"""Google Vertex AI adapter.

Streams ``publishers/google/models/{model}:streamGenerateContent?alt=sse`` over REST, with
either a Vertex API key (express mode) or Application Default Credentials plus project and
location, resolving endpoints the way ``@google/genai`` (used by pi upstream) does.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from typing import Any, Literal

from ..auth.google_adc import get_google_auth_headers
from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..types import Model, StreamOptions, TranscriptContext
from ..utils.misc import get_provider_env_value
from .google_shared import (
    GoogleClient,
    GoogleThinkingOptions,
    ResolvedGoogleThinkingLevel,
    build_google_client_headers,
    new_google_output,
    resolve_google_thinking_level,
    run_google_stream,
    to_google_thinking_level,
    uses_google_thinking_level,
)
from .simple_options import build_base_options

__all__ = [
    "GoogleVertexOptions",
    "stream",
    "stream_simple",
    "get_google_budget",
    "resolve_api_key",
    "resolve_project",
    "resolve_location",
    "resolve_vertex_base_url",
]

API_VERSION = "v1"
GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"
_GLOBAL_ENDPOINT = "https://aiplatform.googleapis.com/"
_MULTI_REGIONAL_LOCATIONS = frozenset(("us", "eu"))
_PLACEHOLDER_API_KEY = re.compile(r"<[^>]+>")
_API_VERSION_SEGMENT = re.compile(r"v\d+(?:beta\d*)?")
_API_VERSION_IN_TEXT = re.compile(r"(?:^|/)v\d+(?:beta\d*)?(?:/|$)")


class GoogleVertexOptions(StreamOptions, total=False):
    tool_choice: Literal["auto", "none", "any"]
    thinking: GoogleThinkingOptions
    project: str
    location: str


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a Vertex AI response."""
    opts = dict(options or {})
    out = AssistantMessageEventStream()
    output = new_google_output(model)
    out.start(
        run_google_stream(
            model,
            context,
            opts,
            out,
            output,
            create_client=lambda: _create_client(model, opts),
            missing_finish_reason_message="Google Vertex stream ended without a finish reason",
        )
    )
    return out


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream with provider-neutral options (``reasoning``, ``tool_choice``, ``thinking_budgets``)."""
    options = options or {}
    base: dict[str, Any] = build_base_options(model, context, options, None)
    if options.get("tool_choice") is not None:
        base["tool_choice"] = options["tool_choice"]
    reasoning = options.get("reasoning")
    if not reasoning:
        return stream(model, context, {**base, "thinking": {"enabled": False}})

    clamped = clamp_thinking_level(model, reasoning)
    if clamped == "off":
        return stream(model, context, {**base, "thinking": {"enabled": False}})
    resolved = resolve_google_thinking_level(model, clamped)

    if uses_google_thinking_level(model):
        return stream(
            model, context, {**base, "thinking": {"enabled": True, "level": to_google_thinking_level(resolved)}}
        )
    return stream(
        model,
        context,
        {
            **base,
            "thinking": {
                "enabled": True,
                "budget_tokens": get_google_budget(model, resolved, options.get("thinking_budgets")),
            },
        },
    )


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------


def resolve_api_key(options: Mapping[str, Any] | None) -> str | None:
    """The Vertex API key, or ``None`` for ADC (no key, the ADC marker, or a ``<placeholder>``)."""
    api_key = ((options or {}).get("api_key") or "").strip()
    if not api_key or api_key == GCP_VERTEX_CREDENTIALS_MARKER or _PLACEHOLDER_API_KEY.fullmatch(api_key):
        return None
    return api_key


def resolve_project(options: Mapping[str, Any] | None) -> str:
    options = options or {}
    env = options.get("env")
    project = (
        options.get("project")
        or get_provider_env_value("GOOGLE_CLOUD_PROJECT", env)
        or get_provider_env_value("GCLOUD_PROJECT", env)
    )
    if not project:
        raise ValueError(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def resolve_location(options: Mapping[str, Any] | None) -> str:
    options = options or {}
    location = options.get("location") or get_provider_env_value("GOOGLE_CLOUD_LOCATION", options.get("env"))
    if not location:
        raise ValueError("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    return location


def _resolve_custom_base_url(base_url: str) -> str | None:
    """A model baseUrl to forward, skipping generated ``{location}`` placeholders."""
    trimmed = base_url.strip()
    if not trimmed or "{location}" in trimmed:
        return None
    return trimmed


def _base_url_includes_api_version(base_url: str) -> bool:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme and parsed.netloc:
        return any(_API_VERSION_SEGMENT.fullmatch(part) for part in parsed.path.split("/"))
    return _API_VERSION_IN_TEXT.search(base_url) is not None


def resolve_vertex_base_url(
    model: Model,
    *,
    project: str | None,
    location: str | None,
    api_key: str | None,
    env: Mapping[str, str] | None = None,
) -> str:
    """The URL prefix that ``{model path}:streamGenerateContent`` is appended to.

    A custom model baseUrl is used as a collection URL (no ``projects/.../locations/...``
    prefix; the API version is appended unless the URL already has one). Otherwise the SDK's
    endpoint rules apply: the global endpoint for API keys and ``location == "global"``,
    ``aiplatform.{us,eu}.rep.googleapis.com`` for multi-regions, ``{location}-aiplatform`` otherwise.
    """
    custom = _resolve_custom_base_url(model.base_url)
    if custom is not None:
        base_url = custom
        api_version = "" if _base_url_includes_api_version(custom) else API_VERSION
        prepend_resource_path = False
    else:
        api_version = API_VERSION
        env_base_url = (get_provider_env_value("GOOGLE_VERTEX_BASE_URL", env) or "").strip()
        if env_base_url:
            base_url = env_base_url
        elif (api_key and not project) or location == "global":
            # Vertex express mode (API key) or the global endpoint.
            base_url = _GLOBAL_ENDPOINT
        elif project and location and location in _MULTI_REGIONAL_LOCATIONS:
            base_url = f"https://aiplatform.{location}.rep.googleapis.com/"
        else:
            base_url = f"https://{location}-aiplatform.googleapis.com/"
        prepend_resource_path = bool(project and location and not api_key)
    parts = [base_url[:-1] if base_url.endswith("/") else base_url]
    if api_version:
        parts.append(api_version)
    if prepend_resource_path:
        parts.append(f"projects/{project}/locations/{location}")
    return "/".join(parts)


def _create_client(model: Model, options: Mapping[str, Any]) -> GoogleClient:
    # Use a Vertex API key if provided, otherwise ADC with project and location.
    api_key = resolve_api_key(options)
    headers = build_google_client_headers(model, options)
    if api_key:
        prefix = resolve_vertex_base_url(model, project=None, location=None, api_key=api_key, env=options.get("env"))

        async def api_key_headers() -> dict[str, str]:
            if api_key.startswith("auth_tokens/"):
                raise ValueError("Ephemeral tokens are only supported by the live API.")
            return {"x-goog-api-key": api_key}

        return GoogleClient(
            vertex=True,
            url=lambda model_path: f"{prefix}/{model_path}:streamGenerateContent?alt=sse",
            headers=headers,
            auth_headers=api_key_headers,
        )

    project = resolve_project(options)
    location = resolve_location(options)
    prefix = resolve_vertex_base_url(model, project=project, location=location, api_key=None, env=options.get("env"))

    async def adc_headers() -> dict[str, str]:
        return await get_google_auth_headers(
            env=options.get("env"),
            http_client=options.get("http_client"),
            signal=options.get("signal"),
            timeout_ms=options.get("timeout_ms"),
        )

    return GoogleClient(
        vertex=True,
        url=lambda model_path: f"{prefix}/{model_path}:streamGenerateContent?alt=sse",
        headers=headers,
        auth_headers=adc_headers,
    )


def get_google_budget(
    model: Model, level: ResolvedGoogleThinkingLevel, custom_budgets: Mapping[str, int] | None = None
) -> int:
    """Token budget for a thinking level on budget-based Gemini models (-1 = dynamic)."""
    if custom_budgets and custom_budgets.get(level) is not None:
        return custom_budgets[level]
    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[level]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[level]
    return -1
