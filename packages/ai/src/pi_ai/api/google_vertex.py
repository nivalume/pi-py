"""Google Vertex AI adapter.

Streams through the official ``google-genai`` SDK (``Client(vertexai=True, ...)``), as pi
upstream does with ``@google/genai``: either with a Vertex API key (express mode) or with
Application Default Credentials (through ``google-auth``) plus project and location.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from collections.abc import Mapping
from typing import Any, Literal

import google.auth
from google.auth.credentials import Credentials
from google.genai import Client
from google.genai import types as genai_types

from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..types import Model, StreamOptions, TranscriptContext
from ..utils.http import get_http_client
from ..utils.misc import get_provider_env_value
from .google_shared import (
    GoogleThinkingOptions,
    ResolvedGoogleThinkingLevel,
    get_google_client,
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
]

API_VERSION = "v1"
GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_GLOBAL_ENDPOINT = "https://aiplatform.googleapis.com/"
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


def _load_credentials_file(path: str) -> Credentials:
    credentials, _project = google.auth.load_credentials_from_file(path, scopes=[CLOUD_PLATFORM_SCOPE])
    return credentials


async def _create_client(model: Model, options: Mapping[str, Any]) -> Client:
    env = options.get("env")
    http_options: dict[str, Any] = {"api_version": API_VERSION}
    custom_base_url = _resolve_custom_base_url(model.base_url)
    if custom_base_url is not None:
        # A model baseUrl is a collection URL: no projects/.../locations/... prefix. The JS SDK
        # still appends the API version unless the URL has one; the Python SDK never does for
        # collection-scoped URLs, so add it here.
        if not _base_url_includes_api_version(custom_base_url):
            custom_base_url = f"{custom_base_url.rstrip('/')}/{API_VERSION}"
        http_options["base_url"] = custom_base_url
        http_options["base_url_resource_scope"] = genai_types.ResourceScope.COLLECTION
    else:
        # The SDK reads GOOGLE_VERTEX_BASE_URL from os.environ itself; this also honors the provider env.
        env_base_url = (get_provider_env_value("GOOGLE_VERTEX_BASE_URL", env) or "").strip()
        if env_base_url:
            http_options["base_url"] = env_base_url
    http_client = get_http_client(http_options.get("base_url") or _GLOBAL_ENDPOINT, options)
    http_options["httpx_async_client"] = http_client
    base_key = (http_options.get("base_url"), http_options.get("base_url_resource_scope"))

    # Use a Vertex API key if provided, otherwise ADC with project and location.
    api_key = resolve_api_key(options)
    if api_key:

        async def api_key_client() -> Client:
            return Client(vertexai=True, api_key=api_key, http_options=genai_types.HttpOptions(**http_options))

        return await get_google_client(http_client, ("vertex-api-key", api_key, *base_key), api_key_client)

    project = resolve_project(options)
    location = resolve_location(options)
    # Upstream passes GOOGLE_APPLICATION_CREDENTIALS to google-auth as keyFilename; without it the
    # SDK uses google.auth.default(). The cached client keeps the credentials and refreshes the token.
    key_filename = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)

    async def adc_client() -> Client:
        credentials = await asyncio.to_thread(_load_credentials_file, key_filename) if key_filename else None
        return Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
            http_options=genai_types.HttpOptions(**http_options),
        )

    return await get_google_client(http_client, ("vertex-adc", project, location, key_filename, *base_key), adc_client)


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
