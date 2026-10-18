"""Google Generative AI (Gemini API) adapter.

Streams ``models/{model}:streamGenerateContent?alt=sse`` over REST with the request body,
headers and error wording that ``@google/genai`` (used by pi upstream) produces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

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

__all__ = ["GoogleOptions", "stream", "stream_simple", "get_google_budget"]

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/"
_DEFAULT_API_VERSION = "v1beta"


class GoogleOptions(StreamOptions, total=False):
    tool_choice: Literal["auto", "none", "any"]
    thinking: GoogleThinkingOptions


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a Gemini API response."""
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
            missing_finish_reason_message="Google stream ended without a finish reason",
        )
    )
    return out


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream with provider-neutral options (``reasoning``, ``tool_choice``, ``thinking_budgets``)."""
    options = options or {}
    api_key = options.get("api_key")
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base: dict[str, Any] = build_base_options(model, context, options, api_key)
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


def _create_client(model: Model, options: Mapping[str, Any]) -> GoogleClient:
    api_key = options.get("api_key")
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    if model.base_url:
        # The catalog baseUrl already includes the version path; don't append one.
        base_url, api_version = model.base_url, ""
    else:
        env_base_url = (get_provider_env_value("GOOGLE_GEMINI_BASE_URL", options.get("env")) or "").strip()
        base_url, api_version = env_base_url or _DEFAULT_BASE_URL, _DEFAULT_API_VERSION
    prefix = base_url[:-1] if base_url.endswith("/") else base_url
    if api_version:
        prefix = f"{prefix}/{api_version}"

    async def auth_headers() -> dict[str, str]:
        if api_key.startswith("auth_tokens/"):
            raise ValueError("Ephemeral tokens are only supported by the live API.")
        return {"x-goog-api-key": api_key}

    return GoogleClient(
        vertex=False,
        url=lambda model_path: f"{prefix}/{model_path}:streamGenerateContent?alt=sse",
        headers=build_google_client_headers(model, options),
        auth_headers=auth_headers,
    )


def get_google_budget(
    model: Model, level: ResolvedGoogleThinkingLevel, custom_budgets: Mapping[str, int] | None = None
) -> int:
    """Token budget for a thinking level on budget-based Gemini models (-1 = dynamic)."""
    if custom_budgets and custom_budgets.get(level) is not None:
        return custom_budgets[level]
    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[level]
    if "2.5-flash-lite" in model.id:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[level]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[level]
    return -1
