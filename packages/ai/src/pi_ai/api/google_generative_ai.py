"""Google Generative AI (Gemini API) adapter.

Streams through the official ``google-genai`` SDK (``client.aio.models.generate_content_stream``),
as pi upstream does with ``@google/genai``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

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

__all__ = ["GoogleOptions", "stream", "stream_simple", "get_google_budget"]

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/"


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


async def _create_client(model: Model, options: Mapping[str, Any]) -> Client:
    api_key = options.get("api_key")
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base_url: str | None
    api_version: str | None = None
    if model.base_url:
        # The catalog baseUrl already includes the version path; don't append one.
        base_url, api_version = model.base_url, ""
    else:
        # The SDK reads GOOGLE_GEMINI_BASE_URL from os.environ itself; this also honors the provider env.
        base_url = (get_provider_env_value("GOOGLE_GEMINI_BASE_URL", options.get("env")) or "").strip() or None
    http_client = get_http_client(base_url or _DEFAULT_BASE_URL, options)

    async def factory() -> Client:
        http_options = genai_types.HttpOptions(
            base_url=base_url, api_version=api_version, httpx_async_client=http_client
        )
        return Client(vertexai=False, api_key=api_key, http_options=http_options)

    # Headers are sent per request (see run_google_stream), so they are not part of the key.
    return await get_google_client(http_client, ("gemini", api_key, base_url, api_version), factory)


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
