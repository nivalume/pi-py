# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""OpenAI Responses API (``openai-responses``).

Talks to ``POST {base_url}/responses`` with the request body and headers the OpenAI
SDK would send. Also used by OpenAI-compatible Responses endpoints (xAI, GitHub
Copilot, OpenRouter, OpenCode, Cloudflare AI Gateway, ...).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import httpx

from ..abort import abort_scope
from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    StartEvent,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from ..utils.callbacks import maybe_await
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.http import ProviderHttpError, build_headers, iter_sse, send_request
from ..utils.misc import get_pi_user_agent, get_provider_env_value, headers_to_record
from ..utils.transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from .constrained_sampling import create_grammar_tool_input_properties
from .github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import convert_responses_messages, convert_responses_tools, process_responses_stream
from .simple_options import build_base_options, resolve_sampling_params

__all__ = [
    "OpenAIResponsesOptions",
    "OPENAI_TOOL_CALL_PROVIDERS",
    "stream",
    "stream_simple",
    "iter_responses_sse_events",
    "openai_sdk_headers",
]

OPENAI_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})
# OpenAI Responses rejects max_output_tokens below 16: https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16
CHATGPT_USAGE_URL = "https://chatgpt.com/settings/usage"
# Fixed reasoning.effort while configuration_update items carry the thinking level.
CONFIGURATION_UPDATE_REQUEST_EFFORT = "medium"


class OpenAIResponsesOptions(StreamOptions, total=False):
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    reasoning_summary: Literal["auto", "detailed", "concise"] | None
    service_tier: str
    tool_choice: Any


def _is_chatgpt_sign_in(model: Model, api_key: str | None) -> bool:
    """OpenAI API keys start with ``sk-``; another credential sent directly to OpenAI is a
    Sign in with ChatGPT access token."""
    return (
        model.provider == "openai"
        and model.base_url == "https://api.openai.com/v1"
        and api_key is not None
        and not api_key.startswith("sk-")
    )


def _has_header(headers: Mapping[str, str | None] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value is not None and value.strip() != "" for key, value in headers.items())


def _get_client_api_key(provider: str, api_key: str | None, headers: Mapping[str, str | None] | None) -> str:
    if api_key:
        return api_key
    if _has_header(headers, "authorization") or _has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise RuntimeError(f"No API key for provider: {provider}")


def _detect_session_affinity_format(model: Model) -> str:
    return "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.base_url else "openai"


def _resolve_cache_retention(cache_retention: str | None, env: Mapping[str, str] | None) -> str:
    """Default ``"short"``; ``PI_CACHE_RETENTION=long`` is honored for backward compatibility."""
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _compat_value(compat: Mapping[str, Any], key: str, default: Any) -> Any:
    value = compat.get(key)
    return default if value is None else value


def _get_compat(model: Model) -> dict[str, Any]:
    compat = model.compat or {}
    return {
        "supportsDeveloperRole": _compat_value(compat, "supportsDeveloperRole", True),
        "supportsMidConvoSystemMessages": _compat_value(compat, "supportsMidConvoSystemMessages", False),
        "sessionAffinityFormat": _compat_value(compat, "sessionAffinityFormat", None)
        or _detect_session_affinity_format(model),
        "supportsLongCacheRetention": _compat_value(compat, "supportsLongCacheRetention", True),
        "supportsStrictMode": _compat_value(compat, "supportsStrictMode", False),
        "supportsOpenAIGrammarTools": _compat_value(compat, "supportsOpenAIGrammarTools", False),
        "supportsAdditionalTools": _compat_value(compat, "supportsAdditionalTools", False),
        "supportsToolSearch": _compat_value(compat, "supportsToolSearch", False),
        "supportsExplicitPromptCacheMode": _compat_value(compat, "supportsExplicitPromptCacheMode", False),
        "supportsMaxOutputTokens": _compat_value(compat, "supportsMaxOutputTokens", True),
        "supportsMidConvoEffort": _compat_value(compat, "supportsMidConvoEffort", False),
    }


def _map_thinking_level(model: Model, level: str) -> str:
    level_map = model.thinking_level_map or {}
    mapped = level_map.get(level)
    return mapped if mapped is not None else level


def _get_configuration_update_effort(model: Model, options: Mapping[str, Any], compat: Mapping[str, Any]) -> str | None:
    """Effort to send in ``configuration_update`` items, or ``None`` to send it as ``reasoning.effort``."""
    reasoning_effort = options.get("reasoning_effort")
    if (
        not compat["supportsMidConvoEffort"]
        or not reasoning_effort
        # Sign in with ChatGPT rejects configuration_update items.
        or _is_chatgpt_sign_in(model, options.get("api_key"))
    ):
        return None
    return _map_thinking_level(model, reasoning_effort)


def _get_prompt_cache_retention(compat: Mapping[str, Any], cache_retention: str) -> str | None:
    if (
        cache_retention == "long"
        and compat["supportsLongCacheRetention"]
        and not compat["supportsExplicitPromptCacheMode"]
    ):
        return "24h"
    return None


def _get_prompt_cache_options(compat: Mapping[str, Any], cache_retention: str) -> dict[str, Any] | None:
    if not compat["supportsExplicitPromptCacheMode"]:
        return None
    if cache_retention == "none":
        return {"mode": "explicit"}
    if cache_retention == "long" and compat["supportsLongCacheRetention"]:
        return {"ttl": "30m"}
    return None


def openai_sdk_headers(env: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Default headers the OpenAI SDK sends before auth and client default headers."""
    return {
        "Accept": "application/json",
        "OpenAI-Organization": get_provider_env_value("OPENAI_ORG_ID", env),
        "OpenAI-Project": get_provider_env_value("OPENAI_PROJECT_ID", env),
    }


def _sdk_stream_error(error: Any, headers: httpx.Headers) -> ProviderHttpError:
    """The OpenAI SDK's ``APIError`` for a stream chunk carrying an ``error`` field."""
    message: Any = error.get("message") if isinstance(error, dict) else None
    if message:
        text = message if isinstance(message, str) else _json_dumps(message)
    else:
        text = _json_dumps(error)
    return ProviderHttpError(text, headers=headers, error=error)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


async def iter_responses_sse_events(response: httpx.Response) -> AsyncIterator[Any]:
    """Decode a Responses SSE stream like the OpenAI SDK's ``Stream`` iterator."""
    async for sse in iter_sse(response):
        if sse.data.startswith("[DONE]"):
            break
        data = sse.json()
        if isinstance(data, dict) and data.get("error"):
            raise _sdk_stream_error(data["error"], response.headers)
        yield data


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a response from the OpenAI Responses API."""
    opts: dict[str, Any] = dict(options or {})
    out = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, _get_compat(model)["supportsMidConvoSystemMessages"])
    out.start(_run(model, normalized_context, opts, out))
    return out


async def _run(
    model: Model, context: TranscriptContext, options: dict[str, Any], out: AssistantMessageEventStream
) -> None:
    compat = _get_compat(model)
    configuration_update_effort = _get_configuration_update_effort(model, options, compat)
    output = AssistantMessage(
        [],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        provider_thinking_level=configuration_update_effort,
    )
    signal = options.get("signal")
    try:
        async with abort_scope(signal):
            api_key = _get_client_api_key(model.provider, options.get("api_key"), options.get("headers"))
            cache_retention = _resolve_cache_retention(options.get("cache_retention"), options.get("env"))
            cache_session_id = None if cache_retention == "none" else options.get("session_id")
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(context.messages), compat["supportsOpenAIGrammarTools"]
            )
            headers = _build_request_headers(model, context, api_key, options, compat, cache_session_id)
            params: Any = _build_params(
                model, context, options, compat, grammar_tool_input_properties, configuration_update_effort
            )
            on_payload = options.get("on_payload")
            if on_payload is not None:
                next_params = await maybe_await(on_payload(params, model))
                if next_params is not None:
                    params = next_params
            response = await send_request(
                "POST",
                f"{model.base_url}/responses",
                headers=headers,
                body=params,
                options=options,
                error_style="openai",
            )
            try:
                on_response = options.get("on_response")
                if on_response is not None:
                    await maybe_await(
                        on_response(ProviderResponse(response.status_code, headers_to_record(response.headers)), model)
                    )
                out.push(StartEvent(output))
                await process_responses_stream(
                    iter_responses_sse_events(response),
                    output,
                    out,
                    model,
                    {
                        "on_provider_stream_event": options.get("on_provider_stream_event"),
                        "service_tier": options.get("service_tier"),
                        "grammar_tool_input_properties": grammar_tool_input_properties,
                        "apply_service_tier_pricing": lambda usage, tier: _apply_service_tier_pricing(
                            usage, tier, model
                        ),
                    },
                )
            finally:
                await response.aclose()

        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        if output.stop_reason == "pending":
            raise RuntimeError("OpenAI Responses stream ended without a stop reason")
        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError(output.error_message or "An unknown error occurred")

        out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()
    except Exception as error:
        aborted = signal is not None and signal.aborted
        output.stop_reason = "aborted" if aborted else "error"
        if aborted:
            error_message = "Request was aborted"
        else:
            prefix = "OpenAI" if model.provider == "openai" else model.provider
            error_message = format_provider_error(normalize_provider_error(error), f"{prefix} API error")
        # Sign in with ChatGPT shares the subscription's usage limit with other apps.
        output.error_message = (
            f"{error_message}\nCheck your ChatGPT usage: {CHATGPT_USAGE_URL}"
            if "subscription_sharing_usage_limit_exceeded" in error_message
            else error_message
        )
        out.push(ErrorEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream with provider-neutral options (``reasoning``, ``tool_choice``, ...)."""
    options = options or {}
    _get_client_api_key(model.provider, options.get("api_key"), options.get("headers"))

    base = build_base_options(model, context, options, options.get("api_key"))
    if options.get("tool_choice") is not None:
        base["tool_choice"] = options["tool_choice"]
    reasoning = options.get("reasoning")
    clamped = clamp_thinking_level(model, reasoning) if reasoning else None
    if clamped is not None and clamped != "off":
        base["reasoning_effort"] = clamped
    return stream(model, context, base)


def _build_request_headers(
    model: Model,
    context: TranscriptContext,
    api_key: str,
    options: Mapping[str, Any],
    compat: Mapping[str, Any],
    session_id: str | None,
) -> dict[str, str]:
    copilot_headers: dict[str, str] | None = None
    if model.provider == "github-copilot":
        copilot_headers = build_copilot_dynamic_headers(
            messages=context.messages, has_images=has_copilot_vision_input(context.messages)
        )

    session_headers: dict[str, str] = {}
    if session_id:
        affinity_format = compat["sessionAffinityFormat"]
        if affinity_format == "openrouter":
            session_headers["x-session-id"] = session_id
        else:
            if affinity_format == "openai":
                session_headers["session_id"] = session_id
            session_headers["x-client-request-id"] = session_id

    return build_headers(
        openai_sdk_headers(options.get("env")),
        {"Authorization": f"Bearer {api_key}"},
        {"User-Agent": get_pi_user_agent()},
        model.headers,
        copilot_headers,
        session_headers,
        # Options headers last so they can override defaults.
        options.get("headers"),
        {"Content-Type": "application/json"},
    )


def _build_params(
    model: Model,
    context: TranscriptContext,
    options: Mapping[str, Any],
    compat: Mapping[str, Any],
    grammar_tool_input_properties: Mapping[str, str],
    configuration_update_effort: str | None,
) -> dict[str, Any]:
    transcript_tools = resolve_transcript_tools(
        context.messages, compat["supportsAdditionalTools"] or compat["supportsToolSearch"]
    )
    tool_options = {
        "supports_strict_mode": compat["supportsStrictMode"],
        "supports_openai_grammar_tools": compat["supportsOpenAIGrammarTools"],
    }
    messages_options: dict[str, Any] = {
        "grammar_tool_input_properties": grammar_tool_input_properties,
        "supports_mid_convo_system_messages": compat["supportsMidConvoSystemMessages"],
        "supports_additional_tools": compat["supportsAdditionalTools"],
        "supports_tool_search": compat["supportsToolSearch"],
        "tool_options": tool_options,
    }
    if configuration_update_effort is not None:
        messages_options["configuration_update_effort"] = configuration_update_effort
    messages = convert_responses_messages(model, context, OPENAI_TOOL_CALL_PROVIDERS, messages_options)  # type: ignore[arg-type]

    cache_retention = _resolve_cache_retention(options.get("cache_retention"), options.get("env"))
    # Sign in with ChatGPT rejects these request fields.
    omit_unsupported_fields = _is_chatgpt_sign_in(model, options.get("api_key"))
    params: dict[str, Any] = {"model": model.id, "input": messages, "stream": True}
    prompt_cache_key = None if cache_retention == "none" else clamp_openai_prompt_cache_key(options.get("session_id"))
    if prompt_cache_key is not None:
        params["prompt_cache_key"] = prompt_cache_key
    if not omit_unsupported_fields:
        retention = _get_prompt_cache_retention(compat, cache_retention)
        if retention is not None:
            params["prompt_cache_retention"] = retention
        cache_options = _get_prompt_cache_options(compat, cache_retention)
        if cache_options is not None:
            params["prompt_cache_options"] = cache_options
    params["store"] = False

    max_tokens = options.get("max_tokens")
    if max_tokens and compat["supportsMaxOutputTokens"] and not omit_unsupported_fields:
        params["max_output_tokens"] = max(max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options.get("temperature") is not None and not omit_unsupported_fields:
        params["temperature"] = options["temperature"]

    if options.get("service_tier") is not None:
        params["service_tier"] = options["service_tier"]

    if transcript_tools.request_tools:
        params["tools"] = convert_responses_tools(transcript_tools.request_tools, tool_options)  # type: ignore[arg-type]

    if options.get("tool_choice") is not None:
        params["tool_choice"] = options["tool_choice"]

    requested = options.get("reasoning_effort")
    reasoning_effort = requested or ("medium" if options.get("reasoning_summary") else None)
    if model.reasoning:
        level_map = model.thinking_level_map or {}
        if reasoning_effort:
            requested_effort = _map_thinking_level(model, requested) if requested else reasoning_effort
            effort = requested_effort if configuration_update_effort is None else CONFIGURATION_UPDATE_REQUEST_EFFORT
            params["reasoning"] = {"effort": effort, "summary": options.get("reasoning_summary") or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot" and not ("off" in level_map and level_map["off"] is None):
            off = level_map.get("off")
            params["reasoning"] = {"effort": off if off is not None else "none"}
        if model.provider == "xai":
            params["include"] = ["reasoning.encrypted_content"]

    # Last so model and request sampling parameters override named request fields.
    sampling_params = resolve_sampling_params(model, reasoning_effort or "off", options.get("sampling_params"))
    if sampling_params:
        params.update(sampling_params)

    return params


def _get_service_tier_cost_multiplier(model: Model, service_tier: Any) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier in ("priority", "fast"):
        return 2.5 if model.id == "gpt-5.5" else 2
    return 1


def _apply_service_tier_pricing(usage: Usage, service_tier: Any, model: Model) -> None:
    multiplier = _get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return
    cost = usage.cost
    cost.input *= multiplier
    cost.output *= multiplier
    cost.cache_read *= multiplier
    cost.cache_write *= multiplier
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write
