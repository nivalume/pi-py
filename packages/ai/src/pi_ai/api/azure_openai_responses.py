# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""Azure OpenAI Responses API (``azure-openai-responses``).

Sends ``POST {base_url}/responses?api-version=<version>`` with an ``api-key`` header,
like the ``AzureOpenAI`` client of the OpenAI SDK. The deployment name goes in the
body's ``model`` field.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

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
    TranscriptContext,
    Usage,
)
from ..utils.callbacks import maybe_await
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.http import build_headers, send_request
from ..utils.misc import get_pi_user_agent, headers_to_record
from ..utils.transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from .azure_openai_config import AzureEndpointOptions, resolve_azure_config, resolve_deployment_name
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses import iter_responses_sse_events, openai_sdk_headers
from .openai_responses_shared import convert_responses_messages, convert_responses_tools, process_responses_stream
from .simple_options import build_base_options, resolve_sampling_params

__all__ = ["AzureOpenAIResponsesOptions", "AZURE_TOOL_CALL_PROVIDERS", "stream", "stream_simple"]

AZURE_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode", "azure"})
# OpenAI Responses rejects max_output_tokens below 16: https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16


class AzureOpenAIResponsesOptions(AzureEndpointOptions, total=False):
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    tool_choice: Any
    reasoning_summary: Literal["auto", "detailed", "concise"] | None


def _format_azure_openai_error(error: Any) -> str:
    return format_provider_error(normalize_provider_error(error), "Azure OpenAI API error")


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a response from an Azure OpenAI Responses deployment."""
    opts: dict[str, Any] = dict(options or {})
    out = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, (model.compat or {}).get("supportsMidConvoSystemMessages"))
    out.start(_run(model, normalized_context, opts, out))
    return out


async def _run(
    model: Model, context: TranscriptContext, options: dict[str, Any], out: AssistantMessageEventStream
) -> None:
    deployment_name = resolve_deployment_name(model, options)
    output = AssistantMessage(
        [],
        api="azure-openai-responses",
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
    )
    signal = options.get("signal")
    try:
        async with abort_scope(signal):
            api_key = options.get("api_key")
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")
            headers = build_headers(
                openai_sdk_headers(options.get("env")),
                {"api-key": api_key},
                {"User-Agent": get_pi_user_agent()},
                model.headers,
                options.get("headers"),
                {"Content-Type": "application/json"},
            )
            base_url, api_version = resolve_azure_config(model, options)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(context.messages), bool((model.compat or {}).get("supportsOpenAIGrammarTools"))
            )
            params: Any = _build_params(model, context, options, deployment_name, grammar_tool_input_properties)
            on_payload = options.get("on_payload")
            if on_payload is not None:
                next_params = await maybe_await(on_payload(params, model))
                if next_params is not None:
                    params = next_params
            response = await send_request(
                "POST",
                _build_url(base_url),
                headers=headers,
                body=params,
                options=options,
                error_style="openai",
                params={"api-version": api_version},
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
                        "grammar_tool_input_properties": grammar_tool_input_properties,
                    },
                )
            finally:
                await response.aclose()

        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        if output.stop_reason == "pending":
            raise RuntimeError("Azure OpenAI Responses stream ended without a stop reason")
        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError(output.error_message or "An unknown error occurred")

        out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()
    except Exception as error:
        aborted = signal is not None and signal.aborted
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = "Request was aborted" if aborted else _format_azure_openai_error(error)
        out.push(ErrorEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream with provider-neutral options (``reasoning``, ``tool_choice``, ...)."""
    options = options or {}
    api_key = options.get("api_key")
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    if options.get("tool_choice") is not None:
        base["tool_choice"] = options["tool_choice"]
    reasoning = options.get("reasoning")
    clamped = clamp_thinking_level(model, reasoning) if reasoning else None
    if clamped is not None and clamped != "off":
        base["reasoning_effort"] = clamped
    # Azure endpoint options are not part of the simple options; keep them when callers pass them.
    for key in ("azure_api_version", "azure_resource_name", "azure_base_url", "azure_deployment_name"):
        if options.get(key) is not None:
            base[key] = options[key]
    return stream(model, context, base)


def _build_url(base_url: str) -> str:
    # Like the SDK: the path is appended to the base URL and the query string is replaced by the
    # ``api-version`` parameter.
    return f"{base_url.partition('?')[0]}/responses"


def _build_params(
    model: Model,
    context: TranscriptContext,
    options: Mapping[str, Any],
    deployment_name: str,
    grammar_tool_input_properties: Mapping[str, str],
) -> dict[str, Any]:
    compat = model.compat or {}
    supports_additional_tools = bool(compat.get("supportsAdditionalTools"))
    supports_tool_search = bool(compat.get("supportsToolSearch"))
    supports_strict_mode = compat.get("supportsStrictMode")
    if supports_strict_mode is None:
        supports_strict_mode = True
    supports_openai_grammar_tools = bool(compat.get("supportsOpenAIGrammarTools"))
    tool_options = {
        "supports_strict_mode": supports_strict_mode,
        "supports_openai_grammar_tools": supports_openai_grammar_tools,
    }
    transcript_tools = resolve_transcript_tools(context.messages, supports_additional_tools or supports_tool_search)
    messages = convert_responses_messages(
        model,
        context,
        AZURE_TOOL_CALL_PROVIDERS,
        {
            "grammar_tool_input_properties": grammar_tool_input_properties,
            "supports_mid_convo_system_messages": bool(compat.get("supportsMidConvoSystemMessages")),
            "supports_additional_tools": supports_additional_tools,
            "supports_tool_search": supports_tool_search,
            "tool_options": tool_options,  # type: ignore[typeddict-item]
        },
    )

    params: dict[str, Any] = {"model": deployment_name, "input": messages, "stream": True}
    prompt_cache_key = clamp_openai_prompt_cache_key(options.get("session_id"))
    if prompt_cache_key is not None:
        params["prompt_cache_key"] = prompt_cache_key
    params["store"] = False

    max_tokens = options.get("max_tokens")
    if max_tokens:
        params["max_output_tokens"] = max(max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options.get("temperature") is not None:
        params["temperature"] = options["temperature"]

    if transcript_tools.request_tools:
        params["tools"] = convert_responses_tools(transcript_tools.request_tools, tool_options)  # type: ignore[arg-type]
    if options.get("tool_choice") is not None:
        params["tool_choice"] = options["tool_choice"]

    requested = options.get("reasoning_effort")
    reasoning_effort = requested or ("medium" if options.get("reasoning_summary") else None)
    if model.reasoning:
        level_map = model.thinking_level_map or {}
        if reasoning_effort:
            effort = reasoning_effort
            if requested:
                mapped = level_map.get(requested)
                effort = mapped if mapped is not None else requested
            params["reasoning"] = {"effort": effort, "summary": options.get("reasoning_summary") or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif not ("off" in level_map and level_map["off"] is None):
            off = level_map.get("off")
            params["reasoning"] = {"effort": off if off is not None else "none"}

    # Last so model and request sampling parameters override named request fields.
    sampling_params = resolve_sampling_params(model, reasoning_effort or "off", options.get("sampling_params"))
    if sampling_params:
        params.update(sampling_params)

    return params
