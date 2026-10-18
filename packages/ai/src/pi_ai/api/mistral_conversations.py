# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""Mistral Chat Completions streaming (``mistral-conversations`` API).

Port of pi's ``api/mistral-conversations.ts``. Like pi, this talks to
``/v1/chat/completions`` directly: the payload is built in the Mistral SDK's
camelCase shape (what ``on_payload`` sees) and remapped to the snake_case wire
format before sending.
"""

from __future__ import annotations

import asyncio
import http
import math
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal, TypedDict

import httpx

from ..abort import abort_scope
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, clamp_thinking_level
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    ProviderResponse,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
)
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.error_body import truncate_error_text
from ..utils.http import ProviderHttpError, build_headers, iter_sse, send_request
from ..utils.json_parse import StreamingJsonParser, parse_tool_call_arguments
from ..utils.misc import get_pi_user_agent, headers_to_record, sanitize_surrogates, short_hash
from ..utils.text import append_text as append_block_text
from ..utils.text import append_thinking as append_block_thinking
from ..utils.text import get_system_message_text, render_system_message_update
from ..utils.transcript import get_current_tools, resolve_transcript
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .simple_options import build_base_options
from .transform_messages import transform_messages

__all__ = ["MistralOptions", "MistralReasoningEffort", "MistralToolChoice", "stream", "stream_simple"]

_MISTRAL_TOOL_CALL_ID_LENGTH = 9
_MAX_MISTRAL_ERROR_BODY_CHARS = 4000
_DEFAULT_HEADER_TIMEOUT_MS = 60_000

MistralReasoningEffort = Literal["none", "low", "medium", "high", "max"]


class _FunctionName(TypedDict):
    name: str


class _FunctionToolChoice(TypedDict):
    type: Literal["function"]
    function: _FunctionName


MistralToolChoice = Literal["auto", "none", "any", "required"] | _FunctionToolChoice


class MistralOptions(StreamOptions, total=False):
    """Provider-specific options for the Mistral API."""

    tool_choice: MistralToolChoice
    prompt_mode: Literal["reasoning"]
    reasoning_effort: MistralReasoningEffort


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream responses from the native Mistral Chat Completions endpoint."""
    opts: dict[str, Any] = dict(options or {})
    out = AssistantMessageEventStream()
    normalized = resolve_transcript(context, (model.compat or {}).get("supportsMidConvoSystemMessages"))
    out.start(_run(model, normalized, opts, out))
    return out


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Map provider-agnostic simple options to Mistral options.

    Raises ``ValueError`` synchronously when no API key is given.
    """
    options = options or {}
    api_key = options.get("api_key")
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base: dict[str, Any] = {**build_base_options(model, context, options, api_key)}
    if options.get("tool_choice") is not None:
        base["tool_choice"] = options["tool_choice"]
    requested = options.get("reasoning")
    clamped = clamp_thinking_level(model, requested) if requested else None
    reasoning = None if clamped == "off" else clamped
    # Models with a thinking level map use `reasoning_effort`; other reasoning models use `prompt_mode`.
    effort_map = model.thinking_level_map if model.reasoning else None
    reasoning_effort: str | None = None
    if effort_map is not None:
        mapped = effort_map.get(reasoning) if reasoning else effort_map.get("off")
        reasoning_effort = "high" if reasoning and mapped is None else mapped
    if model.reasoning and effort_map is None and reasoning:
        base["prompt_mode"] = "reasoning"
    if reasoning_effort:
        base["reasoning_effort"] = reasoning_effort
    return stream(model, context, base)


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


async def _run(
    model: Model, context: TranscriptContext, options: dict[str, Any], out: AssistantMessageEventStream
) -> None:
    output = AssistantMessage(
        [], api=model.api, provider=model.provider, model=model.id, usage=Usage(), stop_reason="pending"
    )
    signal = options.get("signal")
    try:
        api_key = options.get("api_key")
        if not api_key:
            raise RuntimeError(f"No API key for provider: {model.provider}")

        normalize_id = _create_mistral_tool_call_id_normalizer()
        messages = transform_messages(context.messages, model, lambda id_, _model, _message: normalize_id(id_))
        payload: Any = _build_chat_payload(model, context, messages, options)
        on_payload = options.get("on_payload")
        if on_payload is not None:
            replaced = await maybe_await(on_payload(payload, model))
            if replaced is not None:
                payload = replaced

        async with abort_scope(signal):
            response = await _request_mistral_stream(model, payload, api_key, options)
            try:
                out.push(StartEvent(output))
                await _consume_chat_stream(model, output, out, response, options.get("on_provider_stream_event"))
            finally:
                await response.aclose()

        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        if output.stop_reason == "pending":
            raise RuntimeError("Mistral stream ended without a finish reason")
        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError(output.error_message or "An unknown error occurred")

        out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()
    except Exception as error:
        output.stop_reason = "aborted" if signal is not None and signal.aborted else "error"
        output.error_message = _format_mistral_error(error)
        out.push(ErrorEvent(output.stop_reason, output))
        out.end()


def _create_mistral_tool_call_id_normalizer() -> Any:
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(id_: str) -> str:
        existing = id_map.get(id_)
        if existing:
            return existing
        attempt = 0
        while True:
            candidate = _derive_mistral_tool_call_id(id_, attempt)
            owner = reverse_map.get(candidate)
            if not owner or owner == id_:
                id_map[id_] = candidate
                reverse_map[candidate] = id_
                return candidate
            attempt += 1

    return normalize


_NON_ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9]")


def _derive_mistral_tool_call_id(id_: str, attempt: int) -> str:
    normalized = _NON_ALPHANUMERIC.sub("", id_)
    if attempt == 0 and len(normalized) == _MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or id_
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return _NON_ALPHANUMERIC.sub("", short_hash(seed))[:_MISTRAL_TOOL_CALL_ID_LENGTH]


def _format_mistral_error(error: BaseException) -> str:
    if isinstance(error, ProviderHttpError) and error.status is not None:
        body = (error.raw_body or "").strip()
        if body:
            return f"Mistral API error ({error.status}): {truncate_error_text(body, _MAX_MISTRAL_ERROR_BODY_CHARS)}"
        return f"Mistral API error ({error.status}): {_status_text(error.status)}"
    return str(error) or type(error).__name__


def _status_text(status: int) -> str:
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:
        return f"Request failed with status {status}"


def _format_ms(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _chat_completions_url(base_url: str) -> str:
    url = httpx.URL(base_url)
    path = url.path.rstrip("/") + "/v1/chat/completions"
    return str(url.copy_with(path=path, query=None, fragment=None))


async def _request_mistral_stream(
    model: Model, payload: Any, api_key: str, options: Mapping[str, Any]
) -> httpx.Response:
    url = _chat_completions_url(model.base_url)
    headers = _build_mistral_headers(model, api_key, options)
    # The timeout covers only the wait for response headers. Long streams (e.g. extended thinking)
    # must not be cut off by a fixed deadline; body stalls are left to the HTTP client idle timeout.
    timeout_ms = options.get("timeout_ms")
    if timeout_ms is None:
        timeout_ms = _DEFAULT_HEADER_TIMEOUT_MS
    request_options = {key: value for key, value in options.items() if key != "timeout_ms"}
    try:
        async with asyncio.timeout(timeout_ms / 1000):
            response = await send_request(
                "POST",
                url,
                headers=headers,
                content=fastjson.dumps_bytes(_to_mistral_wire_payload(payload)),
                options=request_options,
                error_style="raw",
            )
    except TimeoutError as error:
        raise RuntimeError(f"Mistral response headers timed out after {_format_ms(timeout_ms)}ms") from error
    except ProviderHttpError as error:
        on_response = options.get("on_response")
        if on_response is not None and error.status is not None:
            await maybe_await(
                on_response(ProviderResponse(error.status, headers_to_record(error.headers or {})), model)
            )
        raise

    on_response = options.get("on_response")
    if on_response is not None:
        try:
            await maybe_await(
                on_response(ProviderResponse(response.status_code, headers_to_record(response.headers)), model)
            )
        except BaseException:
            await response.aclose()
            raise
    return response


def _build_mistral_headers(model: Model, api_key: str, options: Mapping[str, Any]) -> dict[str, str]:
    option_headers = options.get("headers")
    headers = build_headers(
        {
            "User-Agent": get_pi_user_agent(),
            "accept": "text/event-stream",
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        model.headers,
        option_headers,
    )
    has_explicit_affinity = _has_header_override(model.headers, "x-affinity") or _has_header_override(
        option_headers, "x-affinity"
    )
    if _should_use_prompt_caching(options) and not has_explicit_affinity:
        headers["x-affinity"] = options["session_id"]
    return headers


def _has_header_override(overrides: Mapping[str, Any] | None, target: str) -> bool:
    return bool(overrides) and any(name.lower() == target for name in overrides)  # type: ignore[union-attr]


_WIRE_PAYLOAD_KEYS = (
    ("topP", "top_p"),
    ("maxTokens", "max_tokens"),
    ("randomSeed", "random_seed"),
    ("responseFormat", "response_format"),
    ("toolChoice", "tool_choice"),
    ("presencePenalty", "presence_penalty"),
    ("frequencyPenalty", "frequency_penalty"),
    ("parallelToolCalls", "parallel_tool_calls"),
    ("reasoningEffort", "reasoning_effort"),
    ("promptMode", "prompt_mode"),
    ("promptCacheKey", "prompt_cache_key"),
    ("safePrompt", "safe_prompt"),
)
_WIRE_CHUNK_KEYS = (
    ("imageUrl", "image_url"),
    ("documentUrl", "document_url"),
    ("documentName", "document_name"),
    ("fileId", "file_id"),
    ("referenceIds", "reference_ids"),
    ("inputAudio", "input_audio"),
)


def _remap(record: dict[str, Any], source: str, target: str) -> None:
    if source in record:
        record[target] = record.pop(source)


def _to_mistral_wire_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    wire = dict(payload)
    for source, target in _WIRE_PAYLOAD_KEYS:
        _remap(wire, source, target)
    messages = payload.get("messages")
    if isinstance(messages, list):
        wire["messages"] = [_to_mistral_wire_message(message) for message in messages]
    response_format = wire.get("response_format")
    if isinstance(response_format, Mapping):
        wire_format = dict(response_format)
        _remap(wire_format, "jsonSchema", "json_schema")
        json_schema = wire_format.get("json_schema")
        if isinstance(json_schema, Mapping):
            wire_schema = dict(json_schema)
            _remap(wire_schema, "schemaDefinition", "schema")
            wire_format["json_schema"] = wire_schema
        wire["response_format"] = wire_format
    return wire


def _to_mistral_wire_message(message: Any) -> Any:
    if not isinstance(message, Mapping):
        return message
    wire = dict(message)
    _remap(wire, "toolCalls", "tool_calls")
    _remap(wire, "toolCallId", "tool_call_id")
    content = message.get("content")
    if isinstance(content, list):
        wire["content"] = [_to_mistral_wire_chunk(chunk) for chunk in content]
    return wire


def _to_mistral_wire_chunk(chunk: Any) -> Any:
    if not isinstance(chunk, Mapping):
        return chunk
    wire = dict(chunk)
    for source, target in _WIRE_CHUNK_KEYS:
        _remap(wire, source, target)
    return wire


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def _build_chat_payload(
    model: Model, context: TranscriptContext, messages: list[Message], options: Mapping[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": _to_chat_messages(messages, "image" in model.input),
    }
    current_tools = get_current_tools(context.messages)
    if current_tools:
        payload["tools"] = _to_function_tools(current_tools)
    if options.get("temperature") is not None:
        payload["temperature"] = options["temperature"]
    if options.get("max_tokens") is not None:
        payload["maxTokens"] = options["max_tokens"]
    tool_choice = _map_tool_choice(options.get("tool_choice"))
    if tool_choice is not None:
        payload["toolChoice"] = tool_choice
    if options.get("prompt_mode"):
        payload["promptMode"] = options["prompt_mode"]
    if options.get("reasoning_effort"):
        payload["reasoningEffort"] = options["reasoning_effort"]
    if _should_use_prompt_caching(options):
        payload["promptCacheKey"] = options["session_id"]
    return payload


def _should_use_prompt_caching(options: Mapping[str, Any]) -> bool:
    return options.get("cache_retention") != "none" and bool(options.get("session_id"))


def _to_function_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, True)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": _plain_json(get_json_schema_tool_parameters(tool, strict)),
                    "strict": strict if strict is not None else False,
                },
            }
        )
    return result


def _plain_json(value: Any) -> Any:
    """Deep copy into plain dicts/lists (pi strips TypeBox symbol keys at this point)."""
    if isinstance(value, Mapping):
        return {key: _plain_json(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _image_url(block: Any) -> str:
    return f"data:{block.mime_type};base64,{block.data}"


def _to_chat_messages(messages: list[Message], supports_images: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.role
        if role == "system":
            text = get_system_message_text(message) if index == 0 else render_system_message_update(message)  # type: ignore[arg-type]
            if text:
                result.append({"role": "system", "content": sanitize_surrogates(text)})
            continue

        if role == "user":
            content = message.content  # type: ignore[union-attr]
            if isinstance(content, str):
                result.append({"role": "user", "content": sanitize_surrogates(content)})
                continue
            had_images = any(item.type == "image" for item in content)
            chunks: list[dict[str, Any]] = []
            for item in content:
                if item.type == "text":
                    chunks.append({"type": "text", "text": sanitize_surrogates(item.text)})
                elif supports_images and item.type == "image":
                    chunks.append({"type": "image_url", "imageUrl": _image_url(item)})
            if chunks:
                result.append({"role": "user", "content": chunks})
                continue
            if had_images and not supports_images:
                result.append({"role": "user", "content": "(image omitted: model does not support images)"})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            for block in message.content:  # type: ignore[union-attr]
                kind = block.type
                if kind == "text":
                    if block.text.strip():
                        parts.append({"type": "text", "text": sanitize_surrogates(block.text)})
                    continue
                if kind == "thinking":
                    if block.thinking.strip():
                        parts.append(
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": sanitize_surrogates(block.thinking)}],
                            }
                        )
                    continue
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": fastjson.dumps(block.arguments or {})},
                        "index": 0,
                    }
                )
            assistant: dict[str, Any] = {"role": "assistant", "prefix": False}
            if parts:
                assistant["content"] = parts
            if tool_calls:
                assistant["toolCalls"] = tool_calls
            if parts or tool_calls:
                result.append(assistant)
            continue

        # toolResult
        tool_content = message.content  # type: ignore[union-attr]
        text_result = "\n".join(sanitize_surrogates(part.text) for part in tool_content if part.type == "text")
        has_images = any(part.type == "image" for part in tool_content)
        chunks = [
            {
                "type": "text",
                "text": _build_tool_result_text(text_result, has_images, supports_images, message.is_error),  # type: ignore[union-attr]
            }
        ]
        if supports_images:
            chunks.extend(
                {"type": "image_url", "imageUrl": _image_url(part)} for part in tool_content if part.type == "image"
            )
        result.append(
            {
                "role": "tool",
                "toolCallId": message.tool_call_id,  # type: ignore[union-attr]
                "name": message.tool_name,  # type: ignore[union-attr]
                "content": chunks,
            }
        )
    return result


def _build_tool_result_text(text: str, has_images: bool, supports_images: bool, is_error: bool) -> str:
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""
    if trimmed:
        image_suffix = (
            "\n[tool image omitted: model does not support images]" if has_images and not supports_images else ""
        )
        return f"{error_prefix}{trimmed}{image_suffix}"
    if has_images:
        if supports_images:
            return "[tool error] (see attached image)" if is_error else "(see attached image)"
        return (
            "[tool error] (image omitted: model does not support images)"
            if is_error
            else "(image omitted: model does not support images)"
        )
    return "[tool error] (no tool output)" if is_error else "(no tool output)"


def _map_tool_choice(choice: Any) -> Any:
    if not choice:
        return None
    if choice in ("auto", "none", "any", "required"):
        return choice
    return {"type": "function", "function": {"name": choice["function"]["name"]}}


# ---------------------------------------------------------------------------
# Stream consumption
# ---------------------------------------------------------------------------


def _map_chat_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    if reason is None or reason == "stop":
        return "stop", None
    if reason in ("length", "model_length"):
        return "length", None
    if reason == "tool_calls":
        return "toolUse", None
    if reason == "error":
        # Mistral reports transient server failures this way; "server error" makes the message retryable.
        return "error", "Provider stopped with: error (server error)"
    return "error", f"Provider stopped with: {reason}"


def _cached_prompt_tokens(usage: Mapping[str, Any], prompt_tokens: int) -> int:
    raw: Any = None
    for details_key, cached_key in (
        ("promptTokensDetails", "cachedTokens"),
        ("prompt_tokens_details", "cached_tokens"),
        ("promptTokenDetails", "cachedTokens"),
        ("prompt_token_details", "cached_tokens"),
    ):
        details = usage.get(details_key)
        if isinstance(details, Mapping) and details.get(cached_key) is not None:
            raw = details[cached_key]
            break
    if raw is None:
        raw = usage.get("numCachedTokens")
    if raw is None:
        raw = usage.get("num_cached_tokens")
    cached = raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw) else 0
    return int(min(prompt_tokens, max(0, cached)))


async def _iter_mistral_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    async for event in iter_sse(response):
        data = event.data.strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        parsed = fastjson.loads(data)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("choices"), list):
            raise RuntimeError("Invalid Mistral streaming event")
        yield parsed


async def _consume_chat_stream(
    model: Model,
    output: AssistantMessage,
    out: AssistantMessageEventStream,
    response: httpx.Response,
    on_provider_stream_event: Any,
) -> None:
    blocks = output.content
    current: TextContent | ThinkingContent | None = None
    tool_blocks_by_key: dict[Any, int] = {}
    parsers: dict[int, StreamingJsonParser] = {}

    def finish_current() -> None:
        if current is None:
            return
        index = len(blocks) - 1
        if current.type == "text":
            out.push(TextEndEvent(index, current.text, output))
        else:
            out.push(ThinkingEndEvent(index, current.thinking, output))

    def append_text(text: str) -> None:
        nonlocal current
        if current is None or current.type != "text":
            finish_current()
            current = TextContent("")
            blocks.append(current)
            out.push(TextStartEvent(len(blocks) - 1, output))
        append_block_text(current, text)
        out.push(TextDeltaEvent(len(blocks) - 1, text, output))

    def append_thinking(text: str) -> None:
        nonlocal current
        if current is None or current.type != "thinking":
            finish_current()
            current = ThinkingContent("")
            blocks.append(current)
            out.push(ThinkingStartEvent(len(blocks) - 1, output))
        append_block_thinking(current, text)
        out.push(ThinkingDeltaEvent(len(blocks) - 1, text, output))

    async for chunk in _iter_mistral_events(response):
        if on_provider_stream_event is not None:
            await maybe_await(on_provider_stream_event(chunk, model))
        # Mistral's streamed CompletionChunk carries an id field. Keep the first non-empty one,
        # mirroring how OpenAI-style streaming exposes a stable response identifier per stream.
        if not output.response_id and chunk.get("id"):
            output.response_id = chunk["id"]

        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("prompt_tokens") or 0
            cached_prompt_tokens = _cached_prompt_tokens(usage, prompt_tokens)
            output.usage.input = max(0, prompt_tokens - cached_prompt_tokens)
            output.usage.output = usage.get("completion_tokens") or 0
            output.usage.cache_read = cached_prompt_tokens
            output.usage.cache_write = 0
            output.usage.total_tokens = usage.get("total_tokens") or (
                output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
            )
            calculate_cost(model, output.usage)

        choices = chunk["choices"]
        choice = choices[0] if choices else None
        if not choice:
            continue

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            output.raw_stop_reason = finish_reason
            stop_reason, error_message = _map_chat_stop_reason(finish_reason)
            output.stop_reason = stop_reason
            if error_message:
                output.error_message = error_message

        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is not None:
            for item in [content] if isinstance(content, str) else content:
                if isinstance(item, str):
                    text = sanitize_surrogates(item)
                    # GLM models on Mistral send empty content deltas around thinking and tool calls.
                    # Opening a block for them splits thinking into multiple blocks, which Mistral rejects on replay.
                    if text:
                        append_text(text)
                    continue
                kind = item.get("type")
                if kind == "thinking":
                    parts = item.get("thinking") or []
                    thinking = sanitize_surrogates("".join(part.get("text") or "" for part in parts))
                    if thinking:
                        append_thinking(thinking)
                elif kind == "text":
                    text = sanitize_surrogates(item.get("text") or "")
                    if text:
                        append_text(text)

        for tool_call in delta.get("tool_calls") or []:
            if current is not None:
                finish_current()
                current = None
            raw_id = tool_call.get("id")
            call_index = tool_call.get("index")
            call_id = (
                raw_id
                if raw_id and raw_id != "null"
                else _derive_mistral_tool_call_id(f"toolcall:{call_index if call_index is not None else 0}", 0)
            )
            key = call_index if call_index is not None else call_id
            function = tool_call.get("function") or {}
            existing_index = tool_blocks_by_key.get(key)
            if existing_index is None or blocks[existing_index].type != "toolCall":
                block = ToolCall(call_id, function.get("name") or "", {})
                blocks.append(block)
                existing_index = len(blocks) - 1
                tool_blocks_by_key[key] = existing_index
                parsers[existing_index] = StreamingJsonParser()
                out.push(ToolCallStartEvent(existing_index, output))
            block = blocks[existing_index]
            raw_arguments = function.get("arguments")
            args_delta = raw_arguments if isinstance(raw_arguments, str) else fastjson.dumps(raw_arguments or {})
            parser = parsers[existing_index]
            parser.feed(args_delta)
            block.arguments = parser.value()  # type: ignore[union-attr]
            out.push(ToolCallDeltaEvent(existing_index, args_delta, output))

    finish_current()
    for index in tool_blocks_by_key.values():
        block = blocks[index]
        if block.type != "toolCall":
            continue
        arguments, arguments_error = parse_tool_call_arguments(parsers[index].text)
        block.arguments = arguments  # type: ignore[union-attr]
        if arguments_error is not None:
            block.arguments_error = arguments_error  # type: ignore[union-attr]
        out.push(ToolCallEndEvent(index, block, output))  # type: ignore[arg-type]
