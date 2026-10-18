# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""pi-messages API implementation.

Streams pi's own message protocol directly to a backend: the request is a single
POST of ``{model, context, options}`` to ``<base_url>/messages``, the response is
an SSE stream of serialized assistant-message events plus a terminal
``done``/``error`` event. This is the wire protocol spoken by the Radius gateway,
but any backend implementing it can be used, e.g. via a models.json custom
provider with ``"api": "pi-messages"``.

``context`` is sent in pi's camelCase JSON (:func:`pi_ai.serialization.to_json_value`),
so the wire format is the one the TypeScript implementation produces.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal, TypeAlias, TypedDict

import httpx

from ..abort import abort_scope
from ..event_stream import AssistantMessageEventStream
from ..serialization import from_dict, snake_case, to_json_value
from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    CacheRetention,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
    now_ms,
)
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.diagnostics import (
    AssistantMessageDiagnostic,
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
)
from ..utils.http import ProviderHttpError, build_headers, send_request
from ..utils.json_parse import StreamingJsonParser
from ..utils.misc import get_provider_env_value, headers_to_record, provider_headers_to_record
from ..utils.text import append_text, append_thinking

__all__ = [
    "PiMessagesOptions",
    "PiMessagesToolChoice",
    "PiMessagesRewriteImpact",
    "PiMessagesEvent",
    "PiMessagesResponseError",
    "stream",
    "stream_simple",
]

PiMessagesToolChoice: TypeAlias = "Literal['auto', 'none', 'required'] | dict[str, Any]"


class PiMessagesOptions(StreamOptions, total=False):
    reasoning: ThinkingLevel
    #: ``"auto" | "none" | "required"`` or ``{"type": "function", "function": {"name": ...}}``.
    tool_choice: PiMessagesToolChoice
    #: Ask the backend for debug metadata (e.g. routing response headers).
    debug: bool


class PiMessagesRewriteImpact(TypedDict):
    """Impact summary of a server-side message rewrite (e.g. a gateway policy). Wire JSON, camelCase keys."""

    policyId: str
    policyVersion: int
    changed: bool
    tokenCountChange: int
    messageCountChange: int
    systemPromptChanged: bool


#: Serialized assistant-message event as sent by a pi-messages backend (camelCase wire JSON):
#:
#: * ``{"type": "start"}``
#: * ``{"type": "text_start" | "thinking_start", "contentIndex": n}``
#: * ``{"type": "text_delta" | "thinking_delta" | "toolcall_delta", "contentIndex": n, "delta": str}``
#: * ``{"type": "text_end", "contentIndex": n, "content": str, "contentSignature"?: str}``
#: * ``{"type": "thinking_end", "contentIndex": n, "content": str, "contentSignature"?: str, "redacted"?: bool}``
#: * ``{"type": "toolcall_start", "contentIndex": n, "id": str, "toolName": str}``
#: * ``{"type": "toolcall_end", "contentIndex": n, "toolCall": {...}}``
#: * ``{"type": "done", "reason": "stop" | "length" | "toolUse", "usage": {...}, "responseId"?: str,
#:   "providerThinkingLevel"?: str, "rewrite"?: PiMessagesRewriteImpact}``
#: * ``{"type": "error", "reason": "aborted" | "error", "usage": {...}, "errorMessage"?: str, "responseId"?: str,
#:   "providerThinkingLevel"?: str, "rewrite"?: PiMessagesRewriteImpact}``
PiMessagesEvent: TypeAlias = dict[str, Any]

_MAX_DIAGNOSTIC_STRING = 8192


class PiMessagesResponseError(Exception):
    """A non-2xx pi-messages response. ``diagnostic_details`` is attached to the error message's diagnostics."""

    def __init__(self, message: str, code: str | None, diagnostic_details: dict[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic_details = diagnostic_details


def _parse_error_body(body: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        return parsed
    return None


def _truncate_diagnostic_string(value: str) -> str:
    return f"{value[:_MAX_DIAGNOSTIC_STRING]}…" if len(value) > _MAX_DIAGNOSTIC_STRING else value


def _create_response_error(model: Model, url: str, status: int, body: str) -> PiMessagesResponseError:
    error_body = _parse_error_body(body)
    error = error_body["error"] if error_body is not None else None
    message = error.get("message") if error is not None else None
    code = error.get("code") if error is not None else None
    message = message if isinstance(message, str) else None
    code = code if isinstance(code, str) else None
    status_text = httpx.codes.get_reason_phrase(status)
    suffix = message if message is not None else body
    code_suffix = f" ({code})" if code else ""
    details: dict[str, Any] = {
        "version": 1,
        "provider": model.provider,
        "model": model.id,
        "url": url,
        "status": status,
        "statusText": status_text,
    }
    if error is not None:
        details["error"] = error
    else:
        details["body"] = _truncate_diagnostic_string(body)
    details["timestampMs"] = now_ms()
    return PiMessagesResponseError(f"{status} {status_text}: {suffix}{code_suffix}", code, details)


def _append_rewrite_diagnostic(message: AssistantMessage, rewrite: Any) -> None:
    if not rewrite:
        return
    append_assistant_message_diagnostic(
        message,
        AssistantMessageDiagnostic(type="pi_messages_rewrite", timestamp=now_ms(), details=dict(rewrite)),
    )


def _usage(value: Any) -> Usage:
    return from_dict(Usage, value) if isinstance(value, Mapping) else Usage()


#: Wire ``toolCall`` keys that are not attributes of :class:`ToolCall`.
_TOOL_CALL_SKIP = frozenset({"type"})
_TOOL_CALL_FIELDS = frozenset(ToolCall.__slots__)  # type: ignore[attr-defined]


class _EventConverter:
    """Folds wire events into one live partial message and returns the matching pi event."""

    __slots__ = ("_blocks", "_tool_json", "partial")

    def __init__(self, model: Model) -> None:
        self.partial = AssistantMessage(
            [],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=now_ms(),
        )
        # Blocks by wire content index (pi's content array may be addressed sparsely).
        self._blocks: dict[int, Any] = {}
        self._tool_json: dict[int, StreamingJsonParser] = {}

    def _set_block(self, index: int, block: Any) -> None:
        content = self.partial.content
        if 0 <= index < len(content):
            content[index] = block
        else:
            content.append(block)
        self._blocks[index] = block

    def _finish(self, event: PiMessagesEvent) -> None:
        partial = self.partial
        partial.stop_reason = event.get("reason")  # type: ignore[assignment]
        partial.usage = _usage(event.get("usage"))
        partial.response_id = event.get("responseId")
        provider_thinking_level = event.get("providerThinkingLevel")
        if provider_thinking_level is not None:
            partial.provider_thinking_level = provider_thinking_level
        _append_rewrite_diagnostic(partial, event.get("rewrite"))

    def convert(self, event: PiMessagesEvent) -> AssistantMessageEvent | None:
        partial = self.partial
        kind = event.get("type")
        if kind == "done":
            self._finish(event)
            return DoneEvent(event.get("reason"), partial)  # type: ignore[arg-type]
        if kind == "error":
            self._finish(event)
            partial.error_message = event.get("errorMessage")
            return ErrorEvent(event.get("reason"), partial)  # type: ignore[arg-type]
        if kind == "start":
            return StartEvent(partial)
        index = event.get("contentIndex")
        if kind == "text_start":
            self._set_block(index, TextContent(""))
            return TextStartEvent(index, partial)
        if kind == "text_delta":
            delta = event["delta"]
            append_text(self._blocks[index], delta)
            return TextDeltaEvent(index, delta, partial)
        if kind == "text_end":
            block = self._blocks[index]
            block.text = event["content"]
            block.text_signature = event.get("contentSignature")
            return TextEndEvent(index, event["content"], partial)
        if kind == "thinking_start":
            self._set_block(index, ThinkingContent(""))
            return ThinkingStartEvent(index, partial)
        if kind == "thinking_delta":
            delta = event["delta"]
            append_thinking(self._blocks[index], delta)
            return ThinkingDeltaEvent(index, delta, partial)
        if kind == "thinking_end":
            block = self._blocks[index]
            block.thinking = event["content"]
            block.thinking_signature = event.get("contentSignature")
            block.redacted = event.get("redacted")
            return ThinkingEndEvent(index, event["content"], partial)
        if kind == "toolcall_start":
            self._set_block(index, ToolCall(event.get("id"), event.get("toolName"), {}))  # type: ignore[arg-type]
            self._tool_json[index] = StreamingJsonParser()
            return ToolCallStartEvent(index, partial)
        if kind == "toolcall_delta":
            delta = event["delta"]
            parser = self._tool_json.get(index)
            if parser is None:
                parser = self._tool_json[index] = StreamingJsonParser()
            parser.feed(delta)
            self._blocks[index].arguments = parser.value()
            return ToolCallDeltaEvent(index, delta, partial)
        if kind == "toolcall_end":
            block = self._blocks[index]
            for key, value in (event.get("toolCall") or {}).items():
                if key in _TOOL_CALL_SKIP:
                    continue
                name = snake_case(key)
                if name in _TOOL_CALL_FIELDS:
                    setattr(block, name, value)
            self._tool_json.pop(index, None)  # pyright: ignore[reportCallIssue]
            return ToolCallEndEvent(index, block, partial)
        # Unknown event types have no pi event to map to.
        return None


def _parse_event(raw: str) -> PiMessagesEvent | None:
    data: str | None = None
    for line in raw.split("\n"):
        if line.startswith("data:"):
            data = line[5:].strip()
            break
    if not data or data == "[DONE]":
        return None
    return fastjson.loads(data)


async def _read_events(response: httpx.Response) -> AsyncIterator[PiMessagesEvent]:
    buffer = ""
    async for chunk in response.aiter_text():
        buffer = (buffer + chunk).replace("\r\n", "\n")
        split = buffer.find("\n\n")
        while split != -1:
            event = _parse_event(buffer[:split])
            if event is not None:
                yield event
            buffer = buffer[split + 2 :]
            split = buffer.find("\n\n")
    if buffer.strip():
        event = _parse_event(buffer)
        if event is not None:
            yield event


def _create_error_event(model: Model, error: Exception, aborted: bool) -> ErrorEvent:
    reason: Literal["aborted", "error"] = "aborted" if aborted else "error"
    message = AssistantMessage(
        [],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason=reason,
        error_message=str(error),
        timestamp=now_ms(),
    )
    if not aborted and isinstance(error, PiMessagesResponseError):
        append_assistant_message_diagnostic(
            message,
            create_assistant_message_diagnostic("pi_messages_response_failure", error, error.diagnostic_details),
        )
    return ErrorEvent(reason, message)


def _resolve_cache_retention(
    cache_retention: CacheRetention | None, env: Mapping[str, str] | None
) -> CacheRetention | None:
    if cache_retention:
        return cache_retention
    # Backend defaults apply when unset; only the legacy env opt-in is mapped.
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else None


def _messages_url(model: Model, debug: bool) -> str:
    url = f"{model.base_url.rstrip('/')}/messages"
    if debug:
        url = str(httpx.URL(url).copy_set_param("debug", "1"))
    return url


async def _notify_response(options: Mapping[str, Any], model: Model, status: int, headers: Any) -> None:
    on_response = options.get("on_response")
    if on_response is not None:
        await maybe_await(on_response(ProviderResponse(status, headers_to_record(headers or {})), model))


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a response from a pi-messages backend (``POST <base_url>/messages``)."""
    opts: dict[str, Any] = dict(options or {})
    out = AssistantMessageEventStream()
    out.start(_run(model, context, opts, out, _EventConverter(model)))
    return out


async def _run(
    model: Model,
    context: TranscriptContext,
    options: dict[str, Any],
    out: AssistantMessageEventStream,
    converter: _EventConverter,
) -> None:
    signal = options.get("signal")
    try:
        async with abort_scope(signal):
            api_key = options.get("api_key")
            if not api_key:
                raise ValueError(f'No API key provided for provider "{model.provider}"')

            url = _messages_url(model, bool(options.get("debug")))
            request_options = {
                "temperature": options.get("temperature"),
                "maxTokens": options.get("max_tokens"),
                "reasoning": options.get("reasoning"),
                "cacheRetention": _resolve_cache_retention(options.get("cache_retention"), options.get("env")),
                "sessionId": options.get("session_id"),
                "toolChoice": options.get("tool_choice"),
            }
            payload: Any = {
                "model": model.id,
                "context": to_json_value(context),
                "options": {key: value for key, value in request_options.items() if value is not None},
            }
            on_payload = options.get("on_payload")
            if on_payload is not None:
                next_payload = await maybe_await(on_payload(payload, model))
                if next_payload is not None:
                    payload = next_payload

            headers = build_headers(
                {
                    "authorization": f"Bearer {api_key}",
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                },
                provider_headers_to_record(options.get("headers")),
            )
            try:
                # The backend owns retries: the request is sent once, like upstream's plain fetch.
                response = await send_request(
                    "POST",
                    url,
                    headers=headers,
                    body=payload,
                    options={**options, "max_retries": 0},
                    error_style="raw",
                )
            except ProviderHttpError as error:
                if error.status is None:
                    raise
                await _notify_response(options, model, error.status, error.headers)
                raise _create_response_error(model, url, error.status, error.raw_body or "") from error
            try:
                await _notify_response(options, model, response.status_code, response.headers)
                if not response.is_success:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise _create_response_error(model, url, response.status_code, body)
                on_event = options.get("on_provider_stream_event")
                async for wire_event in _read_events(response):
                    if on_event is not None:
                        await maybe_await(on_event(wire_event, model))
                    event = converter.convert(wire_event)
                    if event is None:
                        continue
                    out.push(event)
                    if event.type == "done" or event.type == "error":
                        out.end()
                        return
            finally:
                await response.aclose()
        raise RuntimeError(f"{model.provider} stream ended without a terminal event")
    except Exception as error:
        out.push(_create_error_event(model, error, signal is not None and signal.aborted))
        out.end()


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Simple options pass straight through: ``reasoning``, ``tool_choice`` and ``debug`` are forwarded as is."""
    return stream(model, context, options)
