"""pi-ai telemetry: the ``pi.ai`` schema and the ``pi.ai.request`` span.

Every request option surface accepts ``telemetry_context`` (a :class:`pi_telemetry.TelemetryContext`), the
explicit parent of the telemetry a request produces; :func:`pi_ai.api.simple_options.build_base_options` passes
it on to providers, as upstream pi-ai does.

Beyond upstream, :class:`pi_ai.Models` emits one ``pi.ai.request`` span per chat request (``stream``,
``stream_simple``, ``complete``, ``complete_simple``, ``stream_deferred`` and ``fetch_deferred``), so every provider
is covered. The provider then receives the request span as its ``telemetry_context``: provider-level spans nest
under it, and the HTTP retry path records ``pi.ai.retry`` events on it. Without a ``telemetry_context`` (or with
:data:`pi_telemetry.NOOP_TELEMETRY_CONTEXT`) nothing is built or recorded.

By default the span records no content: no prompt, completion, thinking, tool arguments, headers or error text.
Under :func:`pi_telemetry.with_content_capture` it adds ``pi.content`` events for the request's messages and the
response.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Final, Literal

from pi_telemetry import (
    CONTENT_EVENT,
    CONTENT_EVENT_NAME,
    NOOP_TELEMETRY_CONTEXT,
    SchemaTelemetrySpan,
    SpanAttributes,
    SpanOptions,
    SpanStatus,
    TelemetryContext,
    TelemetrySchemaDefinition,
    TelemetrySpan,
    TypedSpanStarter,
    bind_typed_span_starter,
    create_typed_span_starter,
    define_telemetry_schema,
    record_content,
)

from .event_stream import _UNSET as _UNSET_RESULT
from .event_stream import AssistantMessageEventStream, EventStream
from .types import AssistantMessage, ErrorEvent, ProviderResponse, now_ms

if TYPE_CHECKING:
    from pi_telemetry import ContentCapture

    from .types import Model, TranscriptContext

__all__ = [
    "AI_TELEMETRY_SCHEMA",
    "AI_TELEMETRY_SCHEMAS",
    "AI_REQUEST_SPAN",
    "AiRequestSpan",
    "AiRequestOperation",
    "ai_request_span_of",
    "start_ai_span",
    "trace_request",
    "record_retry",
    "record_http_status",
]

AI_REQUEST_SPAN: Final = "pi.ai.request"

AiRequestOperation = Literal["stream", "stream_simple", "fetch_deferred"]

#: Error kinds of a failed request: the span status's error name and ``pi.ai.error_kind``.
AI_ERROR_KINDS: Final = ("aborted", "cancelled", "context_overflow", "rate_limited", "http_error", "provider_error")

AI_TELEMETRY_SCHEMA: Final[TelemetrySchemaDefinition] = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            AI_REQUEST_SPAN: {
                "description": "One model request through pi_ai.Models, from dispatch to the terminal message",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "pi.ai.provider": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "Provider ID",
                        "examples": ["anthropic", "openai"],
                    },
                    "pi.ai.api": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "API implementation",
                        "examples": ["anthropic-messages", "openai-responses"],
                    },
                    "pi.ai.model": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "Requested model ID",
                    },
                    "pi.ai.operation": {
                        "type": "string",
                        "required": True,
                        "values": ["stream", "stream_simple", "fetch_deferred"],
                        "cardinality": "low",
                        "description": "Models entry point: full options, provider-neutral options, or a deferred fetch",
                    },
                    "pi.ai.streaming": {
                        "type": "boolean",
                        "required": True,
                        "description": "Whether the caller consumes events (stream*) or only the result (complete*, fetch_deferred)",
                    },
                    "pi.ai.reasoning": {
                        "type": "string",
                        "required": False,
                        "values": ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
                        "cardinality": "low",
                        "description": "Requested thinking level (stream_simple)",
                    },
                    "pi.ai.tool_count": {
                        "type": "number",
                        "required": True,
                        "description": "Tools declared to the model",
                    },
                    "pi.ai.message_count": {
                        "type": "number",
                        "required": True,
                        "description": "Transcript messages sent",
                    },
                    "pi.ai.session_id": {
                        "type": "string",
                        "required": False,
                        "cardinality": "high",
                        "description": "The request's session_id option, when set",
                    },
                    "pi.ai.request.max_tokens": {
                        "type": "number",
                        "required": False,
                        "description": "Requested output token limit",
                    },
                    "pi.ai.request.temperature": {
                        "type": "number",
                        "required": False,
                        "description": "Requested temperature",
                    },
                },
                "endAttributes": {
                    "pi.ai.usage.input_tokens": {"type": "number", "description": "Input tokens"},
                    "pi.ai.usage.output_tokens": {"type": "number", "description": "Output tokens"},
                    "pi.ai.usage.cache_read_tokens": {"type": "number", "description": "Input tokens read from cache"},
                    "pi.ai.usage.cache_write_tokens": {
                        "type": "number",
                        "description": "Input tokens written to cache",
                    },
                    "pi.ai.usage.reasoning_tokens": {
                        "type": "number",
                        "description": "Output tokens spent on reasoning, when the provider reports them",
                    },
                    "pi.ai.cost": {"type": "number", "description": "Estimated total cost in USD"},
                    "pi.ai.stop_reason": {
                        "type": "string",
                        "values": ["stop", "length", "toolUse", "error", "aborted", "deferred", "pending"],
                        "cardinality": "low",
                        "description": "Stop reason of the terminal message",
                    },
                    "pi.ai.error_kind": {
                        "type": "string",
                        "values": list(AI_ERROR_KINDS),
                        "cardinality": "low",
                        "description": "Why the request failed; never the provider's error text",
                    },
                    "pi.ai.time_to_first_token_ms": {
                        "type": "number",
                        "description": "Milliseconds from the start of the request to its first text, thinking or tool-call delta",
                    },
                    "pi.ai.duration_ms": {"type": "number", "description": "Milliseconds until the terminal message"},
                    "pi.ai.http.status": {
                        "type": "number",
                        "description": "HTTP status of the last provider response, when the provider reports one",
                    },
                    "pi.ai.response.model": {
                        "type": "string",
                        "cardinality": "low",
                        "description": "Model ID the provider reports in its response",
                    },
                    "pi.ai.retry_count": {"type": "number", "description": "HTTP retries within the request"},
                },
                "events": {
                    "pi.ai.retry": {
                        "description": "A retryable HTTP failure; the request is sent again after the delay",
                        "attributes": {
                            "pi.ai.retry.attempt": {
                                "type": "number",
                                "required": True,
                                "description": "Retry number, starting at 1",
                            },
                            "pi.ai.retry.reason": {
                                "type": "string",
                                "required": True,
                                "values": ["http_status", "transport"],
                                "description": "An HTTP error status, or a connection failure or timeout",
                            },
                            "pi.ai.retry.delay_ms": {
                                "type": "number",
                                "required": True,
                                "description": "Backoff before the next attempt",
                            },
                            "pi.ai.retry.http_status": {
                                "type": "number",
                                "required": False,
                                "description": "The failed attempt's HTTP status",
                            },
                        },
                    },
                    "pi.ai.context_overflow": {
                        "description": "The request exceeded the model's context window",
                        "attributes": {
                            "pi.ai.context_window": {
                                "type": "number",
                                "required": False,
                                "description": "The model's context window in tokens",
                            },
                        },
                    },
                    "pi.ai.abort": {"description": "The request was aborted through its signal", "attributes": {}},
                    CONTENT_EVENT_NAME: CONTENT_EVENT,
                },
                "status": {"default": "ok", "errorWhen": "The terminal message's stop reason is error or aborted"},
            },
        },
    }
)

#: The schemas pi_ai owns, for typed starters and schema checks.
AI_TELEMETRY_SCHEMAS: Final = (AI_TELEMETRY_SCHEMA,)

#: Indexed once; bound to each request's parent context.
_STARTER: Final[TypedSpanStarter] = create_typed_span_starter(
    NOOP_TELEMETRY_CONTEXT, AI_TELEMETRY_SCHEMAS, on_invalid="warn"
)

_DELTA_EVENTS: Final = frozenset({"text_delta", "thinking_delta", "toolcall_delta"})


def start_ai_span(
    context: TelemetryContext,
    name: str,
    attributes: SpanAttributes,
    callback: Callable[[SchemaTelemetrySpan, TypedSpanStarter], Any],
) -> Awaitable[Any]:
    """Start a span of the ``pi.ai`` schema under ``context`` (schema violations warn and are dropped)."""
    return bind_typed_span_starter(context, _STARTER)(name, attributes, callback)


class AiRequestSpan:
    """The ``pi.ai.request`` span as providers receive it in ``options["telemetry_context"]``.

    A full :class:`~pi_telemetry.TelemetrySpan` (provider spans nest under it) that the HTTP layer recognizes, so
    retries and response statuses land on the request they belong to.
    """

    __slots__ = ("_span", "http_status", "retries")

    def __init__(self, span: SchemaTelemetrySpan) -> None:
        self._span = span
        self.http_status: int | None = None
        self.retries = 0

    @property
    def content_capture(self) -> ContentCapture | None:
        return self._span.content_capture

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], Any]) -> Awaitable[Any]:
        return self._span.start_span(options, callback)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        self._span.add_event(name, attributes)

    def set_attributes(self, attributes: SpanAttributes) -> None:
        self._span.set_attributes(attributes)

    def set_status(self, status: SpanStatus) -> None:
        self._span.set_status(status)

    def __repr__(self) -> str:
        return f"AiRequestSpan({self._span.span!r})"


def ai_request_span_of(context: Any) -> AiRequestSpan | None:
    """The ``pi.ai.request`` span a provider's ``telemetry_context`` is, or ``None``."""
    return context if isinstance(context, AiRequestSpan) else None


def record_retry(context: Any, attempt: int, delay_ms: float, status: int | None) -> None:
    """Record a ``pi.ai.retry`` event when ``context`` is a request span; cheap no-op otherwise."""
    span = ai_request_span_of(context)
    if span is None:
        return
    span.retries += 1
    attributes: dict[str, Any] = {
        "pi.ai.retry.attempt": attempt,
        "pi.ai.retry.reason": "transport" if status is None else "http_status",
        "pi.ai.retry.delay_ms": round(delay_ms),
    }
    if status is not None:
        attributes["pi.ai.retry.http_status"] = status
        span.http_status = status
    span.add_event("pi.ai.retry", attributes)


def record_http_status(context: Any, status: int | None) -> None:
    """Remember the HTTP status of the request's last provider response."""
    span = ai_request_span_of(context)
    if span is not None and status is not None:
        span.http_status = status


# ─── the request span ───────────────────────────────────────────────────────


def _role_kind(message: Any) -> str:
    role = getattr(message, "role", "message")
    return "tool_result" if role == "toolResult" else str(role)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(block.text)
        elif kind == "image":
            parts.append("[image]")
        elif kind == "thinking":
            continue
        elif kind == "toolCall":
            parts.append(f"[tool call {block.name}]")
    return "\n".join(parts)


def _record_prompt(span: SchemaTelemetrySpan, transcript: TranscriptContext) -> None:
    from .utils.text import get_system_message_text

    for index, message in enumerate(transcript.messages):
        role = getattr(message, "role", None)
        text = get_system_message_text(message) if role == "system" else _message_text(message)  # type: ignore[arg-type]
        name = getattr(message, "tool_name", None)
        record_content(span, _role_kind(message), text, index=index, name=name)


def _record_response(span: SchemaTelemetrySpan, message: AssistantMessage) -> None:
    for block in message.content:
        kind = block.type
        if kind == "text":
            record_content(span, "assistant", block.text)  # type: ignore[union-attr]
        elif kind == "thinking":
            record_content(span, "thinking", block.thinking)  # type: ignore[union-attr]
        elif kind == "toolCall":
            try:
                arguments = json.dumps(block.arguments, ensure_ascii=False, default=str)  # type: ignore[union-attr]
            except Exception:
                arguments = ""
            record_content(span, "tool_call", arguments, name=block.name)  # type: ignore[union-attr]
    if message.error_message:
        record_content(span, "error", message.error_message)


def _start_attributes(
    model: Model, transcript: TranscriptContext, options: Mapping[str, Any], operation: str, streaming: bool
) -> dict[str, Any]:
    from .utils.transcript import get_current_tools

    attributes: dict[str, Any] = {
        "pi.ai.provider": model.provider,
        "pi.ai.api": model.api,
        "pi.ai.model": model.id,
        "pi.ai.operation": operation,
        "pi.ai.streaming": streaming,
        "pi.ai.tool_count": len(get_current_tools(transcript.messages)),
        "pi.ai.message_count": len(transcript.messages),
    }
    reasoning = options.get("reasoning")
    if isinstance(reasoning, str):
        attributes["pi.ai.reasoning"] = reasoning
    elif operation == "stream_simple":
        attributes["pi.ai.reasoning"] = "off"
    session_id = options.get("session_id")
    if isinstance(session_id, str):
        attributes["pi.ai.session_id"] = session_id
    max_tokens = options.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
        attributes["pi.ai.request.max_tokens"] = max_tokens
    temperature = options.get("temperature")
    if isinstance(temperature, int | float) and not isinstance(temperature, bool):
        attributes["pi.ai.request.temperature"] = temperature
    return attributes


def _error_kind(message: AssistantMessage, overflow: bool, http_status: int | None) -> str:
    if message.stop_reason == "aborted":
        return "aborted"
    if overflow:
        return "context_overflow"
    if http_status == 429:
        return "rate_limited"
    if http_status is not None and http_status >= 400:
        return "http_error"
    return "provider_error"


def _finish(
    span: SchemaTelemetrySpan,
    request: AiRequestSpan,
    model: Model,
    message: AssistantMessage | None,
    started: float,
    first_token: float | None,
) -> None:
    attributes: dict[str, Any] = {}
    if first_token is not None:
        attributes["pi.ai.time_to_first_token_ms"] = round((first_token - started) * 1000, 3)
    if request.retries:
        attributes["pi.ai.retry_count"] = request.retries
    if request.http_status is not None:
        attributes["pi.ai.http.status"] = request.http_status
    if message is None:
        attributes["pi.ai.duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        span.set_attributes(attributes)
        return
    usage = message.usage
    attributes.update(
        {
            "pi.ai.usage.input_tokens": usage.input,
            "pi.ai.usage.output_tokens": usage.output,
            "pi.ai.usage.cache_read_tokens": usage.cache_read,
            "pi.ai.usage.cache_write_tokens": usage.cache_write,
            "pi.ai.cost": usage.cost.total,
            "pi.ai.stop_reason": message.stop_reason,
            "pi.ai.duration_ms": message.duration_ms
            if message.duration_ms is not None
            else round((time.perf_counter() - started) * 1000, 3),
        }
    )
    if usage.reasoning is not None:
        attributes["pi.ai.usage.reasoning_tokens"] = usage.reasoning
    if message.response_model:
        attributes["pi.ai.response.model"] = message.response_model
    from .utils.overflow import is_context_overflow

    context_window = getattr(model, "context_window", None)
    overflow = is_context_overflow(message, context_window or None)
    if overflow:
        overflow_attributes: dict[str, Any] = {}
        if context_window:
            overflow_attributes["pi.ai.context_window"] = context_window
        span.add_event("pi.ai.context_overflow", overflow_attributes)
    if message.stop_reason in ("error", "aborted"):
        kind = _error_kind(message, overflow, request.http_status)
        attributes["pi.ai.error_kind"] = kind
        if kind == "aborted":
            span.add_event("pi.ai.abort")
        span.set_status({"status": "error", "error": {"name": kind, "message": ""}})
    span.set_attributes(attributes)
    if span.content_capture is not None:
        _record_response(span, message)


def _with_status_hook(options: dict[str, Any], request: AiRequestSpan) -> dict[str, Any]:
    user = options.get("on_response")

    def on_response(response: ProviderResponse, model: Any) -> Any:
        request.http_status = response.status
        return None if user is None else user(response, model)

    options["on_response"] = on_response
    options["telemetry_context"] = request
    return options


class _TracedStream(AssistantMessageEventStream):
    """The caller's stream: events are forwarded into it without an extra hop, while it notes the first delta.

    The span is woken *before* the terminal message reaches consumers: its wake-up is scheduled ahead of theirs,
    so the span has settled by the time a caller awaiting the result resumes.
    """

    __slots__ = ("_failure", "_finished", "_first_token", "_span_settled", "_terminal")

    def __init__(self) -> None:
        super().__init__()
        self._first_token: float | None = None
        self._finished: Any = None
        self._terminal: AssistantMessage | None = None
        self._failure = False
        #: Resolves once the span settled; ``result()`` waits for it, so a caller sees a finished span.
        self._span_settled: asyncio.Future[None] | None = None

    async def result(self) -> AssistantMessage:
        value = await AssistantMessageEventStream.result(self)
        settled = self._span_settled
        if settled is not None and not settled.done():
            await asyncio.shield(settled)
        return value

    def push(self, event: Any) -> None:
        kind = event.type
        if self._first_token is None and kind in _DELTA_EVENTS:
            self._first_token = time.perf_counter()
        if not self._done and (kind == "done" or kind == "error"):
            self._terminal = event.message if kind == "done" else event.error
            self._wake()
        AssistantMessageEventStream.push(self, event)

    def end(self, result: AssistantMessage | None = None) -> None:
        if not self._done and result is not None:
            self._terminal = result
        self._wake()
        AssistantMessageEventStream.end(self, result)

    def _fail(self, error: BaseException) -> None:
        if not (self._done and self._result is not _UNSET_RESULT):
            self._failure = True
        self._wake()
        EventStream._fail(self, error)

    def _wake(self) -> None:
        finished = self._finished
        if finished is not None and not finished.done():
            finished.set_result(None)


def trace_request(
    model: Model,
    transcript: TranscriptContext,
    options: Mapping[str, Any],
    operation: AiRequestOperation,
    streaming: bool,
    setup: Callable[[Callable[[dict[str, Any]], dict[str, Any]]], Awaitable[AsyncIterable[Any]]],
) -> AssistantMessageEventStream:
    """``lazy_stream(model, setup)`` inside a ``pi.ai.request`` span under ``options["telemetry_context"]``.

    ``setup(instrument)`` opens the provider stream; it passes its final request options through ``instrument``,
    which points their ``telemetry_context`` at the request span.
    """
    from .api.lazy import setup_error_message

    telemetry: TelemetryContext = options["telemetry_context"]
    attributes = _start_attributes(model, transcript, options, operation, streaming)
    outer = _TracedStream()
    started_at = now_ms()

    async def body(span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> None:
        started = time.perf_counter()
        request = AiRequestSpan(span)
        if span.content_capture is not None:
            _record_prompt(span, transcript)
        finished = outer._finished = asyncio.get_running_loop().create_future()
        try:
            try:
                inner = await setup(lambda request_options: _with_status_hook(request_options, request))
            except Exception as error:
                message = setup_error_message(model, error, started_at)
                outer.push(ErrorEvent(message.stop_reason, message))  # type: ignore[arg-type]
                outer.end(message)
                _finish(span, request, model, message, started, None)
                return
            if isinstance(inner, EventStream):
                inner.forward_to(outer)
            else:
                await _pump(inner, outer, model, started_at)
            await finished
        except BaseException:
            # The caller closed the stream (or the task was cancelled): no terminal message.
            span.set_attributes({"pi.ai.error_kind": "cancelled"})
            span.set_status({"status": "error", "error": {"name": "cancelled", "message": ""}})
            _finish(span, request, model, None, started, outer._first_token)
            raise
        message = outer._terminal
        if message is None and outer._failure:
            span.set_attributes({"pi.ai.error_kind": "provider_error"})
            span.set_status({"status": "error", "error": {"name": "provider_error", "message": ""}})
        _finish(span, request, model, message, started, outer._first_token)

    async def run() -> None:
        settled = outer._span_settled = asyncio.get_running_loop().create_future()
        try:
            await start_ai_span(telemetry, AI_REQUEST_SPAN, attributes, body)
        finally:
            settled.set_result(None)

    outer.start(run())
    return outer


async def _pump(inner: AsyncIterable[Any], outer: AssistantMessageEventStream, model: Any, started_at: int) -> None:
    from .api.lazy import setup_error_message

    try:
        async for event in inner:
            outer.push(event)
    except Exception as error:
        message = setup_error_message(model, error, started_at)
        outer.push(ErrorEvent(message.stop_reason, message))  # type: ignore[arg-type]
    result: Callable[[], Awaitable[Any]] | None = getattr(inner, "result", None)
    outer.end(await result() if result is not None else None)
