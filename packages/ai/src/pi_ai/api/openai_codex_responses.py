"""OpenAI Codex Responses API (``openai-codex-responses``): the ChatGPT subscription Codex backend.

Two transports:

* SSE over httpx (``POST {base_url}/codex/responses``). Request bodies are zstd-compressed
  when a zstd codec is available (Python 3.14 ``compression.zstd`` or the ``zstandard``
  package), like the official Codex client.
* WebSocket (``transport="websocket"``, ``"websocket-cached"`` or ``"auto"``, the default),
  using the optional ``websockets`` package (``pip install "pi-ai-python[websocket]"``). Connections are
  pooled per session and account; ``websocket-cached``/``auto`` send only the new input items
  with ``previous_response_id`` when the request extends the previous one. Failures before the
  first event fall back to SSE, and the session then stays on SSE.

The WebSocket connect function can be replaced by assigning :data:`websocket_connect`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import email.utils
import importlib.util
import inspect
import json
import math
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import httpx

from ..abort import abort_scope
from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..session_resources import register_session_resource_cleanup
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
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    format_thrown_value,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.http import DEFAULT_TIMEOUT_MS, get_http_client, iter_sse
from ..utils.misc import get_pi_user_agent, get_provider_env_value, headers_to_record, uuidv7
from ..utils.text import get_system_message_text
from ..utils.transcript import (
    get_declared_tools,
    get_initial_system_message,
    resolve_transcript,
    resolve_transcript_tools,
)
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import convert_responses_messages, convert_responses_tools, process_responses_stream
from .simple_options import build_base_options

__all__ = [
    "OpenAICodexResponsesOptions",
    "OpenAICodexWebSocketDebugStats",
    "CodexWebSocket",
    "CodexWebSocketClosed",
    "WebSocketConnect",
    "CodexApiError",
    "CodexProtocolError",
    "ProviderStreamEventCallbackError",
    "WebSocketCloseError",
    "websocket_connect",
    "stream",
    "stream_simple",
    "get_openai_codex_websocket_debug_stats",
    "reset_openai_codex_websocket_debug_stats",
    "close_openai_codex_websocket_sessions",
]

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_MAX_RETRIES = 0
BASE_DELAY_MS = 1000
DEFAULT_MAX_RETRY_DELAY_MS = 60_000
DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS = 15_000
# The Codex backend accepts zstd-compressed request bodies on the SSE responses
# endpoint (the same endpoint the official Codex client compresses against).
REQUEST_COMPRESSION_ZSTD_LEVEL = 3
CODEX_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})
WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE = 1009
WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
PREVIOUS_RESPONSE_NOT_FOUND_CODE = "previous_response_not_found"
OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
SESSION_WEBSOCKET_CACHE_TTL_MS = 5 * 60 * 1000
SESSION_WEBSOCKET_MAX_AGE_MS = 55 * 60 * 1000
WEBSOCKET_UNAVAILABLE_MESSAGE = (
    "WebSocket transport requires the optional 'websockets' package. Install it with: "
    'pip install "pi-ai-python[websocket]"'
)

_CODEX_RESPONSE_STATUSES = frozenset({"completed", "incomplete", "failed", "cancelled", "queued", "in_progress"})
_TERMINAL_EVENT_TYPES = frozenset({"response.done", "response.completed", "response.incomplete"})
_SSE_TIMEOUT = httpx.Timeout(DEFAULT_TIMEOUT_MS / 1000)

# ============================================================================
# Types
# ============================================================================


class OpenAICodexResponsesOptions(StreamOptions, total=False):
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    reasoning_summary: Literal["auto", "concise", "detailed", "off", "on"] | None
    service_tier: str
    text_verbosity: Literal["low", "medium", "high"]
    tool_choice: Literal["auto", "none", "required"]


def _assert_successful_output(output: AssistantMessage) -> None:
    if output.stop_reason == "pending":
        raise RuntimeError("Codex stream ended without a stop reason")
    if output.stop_reason in ("error", "aborted"):
        raise RuntimeError(output.error_message or "An unknown error occurred")


class CodexApiError(Exception):
    """An error reported by the Codex backend (``error`` or ``response.failed`` events)."""

    def __init__(self, message: str, *, code: str | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload


class CodexProtocolError(Exception):
    """A malformed Codex stream frame."""

    def __init__(self, message: str, *, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


class ProviderStreamEventCallbackError(Exception):
    """``on_provider_stream_event`` raised; kept out of the WebSocket retry and SSE fallback path."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(format_thrown_value(cause))
        self.cause = cause


class RetryDelayExceededError(Exception):
    pass


class WebSocketCloseError(Exception):
    def __init__(
        self, message: str, *, code: int | None = None, reason: str | None = None, was_clean: bool | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.was_clean = was_clean


class CodexWebSocketClosed(Exception):
    """Raised by :meth:`CodexWebSocket.recv` when the connection is closed."""

    def __init__(self, code: int | None = None, reason: str = "", was_clean: bool | None = None) -> None:
        super().__init__(f"WebSocket closed {code} {reason}".strip())
        self.code = code
        self.reason = reason
        self.was_clean = was_clean


class CodexWebSocket(Protocol):
    """The WebSocket connection interface the transport uses.

    ``send`` and ``close`` may be sync or async. ``recv`` returns the next text or binary
    message and raises :class:`CodexWebSocketClosed` when the connection closes. An optional
    integer ``ready_state`` attribute (WebSocket ``readyState`` values, 1 = open) decides
    whether a pooled connection can be reused; without it the connection is assumed open.
    """

    def send(self, data: str) -> Any: ...

    async def recv(self) -> str | bytes: ...

    def close(self, code: int = 1000, reason: str = "") -> Any: ...


#: ``(url, headers, env) -> connection``. Connect timeouts and aborts are applied by the caller.
WebSocketConnect = Callable[[str, dict[str, str], "Mapping[str, str] | None"], Awaitable[CodexWebSocket]]

#: Replace to supply a custom WebSocket implementation (``None`` uses the ``websockets`` package).
websocket_connect: WebSocketConnect | None = None


def _clock_ms() -> float:
    return time.time() * 1000


async def _sleep(delay_ms: float) -> None:
    await asyncio.sleep(max(0.0, delay_ms) / 1000)


# ============================================================================
# Retry Helpers
# ============================================================================

_TERMINAL_RATE_LIMIT_RE = re.compile(
    r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota"
    r"|out of budget|quota exceeded|billing",
    re.IGNORECASE,
)
_RETRYABLE_TEXT_RE = re.compile(
    r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused", re.IGNORECASE
)
_USAGE_LIMIT_CODE_RE = re.compile(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", re.IGNORECASE)


def _is_terminal_rate_limit_error(error_text: str) -> bool:
    return _TERMINAL_RATE_LIMIT_RE.search(error_text) is not None


def _is_retryable_error(status: int, error_text: str) -> bool:
    if status == 429 and _is_terminal_rate_limit_error(error_text):
        return False
    if status in (429, 500, 502, 503, 504):
        return True
    return _RETRYABLE_TEXT_RE.search(error_text) is not None


def _js_number(text: str) -> float | None:
    """``Number(text)`` for header values (``None`` for NaN)."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    try:
        return float(stripped)
    except ValueError:
        return None


def _get_retry_after_delay_ms(headers: httpx.Headers) -> float | None:
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        millis = _js_number(retry_after_ms)
        if millis is not None and math.isfinite(millis):
            return max(0.0, millis)

    retry_after = headers.get("retry-after")
    if not retry_after:
        return None
    seconds = _js_number(retry_after)
    if seconds is not None and math.isfinite(seconds):
        return max(0.0, seconds * 1000)
    try:
        date = email.utils.parsedate_to_datetime(retry_after)
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, date.timestamp() * 1000 - _clock_ms())


def _validate_retry_delay_ms(delay_ms: float, options: Mapping[str, Any]) -> float:
    max_retry_delay_ms = options.get("max_retry_delay_ms")
    if max_retry_delay_ms is None:
        max_retry_delay_ms = DEFAULT_MAX_RETRY_DELAY_MS
    if max_retry_delay_ms > 0 and delay_ms > max_retry_delay_ms:
        raise RetryDelayExceededError(
            f"Server requested {math.ceil(delay_ms / 1000)}s retry delay (max: {math.ceil(max_retry_delay_ms / 1000)}s)"
        )
    return delay_ms


def _normalize_timeout_ms(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise RuntimeError(f"Invalid timeoutMs: {value}")
    return math.floor(value)


# ============================================================================
# Request Compression
# ============================================================================


def _compress_request_body_zstd(body: bytes) -> bytes | None:
    """zstd-compressed body bytes, or ``None`` when no zstd codec is available.

    Callers fall back to sending the uncompressed JSON when this returns ``None``.
    """
    try:
        from compression import zstd  # type: ignore[import-not-found]  # Python 3.14+

        return zstd.compress(body, level=REQUEST_COMPRESSION_ZSTD_LEVEL)
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import zstandard  # type: ignore[import-not-found]

        return zstandard.ZstdCompressor(level=REQUEST_COMPRESSION_ZSTD_LEVEL).compress(body)
    except ImportError:
        return None
    except Exception:
        return None


# ============================================================================
# Main Stream Function
# ============================================================================


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a response from the ChatGPT Codex backend."""
    opts: dict[str, Any] = dict(options or {})
    out = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, (model.compat or {}).get("supportsMidConvoSystemMessages"))
    out.start(_run(model, normalized_context, opts, out))
    return out


class _StartState:
    __slots__ = ("emitted", "out", "output")

    def __init__(self, out: AssistantMessageEventStream, output: AssistantMessage) -> None:
        self.out = out
        self.output = output
        self.emitted = False

    def emit(self) -> None:
        if not self.emitted:
            self.emitted = True
            self.out.push(StartEvent(self.output))


class _WebSocketAttempt:
    __slots__ = ("start", "started")

    def __init__(self, start: _StartState) -> None:
        self.start = start
        self.started = False

    def on_start(self) -> None:
        self.started = True
        self.start.emit()


async def _run(
    model: Model, context: TranscriptContext, options: dict[str, Any], out: AssistantMessageEventStream
) -> None:
    output = AssistantMessage(
        [],
        api="openai-codex-responses",
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
    )
    signal = options.get("signal")
    try:
        async with abort_scope(signal):
            await _run_request(model, context, options, out, output)
    except Exception as error:
        aborted = signal is not None and signal.aborted
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = (
            "Request was aborted" if aborted else format_provider_error(normalize_provider_error(error))
        )
        out.push(ErrorEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()


async def _run_request(
    model: Model,
    context: TranscriptContext,
    options: dict[str, Any],
    out: AssistantMessageEventStream,
    output: AssistantMessage,
) -> None:
    signal = options.get("signal")
    api_key = options.get("api_key")
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    account_id = _extract_account_id(api_key)
    grammar_tool_input_properties = create_grammar_tool_input_properties(
        get_declared_tools(context.messages), bool((model.compat or {}).get("supportsOpenAIGrammarTools"))
    )
    cache_session_id = None if options.get("cache_retention") == "none" else options.get("session_id")
    codex_session_id = clamp_openai_prompt_cache_key(cache_session_id)
    body: Any = _build_request_body(model, context, options, codex_session_id, grammar_tool_input_properties)
    on_payload = options.get("on_payload")
    if on_payload is not None:
        next_body = await maybe_await(on_payload(body, model))
        if next_body is not None:
            body = next_body
    websocket_request_id = codex_session_id or uuidv7()
    sse_headers = _build_sse_headers(model.headers, options.get("headers"), account_id, api_key, codex_session_id)
    websocket_headers = _build_websocket_headers(
        model.headers, options.get("headers"), account_id, api_key, websocket_request_id
    )
    body_bytes = fastjson.dumps_bytes(body)
    http_timeout_ms = _normalize_timeout_ms(options.get("timeout_ms"))
    websocket_connect_timeout_ms = _normalize_timeout_ms(options.get("websocket_connect_timeout_ms"))
    transport = options.get("transport") or "auto"
    start = _StartState(out, output)
    websocket_disabled_for_session = transport != "sse" and _is_websocket_sse_fallback_active(cache_session_id)
    if websocket_disabled_for_session:
        _record_websocket_sse_fallback(cache_session_id)

    if transport != "sse" and not websocket_disabled_for_session:
        connect = _resolve_websocket_connect()
        if connect is None and transport != "auto":
            raise RuntimeError(WEBSOCKET_UNAVAILABLE_MESSAGE)
        # Without a WebSocket implementation "auto" goes straight to SSE.
        if connect is not None:
            retried_websocket_connection_limit = False
            retried_missing_websocket_continuation = False
            while True:
                attempt = _WebSocketAttempt(start)
                try:
                    await _process_websocket_stream(
                        connect,
                        _resolve_codex_websocket_url(model.base_url),
                        body,
                        websocket_headers,
                        output,
                        out,
                        model,
                        attempt.on_start,
                        http_timeout_ms,
                        websocket_connect_timeout_ms,
                        cache_session_id,
                        account_id,
                        grammar_tool_input_properties,
                        options,
                    )
                    if signal is not None and signal.aborted:
                        raise RuntimeError("Request was aborted")
                    _assert_successful_output(output)
                    out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
                    out.end()
                    return
                except Exception as error:
                    aborted = signal is not None and signal.aborted
                    connection_limit_before_start = (
                        not attempt.started and _is_websocket_connection_limit_reached_error(error)
                    )
                    previous_response_not_found = _is_previous_response_not_found_error(error)
                    if not aborted and previous_response_not_found and not retried_missing_websocket_continuation:
                        retried_missing_websocket_continuation = True
                        continue
                    if not aborted and connection_limit_before_start and not retried_websocket_connection_limit:
                        retried_websocket_connection_limit = True
                        continue
                    if aborted or (_is_codex_non_transport_error(error) and not connection_limit_before_start):
                        raise
                    details: dict[str, Any] = {"configuredTransport": transport}
                    if not attempt.started:
                        details["fallbackTransport"] = "sse"
                    details["eventsEmitted"] = attempt.started
                    details["phase"] = (
                        "after_message_stream_start" if attempt.started else "before_message_stream_start"
                    )
                    details["requestBytes"] = len(body_bytes)
                    append_assistant_message_diagnostic(
                        output, create_assistant_message_diagnostic("provider_transport_failure", error, details)
                    )
                    _record_websocket_failure(cache_session_id, error)
                    if attempt.started:
                        raise
                    _record_websocket_sse_fallback(cache_session_id)
                    break

    # Compress the request body once for the SSE path. The Codex backend decodes
    # Content-Encoding: zstd; the WebSocket transport above sends the uncompressed
    # JSON frame, matching the official Codex client.
    compressed_body = _compress_request_body_zstd(body_bytes)
    if compressed_body is not None:
        sse_headers["content-encoding"] = "zstd"
    sse_body = compressed_body if compressed_body is not None else body_bytes

    # Fetch with retry logic for rate limits and transient errors
    url = _resolve_codex_url(model.base_url)
    client = get_http_client(url, options)
    response: httpx.Response | None = None
    last_error: Exception | None = None
    max_retries = options.get("max_retries")
    if max_retries is None:
        max_retries = DEFAULT_MAX_RETRIES
    on_response = options.get("on_response")

    for attempt_index in range(max_retries + 1):
        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        try:
            response = await _send_sse_request(client, url, sse_headers, sse_body, http_timeout_ms)
            if on_response is not None:
                await maybe_await(
                    on_response(ProviderResponse(response.status_code, headers_to_record(response.headers)), model)
                )
            if 200 <= response.status_code < 300:
                break

            try:
                error_text = (await response.aread()).decode("utf-8", "replace")
            finally:
                await response.aclose()
            if attempt_index < max_retries and _is_retryable_error(response.status_code, error_text):
                retry_after_delay_ms = _get_retry_after_delay_ms(response.headers)
                delay_ms = (
                    BASE_DELAY_MS * 2**attempt_index
                    if retry_after_delay_ms is None
                    else _validate_retry_delay_ms(retry_after_delay_ms, options)
                )
                await _sleep(delay_ms)
                continue

            # Parse error for friendly message on final attempt or non-retryable error
            message, friendly_message = _parse_error_response(response.status_code, response.reason_phrase, error_text)
            raise RuntimeError(friendly_message or message)
        except Exception as error:
            if response is not None and not response.is_closed:
                await response.aclose()
            last_error = error
            # Network errors are retryable
            if (
                attempt_index < max_retries
                and not isinstance(error, RetryDelayExceededError)
                and "usage limit" not in str(error)
            ):
                await _sleep(BASE_DELAY_MS * 2**attempt_index)
                continue
            raise

    if response is None or not (200 <= response.status_code < 300):
        raise last_error or RuntimeError("Failed after retries")

    start.emit()
    try:
        await _process_sse_stream(response, output, out, model, grammar_tool_input_properties, options)
    finally:
        await response.aclose()

    if signal is not None and signal.aborted:
        raise RuntimeError("Request was aborted")

    _assert_successful_output(output)
    out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
    out.end()


async def _send_sse_request(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], body: bytes, http_timeout_ms: int | None
) -> httpx.Response:
    request = client.build_request("POST", url, headers=headers, content=body, timeout=_SSE_TIMEOUT)
    if http_timeout_ms is None or http_timeout_ms <= 0:
        return await client.send(request, stream=True)
    timeout = asyncio.timeout(http_timeout_ms / 1000)
    try:
        async with timeout:
            return await client.send(request, stream=True)
    except TimeoutError:
        if timeout.expired():
            raise RuntimeError(f"Codex SSE response headers timed out after {http_timeout_ms}ms") from None
        raise


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
    return stream(model, context, base)


# ============================================================================
# Request Building
# ============================================================================


def _build_request_body(
    model: Model,
    context: TranscriptContext,
    options: Mapping[str, Any],
    cache_session_id: str | None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    compat = model.compat or {}
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), bool(compat.get("supportsOpenAIGrammarTools"))
        )
    supports_strict_mode = compat.get("supportsStrictMode")
    if supports_strict_mode is None:
        supports_strict_mode = True
    supports_openai_grammar_tools = bool(compat.get("supportsOpenAIGrammarTools"))
    supports_additional_tools = bool(compat.get("supportsAdditionalTools"))
    supports_tool_search = bool(compat.get("supportsToolSearch"))
    tool_options: dict[str, Any] = {
        "strict": None,
        "supports_strict_mode": supports_strict_mode,
        "supports_openai_grammar_tools": supports_openai_grammar_tools,
    }
    transcript_tools = resolve_transcript_tools(context.messages, supports_additional_tools or supports_tool_search)
    messages = convert_responses_messages(
        model,
        context,
        CODEX_TOOL_CALL_PROVIDERS,
        {
            "include_system_prompt": False,
            "grammar_tool_input_properties": grammar_tool_input_properties,
            "supports_mid_convo_system_messages": bool(compat.get("supportsMidConvoSystemMessages")),
            "supports_additional_tools": supports_additional_tools,
            "supports_tool_search": supports_tool_search,
            "tool_options": tool_options,  # type: ignore[typeddict-item]
        },
    )

    initial_system_message = get_initial_system_message(context.messages)
    instructions = get_system_message_text(initial_system_message) if initial_system_message else ""
    body: dict[str, Any] = {
        "model": model.id,
        "store": False,
        "stream": True,
        "instructions": instructions or "You are a helpful assistant.",
        "input": messages,
        "text": {"verbosity": options.get("text_verbosity") or "low"},
        "include": ["reasoning.encrypted_content"],
    }
    if cache_session_id is not None:
        body["prompt_cache_key"] = cache_session_id
    tool_choice = options.get("tool_choice")
    body["tool_choice"] = tool_choice if tool_choice is not None else "auto"
    body["parallel_tool_calls"] = True

    if options.get("temperature") is not None:
        body["temperature"] = options["temperature"]

    if options.get("service_tier") is not None:
        body["service_tier"] = options["service_tier"]

    if transcript_tools.request_tools:
        body["tools"] = convert_responses_tools(transcript_tools.request_tools, tool_options)  # type: ignore[arg-type]

    level_map = model.thinking_level_map
    reasoning_effort = options.get("reasoning_effort")
    if reasoning_effort is not None:
        effort: Any
        if reasoning_effort == "none":
            effort = "none" if level_map is None or "off" not in level_map else level_map["off"]
        else:
            mapped = (level_map or {}).get(reasoning_effort)
            effort = mapped if mapped is not None else reasoning_effort
        if effort is not None:
            summary = options.get("reasoning_summary")
            body["reasoning"] = {"effort": effort, "summary": summary if summary is not None else "auto"}
    elif model.reasoning and not (level_map is not None and "off" in level_map and level_map["off"] is None):
        off = (level_map or {}).get("off")
        body["reasoning"] = {"effort": off if off is not None else "none"}

    return body


def _get_service_tier_cost_multiplier(model: Model, service_tier: Any) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
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


def _resolve_codex_service_tier(response_service_tier: Any, request_service_tier: Any) -> Any:
    if response_service_tier == "default" and request_service_tier in ("flex", "priority"):
        return request_service_tier
    return response_service_tier if response_service_tier is not None else request_service_tier


_TRAILING_SLASHES = re.compile(r"/+$")


def _resolve_codex_url(base_url: str | None = None) -> str:
    raw = base_url if base_url and base_url.strip() else DEFAULT_CODEX_BASE_URL
    normalized = _TRAILING_SLASHES.sub("", raw)
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _resolve_codex_websocket_url(base_url: str | None = None) -> str:
    url = _resolve_codex_url(base_url)
    if url.startswith("https:"):
        return "wss:" + url[len("https:") :]
    if url.startswith("http:"):
        return "ws:" + url[len("http:") :]
    return url


# ============================================================================
# Response Processing
# ============================================================================


def _stream_options(
    model: Model, options: Mapping[str, Any], grammar_tool_input_properties: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "service_tier": options.get("service_tier"),
        "grammar_tool_input_properties": grammar_tool_input_properties,
        "resolve_service_tier": _resolve_codex_service_tier,
        "apply_service_tier_pricing": lambda usage, tier: _apply_service_tier_pricing(usage, tier, model),
    }


async def _aclose(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        await aclose()


async def _process_sse_stream(
    response: httpx.Response,
    output: AssistantMessage,
    out: AssistantMessageEventStream,
    model: Model,
    grammar_tool_input_properties: Mapping[str, str],
    options: Mapping[str, Any],
) -> None:
    events = _map_codex_events(_parse_sse(response), output, model, options.get("on_provider_stream_event"))
    try:
        await process_responses_stream(
            events,
            output,
            out,
            model,
            _stream_options(model, options, grammar_tool_input_properties),  # type: ignore[arg-type]
        )
    finally:
        await _aclose(events)


def _is_codex_non_transport_error(error: BaseException) -> bool:
    return isinstance(error, (CodexApiError, CodexProtocolError, ProviderStreamEventCallbackError))


def _is_websocket_connection_limit_reached_error(error: BaseException) -> bool:
    return isinstance(error, CodexApiError) and error.code == WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE


def _is_previous_response_not_found_error(error: BaseException) -> bool:
    return isinstance(error, CodexApiError) and error.code == PREVIOUS_RESPONSE_NOT_FOUND_CODE


def _extract_codex_event_error(event: Mapping[str, Any]) -> tuple[str | None, str | None]:
    nested = event.get("error")
    nested = nested if isinstance(nested, dict) else None
    code = event.get("code")
    if not isinstance(code, str):
        code = nested.get("code") if nested is not None and isinstance(nested.get("code"), str) else None
    message = event.get("message")
    if not isinstance(message, str):
        message = nested.get("message") if nested is not None and isinstance(nested.get("message"), str) else None
    return code, message


async def _map_codex_events(
    events: AsyncIterable[Any],
    output: AssistantMessage,
    model: Model,
    on_provider_stream_event: Callable[[Any, Any], Any] | None,
) -> AsyncIterator[Any]:
    try:
        async for event in events:
            if on_provider_stream_event is not None:
                try:
                    await maybe_await(on_provider_stream_event(event, model))
                except Exception as error:
                    # Keep callback failures out of Codex's WebSocket retry and SSE fallback path.
                    raise ProviderStreamEventCallbackError(error) from error
            event_type = event.get("type") if isinstance(event, dict) else None
            if not isinstance(event_type, str) or not event_type:
                continue

            if event_type == "error":
                code, message = _extract_codex_event_error(event)
                raise CodexApiError(
                    f"Codex error: {message or code or json.dumps(event, separators=(',', ':'))}",
                    code=code,
                    payload=event,
                )

            if event_type == "response.failed":
                response = event.get("response")
                error = response.get("error") if isinstance(response, dict) else None
                error = error if isinstance(error, dict) else {}
                raise CodexApiError(
                    error.get("message") or "Codex response failed", code=error.get("code"), payload=event
                )

            if event_type in _TERMINAL_EVENT_TYPES:
                response = event.get("response")
                normalized_response = response
                if isinstance(response, dict):
                    end_turn = response.get("end_turn")
                    if isinstance(end_turn, bool):
                        output.end_turn = end_turn
                    normalized_response = {**response, "status": _normalize_codex_status(response.get("status"))}
                yield {**event, "type": "response.completed", "response": normalized_response}
                return

            yield event
    finally:
        await _aclose(events)


def _normalize_codex_status(status: Any) -> str | None:
    if not isinstance(status, str):
        return None
    return status if status in _CODEX_RESPONSE_STATUSES else None


# ============================================================================
# SSE Parsing
# ============================================================================


async def _parse_sse(response: httpx.Response) -> AsyncIterator[Any]:
    async for sse in iter_sse(response):
        data = sse.data.strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = fastjson.loads(data)
        except ValueError as cause:
            raise CodexProtocolError(f"Invalid Codex SSE JSON: {format_thrown_value(cause)}", payload=data) from cause
        yield parsed


# ============================================================================
# WebSocket Parsing
# ============================================================================


@dataclass(slots=True)
class OpenAICodexWebSocketDebugStats:
    requests: int = 0
    connections_created: int = 0
    connections_reused: int = 0
    cached_context_requests: int = 0
    store_true_requests: int = 0
    full_context_requests: int = 0
    delta_requests: int = 0
    last_input_items: int = 0
    last_delta_input_items: int | None = None
    last_previous_response_id: str | None = None
    websocket_failures: int = 0
    sse_fallbacks: int = 0
    websocket_fallback_active: bool | None = None
    last_websocket_error: str | None = None


@dataclass(slots=True)
class _CachedWebSocketContinuationState:
    last_request_body: dict[str, Any]
    last_response_id: str
    last_response_items: list[Any]


@dataclass(slots=True)
class _CachedWebSocketConnection:
    socket: Any
    busy: bool
    created_at: float
    loop: asyncio.AbstractEventLoop
    idle_timer: asyncio.TimerHandle | None = None
    continuation: _CachedWebSocketContinuationState | None = None


_websocket_session_cache: dict[str, dict[str, _CachedWebSocketConnection]] = {}
_websocket_debug_stats: dict[str, OpenAICodexWebSocketDebugStats] = {}
_websocket_sse_fallback_sessions: set[str] = set()
_background_tasks: set[asyncio.Future[Any]] = set()


def _get_or_create_websocket_debug_stats(session_id: str) -> OpenAICodexWebSocketDebugStats:
    stats = _websocket_debug_stats.get(session_id)
    if stats is None:
        stats = OpenAICodexWebSocketDebugStats()
        _websocket_debug_stats[session_id] = stats
    return stats


def get_openai_codex_websocket_debug_stats(session_id: str) -> OpenAICodexWebSocketDebugStats | None:
    """A copy of the WebSocket transport counters for ``session_id`` (``None`` when unused)."""
    stats = _websocket_debug_stats.get(session_id)
    return replace(stats) if stats is not None else None


def reset_openai_codex_websocket_debug_stats(session_id: str | None = None) -> None:
    """Forget debug counters and the SSE fallback flag for one session (or all sessions)."""
    if session_id:
        _websocket_debug_stats.pop(session_id, None)
        _websocket_sse_fallback_sessions.discard(session_id)
        return
    _websocket_debug_stats.clear()
    _websocket_sse_fallback_sessions.clear()


def close_openai_codex_websocket_sessions(session_id: str | None = None) -> None:
    """Close pooled WebSocket connections of one session (or all sessions)."""

    def close_entry(entry: _CachedWebSocketConnection) -> None:
        if entry.idle_timer is not None:
            entry.idle_timer.cancel()
            entry.idle_timer = None
        _close_websocket_silently(entry.socket, 1000, "debug_close", entry.loop)

    if session_id:
        for entry in list((_websocket_session_cache.get(session_id) or {}).values()):
            close_entry(entry)
        _websocket_session_cache.pop(session_id, None)
        return
    for account_entries in list(_websocket_session_cache.values()):
        for entry in list(account_entries.values()):
            close_entry(entry)
    _websocket_session_cache.clear()


register_session_resource_cleanup(close_openai_codex_websocket_sessions)


def _is_websocket_sse_fallback_active(session_id: str | None) -> bool:
    return session_id in _websocket_sse_fallback_sessions if session_id else False


def _record_websocket_sse_fallback(session_id: str | None) -> None:
    if not session_id:
        return
    stats = _get_or_create_websocket_debug_stats(session_id)
    stats.sse_fallbacks += 1
    stats.websocket_fallback_active = _is_websocket_sse_fallback_active(session_id)


def _record_websocket_failure(session_id: str | None, error: BaseException) -> None:
    if not session_id:
        return
    _websocket_sse_fallback_sessions.add(session_id)
    stats = _get_or_create_websocket_debug_stats(session_id)
    stats.websocket_failures += 1
    stats.last_websocket_error = format_thrown_value(error)
    stats.websocket_fallback_active = True


_websockets_available: bool | None = None


def _resolve_websocket_connect() -> WebSocketConnect | None:
    global _websockets_available
    if websocket_connect is not None:
        return websocket_connect
    if _websockets_available is None:
        _websockets_available = importlib.util.find_spec("websockets") is not None
    return _connect_with_websockets if _websockets_available else None


def _websocket_proxy_for(url: str, env: Mapping[str, str] | None) -> str | None:
    """Proxy from provider-scoped env overrides (``websockets`` honors ``os.environ`` itself)."""
    if not env:
        return None
    secure = url.startswith("wss:")
    names = ("HTTPS_PROXY", "https_proxy") if secure else ("HTTP_PROXY", "http_proxy")
    for name in (*names, "ALL_PROXY", "all_proxy"):
        value = env.get(name)
        if value:
            no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or get_provider_env_value("NO_PROXY")
            if no_proxy:
                host = httpx.URL(url.replace("wss:", "https:", 1).replace("ws:", "http:", 1)).host
                for entry in (part.strip().lstrip(".") for part in no_proxy.split(",")):
                    if entry and (entry == "*" or host == entry or host.endswith("." + entry)):
                        return None
            return value
    return None


class _WebsocketsConnection:
    """Adapts a ``websockets`` client connection to :class:`CodexWebSocket`."""

    __slots__ = ("_connection",)

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @property
    def ready_state(self) -> int:
        return int(self._connection.state)

    async def send(self, data: str) -> None:
        await self._connection.send(data)

    async def recv(self) -> str | bytes:
        from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

        try:
            return await self._connection.recv()
        except ConnectionClosed as error:
            frame = error.rcvd
            raise CodexWebSocketClosed(
                frame.code if frame is not None else 1006,
                frame.reason if frame is not None else "",
                isinstance(error, ConnectionClosedOK),
            ) from error

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._connection.close(code, reason)


async def _connect_with_websockets(url: str, headers: dict[str, str], env: Mapping[str, str] | None) -> Any:
    try:
        from websockets.asyncio.client import connect
    except ImportError as error:
        raise RuntimeError(WEBSOCKET_UNAVAILABLE_MESSAGE) from error
    kwargs: dict[str, Any] = {"additional_headers": headers, "open_timeout": None, "max_size": None}
    if any(name.lower() == "user-agent" for name in headers):
        kwargs["user_agent_header"] = None
    proxy = _websocket_proxy_for(url, env)
    if proxy:
        kwargs["proxy"] = proxy
    connection = await connect(url, **kwargs)
    return _WebsocketsConnection(connection)


def _websocket_ready_state(socket: Any) -> int | None:
    ready_state = getattr(socket, "ready_state", None)
    if isinstance(ready_state, int) and not isinstance(ready_state, bool):
        return ready_state
    return None


def _is_websocket_reusable(socket: Any) -> bool:
    ready_state = _websocket_ready_state(socket)
    # If ready_state is unavailable, assume the implementation keeps it open/reusable.
    return ready_state is None or ready_state == 1


def _is_websocket_session_expired(entry: _CachedWebSocketConnection) -> bool:
    return _clock_ms() - entry.created_at >= SESSION_WEBSOCKET_MAX_AGE_MS


async def _await_quietly(awaitable: Awaitable[Any]) -> None:
    try:
        await awaitable
    except Exception:
        pass


def _close_websocket_silently(
    socket: Any, code: int = 1000, reason: str = "done", loop: asyncio.AbstractEventLoop | None = None
) -> None:
    try:
        result = socket.close(code, reason)
    except Exception:
        return
    if not inspect.isawaitable(result):
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    target = loop or running
    if target is not None and target is running:
        task = target.create_task(_await_quietly(result))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    elif target is not None and target.is_running() and not target.is_closed():
        asyncio.run_coroutine_threadsafe(_await_quietly(result), target)
    elif inspect.iscoroutine(result):
        result.close()


def _remove_cached_entry(session_id: str, account_id: str, entry: _CachedWebSocketConnection) -> None:
    account_entries = _websocket_session_cache.get(session_id)
    if account_entries is not None and account_entries.get(account_id) is entry:
        del account_entries[account_id]
    if account_entries is not None and not account_entries:
        _websocket_session_cache.pop(session_id, None)


def _schedule_session_websocket_expiry(session_id: str, account_id: str, entry: _CachedWebSocketConnection) -> None:
    if entry.idle_timer is not None:
        entry.idle_timer.cancel()

    def expire() -> None:
        entry.idle_timer = None
        if entry.busy:
            return
        _close_websocket_silently(entry.socket, 1000, "idle_timeout", entry.loop)
        _remove_cached_entry(session_id, account_id, entry)

    entry.idle_timer = entry.loop.call_later(SESSION_WEBSOCKET_CACHE_TTL_MS / 1000, expire)


async def _connect_websocket(
    connect: WebSocketConnect,
    url: str,
    headers: Mapping[str, str],
    connect_timeout_ms: int | None,
    env: Mapping[str, str] | None,
) -> Any:
    ws_headers = dict(headers)
    timeout_ms = DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS if connect_timeout_ms is None else connect_timeout_ms
    if timeout_ms <= 0:
        return await connect(url, ws_headers, env)
    timeout = asyncio.timeout(timeout_ms / 1000)
    try:
        async with timeout:
            return await connect(url, ws_headers, env)
    except TimeoutError:
        if timeout.expired():
            raise RuntimeError(f"WebSocket connect timeout after {timeout_ms}ms") from None
        raise


@dataclass(slots=True)
class _AcquiredWebSocket:
    socket: Any
    entry: _CachedWebSocketConnection | None
    reused: bool
    release: Callable[[bool], None]


async def _acquire_websocket(
    connect: WebSocketConnect,
    url: str,
    headers: Mapping[str, str],
    session_id: str | None,
    account_id: str,
    connect_timeout_ms: int | None,
    env: Mapping[str, str] | None,
) -> _AcquiredWebSocket:
    if not session_id:
        socket = await _connect_websocket(connect, url, headers, connect_timeout_ms, env)
        return _AcquiredWebSocket(socket, None, False, lambda keep=False: _close_websocket_silently(socket))

    loop = asyncio.get_running_loop()
    account_entries = _websocket_session_cache.get(session_id)
    cached = account_entries.get(account_id) if account_entries is not None else None
    if cached is not None and cached.loop is not loop:
        # Connections belong to the event loop that opened them.
        if not cached.busy:
            if cached.idle_timer is not None:
                cached.idle_timer.cancel()
                cached.idle_timer = None
            _close_websocket_silently(cached.socket, 1000, "done", cached.loop)
            _remove_cached_entry(session_id, account_id, cached)
            cached = None
    if cached is not None:
        if cached.idle_timer is not None:
            cached.idle_timer.cancel()
            cached.idle_timer = None
        if cached.busy or cached.loop is not loop:
            socket = await _connect_websocket(connect, url, headers, connect_timeout_ms, env)
            return _AcquiredWebSocket(socket, None, False, lambda keep=False: _close_websocket_silently(socket))
        if _is_websocket_session_expired(cached):
            _close_websocket_silently(cached.socket, 1000, "connection_age_limit", cached.loop)
            _remove_cached_entry(session_id, account_id, cached)
        elif _is_websocket_reusable(cached.socket):
            cached.busy = True
            reused_entry = cached

            def release_reused(keep: bool = False) -> None:
                if not keep or not _is_websocket_reusable(reused_entry.socket):
                    _close_websocket_silently(reused_entry.socket)
                    _remove_cached_entry(session_id, account_id, reused_entry)
                    return
                reused_entry.busy = False
                _schedule_session_websocket_expiry(session_id, account_id, reused_entry)

            return _AcquiredWebSocket(cached.socket, cached, True, release_reused)
        else:
            _close_websocket_silently(cached.socket)
            _remove_cached_entry(session_id, account_id, cached)

    socket = await _connect_websocket(connect, url, headers, connect_timeout_ms, env)
    entry = _CachedWebSocketConnection(socket=socket, busy=True, created_at=_clock_ms(), loop=loop)
    _websocket_session_cache.setdefault(session_id, {})[account_id] = entry

    def release(keep: bool = False) -> None:
        if not keep or not _is_websocket_reusable(entry.socket):
            _close_websocket_silently(entry.socket)
            if entry.idle_timer is not None:
                entry.idle_timer.cancel()
                entry.idle_timer = None
            _remove_cached_entry(session_id, account_id, entry)
            return
        entry.busy = False
        _schedule_session_websocket_expiry(session_id, account_id, entry)

    return _AcquiredWebSocket(socket, entry, False, release)


def _websocket_close_error(code: Any, reason: Any, was_clean: Any) -> Exception:
    code_text = f" {code}" if isinstance(code, int) and not isinstance(code, bool) else ""
    reason_text = f" {reason}" if isinstance(reason, str) and reason else ""
    if not reason_text and code == WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE:
        reason_text = " message too big"
    return WebSocketCloseError(
        f"WebSocket closed{code_text}{reason_text}".strip(),
        code=code if isinstance(code, int) and not isinstance(code, bool) else None,
        reason=reason if isinstance(reason, str) and reason else None,
        was_clean=was_clean if isinstance(was_clean, bool) else None,
    )


def _decode_websocket_data(data: Any) -> str | None:
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data).decode("utf-8", "replace")
    return None


async def _recv_websocket(socket: Any, idle_timeout_ms: int | None) -> Any:
    try:
        if idle_timeout_ms is None or idle_timeout_ms <= 0:
            return await socket.recv()
        timeout = asyncio.timeout(idle_timeout_ms / 1000)
        try:
            async with timeout:
                return await socket.recv()
        except TimeoutError:
            if not timeout.expired():
                raise
            _close_websocket_silently(socket, 1000, "idle_timeout")
            raise RuntimeError(f"WebSocket idle timeout after {idle_timeout_ms}ms") from None
    except CodexWebSocketClosed as closed:
        raise _websocket_close_error(closed.code, closed.reason, closed.was_clean) from closed


async def _parse_websocket(socket: Any, idle_timeout_ms: int | None) -> AsyncIterator[Any]:
    while True:
        try:
            data = await _recv_websocket(socket, idle_timeout_ms)
        except Exception as error:
            if str(error):
                raise
            raise RuntimeError("WebSocket error") from error
        if data is None:
            raise RuntimeError("WebSocket stream closed before response.completed")
        text = _decode_websocket_data(data)
        if not text:
            continue
        try:
            parsed = fastjson.loads(text)
        except ValueError as cause:
            raise CodexProtocolError(
                f"Invalid Codex WebSocket JSON: {format_thrown_value(cause)}", payload=text
            ) from cause
        event_type = parsed.get("type") if isinstance(parsed, dict) else None
        yield parsed
        if event_type in _TERMINAL_EVENT_TYPES:
            return


def _request_body_without_input(body: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key not in ("input", "previous_response_id")}


def _get_cached_websocket_input_delta(
    body: Mapping[str, Any], continuation: _CachedWebSocketContinuationState
) -> list[Any] | None:
    if _request_body_without_input(body) != _request_body_without_input(continuation.last_request_body):
        return None
    current_input = body.get("input") or []
    baseline = [*(continuation.last_request_body.get("input") or []), *continuation.last_response_items]
    if len(current_input) < len(baseline):
        return None
    if current_input[: len(baseline)] != baseline:
        return None
    return current_input[len(baseline) :]


def _build_cached_websocket_request_body(entry: _CachedWebSocketConnection, body: dict[str, Any]) -> dict[str, Any]:
    continuation = entry.continuation
    if continuation is None:
        return body
    delta = _get_cached_websocket_input_delta(body, continuation)
    if delta is None or not continuation.last_response_id:
        entry.continuation = None
        return body
    return {**body, "previous_response_id": continuation.last_response_id, "input": delta}


async def _start_on_first_event(events: AsyncIterable[Any], on_start: Callable[[], None]) -> AsyncIterator[Any]:
    started = False
    try:
        async for event in events:
            if not started:
                started = True
                on_start()
            yield event
    finally:
        await _aclose(events)


async def _process_websocket_stream(
    connect: WebSocketConnect,
    url: str,
    body: dict[str, Any],
    headers: Mapping[str, str],
    output: AssistantMessage,
    out: AssistantMessageEventStream,
    model: Model,
    on_start: Callable[[], None],
    idle_timeout_ms: int | None,
    websocket_connect_timeout_ms: int | None,
    cache_session_id: str | None,
    account_id: str,
    grammar_tool_input_properties: Mapping[str, str],
    options: Mapping[str, Any],
) -> None:
    acquired = await _acquire_websocket(
        connect, url, headers, cache_session_id, account_id, websocket_connect_timeout_ms, options.get("env")
    )
    socket, entry = acquired.socket, acquired.entry
    keep_connection = True
    use_cached_context = options.get("transport") in ("websocket-cached", "auto")
    # ChatGPT Codex Responses rejects `store: true` ("Store must be set to false").
    # WebSocket continuation still works via connection-scoped previous_response_id state.
    full_body = body
    request_body = (
        _build_cached_websocket_request_body(entry, full_body)
        if use_cached_context and entry is not None
        else full_body
    )
    if cache_session_id:
        stats = _get_or_create_websocket_debug_stats(cache_session_id)
        stats.requests += 1
        if acquired.reused:
            stats.connections_reused += 1
        else:
            stats.connections_created += 1
        if use_cached_context:
            stats.cached_context_requests += 1
        if request_body.get("store") is True:
            stats.store_true_requests += 1
        stats.last_input_items = len(request_body.get("input") or [])
        if request_body.get("previous_response_id"):
            stats.delta_requests += 1
            stats.last_delta_input_items = len(request_body.get("input") or [])
            stats.last_previous_response_id = request_body["previous_response_id"]
        else:
            stats.full_context_requests += 1
            stats.last_delta_input_items = None
            stats.last_previous_response_id = None
    try:
        await maybe_await(socket.send(fastjson.dumps({"type": "response.create", **request_body})))
        events = _start_on_first_event(
            _map_codex_events(
                _parse_websocket(socket, idle_timeout_ms), output, model, options.get("on_provider_stream_event")
            ),
            on_start,
        )
        try:
            await process_responses_stream(
                events,
                output,
                out,
                model,
                _stream_options(model, options, grammar_tool_input_properties),  # type: ignore[arg-type]
            )
        finally:
            await _aclose(events)
        signal = options.get("signal")
        if signal is not None and signal.aborted:
            keep_connection = False
        elif use_cached_context and entry is not None and output.response_id:
            response_items = [
                item
                for item in convert_responses_messages(
                    model,
                    TranscriptContext([output]),
                    CODEX_TOOL_CALL_PROVIDERS,
                    {"include_system_prompt": False, "grammar_tool_input_properties": grammar_tool_input_properties},
                )
                if item.get("type") not in ("function_call_output", "custom_tool_call_output")
            ]
            entry.continuation = _CachedWebSocketContinuationState(full_body, output.response_id, response_items)
    except BaseException:
        if entry is not None:
            entry.continuation = None
        keep_connection = False
        raise
    finally:
        acquired.release(keep_connection)


# ============================================================================
# Error Handling
# ============================================================================


def _parse_error_response(status: int, status_text: str, raw: str) -> tuple[str, str | None]:
    """``(message, friendly_message)`` for a failed Codex response."""
    message = raw or status_text or "Request failed"
    friendly_message: str | None = None
    try:
        parsed = json.loads(raw)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if err:
            fields = err if isinstance(err, dict) else {}
            code = fields.get("code") or fields.get("type") or ""
            if (isinstance(code, str) and _USAGE_LIMIT_CODE_RE.search(code)) or status == 429:
                plan_type = fields.get("plan_type")
                plan = f" ({plan_type.lower()} plan)" if plan_type else ""
                resets_at = fields.get("resets_at")
                mins = max(0, math.floor((resets_at * 1000 - _clock_ms()) / 60000 + 0.5)) if resets_at else None
                when = f" Try again in ~{mins} min." if mins is not None else ""
                friendly_message = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
            message = fields.get("message") or friendly_message or message
    except Exception:
        pass
    return message, friendly_message


# ============================================================================
# Auth & Headers
# ============================================================================


def _decode_base64_json(segment: str) -> Any:
    normalized = segment.replace("-", "+").replace("_", "/")
    normalized += "=" * (-len(normalized) % 4)
    return json.loads(base64.b64decode(normalized, validate=True).decode("utf-8"))


def _extract_account_id(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token")
        payload = _decode_base64_json(parts[1])
        claims = payload.get(JWT_CLAIM_PATH) if isinstance(payload, dict) else None
        account_id = claims.get("chatgpt_account_id") if isinstance(claims, dict) else None
        if not account_id:
            raise ValueError("No account ID in token")
        return str(account_id)
    except (ValueError, TypeError, binascii.Error, UnicodeDecodeError):
        raise RuntimeError("Failed to extract accountId from token") from None


def _build_base_codex_headers(
    init_headers: Mapping[str, str] | None,
    additional_headers: Mapping[str, str | None] | None,
    account_id: str,
    token: str,
) -> dict[str, str]:
    # Defaults first so model and caller headers can override them, matching the other providers.
    # Names are lowercased like the Fetch ``Headers`` class does.
    headers: dict[str, str] = {"originator": "pi", "user-agent": get_pi_user_agent()}
    for key, value in (init_headers or {}).items():
        headers[key.lower()] = value
    for key, value in (additional_headers or {}).items():
        if value is None:
            headers.pop(key.lower(), None)
        else:
            headers[key.lower()] = value
    headers["authorization"] = f"Bearer {token}"
    headers["chatgpt-account-id"] = account_id
    return headers


def _build_sse_headers(
    init_headers: Mapping[str, str] | None,
    additional_headers: Mapping[str, str | None] | None,
    account_id: str,
    token: str,
    session_id: str | None = None,
) -> dict[str, str]:
    headers = _build_base_codex_headers(init_headers, additional_headers, account_id, token)
    headers["openai-beta"] = "responses=experimental"
    headers["accept"] = "text/event-stream"
    headers["content-type"] = "application/json"
    if session_id:
        headers["session-id"] = session_id
        headers["x-client-request-id"] = session_id
    return headers


def _build_websocket_headers(
    init_headers: Mapping[str, str] | None,
    additional_headers: Mapping[str, str | None] | None,
    account_id: str,
    token: str,
    request_id: str,
) -> dict[str, str]:
    headers = _build_base_codex_headers(init_headers, additional_headers, account_id, token)
    headers.pop("accept", None)
    headers.pop("content-type", None)
    headers["openai-beta"] = OPENAI_BETA_RESPONSES_WEBSOCKETS
    headers["x-client-request-id"] = request_id
    headers["session-id"] = request_id
    return headers
