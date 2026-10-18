"""Proxy stream function for apps that route LLM calls through a server.

The server manages auth and proxies requests to providers. It strips the
``partial`` field from delta events to reduce bandwidth; the partial message is
reconstructed client-side.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypedDict

import httpx

from pi_ai import (
    AbortSignal,
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    StreamingJsonParser,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
    abort_scope,
)
from pi_ai.serialization import content_from_dict, from_dict, to_json_value
from pi_ai.utils.http import get_http_client
from pi_ai.utils.text import append_text, append_thinking

__all__ = ["ProxyStreamOptions", "stream_proxy", "process_proxy_event"]

#: Simple stream options forwarded to the proxy server (snake_case keys are sent as pi's camelCase).
_SERIALIZABLE_OPTIONS = (
    "temperature",
    "sampling_params",
    "max_tokens",
    "reasoning",
    "cache_retention",
    "session_id",
    "headers",
    "metadata",
    "transport",
    "thinking_budgets",
    "max_retry_delay_ms",
)


class ProxyStreamOptions(TypedDict, total=False):
    #: Local abort signal for the proxy request.
    signal: AbortSignal
    #: Auth token for the proxy server.
    auth_token: str
    #: Proxy server URL, e.g. ``"https://genai.example.com"``.
    proxy_url: str
    http_client: httpx.AsyncClient
    temperature: float
    sampling_params: dict[str, Any]
    max_tokens: int
    reasoning: str
    cache_retention: str
    session_id: str
    headers: dict[str, str | None]
    metadata: dict[str, Any]
    transport: str
    thinking_budgets: dict[str, int]
    max_retry_delay_ms: float


def _request_options(options: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _SERIALIZABLE_OPTIONS:
        value = options.get(key)
        if value is None:
            continue
        camel = "".join(part if index == 0 else part.capitalize() for index, part in enumerate(key.split("_")))
        result[camel] = value
    return result


def stream_proxy(model: Model, context: TranscriptContext, options: Mapping[str, Any]) -> AssistantMessageEventStream:
    """Stream through a pi proxy server (``POST {proxy_url}/api/stream``) instead of a provider.

    Use it as the agent's ``stream_fn``::

        Agent(stream_fn=lambda model, context, options: stream_proxy(
            model, context, {**(options or {}), "auth_token": token, "proxy_url": "https://genai.example.com"}))
    """
    options = dict(options)
    stream = AssistantMessageEventStream()

    async def run() -> None:
        partial = AssistantMessage(
            [], api=model.api, provider=model.provider, model=model.id, usage=Usage(), stop_reason="pending"
        )
        parsers: dict[int, StreamingJsonParser] = {}
        signal: AbortSignal | None = options.get("signal")
        try:
            async with abort_scope(signal):
                proxy_url = str(options.get("proxy_url", "")).rstrip("/")
                client = get_http_client(proxy_url, options)
                body = json.dumps(
                    {
                        "model": to_json_value(model),
                        "context": to_json_value(context),
                        "options": _request_options(options),
                    }
                )
                request = client.build_request(
                    "POST",
                    f"{proxy_url}/api/stream",
                    headers={
                        "Authorization": f"Bearer {options.get('auth_token', '')}",
                        "Content-Type": "application/json",
                    },
                    content=body,
                )
                response = await client.send(request, stream=True)
                try:
                    if response.status_code >= 400:
                        text = (await response.aread()).decode("utf-8", "replace")
                        message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                        try:
                            data = json.loads(text)
                            if isinstance(data, dict) and data.get("error"):
                                message = f"Proxy error: {data['error']}"
                        except ValueError:
                            pass
                        raise RuntimeError(message)
                    saw_terminal = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if not data:
                            continue
                        event = process_proxy_event(json.loads(data), partial, parsers)
                        if event is not None:
                            if event.type in ("done", "error"):
                                saw_terminal = True
                            stream.push(event)
                finally:
                    await response.aclose()
            if signal is not None and signal.aborted:
                raise RuntimeError("Request aborted by user")
            if not saw_terminal:
                # A clean EOF without done/error means the server dropped the response mid-stream.
                partial.stop_reason = "error"
                partial.error_message = "Connection closed by proxy server before the response completed"
                stream.push(ErrorEvent("error", partial))
            stream.end()
        except Exception as error:
            reason = "aborted" if signal is not None and signal.aborted else "error"
            partial.stop_reason = reason
            partial.error_message = str(error) or type(error).__name__
            stream.push(ErrorEvent(reason, partial))
            stream.end()

    stream.start(run())
    return stream


def _usage(data: Any) -> Usage:
    return from_dict(Usage, data) if isinstance(data, Mapping) else Usage()


def process_proxy_event(
    event: Mapping[str, Any], partial: AssistantMessage, parsers: dict[int, StreamingJsonParser] | None = None
) -> Any:
    """Apply one proxy event to ``partial`` and return the matching assistant message event."""
    parsers = parsers if parsers is not None else {}
    kind = event.get("type")
    index = event.get("contentIndex", 0)
    content = partial.content

    def block() -> Any:
        return content[index] if 0 <= index < len(content) else None

    def place(value: Any) -> None:
        while len(content) <= index:
            content.append(TextContent(""))
        content[index] = value

    if kind == "start":
        return StartEvent(partial)
    if kind == "text_start":
        place(TextContent(""))
        return TextStartEvent(index, partial)
    if kind == "text_delta":
        current = block()
        if isinstance(current, TextContent):
            append_text(current, event["delta"])
            return TextDeltaEvent(index, event["delta"], partial)
        raise ValueError("Received text_delta for non-text content")
    if kind == "text_end":
        current = block()
        if isinstance(current, TextContent):
            current.text_signature = event.get("contentSignature")
            return TextEndEvent(index, current.text, partial)
        raise ValueError("Received text_end for non-text content")
    if kind == "thinking_start":
        place(ThinkingContent(""))
        return ThinkingStartEvent(index, partial)
    if kind == "thinking_delta":
        current = block()
        if isinstance(current, ThinkingContent):
            append_thinking(current, event["delta"])
            return ThinkingDeltaEvent(index, event["delta"], partial)
        raise ValueError("Received thinking_delta for non-thinking content")
    if kind == "thinking_end":
        current = block()
        if isinstance(current, ThinkingContent):
            current.thinking_signature = event.get("contentSignature")
            return ThinkingEndEvent(index, current.thinking, partial)
        raise ValueError("Received thinking_end for non-thinking content")
    if kind == "toolcall_start":
        place(ToolCall(event.get("id", ""), event.get("toolName", ""), {}))
        parsers[index] = StreamingJsonParser()
        return ToolCallStartEvent(index, partial)
    if kind == "toolcall_delta":
        current = block()
        if isinstance(current, ToolCall):
            parser = parsers.setdefault(index, StreamingJsonParser())
            parser.feed(event["delta"])
            current.arguments = parser.value()
            return ToolCallDeltaEvent(index, event["delta"], partial)
        raise ValueError("Received toolcall_delta for non-toolCall content")
    if kind == "toolcall_end":
        current = block()
        if isinstance(current, ToolCall):
            final = content_from_dict(event.get("toolCall") or {})
            if isinstance(final, ToolCall):
                current.id = final.id or current.id
                current.name = final.name or current.name
                current.arguments = final.arguments
                current.thought_signature = final.thought_signature
                current.namespace = final.namespace
                current.arguments_error = final.arguments_error
            parsers.pop(index, None)
            return ToolCallEndEvent(index, current, partial)
        return None
    if kind == "done":
        partial.stop_reason = event["reason"]
        partial.usage = _usage(event.get("usage"))
        if event.get("providerThinkingLevel") is not None:
            partial.provider_thinking_level = event["providerThinkingLevel"]
        return DoneEvent(event["reason"], partial)
    if kind == "error":
        partial.stop_reason = event["reason"]
        partial.error_message = event.get("errorMessage")
        partial.usage = _usage(event.get("usage"))
        if event.get("providerThinkingLevel") is not None:
            partial.provider_thinking_level = event["providerThinkingLevel"]
        return ErrorEvent(event["reason"], partial)
    return None
