"""Offline tests for the pi-messages API (port of upstream ``test/pi-messages.test.ts``)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from pi_ai.abort import AbortController
from pi_ai.api import pi_messages
from pi_ai.api.pi_messages import PiMessagesResponseError, stream, stream_simple
from pi_ai.api.registry import API_MODULES, pi_messages_api
from pi_ai.serialization import to_json_value
from pi_ai.types import (
    Context,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

BASE_URL = "https://radius.test/v1"


def create_model(base_url: str = BASE_URL) -> Model:
    return Model(
        id="auto",
        name="Radius Auto",
        api="pi-messages",
        provider="radius",
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1, output=2, cache_read=0.1, cache_write=0.2),
        context_window=128000,
        max_tokens=16384,
    )


TIMESTAMP = 1_700_000_000_000
WIRE_USAGE = {
    "input": 10,
    "output": 5,
    "cacheRead": 0,
    "cacheWrite": 0,
    "totalTokens": 15,
    "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0, "cacheWrite": 0, "total": 0.3},
}
USAGE = Usage(10, 5, 0, 0, 15, UsageCost(0.1, 0.2, 0, 0, 0.3))


def context() -> Any:
    return normalize_context(Context(messages=[UserMessage("Hello", timestamp=TIMESTAMP)]))


class _ChunkStream(httpx.AsyncByteStream):
    """Yields chunks one event-loop turn apart (like network reads), then optionally blocks forever."""

    def __init__(self, chunks: list[bytes], *, block: bool = False) -> None:
        self.chunks = chunks
        self.block = block

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk
        if self.block:
            await asyncio.Event().wait()


def sse_response(events: list[Any]) -> httpx.Response:
    """An SSE response that delivers one frame per network read."""
    frames = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_ChunkStream(frames))


async def test_streams_text_and_tool_calls_and_resolves_the_terminal_message(mock_server):
    mock_server.add(
        sse_response(
            [
                {"type": "start"},
                {"type": "text_start", "contentIndex": 0},
                {"type": "text_delta", "contentIndex": 0, "delta": "Hel"},
                {"type": "text_delta", "contentIndex": 0, "delta": "lo"},
                {"type": "text_end", "contentIndex": 0, "content": "Hello"},
                {"type": "toolcall_start", "contentIndex": 1, "id": "call_1", "toolName": "read"},
                {"type": "toolcall_delta", "contentIndex": 1, "delta": '{"path":'},
                {"type": "toolcall_delta", "contentIndex": 1, "delta": '"a.txt"}'},
                {
                    "type": "toolcall_end",
                    "contentIndex": 1,
                    "toolCall": {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "a.txt"}},
                },
                {
                    "type": "done",
                    "reason": "toolUse",
                    "usage": WIRE_USAGE,
                    "responseId": "resp_1",
                    "providerThinkingLevel": "high",
                },
            ]
        )
    )
    ctx = context()
    events = []
    partial_stop_reasons = []
    partial_arguments = []
    s = stream(
        create_model(),
        ctx,
        {
            "api_key": "test-key",
            "session_id": "session-1",
            "tool_choice": "auto",
            "max_tokens": 100,
            "headers": {"x-custom": "1"},
            "http_client": mock_server.client(),
        },
    )
    async for event in s:
        partial = getattr(event, "partial", None)
        if partial is not None:
            partial_stop_reasons.append(partial.stop_reason)
        if event.type == "toolcall_delta":
            partial_arguments.append(dict(partial.content[1].arguments))
        events.append(event)
    message = await s.result()

    assert partial_stop_reasons[0] == "pending"
    assert message.stop_reason == "toolUse"
    assert message.usage == USAGE
    assert message.response_id == "resp_1"
    assert message.provider_thinking_level == "high"
    assert message.model == "auto"
    assert message.provider == "radius"
    assert message.content == [TextContent("Hello"), ToolCall("call_1", "read", {"path": "a.txt"})]
    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert partial_arguments == [{}, {"path": "a.txt"}]
    assert events[2].delta == "Hel"
    assert events[8].tool_call is message.content[1]
    assert events[-1].message is message

    assert len(mock_server.requests) == 1
    request = mock_server.last
    assert request.url == "https://radius.test/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-custom"] == "1"
    assert request.json() == {
        "model": "auto",
        "context": {"messages": [{"role": "user", "content": "Hello", "timestamp": TIMESTAMP}]},
        "options": {"maxTokens": 100, "sessionId": "session-1", "toolChoice": "auto"},
    }


async def test_serializes_the_context_in_pi_camel_case_json(mock_server):
    mock_server.add_sse([{"type": "done", "reason": "stop", "usage": WIRE_USAGE}])
    tool = Tool(
        "read", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    )
    ctx = normalize_context(
        Context(
            system_prompt="Be brief.",
            tools=[tool],
            messages=[
                UserMessage([TextContent("Read a.txt")], timestamp=TIMESTAMP),
                ToolResultMessage("call_1", "read", [TextContent("contents")], timestamp=TIMESTAMP),
            ],
        )
    )
    await stream(
        create_model(),
        ctx,
        {
            "api_key": "k",
            "temperature": 0.5,
            "reasoning": "high",
            "cache_retention": "short",
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "http_client": mock_server.client(),
        },
    ).result()

    body = mock_server.last.json()
    assert body["context"] == to_json_value(ctx)
    system, user, tool_result = body["context"]["messages"]
    assert system == {
        "role": "system",
        "content": "Be brief.",
        "toolsAdded": [
            {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }
        ],
        "timestamp": 0,
    }
    assert user == {"role": "user", "content": [{"type": "text", "text": "Read a.txt"}], "timestamp": TIMESTAMP}
    assert tool_result == {
        "role": "toolResult",
        "toolCallId": "call_1",
        "toolName": "read",
        "content": [{"type": "text", "text": "contents"}],
        "isError": False,
        "timestamp": TIMESTAMP,
    }
    assert body["options"] == {
        "temperature": 0.5,
        "reasoning": "high",
        "cacheRetention": "short",
        "toolChoice": {"type": "function", "function": {"name": "read"}},
    }


async def test_maps_the_legacy_cache_retention_env_opt_in(mock_server):
    mock_server.add_sse([{"type": "done", "reason": "stop", "usage": WIRE_USAGE}])
    await stream(
        create_model(),
        context(),
        {"api_key": "k", "env": {"PI_CACHE_RETENTION": "long"}, "http_client": mock_server.client()},
    ).result()
    assert mock_server.last.json()["options"] == {"cacheRetention": "long"}


async def test_forwards_parsed_wire_events_in_order_before_converting_them(mock_server):
    wire_events = [
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_delta", "contentIndex": 0, "delta": "Hello", "gatewayField": "upstream-value"},
        {"type": "text_end", "contentIndex": 0, "content": "Hello"},
        {"type": "done", "reason": "stop", "usage": WIRE_USAGE, "responseId": "resp_1"},
    ]
    mock_server.add_sse(wire_events)
    model = create_model()
    received: list[Any] = []
    event_models: list[Any] = []

    async def on_provider_stream_event(event: Any, event_model: Any) -> None:
        await asyncio.sleep(0)
        received.append(event)
        event_models.append(event_model)

    message = await stream_simple(
        model,
        context(),
        {
            "api_key": "test-key",
            "on_provider_stream_event": on_provider_stream_event,
            "http_client": mock_server.client(),
        },
    ).result()

    assert received == wire_events
    assert event_models == [model] * len(wire_events)
    assert message.stop_reason == "stop"
    assert message.response_id == "resp_1"
    assert message.content == [TextContent("Hello")]


async def test_appends_debug_and_reports_response_headers_via_on_response(mock_server):
    mock_server.add_sse(
        [{"type": "done", "reason": "stop", "usage": WIRE_USAGE}],
        headers={"x-pi-gateway-upstream-provider": "anthropic"},
    )
    observed: dict[str, Any] = {}

    def on_response(response: Any, _model: Any) -> None:
        observed["status"] = response.status
        observed["headers"] = response.headers

    message = await stream_simple(
        create_model(),
        context(),
        {"api_key": "test-key", "debug": True, "on_response": on_response, "http_client": mock_server.client()},
    ).result()

    assert message.stop_reason == "stop"
    assert mock_server.last.url == "https://radius.test/v1/messages?debug=1"
    assert observed["status"] == 200
    assert observed["headers"]["x-pi-gateway-upstream-provider"] == "anthropic"


async def test_surfaces_backend_error_responses_with_diagnostics(mock_server):
    mock_server.add_json({"error": {"message": "Token expired", "code": "unauthorized"}}, status=401)
    statuses: list[int] = []

    message = await stream(
        create_model(),
        context(),
        {
            "api_key": "stale",
            "on_response": lambda response, _model: statuses.append(response.status),
            "http_client": mock_server.client(),
        },
    ).result()

    assert message.stop_reason == "error"
    assert message.error_message == "401 Unauthorized: Token expired (unauthorized)"
    assert message.content == []
    assert statuses == [401]
    diagnostic = message.diagnostics[0]
    assert diagnostic.type == "pi_messages_response_failure"
    assert diagnostic.details["status"] == 401
    assert diagnostic.details["statusText"] == "Unauthorized"
    assert diagnostic.details["url"] == "https://radius.test/v1/messages"
    assert diagnostic.details["error"] == {"message": "Token expired", "code": "unauthorized"}
    assert "body" not in diagnostic.details
    assert diagnostic.error.name == "PiMessagesResponseError"
    assert diagnostic.error.code == "unauthorized"


async def test_keeps_non_json_error_bodies_in_diagnostics(mock_server):
    mock_server.add(httpx.Response(502, text="bad gateway"))

    message = await stream(create_model(), context(), {"api_key": "k", "http_client": mock_server.client()}).result()

    assert message.error_message == "502 Bad Gateway: bad gateway"
    assert message.diagnostics[0].details["body"] == "bad gateway"
    assert "error" not in message.diagnostics[0].details


async def test_does_not_retry_failed_requests(mock_server):
    mock_server.add(httpx.Response(503, text="busy", headers={"retry-after-ms": "0"}))
    mock_server.add_sse([{"type": "done", "reason": "stop", "usage": WIRE_USAGE}])

    message = await stream(
        create_model(), context(), {"api_key": "k", "max_retries": 3, "http_client": mock_server.client()}
    ).result()

    assert message.stop_reason == "error"
    assert len(mock_server.requests) == 1


async def test_propagates_server_sent_error_events(mock_server):
    mock_server.add_sse(
        [
            {"type": "start"},
            {"type": "error", "reason": "error", "usage": WIRE_USAGE, "errorMessage": "Upstream failed"},
        ]
    )

    message = await stream(
        create_model(), context(), {"api_key": "test-key", "http_client": mock_server.client()}
    ).result()

    assert message.stop_reason == "error"
    assert message.error_message == "Upstream failed"
    assert message.usage == USAGE


async def test_errors_when_no_api_key_is_provided(mock_server):
    message = await stream(create_model(), context(), {"http_client": mock_server.client()}).result()

    assert message.stop_reason == "error"
    assert message.error_message == 'No API key provided for provider "radius"'
    assert mock_server.requests == []


async def test_errors_when_the_stream_ends_without_a_terminal_event(mock_server):
    mock_server.add_sse(
        [
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "partial"},
        ]
    )

    message = await stream(
        create_model(), context(), {"api_key": "test-key", "http_client": mock_server.client()}
    ).result()

    assert message.stop_reason == "error"
    assert "stream ended without a terminal event" in (message.error_message or "")


async def test_converts_thinking_blocks_and_rewrite_diagnostics(mock_server):
    rewrite = {
        "policyId": "p1",
        "policyVersion": 3,
        "changed": True,
        "tokenCountChange": -10,
        "messageCountChange": -1,
        "systemPromptChanged": False,
    }
    mock_server.add(
        sse_response(
            [
                {"type": "start"},
                {"type": "thinking_start", "contentIndex": 0},
                {"type": "thinking_delta", "contentIndex": 0, "delta": "Let me "},
                {"type": "thinking_delta", "contentIndex": 0, "delta": "think"},
                {"type": "thinking_end", "contentIndex": 0, "content": "", "contentSignature": "sig", "redacted": True},
                {"type": "text_start", "contentIndex": 1},
                {"type": "text_end", "contentIndex": 1, "content": "Hi", "contentSignature": "msg_1"},
                {"type": "toolcall_start", "contentIndex": 2, "id": "call_1", "toolName": "read"},
                {
                    "type": "toolcall_end",
                    "contentIndex": 2,
                    "toolCall": {
                        "type": "toolCall",
                        "id": "call_1",
                        "name": "read",
                        "arguments": {},
                        "thoughtSignature": "ts",
                    },
                },
                {"type": "done", "reason": "stop", "usage": WIRE_USAGE, "rewrite": rewrite},
            ]
        )
    )
    deltas: list[str] = []
    s = stream(create_model(), context(), {"api_key": "k", "http_client": mock_server.client()})
    async for event in s:
        if event.type == "thinking_delta":
            deltas.append(event.partial.content[0].thinking)
    message = await s.result()

    assert deltas == ["Let me ", "Let me think"]
    assert message.content == [
        ThinkingContent("", thinking_signature="sig", redacted=True),
        TextContent("Hi", text_signature="msg_1"),
        ToolCall("call_1", "read", {}, thought_signature="ts"),
    ]
    assert message.provider_thinking_level is None
    assert message.response_id is None
    assert len(message.diagnostics) == 1
    assert message.diagnostics[0].type == "pi_messages_rewrite"
    assert message.diagnostics[0].details == rewrite


async def test_parses_crlf_frames_split_across_chunks(mock_server):
    frames = "".join(
        f"event: message\r\ndata: {json.dumps(event)}\r\n\r\n"
        for event in [
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "ok"},
            {"type": "text_end", "contentIndex": 0, "content": "ok"},
        ]
    )
    frames += ": keep-alive\r\n\r\ndata: [DONE]\r\n\r\n"
    frames += f"data: {json.dumps({'type': 'done', 'reason': 'stop', 'usage': WIRE_USAGE})}"
    raw = frames.encode()
    # Split inside CRLF pairs and inside the JSON to exercise the frame buffer.
    chunks = [raw[index : index + 7] for index in range(0, len(raw), 7)]
    mock_server.add(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_ChunkStream(chunks)))

    message = await stream(create_model(), context(), {"api_key": "k", "http_client": mock_server.client()}).result()

    assert message.stop_reason == "stop"
    assert message.content == [TextContent("ok")]


async def test_aborts_a_running_stream(mock_server):
    first = f"data: {json.dumps({'type': 'start'})}\n\n".encode()
    mock_server.add(
        httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_ChunkStream([first], block=True))
    )
    controller = AbortController()
    s = stream(
        create_model(), context(), {"api_key": "k", "signal": controller.signal, "http_client": mock_server.client()}
    )
    types = []
    async for event in s:
        types.append(event.type)
        if event.type == "start":
            controller.abort()
    message = await s.result()

    assert types == ["start", "error"]
    assert message.stop_reason == "aborted"
    assert message.diagnostics is None


async def test_payload_hook_can_replace_the_request_body(mock_server):
    mock_server.add_sse([{"type": "done", "reason": "stop", "usage": WIRE_USAGE}])
    seen: list[Any] = []

    def on_payload(payload: Any, _model: Any) -> Any:
        seen.append(payload)
        return {**payload, "extra": True}

    await stream(
        create_model(), context(), {"api_key": "k", "on_payload": on_payload, "http_client": mock_server.client()}
    ).result()

    assert seen[0]["model"] == "auto"
    assert mock_server.last.json()["extra"] is True


def test_response_error_carries_code_and_details():
    error = PiMessagesResponseError("boom", "code", {"version": 1})
    assert str(error) == "boom"
    assert error.code == "code"
    assert error.diagnostic_details == {"version": 1}


async def test_is_registered_as_a_builtin_api(mock_server):
    assert API_MODULES["pi-messages"] == "pi_ai.api.pi_messages"
    api = pi_messages_api()
    assert api.load() is pi_messages
    mock_server.add_sse([{"type": "done", "reason": "stop", "usage": WIRE_USAGE}])
    message = await api.stream(
        create_model(), context(), {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "stop"
