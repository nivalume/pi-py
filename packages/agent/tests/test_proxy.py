"""Port of upstream ``test/proxy.test.ts`` (HTTP mocked with ``httpx.MockTransport``)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx

from pi_agent import Agent, AgentInitialState, stream_proxy
from pi_ai import (
    AbortController,
    AssistantMessageEvent,
    Model,
    ModelCost,
    ToolCall,
    UserMessage,
    normalize_context,
)

MODEL = Model(
    id="gpt-5.4",
    name="GPT-5.4",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=400000,
    max_tokens=128000,
)

USAGE: dict[str, Any] = {
    "input": 0,
    "output": 0,
    "cacheRead": 0,
    "cacheWrite": 0,
    "totalTokens": 0,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
}


def sse(events: list[dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def body_client(body: str | bytes, status: int = 200, requests: list[httpx.Request] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return httpx.Response(status, content=body.encode() if isinstance(body, str) else body)

    return client_for(handler)


def proxy_options(client: httpx.AsyncClient, **extra: Any) -> dict[str, Any]:
    return {"auth_token": "test-token", "proxy_url": "https://proxy.example.com", "http_client": client, **extra}


def empty_context() -> Any:
    return normalize_context({"system_prompt": "", "messages": []})


async def drain(stream: Any) -> list[AssistantMessageEvent]:
    return [event async for event in stream]


async def test_preserves_tool_call_metadata_received_only_on_toolcall_end() -> None:
    body = sse(
        [
            {"type": "start"},
            {"type": "toolcall_start", "contentIndex": 0, "id": "call_test|fc_test", "toolName": "lookup"},
            {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"value":"hello"}'},
            {
                "type": "toolcall_end",
                "contentIndex": 0,
                "toolCall": {
                    "type": "toolCall",
                    "id": "call_test|fc_test",
                    "name": "lookup",
                    "arguments": {"value": "hello"},
                    "namespace": "dynamic_tools",
                },
            },
            {"type": "done", "reason": "toolUse", "usage": USAGE},
        ]
    )

    stream = stream_proxy(MODEL, empty_context(), proxy_options(body_client(body)))
    events = await drain(stream)
    result = await stream.result()
    end_event = next(event for event in events if event.type == "toolcall_end")

    assert end_event.tool_call.namespace == "dynamic_tools"
    first = result.content[0]
    assert isinstance(first, ToolCall)
    assert first.arguments == {"value": "hello"}
    assert first.namespace == "dynamic_tools"


# Regression tests for https://github.com/earendil-works/pi/issues/8996
async def test_processes_terminal_metadata_when_the_event_is_not_newline_terminated() -> None:
    start = f"data: {json.dumps({'type': 'start'})}\n\n"
    done = f"data: {json.dumps({'type': 'done', 'reason': 'stop', 'usage': USAGE, 'providerThinkingLevel': 'high'})}"

    stream = stream_proxy(MODEL, empty_context(), proxy_options(body_client(start + done)))
    events = await drain(stream)
    result = await stream.result()

    assert [event.type for event in events] == ["start", "done"]
    assert result.stop_reason == "stop"
    assert result.provider_thinking_level == "high"


async def test_emits_an_error_instead_of_hanging_when_the_stream_ends_without_a_terminal_event() -> None:
    body = f"data: {json.dumps({'type': 'start'})}\n\n"

    stream = stream_proxy(MODEL, empty_context(), proxy_options(body_client(body)))
    events = await drain(stream)
    result = await stream.result()

    assert [event.type for event in events] == ["start", "error"]
    assert result.stop_reason == "error"
    assert "Connection closed by proxy server" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Python-specific coverage
# ---------------------------------------------------------------------------


async def test_sends_the_request_with_auth_and_camel_case_options() -> None:
    requests: list[httpx.Request] = []
    body = sse([{"type": "start"}, {"type": "done", "reason": "stop", "usage": USAGE}])
    context = normalize_context({"system_prompt": "Be brief.", "messages": [UserMessage("hi", timestamp=1)]})

    stream = stream_proxy(
        MODEL,
        context,
        proxy_options(
            body_client(body, requests=requests),
            proxy_url="https://proxy.example.com/",
            reasoning="high",
            session_id="session-1",
            max_tokens=100,
            api_key="must-not-be-forwarded",
        ),
    )
    await stream.result()

    request = requests[0]
    assert str(request.url) == "https://proxy.example.com/api/stream"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer test-token"
    assert request.headers["content-type"] == "application/json"
    payload = json.loads(request.content)
    assert payload["model"]["id"] == "gpt-5.4"
    assert payload["options"] == {"maxTokens": 100, "reasoning": "high", "sessionId": "session-1"}
    roles = [message["role"] for message in payload["context"]["messages"]]
    assert roles == ["system", "user"]


async def test_reconstructs_text_and_thinking_partials() -> None:
    body = sse(
        [
            {"type": "start"},
            {"type": "thinking_start", "contentIndex": 0},
            {"type": "thinking_delta", "contentIndex": 0, "delta": "hmm"},
            {"type": "thinking_end", "contentIndex": 0, "contentSignature": "sig"},
            {"type": "text_start", "contentIndex": 1},
            {"type": "text_delta", "contentIndex": 1, "delta": "Hel"},
            {"type": "text_delta", "contentIndex": 1, "delta": "lo"},
            {"type": "text_end", "contentIndex": 1},
            {"type": "done", "reason": "stop", "usage": {**USAGE, "input": 3, "output": 2, "totalTokens": 5}},
        ]
    )

    stream = stream_proxy(MODEL, empty_context(), proxy_options(body_client(body)))
    events = await drain(stream)
    result = await stream.result()

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert result.content[0].thinking == "hmm"
    assert result.content[0].thinking_signature == "sig"
    assert result.content[1].text == "Hello"
    assert result.usage.input == 3
    assert result.usage.total_tokens == 5


async def test_reports_proxy_http_errors_with_the_server_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid token"})

    stream = stream_proxy(MODEL, empty_context(), proxy_options(client_for(handler)))
    events = await drain(stream)
    result = await stream.result()

    assert [event.type for event in events] == ["error"]
    assert result.stop_reason == "error"
    assert result.error_message == "Proxy error: invalid token"


async def test_reports_proxy_http_errors_without_a_json_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=b"bad gateway")

    stream = stream_proxy(MODEL, empty_context(), proxy_options(client_for(handler)))
    result = await stream.result()

    assert result.stop_reason == "error"
    assert result.error_message == "Proxy error: 502 Bad Gateway"


async def test_server_error_events_become_error_results() -> None:
    body = sse(
        [
            {"type": "start"},
            {"type": "error", "reason": "error", "errorMessage": "upstream failed", "usage": USAGE},
        ]
    )

    stream = stream_proxy(MODEL, empty_context(), proxy_options(body_client(body)))
    events = await drain(stream)
    result = await stream.result()

    assert [event.type for event in events] == ["start", "error"]
    assert result.error_message == "upstream failed"


async def test_aborting_mid_stream_yields_an_aborted_result() -> None:
    controller = AbortController()
    first_line_sent = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[override]
            yield f"data: {json.dumps({'type': 'start'})}\n\n".encode()
            first_line_sent.set()
            await asyncio.sleep(10)
            yield b""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SlowStream())

    stream = stream_proxy(MODEL, empty_context(), proxy_options(client_for(handler), signal=controller.signal))
    iterator = aiter(stream)
    first = await anext(iterator)
    assert first.type == "start"
    await first_line_sent.wait()
    controller.abort()
    result = await asyncio.wait_for(stream.result(), 2)

    assert result.stop_reason == "aborted"
    assert result.error_message


async def test_works_as_an_agent_stream_function() -> None:
    body = sse(
        [
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "proxied"},
            {"type": "text_end", "contentIndex": 0},
            {"type": "done", "reason": "stop", "usage": USAGE},
        ]
    )
    client = body_client(body)

    agent = Agent(
        initial_state=AgentInitialState(model=MODEL),
        stream_fn=lambda model, context, options: stream_proxy(
            model, context, {**(options or {}), "auth_token": "t", "proxy_url": "https://proxy", "http_client": client}
        ),
    )
    await agent.prompt("hi")

    last = agent.state.messages[-1]
    assert last.role == "assistant"
    assert last.content[0].text == "proxied"
    assert last.stop_reason == "stop"
