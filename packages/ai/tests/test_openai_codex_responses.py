"""Offline tests for OpenAI Codex Responses (ports pi's ``openai-codex-stream.test.ts``).

WebSocket cases inject a fake connection through ``openai_codex_responses.websocket_connect``,
the analog of upstream stubbing the global ``WebSocket``.
"""

from __future__ import annotations

import asyncio
import base64
import email.utils
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from pi_ai import (
    AbortController,
    Context,
    Model,
    TextContent,
    Tool,
    ToolResultMessage,
    Type,
    UserMessage,
    normalize_context,
)
from pi_ai.api import openai_codex_responses as codex
from pi_ai.session_resources import cleanup_session_resources
from pi_ai.utils.misc import get_pi_user_agent

CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"


def mock_token(account_id: str = "acc_test") -> str:
    payload = base64.b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode()
    ).decode()
    return f"aaa.{payload}.bbb"


def codex_model(**overrides: Any) -> Model:
    fields: dict[str, Any] = {
        "id": "gpt-5.1-codex",
        "name": "GPT-5.1 Codex",
        "api": "openai-codex-responses",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api",
        "reasoning": True,
        "context_window": 400000,
        "max_tokens": 128000,
    }
    fields.update(overrides)
    return Model(**fields)


def hello_context(text: str = "Say hello", system_prompt: str | None = "You are a helpful assistant.") -> Any:
    return normalize_context(Context(system_prompt=system_prompt, messages=[UserMessage(text, timestamp=1)]))


def hello_events(status: str = "completed", end_turn: bool | None = None, **response: Any) -> list[dict[str, Any]]:
    terminal = "response.incomplete" if status == "incomplete" else "response.completed"
    terminal_response: dict[str, Any] = {
        "status": status,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
        },
        **response,
    }
    if end_turn is not None:
        terminal_response["end_turn"] = end_turn
    return [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []},
        },
        {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
        {"type": "response.output_text.delta", "delta": "Hello"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
        },
        {"type": terminal, "response": terminal_response},
    ]


def sse_text(events: list[Any]) -> str:
    return "".join(f"data: {json.dumps(event) if not isinstance(event, str) else event}\n\n" for event in events)


def sse_response(events: list[Any] | None = None) -> httpx.Response:
    return httpx.Response(
        200, content=sse_text(events or hello_events()).encode(), headers={"content-type": "text/event-stream"}
    )


def text_of(message: Any) -> str | None:
    return next((block.text for block in message.content if isinstance(block, TextContent)), None)


class SseServer:
    """Async mock transport that records requests and serves queued handlers."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.handlers: list[Callable[[httpx.Request], Any]] = []

    def add(self, handler: Callable[[httpx.Request], Any] | httpx.Response) -> None:
        self.handlers.append(handler if callable(handler) else (lambda _request, response=handler: response))

    async def handle(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if not self.handlers:
            return httpx.Response(500, text="unexpected fetch")
        result = self.handlers.pop(0)(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def last_body(self) -> dict[str, Any]:
        return json.loads(self.last.content)


@pytest.fixture
def sse() -> SseServer:
    return SseServer()


class FakeWebSocket:
    def __init__(self, server: FakeWebSocketServer, url: str, headers: dict[str, str]) -> None:
        self.server = server
        self.url = url
        self.headers = headers
        self.connection_id = len(server.connections) + 1
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.ready_state = 1
        self.closes: list[tuple[int, str]] = []

    async def send(self, data: str) -> None:
        body = json.loads(data)
        self.server.sent.append({**body, "connection_id": self.connection_id})
        reply = self.server.respond(self, body, len(self.server.sent))
        if reply is None:
            return
        if isinstance(reply, BaseException):
            self.queue.put_nowait(reply)
            return
        for event in reply:
            self.queue.put_nowait(event if isinstance(event, (str, bytes, BaseException)) else json.dumps(event))

    async def recv(self) -> str | bytes:
        item = await self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.closes.append((code, reason))
        self.ready_state = 3


class FakeWebSocketServer:
    def __init__(self, respond: Callable[[FakeWebSocket, dict[str, Any], int], Any]) -> None:
        self.respond = respond
        self.connections: list[FakeWebSocket] = []
        self.sent: list[dict[str, Any]] = []
        self.hang = False

    async def connect(self, url: str, headers: dict[str, str], env: Any) -> FakeWebSocket:
        if self.hang:
            await asyncio.Event().wait()
        socket = FakeWebSocket(self, url, headers)
        self.connections.append(socket)
        return socket


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> Any:
    async def unexpected_connect(url: str, headers: dict[str, str], env: Any) -> Any:
        raise AssertionError("unexpected websocket connect")

    monkeypatch.setattr(codex, "websocket_connect", unexpected_connect)
    delays: list[float] = []

    async def fake_sleep(delay_ms: float) -> None:
        delays.append(delay_ms)

    monkeypatch.setattr(codex, "_sleep", fake_sleep)
    yield delays
    codex.close_openai_codex_websocket_sessions()
    codex.reset_openai_codex_websocket_debug_stats()


def use_websocket(
    monkeypatch: pytest.MonkeyPatch, respond: Callable[[FakeWebSocket, dict[str, Any], int], Any]
) -> FakeWebSocketServer:
    server = FakeWebSocketServer(respond)
    monkeypatch.setattr(codex, "websocket_connect", server.connect)
    return server


async def run_sse(sse: SseServer, model: Model | None = None, context: Any = None, **options: Any) -> Any:
    return await codex.stream(
        model or codex_model(),
        context or hello_context(),
        {"api_key": mock_token(), "transport": "sse", "http_client": sse.client(), **options},
    ).result()


# ---------------------------------------------------------------------------
# SSE transport
# ---------------------------------------------------------------------------


async def test_streams_sse_and_forwards_raw_events(sse: SseServer) -> None:
    sse.add(sse_response(hello_events()))
    token = mock_token()
    model = codex_model()
    provider_events: list[Any] = []
    event_models: list[Any] = []
    stream = codex.stream(
        model,
        hello_context(),
        {
            "api_key": token,
            "transport": "sse",
            "http_client": sse.client(),
            "on_provider_stream_event": lambda event, m: (provider_events.append(event), event_models.append(m)),
        },
    )
    types = [event.type async for event in stream]
    result = await stream.result()
    assert "text_delta" in types and types[-1] == "done"
    assert text_of(result) == "Hello"
    headers = sse.last.headers
    assert str(sse.last.url) == CODEX_URL
    assert headers["authorization"] == f"Bearer {token}"
    assert headers["chatgpt-account-id"] == "acc_test"
    assert headers["openai-beta"] == "responses=experimental"
    assert headers["originator"] == "pi"
    assert headers["user-agent"] == get_pi_user_agent()
    assert headers["accept"] == "text/event-stream"
    assert "x-api-key" not in headers
    assert [event["type"] for event in provider_events] == [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert event_models == [model] * 5
    assert result.usage.input == 5 and result.usage.output == 3


async def test_request_body_shape(sse: SseServer) -> None:
    sse.add(sse_response())
    await run_sse(sse)
    body = sse.last_body()
    assert body == {
        "model": "gpt-5.1-codex",
        "store": False,
        "stream": True,
        "instructions": "You are a helpful assistant.",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Say hello"}]}],
        "text": {"verbosity": "low"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {"effort": "none"},
    }


async def test_terminal_event_without_trailing_blank_line(sse: SseServer) -> None:
    sse.add(httpx.Response(200, content=sse_text(hello_events()).rstrip().encode()))
    result = await run_sse(sse)
    assert result.stop_reason == "stop"
    assert text_of(result) == "Hello"


def open_body(events: list[Any]) -> httpx.Response:
    async def body() -> AsyncIterator[bytes]:
        yield sse_text(events).encode()
        await asyncio.Event().wait()

    return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})


async def test_completes_after_response_completed_while_body_stays_open(sse: SseServer) -> None:
    sse.add(open_body([*hello_events(end_turn=False), "[DONE]"]))
    result = await asyncio.wait_for(run_sse(sse), 2)
    assert text_of(result) == "Hello"
    assert result.stop_reason == "stop"
    assert result.end_turn is False


async def test_incomplete_maps_to_length_while_body_stays_open(sse: SseServer) -> None:
    sse.add(open_body(hello_events("incomplete")))
    result = await asyncio.wait_for(run_sse(sse), 2)
    assert text_of(result) == "Hello"
    assert result.stop_reason == "length"


async def test_headers_timeout(sse: SseServer) -> None:
    async def hang(_request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    sse.add(hang)
    result = await run_sse(sse, timeout_ms=10)
    assert len(sse.requests) == 1
    assert result.stop_reason == "error"
    assert result.error_message == "Codex SSE response headers timed out after 10ms"


async def test_abort_during_body_read(sse: SseServer) -> None:
    state = {"closed": False}

    async def body() -> AsyncIterator[bytes]:
        try:
            yield sse_text(hello_events()[:3]).replace("Hello", "one").encode()
            await asyncio.sleep(0.05)
            yield sse_text([{"type": "response.output_text.delta", "delta": "two"}]).encode()
            await asyncio.sleep(0.05)
            yield sse_text(hello_events()[3:]).encode()
        finally:
            state["closed"] = True

    sse.add(httpx.Response(200, content=body()))
    controller = AbortController()
    stream = codex.stream(
        codex_model(),
        hello_context(),
        {"api_key": mock_token(), "transport": "sse", "http_client": sse.client(), "signal": controller.signal},
    )
    events: list[str] = []
    async for event in stream:
        events.append(f"text_delta:{event.delta}" if event.type == "text_delta" else event.type)
        if event.type == "text_delta" and event.delta == "one":
            controller.abort()
    result = await stream.result()
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert "text_delta:one" in events
    assert "text_delta:two" not in events
    await asyncio.sleep(0)
    assert state["closed"] is True


async def test_session_headers_and_prompt_cache_key(sse: SseServer) -> None:
    sse.add(sse_response())
    await run_sse(sse, session_id="test-session-123")
    headers = sse.last.headers
    assert headers["session-id"] == "test-session-123"
    assert "session_id" not in headers
    assert headers["x-client-request-id"] == "test-session-123"
    assert sse.last_body()["prompt_cache_key"] == "test-session-123"


async def test_no_session_headers_without_session(sse: SseServer) -> None:
    sse.add(sse_response())
    await run_sse(sse)
    headers = sse.last.headers
    assert "session-id" not in headers and "session_id" not in headers and "x-client-request-id" not in headers


async def test_cache_retention_none_omits_affinity(sse: SseServer) -> None:
    sse.add(sse_response())
    await run_sse(sse, cache_retention="none", session_id="one-off-summary")
    assert "session-id" not in sse.last.headers
    assert "x-client-request-id" not in sse.last.headers
    assert "prompt_cache_key" not in sse.last_body()


async def test_clamps_session_ids_to_64_characters(sse: SseServer) -> None:
    sse.add(sse_response())
    captured: dict[str, Any] = {}
    await run_sse(sse, session_id="x" * 67, on_payload=lambda payload, _m: captured.update(payload))
    assert captured["prompt_cache_key"] == "x" * 64
    assert sse.last.headers["session-id"] == "x" * 64
    assert sse.last.headers["x-client-request-id"] == "x" * 64


async def test_model_and_caller_headers_override_defaults(sse: SseServer) -> None:
    sse.add(sse_response())
    token = mock_token()
    await codex.stream(
        codex_model(headers={"originator": "my-app"}),
        hello_context(),
        {
            "api_key": token,
            "transport": "sse",
            "http_client": sse.client(),
            "headers": {"user-agent": "my-app/1.0", "Authorization": "Bearer ignored", "X-Drop": None},
        },
    ).result()
    headers = sse.last.headers
    assert headers["originator"] == "my-app"
    assert headers["user-agent"] == "my-app/1.0"
    assert headers["authorization"] == f"Bearer {token}"


async def test_preserves_xhigh_from_simple_options(sse: SseServer) -> None:
    sse.add(sse_response())
    model = codex_model(id="gpt-5.5", name="GPT-5.5", thinking_level_map={"xhigh": "xhigh"})
    await codex.stream_simple(
        model,
        hello_context(),
        {"api_key": mock_token(), "reasoning": "xhigh", "transport": "sse", "http_client": sse.client()},
    ).result()
    assert sse.last_body()["reasoning"] == {"effort": "xhigh", "summary": "auto"}


async def test_forwards_required_tool_choice(sse: SseServer) -> None:
    sse.add(sse_response())
    context = normalize_context(
        Context(
            messages=[UserMessage("Do not call ping.")],
            tools=[Tool("ping", "Ping", Type.Object({"value": Type.String()}))],
        )
    )
    await run_sse(sse, codex_model(id="gpt-5.5"), context, tool_choice="required")
    assert sse.last_body()["tool_choice"] == "required"


async def test_strict_mode_is_explicit(sse: SseServer) -> None:
    sse.add(sse_response())
    context = normalize_context(
        Context(
            messages=[UserMessage("Use a tool")],
            tools=[
                Tool("optional", "Optional", Type.Object({"value": Type.String()}), constrained_sampling=False),
                Tool(
                    "strict",
                    "Strict",
                    Type.Object({"value": Type.String()}, additional_properties=False),
                    constrained_sampling={"type": "json_schema", "strict": "prefer"},
                ),
            ],
        )
    )
    captured: dict[str, Any] = {}
    await run_sse(sse, codex_model(id="gpt-5.5"), context, on_payload=lambda payload, _m: captured.update(payload))
    tools = captured["tools"]
    assert [(tool["type"], tool["name"], tool["strict"]) for tool in tools] == [
        ("function", "optional", None),
        ("function", "strict", True),
    ]
    assert json.loads(sse.last.content)["tools"][0]["strict"] is None


@pytest.mark.parametrize("model_id", ["gpt-5.3-codex", "gpt-5.4", "gpt-5.5"])
async def test_clamps_minimal_reasoning_effort(sse: SseServer, model_id: str) -> None:
    sse.add(sse_response())
    model = codex_model(id=model_id, name=model_id, thinking_level_map={"minimal": "low"})
    await run_sse(sse, model, reasoning_effort="minimal")
    assert sse.last_body()["reasoning"] == {"effort": "low", "summary": "auto"}


def test_reasoning_none_and_off_mapping() -> None:
    context = hello_context()
    body = codex._build_request_body(codex_model(), context, {"reasoning_effort": "none"}, None)
    assert body["reasoning"] == {"effort": "none", "summary": "auto"}
    body = codex._build_request_body(
        codex_model(thinking_level_map={"off": None}), context, {"reasoning_effort": "none"}, None
    )
    assert "reasoning" not in body
    body = codex._build_request_body(codex_model(thinking_level_map={"off": None}), context, {}, None)
    assert "reasoning" not in body
    body = codex._build_request_body(codex_model(reasoning=False), context, {}, None)
    assert "reasoning" not in body
    body = codex._build_request_body(codex_model(), normalize_context(Context(messages=[UserMessage("x")])), {}, None)
    assert body["instructions"] == "You are a helpful assistant."


@pytest.mark.parametrize(
    ("model_id", "service_tier", "multiplier"),
    [
        ("gpt-5.1-codex", "flex", 0.5),
        ("gpt-5.1-codex", "priority", 2),
        ("gpt-5.5", "flex", 0.5),
        ("gpt-5.5", "priority", 2.5),
    ],
)
async def test_uses_client_service_tier_when_codex_echoes_default(
    sse: SseServer, model_id: str, service_tier: str, multiplier: float
) -> None:
    events = hello_events(
        service_tier="default",
        usage={
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "total_tokens": 2_000_000,
            "input_tokens_details": {"cached_tokens": 0},
        },
    )
    sse.add(sse_response(events))
    model = codex_model(id=model_id, cost={"input": 1, "output": 2})  # type: ignore[arg-type]
    from pi_ai import ModelCost

    model.cost = ModelCost(input=1, output=2)
    result = await run_sse(sse, model, service_tier=service_tier)
    assert result.usage.cost.input == pytest.approx(1 * multiplier)
    assert result.usage.cost.output == pytest.approx(2 * multiplier)
    assert result.usage.cost.total == pytest.approx(3 * multiplier)


def rate_limited(headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        429,
        json={"error": {"code": "rate_limit_exceeded", "message": "rate limited"}},
        headers=headers or {},
    )


@pytest.mark.parametrize(
    ("name", "headers", "expected_delay"),
    [
        ("retry-after-ms", {"retry-after-ms": "1500"}, 1500),
        ("retry-after seconds", {"retry-after": "60"}, 60_000),
        ("retry-after HTTP date", None, 45_000),
    ],
)
async def test_sse_retry_after_headers(
    sse: SseServer,
    isolate: list[float],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    headers: Any,
    expected_delay: float,
) -> None:
    now_ms = 1_778_630_400_000.0  # 2026-05-13T00:00:00Z
    monkeypatch.setattr(codex, "_clock_ms", lambda: now_ms)
    if headers is None:
        headers = {"retry-after": email.utils.formatdate((now_ms + 45_000) / 1000, usegmt=True)}
    sse.add(rate_limited(headers))
    sse.add(sse_response())
    result = await run_sse(sse, max_retries=1)
    assert isolate == [pytest.approx(expected_delay)]
    assert text_of(result) == "Hello"
    assert len(sse.requests) == 2


@pytest.mark.parametrize("status", [429, 503])
async def test_fails_when_retry_delay_exceeds_limit(sse: SseServer, status: int) -> None:
    sse.add(
        httpx.Response(
            status,
            json={"error": {"code": "temporarily_unavailable", "message": "retry later"}},
            headers={"retry-after": "2"},
        )
    )
    result = await run_sse(sse, max_retries=3, max_retry_delay_ms=1000)
    assert result.stop_reason == "error"
    assert result.error_message == "Server requested 2s retry delay (max: 1s)"
    assert len(sse.requests) == 1


async def test_exponential_backoff_without_retry_headers(sse: SseServer, isolate: list[float]) -> None:
    for _ in range(3):
        sse.add(rate_limited())
    sse.add(sse_response())
    result = await run_sse(sse, max_retries=3)
    assert isolate == [1000, 2000, 4000]
    assert text_of(result) == "Hello"
    assert len(sse.requests) == 4


async def test_usage_limit_friendly_message(sse: SseServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex, "_clock_ms", lambda: 1_000_000_000.0)
    sse.add(
        httpx.Response(
            429,
            json={"error": {"code": "usage_limit_reached", "plan_type": "PLUS", "resets_at": 1_000_000 + 600}},
        )
    )
    result = await run_sse(sse)
    assert result.error_message == "You have hit your ChatGPT usage limit (plus plan). Try again in ~10 min."


async def test_non_retryable_error_message(sse: SseServer) -> None:
    sse.add(httpx.Response(400, json={"error": {"message": "Bad input"}}))
    sse.add(httpx.Response(401, text="unauthorized"))
    result = await run_sse(sse)
    assert result.error_message == "Bad input"
    result = await run_sse(sse)
    assert result.error_message == "unauthorized"


async def test_on_response_is_called_per_attempt(sse: SseServer) -> None:
    sse.add(rate_limited())
    sse.add(sse_response())
    statuses: list[int] = []
    await run_sse(sse, max_retries=1, on_response=lambda response, _m: statuses.append(response.status))
    assert statuses == [429, 200]


async def test_zstd_compresses_sse_bodies_when_available(sse: SseServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex, "_compress_request_body_zstd", lambda body: b"ZSTD:" + body)
    sse.add(sse_response())
    large_text = "compress me " * 400
    await run_sse(sse, context=hello_context(large_text))
    assert sse.last.headers["content-encoding"] == "zstd"
    assert sse.last.content.startswith(b"ZSTD:")
    assert json.loads(sse.last.content[5:])["input"][0]["content"][0]["text"] == large_text


async def test_uncompressed_body_without_zstd_codec(sse: SseServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex, "_compress_request_body_zstd", lambda body: None)
    sse.add(sse_response())
    await run_sse(sse)
    assert "content-encoding" not in sse.last.headers
    assert sse.last_body()["model"] == "gpt-5.1-codex"


def test_real_zstd_codec_round_trips() -> None:
    compressed = codex._compress_request_body_zstd(b'{"a":1}')
    if compressed is None:
        pytest.skip("no zstd codec available")
    try:
        from compression import zstd  # type: ignore[import-not-found]

        assert zstd.decompress(compressed) == b'{"a":1}'
    except ImportError:
        import zstandard  # type: ignore[import-not-found]

        assert zstandard.ZstdDecompressor().decompress(compressed) == b'{"a":1}'


async def test_codex_error_events(sse: SseServer) -> None:
    sse.add(sse_response([{"type": "error", "error": {"code": "bad_thing", "message": "Something broke"}}]))
    result = await run_sse(sse)
    assert result.error_message == "Codex error: Something broke"
    sse.add(sse_response([{"type": "response.failed", "response": {"error": {"code": "x", "message": "failed hard"}}}]))
    result = await run_sse(sse)
    assert result.error_message == "failed hard"
    sse.add(httpx.Response(200, content=b"data: {not json}\n\n"))
    result = await run_sse(sse)
    assert (result.error_message or "").startswith("Invalid Codex SSE JSON:")


async def test_auth_errors() -> None:
    result = await codex.stream(codex_model(), hello_context(), {}).result()
    assert result.error_message == "No API key for provider: openai-codex"
    result = await codex.stream(codex_model(), hello_context(), {"api_key": "not-a-jwt"}).result()
    assert result.error_message == "Failed to extract accountId from token"
    with pytest.raises(RuntimeError, match="No API key"):
        codex.stream_simple(codex_model(), hello_context(), {})


def test_url_resolution() -> None:
    assert codex._resolve_codex_url(None) == CODEX_URL
    assert codex._resolve_codex_url("https://x.test/api/") == "https://x.test/api/codex/responses"
    assert codex._resolve_codex_url("https://x.test/codex") == "https://x.test/codex/responses"
    assert codex._resolve_codex_url("https://x.test/codex/responses") == "https://x.test/codex/responses"
    assert codex._resolve_codex_websocket_url("https://x.test") == "wss://x.test/codex/responses"
    assert codex._resolve_codex_websocket_url("http://x.test") == "ws://x.test/codex/responses"


# ---------------------------------------------------------------------------
# WebSocket transport
# ---------------------------------------------------------------------------


def respond_hello(_socket: FakeWebSocket, _body: dict[str, Any], _index: int) -> list[Any]:
    events = hello_events(end_turn=False)
    events[-1] = {**events[-1], "type": "response.done"}
    return events


async def test_auto_transport_over_websocket_from_stream_simple(
    monkeypatch: pytest.MonkeyPatch, sse: SseServer
) -> None:
    server = use_websocket(monkeypatch, respond_hello)
    model = codex_model()
    provider_events: list[Any] = []
    event_models: list[Any] = []
    result = await codex.stream_simple(
        model,
        hello_context(),
        {
            "api_key": mock_token(),
            "session_id": "session-auto",
            "transport": "auto",
            "http_client": sse.client(),
            "on_provider_stream_event": lambda event, m: (provider_events.append(event), event_models.append(m)),
        },
    ).result()
    assert result.stop_reason == "stop", result.error_message
    assert result.end_turn is False
    assert len(server.sent) == 1
    assert server.sent[0]["type"] == "response.create"
    assert [event["type"] for event in provider_events] == [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_item.done",
        "response.done",
    ]
    assert event_models == [model] * 5
    headers = server.connections[0].headers
    assert server.connections[0].url == "wss://chatgpt.com/backend-api/codex/responses"
    assert headers["session-id"] == "session-auto"
    assert "session_id" not in headers
    assert headers["x-client-request-id"] == "session-auto"
    assert headers["openai-beta"] == codex.OPENAI_BETA_RESPONSES_WEBSOCKETS
    assert "accept" not in headers and "content-type" not in headers
    assert not sse.requests
    stats = codex.get_openai_codex_websocket_debug_stats("session-auto")
    assert stats is not None
    assert stats.cached_context_requests == 1 and stats.full_context_requests == 1


async def test_websocket_without_session_uses_generated_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    server = use_websocket(monkeypatch, respond_hello)
    result = await codex.stream(
        codex_model(), hello_context(), {"api_key": mock_token(), "transport": "websocket"}
    ).result()
    assert result.stop_reason == "stop"
    headers = server.connections[0].headers
    assert headers["session-id"] == headers["x-client-request-id"]
    assert len(headers["session-id"]) == 36
    assert server.connections[0].closes == [(1000, "done")]


async def test_cached_websockets_are_scoped_to_account(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = {"n": 0}

    def respond(_socket: FakeWebSocket, _body: dict[str, Any], _index: int) -> list[Any]:
        counter["n"] += 1
        return [
            {
                "type": "response.completed",
                "response": {
                    "id": f"resp_{counter['n']}",
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            }
        ]

    server = use_websocket(monkeypatch, respond)
    context = normalize_context(Context(system_prompt="", messages=[]))
    options = {"session_id": "shared-session", "transport": "websocket-cached"}
    for account in ("account-a", "account-b", "account-a"):
        result = await codex.stream(codex_model(), context, {**options, "api_key": mock_token(account)}).result()
        assert result.stop_reason == "stop", result.error_message
    assert [socket.headers["chatgpt-account-id"] for socket in server.connections] == ["account-a", "account-b"]
    assert [socket.headers["authorization"] for socket in server.connections] == [
        f"Bearer {mock_token('account-a')}",
        f"Bearer {mock_token('account-b')}",
    ]
    stats = codex.get_openai_codex_websocket_debug_stats("shared-session")
    assert stats is not None and (stats.connections_created, stats.connections_reused) == (2, 1)


async def test_one_shot_websockets_close_with_cache_retention_none(
    monkeypatch: pytest.MonkeyPatch, sse: SseServer
) -> None:
    def respond(socket: FakeWebSocket, _body: dict[str, Any], _index: int) -> list[Any]:
        return [
            {"type": "response.completed", "response": {"id": f"resp_{socket.connection_id}", "status": "completed"}}
        ]

    server = use_websocket(monkeypatch, respond)
    options = {
        "api_key": mock_token(),
        "cache_retention": "none",
        "session_id": "one-off-summary",
        "transport": "auto",
        "http_client": sse.client(),
    }
    await codex.stream(codex_model(), hello_context(), options).result()
    await codex.stream(codex_model(), hello_context(), options).result()
    assert len(server.connections) == 2
    assert sum(len(socket.closes) for socket in server.connections) == 2
    assert len(server.sent) == 2
    assert all("prompt_cache_key" not in body for body in server.sent)
    assert codex.get_openai_codex_websocket_debug_stats("one-off-summary") is None
    assert not sse.requests


async def test_falls_back_to_sse_on_connect_timeout(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    server = use_websocket(monkeypatch, respond_hello)
    server.hang = True
    sse.add(sse_response())
    result = await codex.stream(
        codex_model(),
        hello_context(),
        {
            "api_key": mock_token(),
            "session_id": "ws-connect-timeout",
            "transport": "auto",
            "timeout_ms": 300_000,
            "websocket_connect_timeout_ms": 50,
            "http_client": sse.client(),
        },
    ).result()
    assert text_of(result) == "Hello"
    assert len(sse.requests) == 1
    stats = codex.get_openai_codex_websocket_debug_stats("ws-connect-timeout")
    assert stats is not None
    assert (stats.websocket_failures, stats.sse_fallbacks, stats.websocket_fallback_active) == (1, 1, True)
    assert stats.last_websocket_error == "WebSocket connect timeout after 50ms"
    diagnostic = (result.diagnostics or [])[0]
    assert diagnostic.type == "provider_transport_failure"
    assert diagnostic.details == {
        "configuredTransport": "auto",
        "fallbackTransport": "sse",
        "eventsEmitted": False,
        "phase": "before_message_stream_start",
        "requestBytes": diagnostic.details["requestBytes"],  # type: ignore[index]
    }

    # The session now stays on SSE.
    sse.add(sse_response())
    await codex.stream(
        codex_model(),
        hello_context(),
        {"api_key": mock_token(), "session_id": "ws-connect-timeout", "transport": "auto", "http_client": sse.client()},
    ).result()
    assert len(server.connections) == 0
    assert len(sse.requests) == 2
    stats = codex.get_openai_codex_websocket_debug_stats("ws-connect-timeout")
    assert stats is not None and stats.sse_fallbacks == 2


async def test_reconnects_once_when_connection_limit_reached(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    def respond(socket: FakeWebSocket, _body: dict[str, Any], _index: int) -> list[Any]:
        if socket.connection_id == 1:
            return [{"type": "error", "error": {"code": "websocket_connection_limit_reached"}}]
        return [{"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}}]

    server = use_websocket(monkeypatch, respond)
    result = await codex.stream(
        codex_model(), normalize_context(Context(messages=[])), {"api_key": mock_token(), "http_client": sse.client()}
    ).result()
    assert result.stop_reason == "stop", result.error_message
    assert len(server.connections) == 2
    assert not sse.requests


async def test_falls_back_to_sse_when_idle_before_first_event(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    server = use_websocket(monkeypatch, lambda *_args: None)
    sse.add(sse_response())
    result = await codex.stream(
        codex_model(),
        hello_context(),
        {
            "api_key": mock_token(),
            "session_id": "ws-idle-before-start",
            "transport": "auto",
            "timeout_ms": 50,
            "http_client": sse.client(),
        },
    ).result()
    assert len(server.sent) == 1
    assert text_of(result) == "Hello"
    assert len(sse.requests) == 1
    stats = codex.get_openai_codex_websocket_debug_stats("ws-idle-before-start")
    assert stats is not None
    assert (stats.websocket_failures, stats.sse_fallbacks, stats.websocket_fallback_active) == (1, 1, True)
    assert stats.last_websocket_error == "WebSocket idle timeout after 50ms"
    assert (1000, "idle_timeout") in server.connections[0].closes


async def test_errors_when_idle_after_stream_started(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    use_websocket(monkeypatch, lambda *_args: hello_events()[:1])
    result = await codex.stream(
        codex_model(),
        hello_context(),
        {"api_key": mock_token(), "transport": "auto", "timeout_ms": 50, "http_client": sse.client()},
    ).result()
    assert result.stop_reason == "error"
    assert result.error_message == "WebSocket idle timeout after 50ms"
    assert not sse.requests
    assert (result.diagnostics or [])[0].details["phase"] == "after_message_stream_start"  # type: ignore[index]


async def test_websocket_close_before_completion(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    use_websocket(monkeypatch, lambda *_args: [*hello_events()[:1], codex.CodexWebSocketClosed(1009, "")])
    result = await codex.stream(
        codex_model(), hello_context(), {"api_key": mock_token(), "http_client": sse.client()}
    ).result()
    assert result.error_message == "WebSocket closed 1009 message too big"


async def test_opens_fresh_cached_websocket_before_age_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"ms": 1_783_036_800_000.0}
    monkeypatch.setattr(codex, "_clock_ms", lambda: now["ms"])

    def respond(socket: FakeWebSocket, _body: dict[str, Any], _index: int) -> list[Any]:
        return [
            {"type": "response.completed", "response": {"id": f"resp_{socket.connection_id}", "status": "completed"}}
        ]

    server = use_websocket(monkeypatch, respond)
    session = "aged-ws-session"
    first_context = hello_context()
    first = await codex.stream(
        codex_model(), first_context, {"api_key": mock_token(), "session_id": session, "transport": "websocket-cached"}
    ).result()
    now["ms"] += 56 * 60 * 1000
    second_context = normalize_context(
        Context(messages=[*first_context.messages, first, UserMessage("Now finish", timestamp=2)])
    )
    await codex.stream(
        codex_model(), second_context, {"api_key": mock_token(), "session_id": session, "transport": "websocket-cached"}
    ).result()
    assert len(server.connections) == 2
    assert [body["connection_id"] for body in server.sent] == [1, 2]
    stats = codex.get_openai_codex_websocket_debug_stats(session)
    assert stats is not None and (stats.connections_created, stats.connections_reused) == (2, 0)
    assert server.connections[0].closes == [(1000, "connection_age_limit")]


async def test_websocket_cached_sends_input_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(_socket: FakeWebSocket, _body: dict[str, Any], index: int) -> list[Any]:
        response_id = f"resp_{index}"
        output_events: list[Any] = []
        if index == 1:
            item = {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": ""}
            output_events = [
                {"type": "response.output_item.added", "item": item},
                {"type": "response.custom_tool_call_input.delta", "item_id": "ctc_1", "delta": "abc"},
                {"type": "response.custom_tool_call_input.done", "item_id": "ctc_1", "input": "abc"},
                {"type": "response.output_item.done", "item": {**item, "input": "abc"}},
            ]
        return [
            {"type": "response.created", "response": {"id": response_id}},
            *output_events,
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "total_tokens": 8,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            },
        ]

    server = use_websocket(monkeypatch, respond)
    model = codex_model(compat={"supportsOpenAIGrammarTools": True})
    tool = Tool(
        "sample_tool",
        "Sample tool",
        Type.Object({"payload": Type.String()}),
        constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /[a-z]+/"}},
    )
    first_context = Context(
        system_prompt="You are a helpful assistant.", messages=[UserMessage("Use the tool", timestamp=1)], tools=[tool]
    )
    options = {"api_key": mock_token(), "session_id": "session-1", "transport": "websocket-cached"}
    first = await codex.stream(model, normalize_context(first_context), options).result()
    assert first.stop_reason == "toolUse", first.error_message
    second_context = normalize_context(
        Context(
            system_prompt=first_context.system_prompt,
            tools=[tool],
            messages=[
                *first_context.messages,
                first,
                ToolResultMessage("call_1|ctc_1", "sample_tool", [TextContent("real result")], False, timestamp=2),
                UserMessage("Now finish", timestamp=3),
            ],
        )
    )
    await codex.stream(model, second_context, options).result()

    assert len(server.sent) == 2
    first_body, second_body = server.sent
    assert first_body["store"] is False
    assert "previous_response_id" not in first_body
    assert first_body["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Use the tool"}]}]
    assert second_body["store"] is False
    assert second_body["previous_response_id"] == "resp_1"
    assert second_body["input"] == [
        {"type": "custom_tool_call_output", "call_id": "call_1", "output": "real result"},
        {"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]},
    ]
    stats = codex.get_openai_codex_websocket_debug_stats("session-1")
    assert stats is not None
    assert (
        stats.requests,
        stats.connections_created,
        stats.connections_reused,
        stats.cached_context_requests,
        stats.store_true_requests,
        stats.full_context_requests,
        stats.delta_requests,
        stats.last_delta_input_items,
        stats.last_previous_response_id,
    ) == (2, 1, 1, 2, 0, 1, 1, 2, "resp_1")


@pytest.mark.parametrize("recovery", ["websocket", "sse"])
async def test_recovers_missing_cached_continuation(
    monkeypatch: pytest.MonkeyPatch, sse: SseServer, recovery: str
) -> None:
    session = f"missing-continuation-{recovery}"
    sse.add(sse_response())

    def respond(_socket: FakeWebSocket, _body: dict[str, Any], index: int) -> Any:
        if index == 2:
            return [
                {"type": "codex.rate_limits", "plan_type": "plus", "rate_limits": {"allowed": True}},
                {
                    "type": "error",
                    "status": 400,
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "Previous response with id 'resp_1' not found.",
                        "param": "previous_response_id",
                    },
                },
            ]
        if index == 3 and recovery == "sse":
            return ConnectionError("retry websocket failed")
        response_id, message_id, text = ("resp_1", "msg_1", "Hello") if index == 1 else ("resp_2", "msg_2", "Recovered")
        message = {"type": "message", "id": message_id, "role": "assistant"}
        return [
            {"type": "response.created", "response": {"id": response_id}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**message, "status": "in_progress", "content": []},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {**message, "status": "completed", "content": [{"type": "output_text", "text": text}]},
            },
            {"type": "response.completed", "response": {"id": response_id, "status": "completed"}},
        ]

    server = use_websocket(monkeypatch, respond)
    first_context = hello_context()
    options = {
        "api_key": mock_token(),
        "session_id": session,
        "transport": "websocket-cached",
        "http_client": sse.client(),
    }
    first = await codex.stream(codex_model(), first_context, options).result()
    second_context = normalize_context(
        Context(messages=[*first_context.messages, first, UserMessage("Now finish", timestamp=2)])
    )
    stream = codex.stream(codex_model(), second_context, options)
    event_types = [event.type async for event in stream]
    second = await stream.result()

    assert second.stop_reason == "stop", second.error_message
    assert text_of(second) == ("Hello" if recovery == "sse" else "Recovered")
    assert event_types.count("start") == 1
    assert "error" not in event_types
    assert len(server.connections) == 2
    assert [body["connection_id"] for body in server.sent] == [1, 1, 2]
    assert server.sent[1]["previous_response_id"] == "resp_1"
    assert server.sent[1]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]}]
    assert "previous_response_id" not in server.sent[2]
    assert len(server.sent[2]["input"]) == 3
    assert server.sent[2]["input"][-1] == {"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]}
    assert len(sse.requests) == (1 if recovery == "sse" else 0)
    stats = codex.get_openai_codex_websocket_debug_stats(session)
    assert stats is not None
    expected_failures = 1 if recovery == "sse" else 0
    assert (
        stats.requests,
        stats.connections_created,
        stats.connections_reused,
        stats.full_context_requests,
        stats.delta_requests,
        stats.websocket_failures,
        stats.sse_fallbacks,
    ) == (3, 2, 1, 2, 1, expected_failures, expected_failures)


async def test_callback_errors_do_not_fall_back(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    use_websocket(monkeypatch, respond_hello)

    def boom(_event: Any, _model: Any) -> None:
        raise ValueError("callback failed")

    result = await codex.stream(
        codex_model(),
        hello_context(),
        {"api_key": mock_token(), "session_id": "cb", "http_client": sse.client(), "on_provider_stream_event": boom},
    ).result()
    assert result.stop_reason == "error"
    assert result.error_message == "callback failed"
    assert not sse.requests


async def test_websocket_abort(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    server = use_websocket(monkeypatch, lambda *_args: hello_events()[:3])
    controller = AbortController()
    stream = codex.stream(
        codex_model(),
        hello_context(),
        {"api_key": mock_token(), "session_id": "abort", "transport": "websocket", "signal": controller.signal},
    )
    async for event in stream:
        if event.type == "text_delta":
            controller.abort()
    result = await stream.result()
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert server.connections[0].closes
    assert "abort" not in codex._websocket_session_cache


async def test_session_resource_cleanup_closes_pooled_websockets(monkeypatch: pytest.MonkeyPatch) -> None:
    server = use_websocket(monkeypatch, respond_hello)
    await codex.stream(
        codex_model(),
        hello_context(),
        {"api_key": mock_token(), "session_id": "pooled", "transport": "websocket-cached"},
    ).result()
    assert "pooled" in codex._websocket_session_cache
    assert server.connections[0].closes == []
    cleanup_session_resources("pooled")
    assert "pooled" not in codex._websocket_session_cache
    assert server.connections[0].closes == [(1000, "debug_close")]


async def test_explicit_websocket_without_package(monkeypatch: pytest.MonkeyPatch, sse: SseServer) -> None:
    monkeypatch.setattr(codex, "websocket_connect", None)
    monkeypatch.setattr(codex, "_websockets_available", False)
    result = await codex.stream(
        codex_model(), hello_context(), {"api_key": mock_token(), "transport": "websocket"}
    ).result()
    assert result.stop_reason == "error"
    assert "pip install pi-ai[websocket]" in (result.error_message or "")

    sse.add(sse_response())
    result = await codex.stream(
        codex_model(), hello_context(), {"api_key": mock_token(), "transport": "auto", "http_client": sse.client()}
    ).result()
    assert text_of(result) == "Hello"
    assert len(sse.requests) == 1
    assert result.diagnostics is None


async def test_default_connector_uses_websockets_package(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("websockets")
    from websockets.asyncio.server import serve

    monkeypatch.setattr(codex, "websocket_connect", None)  # use the real connector
    received: dict[str, Any] = {}

    async def handler(connection: Any) -> None:
        received["headers"] = {key.lower(): value for key, value in connection.request.headers.raw_items()}
        received["body"] = json.loads(await connection.recv())
        for event in respond_hello(None, {}, 1):  # type: ignore[arg-type]
            await connection.send(json.dumps(event))
        await asyncio.sleep(0.05)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        result = await codex.stream(
            codex_model(base_url=f"http://127.0.0.1:{port}"),
            hello_context(),
            {"api_key": mock_token(), "transport": "websocket", "headers": {"X-Test": "1"}},
        ).result()
    assert result.stop_reason == "stop", result.error_message
    assert text_of(result) == "Hello"
    assert received["body"]["type"] == "response.create"
    headers = received["headers"]
    assert headers["chatgpt-account-id"] == "acc_test"
    assert headers["user-agent"] == get_pi_user_agent()
    assert headers["x-test"] == "1"
    assert headers["openai-beta"] == codex.OPENAI_BETA_RESPONSES_WEBSOCKETS
