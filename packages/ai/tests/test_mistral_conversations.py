"""Offline tests for the Mistral adapter (ports of upstream ``test/mistral-*.test.ts``)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import platform
import sys
from typing import Any

import httpx
import pytest

from pi_ai import AbortController, Context, Model, Tool, Type, normalize_context
from pi_ai.api.mistral_conversations import stream, stream_simple
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.misc import get_pi_user_agent
from pi_ai.utils.retry import is_retryable_assistant_error


def _model(model_id: str = "mistral-large-latest") -> Model:
    model = get_builtin_model("mistral", model_id)
    assert model is not None
    return model


def _sse(events: list[Any], sep: str = "\r\n\r\n") -> bytes:
    body = sep.join(f"data: {json.dumps(event)}" for event in events)
    return f"{body}{sep}data: [DONE]{sep}".encode()


def _terminal_event(finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "mistral-response-id",
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _sse_response(events: list[Any], headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream", **(headers or {})})


def _hello() -> Any:
    return normalize_context(Context(messages=[UserMessage("hello", timestamp=1)]))


async def test_serializes_sdk_style_payloads_to_the_wire_format(mock_server):
    model = _model()
    context = normalize_context(
        Context(
            system_prompt="Be precise",
            messages=[
                UserMessage([TextContent("describe"), ImageContent("aGVsbG8=", "image/png")], timestamp=1),
            ],
            tools=[Tool("lookup", "Look something up", Type.Object({"query": Type.String()}))],
        )
    )
    mock_server.add(_sse_response([_terminal_event()], {"x-request-id": "request-1"}))
    captured: dict[str, Any] = {}
    responses: list[Any] = []

    def on_payload(payload: Any, _model: Any) -> Any:
        captured.update(payload)
        return {
            **payload,
            "topP": 0.9,
            "randomSeed": 42,
            "responseFormat": {
                "type": "json_schema",
                "jsonSchema": {
                    "name": "result",
                    "schemaDefinition": {"type": "object", "properties": {"maxTokens": {"type": "number"}}},
                },
            },
            "presencePenalty": 0.1,
            "frequencyPenalty": 0.2,
            "parallelToolCalls": True,
            "safePrompt": True,
        }

    message = await stream(
        model,
        context,
        {
            "api_key": "secret",
            "http_client": mock_server.client(),
            "headers": {"x-custom": "value"},
            "max_tokens": 123,
            "prompt_mode": "reasoning",
            "reasoning_effort": "high",
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "session_id": "session-1",
            "on_payload": on_payload,
            "on_response": lambda response, _m: responses.append(response),
        },
    ).result()

    assert message.stop_reason == "stop"
    request = mock_server.last
    assert request.url == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["x-affinity"] == "session-1"
    assert request.headers["x-custom"] == "value"
    assert request.headers["user-agent"] == get_pi_user_agent()
    assert get_pi_user_agent().startswith(f"pi ({sys.platform} {platform.release()};")
    assert captured["maxTokens"] == 123
    assert captured["promptMode"] == "reasoning"
    assert captured["promptCacheKey"] == "session-1"
    assert len(responses) == 1
    assert responses[0].status == 200
    assert responses[0].headers["content-type"] == "text/event-stream"
    assert responses[0].headers["x-request-id"] == "request-1"

    wire = request.json()
    assert wire["max_tokens"] == 123
    assert wire["prompt_mode"] == "reasoning"
    assert wire["reasoning_effort"] == "high"
    assert wire["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert wire["prompt_cache_key"] == "session-1"
    assert wire["top_p"] == 0.9
    assert wire["random_seed"] == 42
    assert wire["presence_penalty"] == 0.1
    assert wire["frequency_penalty"] == 0.2
    assert wire["parallel_tool_calls"] is True
    assert wire["safe_prompt"] is True
    assert wire["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "schema": {"type": "object", "properties": {"maxTokens": {"type": "number"}}},
        },
    }
    for key in ("maxTokens", "promptMode", "promptCacheKey"):
        assert key not in wire
    assert wire["messages"] == [
        {"role": "system", "content": "Be precise"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            ],
        },
    ]
    assert wire["tools"][0]["function"]["name"] == "lookup"
    assert wire["tools"][0]["function"]["strict"] is False


async def test_serializes_assistant_thinking_tool_calls_and_tool_results(mock_server):
    model = _model()
    context = normalize_context(
        Context(
            messages=[
                AssistantMessage(
                    [
                        ThinkingContent("reason"),
                        TextContent("answer"),
                        ToolCall("abc123456", "lookup", {"query": "pi"}),
                    ],
                    api="mistral-conversations",
                    provider="mistral",
                    model=model.id,
                    stop_reason="toolUse",
                    timestamp=1,
                ),
                ToolResultMessage(
                    "abc123456",
                    "lookup",
                    [TextContent("found"), ImageContent("aGVsbG8=", "image/png")],
                    False,
                    timestamp=2,
                ),
            ]
        )
    )
    mock_server.add(_sse_response([_terminal_event()]))
    message = await stream(model, context, {"api_key": "test", "http_client": mock_server.client()}).result()

    assert message.stop_reason == "stop"
    assert mock_server.last.json()["messages"] == [
        {
            "role": "assistant",
            "prefix": False,
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
                {"type": "text", "text": "answer"},
            ],
            "tool_calls": [
                {
                    "id": "abc123456",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"pi"}'},
                    "index": 0,
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "abc123456",
            "name": "lookup",
            "content": [
                {"type": "text", "text": "found"},
                {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            ],
        },
    ]


async def test_normalizes_foreign_tool_call_ids_with_short_hash(mock_server):
    model = _model()
    context = normalize_context(
        Context(
            messages=[
                AssistantMessage(
                    [ToolCall("call_abc123|fc_xyz", "lookup", {})],
                    api="openai-responses",
                    provider="openai",
                    model="gpt-5",
                    stop_reason="toolUse",
                ),
                ToolResultMessage("call_abc123|fc_xyz", "lookup", [], True),
            ]
        )
    )
    mock_server.add(_sse_response([_terminal_event()]))
    await stream(model, context, {"api_key": "test", "http_client": mock_server.client()}).result()

    messages = mock_server.last.json()["messages"]
    # Matches pi: shortHash("callabc123fcxyz") truncated to 9 alphanumerics.
    assert messages[0]["tool_calls"][0]["id"] == "nrregu2qr"
    assert messages[1]["tool_call_id"] == "nrregu2qr"
    assert messages[1]["content"] == [{"type": "text", "text": "[tool error] (no tool output)"}]


async def test_keeps_mid_conversation_system_messages_when_supported(mock_server):
    model = dataclasses.replace(_model(), compat={"supportsMidConvoSystemMessages": True})
    messages = [
        SystemMessage("Base"),
        UserMessage("hi"),
        SystemMessage("Later", sections={"style": "terse"}),
        UserMessage("again"),
    ]
    mock_server.add(_sse_response([_terminal_event()]))
    await stream(
        model, normalize_context(Context(messages=messages)), {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    wire = mock_server.last.json()["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "system", "user"]
    assert wire[2]["content"] == 'Later\n\nUpdated system prompt section "style":\n\nterse'

    mock_server.add(_sse_response([_terminal_event()]))
    await stream(
        _model(), normalize_context(Context(messages=messages)), {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    wire = mock_server.last.json()["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "user"]
    assert wire[0]["content"] == "Base\n\nLater\n\nterse"


async def test_parses_thinking_text_fragmented_tool_calls_and_cached_usage(mock_server):
    model = _model()

    def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
        return {
            "id": "response-1",
            "model": model.id,
            "choices": [{"index": 0, "finish_reason": finish, "delta": delta}],
            **extra,
        }

    events = [
        chunk({"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]}]}),
        chunk({"content": [{"type": "text", "text": "answer"}]}),
        chunk(
            {"tool_calls": [{"id": "abc123456", "index": 0, "function": {"name": "lookup", "arguments": '{"query":'}}]}
        ),
        chunk(
            {"tool_calls": [{"index": 0, "function": {"name": "", "arguments": '"pi"}'}}]},
            "tool_calls",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        ),
    ]
    mock_server.add(_sse_response(events))
    s = stream(model, _hello(), {"api_key": "test", "http_client": mock_server.client()})
    types = [event.type async for event in s]
    message = await s.result()

    assert types == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert message.stop_reason == "toolUse"
    assert message.raw_stop_reason == "tool_calls"
    assert message.response_id == "response-1"
    assert message.content == [
        ThinkingContent("reason"),
        TextContent("answer"),
        ToolCall("abc123456", "lookup", {"query": "pi"}),
    ]
    usage = message.usage
    assert (usage.input, usage.output, usage.cache_read, usage.cache_write, usage.total_tokens) == (7, 4, 3, 0, 14)
    assert usage.cost.total > 0


async def test_ignores_empty_content_deltas(mock_server):
    # #9674: GLM models on Mistral send empty content deltas that must not open blocks or split thinking.
    model = _model("zai-glm-5-3")

    def thinking(text: str) -> dict[str, Any]:
        return {"type": "thinking", "thinking": [{"type": "text", "text": text}]}

    def tool_call(args: str, first: bool) -> dict[str, Any]:
        return {
            "index": 0,
            **({"id": "abc123456"} if first else {}),
            "function": {"name": "read" if first else "", "arguments": args},
        }

    deltas = [
        {"content": ""},
        {"content": [thinking("first part,")]},
        {"content": ""},
        {"content": [{"type": "text", "text": ""}]},
        {"content": [thinking(" second part."), {"type": "text", "text": "Reading."}]},
        {"content": "", "tool_calls": [tool_call("", True)]},
        {"content": "", "tool_calls": [tool_call('{"path":', False)]},
        {"content": "", "tool_calls": [tool_call('"a.txt"}', False)]},
        {"content": ""},
    ]
    events = [
        {
            "id": "response-1",
            "model": model.id,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if i == len(deltas) - 1 else None, "delta": d}],
        }
        for i, d in enumerate(deltas)
    ]
    mock_server.add(_sse_response(events))
    message = await stream(model, _hello(), {"api_key": "test", "http_client": mock_server.client()}).result()
    assert message.content == [
        ThinkingContent("first part, second part."),
        TextContent("Reading."),
        ToolCall("abc123456", "read", {"path": "a.txt"}),
    ]


async def test_forwards_each_parsed_sse_payload_before_normalizing(mock_server):
    model = _model()
    events = [
        {
            "id": "response-1",
            "provider_metadata": {"request": "test"},
            "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "hello"}}],
        },
        {
            "id": "response-1",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    mock_server.add(_sse_response(events))
    received: list[Any] = []
    models: list[Any] = []

    async def on_event(event: Any, event_model: Any) -> None:
        await asyncio.sleep(0)
        received.append(event)
        models.append(event_model)

    result = await stream(
        model, _hello(), {"api_key": "test", "http_client": mock_server.client(), "on_provider_stream_event": on_event}
    ).result()
    assert received == events
    assert models == [model, model]
    assert result.stop_reason == "stop"
    assert result.content == [TextContent("hello")]


async def test_parses_sse_and_utf8_split_across_transport_chunks(mock_server):
    model = _model()
    raw = _sse(
        [
            {
                "id": "response-bytewise",
                "model": model.id,
                "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "héllo 🌍"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        ]
    )

    async def bytewise():
        for byte in raw:
            yield bytes([byte])

    mock_server.add(httpx.Response(200, content=bytewise(), headers={"content-type": "text/event-stream"}))
    message = await stream(model, _hello(), {"api_key": "test", "http_client": mock_server.client()}).result()
    assert message.stop_reason == "stop"
    assert message.content == [TextContent("héllo 🌍")]


async def test_honors_case_insensitive_header_overrides_and_affinity_suppression(mock_server):
    model = dataclasses.replace(_model(), headers={"Authorization": "Bearer model-key", "X-Affinity": "model-affinity"})
    mock_server.add(_sse_response([_terminal_event()]))
    await stream(
        model,
        _hello(),
        {
            "api_key": "request-key",
            "http_client": mock_server.client(),
            "session_id": "automatic-affinity",
            "headers": {"authorization": None, "x-affinity": None, "User-Agent": "custom-agent"},
        },
    ).result()
    headers = mock_server.last.headers
    assert "authorization" not in headers
    assert "x-affinity" not in headers
    assert headers["user-agent"] == "custom-agent"


async def test_aborts_while_waiting_for_an_sse_chunk(mock_server):
    controller = AbortController()

    async def never():
        await asyncio.Event().wait()
        yield b""

    mock_server.add(httpx.Response(200, content=never(), headers={"content-type": "text/event-stream"}))
    s = stream(
        _model(), _hello(), {"api_key": "test", "http_client": mock_server.client(), "signal": controller.signal}
    )
    result = asyncio.ensure_future(s.result())
    await asyncio.sleep(0.05)
    controller.abort()
    message = await result
    assert message.stop_reason == "aborted"


async def test_applies_the_request_timeout_while_waiting_for_headers(mock_server):
    async def hang(_request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    mock_server.add(hang)
    message = await stream(
        _model(), _hello(), {"api_key": "test", "http_client": mock_server.client(), "timeout_ms": 5}
    ).result()
    assert message.stop_reason == "error"
    assert message.error_message == "Mistral response headers timed out after 5ms"


async def test_does_not_abort_an_active_stream_longer_than_the_timeout(mock_server):
    # Regression test for #10609: an active stream must not be cut off after timeout_ms.
    thinking_event = {
        "choices": [
            {"index": 0, "delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "x"}]}]}}
        ]
    }

    async def body():
        for _ in range(5):
            yield f"data: {json.dumps(thinking_event)}\n\n".encode()
            await asyncio.sleep(0.01)
        yield f"data: {json.dumps(_terminal_event())}\n\ndata: [DONE]\n\n".encode()

    mock_server.add(httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"}))
    message = await stream(
        _model(), _hello(), {"api_key": "test", "http_client": mock_server.client(), "timeout_ms": 20}
    ).result()
    assert message.stop_reason == "stop"
    assert message.content == [ThinkingContent("xxxxx")]


async def test_preserves_http_status_and_response_body_in_errors(mock_server):
    mock_server.add(httpx.Response(403, content=b'{"message":"blocked by gateway"}'))
    responses: list[Any] = []
    message = await stream(
        _model(),
        _hello(),
        {"api_key": "test", "http_client": mock_server.client(), "on_response": lambda r, _m: responses.append(r)},
    ).result()
    assert message.stop_reason == "error"
    assert message.error_message == 'Mistral API error (403): {"message":"blocked by gateway"}'
    assert [r.status for r in responses] == [403]


async def test_missing_api_key():
    message = await stream(_model(), _hello(), {}).result()
    assert message.error_message == "No API key for provider: mistral"
    with pytest.raises(ValueError, match="No API key for provider: mistral"):
        stream_simple(_model(), _hello(), {})


# -- raw stop reasons ------------------------------------------------------------------------------


def _stop_response(finish_reason: str) -> httpx.Response:
    event = {
        "id": "mistral-response-id",
        "model": "devstral-medium-latest",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    return httpx.Response(200, content=_sse([event], "\n\n"), headers={"content-type": "text/event-stream"})


async def test_raw_stop_reason_for_successful_stops(mock_server):
    mock_server.add(_stop_response("stop"))
    message = await stream(
        _model("devstral-medium-latest"), _hello(), {"api_key": "test", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "stop"
    assert message.error_message is None


async def test_raw_stop_reason_for_provider_error_stops(mock_server):
    mock_server.add(_stop_response("error"))
    message = await stream(
        _model("devstral-medium-latest"), _hello(), {"api_key": "test", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "error"
    assert message.error_message == "Provider stopped with: error (server error)"
    assert is_retryable_assistant_error(message)  # #10487


async def test_unknown_finish_reasons_are_provider_error_stops(mock_server):
    mock_server.add(_stop_response("unmapped_error"))
    message = await stream(
        _model("devstral-medium-latest"), _hello(), {"api_key": "test", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "unmapped_error"
    assert message.error_message == "Provider stopped with: unmapped_error"
    assert not is_retryable_assistant_error(message)


# -- reasoning mode selection (stream_simple) -----------------------------------------------------

NONE_HIGH_LEVELS = {
    "off": "none",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": None,
}
GLM_5_2_LEVELS = {**NONE_HIGH_LEVELS, "max": "max"}
GLM_5_3_LEVELS = {**NONE_HIGH_LEVELS, "off": None, "low": "low", "max": "max"}


def _make_model(model_id: str, reasoning: bool, level_map: dict[str, Any] | None = None) -> Model:
    return Model(
        id=model_id,
        api="mistral-conversations",
        provider="mistral",
        base_url="http://127.0.0.1:9",
        reasoning=reasoning,
        thinking_level_map=level_map,
        context_window=128000,
        max_tokens=16384,
    )


async def _capture_payload(mock_server, model: Model, **options: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def on_payload(payload: Any, _model: Any) -> Any:
        captured.update(payload)
        return payload

    mock_server.add(httpx.Response(400, json={"message": "stop"}))
    await stream_simple(
        model,
        normalize_context(Context(messages=[UserMessage("Hello")])),
        {**options, "api_key": "fake-key", "on_payload": on_payload, "http_client": mock_server.client()},
    ).result()
    assert captured
    return captured


async def test_prompt_mode_for_reasoning_models_without_level_map(mock_server):
    payload = await _capture_payload(mock_server, _make_model("magistral-medium-latest", True), reasoning="medium")
    assert payload["promptMode"] == "reasoning"
    assert "reasoningEffort" not in payload


async def test_omits_reasoning_controls_for_magistral_when_off(mock_server):
    payload = await _capture_payload(mock_server, _make_model("magistral-medium-latest", True))
    assert "promptMode" not in payload
    assert "reasoningEffort" not in payload


@pytest.mark.parametrize("model_id", ["mistral-small-2603", "mistral-medium-latest", "zai-glm-5-2"])
async def test_reasoning_effort_models(mock_server, model_id):
    # Regression for #8700 and #9375: Medium and GLM-5.2 ignore Magistral's prompt_mode.
    level_map = GLM_5_2_LEVELS if model_id == "zai-glm-5-2" else NONE_HIGH_LEVELS
    model = _make_model(model_id, True, level_map)

    payload = await _capture_payload(mock_server, model, reasoning="high")
    assert payload["reasoningEffort"] == "high"
    assert "promptMode" not in payload

    payload = await _capture_payload(mock_server, model, reasoning="low")
    assert payload["reasoningEffort"] == "high"

    payload = await _capture_payload(mock_server, model)
    assert payload["reasoningEffort"] == "none"
    assert "promptMode" not in payload


async def test_sends_max_for_glm_5_2(mock_server):
    # Regression for #9678.
    payload = await _capture_payload(mock_server, _make_model("zai-glm-5-2", True, GLM_5_2_LEVELS), reasoning="max")
    assert payload["reasoningEffort"] == "max"


@pytest.mark.parametrize(("level", "expected"), [("low", "low"), ("high", "high"), ("max", "max"), ("medium", "high")])
async def test_glm_5_3_levels(mock_server, level, expected):
    payload = await _capture_payload(mock_server, _make_model("zai-glm-5-3", True, GLM_5_3_LEVELS), reasoning=level)
    assert payload["reasoningEffort"] == expected
    assert "promptMode" not in payload


async def test_omits_reasoning_controls_for_non_reasoning_models(mock_server):
    payload = await _capture_payload(mock_server, _make_model("mistral-medium-2505", False), reasoning="medium")
    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


async def test_session_id_is_prompt_cache_key(mock_server):
    payload = await _capture_payload(mock_server, _make_model("mistral-large-latest", False), session_id="session-123")
    assert payload["promptCacheKey"] == "session-123"
    payload = await _capture_payload(
        mock_server, _make_model("mistral-large-latest", False), session_id="session-123", cache_retention="none"
    )
    assert "promptCacheKey" not in payload


# -- tool schema ----------------------------------------------------------------------------------


async def test_strict_tool_schema_serialization(mock_server):
    model = dataclasses.replace(_model("devstral-medium-latest"), base_url="http://127.0.0.1:9")
    parameters = Type.Object({"nested": Type.Object({"value": Type.String()})})
    context = normalize_context(
        Context(
            messages=[UserMessage("Hi")],
            tools=[
                Tool(
                    "inspect_schema",
                    "Inspect the schema",
                    parameters,
                    constrained_sampling={"type": "json_schema", "strict": "require"},
                )
            ],
        )
    )
    captured: dict[str, Any] = {}
    mock_server.add(httpx.Response(400, json={"message": "nope"}))
    response = await stream(
        model,
        context,
        {"api_key": "fake-key", "http_client": mock_server.client(), "on_payload": lambda p, _m: captured.update(p)},
    ).result()
    tools = captured["tools"]
    assert len(tools) == 1
    function = tools[0]["function"]
    assert function["strict"] is True
    assert type(function["parameters"]) is dict
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["properties"]["nested"]["required"] == ["value"]
    assert response.stop_reason == "error"
    assert response.error_message == 'Mistral API error (400): {"message":"nope"}'
