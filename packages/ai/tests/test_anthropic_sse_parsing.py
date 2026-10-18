"""Anthropic Messages SSE parsing (ports test/anthropic-sse-parsing.test.ts and anthropic-cache-write-1h-cost)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
import pytest

from pi_ai import Context, Model, ModelCost, Tool, ToolCall, Type, UserMessage, normalize_context
from pi_ai.api.anthropic_messages import stream
from pi_ai.api.transform_messages import transform_messages
from pi_ai.providers.catalog import get_builtin_model


def sse(events: list[dict[str, Any] | tuple[str, str]]) -> bytes:
    """Encode events like upstream's createSseResponse (``event: <type>`` + ``data:``)."""
    chunks = []
    for item in events:
        if isinstance(item, tuple):
            name, data = item
        else:
            name, data = item["type"], json.dumps(item)
        chunks.append(f"event: {name}\ndata: {data}\n")
    return "\n".join(chunks).encode()


def sse_response(events: list[Any]) -> httpx.Response:
    return httpx.Response(200, content=sse(events), headers={"content-type": "text/event-stream"})


MINIMAL_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {
            "input_tokens": 12,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    },
    {"type": "message_stop"},
]


def model_of(provider: str, model_id: str) -> Model:
    model = get_builtin_model(provider, model_id)
    assert model is not None
    return model


def hello_context() -> Any:
    return normalize_context(Context(messages=[UserMessage("Hello", timestamp=1)]))


def response_model_events(model: str, content_block: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {"id": "msg_response_model", "model": model, "usage": {"input_tokens": 100, "output_tokens": 0}},
        },
        {"type": "content_block_start", "index": 0, "content_block": content_block},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
        {"type": "message_stop"},
    ]


async def run(mock_server: Any, model: Model, events: list[Any], context: Any = None, **options: Any) -> Any:
    mock_server.add(sse_response(events))
    s = stream(model, context or hello_context(), {"api_key": "k", "http_client": mock_server.client(), **options})
    return s, await s.result()


async def test_forwards_parsed_provider_stream_events_in_order(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-haiku-4-5")
    provider_events: list[Any] = []
    event_models: list[Any] = []

    async def on_event(event: Any, event_model: Any) -> None:
        provider_events.append(event)
        event_models.append(event_model)

    _, result = await run(mock_server, model, MINIMAL_EVENTS, on_provider_stream_event=on_event)

    assert result.stop_reason == "stop"
    assert [event["type"] for event in provider_events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert all(event_model is model for event_model in event_models)
    assert len(event_models) == 6


async def test_emits_events_and_posts_to_messages_endpoint(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-haiku-4-5")
    s, result = await run(mock_server, model, MINIMAL_EVENTS)
    events = [event.type async for event in s]
    assert events == ["start", "text_start", "text_delta", "text_end", "done"]
    assert result.response_id == "msg_test"
    assert result.response_model is None
    assert result.usage.input == 12
    assert result.usage.output == 5
    assert result.usage.total_tokens == 17
    assert result.raw_stop_reason == "end_turn"
    request = mock_server.last
    assert request.method == "POST"
    assert request.url == "https://api.anthropic.com/v1/messages?beta=true"
    assert request.headers["x-api-key"] == "k"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "application/json"
    body = request.json()
    assert body["model"] == "claude-haiku-4-5"
    assert body["stream"] is True
    assert "betas" not in body


async def test_keeps_signed_thinking_replayable_when_a_proxy_relabels_the_model(mock_server: Any) -> None:
    # Regression test for earendil-works/pi#9188.
    model = model_of("anthropic", "claude-opus-5")
    initial = hello_context()
    _, first = await run(
        mock_server,
        model,
        response_model_events(
            "kimi-for-coding", {"type": "thinking", "thinking": "reasoning", "signature": "signature"}
        ),
        initial,
    )
    assert first.model == model.id
    assert first.response_model == "kimi-for-coding"

    transformed = transform_messages([*initial.messages, first], model)
    replayed = next(message for message in transformed if message.role == "assistant")
    assert len(replayed.content) == 1
    block = replayed.content[0]
    assert block.type == "thinking"
    assert block.thinking == "reasoning"
    assert block.thinking_signature == "signature"
    assert block.redacted is None


async def test_uses_a_returned_fallback_model_for_cost_attribution(mock_server: Any) -> None:
    fallback_model = "fallback-model"
    model = dataclasses.replace(
        model_of("anthropic", "claude-opus-5"),
        compat={
            "allowedFallbackModels": [
                {
                    "provider": "anthropic",
                    "model": fallback_model,
                    "cost": {"input": 3, "output": 5, "cacheRead": 0, "cacheWrite": 0},
                }
            ]
        },
    )
    _, result = await run(mock_server, model, response_model_events(fallback_model, {"type": "text", "text": "done"}))

    assert result.model == model.id
    assert result.response_model == fallback_model
    assert result.usage.cost.input == pytest.approx(0.0003, abs=1e-10)
    assert result.usage.cost.output == pytest.approx(0.0001, abs=1e-10)
    body = mock_server.last.json()
    assert body["fallbacks"] == [{"model": fallback_model}]
    assert mock_server.last.headers["anthropic-beta"] == "server-side-fallback-2026-07-01"


async def test_fails_safely_when_anthropic_falls_back_after_output_begins(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-opus-5")
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_fallback",
                "model": "claude-opus-5",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "partial"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "fallback",
                "from": {"model": "claude-opus-5"},
                "to": {"model": "claude-opus-4-8"},
            },
        },
    ]
    _, result = await run(mock_server, model, events)
    assert result.stop_reason == "error"
    assert "unsupported mid-output model fallback" in result.error_message


async def test_ignores_a_fallback_block_before_output(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-opus-5")
    events = [
        MINIMAL_EVENTS[0],
        {"type": "content_block_start", "index": 0, "content_block": {"type": "fallback"}},
        {"type": "content_block_stop", "index": 0},
        *[{**event, "index": 1} if "index" in event else event for event in MINIMAL_EVENTS[1:]],
    ]
    _, result = await run(mock_server, model, events)
    assert result.stop_reason == "stop"
    assert [block.text for block in result.content] == ["Hello"]


async def test_forces_streaming_after_an_on_payload_replacement(mock_server: Any) -> None:
    _, result = await run(
        mock_server,
        model_of("anthropic", "claude-fable-5-1"),
        MINIMAL_EVENTS,
        on_payload=lambda payload, _model: {**payload, "stream": False},
    )
    assert result.stop_reason == "stop"
    assert mock_server.last.json()["stream"] is True


async def test_omits_the_interleaved_thinking_beta_when_thinking_is_disabled(mock_server: Any) -> None:
    betas: list[Any] = []
    await run(
        mock_server,
        model_of("openrouter", "anthropic/claude-haiku-4.5"),
        MINIMAL_EVENTS,
        thinking_enabled=False,
        on_payload=lambda payload, _model: betas.append(payload.get("betas")),
    )
    assert "interleaved-thinking-2025-05-14" not in (betas[0] or [])
    assert "interleaved-thinking-2025-05-14" not in mock_server.last.headers.get("anthropic-beta", "")


async def test_sends_managed_beta_features(mock_server: Any) -> None:
    betas: list[Any] = []
    _, result = await run(
        mock_server,
        model_of("anthropic", "claude-fable-5-1"),
        MINIMAL_EVENTS,
        on_payload=lambda payload, _model: betas.append(payload.get("betas")),
    )
    assert result.stop_reason == "stop"
    assert "mid-conversation-output-config-2026-07-01" in betas[0]
    assert "thinking-binding-controls-2026-08-01" in betas[0]
    header = mock_server.last.headers["anthropic-beta"]
    assert "mid-conversation-output-config-2026-07-01" in header
    assert "thinking-binding-controls-2026-08-01" in header
    assert "betas" not in mock_server.last.json()


async def test_uses_the_serving_model_input_transformations_from_the_final_stream_event(mock_server: Any) -> None:
    events = [dict(event) for event in MINIMAL_EVENTS]
    events[0] = {
        "type": "message_start",
        "message": {
            "id": "msg_transformations",
            "model": "claude-fable-5-1",
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "input_transformations": [
                {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch"}
            ],
        },
    }
    events[4] = {
        **events[4],
        "input_transformations": [
            {"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch"}
        ],
    }
    _, result = await run(mock_server, model_of("anthropic", "claude-fable-5-1"), events)

    assert result.diagnostics is not None
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.type == "anthropic_input_transformations"
    assert isinstance(diagnostic.timestamp, int)
    assert diagnostic.details == {
        "transformations": [
            {"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch"}
        ]
    }


async def test_repairs_malformed_sse_json_and_malformed_streamed_tool_json(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-haiku-4-5")
    context = normalize_context(
        Context(
            messages=[UserMessage("Use the edit tool.")],
            tools=[Tool("edit", "Edit a file.", Type.Object({"path": Type.String(), "text": Type.String()}))],
        )
    )
    malformed_tool_json_delta = (
        r'{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
        r'"partial_json":"{\"path\":\"A\H\",\"text\":\"col1' + "\t" + r'col2\"}"}}'
    )
    events = [
        MINIMAL_EVENTS[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_test", "name": "edit", "input": {}},
        },
        ("content_block_delta", malformed_tool_json_delta),
        {"type": "content_block_stop", "index": 0},
        {**MINIMAL_EVENTS[4], "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    s, result = await run(mock_server, model, events, context)
    kinds = [event.type async for event in s]

    assert result.stop_reason == "toolUse"
    assert result.error_message is None
    assert kinds == ["start", "toolcall_start", "toolcall_delta", "toolcall_end", "done"]
    tool_call = next(block for block in result.content if isinstance(block, ToolCall))
    assert tool_call.id == "toolu_test"
    assert tool_call.name == "edit"
    assert tool_call.arguments == {"path": "A\\H", "text": "col1\tcol2"}
    assert tool_call.arguments_error is None


async def test_streams_partial_tool_arguments(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-haiku-4-5")
    events = [
        MINIMAL_EVENTS[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"value": "a'},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": 'bc"}'}},
        {"type": "content_block_stop", "index": 0},
        {**MINIMAL_EVENTS[4], "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    mock_server.add(sse_response(events))
    s = stream(model, hello_context(), {"api_key": "k", "http_client": mock_server.client()})
    deltas = [event.delta async for event in s if event.type == "toolcall_delta"]
    result = await s.result()
    assert deltas == ['{"value": "a', 'bc"}']
    assert result.content[0].arguments == {"value": "abc"}


async def test_reports_invalid_complete_tool_arguments(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-haiku-4-5")
    events = [
        MINIMAL_EVENTS[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "[1, 2]"}},
        {"type": "content_block_stop", "index": 0},
        {**MINIMAL_EVENTS[4], "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model, events)
    assert result.content[0].arguments == {}
    assert "Received arguments:\n[1, 2]" in result.content[0].arguments_error


async def test_preserves_content_from_content_block_start_events(mock_server: Any) -> None:
    events = [
        MINIMAL_EVENTS[0],
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "Initial text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " plus delta"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": "Initial thinking", "signature": "initial signature"},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": " plus delta"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": " plus delta"}},
        {"type": "content_block_stop", "index": 1},
        MINIMAL_EVENTS[4],
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert len(result.content) == 2
    text, thinking = result.content
    assert text.type == "text"
    assert text.text == "Initial text plus delta"
    assert text.text_signature is None
    assert thinking.type == "thinking"
    assert thinking.thinking == "Initial thinking plus delta"
    assert thinking.thinking_signature == "initial signature plus delta"
    assert thinking.redacted is None


async def test_redacted_thinking_blocks(mock_server: Any) -> None:
    events = [
        MINIMAL_EVENTS[0],
        {"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": "opaque"}},
        {"type": "content_block_stop", "index": 0},
        MINIMAL_EVENTS[4],
        {"type": "message_stop"},
    ]
    s, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert [event.type async for event in s] == ["start", "thinking_start", "thinking_end", "done"]
    block = result.content[0]
    assert block.thinking == "[Reasoning redacted]"
    assert block.thinking_signature == "opaque"
    assert block.redacted is True


async def test_preserves_refusal_stop_details_from_message_delta(mock_server: Any) -> None:
    explanation = (
        "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage "
        "Policy. To learn more, provide feedback, or request an exemption based on how you use Claude, visit our "
        "help center: https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude."
    )
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XFUDYJgAACzvnptvVoYEL",
                "usage": {
                    "input_tokens": 412,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "cyber", "explanation": explanation},
            },
            "usage": {"input_tokens": 412, "output_tokens": 0},
        },
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model_of("anthropic", "claude-fable-5"), events)
    assert result.stop_reason == "error"
    assert result.raw_stop_reason == "refusal"
    assert result.error_message == explanation


async def test_refusal_without_explanation(mock_server: Any) -> None:
    events = [
        MINIMAL_EVENTS[0],
        {"type": "message_delta", "delta": {"stop_reason": "refusal"}},
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.stop_reason == "error"
    assert result.error_message == "The model refused to complete the request"


async def test_preserves_sensitive_stop_reasons_with_a_descriptive_error_message(mock_server: Any) -> None:
    events = [
        MINIMAL_EVENTS[0],
        {"type": "message_delta", "delta": {"stop_reason": "sensitive"}, "usage": {"input_tokens": 12}},
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.stop_reason == "error"
    assert result.raw_stop_reason == "sensitive"
    assert result.error_message == "Provider stopped with: sensitive"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "toolUse"),
        ("pause_turn", "stop"),
        ("stop_sequence", "stop"),
    ],
)
async def test_maps_stop_reasons(mock_server: Any, raw: str, expected: str) -> None:
    events = [MINIMAL_EVENTS[0], {"type": "message_delta", "delta": {"stop_reason": raw}}, {"type": "message_stop"}]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.stop_reason == expected
    assert result.raw_stop_reason == raw


async def test_unknown_stop_reason_is_an_error(mock_server: Any) -> None:
    events = [
        MINIMAL_EVENTS[0],
        {"type": "message_delta", "delta": {"stop_reason": "new_reason"}},
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.stop_reason == "error"
    assert result.error_message == "Unhandled stop reason: new_reason"


async def test_treats_message_delta_without_usage_as_a_no_op_for_usage_accumulation(mock_server: Any) -> None:
    events = [
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}} if event["type"] == "message_delta" else event
        for event in MINIMAL_EVENTS
    ]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert [block.text for block in result.content] == ["Hello"]
    assert result.usage.input == 12
    assert result.usage.total_tokens == 12


async def test_reports_reasoning_tokens(mock_server: Any) -> None:
    events = [
        MINIMAL_EVENTS[0],
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 30, "output_tokens_details": {"thinking_tokens": 20}},
        },
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.usage.output == 30
    assert result.usage.reasoning == 20


async def test_ignores_unknown_sse_events_after_message_stop(mock_server: Any) -> None:
    events = [*MINIMAL_EVENTS, ("done", "[DONE]"), ("proxy.stats", "not json")]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert [block.text for block in result.content] == ["Hello"]


async def test_ping_events_are_ignored(mock_server: Any) -> None:
    events = [MINIMAL_EVENTS[0], ("ping", '{"type": "ping"}'), *MINIMAL_EVENTS[1:]]
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), events)
    assert result.stop_reason == "stop"


async def test_sse_error_event_fails_the_stream(mock_server: Any) -> None:
    data = json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), [MINIMAL_EVENTS[0], ("error", data)])
    assert result.stop_reason == "error"
    assert result.error_message == data
    assert result.usage.input == 12


async def test_unparseable_known_event_reports_raw_lines(mock_server: Any) -> None:
    _, result = await run(
        mock_server, model_of("anthropic", "claude-haiku-4-5"), [MINIMAL_EVENTS[0], ("content_block_delta", "{nope")]
    )
    assert result.stop_reason == "error"
    assert result.error_message.startswith("Could not parse Anthropic SSE event content_block_delta: ")
    assert result.error_message.endswith("; data={nope; raw=event: content_block_delta\\ndata: {nope")


async def test_stream_ended_before_message_stop(mock_server: Any) -> None:
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), MINIMAL_EVENTS[:-1])
    assert result.stop_reason == "error"
    assert result.error_message == "Anthropic stream ended before message_stop"
    # Content and usage parsed so far are kept.
    assert [block.text for block in result.content] == ["Hello"]


async def test_stream_without_events_has_no_stop_reason(mock_server: Any) -> None:
    _, result = await run(mock_server, model_of("anthropic", "claude-haiku-4-5"), [])
    assert result.stop_reason == "error"
    assert result.error_message == "Anthropic stream ended without a stop reason"


async def test_http_errors_use_sdk_message_format(mock_server: Any) -> None:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}
    mock_server.add_json(body, status=400)
    s = stream(
        model_of("anthropic", "claude-haiku-4-5"),
        hello_context(),
        {"api_key": "k", "http_client": mock_server.client()},
    )
    events = [event.type async for event in s]
    result = await s.result()
    assert events == ["error"]
    assert result.stop_reason == "error"
    assert (
        result.error_message
        == '400 {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}'
    )


async def test_calls_on_response_with_status_and_headers(mock_server: Any) -> None:
    responses: list[Any] = []
    mock_server.add(
        httpx.Response(
            200, content=sse(MINIMAL_EVENTS), headers={"content-type": "text/event-stream", "request-id": "req_1"}
        )
    )
    s = stream(
        model_of("anthropic", "claude-haiku-4-5"),
        hello_context(),
        {
            "api_key": "k",
            "http_client": mock_server.client(),
            "on_response": lambda response, _model: responses.append(response),
        },
    )
    await s.result()
    assert responses[0].status == 200
    assert responses[0].headers["request-id"] == "req_1"


# ---------------------------------------------------------------------------
# 1h cache write cost (test/anthropic-cache-write-1h-cost.test.ts)
# ---------------------------------------------------------------------------


def events_with_cache_creation(cache_creation: dict[str, int] | None) -> list[dict[str, Any]]:
    start_usage: dict[str, Any] = {
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000_000,
    }
    if cache_creation:
        start_usage["cache_creation"] = cache_creation
    return [
        {"type": "message_start", "message": {"id": "msg_test", "usage": start_usage}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1_000_000,
            },
        },
        {"type": "message_stop"},
    ]


async def test_prices_the_1h_portion_at_2x_input_and_the_rest_at_the_5m_rate(mock_server: Any) -> None:
    # claude-opus-4-8: input 5, cacheWrite (5m) 6.25 per Mtok. 1h write = 2x input = 10.
    model = model_of("anthropic", "claude-opus-4-8")
    events = events_with_cache_creation({"ephemeral_5m_input_tokens": 600_000, "ephemeral_1h_input_tokens": 400_000})
    _, result = await run(mock_server, model, events)
    assert result.usage.cache_write == 1_000_000
    assert result.usage.cache_write_1h == 400_000
    # 600k * 6.25/Mtok + 400k * 10/Mtok = 3.75 + 4.0 = 7.75
    assert result.usage.cost.cache_write == pytest.approx(7.75, abs=1e-10)


async def test_prices_1h_cache_writes_reported_only_in_message_delta(mock_server: Any) -> None:
    # Regression for #9210: Vercel AI Gateway sends cache usage in message_delta, not message_start.
    model = model_of("vercel-ai-gateway", "anthropic/claude-haiku-4.5")
    events = [
        {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 0, "output_tokens": 0}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_creation_input_tokens": 6535,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6535},
            },
        },
        {"type": "message_stop"},
    ]
    _, result = await run(mock_server, model, events)
    assert result.usage.cache_write == 6535
    assert result.usage.cache_write_1h == 6535
    assert result.usage.cost.cache_write == pytest.approx((6535 * model.cost.input * 2) / 1_000_000, abs=1e-10)


async def test_falls_back_to_the_5m_rate_when_no_breakdown_is_reported(mock_server: Any) -> None:
    model = model_of("anthropic", "claude-opus-4-8")
    _, result = await run(mock_server, model, events_with_cache_creation(None))
    assert result.usage.cache_write == 1_000_000
    assert (result.usage.cache_write_1h or 0) == 0
    # 1M * 6.25/Mtok = 6.25
    assert result.usage.cost.cache_write == pytest.approx(6.25, abs=1e-10)


async def test_fallback_cost_accepts_model_cost_instances(mock_server: Any) -> None:
    model = dataclasses.replace(
        model_of("anthropic", "claude-opus-5"),
        compat={"allowedFallbackModels": [{"provider": "anthropic", "model": "fb", "cost": ModelCost(input=1)}]},
    )
    _, result = await run(mock_server, model, response_model_events("fb", {"type": "text", "text": "x"}))
    assert result.usage.cost.input == pytest.approx(100 / 1_000_000)
