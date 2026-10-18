from __future__ import annotations

import json
import math
from typing import Any

import pytest

from pi_ai import (
    AbortController,
    Context,
    FauxModelDefinition,
    ImageContent,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Type,
    UserMessage,
    create_models,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


def _setup(**options: Any):
    faux = faux_provider(**options)
    models = create_models()
    models.set_provider(faux.provider)
    return faux, models


def _hi() -> Context:
    return Context(messages=[UserMessage("hi")])


async def _events(stream: Any) -> list[Any]:
    return [event async for event in stream]


async def test_registers_provider_and_estimates_usage() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message("hello world")])
    response = await models.complete(
        faux.get_model(), Context(system_prompt="Be concise.", messages=[UserMessage("hi there")])
    )
    assert response.content == [TextContent("hello world")]
    assert response.usage.input > 0
    assert response.usage.output > 0
    assert response.usage.total_tokens == response.usage.input + response.usage.output
    assert faux.state.call_count == 1


async def test_helper_blocks_for_text_thinking_and_tool_calls() -> None:
    faux, models = _setup()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("think"), faux_tool_call("echo", {"text": "hi"}), faux_text("done")],
                stop_reason="toolUse",
            )
        ]
    )
    response = await models.complete(faux.get_model(), _hi())
    assert response.content[0] == ThinkingContent("think")
    call = response.content[1]
    assert isinstance(call, ToolCall) and call.name == "echo" and call.arguments == {"text": "hi"} and call.id
    assert response.content[2] == TextContent("done")
    assert response.stop_reason == "toolUse"


async def test_multiple_models_with_model_aware_factories() -> None:
    faux, models = _setup(
        models=[
            FauxModelDefinition(id="faux-fast", name="Faux Fast", reasoning=False),
            {"id": "faux-thinker", "name": "Faux Thinker", "reasoning": True},
        ]
    )
    factory = lambda _context, _options, _state, model: faux_assistant_message(f"{model.id}:{model.reasoning}")  # noqa: E731
    faux.set_responses([factory, factory])
    assert [model.id for model in faux.models] == ["faux-fast", "faux-thinker"]
    assert faux.get_model() is faux.models[0]
    assert faux.get_model("faux-fast").reasoning is False  # type: ignore[union-attr]
    assert faux.get_model("faux-thinker").reasoning is True  # type: ignore[union-attr]
    assert faux.get_model("missing") is None
    fast = await models.complete(faux.get_model("faux-fast"), _hi())  # type: ignore[arg-type]
    thinker = await models.complete(faux.get_model("faux-thinker"), _hi())  # type: ignore[arg-type]
    assert fast.content == [TextContent("faux-fast:False")]
    assert thinker.content == [TextContent("faux-thinker:True")]


async def test_rewrites_api_provider_and_model() -> None:
    faux, models = _setup(api="faux:test", provider="faux-provider", models=[{"id": "faux-model"}])
    faux.set_responses([faux_assistant_message("hello")])
    response = await models.complete(faux.get_model(), _hi())
    assert (response.api, response.provider, response.model) == ("faux:test", "faux-provider", "faux-model")


async def test_consumes_queue_in_order_and_errors_when_exhausted() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    first = await models.complete(faux.get_model(), _hi())
    second = await models.complete(faux.get_model(), _hi())
    exhausted = await models.complete(faux.get_model(), _hi())
    assert first.content == [TextContent("first")]
    assert second.content == [TextContent("second")]
    assert exhausted.stop_reason == "error"
    assert exhausted.error_message == "No more faux responses queued"
    assert faux.get_pending_response_count() == 0
    assert faux.state.call_count == 3


async def test_replace_and_append_responses() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message("first")])
    assert (await models.complete(faux.get_model(), _hi())).content == [TextContent("first")]
    faux.set_responses([faux_assistant_message("second")])
    assert faux.get_pending_response_count() == 1
    assert (await models.complete(faux.get_model(), _hi())).content == [TextContent("second")]
    faux.append_responses([faux_assistant_message("third"), faux_assistant_message("fourth")])
    assert faux.get_pending_response_count() == 2
    assert (await models.complete(faux.get_model(), _hi())).content == [TextContent("third")]
    assert (await models.complete(faux.get_model(), _hi())).content == [TextContent("fourth")]


async def test_async_response_factories() -> None:
    faux, models = _setup()

    async def factory(context: Any, _options: Any, state: Any, _model: Any) -> Any:
        return faux_assistant_message(f"{len(context.messages)}:{state.call_count}")

    faux.set_responses([factory])
    assert (await models.complete(faux.get_model(), _hi())).content == [TextContent("1:1")]


async def test_error_when_factory_raises() -> None:
    faux, models = _setup()

    def factory(*_args: Any) -> Any:
        raise RuntimeError("boom")

    faux.set_responses([factory])
    events = await _events(models.stream(faux.get_model(), _hi()))
    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].error.stop_reason == "error"
    assert events[0].error.error_message == "boom"


async def test_rejects_queued_response_without_terminal_stop_reason() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message("partial", stop_reason="pending")])
    events = await _events(models.stream(faux.get_model(), _hi()))
    assert not any(event.type == "done" for event in events)
    assert events[-1].type == "error"
    assert events[-1].error.error_message == "Faux response ended without a stop reason"


async def test_estimates_prompt_and_output_tokens() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message("done")])
    tool = Tool("echo", "Echo back text", Type.Object({"text": Type.String()}))
    context = Context(
        system_prompt="sys",
        messages=[
            UserMessage([TextContent("hello"), ImageContent("abcd", "image/png")], timestamp=1),
            faux_assistant_message("prior"),
            ToolResultMessage("tool-1", "echo", [TextContent("tool out")], False, timestamp=2),
        ],
        tools=[tool],
    )
    response = await models.complete(faux.get_model(), context)
    tool_json = json.dumps(
        {"name": "echo", "description": "Echo back text", "parameters": tool.parameters}, separators=(",", ":")
    )
    prompt_text = "\n\n".join(
        [
            f"system:sys\ntool+:{tool_json}",
            "user:hello\n[image:image/png:4]",
            "assistant:prior",
            "toolResult:echo\ntool out",
        ]
    )
    assert response.usage.input == math.ceil(len(prompt_text) / 4)
    assert response.usage.output == math.ceil(len("done") / 4)
    assert response.usage.cache_read == 0 and response.usage.cache_write == 0


async def test_no_cache_sharing_across_sessions_or_without_session_id() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message(text) for text in ("first", "second", "third")])
    context = Context(messages=[UserMessage("hello")])
    first = await models.complete(faux.get_model(), context, session_id="session-1", cache_retention="short")
    assert first.usage.cache_write > 0
    context.messages.extend([first, UserMessage("follow up")])
    second = await models.complete(faux.get_model(), context, session_id="session-2", cache_retention="short")
    assert second.usage.cache_read == 0 and second.usage.cache_write > 0
    third = await models.complete(faux.get_model(), context)
    assert third.usage.cache_read == 0 and third.usage.cache_write == 0


async def test_simulates_prompt_caching_per_session() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = Context(system_prompt="Be concise.", messages=[UserMessage("hello")])
    first = await models.complete(faux.get_model(), context, {"session_id": "session-1", "cache_retention": "short"})
    assert first.usage.cache_read == 0 and first.usage.cache_write > 0
    context.messages.extend([first, UserMessage("follow up")])
    second = await models.complete(faux.get_model(), context, session_id="session-1", cache_retention="short")
    assert second.usage.cache_read > 0


async def test_counts_cached_characters_up_to_first_difference() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message(text) for text in "abc"])
    options = {"session_id": "session-1", "cache_retention": "short"}

    def user(text: str) -> UserMessage:
        return UserMessage(text, timestamp=1)

    await models.complete(faux.get_model(), Context(messages=[user("hello world")]), options)
    extended = await models.complete(faux.get_model(), Context(messages=[user("hello world"), user("next")]), options)
    assert (extended.usage.input, extended.usage.cache_read, extended.usage.cache_write) == (3, 4, 3)
    edited = await models.complete(faux.get_model(), Context(messages=[user("hello wide"), user("next")]), options)
    assert (edited.usage.input, edited.usage.cache_read, edited.usage.cache_write) == (4, 3, 4)


async def test_no_caching_when_retention_is_none() -> None:
    faux, models = _setup()
    faux.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = Context(messages=[UserMessage("hello")])
    await models.complete(faux.get_model(), context, session_id="session-1", cache_retention="none")
    context.messages.extend([faux_assistant_message("first"), UserMessage("follow up")])
    second = await models.complete(faux.get_model(), context, session_id="session-1", cache_retention="none")
    assert second.usage.cache_read == 0 and second.usage.cache_write == 0


async def test_streams_thinking_text_and_partial_tool_call_deltas() -> None:
    faux, models = _setup()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_thinking("thinking text"),
                    faux_text("answer text"),
                    faux_tool_call("echo", {"text": "hi", "count": 12}, id="tool-1"),
                ],
                stop_reason="toolUse",
            )
        ]
    )
    types: list[str] = []
    deltas: list[str] = []
    async for event in models.stream(faux.get_model(), _hi()):
        types.append(event.type)
        if event.type == "toolcall_delta":
            deltas.append(event.delta)
    for expected in (
        "thinking_start",
        "thinking_delta",
        "text_start",
        "text_delta",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
    ):
        assert expected in types
    assert len(deltas) > 1
    assert json.loads("".join(deltas)) == {"text": "hi", "count": 12}


async def test_exact_event_order_for_fixed_size_chunks() -> None:
    faux, models = _setup(token_size={"min": 1, "max": 1})
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("go"), faux_text("ok"), faux_tool_call("echo", {}, id="tool-1")], stop_reason="toolUse"
            )
        ]
    )
    events = await _events(models.stream(faux.get_model(), _hi()))
    assert events[0].type == "start" and events[0].partial.stop_reason == "pending"
    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]


async def test_streams_multiple_tool_calls() -> None:
    faux, models = _setup()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("echo", {"text": "one"}, id="tool-1"),
                    faux_tool_call("echo", {"text": "two"}, id="tool-2"),
                ],
                stop_reason="toolUse",
            )
        ]
    )
    events = await _events(models.stream(faux.get_model(), _hi()))
    assert sum(event.type == "toolcall_start" for event in events) == 2
    assert sum(event.type == "toolcall_end" for event in events) == 2


@pytest.mark.parametrize(("stop_reason", "message"), [("error", "upstream failed"), ("aborted", "Request was aborted")])
async def test_explicit_error_messages_stream_as_terminal_errors(stop_reason: str, message: str) -> None:
    faux, models = _setup(token_size={"min": 2, "max": 2})
    faux.set_responses([faux_assistant_message("partial", stop_reason=stop_reason, error_message=message)])  # type: ignore[arg-type]
    events = await _events(models.stream(faux.get_model(), _hi()))
    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    terminal = events[-1]
    assert terminal.reason == stop_reason
    assert terminal.error.stop_reason == stop_reason
    assert terminal.error.error_message == message


async def test_abort_before_first_chunk() -> None:
    faux, models = _setup(tokens_per_second=50, token_size={"min": 3, "max": 3})
    faux.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])
    controller = AbortController()
    controller.abort()
    events = await _events(models.stream(faux.get_model(), _hi(), signal=controller.signal))
    assert len(events) == 1
    assert events[0].type == "error" and events[0].reason == "aborted"
    assert events[0].error.stop_reason == "aborted"


@pytest.mark.parametrize(
    ("content", "delta_type", "end_type"),
    [
        ([TextContent("abcdefghijklmnopqrstuvwxyz")], "text_delta", "text_end"),
        ([ThinkingContent("abcdefghijklmnopqrstuvwxyz")], "thinking_delta", "thinking_end"),
        (
            [ToolCall("tool-1", "echo", {"text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789})],
            "toolcall_delta",
            "toolcall_end",
        ),
    ],
)
async def test_abort_mid_stream_when_paced(content: list[Any], delta_type: str, end_type: str) -> None:
    faux, models = _setup(tokens_per_second=100, token_size={"min": 3, "max": 3})
    faux.set_responses(
        [faux_assistant_message(content, stop_reason="toolUse" if delta_type == "toolcall_delta" else "stop")]
    )
    controller = AbortController()
    types: list[str] = []
    deltas = 0
    async for event in models.stream(faux.get_model(), _hi(), signal=controller.signal):
        types.append(event.type)
        if event.type == delta_type:
            deltas += 1
            controller.abort()
    assert deltas == 1
    assert "error" in types
    assert end_type not in types


async def test_deferred_responses() -> None:
    faux, models = _setup(deferred={"pending_fetches": 1, "poll_after_ms": 5})
    faux.set_responses([faux_assistant_message("later")])
    model = faux.get_model()
    submitted = await models.complete_simple(model, _hi(), deferred=True)
    assert submitted.stop_reason == "deferred" and submitted.deferred is not None
    handle = submitted.deferred
    assert handle.poll_after_ms == 5
    pending = await models.fetch_deferred(model, handle)
    assert pending.stop_reason == "deferred"
    final = await models.fetch_deferred(model, handle)
    assert final.content == [TextContent("later")]
    await models.cancel_deferred(model, handle)
    assert faux.state.cancelled_deferred[0].id == handle.id
    assert faux.state.deferred_fetch_count == 2
