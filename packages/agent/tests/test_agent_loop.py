"""Port of upstream ``test/agent-loop.test.ts``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from agent_helpers import (
    create_assistant_message,
    create_model,
    create_user_message,
    identity_converter,
    respond,
)

from pi_agent import (
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    AgentTurnDecision,
    BeforeToolCallResult,
    RunToolCallOptions,
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_tool_call,
    set_default_stream_fn,
)
from pi_ai import (
    AssistantMessageEventStream,
    Message,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Type,
    Usage,
    UsageCost,
    UserMessage,
    now_ms,
)

ECHO_SCHEMA = Type.Object({"value": Type.String()})


def echo_tool(
    execute: Any = None,
    *,
    name: str = "echo",
    execution_mode: Any = None,
    prepare_arguments: Any = None,
    parameters: Any = None,
) -> AgentTool:
    async def default_execute(_tool_call_id: str, params: Any, _signal: Any = None, _on_update: Any = None) -> Any:
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]})

    return AgentTool(
        name,
        f"{name.capitalize()} tool",
        parameters if parameters is not None else ECHO_SCHEMA,
        label=name.capitalize(),
        execute=execute or default_execute,
        execution_mode=execution_mode,
        prepare_arguments=prepare_arguments,
    )


def noop_tool() -> AgentTool:
    async def execute(*_args: Any) -> AgentToolResult:
        return AgentToolResult([TextContent("done")], None)

    return AgentTool("noop", "Noop tool", Type.Object({}), label="Noop", execute=execute)


def tool_call(id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id, name, arguments)


async def collect(stream: Any) -> list[AgentEvent]:
    return [event async for event in stream]


def config(**overrides: Any) -> AgentLoopConfig:
    return AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, **overrides)


# ---------------------------------------------------------------------------
# default stream function compatibility
# ---------------------------------------------------------------------------


async def test_uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() -> None:
    calls = 0

    def default_stream_fn(_model: Any, _context: Any, _options: Any) -> AssistantMessageEventStream:
        nonlocal calls
        calls += 1
        return respond(create_assistant_message([TextContent("fallback")]))

    set_default_stream_fn(default_stream_fn)
    try:
        context = AgentContext(messages=[], tools=[])
        stream = agent_loop([create_user_message("Hello")], context, config())
        await stream.result()
        assert calls == 1
    finally:
        set_default_stream_fn(None)


# ---------------------------------------------------------------------------
# agent_loop with agent messages
# ---------------------------------------------------------------------------


async def test_should_emit_events_with_agent_message_types() -> None:
    context = AgentContext(messages=[], tools=[])

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(create_assistant_message([TextContent("Hi there!")]))

    stream = agent_loop([create_user_message("Hello")], context, config(), None, stream_fn)
    events = await collect(stream)
    messages = await stream.result()

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    event_types = [event.type for event in events]
    for expected in ("agent_start", "turn_start", "message_start", "message_end", "turn_end", "agent_end"):
        assert expected in event_types


async def test_should_build_provider_context_exclusively_from_transcript_messages() -> None:
    initial_system = SystemMessage("Transcript prompt", tools_added=[], timestamp=1)
    context = AgentContext(messages=[], tools=[])
    provider_contexts: list[Any] = []

    def stream_fn(_model: Any, provider_context: Any, _options: Any) -> AssistantMessageEventStream:
        provider_contexts.append(provider_context)
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop([initial_system, create_user_message("Hello")], context, config(), None, stream_fn)
    await stream.result()

    # The provider receives a transcript: no top-level prompt or tool fields.
    provider_context = provider_contexts[0]
    assert isinstance(provider_context, TranscriptContext)
    assert not hasattr(provider_context, "system_prompt")
    assert not hasattr(provider_context, "tools")
    assert provider_context.messages[0] is initial_system


@dataclass
class CustomNotification:
    text: str
    role: str = "notification"
    timestamp: int = field(default_factory=now_ms)


async def test_should_handle_custom_message_types_via_convert_to_llm() -> None:
    notification = CustomNotification("This is a notification")
    context = AgentContext(messages=[notification], tools=[])
    converted_messages: list[Message] = []

    def convert_to_llm(messages: list[Any]) -> list[Message]:
        nonlocal converted_messages
        converted_messages = [
            message
            for message in messages
            if message.role != "notification" and message.role in ("user", "assistant", "toolResult")
        ]
        return converted_messages

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(create_assistant_message([TextContent("Response")]))

    stream = agent_loop(
        [create_user_message("Hello")],
        context,
        AgentLoopConfig(model=create_model(), convert_to_llm=convert_to_llm),
        None,
        stream_fn,
    )
    await collect(stream)

    # The notification was filtered out in convert_to_llm.
    assert len(converted_messages) == 1
    assert converted_messages[0].role == "user"


async def test_should_apply_transform_context_before_convert_to_llm() -> None:
    context = AgentContext(
        messages=[
            create_user_message("old message 1"),
            create_assistant_message([TextContent("old response 1")]),
            create_user_message("old message 2"),
            create_assistant_message([TextContent("old response 2")]),
        ],
        tools=[],
    )
    transformed_messages: list[Any] = []
    converted_messages: list[Message] = []

    async def transform_context(messages: list[Any], _signal: Any) -> list[Any]:
        nonlocal transformed_messages
        transformed_messages = messages[-2:]
        return transformed_messages

    def convert_to_llm(messages: list[Any]) -> list[Message]:
        nonlocal converted_messages
        converted_messages = [message for message in messages if message.role in ("user", "assistant", "toolResult")]
        return converted_messages

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(create_assistant_message([TextContent("Response")]))

    stream = agent_loop(
        [create_user_message("new message")],
        context,
        AgentLoopConfig(model=create_model(), transform_context=transform_context, convert_to_llm=convert_to_llm),
        None,
        stream_fn,
    )
    await collect(stream)

    assert len(transformed_messages) == 2
    assert len(converted_messages) == 2


async def test_should_handle_tool_calls_and_results() -> None:
    executed: list[str] = []
    tool_usage = Usage(1, 2, 3, 4, 10, UsageCost(0.1, 0.2, 0.3, 0.4, 1))
    patched_tool_usage = Usage(5, 6, 7, 8, 26, UsageCost(0.5, 0.6, 0.7, 0.8, 2.6))
    observed_tool_usage: list[Any] = []

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(
            [TextContent(f"echoed: {params['value']}")], {"value": params["value"]}, usage=tool_usage
        )

    async def after_tool_call(ctx: Any, _signal: Any) -> AfterToolCallResult:
        observed_tool_usage.append(ctx.result.usage)
        return AfterToolCallResult(usage=patched_tool_usage)

    context = AgentContext(messages=[], tools=[echo_tool(execute)])
    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse"))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo something")], context, config(after_tool_call=after_tool_call), None, stream_fn
    )
    events = await collect(stream)

    assert executed == ["hello"]
    tool_start = next((event for event in events if event.type == "tool_execution_start"), None)
    tool_end = next((event for event in events if event.type == "tool_execution_end"), None)
    assert tool_start is not None
    assert tool_end is not None
    assert tool_end.is_error is False
    assert observed_tool_usage == [tool_usage]
    messages = await stream.result()
    tool_result = next(message for message in messages if message.role == "toolResult")
    assert tool_result.usage == patched_tool_usage


# #10549
async def test_records_how_long_execute_took_on_the_tool_result_excluding_hooks() -> None:
    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        await asyncio.sleep(0.03)
        return AgentToolResult([TextContent(params["value"])], {"value": params["value"]})

    async def before_tool_call(ctx: Any, _signal: Any) -> BeforeToolCallResult | None:
        await asyncio.sleep(0.1)
        return BeforeToolCallResult(block=True, reason="no") if ctx.tool_call.id == "blocked" else None

    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return respond(
                create_assistant_message(
                    [
                        tool_call("ran", "echo", {"value": "a"}),
                        tool_call("blocked", "echo", {"value": "b"}),
                    ],
                    "toolUse",
                )
            )
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("go")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(before_tool_call=before_tool_call),
        None,
        stream_fn,
    )
    await collect(stream)
    results = [message for message in await stream.result() if message.role == "toolResult"]
    ran, blocked = results
    assert ran.duration_ms is not None
    assert 25 <= ran.duration_ms < 100
    assert blocked.is_error is True
    assert blocked.duration_ms is None


async def test_should_not_execute_tool_calls_from_a_length_truncated_assistant_message() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]})

    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            # Output hit the token limit mid tool call. The salvage parser can produce arguments
            # that validate but are silently truncated, so nothing in this message may execute.
            return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hel"})], "length"))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo something")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(),
        None,
        stream_fn,
    )
    events = await collect(stream)

    # The tool must never execute with potentially truncated arguments.
    assert executed == []
    tool_end = next((event for event in events if event.type == "tool_execution_end"), None)
    assert tool_end is not None
    assert tool_end.is_error is True
    text = next(block for block in tool_end.result.content if block.type == "text")
    assert "output token limit" in text.text

    # The loop continues so the model can re-issue the tool call.
    assert call_index == 2
    messages = await stream.result()
    assert messages[-1].role == "assistant"


async def test_should_execute_mutated_before_tool_call_args_without_revalidation() -> None:
    executed: list[Any] = []

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]})

    async def before_tool_call(ctx: Any, _signal: Any) -> None:
        ctx.args["value"] = 123
        return None

    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse"))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo something")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(before_tool_call=before_tool_call),
        None,
        stream_fn,
    )
    await collect(stream)

    assert executed == [123]


async def test_should_prepare_tool_arguments_for_validation() -> None:
    replace_schema = Type.Object({"oldText": Type.String(), "newText": Type.String()})
    schema = Type.Object({"edits": Type.Array(replace_schema)})
    executed: list[Any] = []

    def prepare_arguments(args: Any) -> Any:
        if not isinstance(args, dict):
            return args
        if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
            return args
        return {"edits": [*(args.get("edits") or []), {"oldText": args["oldText"], "newText": args["newText"]}]}

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        executed.append(params["edits"])
        return AgentToolResult([TextContent(f"edited {len(params['edits'])}")], {"count": len(params["edits"])})

    tool = echo_tool(execute, name="edit", parameters=schema, prepare_arguments=prepare_arguments)
    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return respond(
                create_assistant_message(
                    [tool_call("tool-1", "edit", {"oldText": "before", "newText": "after"})], "toolUse"
                )
            )
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("edit something")], AgentContext(messages=[], tools=[tool]), config(), None, stream_fn
    )
    await collect(stream)

    assert executed == [[{"oldText": "before", "newText": "after"}]]


def _two_echo_calls(first_name: str = "echo", second_name: str = "echo", values: tuple[str, str] = ("first", "second")):
    return create_assistant_message(
        [
            tool_call("tool-1", first_name, {"value": values[0]}),
            tool_call("tool-2", second_name, {"value": values[1]}),
        ],
        "toolUse",
    )


async def test_should_emit_tool_execution_end_in_completion_order_but_persist_tool_results_in_source_order() -> None:
    first_resolved = False
    parallel_observed = False
    release_first = asyncio.Event()

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            await release_first.wait()
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]})

    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            asyncio.get_running_loop().call_later(0.02, release_first.set)
            return respond(_two_echo_calls())
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo both")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(tool_execution="parallel"),
        None,
        stream_fn,
    )
    events = await collect(stream)

    tool_execution_end_ids = [event.tool_call_id for event in events if event.type == "tool_execution_end"]
    tool_result_ids = [
        event.message.tool_call_id
        for event in events
        if event.type == "message_end" and event.message.role == "toolResult"
    ]
    turn_tool_result_ids = [
        result.tool_call_id for event in events if event.type == "turn_end" for result in event.tool_results
    ]

    assert parallel_observed is True
    assert tool_execution_end_ids == ["tool-2", "tool-1"]
    assert tool_result_ids == ["tool-1", "tool-2"]
    assert turn_tool_result_ids == ["tool-1", "tool-2"]


async def test_should_inject_queued_messages_after_all_tool_calls_complete() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult([TextContent(f"ok:{params['value']}")], {"value": params["value"]})

    queued_user_message = create_user_message("interrupt")
    queued_delivered = False
    call_index = 0
    saw_interrupt_in_context = False

    async def get_steering_messages() -> list[Any]:
        nonlocal queued_delivered
        # Return the steering message after tool execution has started.
        if len(executed) >= 1 and not queued_delivered:
            queued_delivered = True
            return [queued_user_message]
        return []

    def stream_fn(_model: Any, ctx: Any, _options: Any) -> AssistantMessageEventStream:
        nonlocal call_index, saw_interrupt_in_context
        if call_index == 1:
            saw_interrupt_in_context = any(
                message.role == "user" and message.content == "interrupt" for message in ctx.messages
            )
        call_index += 1
        if call_index == 1:
            return respond(_two_echo_calls())
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("start")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(tool_execution="sequential", get_steering_messages=get_steering_messages),
        None,
        stream_fn,
    )
    events = await collect(stream)

    # Both tools execute before steering is injected.
    assert executed == ["first", "second"]
    tool_ends = [event for event in events if event.type == "tool_execution_end"]
    assert len(tool_ends) == 2
    assert tool_ends[0].is_error is False
    assert tool_ends[1].is_error is False

    # The queued message appears after both tool result messages.
    sequence: list[str] = []
    for event in events:
        if event.type != "message_start":
            continue
        if event.message.role == "toolResult":
            sequence.append(f"tool:{event.message.tool_call_id}")
        elif event.message.role == "user" and isinstance(event.message.content, str):
            sequence.append(event.message.content)
    assert "interrupt" in sequence
    assert sequence.index("tool:tool-1") < sequence.index("interrupt")
    assert sequence.index("tool:tool-2") < sequence.index("interrupt")

    # The interrupt message is in context when the second LLM call is made.
    assert saw_interrupt_in_context is True


async def test_should_force_sequential_execution_when_a_tool_has_execution_mode_sequential() -> None:
    first_resolved = False
    parallel_observed = False
    release_first = asyncio.Event()

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            await release_first.wait()
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult([TextContent(f"slow: {params['value']}")], {"value": params["value"]})

    slow_tool = echo_tool(execute, name="slow", execution_mode="sequential")
    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            asyncio.get_running_loop().call_later(0.02, release_first.set)
            return respond(_two_echo_calls("slow", "slow"))
        return respond(create_assistant_message([TextContent("done")]))

    # The config is parallel (default), but the tool forces sequential execution.
    stream = agent_loop(
        [create_user_message("run both")], AgentContext(messages=[], tools=[slow_tool]), config(), None, stream_fn
    )
    events = await collect(stream)

    # With sequential execution the second tool does not start before the first finishes.
    assert parallel_observed is False
    tool_result_ids = [
        event.message.tool_call_id
        for event in events
        if event.type == "message_end" and event.message.role == "toolResult"
    ]
    assert tool_result_ids == ["tool-1", "tool-2"]


async def test_should_force_sequential_execution_when_one_of_multiple_tools_has_execution_mode_sequential() -> None:
    execution_order: list[str] = []
    release_slow = asyncio.Event()

    async def slow_execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        execution_order.append(f"slow:{params['value']}")
        if params["value"] == "a":
            await release_slow.wait()
        return AgentToolResult([TextContent(f"slow: {params['value']}")], {"value": params["value"]})

    async def fast_execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        execution_order.append(f"fast:{params['value']}")
        return AgentToolResult([TextContent(f"fast: {params['value']}")], {"value": params["value"]})

    slow_tool = echo_tool(slow_execute, name="slow", execution_mode="sequential")
    fast_tool = echo_tool(fast_execute, name="fast")  # no execution_mode: defaults to parallel
    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            asyncio.get_running_loop().call_later(0.02, release_slow.set)
            return respond(_two_echo_calls("slow", "fast", ("a", "b")))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("run both")],
        AgentContext(messages=[], tools=[slow_tool, fast_tool]),
        config(),
        None,
        stream_fn,
    )
    await collect(stream)

    # The fast tool does not run before the slow tool finishes.
    assert execution_order[0] == "slow:a"
    assert "fast:b" in execution_order


async def test_should_allow_parallel_execution_when_all_tools_have_execution_mode_parallel() -> None:
    first_resolved = False
    parallel_observed = False
    release_first = asyncio.Event()

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            await release_first.wait()
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]})

    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            asyncio.get_running_loop().call_later(0.02, release_first.set)
            return respond(_two_echo_calls())
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo both")],
        AgentContext(messages=[], tools=[echo_tool(execute, execution_mode="parallel")]),
        config(),
        None,
        stream_fn,
    )
    await collect(stream)

    # With execution_mode="parallel" the second tool starts before the first finishes.
    assert parallel_observed is True


async def test_runs_finish_turn_after_tool_result_messages_and_before_turn_end() -> None:
    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        return AgentToolResult([TextContent(params["value"])], {"value": params["value"]}, terminate=True)

    ordering: list[str] = []

    def finish_turn(turn: Any, _signal: Any) -> None:
        ordering.append("finishTurn")
        assert len(turn.tool_results) == 1
        assert turn.context.messages[-1].role == "toolResult"

    def emit(event: AgentEvent) -> None:
        if event.type == "message_end":
            ordering.append(f"message_end:{event.message.role}")
        if event.type == "turn_end":
            ordering.append("turn_end")

    await run_agent_loop(
        [create_user_message("echo")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(finish_turn=finish_turn),
        emit,
        None,
        lambda *_args: respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse")),
    )

    assert ordering[-3:] == ["message_end:toolResult", "finishTurn", "turn_end"]


@pytest.mark.parametrize("reason", ["error", "aborted"])
async def test_runs_finish_turn_for_a_failed_assistant_before_turn_end_without_changing_the_hard_exit(
    reason: str,
) -> None:
    ordering: list[str] = []
    provider_calls = 0
    steering_polls = 0
    follow_up_polls = 0

    def finish_turn(turn: Any, _signal: Any) -> AgentTurnDecision:
        assert turn.message.stop_reason == reason
        ordering.append("finishTurn")
        return AgentTurnDecision("continue")

    async def get_steering_messages() -> list[Any]:
        nonlocal steering_polls
        steering_polls += 1
        return []

    async def get_follow_up_messages() -> list[Any]:
        nonlocal follow_up_polls
        follow_up_polls += 1
        return [create_user_message("queued")]

    def emit(event: AgentEvent) -> None:
        if event.type == "turn_end":
            ordering.append("turn_end")

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal provider_calls
        provider_calls += 1
        message = create_assistant_message([], reason)
        message.error_message = reason
        return respond(message)

    await run_agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[]),
        config(
            finish_turn=finish_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        ),
        emit,
        None,
        stream_fn,
    )

    assert ordering == ["finishTurn", "turn_end"]
    assert provider_calls == 1
    assert steering_polls == 1
    assert follow_up_polls == 0


async def test_action_end_skips_queue_polling_and_next_turn_preparation() -> None:
    provider_calls = 0
    steering_polls = 0
    follow_up_polls = 0
    prepare_next_turn_calls = 0

    def prepare_next_turn(_turn: Any) -> None:
        nonlocal prepare_next_turn_calls
        prepare_next_turn_calls += 1
        return None

    async def get_steering_messages() -> list[Any]:
        nonlocal steering_polls
        steering_polls += 1
        return []

    async def get_follow_up_messages() -> list[Any]:
        nonlocal follow_up_polls
        follow_up_polls += 1
        return [create_user_message("queued")]

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal provider_calls
        provider_calls += 1
        return respond(create_assistant_message([tool_call("tool-1", "noop", {})], "toolUse"))

    stream = agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[noop_tool()]),
        config(
            finish_turn=lambda _turn, _signal: {"action": "end"},
            prepare_next_turn=prepare_next_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        ),
        None,
        stream_fn,
    )
    await stream.result()

    assert provider_calls == 1
    assert steering_polls == 1
    assert follow_up_polls == 0
    assert prepare_next_turn_calls == 0


async def test_makes_exactly_one_context_only_request_when_no_natural_request_satisfies_continuation() -> None:
    provider_calls = 0
    finish_calls = 0

    def finish_turn(_turn: Any, _signal: Any) -> AgentTurnDecision | None:
        nonlocal finish_calls
        finish_calls += 1
        return AgentTurnDecision("continue") if finish_calls == 1 else None

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal provider_calls
        provider_calls += 1
        return respond(create_assistant_message([TextContent(f"response {provider_calls}")]))

    stream = agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[]),
        config(finish_turn=finish_turn),
        None,
        stream_fn,
    )
    await stream.result()

    assert provider_calls == 2
    assert finish_calls == 2


async def test_lets_a_natural_tool_result_request_satisfy_continuation() -> None:
    provider_calls = 0
    finish_calls = 0

    def finish_turn(_turn: Any, _signal: Any) -> AgentTurnDecision | None:
        nonlocal finish_calls
        finish_calls += 1
        return AgentTurnDecision("continue") if finish_calls == 1 else None

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return respond(create_assistant_message([tool_call("tool-1", "noop", {})], "toolUse"))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[noop_tool()]),
        config(finish_turn=finish_turn),
        None,
        stream_fn,
    )
    await stream.result()

    assert provider_calls == 2
    assert finish_calls == 2


@pytest.mark.parametrize("queue_kind", ["steering", "follow-up"])
async def test_lets_a_natural_queued_request_satisfy_continuation(queue_kind: str) -> None:
    queued_message = create_user_message(queue_kind)
    provider_calls = 0
    finish_calls = 0
    steering_polls = 0
    follow_up_delivered = False
    second_request_users: list[str] = []

    def finish_turn(_turn: Any, _signal: Any) -> AgentTurnDecision | None:
        nonlocal finish_calls
        finish_calls += 1
        return AgentTurnDecision("continue") if finish_calls == 1 else None

    async def get_steering_messages() -> list[Any]:
        nonlocal steering_polls
        steering_polls += 1
        return [queued_message] if queue_kind == "steering" and steering_polls == 2 else []

    async def get_follow_up_messages() -> list[Any]:
        nonlocal follow_up_delivered
        if queue_kind != "follow-up" or follow_up_delivered:
            return []
        follow_up_delivered = True
        return [queued_message]

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 2:
            second_request_users.extend(
                message.content
                for message in context.messages
                if message.role == "user" and isinstance(message.content, str)
            )
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[]),
        config(
            finish_turn=finish_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        ),
        None,
        stream_fn,
    )
    await stream.result()

    assert provider_calls == 2
    assert finish_calls == 2
    assert queue_kind in second_request_users


async def test_prepares_the_initial_request_after_pending_messages_and_can_replace_request_state() -> None:
    replacement_model = create_model()
    replacement_model.id = "replacement"
    replacement_model.name = "replacement"
    canonical_message = create_user_message("canonical projection")
    steering_message = create_user_message("steering")
    completed_messages: list[Any] = []
    steering_delivered = False
    prepare_calls = 0
    requests: list[tuple[Any, list[Any], Any]] = []

    async def get_steering_messages() -> list[Any]:
        nonlocal steering_delivered
        if steering_delivered:
            return []
        steering_delivered = True
        return [steering_message]

    def prepare_request(request: Any, _signal: Any) -> AgentRequestUpdate:
        nonlocal prepare_calls
        prepare_calls += 1
        assert steering_message in completed_messages
        assert steering_message in request.context.messages
        return AgentRequestUpdate(
            context=AgentContext(messages=[canonical_message], tools=request.context.tools),
            model=replacement_model,
            thinking_level="high",
        )

    def emit(event: AgentEvent) -> None:
        if event.type == "message_end":
            completed_messages.append(event.message)

    def stream_fn(model: Any, context: Any, options: Any) -> AssistantMessageEventStream:
        requests.append((model, list(context.messages), (options or {}).get("reasoning")))
        return respond(create_assistant_message([TextContent("done")]))

    await run_agent_loop(
        [create_user_message("prompt")],
        AgentContext(messages=[], tools=[]),
        config(get_steering_messages=get_steering_messages, prepare_request=prepare_request),
        emit,
        None,
        stream_fn,
    )

    assert prepare_calls == 1
    assert len(requests) == 1
    model, messages, reasoning = requests[0]
    assert model is replacement_model
    assert messages == [canonical_message]
    assert reasoning == "high"


async def test_does_not_poll_steering_after_prepare_request() -> None:
    queued: list[Any] = []
    late_steering = create_user_message("late steering")
    request_included_steering: list[bool] = []
    request_preparations = 0
    steering_polls = 0

    async def get_steering_messages() -> list[Any]:
        nonlocal steering_polls
        steering_polls += 1
        drained = list(queued)
        queued.clear()
        return drained

    def prepare_request(_request: Any, _signal: Any) -> None:
        nonlocal request_preparations
        request_preparations += 1
        if request_preparations == 1:
            queued.append(late_steering)

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        request_included_steering.append(any(message is late_steering for message in context.messages))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[]),
        config(get_steering_messages=get_steering_messages, prepare_request=prepare_request),
        None,
        stream_fn,
    )
    await stream.result()

    assert request_included_steering == [False, True]
    assert request_preparations == 2
    # Startup, post-turn delivery, then the final natural-stop check.
    assert steering_polls == 3


async def test_should_use_prepare_next_turn_snapshot_before_continuing() -> None:
    converted_second_turn_has_update = False
    prepare_calls = 0
    prepared = False

    async def prepare_next_turn(turn: Any) -> AgentLoopTurnUpdate | None:
        nonlocal prepare_calls, prepared
        prepare_calls += 1
        if prepared:
            return None
        prepared = True
        return AgentLoopTurnUpdate(
            context=AgentContext(messages=list(turn.context.messages), tools=turn.context.tools),
            messages=[SystemMessage("updated guidance", timestamp=1)],
        )

    llm_calls = 0

    def stream_fn(_model: Any, ctx: Any, _options: Any) -> AssistantMessageEventStream:
        nonlocal llm_calls, converted_second_turn_has_update
        llm_calls += 1
        if llm_calls == 2:
            converted_second_turn_has_update = any(
                message.role == "system" and message.content == "updated guidance" for message in ctx.messages
            )
        if llm_calls == 1:
            return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse"))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo something")],
        AgentContext(messages=[], tools=[echo_tool()]),
        config(prepare_next_turn=prepare_next_turn),
        None,
        stream_fn,
    )
    await collect(stream)

    assert llm_calls == 2
    assert prepare_calls == 1
    assert converted_second_turn_has_update is True


async def test_picks_up_steering_queued_during_prepare_next_turn_before_the_next_request() -> None:
    queued: list[Any] = []
    late_steering = create_user_message("late steering")
    provider_calls = 0
    second_request_included_steering = False

    def prepare_next_turn(_turn: Any) -> None:
        queued.append(late_steering)
        return None

    async def get_steering_messages() -> list[Any]:
        drained = list(queued)
        queued.clear()
        return drained

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        nonlocal provider_calls, second_request_included_steering
        provider_calls += 1
        if provider_calls == 2:
            second_request_included_steering = any(message is late_steering for message in context.messages)
        if provider_calls == 1:
            return respond(create_assistant_message([tool_call("tool-1", "noop", {})], "toolUse"))
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("run")],
        AgentContext(messages=[], tools=[noop_tool()]),
        config(prepare_next_turn=prepare_next_turn, get_steering_messages=get_steering_messages),
        None,
        stream_fn,
    )
    await stream.result()

    assert provider_calls == 2
    assert second_request_included_steering is True


async def test_action_end_receives_finalized_turn_context_and_stops_before_queue_polling() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]})

    steering_polls = 0
    follow_up_polls = 0
    callback_tool_result_ids: list[str] = []
    callback_context_roles: list[str] = []

    async def finish_turn(turn: Any, _signal: Any) -> AgentTurnDecision:
        nonlocal callback_tool_result_ids, callback_context_roles
        assert turn.message.role == "assistant"
        callback_tool_result_ids = [result.tool_call_id for result in turn.tool_results]
        callback_context_roles = [message.role for message in turn.context.messages]
        return AgentTurnDecision("end")

    async def get_steering_messages() -> list[Any]:
        nonlocal steering_polls
        steering_polls += 1
        return []

    async def get_follow_up_messages() -> list[Any]:
        nonlocal follow_up_polls
        follow_up_polls += 1
        return [create_user_message("follow up should stay queued")]

    llm_calls = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse"))
        return respond(create_assistant_message([TextContent("should not run")]))

    stream = agent_loop(
        [create_user_message("echo something")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(
            finish_turn=finish_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        ),
        None,
        stream_fn,
    )
    events = await collect(stream)

    messages = await stream.result()
    assert llm_calls == 1
    assert executed == ["hello"]
    assert steering_polls == 1
    assert follow_up_polls == 0
    assert callback_tool_result_ids == ["tool-1"]
    assert callback_context_roles == ["system", "user", "assistant", "toolResult"]
    # The context declares no tools, so the loop announces the loadout with a system message.
    assert [message.role for message in messages] == ["system", "user", "assistant", "toolResult"]
    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


async def test_should_stop_after_a_tool_batch_when_every_tool_result_sets_terminate() -> None:
    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]}, terminate=True)

    llm_calls = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal llm_calls
        llm_calls += 1
        return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse"))

    stream = agent_loop(
        [create_user_message("echo something")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(),
        None,
        stream_fn,
    )
    events = await collect(stream)

    messages = await stream.result()
    assert llm_calls == 1
    assert [message.role for message in messages] == ["system", "user", "assistant", "toolResult"]
    assert len([event for event in events if event.type == "turn_end"]) == 1


async def test_should_stop_after_a_blocked_tool_call_when_before_tool_call_sets_terminate() -> None:
    executed = False

    async def execute(*_args: Any) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult([TextContent("should not execute")], {"value": "unexpected"})

    async def before_tool_call(_ctx: Any, _signal: Any) -> BeforeToolCallResult:
        return BeforeToolCallResult(block=True, reason="Blocked by policy", terminate=True)

    llm_calls = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse"))
        return respond(create_assistant_message([TextContent("should not run")]))

    stream = agent_loop(
        [create_user_message("echo something")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(before_tool_call=before_tool_call),
        None,
        stream_fn,
    )
    await collect(stream)

    messages = await stream.result()
    tool_result = next(message for message in messages if message.role == "toolResult")
    assert executed is False
    assert llm_calls == 1
    assert tool_result.is_error is True
    assert TextContent("Blocked by policy") in tool_result.content


async def test_should_continue_after_a_mixed_batch_with_one_terminating_blocked_call() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult([TextContent(f"echoed: {params['value']}")], {"value": params["value"]})

    async def before_tool_call(ctx: Any, _signal: Any) -> BeforeToolCallResult | None:
        if ctx.args["value"] == "first":
            return BeforeToolCallResult(block=True, reason="Blocked first", terminate=True)
        return None

    llm_calls = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return respond(_two_echo_calls())
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo both")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(tool_execution="parallel", before_tool_call=before_tool_call),
        None,
        stream_fn,
    )
    await collect(stream)

    assert executed == ["second"]
    assert llm_calls == 2


async def test_should_continue_after_parallel_tool_calls_when_not_all_tool_results_terminate() -> None:
    async def execute(_tool_call_id: str, params: Any, *_rest: Any) -> AgentToolResult:
        return AgentToolResult(
            [TextContent(f"echoed: {params['value']}")],
            {"value": params["value"]},
            terminate=params["value"] == "first",
        )

    call_index = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return respond(_two_echo_calls())
        return respond(create_assistant_message([TextContent("done")]))

    stream = agent_loop(
        [create_user_message("echo both")],
        AgentContext(messages=[], tools=[echo_tool(execute)]),
        config(tool_execution="parallel"),
        None,
        stream_fn,
    )
    await collect(stream)

    messages = await stream.result()
    assert call_index == 2
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "toolResult",
        "toolResult",
        "assistant",
    ]


async def test_should_allow_after_tool_call_to_mark_a_tool_batch_as_terminating() -> None:
    async def after_tool_call(_ctx: Any, _signal: Any) -> AfterToolCallResult:
        return AfterToolCallResult(terminate=True)

    llm_calls = 0

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal llm_calls
        llm_calls += 1
        return respond(create_assistant_message([tool_call("tool-1", "echo", {"value": "hello"})], "toolUse"))

    stream = agent_loop(
        [create_user_message("echo something")],
        AgentContext(messages=[], tools=[echo_tool()]),
        config(after_tool_call=after_tool_call),
        None,
        stream_fn,
    )
    await collect(stream)

    assert llm_calls == 1


# ---------------------------------------------------------------------------
# agent_loop_continue with agent messages
# ---------------------------------------------------------------------------


def test_continue_should_throw_when_context_has_no_messages() -> None:
    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        raise AssertionError("Unexpected stream call")

    with pytest.raises(ValueError, match="Cannot continue: no messages in context"):
        agent_loop_continue(AgentContext(messages=[], tools=[]), config(), None, stream_fn)


async def test_continue_should_continue_from_existing_context_without_emitting_user_message_events() -> None:
    context = AgentContext(messages=[create_user_message("Hello")], tools=[])

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(create_assistant_message([TextContent("Response")]))

    stream = agent_loop_continue(context, config(), None, stream_fn)
    events = await collect(stream)
    messages = await stream.result()

    # Only the new assistant message is returned (not the existing user message).
    assert len(messages) == 1
    assert messages[0].role == "assistant"

    # No user message events: that is the key difference from agent_loop.
    message_end_events = [event for event in events if event.type == "message_end"]
    assert len(message_end_events) == 1
    assert message_end_events[0].message.role == "assistant"


@dataclass
class CustomMessage:
    text: str
    role: str = "custom"
    timestamp: int = field(default_factory=now_ms)


async def test_continue_should_allow_custom_message_types_as_last_message() -> None:
    context = AgentContext(messages=[CustomMessage("Hook content")], tools=[])

    def convert_to_llm(messages: list[Any]) -> list[Message]:
        converted: list[Any] = [
            UserMessage(message.text, timestamp=message.timestamp) if message.role == "custom" else message
            for message in messages
        ]
        return [message for message in converted if message.role in ("user", "assistant", "toolResult")]

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(create_assistant_message([TextContent("Response to custom message")]))

    # Does not raise: the custom message is converted to a user message.
    stream = agent_loop_continue(
        context, AgentLoopConfig(model=create_model(), convert_to_llm=convert_to_llm), None, stream_fn
    )
    await collect(stream)

    messages = await stream.result()
    assert len(messages) == 1
    assert messages[0].role == "assistant"


# ---------------------------------------------------------------------------
# run_tool_call
# ---------------------------------------------------------------------------


async def _echo_execute(_tool_call_id: str, params: Any, _signal: Any = None, on_update: Any = None) -> AgentToolResult:
    if on_update is not None:
        on_update(AgentToolResult([TextContent("partial")], {}))
    return AgentToolResult([TextContent(params["value"])], {}, structured_content={"value": params["value"]})


async def _failing_execute(*_args: Any) -> AgentToolResult:
    return AgentToolResult([TextContent("bad")], {"partial": True}, is_error=True)


RUN_ECHO = AgentTool(
    "echo",
    "Echo tool",
    ECHO_SCHEMA,
    label="Echo",
    output_schema=Type.Object({"value": Type.String()}),
    execute=_echo_execute,
)
RUN_FAILING = AgentTool(
    "failing", "Returns an error result", Type.Object({}), label="Failing", execute=_failing_execute
)


async def test_run_tool_call_validates_runs_the_hooks_and_reports_failures_as_error_outcomes() -> None:
    hook_calls: list[str] = []
    updates: list[Any] = []

    async def before_tool_call(ctx: Any, _signal: Any) -> BeforeToolCallResult | None:
        hook_calls.append(f"before {ctx.tool_call.id}")
        if ctx.args.get("value") == "blocked":
            return BeforeToolCallResult(block=True, reason="nope")
        return None

    async def after_tool_call(ctx: Any, _signal: Any) -> None:
        hook_calls.append(f"after {ctx.tool_call.id}")
        return None

    options = RunToolCallOptions(
        tools=[RUN_ECHO, RUN_FAILING],
        assistant_message=create_assistant_message([]),
        context=AgentContext(messages=[]),
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        on_update=updates.append,
    )

    outcome = await run_tool_call(tool_call("a", "echo", {"value": "a"}), options)
    assert outcome.tool_call.id == "a"
    assert outcome.result.structured_content == {"value": "a"}
    assert outcome.is_error is False

    outcome = await run_tool_call(tool_call("b", "echo", {"value": {"nested": True}}), options)
    assert outcome.is_error is True

    outcome = await run_tool_call(tool_call("c", "echo", {"value": "blocked"}), options)
    assert outcome.result.content == [TextContent("nope")]
    assert outcome.is_error is True

    outcome = await run_tool_call(tool_call("d", "missing", {}), options)
    assert outcome.result.content == [TextContent("Tool missing not found")]
    assert outcome.is_error is True

    # Error results keep their details.
    outcome = await run_tool_call(tool_call("e", "failing", {}), options)
    assert outcome.result.details == {"partial": True}
    assert outcome.is_error is True

    assert updates == [AgentToolResult([TextContent("partial")], {})]
    # Validation failures and unknown tools never reach the hooks; blocked calls skip after_tool_call.
    assert hook_calls == ["before a", "after a", "before c", "before e", "after e"]


async def test_run_tool_call_lets_after_tool_call_replace_structured_content() -> None:
    """And drops structured content when only the content is replaced."""
    redacted = [TextContent("redacted")]
    results = [
        AfterToolCallResult(content=redacted),
        AfterToolCallResult(structured_content={"value": "replaced"}),
        AfterToolCallResult(content=redacted, structured_content={"value": "both"}),
        AfterToolCallResult(details={"note": "kept"}),
    ]
    seen: list[Any] = []
    for after_result in results:

        async def after_tool_call(_ctx: Any, _signal: Any, after_result: AfterToolCallResult = after_result) -> Any:
            return after_result

        outcome = await run_tool_call(
            tool_call("x", "echo", {"value": "original"}),
            RunToolCallOptions(
                tools=[RUN_ECHO],
                assistant_message=create_assistant_message([]),
                context=AgentContext(messages=[]),
                after_tool_call=after_tool_call,
            ),
        )
        seen.append(outcome.result.structured_content)

    assert seen == [None, {"value": "replaced"}, {"value": "both"}, {"value": "original"}]


async def test_tool_results_are_typed_tool_result_messages() -> None:
    """The persisted tool results are ``ToolResultMessage`` instances carrying the tool's details."""
    stream = agent_loop(
        [create_user_message("echo")],
        AgentContext(messages=[], tools=[echo_tool()]),
        config(after_tool_call=None),
        None,
        _scripted(
            create_assistant_message([tool_call("tool-1", "echo", {"value": "hi"})], "toolUse"),
            create_assistant_message([TextContent("done")]),
        ),
    )
    messages = await stream.result()
    tool_result = messages[3]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.details == {"value": "hi"}
    assert tool_result.content == [TextContent("echoed: hi")]
    assert tool_result.duration_ms is not None


def _scripted(*messages: Any) -> Any:
    remaining = list(messages)

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(remaining.pop(0))

    return stream_fn
