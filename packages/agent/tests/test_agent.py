"""Port of upstream ``test/agent.test.ts``."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from agent_helpers import (
    create_assistant_message,
    create_assistant_tool_use_message,
    create_user_message,
    respond,
    user_texts,
)

from pi_agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentTool,
    AgentToolResult,
    AgentTurnDecision,
    set_default_stream_fn,
)
from pi_ai import (
    AbortSignal,
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    StartEvent,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    Type,
    UserMessage,
    get_current_system_message,
    now_ms,
    to_tool_declaration,
)


def create_tool(name: str, description: str | None = None) -> AgentTool:
    async def execute(*_args: Any) -> AgentToolResult:
        return AgentToolResult([TextContent(name)], {})

    return AgentTool(
        name,
        description if description is not None else f"{name} tool",
        Type.Object({}),
        label=name,
        execute=execute,
    )


def unused_stream_function(*_args: Any) -> AssistantMessageEventStream:
    raise AssertionError("Unexpected stream call")


def done_stream_fn(text: str = "done") -> Any:
    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(create_assistant_message(text))

    return stream_fn


def abortable_stream_fn(captured: list[Any] | None = None) -> Any:
    """Starts a response, then waits until the request signal aborts (polling like upstream)."""

    def stream_fn(_model: Any, _context: Any, options: Any) -> AssistantMessageEventStream:
        signal = (options or {}).get("signal")
        if captured is not None:
            captured.append(signal)
        stream = AssistantMessageEventStream()

        async def produce() -> None:
            stream.push(StartEvent(create_assistant_message("")))
            while not (signal is not None and signal.aborted):
                await asyncio.sleep(0.005)
            aborted = create_assistant_message("Aborted", "aborted")
            aborted.error_message = "Request was aborted"
            stream.push(ErrorEvent("aborted", aborted))

        stream.start(produce())
        return stream

    return stream_fn


def respond_done(text: str) -> DoneEvent:
    return DoneEvent("stop", create_assistant_message(text))


def model_named(provider: str, model_id: str) -> Model:
    return Model(id=model_id, name=model_id, api="openai-responses", provider=provider)


def is_user_with_text(message: Any, text: str) -> bool:
    if getattr(message, "role", None) != "user":
        return False
    if isinstance(message.content, str):
        return message.content == text
    return any(part.type == "text" and part.text == text for part in message.content)


async def test_uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() -> None:
    calls = 0

    def default_stream_fn(*_args: Any) -> AssistantMessageEventStream:
        nonlocal calls
        calls += 1
        return respond(create_assistant_message("fallback"))

    set_default_stream_fn(default_stream_fn)
    try:
        agent = Agent()
        await agent.prompt("Hello")
        assert calls == 1
    finally:
        set_default_stream_fn(None)


def test_should_create_an_agent_instance_with_default_state() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    assert agent.state is not None
    assert agent.state.model is not None
    assert agent.state.thinking_level == "off"
    assert agent.state.tools == []
    assert agent.state.messages == []
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == frozenset()
    assert agent.state.error_message is None


def test_should_create_an_agent_instance_with_custom_initial_state() -> None:
    custom_model = model_named("openai", "gpt-4o-mini")
    agent = Agent(
        stream_fn=unused_stream_function,
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant.", model=custom_model, thinking_level="low"
        ),
    )

    assert agent.state.messages == [SystemMessage("You are a helpful assistant.", timestamp=0)]
    assert agent.state.model is custom_model
    assert agent.state.thinking_level == "low"


def test_accepts_initial_state_as_a_mapping() -> None:
    agent = Agent(stream_fn=unused_stream_function, initial_state={"system_prompt": "Hi", "thinking_level": "high"})

    assert agent.state.system_prompt == "Hi"
    assert agent.state.thinking_level == "high"


def test_converts_initial_prompt_and_tools_into_transcript_state() -> None:
    tool = create_tool("echo", "Echo input")
    agent = Agent(
        initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[tool]),
        stream_fn=unused_stream_function,
    )

    initial = agent.state.messages[0]
    assert initial.role == "system"
    assert initial.content == "You are helpful."
    assert [value.name for value in initial.tools_added] == ["echo"]


async def test_declares_tool_loadout_changes_to_the_model_before_the_next_request() -> None:
    first = create_tool("first")
    second = create_tool("second")
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        entries: list[str] = []
        for message in context.messages:
            if message.role == "system":
                entries.append("+" + ",".join(tool.name for tool in message.tools_added or []))
                entries.append("-" + ",".join(tool.name for tool in message.tools_removed or []))
        requests.append(entries)
        return respond(create_assistant_message("done"))

    agent = Agent(
        initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[first]),
        stream_fn=stream_fn,
    )

    await agent.prompt("one")
    agent.state.tools = [second]
    await agent.prompt("two")
    await agent.prompt("three")

    assert requests == [
        ["+first", "-"],
        ["+first", "-", "+second", "-first"],
        ["+first", "-", "+second", "-first"],
    ]
    update = next(message for message in agent.state.messages if message.role == "system" and message.tools_removed)
    assert update == SystemMessage(
        "",
        tools_added=[Tool("second", "second tool", Type.Object({}))],
        tools_removed=[ToolReference("first")],
        timestamp=update.timestamp,
    )
    assert isinstance(update.timestamp, int)
    initial = agent.state.messages[0]
    assert initial.role == "system"
    # Declarations are plain tools: no executable fields.
    assert type(initial.tools_added[0]) is Tool
    assert not hasattr(initial.tools_added[0], "execute")


async def test_merges_tool_changes_into_a_pending_system_message() -> None:
    tool = create_tool("echo", "Echo input")
    system_counts: list[int] = []

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        system_counts.append(len([message for message in context.messages if message.role == "system"]))
        return respond(create_assistant_message("done"))

    agent = Agent(initial_state=AgentInitialState(system_prompt="You are helpful."), stream_fn=stream_fn)

    agent.state.tools = [tool]
    await agent.prompt(
        [
            SystemMessage("", sections={"skills": "<skills>x</skills>"}, timestamp=1),
            UserMessage("hi", timestamp=2),
        ]
    )

    assert system_counts == [2]
    assert agent.state.messages[1] == SystemMessage(
        "",
        sections={"skills": "<skills>x</skills>"},
        tools_added=[Tool("echo", "Echo input", Type.Object({}))],
        timestamp=1,
    )


async def test_rewrites_pending_tool_declarations_to_match_the_executable_set() -> None:
    agent = Agent(
        initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[create_tool("first")]),
        stream_fn=done_stream_fn(),
    )

    # The pending message claims to add `second` and remove `first`, but the executable
    # set still has `first` and lacks `second`: the executable set wins.
    await agent.prompt(
        [
            SystemMessage(
                "",
                sections={"note": "<note>x</note>"},
                tools_added=[to_tool_declaration(create_tool("second"))],
                tools_removed=[ToolReference("first")],
                timestamp=1,
            ),
            UserMessage("hi", timestamp=2),
        ]
    )

    assert agent.state.messages[1] == SystemMessage("", sections={"note": "<note>x</note>"}, timestamp=1)
    current = get_current_system_message(agent.state.messages)
    assert current is not None
    assert [tool.name for tool in current.tools_added or []] == ["first"]


def test_restores_the_transcript_baseline_when_reset() -> None:
    agent = Agent(
        initial_state=AgentInitialState(
            system_prompt="You are helpful.",
            tools=[create_tool("echo", "Echo input")],
            messages=[UserMessage("old", timestamp=1)],
        ),
        stream_fn=unused_stream_function,
    )

    agent.reset()

    assert len(agent.state.messages) == 1
    initial = agent.state.messages[0]
    assert initial.role == "system"
    assert initial.content == "You are helpful."
    assert [value.name for value in initial.tools_added] == ["echo"]


def test_should_subscribe_to_events() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    event_count = 0

    def listener(_event: AgentEvent, _signal: AbortSignal) -> None:
        nonlocal event_count
        event_count += 1

    unsubscribe = agent.subscribe(listener)

    # No initial event on subscribe.
    assert event_count == 0

    # State mutators don't emit events.
    agent.state.thinking_level = "low"
    assert event_count == 0
    assert agent.state.thinking_level == "low"

    # Unsubscribe works.
    unsubscribe()
    agent.state.thinking_level = "high"
    assert event_count == 0


async def test_emits_full_lifecycle_events_for_thrown_run_failures() -> None:
    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        raise RuntimeError("provider exploded")

    agent = Agent(stream_fn=stream_fn)
    events: list[str] = []
    agent.subscribe(lambda event, _signal: events.append(event.type))

    await agent.prompt("hello")

    assert events == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert last_message.stop_reason == "error"
    assert last_message.error_message == "provider exploded"
    assert agent.state.error_message == "provider exploded"


async def test_should_await_async_subscribers_before_prompt_resolves() -> None:
    barrier = asyncio.Event()
    agent = Agent(stream_fn=done_stream_fn("ok"))

    listener_finished = False

    async def listener(event: AgentEvent, _signal: AbortSignal) -> None:
        nonlocal listener_finished
        if event.type == "agent_end":
            await barrier.wait()
            listener_finished = True

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("hello"))

    await asyncio.sleep(0.01)
    assert prompt_task.done() is False
    assert listener_finished is False
    assert agent.state.is_streaming is True

    barrier.set()
    await prompt_task

    assert listener_finished is True
    assert agent.state.is_streaming is False


async def test_wait_for_idle_should_wait_for_async_subscribers() -> None:
    barrier = asyncio.Event()
    agent = Agent(stream_fn=done_stream_fn("ok"))

    async def listener(event: AgentEvent, _signal: AbortSignal) -> None:
        if event.type == "message_end" and event.message.role == "assistant":
            await barrier.wait()

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("hello"))
    # Unlike a JS promise, the coroutine only starts when the loop runs it.
    await asyncio.sleep(0)
    idle_task = asyncio.create_task(agent.wait_for_idle())

    await asyncio.sleep(0.01)
    assert idle_task.done() is False
    assert agent.state.is_streaming is True

    barrier.set()
    await asyncio.gather(prompt_task, idle_task)

    assert idle_task.done() is True
    assert agent.state.is_streaming is False


async def test_should_pass_the_active_abort_signal_to_subscribers() -> None:
    received: list[AbortSignal] = []
    agent = Agent(stream_fn=abortable_stream_fn())

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "agent_start":
            received.append(signal)

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("hello"))
    await asyncio.sleep(0.01)

    assert len(received) == 1
    assert received[0].aborted is False

    agent.abort()
    await prompt_task

    assert received[0].aborted is True


async def test_should_ignore_tool_updates_after_the_tool_execution_settles() -> None:
    delayed_update: list[Any] = []
    events: list[AgentEvent] = []

    async def execute(_tool_call_id: str, _params: Any, _signal: Any = None, on_update: Any = None) -> AgentToolResult:
        delayed_update.append(on_update)
        on_update(AgentToolResult([TextContent("running")], {"status": "running"}))
        return AgentToolResult([TextContent("ok")], {"status": "done"}, terminate=True)

    tool = AgentTool(
        "delayed_tool", "Captures progress callbacks", Type.Object({}), label="Delayed Tool", execute=execute
    )

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(create_assistant_tool_use_message([ToolCall("call-1", "delayed_tool", {})]))

    agent = Agent(initial_state=AgentInitialState(tools=[tool]), stream_fn=stream_fn)
    agent.subscribe(lambda event, _signal: events.append(event))

    loop = asyncio.get_running_loop()
    unhandled: list[Any] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        await agent.prompt("run tool")
        event_count_after_prompt = len(events)

        result = delayed_update[0](AgentToolResult([TextContent("late")], {"status": "late"}))
        assert result is None
        await asyncio.sleep(0)

        assert len([event for event in events if event.type == "tool_execution_update"]) == 1
        assert len(events) == event_count_after_prompt
        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_handler)


async def test_should_ignore_a_settled_parallel_tool_update_while_another_tool_is_still_running() -> None:
    slow_started = asyncio.Event()
    settled_tool_ended = asyncio.Event()
    release_slow = asyncio.Event()
    settled_tool_update: list[Any] = []
    events: list[AgentEvent] = []

    async def settled_execute(
        _tool_call_id: str, _params: Any, _signal: Any = None, on_update: Any = None
    ) -> AgentToolResult:
        settled_tool_update.append(on_update)
        return AgentToolResult([TextContent("done")], {"status": "done"}, terminate=True)

    async def slow_execute(*_args: Any) -> AgentToolResult:
        slow_started.set()
        await release_slow.wait()
        return AgentToolResult([TextContent("done")], {"status": "done"}, terminate=True)

    settled_tool = AgentTool(
        "settled_tool", "Captures progress callbacks", Type.Object({}), label="Settled Tool", execute=settled_execute
    )
    slow_tool = AgentTool(
        "slow_tool", "Keeps the agent run active", Type.Object({}), label="Slow Tool", execute=slow_execute
    )

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        return respond(
            create_assistant_tool_use_message(
                [ToolCall("call-1", "settled_tool", {}), ToolCall("call-2", "slow_tool", {})]
            )
        )

    agent = Agent(initial_state=AgentInitialState(tools=[settled_tool, slow_tool]), stream_fn=stream_fn)

    def listener(event: AgentEvent, _signal: AbortSignal) -> None:
        events.append(event)
        if event.type == "tool_execution_end" and event.tool_call_id == "call-1":
            settled_tool_ended.set()

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("run tools"))
    await asyncio.gather(slow_started.wait(), settled_tool_ended.wait())
    event_count_before_late_update = len(events)

    settled_tool_update[0](AgentToolResult([TextContent("late")], {"status": "late"}))
    await asyncio.sleep(0)
    assert len(events) == event_count_before_late_update

    release_slow.set()
    await prompt_task
    assert len([event for event in events if event.type == "tool_execution_update"]) == 0


def test_should_update_state_with_mutators() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    # set model
    new_model = model_named("google", "gemini-2.5-flash")
    agent.state.model = new_model
    assert agent.state.model is new_model

    # set thinking level
    agent.state.thinking_level = "high"
    assert agent.state.thinking_level == "high"

    # set tools (copied)
    tools = [create_tool("test")]
    agent.state.tools = tools
    assert agent.state.tools == tools
    assert agent.state.tools is not tools

    # replace messages (copied)
    messages = [UserMessage("Hello", timestamp=now_ms())]
    agent.state.messages = messages
    assert agent.state.messages == messages
    assert agent.state.messages is not messages

    # append a message
    new_message = AssistantMessage([TextContent("Hi")], api="openai-responses", provider="openai", model="mock")
    agent.state.messages.append(new_message)
    assert len(agent.state.messages) == 2
    assert agent.state.messages[1] is new_message

    # clear messages
    agent.state.messages = []
    assert agent.state.messages == []


def test_should_support_steering_message_queue() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    message = UserMessage("Steering message", timestamp=now_ms())
    agent.steer(message)

    # The message is queued but not yet in state.messages.
    assert message not in agent.state.messages
    assert agent.has_queued_messages() is True


def test_should_support_follow_up_message_queue() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    message = UserMessage("Follow-up message", timestamp=now_ms())
    agent.follow_up(message)

    # The message is queued but not yet in state.messages.
    assert message not in agent.state.messages
    assert agent.has_queued_messages() is True


def test_should_handle_abort_controller() -> None:
    agent = Agent(stream_fn=unused_stream_function)

    # Does not raise even if nothing is running.
    agent.abort()


async def test_should_reject_reset_while_processing_without_corrupting_the_transcript() -> None:
    stream_started = asyncio.Event()
    release_response = asyncio.Event()

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def produce() -> None:
            stream.push(StartEvent(create_assistant_message("")))
            stream_started.set()
            await release_response.wait()
            stream.push(respond_done("Done"))

        stream.start(produce())
        return stream

    agent = Agent(stream_fn=stream_fn)

    prompt_task = asyncio.create_task(agent.prompt("Hello"))
    await stream_started.wait()

    try:
        assert agent.state.is_streaming is True
        assert [message.role for message in agent.state.messages] == ["user"]
        with pytest.raises(
            RuntimeError, match=re.escape("Agent is already processing. Wait for completion before resetting.")
        ):
            agent.reset()
        assert agent.state.is_streaming is True
        assert [message.role for message in agent.state.messages] == ["user"]
    finally:
        release_response.set()
        await prompt_task

    assert agent.state.is_streaming is False
    assert [message.role for message in agent.state.messages] == ["user", "assistant"]


async def test_should_throw_when_prompt_called_while_streaming() -> None:
    agent = Agent(stream_fn=abortable_stream_fn())

    # Start the first prompt (not awaited: it blocks until abort).
    first_prompt = asyncio.create_task(agent.prompt("First message"))

    # Wait a tick for is_streaming to be set.
    await asyncio.sleep(0.01)
    assert agent.state.is_streaming is True

    # The second prompt is rejected.
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, "
            "or wait for completion."
        ),
    ):
        await agent.prompt("Second message")

    # Cleanup: abort to stop the stream.
    agent.abort()
    await first_prompt


async def test_should_throw_when_continue_called_while_streaming() -> None:
    agent = Agent(stream_fn=abortable_stream_fn())

    first_prompt = asyncio.create_task(agent.prompt("First message"))
    await asyncio.sleep(0.01)
    assert agent.state.is_streaming is True

    with pytest.raises(
        RuntimeError, match=re.escape("Agent is already processing. Wait for completion before continuing.")
    ):
        await agent.continue_()

    agent.abort()
    await first_prompt


async def test_continue_should_process_queued_follow_up_messages_after_an_assistant_turn() -> None:
    agent = Agent(stream_fn=done_stream_fn("Processed"))

    agent.state.messages = [
        UserMessage([TextContent("Initial")], timestamp=now_ms() - 10),
        create_assistant_message("Initial response"),
    ]

    agent.follow_up(UserMessage([TextContent("Queued follow-up")], timestamp=now_ms()))

    assert await agent.continue_() is None

    assert any(is_user_with_text(message, "Queued follow-up") for message in agent.state.messages)
    assert agent.state.messages[-1].role == "assistant"


@pytest.mark.parametrize(("mode", "expected_requests"), [("one-at-a-time", 2), ("all", 1)])
async def test_continue_keeps_steering_semantics_for_assistant_tail_fallback(mode: Any, expected_requests: int) -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        requests.append(user_texts(context.messages))
        return respond(create_assistant_message("Processed"))

    agent = Agent(steering_mode=mode, stream_fn=stream_fn)
    agent.state.messages = [create_user_message("Initial"), create_assistant_message("Initial response")]
    agent.steer(create_user_message("Steering 1"))
    agent.steer(create_user_message("Steering 2"))

    assert await agent.continue_() is None

    assert len(requests) == expected_requests
    assert "Steering 1" in requests[0]
    if mode == "one-at-a-time":
        assert "Steering 2" not in requests[0]
        assert "Steering 2" in requests[1]
    else:
        assert "Steering 2" in requests[0]


def _noop_tool_agent(**options: Any) -> tuple[Agent, list[int]]:
    request_count = [0]

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        request_count[0] += 1
        if request_count[0] == 1:
            return respond(create_assistant_tool_use_message([ToolCall("tool-1", "noop", {})]))
        return respond(create_assistant_message("done"))

    async def execute(*_args: Any) -> AgentToolResult:
        return AgentToolResult([TextContent("ok")], {})

    tool = AgentTool("noop", "Noop tool", Type.Object({}), label="Noop", execute=execute)
    return Agent(initial_state=AgentInitialState(tools=[tool]), stream_fn=stream_fn, **options), request_count


async def test_keeps_legacy_prepare_next_turn_signal_callback_behavior() -> None:
    saw_abort_signal = False

    async def prepare_next_turn(signal: Any) -> None:
        nonlocal saw_abort_signal
        saw_abort_signal = isinstance(signal, AbortSignal)
        return None

    agent, request_count = _noop_tool_agent(prepare_next_turn=prepare_next_turn)
    await agent.prompt("start")

    assert request_count[0] == 2
    assert saw_abort_signal is True


async def test_prepare_next_turn_with_context_receives_the_completed_turn() -> None:
    seen: list[tuple[list[str], bool]] = []

    def prepare_next_turn_with_context(turn: Any, signal: Any) -> None:
        seen.append(([message.role for message in turn.context.messages], isinstance(signal, AbortSignal)))
        return None

    agent, request_count = _noop_tool_agent(prepare_next_turn_with_context=prepare_next_turn_with_context)
    await agent.prompt("start")

    assert request_count[0] == 2
    assert seen == [(["system", "user", "assistant", "toolResult"], True)]


async def test_forwards_finish_turn_through_agent_options_with_the_active_abort_signal() -> None:
    saw_abort_signal = False
    callback_context_roles: list[str] = []

    def finish_turn(turn: Any, signal: Any) -> AgentTurnDecision:
        nonlocal saw_abort_signal, callback_context_roles
        saw_abort_signal = isinstance(signal, AbortSignal)
        callback_context_roles = [message.role for message in turn.context.messages]
        return AgentTurnDecision("end")

    agent, request_count = _noop_tool_agent(finish_turn=finish_turn)
    await agent.prompt("start")

    assert request_count[0] == 1
    assert saw_abort_signal is True
    assert callback_context_roles == ["system", "user", "assistant", "toolResult"]


@pytest.mark.parametrize(
    "messages",
    [[], [SystemMessage("system only", timestamp=1)]],
    ids=["empty", "system-only"],
)
async def test_rejects_a_queued_continuation_without_draining_queues(messages: list[Any]) -> None:
    agent = Agent(initial_state=AgentInitialState(messages=messages), stream_fn=unused_stream_function)
    steering = create_user_message("steering")
    follow_up = create_user_message("follow-up")
    agent.steer(steering)
    agent.follow_up(follow_up)

    with pytest.raises(RuntimeError, match="No messages to continue from"):
        await agent.continue_()
    assert agent.peek_queued_messages() == [steering]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


async def test_continue_rejects_an_assistant_tail_without_queued_messages() -> None:
    agent = Agent(stream_fn=unused_stream_function)
    agent.state.messages = [create_user_message("hi"), create_assistant_message("hello")]

    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        await agent.continue_()


@pytest.mark.parametrize(
    "messages",
    [
        [create_user_message("existing user")],
        [
            create_user_message("existing user"),
            create_assistant_tool_use_message([ToolCall("call-1", "noop", {})]),
            ToolResultMessage("call-1", "noop", [TextContent("done")], False, timestamp=1),
        ],
    ],
    ids=["user", "toolResult"],
)
async def test_defers_follow_up_input_on_the_first_continuation_request(messages: list[Any]) -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        requests.append(user_texts(context.messages))
        return respond(create_assistant_message("done"))

    agent = Agent(initial_state=AgentInitialState(messages=messages), stream_fn=stream_fn)
    agent.follow_up(create_user_message("follow-up"))

    await agent.continue_()

    assert len(requests) == 2
    assert "follow-up" not in requests[0]
    assert "follow-up" in requests[1]


@pytest.mark.parametrize(("mode", "expected_requests"), [("one-at-a-time", 2), ("all", 1)])
async def test_polls_steering_at_continuation_startup(mode: Any, expected_requests: int) -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        requests.append(user_texts(context.messages))
        return respond(create_assistant_message("done"))

    agent = Agent(
        initial_state=AgentInitialState(messages=[create_user_message("existing")]),
        steering_mode=mode,
        stream_fn=stream_fn,
    )
    agent.steer(create_user_message("first"))
    agent.steer(create_user_message("second"))

    await agent.continue_()

    assert len(requests) == expected_requests
    assert "first" in requests[0]
    if mode == "one-at-a-time":
        assert "second" not in requests[0]
        assert "second" in requests[1]
    else:
        assert "second" in requests[0]


async def test_keeps_steering_ahead_of_follow_up_from_a_non_assistant_continuation_tail() -> None:
    requests: list[list[str]] = []

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        requests.append(user_texts(context.messages))
        return respond(create_assistant_message("done"))

    agent = Agent(initial_state=AgentInitialState(messages=[create_user_message("existing")]), stream_fn=stream_fn)
    agent.steer(create_user_message("steering"))
    agent.follow_up(create_user_message("follow-up"))

    await agent.continue_()

    assert len(requests) == 2
    assert "steering" in requests[0]
    assert "follow-up" not in requests[0]
    assert "follow-up" in requests[1]


@pytest.mark.parametrize("stop_reason", ["error", "aborted"])
async def test_keeps_queues_on_a_failed_response_even_when_finish_turn_requests_continuation(stop_reason: str) -> None:
    queued_during_response = create_user_message("steering")
    follow_up = create_user_message("follow-up")

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        message = create_assistant_message(stop_reason, stop_reason)
        message.error_message = stop_reason
        return respond(message)

    agent = Agent(finish_turn=lambda _turn, _signal: AgentTurnDecision("continue"), stream_fn=stream_fn)
    agent.follow_up(follow_up)

    def listener(event: AgentEvent, _signal: AbortSignal) -> None:
        if event.type == "message_end" and event.message.role == "assistant":
            agent.steer(queued_during_response)

    agent.subscribe(listener)

    await agent.prompt("start")

    assert agent.peek_queued_messages() == [queued_during_response]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]
    assert agent.state.error_message == stop_reason


async def test_keeps_queues_when_finish_turn_ends_the_run() -> None:
    queued_during_response = create_user_message("steering")
    follow_up = create_user_message("follow-up")
    agent = Agent(finish_turn=lambda _turn, _signal: AgentTurnDecision("end"), stream_fn=done_stream_fn())
    agent.follow_up(follow_up)

    def listener(event: AgentEvent, _signal: AbortSignal) -> None:
        if event.type == "message_end" and event.message.role == "assistant":
            agent.steer(queued_during_response)

    agent.subscribe(listener)

    await agent.prompt("start")

    assert agent.peek_queued_messages() == [queued_during_response]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


def test_previews_the_next_selected_queued_messages_without_consuming_them() -> None:
    agent = Agent(
        steering_mode="one-at-a-time",
        follow_up_mode="all",
        stream_fn=lambda *_args: AssistantMessageEventStream(),
    )
    first = create_user_message("first steering")
    second = create_user_message("second steering")
    follow_up = create_user_message("follow-up")
    agent.steer(first)
    agent.steer(second)
    agent.follow_up(follow_up)

    assert agent.peek_queued_messages() == [first]
    assert agent.peek_queued_messages() == [first]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


async def test_forwards_provider_stream_event_observers_through_agent_options() -> None:
    provider_events: list[Any] = []

    def stream_fn(model: Any, _context: Any, options: Any) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def produce() -> None:
            observer = (options or {}).get("on_provider_stream_event")
            if observer is not None:
                result = observer({"request_cost": 0.01}, model)
                if asyncio.iscoroutine(result):
                    await result
            stream.push(respond_done("ok"))

        stream.start(produce())
        return stream

    agent = Agent(
        on_provider_stream_event=lambda data, _model: provider_events.append(data),
        stream_fn=stream_fn,
    )

    await agent.prompt("hello")

    assert provider_events == [{"request_cost": 0.01}]


async def test_forwards_session_id_to_stream_function_options() -> None:
    received_session_ids: list[Any] = []

    def stream_fn(_model: Any, _context: Any, options: Any) -> AssistantMessageEventStream:
        received_session_ids.append((options or {}).get("session_id"))
        return respond(create_assistant_message("ok"))

    agent = Agent(session_id="session-abc", stream_fn=stream_fn)

    await agent.prompt("hello")
    assert received_session_ids[-1] == "session-abc"

    # Setter
    agent.session_id = "session-def"
    assert agent.session_id == "session-def"

    await agent.prompt("hello again")
    assert received_session_ids[-1] == "session-def"


# ---------------------------------------------------------------------------
# Python-specific behavior
# ---------------------------------------------------------------------------


async def test_forwards_reasoning_api_key_and_stream_options() -> None:
    received: list[dict[str, Any]] = []

    def stream_fn(_model: Any, _context: Any, options: Any) -> AssistantMessageEventStream:
        received.append(dict(options or {}))
        return respond(create_assistant_message("ok"))

    agent = Agent(
        initial_state=AgentInitialState(thinking_level="medium"),
        get_api_key=lambda provider: f"key-for-{provider}",
        stream_options={"temperature": 0.2},
        transport="sse",
        stream_fn=stream_fn,
    )

    await agent.prompt("hello")

    options = received[0]
    assert options["reasoning"] == "medium"
    assert options["api_key"] == "key-for-unknown"
    assert options["temperature"] == 0.2
    assert options["transport"] == "sse"
    assert isinstance(options["signal"], AbortSignal)
    assert agent.state.messages[-1].thinking_level == "medium"


async def test_off_thinking_level_omits_reasoning() -> None:
    received: list[dict[str, Any]] = []

    def stream_fn(_model: Any, _context: Any, options: Any) -> AssistantMessageEventStream:
        received.append(dict(options or {}))
        return respond(create_assistant_message("ok"))

    agent = Agent(stream_fn=stream_fn)
    await agent.prompt("hello")

    assert "reasoning" not in received[0]
    assert agent.state.messages[-1].thinking_level == "off"


async def test_prompt_accepts_images_and_message_lists() -> None:
    requests: list[list[Any]] = []

    def stream_fn(_model: Any, context: Any, _options: Any) -> AssistantMessageEventStream:
        requests.append(list(context.messages))
        return respond(create_assistant_message("ok"))

    agent = Agent(stream_fn=stream_fn)
    image = ImageContent("aGVsbG8=", "image/png")
    await agent.prompt("look", [image])

    prompt = agent.state.messages[0]
    assert prompt.role == "user"
    assert prompt.content == [TextContent("look"), image]

    await agent.prompt([create_user_message("a"), create_user_message("b")])
    assert user_texts(requests[-1]) == ["a", "b"]


async def test_state_tracks_streaming_message_and_pending_tool_calls() -> None:
    observed: list[tuple[str, Any, frozenset[str]]] = []

    async def execute(*_args: Any) -> AgentToolResult:
        return AgentToolResult([TextContent("ok")], {})

    tool = AgentTool("noop", "Noop tool", Type.Object({}), execute=execute)
    agent = Agent(initial_state=AgentInitialState(tools=[tool]), stream_fn=_scripted_tool_then_text())

    def listener(event: AgentEvent, _signal: AbortSignal) -> None:
        observed.append((event.type, agent.state.streaming_message, agent.state.pending_tool_calls))

    agent.subscribe(listener)
    await agent.prompt("go")

    by_type = {kind: (streaming, pending) for kind, streaming, pending in observed}
    assert by_type["tool_execution_start"][1] == frozenset({"tool-1"})
    assert by_type["tool_execution_end"][1] == frozenset()
    assert by_type["agent_end"][0] is None
    message_start = next(entry for entry in observed if entry[0] == "message_start" and entry[1].role == "assistant")
    assert message_start[1].role == "assistant"
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == frozenset()
    assert tool.label == "noop"


def _scripted_tool_then_text() -> Any:
    calls = [0]

    def stream_fn(*_args: Any) -> AssistantMessageEventStream:
        calls[0] += 1
        if calls[0] == 1:
            return respond(create_assistant_tool_use_message([ToolCall("tool-1", "noop", {})]))
        return respond(create_assistant_message("done"))

    return stream_fn


async def test_abort_produces_an_aborted_assistant_message_and_error_state() -> None:
    signals: list[Any] = []
    agent = Agent(stream_fn=abortable_stream_fn(signals))

    prompt_task = asyncio.create_task(agent.prompt("hello"))
    await asyncio.sleep(0.01)
    assert agent.is_running is True
    assert agent.signal is signals[0]
    agent.abort()
    await prompt_task

    assert agent.is_running is False
    assert agent.signal is None
    last = agent.state.messages[-1]
    assert last.role == "assistant"
    assert last.stop_reason == "aborted"
    assert agent.state.error_message == "Request was aborted"


async def test_sync_and_async_listeners_are_awaited_in_subscription_order() -> None:
    order: list[str] = []
    agent = Agent(stream_fn=done_stream_fn())

    async def slow(event: AgentEvent, _signal: AbortSignal) -> None:
        if event.type == "agent_end":
            await asyncio.sleep(0.01)
            order.append("slow")

    def fast(event: AgentEvent, _signal: AbortSignal) -> None:
        if event.type == "agent_end":
            order.append("fast")

    agent.subscribe(slow)
    agent.subscribe(fast)
    await agent.prompt("hi")

    assert order == ["slow", "fast"]
