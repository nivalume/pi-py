"""Stateful wrapper around the low-level agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pi_ai import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    UserMessage,
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    now_ms,
    to_tool_declaration,
)
from pi_ai.utils.callbacks import maybe_await
from pi_telemetry import TelemetryContext

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentTool,
    AgentTurnContext,
    BeforeToolCallContext,
    BeforeToolCallResult,
    FinishTurn,
    MessageEndEvent,
    MessageStartEvent,
    PrepareRequest,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
    TurnEndEvent,
)

__all__ = ["Agent", "AgentState", "AgentInitialState", "AgentListener", "default_convert_to_llm"]

_LLM_ROLES = frozenset({"system", "user", "assistant", "toolResult"})


def default_convert_to_llm(messages: list[Any]) -> list[Message]:
    """Keep only messages the LLM understands (``system``, ``user``, ``assistant``, ``toolResult``)."""
    return [message for message in messages if getattr(message, "role", None) in _LLM_ROLES]


_DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="unknown",
    provider="unknown",
    base_url="",
    reasoning=False,
    input=[],
    context_window=0,
    max_tokens=0,
)

AgentListener = Callable[[AgentEvent, AbortSignal], "Awaitable[None] | None"]


@dataclass(slots=True)
class AgentInitialState:
    """Initial state for :class:`Agent`.

    ``system_prompt`` and ``tools`` become the leading system message unless
    ``messages`` already starts with one.
    """

    system_prompt: str | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)


class AgentState:
    """Public agent state.

    Assigning ``tools`` or ``messages`` copies the top-level list. ``system_prompt`` is
    read-only and replays the transcript's system messages; to change the prompt,
    append a system message with ``content`` or ``sections``.
    """

    __slots__ = (
        "model",
        "thinking_level",
        "_tools",
        "_messages",
        "is_streaming",
        "streaming_message",
        "pending_tool_calls",
        "error_message",
    )

    def __init__(self, initial: AgentInitialState | Mapping[str, Any] | None = None) -> None:
        if isinstance(initial, Mapping):
            initial = AgentInitialState(**initial)
        initial = initial or AgentInitialState()
        tools = list(initial.tools or [])
        messages = _coerce_messages(list(initial.messages or []))
        initial_message = create_initial_system_message(
            initial.system_prompt, [to_tool_declaration(tool) for tool in tools]
        )
        if initial_message is not None and (not messages or getattr(messages[0], "role", None) != "system"):
            messages.insert(0, initial_message)
        self.model: Model = initial.model or _DEFAULT_MODEL
        self.thinking_level: ThinkingLevel = initial.thinking_level or "off"
        self._tools: list[AgentTool] = tools
        self._messages: list[Any] = messages
        #: True while a prompt or continuation runs, until awaited ``agent_end`` listeners settle.
        self.is_streaming = False
        #: Partial assistant message for the current streamed response.
        self.streaming_message: Any = None
        #: Tool call ids currently executing.
        self.pending_tool_calls: frozenset[str] = frozenset()
        #: Error message from the most recent failed or aborted assistant turn.
        self.error_message: str | None = None

    @property
    def system_prompt(self) -> str:
        return get_current_system_prompt(self._messages)

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, tools: Sequence[AgentTool]) -> None:
        self._tools = list(tools)

    @property
    def messages(self) -> list[Any]:
        return self._messages

    @messages.setter
    def messages(self, messages: Sequence[Any]) -> None:
        self._messages = _coerce_messages(list(messages))

    def __repr__(self) -> str:
        return (
            f"AgentState(model={self.model.provider}/{self.model.id}, messages={len(self._messages)}, "
            f"tools={[tool.name for tool in self._tools]}, is_streaming={self.is_streaming})"
        )


def _coerce_messages(messages: list[Any]) -> list[Any]:
    if any(isinstance(message, Mapping) for message in messages):
        from pi_ai.serialization import message_from_dict

        return [message_from_dict(message) if isinstance(message, Mapping) else message for message in messages]
    return messages


class _PendingMessageQueue:
    __slots__ = ("mode", "_messages")

    def __init__(self, mode: QueueMode) -> None:
        self.mode: QueueMode = mode
        self._messages: list[Any] = []

    def enqueue(self, message: Any) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def peek(self) -> list[Any]:
        if self.mode == "all":
            return list(self._messages)
        return self._messages[:1]

    def drain(self) -> list[Any]:
        drained = self.peek()
        self._messages = self._messages[len(drained) :]
        return drained

    def clear(self) -> None:
        self._messages = []


@dataclass(slots=True)
class _ActiveRun:
    done: asyncio.Event
    abort_controller: AbortController


class Agent:
    """Stateful agent: owns the transcript, emits lifecycle events, executes tools,
    and queues steering and follow-up messages.

    ::

        models = builtin_models()
        agent = Agent(
            initial_state=AgentInitialState(system_prompt="You are helpful.", model=models.get_model("anthropic", "claude-sonnet-4-5")),
            stream_fn=models.stream_simple,
        )
        agent.subscribe(lambda event, signal: print(event.type))
        await agent.prompt("Hello!")
    """

    def __init__(
        self,
        *,
        initial_state: AgentInitialState | Mapping[str, Any] | None = None,
        stream_fn: StreamFn | Any = None,
        convert_to_llm: Callable[[list[Any]], list[Message] | Awaitable[list[Message]]] | None = None,
        transform_context: Callable[[list[Any], AbortSignal | None], Awaitable[list[Any]]] | None = None,
        get_api_key: Callable[[str], str | Awaitable[str | None] | None] | None = None,
        on_payload: Callable[[Any, Model], Any] | None = None,
        on_response: Callable[[Any, Model], Any] | None = None,
        on_provider_stream_event: Callable[[Any, Model], Any] | None = None,
        before_tool_call: Callable[[BeforeToolCallContext, AbortSignal | None], Awaitable[BeforeToolCallResult | None]]
        | None = None,
        after_tool_call: Callable[[AfterToolCallContext, AbortSignal | None], Awaitable[AfterToolCallResult | None]]
        | None = None,
        finish_turn: FinishTurn | None = None,
        prepare_request: PrepareRequest | None = None,
        prepare_next_turn: Callable[[AbortSignal | None], Any] | None = None,
        prepare_next_turn_with_context: Callable[[AgentTurnContext, AbortSignal | None], Any] | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        session_id: str | None = None,
        thinking_budgets: Mapping[str, int] | None = None,
        transport: str = "auto",
        max_retry_delay_ms: float | None = None,
        tool_execution: ToolExecutionMode = "parallel",
        stream_options: Mapping[str, Any] | None = None,
        telemetry_context: TelemetryContext | None = None,
    ) -> None:
        self._state = AgentState(initial_state)
        self._listeners: list[AgentListener] = []
        self._steering_queue = _PendingMessageQueue(steering_mode)
        self._follow_up_queue = _PendingMessageQueue(follow_up_mode)
        self._active_run: _ActiveRun | None = None
        if stream_fn is not None and not callable(stream_fn) and hasattr(stream_fn, "stream_simple"):
            stream_fn = stream_fn.stream_simple  # accept a Models collection directly
        self.stream_function: StreamFn = cast(StreamFn, stream_fn) if stream_fn is not None else get_default_stream_fn()
        self.convert_to_llm = convert_to_llm or default_convert_to_llm
        self.transform_context = transform_context
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.on_provider_stream_event = on_provider_stream_event
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.finish_turn = finish_turn
        self.prepare_request = prepare_request
        self.prepare_next_turn = prepare_next_turn
        self.prepare_next_turn_with_context = prepare_next_turn_with_context
        #: Session identifier forwarded to providers for cache-aware backends.
        self.session_id = session_id
        #: Optional per-level thinking token budgets.
        self.thinking_budgets = dict(thinking_budgets) if thinking_budgets else None
        #: Preferred transport forwarded to the stream function.
        self.transport = transport
        #: Optional cap for provider-requested retry delays.
        self.max_retry_delay_ms = max_retry_delay_ms
        #: Tool execution strategy for assistant messages with multiple tool calls.
        self.tool_execution: ToolExecutionMode = tool_execution
        #: Extra stream options (temperature, max_tokens, cache_retention, headers...).
        self.stream_options: dict[str, Any] = dict(stream_options or {})
        #: Parent context of each run's ``pi.agent.run`` span; ``None`` records nothing (see :mod:`pi_agent.telemetry`).
        self.telemetry_context = telemetry_context

    # -- events and state -------------------------------------------------------

    def subscribe(self, listener: AgentListener) -> Callable[[], None]:
        """Subscribe to lifecycle events. Returns an unsubscribe function.

        Listeners (sync or async) are awaited in subscription order and count toward
        the current run's settlement. They receive the run's abort signal.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def steer(self, message: Any) -> None:
        """Queue a message to inject after the current assistant turn finishes."""
        self._steering_queue.enqueue(_coerce_messages([message])[0])

    def follow_up(self, message: Any) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(_coerce_messages([message])[0])

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    def peek_queued_messages(self) -> list[Any]:
        """Preview the messages selected for the next turn without consuming them."""
        steering = self._steering_queue.peek()
        return steering if steering else self._follow_up_queue.peek()

    @property
    def signal(self) -> AbortSignal | None:
        """Abort signal of the current run, if any."""
        return self._active_run.abort_controller.signal if self._active_run else None

    @property
    def is_running(self) -> bool:
        return self._active_run is not None

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        if self._active_run is not None:
            self._active_run.abort_controller.abort()

    async def wait_for_idle(self) -> None:
        """Wait until the current run and all awaited listeners (incl. ``agent_end``) finished."""
        run = self._active_run
        if run is not None:
            await run.done.wait()

    def reset(self) -> None:
        """Clear conversation state and queues, keeping the replayed prompt/tool baseline."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")
        baseline = get_current_system_message(self._state.messages)
        self._state.messages = [baseline] if baseline else []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = frozenset()
        self._state.error_message = None
        self.clear_all_queues()

    # -- running ----------------------------------------------------------------

    async def prompt(self, input: str | Any | Sequence[Any], images: Sequence[ImageContent] | None = None) -> None:
        """Start a run from text (with optional images), a message, or a list of messages."""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )
        await self._run_prompt_messages(self._normalize_prompt_input(input, images))

    async def continue_(self) -> None:
        """Continue from the current transcript (``continue`` is a Python keyword).

        A non-assistant tail is retried. An assistant tail falls back to one queued steering
        batch, then one queued follow-up batch.
        """
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")
        messages = self._state.messages
        if not messages or all(getattr(message, "role", None) == "system" for message in messages):
            raise RuntimeError("No messages to continue from")
        if getattr(messages[-1], "role", None) == "assistant":
            steering = self._steering_queue.drain()
            if steering:
                await self._run_prompt_messages(steering, skip_initial_steering_poll=True)
                return
            follow_ups = self._follow_up_queue.drain()
            if follow_ups:
                await self._run_prompt_messages(follow_ups)
                return
            raise RuntimeError("Cannot continue from message role: assistant")
        await self._run_continuation()

    #: Alias of :meth:`continue_`.
    resume = continue_

    async def events(
        self, input: str | Any | Sequence[Any], images: Sequence[ImageContent] | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Run a prompt and yield its events as an async iterator.

        Convenience over :meth:`subscribe` + :meth:`prompt`; state reduction happens
        before each event is yielded, and the run waits for the consumer (the same
        barrier semantics as an awaited listener).
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[AgentEvent, asyncio.Future[None]] | None] = asyncio.Queue()
        closed = False

        async def listener(event: AgentEvent, _signal: AbortSignal) -> None:
            if closed:
                return
            consumed: asyncio.Future[None] = loop.create_future()
            queue.put_nowait((event, consumed))
            await consumed

        unsubscribe = self.subscribe(listener)

        async def run() -> None:
            try:
                await self.prompt(input, images)
            finally:
                queue.put_nowait(None)

        task = asyncio.ensure_future(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, consumed = item
                try:
                    yield event
                finally:
                    if not consumed.done():
                        consumed.set_result(None)
            await task
        finally:
            closed = True
            unsubscribe()
            while not queue.empty():
                pending = queue.get_nowait()
                if pending is not None and not pending[1].done():
                    pending[1].set_result(None)
            if not task.done():
                self.abort()
                await asyncio.gather(task, return_exceptions=True)

    def _normalize_prompt_input(self, input: Any, images: Sequence[ImageContent] | None) -> list[Any]:
        if isinstance(input, (list, tuple)):
            return _coerce_messages(list(input))
        if not isinstance(input, str):
            return _coerce_messages([input])
        content: list[TextContent | ImageContent] = [TextContent(input)]
        if images:
            content.extend(images)
        return [UserMessage(content, timestamp=now_ms())]

    async def _run_prompt_messages(self, messages: list[Any], *, skip_initial_steering_poll: bool = False) -> None:
        async def executor(signal: AbortSignal) -> None:
            await run_agent_loop(
                messages,
                self._context_snapshot(),
                self._loop_config(skip_initial_steering_poll),
                self._process_event,
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        async def executor(signal: AbortSignal) -> None:
            await run_agent_loop_continue(
                self._context_snapshot(), self._loop_config(), self._process_event, signal, self.stream_function
            )

        await self._run_with_lifecycle(executor)

    def _context_snapshot(self) -> AgentContext:
        return AgentContext(messages=list(self._state.messages), tools=list(self._state.tools))

    def _loop_config(self, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        skip = [skip_initial_steering_poll]

        async def get_steering_messages() -> list[Any]:
            if skip[0]:
                skip[0] = False
                return []
            return self._steering_queue.drain()

        async def get_follow_up_messages() -> list[Any]:
            return self._follow_up_queue.drain()

        prepare_next_turn: Any = None
        if self.prepare_next_turn_with_context is not None or self.prepare_next_turn is not None:

            async def prepare(context: AgentTurnContext) -> AgentLoopTurnUpdate | None:
                if self.prepare_next_turn_with_context is not None:
                    return await maybe_await(self.prepare_next_turn_with_context(context, self.signal))
                assert self.prepare_next_turn is not None
                return await maybe_await(self.prepare_next_turn(self.signal))

            prepare_next_turn = prepare

        stream_options: dict[str, Any] = dict(self.stream_options)
        for key, value in (
            ("session_id", self.session_id),
            ("on_payload", self.on_payload),
            ("on_response", self.on_response),
            ("on_provider_stream_event", self.on_provider_stream_event),
            ("transport", self.transport),
            ("thinking_budgets", self.thinking_budgets),
            ("max_retry_delay_ms", self.max_retry_delay_ms),
        ):
            if value is not None:
                stream_options[key] = value

        thinking_level = self._state.thinking_level
        return AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            reasoning=None if thinking_level == "off" else thinking_level,
            stream_options=stream_options,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            finish_turn=self.finish_turn,
            prepare_request=self.prepare_request,
            prepare_next_turn=prepare_next_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            telemetry_context=self.telemetry_context,
        )

    async def _run_with_lifecycle(self, executor: Callable[[AbortSignal], Awaitable[None]]) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")
        controller = AbortController()
        self._active_run = _ActiveRun(asyncio.Event(), controller)
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None
        try:
            await executor(controller.signal)
        except Exception as error:
            await self._handle_run_failure(error, controller.signal.aborted)
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: BaseException, aborted: bool) -> None:
        model = self._state.model
        failure = AssistantMessage(
            [TextContent("")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason="aborted" if aborted else "error",
            error_message=str(error) or type(error).__name__,
            timestamp=now_ms(),
        )
        await self._process_event(MessageStartEvent(failure))
        await self._process_event(MessageEndEvent(failure))
        await self._process_event(TurnEndEvent(failure, []))
        await self._process_event(AgentEndEvent([failure]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = frozenset()
        run = self._active_run
        self._active_run = None
        if run is not None:
            run.done.set()

    async def _process_event(self, event: AgentEvent) -> None:
        """Reduce internal state for a loop event, then await listeners in order."""
        state = self._state
        kind = event.type
        if kind == "message_start" or kind == "message_update":
            state.streaming_message = event.message  # type: ignore[union-attr]
        elif kind == "message_end":
            state.streaming_message = None
            state._messages.append(event.message)  # type: ignore[union-attr]
        elif kind == "tool_execution_start":
            state.pending_tool_calls = state.pending_tool_calls | {event.tool_call_id}  # type: ignore[union-attr]
        elif kind == "tool_execution_end":
            state.pending_tool_calls = state.pending_tool_calls - {event.tool_call_id}  # type: ignore[union-attr]
        elif kind == "turn_end":
            message = event.message  # type: ignore[union-attr]
            if getattr(message, "role", None) == "assistant" and message.error_message:
                state.error_message = message.error_message
        elif kind == "agent_end":
            state.streaming_message = None

        run = self._active_run
        if run is None:
            raise RuntimeError("Agent listener invoked outside active run")
        signal = run.abort_controller.signal
        for listener in tuple(self._listeners):
            result = listener(event, signal)
            if result is not None:
                await result
