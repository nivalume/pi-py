"""Agent loop that works with agent messages throughout.

Messages are converted to LLM messages only at the provider-call boundary.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pi_ai import (
    AbortSignal,
    AssistantMessage,
    EventStream,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    ToolStateChanges,
    get_current_tools,
    get_tool_state_changes,
    normalize_context,
    now_ms,
    to_tool_declaration,
    validate_tool_arguments,
)
from pi_ai.utils.callbacks import maybe_await
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, SchemaTelemetrySpan

from .stream_fn import get_default_stream_fn
from .telemetry import trace_agent_run
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentStartEvent,
    AgentTool,
    AgentToolCallOutcome,
    AgentToolResult,
    AgentTurnContext,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareRequestContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = [
    "AgentEventSink",
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
    "RunToolCallOptions",
    "run_tool_call",
    "ToolCallHooks",
]

AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]

_UPDATE_EVENT_TYPES = frozenset(
    {
        "text_start",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
    }
)


def _create_agent_stream() -> EventStream[AgentEvent, list[Any]]:
    return EventStream(
        lambda event: event.type == "agent_end",
        lambda event: event.messages if event.type == "agent_end" else [],  # type: ignore[union-attr]
    )


def agent_loop(
    prompts: list[Any],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[Any]]:
    """Start an agent loop with new prompt messages and stream its events.

    The returned stream is observational: it preserves event order but does not wait
    for consumers before later loop phases continue. Use :class:`pi_agent.Agent` when
    message processing must act as a barrier.
    """
    stream = _create_agent_stream()

    async def run() -> None:
        messages = await run_agent_loop(prompts, context, config, _push_sink(stream), signal, stream_fn)
        stream.end(messages)

    stream.start(run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[Any]]:
    """Continue from the current context without adding a message (e.g. retries).

    The last message must convert to a ``user`` or ``toolResult`` message.
    """
    _check_continuable(context)
    stream = _create_agent_stream()

    async def run() -> None:
        messages = await run_agent_loop_continue(context, config, _push_sink(stream), signal, stream_fn)
        stream.end(messages)

    stream.start(run())
    return stream


def _push_sink(stream: EventStream[AgentEvent, list[Any]]) -> AgentEventSink:
    def sink(event: AgentEvent) -> None:
        stream.push(event)

    return sink


def _check_continuable(context: AgentContext) -> None:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if result is not None:
        await result


async def run_agent_loop(
    prompts: list[Any],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[Any]:
    """Run the loop for new prompt messages, emitting events through ``emit``."""
    initial = _declare_tool_changes(context, prompts)
    new_messages: list[Any] = list(initial)
    current = AgentContext(messages=[*context.messages, *initial], tools=context.tools)
    stream_function = stream_fn or get_default_stream_fn()

    async def run(span: SchemaTelemetrySpan | None) -> str:
        await _emit(emit, AgentStartEvent())
        await _emit(emit, TurnStartEvent())
        for message in initial:
            await _emit(emit, MessageStartEvent(message))
            await _emit(emit, MessageEndEvent(message))
        return await _run_loop(current, new_messages, config, signal, emit, stream_function, span)

    await _traced(config, current, "prompt", new_messages, run)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[Any]:
    """Run the loop from the existing context, emitting events through ``emit``."""
    _check_continuable(context)
    new_messages: list[Any] = []
    current = AgentContext(messages=context.messages, tools=context.tools)
    stream_function = stream_fn or get_default_stream_fn()

    async def run(span: SchemaTelemetrySpan | None) -> str:
        await _emit(emit, AgentStartEvent())
        await _emit(emit, TurnStartEvent())
        return await _run_loop(current, new_messages, config, signal, emit, stream_function, span)

    await _traced(config, current, "continue", new_messages, run)
    return new_messages


async def _traced(
    config: AgentLoopConfig,
    context: AgentContext,
    mode: str,
    new_messages: list[Any],
    run: Callable[[SchemaTelemetrySpan | None], Awaitable[str]],
) -> None:
    """Run the loop, inside a ``pi.agent.run`` span when the config has a telemetry context."""
    telemetry = config.telemetry_context
    if telemetry is None or telemetry is NOOP_TELEMETRY_CONTEXT:
        await run(None)
        return
    await trace_agent_run(telemetry, config, len(context.tools or ()), mode, new_messages, run)


def _apply_thinking_level(reasoning: Any, level: Any) -> Any:
    if level is None:
        return reasoning
    return None if level == "off" else level


def _decision_action(decision: Any) -> str | None:
    if decision is None:
        return None
    if isinstance(decision, str):
        return decision
    if isinstance(decision, dict):
        return decision.get("action")
    return getattr(decision, "action", None)


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[Any],
    initial_config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
    span: SchemaTelemetrySpan | None = None,
) -> str:
    """The loop; returns why it ended (``AGENT_END_REASONS``)."""
    current = initial_context
    config = initial_config
    last_completed_turn: AgentTurnContext | None = None
    explicit_continuation = False
    pending: list[Any] = await _poll(config.get_steering_messages)
    if span is not None and pending:
        span.add_event("pi.agent.steering", {"pi.agent.message_count": len(pending)})

    while True:
        has_more_tool_calls = True
        while has_more_tool_calls or pending:
            prepared: list[Any] = []
            if last_completed_turn is not None:
                if config.prepare_next_turn is not None:
                    update = await maybe_await(config.prepare_next_turn(last_completed_turn))
                    if update is not None:
                        current = update.context or current
                        prepared = list(update.messages or [])
                        config = dataclasses.replace(
                            config,
                            model=update.model or config.model,
                            reasoning=_apply_thinking_level(config.reasoning, update.thinking_level),
                        )
                # Preparation can be long-running (e.g. compaction): pick up steering queued meanwhile,
                # unless the earlier poll already returned messages (one-at-a-time would deliver two).
                if not pending:
                    pending = await _poll(config.get_steering_messages)
                    if span is not None and pending:
                        span.add_event("pi.agent.steering", {"pi.agent.message_count": len(pending)})
                await _emit(emit, TurnStartEvent())

            for message in _declare_tool_changes(current, [*prepared, *pending]):
                await _emit(emit, MessageStartEvent(message))
                await _emit(emit, MessageEndEvent(message))
                current.messages.append(message)
                new_messages.append(message)
            pending = []

            if config.prepare_request is not None:
                request_update = await maybe_await(
                    config.prepare_request(
                        PrepareRequestContext(current, config.model, config.reasoning or "off"), signal
                    )
                )
                if request_update is not None:
                    current = request_update.context or current
                    config = dataclasses.replace(
                        config,
                        model=request_update.model or config.model,
                        reasoning=_apply_thinking_level(config.reasoning, request_update.thinking_level),
                    )

            message = await _stream_assistant_response(current, config, signal, emit, stream_function, span)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                last_completed_turn = AgentTurnContext(message, [], current, new_messages)
                if config.finish_turn is not None:
                    await maybe_await(config.finish_turn(last_completed_turn, signal))
                await _emit(emit, TurnEndEvent(message, []))
                await _emit(emit, AgentEndEvent(new_messages))
                return message.stop_reason

            tool_calls = [block for block in message.content if block.type == "toolCall"]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                # A "length" stop means the output was cut off, so every tool call may carry
                # truncated arguments. Fail them all instead of executing borked calls.
                if message.stop_reason == "length":
                    batch = await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    batch = await _execute_tool_calls(current, message, config, signal, emit)
                tool_results.extend(batch.messages)
                has_more_tool_calls = not batch.terminate
                for result in tool_results:
                    current.messages.append(result)
                    new_messages.append(result)

            last_completed_turn = AgentTurnContext(message, tool_results, current, new_messages)
            decision = None
            if config.finish_turn is not None:
                decision = await maybe_await(config.finish_turn(last_completed_turn, signal))
            await _emit(emit, TurnEndEvent(message, tool_results))

            action = _decision_action(decision)
            if action == "end":
                await _emit(emit, AgentEndEvent(new_messages))
                return "ended"
            explicit_continuation = action == "continue"
            pending = await _poll(config.get_steering_messages)
            if span is not None and pending:
                span.add_event("pi.agent.steering", {"pi.agent.message_count": len(pending)})
            if has_more_tool_calls or pending:
                explicit_continuation = False

        follow_ups = await _poll(config.get_follow_up_messages)
        if follow_ups:
            if span is not None:
                span.add_event("pi.agent.follow_up", {"pi.agent.message_count": len(follow_ups)})
            explicit_continuation = False
            pending = follow_ups
            continue
        # No natural request was selected: fulfill the continuation decision with one context-only turn.
        if explicit_continuation:
            explicit_continuation = False
            continue
        break

    await _emit(emit, AgentEndEvent(new_messages))
    return "completed"


async def _poll(source: Callable[[], Awaitable[list[Any]]] | None) -> list[Any]:
    if source is None:
        return []
    return list(await maybe_await(source()) or [])


_NO_CHANGES = ToolStateChanges([], [])


def _with_tool_changes(message: SystemMessage, changes: ToolStateChanges) -> SystemMessage:
    return dataclasses.replace(
        message,
        tools_added=changes.tools_added or None,
        tools_removed=changes.tools_removed or None,
    )


def _declare_tool_changes(context: AgentContext, pending: list[Any]) -> list[Any]:
    """Announce differences between the executable tools and the transcript's declared tools.

    ``context.tools`` is what the runtime can execute; the transcript's system messages
    declare what the model may call. A pending system message has its tool fields
    replaced with the delta; otherwise a new system message is inserted before the
    first non-system pending message.
    """
    system_index = -1
    for index in range(len(pending) - 1, -1, -1):
        if getattr(pending[index], "role", None) == "system":
            system_index = index
            break
    pending_system: SystemMessage | None = pending[system_index] if system_index >= 0 else None
    if pending_system is not None:
        baseline = [
            _with_tool_changes(message, _NO_CHANGES) if index == system_index else message
            for index, message in enumerate(pending)
        ]
    else:
        baseline = pending
    changes = get_tool_state_changes(
        get_current_tools([*context.messages, *baseline]),
        [to_tool_declaration(tool) for tool in context.tools or []],
    )
    unchanged = not changes.tools_added and not changes.tools_removed

    if pending_system is not None:
        if unchanged and not pending_system.tools_added and not pending_system.tools_removed:
            return pending
        return [
            _with_tool_changes(pending_system, changes) if index == system_index else message
            for index, message in enumerate(baseline)
        ]
    if unchanged:
        return pending
    update = _with_tool_changes(SystemMessage("", timestamp=now_ms()), changes)
    insert_at = next(
        (index for index, message in enumerate(pending) if getattr(message, "role", None) != "system"), len(pending)
    )
    return [*pending[:insert_at], update, *pending[insert_at:]]


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
    span: SchemaTelemetrySpan | None = None,
) -> AssistantMessage:
    messages = context.messages
    if config.transform_context is not None:
        messages = await config.transform_context(messages, signal)
    llm_messages = await maybe_await(config.convert_to_llm(messages))
    llm_context = normalize_context({"messages": llm_messages})

    resolved_key = None
    if config.get_api_key is not None:
        resolved_key = await maybe_await(config.get_api_key(config.model.provider))
    resolved_key = resolved_key or config.api_key

    options: dict[str, Any] = dict(config.stream_options)
    if config.reasoning is not None:
        options["reasoning"] = config.reasoning
    if resolved_key is not None:
        options["api_key"] = resolved_key
    if signal is not None:
        options["signal"] = signal
    if span is not None:
        # The run span is the explicit parent of the request's pi.ai.request span.
        options["telemetry_context"] = span
    response = await maybe_await(stream_function(config.model, llm_context, options))
    thinking_level = config.reasoning or "off"

    async def final_result() -> AssistantMessage:
        message = await response.result()
        message.thinking_level = thinking_level
        return message

    partial: AssistantMessage | None = None
    added_partial = False
    async for event in response:
        kind = event.type
        if kind == "start":
            partial = event.partial  # type: ignore[union-attr]
            context.messages.append(partial)
            added_partial = True
            await _emit(emit, MessageStartEvent(partial.shallow_copy()))  # type: ignore[union-attr]
        elif kind in _UPDATE_EVENT_TYPES:
            if partial is not None:
                partial = event.partial  # type: ignore[union-attr]
                context.messages[-1] = partial
                await _emit(emit, MessageUpdateEvent(partial.shallow_copy(), event))  # type: ignore[union-attr]
        elif kind in ("done", "error"):
            final = await final_result()
            if added_partial:
                context.messages[-1] = final
            else:
                context.messages.append(final)
                await _emit(emit, MessageStartEvent(final.shallow_copy()))
            await _emit(emit, MessageEndEvent(final))
            return final

    final = await final_result()
    if added_partial:
        context.messages[-1] = final
    else:
        context.messages.append(final)
        await _emit(emit, MessageStartEvent(final.shallow_copy()))
    await _emit(emit, MessageEndEvent(final))
    return final


@dataclass(slots=True)
class _ExecutedToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass(slots=True)
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: Any


@dataclass(slots=True)
class _ImmediateOutcome:
    result: AgentToolResult
    is_error: bool


@dataclass(slots=True)
class _ExecutedOutcome:
    result: AgentToolResult
    is_error: bool
    duration_ms: int


async def _fail_tool_calls_from_truncated_message(
    tool_calls: list[ToolCall], emit: AgentEventSink
) -> _ExecutedToolCallBatch:
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(emit, ToolExecutionStartEvent(tool_call.id, tool_call.name, tool_call.arguments))
        finalized = AgentToolCallOutcome(
            tool_call,
            _error_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output token limit, '
                "so its arguments may be truncated. Re-issue the tool call with complete arguments."
            ),
            True,
        )
        await _emit_tool_execution_end(finalized, emit)
        result_message = _tool_result_message(finalized)
        await _emit_tool_result_message(result_message, emit)
        messages.append(result_message)
    return _ExecutedToolCallBatch(messages, False)


async def _execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    tool_calls = [block for block in assistant_message.content if block.type == "toolCall"]
    tools_by_name = {tool.name: tool for tool in context.tools or []}
    has_sequential = any(
        getattr(tools_by_name.get(call.name), "execution_mode", None) == "sequential" for call in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential:
        return await _execute_sequential(context, assistant_message, tool_calls, config, signal, emit)
    return await _execute_parallel(context, assistant_message, tool_calls, config, signal, emit)


async def _execute_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    finalized_calls: list[AgentToolCallOutcome] = []
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(emit, ToolExecutionStartEvent(tool_call.id, tool_call.name, tool_call.arguments))
        preparation = await _prepare_tool_call(context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _ImmediateOutcome):
            finalized = AgentToolCallOutcome(tool_call, preparation.result, preparation.is_error)
        else:
            executed = await _execute_prepared(preparation, signal, _tool_update_sink(tool_call, emit))
            finalized = await _finalize_executed(context, assistant_message, preparation, executed, config, signal)
        await _emit_tool_execution_end(finalized, emit)
        result_message = _tool_result_message(finalized)
        await _emit_tool_result_message(result_message, emit)
        finalized_calls.append(finalized)
        messages.append(result_message)
        if signal is not None and signal.aborted:
            break
    return _ExecutedToolCallBatch(messages, _should_terminate(finalized_calls))


async def _execute_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    entries: list[AgentToolCallOutcome | Callable[[], Awaitable[AgentToolCallOutcome]]] = []
    for tool_call in tool_calls:
        await _emit(emit, ToolExecutionStartEvent(tool_call.id, tool_call.name, tool_call.arguments))
        preparation = await _prepare_tool_call(context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _ImmediateOutcome):
            finalized = AgentToolCallOutcome(tool_call, preparation.result, preparation.is_error)
            await _emit_tool_execution_end(finalized, emit)
            entries.append(finalized)
            if signal is not None and signal.aborted:
                break
            continue

        def run(
            prepared: _PreparedToolCall = preparation, call: ToolCall = tool_call
        ) -> Awaitable[AgentToolCallOutcome]:
            async def execute() -> AgentToolCallOutcome:
                if signal is not None and signal.aborted:
                    aborted = AgentToolCallOutcome(call, _error_result("Operation aborted"), True)
                    await _emit_tool_execution_end(aborted, emit)
                    return aborted
                executed = await _execute_prepared(prepared, signal, _tool_update_sink(call, emit))
                outcome = await _finalize_executed(context, assistant_message, prepared, executed, config, signal)
                await _emit_tool_execution_end(outcome, emit)
                return outcome

            return execute()

        entries.append(run)
        if signal is not None and signal.aborted:
            break

    ordered = await asyncio.gather(
        *(entry() if callable(entry) else _completed(entry) for entry in entries)  # type: ignore[arg-type]
    )
    messages: list[ToolResultMessage] = []
    for finalized in ordered:
        result_message = _tool_result_message(finalized)
        await _emit_tool_result_message(result_message, emit)
        messages.append(result_message)
    return _ExecutedToolCallBatch(messages, _should_terminate(list(ordered)))


async def _completed(value: AgentToolCallOutcome) -> AgentToolCallOutcome:
    return value


def _should_terminate(finalized: Sequence[AgentToolCallOutcome]) -> bool:
    return bool(finalized) and all(outcome.result.terminate is True for outcome in finalized)


def _prepare_arguments(tool: AgentTool, tool_call: ToolCall) -> ToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared = tool.prepare_arguments(tool_call.arguments)
    if prepared is tool_call.arguments:
        return tool_call
    return dataclasses.replace(tool_call, arguments=prepared)


#: The ``before_tool_call`` and ``after_tool_call`` hooks (an ``AgentLoopConfig`` or ``RunToolCallOptions``).
ToolCallHooks = Any


async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    hooks: ToolCallHooks,
    signal: AbortSignal | None,
    tools: Sequence[AgentTool] | None = None,
) -> _PreparedToolCall | _ImmediateOutcome:
    available = tools if tools is not None else context.tools or []
    tool = next((candidate for candidate in available if candidate.name == tool_call.name), None)
    if tool is None:
        return _ImmediateOutcome(_error_result(f"Tool {tool_call.name} not found"), True)
    try:
        prepared_call = _prepare_arguments(tool, tool_call)
        validated = validate_tool_arguments(tool, prepared_call)
        before = getattr(hooks, "before_tool_call", None)
        if before is not None:
            before_result: BeforeToolCallResult | None = await maybe_await(
                before(BeforeToolCallContext(assistant_message, tool_call, validated, context), signal)
            )
            if signal is not None and signal.aborted:
                return _ImmediateOutcome(_error_result("Operation aborted"), True)
            if before_result is not None and _get(before_result, "block"):
                result = _error_result(_get(before_result, "reason") or "Tool execution was blocked")
                if _get(before_result, "terminate") is True:
                    result.terminate = True
                return _ImmediateOutcome(result, True)
        if signal is not None and signal.aborted:
            return _ImmediateOutcome(_error_result("Operation aborted"), True)
        return _PreparedToolCall(tool_call, tool, validated)
    except Exception as error:
        return _ImmediateOutcome(_error_result(_error_text(error)), True)


def _get(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _tool_update_sink(tool_call: ToolCall, emit: AgentEventSink) -> Callable[[AgentToolResult], Any]:
    def on_update(partial_result: AgentToolResult) -> Any:
        return emit(ToolExecutionUpdateEvent(tool_call.id, tool_call.name, tool_call.arguments, partial_result))

    return on_update


@dataclass(slots=True, kw_only=True)
class RunToolCallOptions:
    """Options for :func:`run_tool_call`."""

    #: Tools the call resolves against.
    tools: Sequence[AgentTool]
    #: Passed to the hooks as the message that issued the call.
    assistant_message: AssistantMessage
    #: Passed to the hooks as the current agent context.
    context: AgentContext
    signal: AbortSignal | None = None
    on_update: Callable[[AgentToolResult], Any] | None = None
    before_tool_call: Any = None
    after_tool_call: Any = None


async def run_tool_call(tool_call: ToolCall, options: RunToolCallOptions) -> AgentToolCallOutcome:
    """Run one tool call through the same steps as a model-issued call.

    Argument preparation, schema validation, ``before_tool_call``, execution and
    ``after_tool_call`` apply; no events are emitted and no messages added. Never
    raises for tool failures: unknown tools, validation errors, blocked calls and
    exceptions come back with ``is_error=True``.
    """
    preparation = await _prepare_tool_call(
        options.context, options.assistant_message, tool_call, options, options.signal, options.tools
    )
    if isinstance(preparation, _ImmediateOutcome):
        return AgentToolCallOutcome(tool_call, preparation.result, preparation.is_error)
    executed = await _execute_prepared(preparation, options.signal, options.on_update or (lambda _result: None))
    return await _finalize_executed(
        options.context, options.assistant_message, preparation, executed, options, options.signal
    )


async def _execute_prepared(
    prepared: _PreparedToolCall,
    signal: AbortSignal | None,
    on_update: Callable[[AgentToolResult], Any],
) -> _ExecutedOutcome:
    update_events: list[Any] = []
    accepting = True
    started = time.perf_counter()

    def elapsed() -> int:
        return round((time.perf_counter() - started) * 1000)

    def update(partial_result: AgentToolResult) -> None:
        if not accepting:
            return
        pending = on_update(partial_result)
        if pending is not None and asyncio.iscoroutine(pending):
            update_events.append(asyncio.ensure_future(pending))
        elif asyncio.isfuture(pending):
            update_events.append(pending)

    try:
        result = await prepared.tool.execute(prepared.tool_call.id, prepared.args, signal, update)
        duration = elapsed()
        accepting = False
        if update_events:
            await asyncio.gather(*update_events)
        if result is None:
            result = AgentToolResult()
        return _ExecutedOutcome(result, result.is_error is True, duration)
    except Exception as error:
        duration = elapsed()
        accepting = False
        if update_events:
            await asyncio.gather(*update_events, return_exceptions=True)
        return _ExecutedOutcome(_error_result(_error_text(error)), True, duration)
    finally:
        accepting = False


async def _finalize_executed(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedOutcome,
    hooks: ToolCallHooks,
    signal: AbortSignal | None,
) -> AgentToolCallOutcome:
    result = executed.result
    is_error = executed.is_error
    after = getattr(hooks, "after_tool_call", None)
    if after is not None:
        try:
            override: AfterToolCallResult | None = await maybe_await(
                after(
                    AfterToolCallContext(
                        assistant_message, prepared.tool_call, prepared.args, result, is_error, context
                    ),
                    signal,
                )
            )
            if override is not None:
                content = _get(override, "content")
                structured = _get(override, "structured_content")
                if structured is None and content is None:
                    structured = result.structured_content
                details = _get(override, "details")
                usage = _get(override, "usage")
                terminate = _get(override, "terminate")
                result = AgentToolResult(
                    content if content is not None else result.content,
                    details if details is not None else result.details,
                    structured_content=structured,
                    usage=usage if usage is not None else result.usage,
                    is_error=result.is_error,
                    terminate=terminate if terminate is not None else result.terminate,
                )
                override_error = _get(override, "is_error")
                if override_error is not None:
                    is_error = override_error
        except Exception as error:
            result = _error_result(_error_text(error))
            is_error = True
    return AgentToolCallOutcome(prepared.tool_call, result, is_error, executed.duration_ms)


def _error_text(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _error_result(message: str) -> AgentToolResult:
    return AgentToolResult([TextContent(message)], {})


async def _emit_tool_execution_end(finalized: AgentToolCallOutcome, emit: AgentEventSink) -> None:
    await _emit(
        emit,
        ToolExecutionEndEvent(
            finalized.tool_call.id,
            finalized.tool_call.name,
            finalized.result,
            finalized.is_error,
            finalized.duration_ms,
        ),
    )


def _tool_result_message(finalized: AgentToolCallOutcome) -> ToolResultMessage:
    return ToolResultMessage(
        finalized.tool_call.id,
        finalized.tool_call.name,
        # Untyped tools can return results without content; never let None into the transcript.
        list(finalized.result.content or []),
        finalized.is_error,
        details=finalized.result.details,
        usage=finalized.result.usage,
        duration_ms=finalized.duration_ms,
        timestamp=now_ms(),
    )


async def _emit_tool_result_message(message: ToolResultMessage, emit: AgentEventSink) -> None:
    await _emit(emit, MessageStartEvent(message))
    await _emit(emit, MessageEndEvent(message))
