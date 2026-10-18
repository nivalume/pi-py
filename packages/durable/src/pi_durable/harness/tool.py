"""The built-in tool task (port of ``harness/tool.ts``).

Task inputs, checkpoints, and nested results are strict JSON with upstream's keys: a nested call's
``ToolCall`` is stored as its ``pi_ai.serialization`` JSON, as is any ``Usage``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict, TypeVar, cast

from pi_ai.serialization import content_from_dict, to_json_value
from pi_ai.types import ImageContent, TextContent, ToolCall, ToolResultMessage
from pi_ai.utils.schema import is_pydantic_model, parameters_schema
from pi_ai.utils.validation import check_schema, schema_errors, validate_tool_arguments
from pi_chord.context import Context, await_with_context
from pi_chord.delta import overlap
from pi_chord.json import UNDEFINED, copy_json
from pi_chord.output import (
    PROGRESS_BYTES_PER_SECOND,
    OutputBuffer,
    OutputLimits,
    OutputSkip,
    Progress,
    ProgressClock,
    bound_output,
    utf8_byte_length,
)

from ..documents import define_doc_family
from ..entries import AssistantEntry, ToolResultEntry
from ..errors import error_message
from ..tasks import define_task
from ..truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from ..types import ConversationId, EntryId, Task, TaskDocFamilyToken, TaskId, Tx
from .agent import tool_extension
from .json import assign_json
from .live import LiveDoc, clear_progress, finish_slot, remove_nested_slots, tool_slot
from .types import (
    NestedToolExecutionResult,
    RetainedOutput,
    ToolControl,
    ToolDiagnostic,
    ToolExecutionResult,
    ToolHookCall,
    ToolHooks,
    ToolRegistration,
)
from .usage import record_usage, usage_json
from .util import json_equal, optional_extension

__all__ = [
    "NestedCallDoc",
    "NestedCallState",
    "ToolTask",
    "ToolTaskCheckpoint",
    "ToolTaskInput",
    "ToolTaskInputModel",
    "ToolTaskInputNested",
    "ToolTaskResult",
    "append_tool_result",
    "harness_error",
]

T = TypeVar("T")


class ToolTaskInputModel(TypedDict):
    """A model-issued call, read from its assistant entry."""

    kind: Literal["model"]
    assistant: EntryId
    callId: str


class ToolTaskInputNested(TypedDict):
    """A nested call a running tool made through ``execute_tool()``; no entry holds the call, so the input does."""

    kind: Literal["nested"]
    parent: TaskId[Any]
    parentCallId: str
    key: str
    #: The ``ToolCall`` as JSON.
    call: dict[str, Any]
    #: ``False``: commit no running output, details, or diagnostics to the slot; the result still gets them.
    progress: NotRequired[Literal[False]]


ToolTaskInput: TypeAlias = ToolTaskInputModel | ToolTaskInputNested


class ToolTaskCheckpointCall(TypedDict):
    phase: Literal["call"]


class ToolTaskCheckpointExecute(TypedDict):
    """Durable intent: the final arguments and the replay policy recorded before execution."""

    phase: Literal["execute"]
    arguments: dict[str, Any]
    replay: Literal["safe", "unsafe"]


ToolTaskCheckpoint: TypeAlias = ToolTaskCheckpointCall | ToolTaskCheckpointExecute


class ToolTaskResultModel(TypedDict):
    kind: Literal["model"]
    entryId: EntryId
    control: NotRequired[ToolControl]


class ToolTaskResultNested(TypedDict):
    kind: Literal["nested"]


#: A model-issued call's result is its transcript entry; a nested call's result is in its caller's ``NestedCallDoc``.
#: A version 1 result has no ``kind``: treat it as a model-issued call's.
ToolTaskResult: TypeAlias = ToolTaskResultModel | ToolTaskResultNested

#: Memo of a tool task that made a nested call, in any attempt, committed before the first admission of each attempt.
_MADE_NESTED_CALLS: Final = "pi.tool.madeNestedCalls"


class NestedCallState(TypedDict, total=False):
    """One nested call of a running tool: its task, and once it ended, its result."""

    #: Absent only inside the commit that creates the call.
    taskId: TaskId[ToolTaskResult]
    #: Exactly what ``execute_tool()`` returns; absent while running, and for a call the scheduler faulted or orphaned.
    result: NestedToolExecutionResult


#: The nested calls of a tool call, a task-scoped family of the calling task keyed by the nested call's key.
NestedCallDoc: TaskDocFamilyToken[NestedCallState, None] = define_doc_family(
    kind="pi.tool.nested-call",
    version=1,
    scope="task",
    initial=lambda _seed: {},
)

Content: TypeAlias = list[TextContent | ImageContent]


async def _call(task: Any, runtime: Any, context: Context) -> None:
    task_input = task["input"]
    call = await _read_call(runtime, task_input, context)
    tool = await _resolve_tool(runtime, task_input, call.name, context)
    if tool is None:
        error = harness_error("tool_unavailable", f"Tool {call.name} is not available")
        return await _settle(runtime, task_input, call, _COMPLETED, lambda _slot: error, context)
    prepared = _prepare(tool, call.arguments)
    checked = prepared if "error" in prepared else _validate(tool, call, prepared["args"])
    if "error" in checked:
        failure = checked["error"]
        return await _settle(runtime, task_input, call, _COMPLETED, lambda _slot: _invalid(failure), context)
    state: dict[str, Any] = {"args": checked["json"], "block": None}

    async def before_tool(hook: Any) -> None:
        if state["block"] is not None:
            return
        try:
            result = hook(dataclasses.replace(call, arguments=state["args"]), runtime, context)
            decision = await result if inspect.isawaitable(result) else result
            if decision is not None and decision.get("block") is not None:
                state["block"] = decision["block"]
            elif decision is not None and decision.get("arguments") is not None:
                state["args"] = decision["arguments"]
        except Exception as error:
            if runtime.signal.aborted:
                raise
            state["block"] = error_message(error)

    await runtime.hooks.each("before_tool", before_tool)
    if state["block"] is not None:
        blocked = harness_error("blocked", f"Tool call blocked: {state['block']}")
        return await _settle(runtime, task_input, call, _COMPLETED, lambda _slot: blocked, context)
    validated = _validate(tool, call, state["args"])
    if "error" in validated:
        failure = validated["error"]
        return await _settle(runtime, task_input, call, _COMPLETED, lambda _slot: _invalid(failure), context)
    # Strict JSON: repair may set optional properties to UNDEFINED, which the slot and checkpoint cannot hold.
    final: dict[str, Any] = copy_json(validated["json"], omit_undefined_properties=True)  # type: ignore[assignment]

    async def intent(tx: Tx, _current: Any) -> Any:
        slot = tool_slot(await tx.doc(LiveDoc, runtime.conversation_id), runtime.task_id)
        if slot is not None:
            slot["status"] = "running"
        # A nested slot shows the arguments the call runs with, which repair, hooks, and coercion may have changed.
        # Compared with the input, which the slot got at admission: reading the draft would track every leaf.
        changed = task_input["kind"] == "nested" and not json_equal(task_input["call"]["arguments"], final)
        if changed and slot is not None and "parentTaskId" in slot:
            slot["arguments"] = final
        checkpoint = {"phase": "execute", "arguments": final, "replay": tool.replay or "unsafe"}
        return {"status": "running", "checkpoint": checkpoint}

    await runtime.commit(intent, context)
    await _run(runtime, task_input, call, tool, final, validated["args"], context)


async def _execute(task: Any, runtime: Any, context: Context) -> None:
    """Recovery after intent: rerun only when the stored and the current policy both say ``safe``."""
    checkpoint = task["state"]["checkpoint"]
    args = checkpoint["arguments"]
    task_input = task["input"]
    call = await _read_call(runtime, task_input, context)
    tool = await _resolve_tool(runtime, task_input, call.name, context)
    if checkpoint["replay"] == "safe" and tool is not None and tool.replay == "safe":
        # The rerun reports from scratch; clear what the interrupted attempt published.
        async def clear(tx: Tx, _current: Any) -> None:
            slot = tool_slot(await tx.doc(LiveDoc, runtime.conversation_id), runtime.task_id)
            if slot is not None:
                clear_progress(slot)

        await runtime.commit(clear, context)
        return await _run(runtime, task_input, call, tool, args, _execution_args(tool, args), context)
    message = f"Tool {call.name} was interrupted and may have partially run"
    # ``failed`` records cancellation intent, so the call's owned conversations, left unsupervised, are aborted.
    ending = {"status": "failed", "message": message}
    await _settle(runtime, task_input, call, ending, lambda slot: _from_slot(slot, "interrupted", message), context)


async def _abort(task: Any, runtime: Any, context: Context) -> None:
    call = await _read_call(runtime, task["input"], context)
    # Abandoned after a restart (``TaskOptions.abandonOnRestart``): say whether the call may have run.
    if task.get("abortReason") != "restart":
        code, message = "aborted", f"Tool {call.name} was aborted"
    elif task["state"]["checkpoint"]["phase"] == "execute":
        code, message = "interrupted", f"Tool {call.name} was interrupted by a restart and may have partially run"
    else:
        code, message = "abandoned", f"Tool {call.name} was not started: its caller ended with a restart"
    await _settle(
        runtime, task["input"], call, {"status": "aborted"}, lambda slot: _from_slot(slot, code, message), context
    )


def _migrate(input: Any, checkpoint: Any, _version: int) -> Any:
    # Version 1 had only model-issued calls.
    return {"input": {"kind": "model", **input}, "checkpoint": checkpoint}


#: Built-in tool task: resolves the called tool among its phase agent's tools, validates, runs ``before_tool``,
#: records intent, executes, runs ``after_tool``, and settles the result, all in one ``call`` handler. ``execute`` is
#: reached only by recovery and applies the replay rule. A model-issued call settles by appending its result
#: entry; a nested call by writing its result to its caller's ``NestedCallDoc``.
ToolTask: Task[ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult, ToolHooks] = define_task(
    name="pi.tool",
    version=2,
    initial=lambda _input: {"phase": "call"},
    migrate=_migrate,
    phases={"call": _call, "execute": _execute},
    abort=_abort,
)


async def _resolve_tool(runtime: Any, input: Any, name: str, context: Context) -> ToolRegistration | None:
    """The called tool: among the tools the model is offered for a model-issued call, else among those tools may call."""
    agent = await runtime.agent(context)
    tools = agent["tools"] if input["kind"] == "model" else agent["callable"]
    return next((tool for tool in tools if tool.name == name), None)


def _hook_call(call: ToolCall, parent: Any = None) -> ToolHookCall:
    return ToolHookCall(
        call.id,
        call.name,
        call.arguments,
        thought_signature=call.thought_signature,
        namespace=call.namespace,
        arguments_error=call.arguments_error,
        parent=parent,
    )


async def _read_call(runtime: Any, input: Any, context: Context) -> ToolHookCall:
    """The call: from the input for a nested call, which hooks see with its parent, else from the assistant entry."""
    if input["kind"] == "nested":
        stored: ToolCall = content_from_dict(dict(input["call"]))
        return _hook_call(stored, {"taskId": input["parent"], "callId": input["parentCallId"]})
    entry = await runtime.entry(AssistantEntry, input["assistant"], context)
    model = None if entry is None else entry.get("model")
    message = model[0] if model else None
    call = (
        next(
            (
                content
                for content in getattr(message, "content", ())
                if isinstance(content, ToolCall) and content.id == input["callId"]
            ),
            None,
        )
        if message is not None and message.role == "assistant"
        else None
    )
    if call is None:
        raise RuntimeError(f"Entry {input['assistant']} has no tool call {input['callId']}")
    return _hook_call(call)


def _prepare(tool: ToolRegistration, args: Any) -> dict[str, Any]:
    """The call's arguments as repaired by the tool; a raising repair makes them invalid."""
    if tool.prepare_arguments is None:
        return {"args": args}
    try:
        return {"args": tool.prepare_arguments(args)}
    except Exception as error:
        return {"error": error_message(error)}


def _validate(tool: ToolRegistration, call: ToolCall, args: Any) -> dict[str, Any]:
    """Arguments validated and coerced against the implementation's schema: ``{"args", "json"}`` or ``{"error"}``.

    ``args`` is what ``execute()`` receives (a model instance for Pydantic parameters), ``json`` its JSON.
    A property repair set to ``UNDEFINED`` is absent, as JavaScript's schema check treats ``undefined``.
    """
    try:
        args = _omit_undefined(args)
        validated = validate_tool_arguments(tool, dataclasses.replace(call, arguments=args))
    except Exception as error:
        return {"error": error_message(error)}
    dump = getattr(validated, "model_dump", None)
    if callable(dump) and not isinstance(validated, dict):
        return {"args": validated, "json": dump(mode="json")}
    return {"args": validated, "json": validated}


def _execution_args(tool: ToolRegistration, args: Any) -> Any:
    """What ``execute()`` receives for stored JSON arguments: a model instance for Pydantic parameters."""
    if is_pydantic_model(tool.parameters):
        return tool.parameters.model_validate(args)
    return args


def _invalid(message: str) -> ToolExecutionResult:
    return harness_error("invalid_arguments", message)


class _Reported:
    """What a running tool reported through its api: output, the last details, and diagnostics."""

    __slots__ = ("details", "diagnostics", "limits", "output")

    def __init__(self, limits: OutputLimits) -> None:
        self.output = OutputBuffer(limits)
        self.limits = limits
        self.diagnostics: list[ToolDiagnostic] = []
        self.details: Any = None


class _ToolApi:
    """Operations available to one tool invocation (``ToolExecutionApi``). Every operation raises after it ends."""

    def __init__(
        self,
        runtime: Any,
        call: ToolHookCall,
        reported: Any,
        progress: Progress,
        state: dict[str, Any],
        execute_tool: Callable[..., Awaitable[NestedToolExecutionResult]],
        create_task: Callable[..., Awaitable[Any]],
    ) -> None:
        self._runtime = runtime
        self._reported = reported
        self._progress = progress
        self._state = state
        self._execute_tool = execute_tool
        self._create_task = create_task
        self.task_id = runtime.task_id
        self.conversation_id = runtime.conversation_id
        self.call_id = call.id
        self.registry = runtime.registry
        limits = reported.limits
        self.output_window = (
            {
                "maxBytes": limits.max_bytes,
                "maxLines": limits.max_lines,
                "minIntervalMs": runtime.settings["progress"]["outputIntervalMs"],
                "bytesPerSecond": PROGRESS_BYTES_PER_SECOND,
            }
            if limits.retain == "tail"
            else None
        )
        self.memo = runtime.memo
        self.get_task = runtime.get_task
        self.wait_for_task = runtime.wait_for_task
        self.conversation = runtime.conversation
        self.snapshot = runtime.snapshot
        self.snapshot_as_of = runtime.snapshot_as_of
        self.watch_doc = runtime.watch_doc

    def _assert_live(self) -> None:
        if self._state["ended"]:
            raise RuntimeError(f"Tool call {self.call_id} has settled")

    def agent(self, context: Context) -> Awaitable[Any]:
        return self._runtime.agent(context)

    @property
    def models(self) -> Any:
        return self._runtime.models

    @property
    def env(self) -> Any:
        return self._state["env"]

    def now(self) -> float:
        return self._runtime.now()

    def output(self, chunk: str | bytes, skipped: Any = None) -> None:
        self._assert_live()
        skip = (
            None
            if skipped is None
            else skipped
            if isinstance(skipped, OutputSkip)
            else OutputSkip(skipped["bytes"], skipped["newlines"], skipped["endsWithNewline"])
        )
        if self._reported.output.push(chunk, skip):
            self._progress.mark()

    def retained_output(self) -> RetainedOutput:
        self._assert_live()
        retained = self._reported.output.snapshot()
        return {"text": retained.text, "truncated": retained.dropped_bytes > 0}

    def diagnostic(self, diagnostic: ToolDiagnostic) -> None:
        self._assert_live()
        self._reported.diagnostics.append(copy_json(diagnostic, omit_undefined_properties=True))
        self._progress.mark()

    def details(self, value: Any, context: Context) -> asyncio.Future[None]:
        """Report details; the checks and the update run at the call (upstream's synchronous part)."""
        try:
            self._assert_live()
            context.throw_if_aborted()
            self._reported.details = copy_json(value, omit_undefined_properties=True)
            committed = self._progress.mark_and_wait()
        except BaseException as error:
            failed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            failed.set_exception(error)
            return failed
        # Cancelling the wait leaves the update in place; the commit's own outcome stays observed.
        committed.add_done_callback(_observe)
        return asyncio.ensure_future(await_with_context(asyncio.shield(committed), context))

    async def commit(self, change: Callable[[Tx], Any], context: Context) -> Any:
        result: list[Any] = []

        async def run(tx: Tx, _current: Any) -> None:
            outcome = change(tx)
            result.append(await outcome if inspect.isawaitable(outcome) else outcome)

        await self._runtime.commit(run, context)
        return result[0]

    def create_task(self, task: Any, input: Any, options: Any, context: Context) -> Awaitable[Any]:
        return self._create_task(task, input, options, context)

    def execute_tool(
        self, name: str, args: Any, context: Context, *, key: str | None = None, progress: bool | None = None
    ) -> Awaitable[NestedToolExecutionResult]:
        return self._execute_tool(name, args, context, key, progress)


def _observe(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


async def _run(
    runtime: Any,
    input: Any,
    call: ToolHookCall,
    tool: ToolRegistration,
    args: Any,
    execution_args: Any,
    context: Context,
) -> None:
    """Execute with the resolved implementation, then settle its result."""
    limits_option = tool.output_limits or {}
    limits = OutputLimits(
        max_bytes=limits_option.get("maxBytes", DEFAULT_MAX_BYTES),
        max_lines=limits_option.get("maxLines", DEFAULT_MAX_LINES),
        retain=limits_option.get("retain", "head"),
    )
    reported = _Reported(limits)
    streams = input["kind"] == "model" or input.get("progress") is not False
    progress = _publish_progress(runtime, reported, streams, context)
    #: Nested call admissions this invocation started; cleanup waits for them, so it sees every nested call.
    admissions: list[asyncio.Future[Any]] = []
    # A caller that can rerun reattaches to its nested calls and child tasks after a restart; any other caller never
    # resumes, so they are abandoned (``TaskOptions.abandonOnRestart``).
    resumes = tool.replay == "safe"
    state: dict[str, Any] = {"ended": False, "env": None, "sequence": 0, "memo": None}

    async def create_task(task: Any, task_input: Any, options: Any, task_context: Context) -> Any:
        child = (
            {"abandonOnRestart": True, **options} if options["ownership"]["kind"] == "task" and not resumes else options
        )
        created: list[Any] = []

        async def change(tx: Tx, _current: Any) -> None:
            created.append(await tx.create_task(task, task_input, cast(Any, child)))

        await runtime.commit(change, task_context)
        return created[0]

    def execute_tool(
        name: str, tool_args: Any, call_context: Context, key: str | None, nested_progress: bool | None
    ) -> asyncio.Future[NestedToolExecutionResult]:
        """Eager (section 5 of DESIGN.md): the checks, the key, and the admission start at the call, as upstream's
        ``async`` body runs them before its first ``await``; a caller that returns right after still has it awaited
        by cleanup."""
        try:
            nested_key, admission = start_nested(name, tool_args, call_context, key, nested_progress)
        except Exception as error:
            failed: asyncio.Future[NestedToolExecutionResult] = asyncio.get_running_loop().create_future()
            failed.set_exception(error)
            return failed

        async def finish() -> NestedToolExecutionResult:
            id = await admission
            settled = await runtime.wait_for_task(id, call_context)
            stored = await runtime.snapshot(NestedCallDoc, runtime.task_id, nested_key, call_context)
            result = None if stored is None else stored.get("result")
            # A copy the tool may change; a nested call the scheduler faulted or orphaned stored none.
            if result is None:
                return _fallback_result(id, name, settled["state"]["outcome"])
            return _nested_from_json(result)

        return asyncio.ensure_future(finish())

    def start_nested(
        name: str, tool_args: Any, call_context: Context, key: str | None, nested_progress: bool | None
    ) -> tuple[str, asyncio.Future[TaskId[Any]]]:
        if state["ended"]:
            raise RuntimeError(f"Tool call {call.id} has settled")
        if key is not None:
            _check_key(key)
            nested_key = key
        else:
            state["sequence"] += 1
            nested_key = str(state["sequence"])
        # The memo first, once per attempt; a failed one is tried again by the next call.
        if state["memo"] is None:
            memo = asyncio.ensure_future(runtime.memo(_MADE_NESTED_CALLS, True, call_context))

            def forget(future: asyncio.Future[Any]) -> None:
                if not future.cancelled() and future.exception() is not None and state["memo"] is future:
                    state["memo"] = None

            memo.add_done_callback(forget)
            state["memo"] = memo
        memo_future: asyncio.Future[Any] = state["memo"]

        async def admit() -> TaskId[Any]:
            await asyncio.shield(memo_future)
            return await _start_nested_call(
                runtime,
                call.id,
                name,
                tool_args,
                nested_key,
                nested_progress,
                not resumes,
                call_context,
            )

        # Registered now, so cleanup waits for it.
        admission = asyncio.ensure_future(admit())
        admissions.append(admission)
        return nested_key, admission

    api = _ToolApi(runtime, call, reported, progress, state, execute_tool, create_task)
    ending: dict[str, Any] = _COMPLETED
    duration_ms: float | None = None
    try:
        state["env"] = await runtime.env(context)
        started_at = time.perf_counter()
        try:
            executed: Any = tool.execute(execution_args, api, context)  # type: ignore[arg-type]
            result: ToolExecutionResult = await executed if inspect.isawaitable(executed) else executed
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000)
    except Exception as error:
        if runtime.signal.aborted:
            state["ended"] = True
            for waiter in await progress.stop():
                if not waiter.done():
                    waiter.set_exception(error)
            raise
        result = {"isError": True, "diagnostics": [_tool_diagnostic("tool_error", error_message(error))]}
        # A raise, from ``execute()`` or from building the environment, ends the task ``failed``, which cancels what
        # the call owned; it no longer supervises it. The error text is already in the result.
        ending = {"status": "failed", "message": f"Tool {call.name} threw"}
    state["ended"] = True
    # No admission starts after ``ended``; let those underway commit, so settlement aborts and lists their calls.
    if admissions:
        await asyncio.wait(admissions)
    reported.output.end()
    # Details still waiting for a progress commit settle with the terminal commit, the final flush.
    pending = await progress.stop()
    try:
        settled_result = await _final_result(runtime, input, call, tool, result, reported, context)
        await _settle(runtime, input, call, ending, lambda _slot: settled_result, context, duration_ms)
    except BaseException as error:
        for waiter in pending:
            if not waiter.done():
                waiter.set_exception(error)
        raise
    for waiter in pending:
        if not waiter.done():
            waiter.set_result(None)


def _progress_clock(runtime: Any) -> ProgressClock:
    """The Harness clock and timers for progress pacing; after the invocation ends the clock keeps its last time."""
    last = [0.0]

    def now() -> float:
        try:
            last[0] = runtime.now()
        except Exception:
            # The invocation ended.
            pass
        return last[0]

    return ProgressClock(now=now, set_timer=lambda callback, delay_ms: runtime.set_timer(callback, delay_ms))


def _publish_progress(runtime: Any, reported: Any, enabled: bool, context: Context) -> Progress:
    """Throttled commits of what the tool reported into its slot, each writing only what changed since the last one."""
    written: dict[str, Any] = {"text": "", "details": None, "diagnostics": 0}

    async def write() -> int:
        if not enabled:
            return 0
        # Capture everything synchronously: the tool keeps reporting while the commit is in flight.
        snapshot = reported.output.snapshot()
        current = {"text": snapshot.text, "details": reported.details, "diagnostics": len(reported.diagnostics)}
        added = reported.diagnostics[written["diagnostics"] : current["diagnostics"]]
        details_changed = current["details"] is not written["details"]
        # What the commit writes, as Chord diffs the string: an append, a trim plus an append of what follows the
        # shared part, or the whole window when its bounded overlap search finds nothing.
        size = 0
        if snapshot.text != written["text"]:
            shared = (
                len(written["text"])
                if snapshot.text.startswith(written["text"])
                else overlap(written["text"], snapshot.text, 65_536)
            )
            size += utf8_byte_length(snapshot.text[shared:])
        if details_changed:
            size += utf8_byte_length(json.dumps(current["details"], separators=(",", ":"), ensure_ascii=False))
        if len(added) > 0:
            size += utf8_byte_length(json.dumps(added, separators=(",", ":"), ensure_ascii=False))

        async def change(tx: Tx, _current: Any) -> None:
            slot = tool_slot(await tx.doc(LiveDoc, runtime.conversation_id), runtime.task_id)
            if slot is None:
                return
            # REMINDER: assign ``output`` as one string field. Chord then diffs it into an append, or a trim plus an
            # append for a sliding tail; replacing the slot object would record the whole window on every commit.
            if slot.get("output", "") != snapshot.text:
                slot["output"] = snapshot.text
            if snapshot.dropped_bytes > 0:
                slot["droppedBytes"] = snapshot.dropped_bytes
            if snapshot.dropped_lines > 0:
                slot["droppedLines"] = snapshot.dropped_lines
            # Diff details leaf by leaf and append new diagnostics, so each commit writes only what changed.
            if details_changed and current["details"] is not None:
                assign_json(slot, "details", current["details"])
            if len(added) > 0:
                if "diagnostics" not in slot:
                    slot["diagnostics"] = []
                for diagnostic in added:
                    slot["diagnostics"].append(diagnostic)

        await runtime.commit(change, context)
        written.update(current)
        return size

    def on_error(error: BaseException) -> None:
        # Rejections after an abort mark or close are expected; the committed state stays consistent.
        if not runtime.signal.aborted:
            runtime.report(error)

    return Progress(write, on_error, runtime.settings["progress"]["outputIntervalMs"], _progress_clock(runtime))


async def _final_result(
    runtime: Any,
    input: Any,
    call: ToolHookCall,
    tool: ToolRegistration,
    result: ToolExecutionResult,
    reported: Any,
    context: Context,
) -> dict[str, Any]:
    """The settled result: the tool's result with the retained output and last details as fallbacks, its
    diagnostics after those reported through the api, ``after_tool`` applied, explicit text bounded, and
    ``structuredOutput`` checked against the tool's schema, with the Harness's diagnostics last."""
    harness: list[ToolDiagnostic] = []
    retained = reported.output.snapshot() if "output" not in result else None
    output: Content = (
        list(cast(Any, result)["output"])
        if retained is None
        else []
        if retained.text == ""
        else [TextContent(retained.text)]
    )
    final: dict[str, Any] = {**result, "output": output}
    # An explicit ``None`` (JSON null) replaces the reported details; only an absent key falls back to them.
    details = result.get("details", reported.details)
    if details is None and "details" not in result:
        final.pop("details", None)
    else:
        final["details"] = details
    final["diagnostics"] = [*reported.diagnostics, *result.get("diagnostics", [])]
    holder = [final]

    async def after_tool(hook: Any) -> None:
        outcome = hook(call, holder[0], runtime, context)
        replaced = await outcome if inspect.isawaitable(outcome) else outcome
        if replaced is not None:
            holder[0] = replaced

    await runtime.hooks.each("after_tool", after_tool)
    final = dict(holder[0])
    # The retained output's truncation applies only while after_tool kept that output.
    if retained is not None and retained.dropped_bytes > 0 and final.get("output") is output:
        harness.append(_truncated(retained.dropped_lines, retained.dropped_bytes, reported.limits.retain))
    bounded, dropped_bytes, dropped_lines = _bound_content(list(final.get("output") or []), reported.limits)
    if dropped_bytes > 0:
        harness.append(_truncated(dropped_lines, dropped_bytes, reported.limits.retain))
    broken = _structured_output_error(tool, final)
    if broken is not None:
        rest = {key: value for key, value in final.items() if key != "structuredOutput"}
        if input["kind"] == "nested":
            final = {**rest, "isError": True}
            harness.append(_tool_diagnostic("invalid_structured_output", broken))
        else:
            # The model never sees structured output, so its call stands; the host learns of the broken tool.
            final = rest
            # Only names the extension: a failed resolution leaves the report without the name.
            try:
                extension = tool_extension(await runtime.agent(context), tool.name)
            except Exception:
                extension = None
            runtime.report(RuntimeError(broken), {"type": "extension", **optional_extension(extension)})
    # Programs get the tool's own value: without a schema, its output. The model reads only ``output``.
    if input["kind"] == "nested" and tool.structured_output_schema is None:
        final = {**final, "structuredOutput": _output_value(bounded)}
    # A schema tool's error result without structured output has its text only in the output, which programs never
    # get: it becomes an error diagnostic, as a thrown error's message does, unless one says what went wrong already.
    if (
        input["kind"] == "nested"
        and tool.structured_output_schema is not None
        and final.get("isError") is True
        and not _defined(final, "structuredOutput")
        and not any(diagnostic["severity"] == "error" for diagnostic in [*final.get("diagnostics", []), *harness])
    ):
        text = "\n".join(item.text for item in bounded if isinstance(item, TextContent))
        if text != "":
            harness.append(_tool_diagnostic("tool_error", text))
    return {**final, "output": bounded, "diagnostics": [*final.get("diagnostics", []), *harness]}


def _structured_output_error(tool: ToolRegistration, result: dict[str, Any]) -> str | None:
    """Why a result's ``structuredOutput`` breaks the tool's contract, or ``None`` when it keeps it."""
    schema = tool.structured_output_schema
    value = result.get("structuredOutput")
    # JSON null is a value (a ``Type.Null()`` schema returns it); only an absent key is "no structuredOutput".
    if schema is None:
        return (
            None
            if not _defined(result, "structuredOutput")
            else f"Tool {tool.name} returned structuredOutput but declares no structuredOutputSchema"
        )
    if not _defined(result, "structuredOutput"):
        return None if result.get("isError") is True else f"Tool {tool.name} returned no structuredOutput"
    plain = parameters_schema(schema) if not isinstance(schema, dict) else schema
    if check_schema(plain, value):
        return None
    errors = schema_errors(plain, value)
    path, message = errors[0] if errors else ("root", "invalid")
    return f"Tool {tool.name} returned structuredOutput that does not match its schema: {path or 'root'}: {message}"


async def _settle(
    runtime: Any,
    input: Any,
    call: ToolHookCall,
    ending: dict[str, Any],
    build: Callable[[Any], Any],
    context: Context,
    duration_ms: float | None = None,
) -> None:
    """Commit the tool's terminal state and mark its slot done.

    A model-issued call appends its result entry and ends with the entry ID; a nested call writes its result to
    the caller's ``NestedCallDoc``, records its usage, and ends with a small receipt. Nested calls this call left
    running are aborted first, so they report into their slots before the slots below the call leave ``pi.live``.
    ``build`` receives the slot so interruption and abort can report the durable partial output.
    """
    # Every live nested call the task owns, also those an earlier attempt made and this one never called again.
    nested_ids = [
        record["id"]
        for record in await runtime.owned_tasks(context)
        if record["kind"] == ToolTask.definition.name and record["input"].get("kind") == "nested"
    ]
    # Read before the settling commit: no admission starts after ``ended``, and those underway were awaited.
    made_nested = (await runtime.memo(_MADE_NESTED_CALLS, context)) is True
    # Mark them all before waiting for any: one may run until a sibling is cancelled.
    if nested_ids:
        await asyncio.gather(*(runtime.abort_owned(id, context) for id in nested_ids))

    async def change(tx: Tx, _current: Any) -> Any:
        conversation_id = runtime.conversation_id
        live = await tx.doc(LiveDoc, conversation_id)
        slot = tool_slot(live, runtime.task_id)
        result = build(None if slot is None else copy_json(slot))
        settled: Any
        if input["kind"] == "nested":
            nested = _nested_result(runtime.task_id, result, duration_ms)
            if "usage" in nested:
                await record_usage(tx, conversation_id, "tools", call.name, nested["usage"])
            # The caller is never terminal before its owned nested calls, so its document is open; another task in it
            # is a bug, not a race.
            state = await tx.doc(NestedCallDoc, input["parent"], input["key"], None)
            if state.get("taskId") != runtime.task_id:
                raise RuntimeError(f"Nested call {call.id} is not in its caller's index")
            state["result"] = cast(Any, nested)
            if slot is not None and "parentTaskId" in slot:
                finish_slot(slot)
                slot["summary"] = _summary_of(nested, result.get("output") or [])
            settled = {"kind": "nested"}
        else:
            entry = await append_tool_result(
                tx, conversation_id, call, result, timestamp=runtime.now(), duration_ms=duration_ms
            )
            if slot is not None and "parentTaskId" not in slot:
                finish_slot(slot)
                slot["entry"] = entry["id"]
            settled = {"kind": "model", "entryId": entry["id"]}
            control = result.get("control")
            if control is not None and ending["status"] == "completed":
                # Tools build control objects freely; drop keys set to UNDEFINED so the task result is strict JSON.
                settled["control"] = copy_json(dict(control), omit_undefined_properties=True)
        # A call that made no nested calls, in any attempt, has no slots below it; skipping the scan keeps many
        # leaves linear.
        if made_nested:
            remove_nested_slots(live, runtime.task_id)
        if ending["status"] == "aborted":
            return {"status": "terminal", "outcome": {"status": "aborted", "result": settled}}
        if ending["status"] == "failed":
            error = {"message": ending["message"]}
            return {"status": "terminal", "outcome": {"status": "failed", "error": error, "result": settled}}
        return {"status": "terminal", "outcome": {"status": "completed", "result": settled}}

    await runtime.commit(change, context)


def _nested_result(task_id: TaskId[Any], result: Any, duration_ms: float | None) -> dict[str, Any]:
    """A nested call's result, as stored and returned: strict JSON, without ``output`` and ``control``."""
    nested: dict[str, Any] = {"taskId": task_id}
    # Upstream's ``copyJson`` drops only undefined properties: a JSON null result or details is kept.
    if _defined(result, "structuredOutput"):
        nested["structuredOutput"] = _plain_json(result["structuredOutput"])
    nested["isError"] = result.get("isError", False)
    if _defined(result, "details"):
        nested["details"] = _plain_json(result["details"])
    nested["diagnostics"] = [_plain_json(diagnostic) for diagnostic in result.get("diagnostics", [])]
    if result.get("usage") is not None:
        nested["usage"] = usage_json(result["usage"])
    if duration_ms is not None:
        nested["durationMs"] = duration_ms
    return copy_json(nested, omit_undefined_properties=True)  # type: ignore[return-value]


def _omit_undefined(value: Any) -> Any:
    """``value`` without the dict entries (at any depth) whose value is ``UNDEFINED``; other values unchanged."""
    if isinstance(value, dict):
        return {key: _omit_undefined(item) for key, item in value.items() if item is not UNDEFINED}
    if isinstance(value, list):
        return [_omit_undefined(item) for item in value]
    return value


def _defined(result: Any, key: str) -> bool:
    """Whether ``result[key]`` is not JS ``undefined``: present and not ``UNDEFINED`` (``None`` is JSON null)."""
    return key in result and result[key] is not UNDEFINED


def _plain_json(value: Any) -> Any:
    """A value as strict JSON: pi-ai data classes (an image) serialized."""
    return copy_json(to_json_value(value), omit_undefined_properties=True)


def _nested_from_json(stored: Any) -> NestedToolExecutionResult:
    """What ``execute_tool()`` returns for a stored result: a copy the tool may change."""
    result: Any = copy_json(stored)
    if "usage" in result:
        from pi_ai.serialization import from_dict
        from pi_ai.types import Usage

        result["usage"] = from_dict(Usage, result["usage"])
    return result


async def _start_nested_call(
    runtime: Any,
    parent_call_id: str,
    name: str,
    args: Any,
    key: str,
    progress: bool | None,
    abandon_on_restart: bool,
    context: Context,
) -> TaskId[Any]:
    """Create nested call ``key`` of the call ``parent_call_id`` in one commit: its tool task, owned by the calling
    task, its slot, and its ``NestedCallDoc``. A key already made returns that call's task when it names the same
    tool and arguments."""
    arguments = copy_json(args, omit_undefined_properties=True)
    call = {"type": "toolCall", "id": f"{parent_call_id}/{key}", "name": name, "arguments": arguments}
    created: list[Any] = []

    async def change(tx: Tx, _current: Any) -> None:
        # Opened, or created empty for a key not made yet.
        state = await tx.doc(NestedCallDoc, runtime.task_id, key, None)
        existing = state.get("taskId")
        if existing is not None:
            record = await tx.task(existing)
            existing_input = None if record is None else record["input"]
            same = (
                existing_input is not None
                and existing_input.get("kind") == "nested"
                and existing_input["call"]["name"] == name
                and json_equal(existing_input["call"]["arguments"], arguments)
            )
            if not same:
                raise RuntimeError(f"Nested call {call['id']} was already made with another tool or other arguments")
            created.append(existing)
            return
        task_input: dict[str, Any] = {
            "kind": "nested",
            "parent": runtime.task_id,
            "parentCallId": parent_call_id,
            "key": key,
            "call": call,
        }
        if progress is False:
            task_input["progress"] = False
        task_id = await tx.create_task(
            ToolTask,
            cast(Any, task_input),
            {"ownership": {"kind": "task", "taskId": runtime.task_id}, "abandonOnRestart": abandon_on_restart},
        )
        state["taskId"] = task_id
        live = await tx.doc(LiveDoc, runtime.conversation_id)
        if "nestedTools" not in live:
            live["nestedTools"] = []
        slot: Any = {
            "callId": call["id"],
            "parentCallId": parent_call_id,
            "parentTaskId": runtime.task_id,
            "name": name,
            "taskId": task_id,
            "arguments": arguments,
            "status": "pending",
        }
        live["nestedTools"].append(slot)
        created.append(task_id)

    await runtime.commit(change, context)
    return created[0]


def _output_value(output: Content) -> Any:
    """What programs get of a tool's output when it declares no schema: one text item as its string, one image as
    its JSON, nothing as ``""``, and anything else as the content list's JSON."""
    if len(output) == 0:
        return ""
    if len(output) > 1:
        return _plain_json(output)
    only = output[0]
    return only.text if isinstance(only, TextContent) else _plain_json(only)


def _fallback_result(task_id: TaskId[Any], name: str, outcome: Any) -> NestedToolExecutionResult:
    """What the caller gets for a nested call without a stored result: one the scheduler faulted or orphaned."""
    status = outcome["status"]
    if status == "faulted":
        message = f"Tool {name} failed: {outcome['error']['message']}"
    elif status == "orphaned":
        message = f"Tool {name} could not resume: {outcome['reason']}"
    else:
        message = f"Tool {name} ended without a result"
    error = harness_error(status, message)
    return {"taskId": task_id, "isError": True, "diagnostics": list(error.get("diagnostics", []))}


_COMPLETED: Final[dict[str, Any]] = {"status": "completed"}


def _from_slot(slot: Any, code: str, message: str) -> ToolExecutionResult:
    """An error result from the slot's durable partial output, details, and diagnostics."""
    slot = slot or {}
    diagnostics: list[ToolDiagnostic] = list(slot.get("diagnostics", []))
    dropped_bytes = slot.get("droppedBytes", 0)
    if dropped_bytes > 0:
        diagnostics.append(_truncated(slot.get("droppedLines", 0), dropped_bytes))
    diagnostics.append(_tool_diagnostic(code, message))
    output = slot.get("output")
    result: dict[str, Any] = {
        "output": [] if output is None or output == "" else [TextContent(output)],
        "isError": True,
    }
    if slot.get("details") is not None:
        result["details"] = slot["details"]
    result["diagnostics"] = diagnostics
    return result  # type: ignore[return-value]


def harness_error(code: str, message: str) -> ToolExecutionResult:
    """An error result the Harness writes itself: no content and one ``error`` diagnostic with ``code``."""
    return {"output": [], "isError": True, "diagnostics": [_tool_diagnostic(code, message)]}


def _tool_diagnostic(code: str, message: str) -> ToolDiagnostic:
    return {"severity": "error", "code": code, "message": message}


def _truncated(dropped_lines: int, dropped_bytes: int, retain: str | None = None) -> ToolDiagnostic:
    """The Harness's truncation diagnostic; ``retain`` is unknown when rebuilt from a slot after recovery."""
    kept = "" if retain is None else f" to its {'beginning' if retain == 'head' else 'end'}"
    return {
        "severity": "warn",
        "code": "truncated",
        "message": f"Output truncated{kept}: {dropped_lines} lines, {dropped_bytes} bytes dropped",
    }


#: Error text a nested call's summary keeps.
_MAX_SUMMARY_ERROR_CHARS: Final = 500


def _summary_of(result: dict[str, Any], output: Sequence[Any]) -> dict[str, Any]:
    """How a nested call ended, for its slot: error text, from its diagnostics or else its ``output``, bounded."""
    error = _error_text_of(result, output)[:_MAX_SUMMARY_ERROR_CHARS] if result["isError"] else ""
    summary: dict[str, Any] = {"isError": result["isError"]}
    if "durationMs" in result:
        summary["durationMs"] = result["durationMs"]
    if "usage" in result:
        summary["usage"] = result["usage"]
    if error != "":
        summary["error"] = error
    return summary


def _error_text_of(result: dict[str, Any], output: Sequence[Any]) -> str:
    """The error messages of a nested result, else the text items of its output, joined with newlines."""
    errors = [diagnostic for diagnostic in result["diagnostics"] if diagnostic["severity"] == "error"]
    if len(errors) > 0:
        return "\n".join(diagnostic["message"] for diagnostic in errors)
    return "\n".join(item.text for item in output if isinstance(item, TextContent))


async def append_tool_result(
    tx: Tx,
    conversation_id: ConversationId,
    call: ToolCall,
    result: Any,
    *,
    timestamp: float,
    duration_ms: float | None = None,
) -> Any:
    """Append a ``pi.tool-result`` entry.

    The content ends with the rendered diagnostics, so the stored message is exactly what the model sees;
    ``data`` keeps the structured list. A result's usage is added to ``pi.usage`` in the same commit.
    """
    diagnostics = [
        copy_json(diagnostic, omit_undefined_properties=True) for diagnostic in result.get("diagnostics", [])
    ]
    content: Content = list(result.get("output") or [])
    if len(diagnostics) > 0:
        content.append(TextContent(_render_diagnostics(cast(Any, diagnostics))))
    usage = result.get("usage")
    if usage is not None and isinstance(usage, dict):
        from pi_ai.serialization import from_dict
        from pi_ai.types import Usage

        usage = from_dict(Usage, usage)
    message = ToolResultMessage(
        call.id,
        call.name,
        content,
        result.get("isError", False),
        details=result.get("details"),
        usage=usage,
        timestamp=timestamp,  # type: ignore[arg-type]
        duration_ms=duration_ms,  # type: ignore[arg-type]
    )
    if usage is not None:
        await record_usage(tx, conversation_id, "tools", call.name, usage)
    return await tx.append_entry(
        ToolResultEntry, conversation_id, {"model": [message], "data": {"diagnostics": diagnostics}}
    )


def _render_diagnostics(diagnostics: Sequence[ToolDiagnostic]) -> str:
    lines = "\n".join(f"[{diagnostic['severity']}] {diagnostic['message']}" for diagnostic in diagnostics)
    return f"<harness>\n{lines}\n</harness>"


def _bound_content(content: Content, limits: OutputLimits) -> tuple[Content, int, int]:
    """Bound the text of result content.

    When the joined text exceeds the limits, the text items are replaced by one bounded item at the position of
    the first (head) or last (tail) text item; other content is kept.
    """
    texts = [item for item in content if isinstance(item, TextContent)]
    bounded = bound_output("".join(item.text for item in texts), limits)
    if bounded.dropped_bytes == 0:
        return content, 0, 0
    keep = texts[0] if limits.retain == "head" else texts[-1]
    result: Content = []
    for item in content:
        if not isinstance(item, TextContent):
            result.append(item)
        elif item is keep:
            result.append(dataclasses.replace(item, text=bounded.text))
    return result, bounded.dropped_bytes, bounded.dropped_lines


_POSITIVE_INTEGER: Final = re.compile(r"^[1-9][0-9]*$")


def _check_key(key: str) -> None:
    """Explicit nested call keys are path segments of the call ID, never plain positive integers (default keys)."""
    if key == "" or "/" in key or key == "__proto__" or _POSITIVE_INTEGER.match(key):
        raise RuntimeError(
            f'Nested call key {json.dumps(key)} must be non-empty, without "/", not "__proto__", and not a positive integer'
        )
