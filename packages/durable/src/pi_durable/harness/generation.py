"""The built-in generation task (port of ``harness/generation.ts``).

Checkpoints are strict JSON with upstream's keys: a deferred handle is stored as its ``pi_ai.serialization``
JSON, and a committed partial in ``pi.live`` as the message's JSON.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
from collections.abc import Sequence
from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict, cast

from pi_ai.serialization import from_dict, message_from_dict, to_json_value
from pi_ai.types import AssistantMessage, DeferredHandle, Message, ToolCall, UserMessage
from pi_ai.utils.overflow import is_context_overflow
from pi_ai.utils.retry import is_retryable_assistant_error
from pi_ai.utils.transcript import get_current_tools
from pi_chord.context import Context
from pi_chord.json import copy_json

from ..entries import AssistantEntry, ResetEntry, SystemEntry, UserEntry
from ..tasks import define_task
from ..types import ConversationId, EntryId, SubmissionId, SubmissionSettlement, Task, TaskId, Tx
from .agent import add_tools, section_extension
from .compaction import compaction_policy, create_compaction, estimate_context, select_cut
from .inbox import Boundary, apply_boundary, prepare_boundary, withdraw_queued_inputs
from .json import assign_json
from .live import LiveDoc, end_run
from .prompt import plan_system_entries, render_sections, replay_sections
from .provider import ensure_provider_session_id
from .stream import retry_delay, stream_options
from .types import (
    CompactionPolicy,
    CompactionResult,
    ContextView,
    ConversationStreamOptions,
    GenerationHooks,
    ModelRef,
    PromptInput,
    UserInput,
)
from .usage import record_usage
from .util import optional_extension

__all__ = [
    "GenerationCheckpoint",
    "GenerationInput",
    "GenerationResult",
    "GenerationTask",
    "append_assistant",
    "convert_partial",
    "hand_over",
    "start_run",
]

GenerationInput: TypeAlias = dict[str, Any]


class GenerationCheckpointPrepare(TypedDict):
    phase: Literal["prepare"]
    attempt: int
    #: The blocking compaction this generation waited for; it starts no other compaction (spec §8.3).
    compacted: NotRequired[TaskId[CompactionResult]]
    #: Error text of the overflow that started ``compacted``; checked once when ``prepare`` resumes.
    overflow: NotRequired[str]
    #: Set when the blocking threshold started ``compacted``; its outcome is checked once when ``prepare`` resumes.
    threshold: NotRequired[Literal[True]]


class GenerationCheckpointRequest(TypedDict):
    phase: Literal["request"]
    attempt: int
    compacted: NotRequired[TaskId[CompactionResult]]
    model: ModelRef
    thinkingLevel: str
    #: The settings' request options when preparation committed; a resend after recovery uses them unchanged.
    streamOptions: ConversationStreamOptions
    #: Newest entry included in the request.
    cutoff: EntryId


class GenerationCheckpointRetry(TypedDict):
    phase: Literal["retry"]
    attempt: int
    compacted: NotRequired[TaskId[CompactionResult]]
    until: float


class GenerationCheckpointPoll(TypedDict):
    phase: Literal["poll"]
    attempt: int
    compacted: NotRequired[TaskId[CompactionResult]]
    model: ModelRef
    cutoff: EntryId
    #: The ``DeferredHandle`` as JSON.
    handle: dict[str, Any]
    pollAt: float


class GenerationCheckpointTools(TypedDict):
    """Waiting on the round's tool tasks, which the generation owns (spec §8.5)."""

    phase: Literal["tools"]
    #: The tool-calling answer.
    assistant: EntryId
    #: Tool tasks created so far, in call order; grows by one per started call of a sequential round.
    tools: list[TaskId[Any]]
    #: Calls of a sequential round not started yet, in call order.
    pending: list[str]


GenerationCheckpoint: TypeAlias = (
    GenerationCheckpointPrepare
    | GenerationCheckpointRequest
    | GenerationCheckpointRetry
    | GenerationCheckpointPoll
    | GenerationCheckpointTools
)


class GenerationResult(TypedDict):
    entryId: EntryId


_DEFAULT_POLL_AFTER_MS: Final = 5000


def _with_compacted(checkpoint: dict[str, Any], compacted: Any) -> dict[str, Any]:
    if compacted is not None:
        checkpoint["compacted"] = compacted
    return checkpoint


async def _prepare(task: Any, runtime: Any, context: Context) -> None:
    """Render the system prompt and tool loadout and append the positional ``pi.system`` entries they need, then
    move to ``request``. The agent and settings resolved here are fixed for this request."""
    conversation_id = runtime.conversation_id
    agent = await runtime.agent(context)
    settings = runtime.settings
    model = agent.get("model")
    thinking_level = agent["thinkingLevel"]
    resolved = None if model is None else runtime.models.get_model(model["provider"], model["modelId"])
    if model is None or resolved is None:
        return await _fail_no_model(runtime, model, context)
    checkpoint = task["state"]["checkpoint"]
    attempt = checkpoint["attempt"]
    compacted = checkpoint.get("compacted")
    overflow = checkpoint.get("overflow")
    blocked = checkpoint.get("threshold")
    if compacted is not None and overflow is not None:
        [outcome] = await runtime.outcomes([compacted], context)
        if outcome["status"] != "completed" or outcome["result"].get("entryId") is None:
            return await _fail_model_error(runtime, overflow, context)
    # A threshold compaction that found nothing to compact or was declined leaves a request that may still fit. One
    # that failed left a context it judged too large: sending it would at best leave no room for the answer.
    if compacted is not None and blocked is True:
        [outcome] = await runtime.outcomes([compacted], context)
        if outcome["status"] != "completed":
            return await _fail_model_error(runtime, _compaction_failure(outcome), context)
    view = await runtime.context(conversation_id, context)
    shown = replay_sections(view["messages"])
    env = None
    try:
        env = await runtime.env(context)
    except Exception as error:
        signal = context.abort_signal
        if signal is not None and signal.aborted:
            raise
        runtime.report(error, {"type": "host", "callback": "env"})
    prompt_input: PromptInput = {
        "conversationId": conversation_id,
        "agent": agent,
        "env": env,
        "shown": dict(shown),
        "read": runtime,
    }
    desired = await render_sections(
        agent["sections"],
        prompt_input,
        shown,
        lambda error, section: runtime.report(
            error, {"type": "extension", **optional_extension(section_extension(agent, section.key))}
        ),
        context,
    )
    entries = plan_system_entries(view, desired, agent["tools"], runtime.now())
    threshold = (
        _threshold_compaction(view, entries, resolved.context_window, compaction_policy(settings, model))
        if compacted is None
        else None
    )
    if threshold == "blocking":
        # Compact first and prepare again; the transcript is unchanged until the compaction appends.
        async def wait_for_compaction(tx: Tx, _current: Any) -> Any:
            child = await create_compaction(tx, conversation_id, {"reason": "threshold"}, runtime.task_id)
            following = {"phase": "prepare", "attempt": attempt, "compacted": child, "threshold": True}
            return {"status": "waiting", "checkpoint": following, "on": [child], "policy": "allSettled"}

        await runtime.commit(wait_for_compaction, context)
        return

    async def commit_request(tx: Tx, _current: Any) -> Any:
        page = await tx.scan_entries({"conversationId": conversation_id}, 1)
        cutoff = page["items"][0]["id"] if len(page["items"]) > 0 else None
        for entry in entries:
            cutoff = (await tx.append_entry(SystemEntry, conversation_id, entry))["id"]
        if cutoff is None:
            raise RuntimeError(f"Conversation {conversation_id} has no entries to send")
        # Checked in this commit, so a compaction admitted during preparation counts.
        if threshold == "background" and "compactions" not in await tx.doc(LiveDoc, conversation_id):
            await create_compaction(tx, conversation_id, {"reason": "threshold"})
        request = _with_compacted({"phase": "request", "attempt": attempt}, compacted)
        request.update(
            {"model": model, "thinkingLevel": thinking_level, "streamOptions": settings["stream"], "cutoff": cutoff}
        )
        return {"status": "running", "checkpoint": request}

    await runtime.commit(commit_request, context)


async def _request(task: Any, runtime: Any, context: Context) -> None:
    checkpoint = task["state"]["checkpoint"]
    attempt = checkpoint["attempt"]
    compacted = checkpoint.get("compacted")
    ref = checkpoint["model"]
    thinking_level = checkpoint["thinkingLevel"]
    cutoff = checkpoint["cutoff"]
    conversation_id = runtime.conversation_id

    async def start_attempt(tx: Tx, _current: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        await convert_partial(tx, live, conversation_id)
        live["generation"] = {"attempt": attempt}

    await runtime.commit(start_attempt, context)
    model = runtime.models.get_model(ref["provider"], ref["modelId"])
    if model is None:
        return await _fail_no_model(runtime, ref, context)
    view = await runtime.context(conversation_id, context, at=cutoff)
    messages: list[list[Message]] = [list(view["messages"])]

    async def before_request(hook: Any) -> None:
        result = hook({"messages": messages[0]}, runtime, context)
        replaced = await result if inspect.isawaitable(result) else result
        if replaced is not None:
            messages[0] = list(replaced["messages"])

    await runtime.hooks.each("before_request", before_request)
    options = stream_options(checkpoint["streamOptions"])
    options["signal"] = runtime.signal
    options["session_id"] = await ensure_provider_session_id(runtime, context)
    if thinking_level != "off":
        options["reasoning"] = thinking_level
    message = await _stream_response(runtime, model, messages[0], options, attempt, context)
    request = {"attempt": attempt, "compacted": compacted, "model": ref, "cutoff": cutoff, "messages": view["messages"]}
    await _classify(runtime, request, message, context)


async def _retry(task: Any, runtime: Any, context: Context) -> None:
    """After the backoff's sleep; a run before ``until`` (an early wake, or an older record) sleeps again."""
    checkpoint = task["state"]["checkpoint"]
    attempt = checkpoint["attempt"]
    until = checkpoint["until"]
    if until > runtime.now():
        return await runtime.sleep_until(until, checkpoint, context)

    async def change(tx: Tx, _current: Any) -> Any:
        (await tx.doc(LiveDoc, runtime.conversation_id))["generation"] = {"attempt": attempt + 1}
        following = _with_compacted({"phase": "prepare", "attempt": attempt + 1}, checkpoint.get("compacted"))
        return {"status": "running", "checkpoint": following}

    await runtime.commit(change, context)


async def _poll(task: Any, runtime: Any, context: Context) -> None:
    """After the sleep until ``pollAt``; a run before it (an early wake, or an older record) sleeps again."""
    checkpoint = task["state"]["checkpoint"]
    poll_at = checkpoint["pollAt"]
    if poll_at > runtime.now():
        return await runtime.sleep_until(poll_at, checkpoint, context)
    ref = checkpoint["model"]
    model = runtime.models.get_model(ref["provider"], ref["modelId"])
    if model is None:
        return await _fail_no_model(runtime, ref, context)
    handle = from_dict(DeferredHandle, checkpoint["handle"])
    message = await runtime.models.fetch_deferred(model, handle, {"signal": runtime.signal})
    request = {
        "attempt": checkpoint["attempt"],
        "compacted": checkpoint.get("compacted"),
        "model": ref,
        "cutoff": checkpoint["cutoff"],
        "pollAt": poll_at,
    }
    await _classify(runtime, request, message, context)


async def _tools(task: Any, runtime: Any, context: Context) -> None:
    checkpoint = task["state"]["checkpoint"]
    assistant = checkpoint["assistant"]
    tools = checkpoint["tools"]
    pending = checkpoint["pending"]
    if len(pending) == 0:
        return await _finish_tool_round(runtime, assistant, tools, context)
    following_call, rest = pending[0], pending[1:]

    # Sequential round: start the next call and wait for it.
    async def change(tx: Tx, _current: Any) -> Any:
        live = await tx.doc(LiveDoc, runtime.conversation_id)
        task_id = await _create_tool_task(tx, runtime, assistant, following_call)
        for slot in live.get("tools", ()):
            if slot["callId"] == following_call and "taskId" not in slot:
                slot["taskId"] = task_id
                break
        following = {"phase": "tools", "assistant": assistant, "tools": [*tools, task_id], "pending": rest}
        return {"status": "waiting", "checkpoint": following, "on": [task_id], "policy": "allSettled"}

    await runtime.commit(change, context)


async def _abort(task: Any, runtime: Any, context: Context) -> None:
    from .tool import append_tool_result, harness_error

    checkpoint = task["state"]["checkpoint"]
    if checkpoint["phase"] == "poll":
        model = runtime.models.get_model(checkpoint["model"]["provider"], checkpoint["model"]["modelId"])
        if model is not None:
            try:
                handle = from_dict(DeferredHandle, checkpoint["handle"])
                await runtime.models.cancel_deferred(model, handle, {"signal": runtime.signal})
            except Exception as error:
                runtime.report(error, {"type": "host", "callback": "models"})
    conversation_id = runtime.conversation_id
    # Runs after the round's tool tasks are terminal; calls never started get ``aborted`` results (spec §8.5).
    unstarted = (
        await _read_calls(runtime, checkpoint["assistant"], checkpoint["pending"], context)
        if checkpoint["phase"] == "tools"
        else []
    )

    async def change(tx: Tx, _current: Any) -> Any:
        live = await tx.doc(LiveDoc, conversation_id)
        await convert_partial(tx, live, conversation_id)
        for call in unstarted:
            result = harness_error("aborted", f"Tool {call.name} was aborted")
            await append_tool_result(tx, conversation_id, call, result, timestamp=runtime.now())
        end_run(tx, live, runtime.task_id, {"status": "unanswered", "reason": "aborted"})
        return {"status": "terminal", "outcome": {"status": "aborted"}}

    await runtime.commit(change, context)


#: Built-in generation task: prepares the positional system prompt and tool loadout, requests or polls the model,
#: retries, and classifies the response. The run's inputs live in ``pi.live.run``.
GenerationTask: Task[GenerationInput, GenerationCheckpoint, GenerationResult, GenerationHooks] = define_task(
    name="pi.generation",
    version=1,
    initial=lambda _input: {"phase": "prepare", "attempt": 1},
    phases={"prepare": _prepare, "request": _request, "retry": _retry, "poll": _poll, "tools": _tools},
    abort=_abort,
)


async def _read_calls(runtime: Any, assistant: EntryId, call_ids: Sequence[str], context: Context) -> list[ToolCall]:
    """The calls ``call_ids`` of the assistant entry, in the given order."""
    entry = await runtime.entry(AssistantEntry, assistant, context)
    model = None if entry is None else entry.get("model")
    message = model[0] if model else None
    calls = (
        [content for content in message.content if isinstance(content, ToolCall)]
        if isinstance(message, AssistantMessage)
        else []
    )
    found: list[ToolCall] = []
    for id in call_ids:
        match = next((call for call in calls if call.id == id), None)
        if match is not None:
            found.append(match)
    return found


def _create_tool_task(tx: Tx, runtime: Any, assistant: EntryId, call_id: str) -> Any:
    """A tool task for call ``call_id``, owned by the generation."""
    from .tool import ToolTask

    return tx.create_task(
        ToolTask,
        {"kind": "model", "assistant": assistant, "callId": call_id},
        {"ownership": {"kind": "task", "taskId": runtime.task_id}},
    )


def _threshold_compaction(
    view: ContextView, planned: Sequence[Any], context_window: float, policy: CompactionPolicy
) -> Literal["blocking", "background"] | None:
    """Which threshold compaction preparation starts before its request (spec §8.3).

    ``blocking`` above ``contextWindow - reserveTokens``, ``background`` above the background threshold, and
    only when range selection finds a cut. The caller starts a background one only while no compaction is listed.
    """
    if not policy["enabled"] or context_window <= 0:
        return None
    tokens = estimate_context(view, [message for entry in planned for message in entry.get("model", ())])
    blocking = context_window - policy["reserveTokens"]
    background = blocking - policy["backgroundTokens"]
    over: Literal["blocking", "background"] | None = (
        "blocking"
        if tokens > blocking
        else "background"
        if policy["backgroundTokens"] > 0 and tokens > background
        else None
    )
    if over is None or select_cut(view, policy["keepRecentTokens"]) is None:
        return None
    return over


def _compaction_failure(outcome: Any) -> str:
    """Error text for a generation whose blocking compaction ended without completing."""
    status = outcome["status"]
    if status in ("failed", "faulted"):
        return f"Compaction failed: {outcome['error']['message']}"
    if status == "orphaned":
        return f"Compaction failed: {outcome['reason']}"
    return "Compaction was aborted"


async def _fail_model_error(runtime: Any, text: str, context: Context) -> None:
    """Settle the run's inputs ``unanswered`` with ``model_error`` and fail with ``text``."""

    async def change(tx: Tx, _current: Any) -> Any:
        failure = await _prepare_failure(tx, runtime)
        live = await tx.doc(LiveDoc, runtime.conversation_id)
        await _end_failed_run(
            tx, runtime, live, failure, {"status": "unanswered", "reason": "model_error", "detail": text}
        )
        return {
            "status": "terminal",
            "outcome": {"status": "failed", "error": {"message": text, "detail": {"reason": "model_error"}}},
        }

    await runtime.commit(change, context)


async def _fail_no_model(runtime: Any, ref: ModelRef | None, context: Context) -> None:
    """Settle the run's inputs ``unanswered`` with ``no_model`` and fail."""
    message = "No model is configured" if ref is None else f"Model {ref['provider']}/{ref['modelId']} is not available"

    async def change(tx: Tx, _current: Any) -> Any:
        failure = await _prepare_failure(tx, runtime)
        live = await tx.doc(LiveDoc, runtime.conversation_id)
        await _end_failed_run(tx, runtime, live, failure, {"status": "unanswered", "reason": "no_model"})
        return {
            "status": "terminal",
            "outcome": {"status": "failed", "error": {"message": message, "detail": {"reason": "no_model"}}},
        }

    await runtime.commit(change, context)


async def _prepare_failure(tx: Tx, runtime: Any) -> tuple[str, Boundary | None]:
    """What a failed run's end needs from the table, read before the commit's first table write."""
    settings = runtime.settings
    policy = settings["afterFailedRun"]
    if policy != "continue":
        return policy, None
    return policy, await prepare_boundary(tx, runtime.conversation_id, settings)


async def _end_failed_run(
    tx: Tx, runtime: Any, live: Any, failure: tuple[str, Boundary | None], settlement: SubmissionSettlement
) -> None:
    """End a failed run, settling its inputs with ``settlement``, and do with the inputs queued behind it what
    ``settings.afterFailedRun`` says (spec §6): keep them, start the next run with them, or withdraw them."""
    conversation_id = runtime.conversation_id
    policy, boundary = failure
    end_run(tx, live, runtime.task_id, settlement)
    if policy == "withdraw":
        await withdraw_queued_inputs(tx, conversation_id, "run_failed")
    if boundary is None:
        return
    result = await apply_boundary(tx, boundary, "final", runtime.now())
    if len(result.users) > 0:
        await start_run(tx, conversation_id, live, result.users)


async def convert_partial(tx: Tx, live: Any, conversation_id: ConversationId) -> None:
    """Append a committed partial left by an interrupted, aborted, faulted, or orphaned attempt as an aborted
    assistant entry; the caller replaces or removes ``generation``."""
    generation = live.get("generation")
    partial = None if generation is None else generation.get("message")
    if partial is None:
        return
    message: AssistantMessage = message_from_dict(copy_json(partial))
    await append_assistant(tx, conversation_id, dataclasses.replace(message, stop_reason="aborted"))


async def _stream_response(
    runtime: Any,
    model: Any,
    messages: Sequence[Message],
    options: dict[str, Any],
    attempt: int,
    context: Context,
) -> AssistantMessage:
    """Stream one request and return the terminal message.

    Partials commit as trailing writes at most every ``progress.partialIntervalMs`` (default 100 ms) with one
    commit in flight; ``finally`` stops the throttle and awaits that commit, so no stale partial lands after the
    outcome.
    """
    interval = runtime.settings["progress"]["partialIntervalMs"]
    state: dict[str, Any] = {"pending": None, "timer": None, "in_flight": None, "stopped": False}

    def flush() -> None:
        state["timer"] = None
        partial = state["pending"]
        state["pending"] = None
        if partial is None or state["stopped"]:
            return
        # Copy synchronously: the provider keeps mutating its partial.
        message = copy_json(to_json_value(partial))

        async def change(tx: Tx, _current: Any) -> None:
            live = await tx.doc(LiveDoc, runtime.conversation_id)
            if "generation" not in live:
                live["generation"] = {"attempt": attempt}
            assign_json(cast(Any, live["generation"]), "message", message)

        async def commit() -> None:
            try:
                await runtime.commit(change, context)
            except Exception as error:
                # Rejections after an abort mark or close are expected; the committed state stays consistent.
                if not runtime.signal.aborted:
                    runtime.report(error)
            finally:
                state["in_flight"] = None
                if state["pending"] is not None and not state["stopped"]:
                    state["timer"] = runtime.set_timer(flush, interval)

        state["in_flight"] = asyncio.ensure_future(commit())

    try:
        events = runtime.models.stream_simple(model, {"messages": list(messages)}, options)
        async for event in events:
            # A partial without content, such as pi-ai's opening ``start`` event, shows nothing; a deferred
            # response never gets past it, so it never leaves a partial.
            if event.type in ("done", "error") or len(event.partial.content) == 0:
                continue
            state["pending"] = event.partial
            if state["timer"] is None and state["in_flight"] is None:
                state["timer"] = runtime.set_timer(flush, interval)
        return await events.result()
    finally:
        state["stopped"] = True
        timer = state["timer"]
        if timer is not None:
            timer()
        in_flight = state["in_flight"]
        if in_flight is not None:
            await in_flight


async def _classify(runtime: Any, request: dict[str, Any], message: AssistantMessage, context: Context) -> None:
    """Classify a terminal provider message in one commit that also clears the partial."""
    # An abort mark or close: the abort invocation or the reopened run handles the committed state.
    runtime.signal.throw_if_aborted()
    conversation_id = runtime.conversation_id
    attempt = request["attempt"]
    compacted = request.get("compacted")
    ref = request["model"]
    cutoff = request["cutoff"]
    if message.stop_reason == "deferred" and message.deferred is not None:
        handle = message.deferred
        poll_after = handle.poll_after_ms if handle.poll_after_ms is not None else _DEFAULT_POLL_AFTER_MS
        previous_poll = request.get("pollAt")
        poll_at = max(runtime.now() + poll_after, -math.inf if previous_poll is None else previous_poll + 1)

        async def sleep(tx: Tx, _current: Any) -> Any:
            (await tx.doc(LiveDoc, conversation_id))["generation"] = {
                "attempt": attempt,
                "deferred": {"pollAt": poll_at},
            }
            checkpoint = _with_compacted({"phase": "poll", "attempt": attempt}, compacted)
            checkpoint.update(
                {"model": ref, "cutoff": cutoff, "handle": copy_json(to_json_value(handle)), "pollAt": poll_at}
            )
            return {"status": "sleeping", "checkpoint": checkpoint, "until": poll_at}

        await runtime.commit(sleep, context)
        return

    async def after_response(hook: Any) -> None:
        result = hook(message, runtime, context)
        if inspect.isawaitable(result):
            await result

    await runtime.hooks.each("after_response", after_response)
    calls = [content for content in message.content if isinstance(content, ToolCall)]
    if message.stop_reason == "toolUse" and len(calls) > 0:
        return await _start_tool_round(runtime, request, message, calls, context)
    if message.stop_reason in ("stop", "length", "toolUse"):
        return await _answer(runtime, message, context)
    # The retry and compaction policies govern the next attempt, so they are read now rather than pinned.
    settings = runtime.settings
    overflow = message.stop_reason == "error" and is_context_overflow(message)
    compaction = compaction_policy(settings, ref)
    if overflow and compacted is None and compaction["enabled"]:
        view = await runtime.context(conversation_id, context, at=cutoff)
        if select_cut(view, compaction["keepRecentTokens"]) is not None:
            text = message.error_message or "Context overflow"

            async def compact(tx: Tx, _current: Any) -> Any:
                live = await tx.doc(LiveDoc, conversation_id)
                await append_assistant(tx, conversation_id, message)
                live.pop("generation", None)
                child = await create_compaction(tx, conversation_id, {"reason": "overflow"}, runtime.task_id)
                checkpoint = {"phase": "prepare", "attempt": attempt, "compacted": child, "overflow": text}
                return {"status": "waiting", "checkpoint": checkpoint, "on": [child], "policy": "allSettled"}

            await runtime.commit(compact, context)
            return
    policy = settings["retry"]
    # An overflow is never retried: only a compaction can make the next request fit.
    retry = (
        message.stop_reason == "error"
        and not overflow
        and is_retryable_assistant_error(message)
        and policy["enabled"]
        and attempt <= policy["maxRetries"]
    )
    until = runtime.now() + retry_delay(policy, attempt) if retry else 0

    async def settle(tx: Tx, _current: Any) -> Any:
        # Read before the first table write; only a failure uses it.
        failure = None if retry else await _prepare_failure(tx, runtime)
        live = await tx.doc(LiveDoc, conversation_id)
        await append_assistant(tx, conversation_id, message)
        if failure is None:
            live["generation"] = {"attempt": attempt, "retry": {"at": until, "error": message.error_message or ""}}
            checkpoint = _with_compacted({"phase": "retry", "attempt": attempt}, compacted)
            checkpoint["until"] = until
            return {"status": "sleeping", "checkpoint": checkpoint, "until": until}
        text = message.error_message or f"Model response ended with stop reason {message.stop_reason}"
        await _end_failed_run(
            tx, runtime, live, failure, {"status": "unanswered", "reason": "model_error", "detail": text}
        )
        return {
            "status": "terminal",
            "outcome": {"status": "failed", "error": {"message": text, "detail": {"reason": "model_error"}}},
        }

    await runtime.commit(settle, context)


async def _answer(runtime: Any, message: AssistantMessage, context: Context) -> None:
    """A final answer; the final boundary places queued items (spec §6).

    The first ``on_yield`` continuation appends a user message and hands the run to a successor generation, but
    only when the boundary selected no user item and no reset. Otherwise the run's inputs settle ``done``, and
    selected user items start the next run.
    """
    continuation: list[UserInput] = []

    async def on_yield(hook: Any) -> None:
        if continuation:
            return
        result = hook(message, runtime, context)
        decided = await result if inspect.isawaitable(result) else result
        if decided is not None and "continue" in decided:
            continuation.append(decided["continue"])

    await runtime.hooks.each("on_yield", on_yield)
    conversation_id = runtime.conversation_id

    async def change(tx: Tx, _current: Any) -> Any:
        # Queue modes are read on the Session line, when the boundary is decided.
        boundary = await prepare_boundary(tx, conversation_id, runtime.settings)
        live = await tx.doc(LiveDoc, conversation_id)
        entry = await append_assistant(tx, conversation_id, message)
        result = {"status": "terminal", "outcome": {"status": "completed", "result": {"entryId": entry["id"]}}}
        placed = await apply_boundary(tx, boundary, "final", runtime.now())
        if continuation and len(placed.users) == 0 and not placed.reset:
            content = continuation[0]
            user = UserMessage(content if isinstance(content, str) else list(content), timestamp=runtime.now())
            await tx.append_entry(UserEntry, conversation_id, {"model": [user]})
            hand_over(live, runtime.task_id, await _create_generation(tx, conversation_id))
            live.pop("generation", None)
            return result
        end_run(tx, live, runtime.task_id, {"status": "done", "answer": entry["id"]})
        if len(placed.users) > 0:
            await start_run(tx, conversation_id, live, placed.users)
        return result

    await runtime.commit(change, context)


async def _start_tool_round(
    runtime: Any, request: dict[str, Any], message: AssistantMessage, calls: Sequence[ToolCall], context: Context
) -> None:
    """Append the tool-calling answer and start its tool round in one commit (spec §8.3).

    A call to a tool the request did not offer gets its ``tool_unavailable`` result here; every other call gets
    a tool task owned by the generation, only the first one now when the round is sequential. The generation
    then waits for them in its ``tools`` phase, keeping the run.
    """
    from .tool import append_tool_result, harness_error

    conversation_id = runtime.conversation_id
    messages = request.get("messages")
    if messages is None:
        messages = (await runtime.context(conversation_id, context, at=request["cutoff"]))["messages"]
    offered = {tool.name for tool in get_current_tools(messages)}
    # Read as the round starts; a tool is resolved as its tool task resolves it.
    agent_tools = (await runtime.agent(context))["tools"]

    def mode_of(name: str) -> Any:
        tool = next((tool for tool in agent_tools if tool.name == name), None)
        return None if tool is None else tool.execution_mode

    sequential = runtime.settings["toolExecution"] == "sequential" or any(
        call.name in offered and mode_of(call.name) == "sequential" for call in calls
    )

    async def change(tx: Tx, _current: Any) -> Any:
        live = await tx.doc(LiveDoc, conversation_id)
        entry = await append_assistant(tx, conversation_id, message)
        slots: list[dict[str, Any]] = []
        tools: list[Any] = []
        pending: list[str] = []
        for call in calls:
            if call.name not in offered:
                unavailable = harness_error("tool_unavailable", f"Tool {call.name} is not available")
                result = await append_tool_result(tx, conversation_id, call, unavailable, timestamp=runtime.now())
                slots.append({"callId": call.id, "name": call.name, "status": "done", "entry": result["id"]})
                continue
            if sequential and len(tools) > 0:
                pending.append(call.id)
                slots.append({"callId": call.id, "name": call.name, "status": "pending"})
                continue
            task_id = await _create_tool_task(tx, runtime, entry["id"], call.id)
            tools.append(task_id)
            slots.append({"callId": call.id, "name": call.name, "taskId": task_id, "status": "pending"})
        live.pop("generation", None)
        live["tools"] = cast(Any, slots)
        checkpoint = {"phase": "tools", "assistant": entry["id"], "tools": tools, "pending": pending}
        return {"status": "waiting", "checkpoint": checkpoint, "on": tools, "policy": "allSettled"}

    await runtime.commit(change, context)


async def _finish_tool_round(runtime: Any, assistant: EntryId, tools: Sequence[TaskId[Any]], context: Context) -> None:
    """The round's tools are terminal: apply their controls and either end the run at the final boundary
    (``terminate``, ``handoff``, or a queued reset) or hand it to the next generation at ``postTools`` (spec §8.5)."""
    conversation_id = runtime.conversation_id
    controls: dict[Any, Any] = {}
    outcomes = await runtime.outcomes(list(tools), context)
    for id, outcome in zip(tools, outcomes, strict=True):
        # Version 1 results have no kind; every result a generation's round reads is a model-issued call's.
        result = outcome.get("result") if outcome["status"] == "completed" else None
        controls[id] = None if result is None or result.get("kind") == "nested" else result.get("control")
    live_value = await runtime.snapshot(LiveDoc, conversation_id, context)
    slots = [] if live_value is None else live_value.get("tools", [])
    results = [slot["entry"] for slot in slots if "entry" in slot]

    async def after_tools(hook: Any) -> None:
        outcome = hook(assistant, results, runtime, context)
        if inspect.isawaitable(outcome):
            await outcome

    await runtime.hooks.each("after_tools", after_tools)
    # Every call of the round, including those answered without a task, must ask to terminate.
    terminate = len(slots) > 0 and all(
        "taskId" in slot and (controls.get(slot["taskId"]) or {}).get("terminate") is True for slot in slots
    )
    added = [name for control in controls.values() for name in (control or {}).get("addTools", [])]
    # The last handoff in call order wins.
    handoff = None
    for control in controls.values():
        if control is not None and control.get("handoff") is not None:
            handoff = control["handoff"]

    async def change(tx: Tx, _current: Any) -> Any:
        boundary = await prepare_boundary(tx, conversation_id, runtime.settings)
        if len(added) > 0:
            await add_tools(tx, conversation_id, added)
        live = await tx.doc(LiveDoc, conversation_id)
        now = runtime.now()
        if terminate or handoff is not None:
            if handoff is not None:
                user = UserMessage(handoff, timestamp=now)
                entry = await tx.append_entry(ResetEntry, conversation_id, {"head": "self", "model": [user]})
                boundary.head = entry["id"]
            placed = await apply_boundary(tx, boundary, "final", now)
            end_run(tx, live, runtime.task_id, {"status": "done", "answer": assistant})
            if len(placed.users) > 0:
                await start_run(tx, conversation_id, live, placed.users)
        else:
            placed = await apply_boundary(tx, boundary, "postTools", now)
            if placed.reset:
                # The queued reset cut the run's context before an answer.
                end_run(tx, live, runtime.task_id, {"status": "unanswered", "reason": "reset"})
                if len(placed.users) > 0:
                    await start_run(tx, conversation_id, live, placed.users)
            else:
                live.pop("tools", None)
                live.pop("nestedTools", None)
                run = live.get("run")
                if run is not None and run["taskId"] == runtime.task_id:
                    run["inputs"].extend(placed.users)
                hand_over(live, runtime.task_id, await _create_generation(tx, conversation_id))
        return {"status": "terminal", "outcome": {"status": "completed", "result": {"entryId": assistant}}}

    await runtime.commit(change, context)


async def append_assistant(tx: Tx, conversation_id: ConversationId, message: AssistantMessage) -> Any:
    """Append a provider result and add its usage to ``pi.usage`` in the same commit.

    REMINDER: every built-in writer of assistant entries goes through here, so the usage ledger stays complete.
    """
    await record_usage(tx, conversation_id, "models", f"{message.provider}/{message.model}", message.usage)
    return await tx.append_entry(AssistantEntry, conversation_id, {"model": [message]})


async def start_run(tx: Tx, conversation_id: ConversationId, live: Any, inputs: list[SubmissionId]) -> None:
    """Start a run for ``inputs``, placed input submissions: a new generation takes ``pi.live.run``."""
    live["run"] = {"taskId": await _create_generation(tx, conversation_id), "inputs": list(inputs)}


def _create_generation(tx: Tx, conversation_id: ConversationId) -> Any:
    """A generation owned by its conversation."""
    return tx.create_task(
        GenerationTask, {}, {"ownership": {"kind": "conversation"}, "conversationId": conversation_id}
    )


def hand_over(live: Any, from_task: TaskId[Any], to_task: TaskId[Any]) -> None:
    """Hand run control from ``from_task`` to ``to_task``; the run's inputs move with it."""
    run = live.get("run")
    if run is not None and run["taskId"] == from_task:
        run["taskId"] = to_task
