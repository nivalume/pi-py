"""The live document: run control and presentation of the current generation and tool round (port of ``harness/live.ts``).

``pi.live`` is strict JSON: an in-flight message is stored as its ``pi_ai.serialization`` JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypedDict

from ..documents import define_doc
from ..types import AnyTaskRecord, ConversationDocToken, EntryId, SubmissionId, SubmissionSettlement, TaskId, Tx
from .types import CompactionReason, CompactionResult, ToolDiagnostic

if TYPE_CHECKING:
    from ..session.transaction import Transaction
    from .scheduler import SchedulerOutcome

__all__ = [
    "CompactionFailure",
    "CompactionStatus",
    "LiveDoc",
    "LiveState",
    "NestedToolSlot",
    "NestedToolSummary",
    "ToolSlot",
    "add_compaction_status",
    "clear_progress",
    "compaction_status",
    "end_run",
    "fail_compaction_status",
    "finish_slot",
    "remove_compaction_status",
    "remove_nested_slots",
    "settle_scheduler_outcome",
    "tool_slot",
]


class _SlotProgress(TypedDict):
    callId: str
    name: str
    status: Literal["pending", "running", "done"]
    #: Retained running output and what the bounds dropped.
    output: NotRequired[str]
    droppedBytes: NotRequired[int]
    droppedLines: NotRequired[int]
    #: Last ``details()`` value.
    details: NotRequired[Any]
    #: Diagnostics recorded through ``api.diagnostic()``.
    diagnostics: NotRequired[list[ToolDiagnostic]]


class ToolSlot(_SlotProgress):
    """Presentation of one tool call of the current round.

    ``taskId`` is absent for a call not started yet (sequential round) and for a call its request did not
    offer, which starts ``done`` with the ``entry`` generation wrote.
    """

    taskId: NotRequired[TaskId[Any]]
    #: Result entry once done; absent when the tool task faulted or was orphaned.
    entry: NotRequired[EntryId]


class NestedToolSummary(TypedDict):
    """How a nested call ended: an error flag, its execution time, its spend, and up to 500 characters of error text."""

    isError: bool
    durationMs: NotRequired[float]
    usage: NotRequired[Any]
    error: NotRequired[str]


class NestedToolSlot(_SlotProgress):
    """Presentation of one nested call: a call a running tool made through ``execute_tool()``.

    ``callId`` is ``<parentCallId>/<key>``. ``taskId`` is the call's tool task, ``parentTaskId`` the calling one.
    """

    parentCallId: str
    parentTaskId: TaskId[Any]
    taskId: TaskId[Any]
    #: What the call was made with, replaced by the arguments it runs with once it starts executing.
    arguments: dict[str, Any]
    #: Once done: how the call ended, for a status line.
    summary: NotRequired[NestedToolSummary]


class CompactionRetry(TypedDict):
    at: float
    error: str


class CompactionStatus(TypedDict):
    """Presentation of one live compaction task (spec §8.7)."""

    taskId: TaskId[CompactionResult]
    reason: CompactionReason
    #: Whether a generation waits for it: a compaction the generation owns.
    blocking: bool
    attempt: int
    #: Durable backoff before the next summarization attempt.
    retry: NotRequired[CompactionRetry]


class CompactionFailure(TypedDict):
    """The newest compaction that failed since a compaction last produced a summary (spec §8.7)."""

    taskId: TaskId[CompactionResult]
    reason: CompactionReason
    #: The failed or faulted outcome's error message, or why the scheduler orphaned the task.
    error: str


class LiveRun(TypedDict):
    taskId: TaskId[Any]
    inputs: list[SubmissionId]
    #: ``on_yield`` continuations the run took; absent while none.
    continuations: NotRequired[int]
    #: The ``on_input`` hooks saw the run's inputs; absent until then.
    inputChecked: NotRequired[Literal[True]]


class LiveGeneration(TypedDict):
    attempt: int
    #: Committed throttled partial of the in-flight response (``AssistantMessage`` JSON).
    message: NotRequired[Any]
    #: Durable backoff before the next attempt.
    retry: NotRequired[CompactionRetry]
    #: Provider-side deferred response being polled.
    deferred: NotRequired[dict[str, float]]


class LiveState(TypedDict, total=False):
    """Built-in live conversation state: run control and presentation of the current generation and tool round."""

    #: Run control: the task that settles the run's inputs, and those inputs; present exactly while busy.
    run: LiveRun
    #: Presentation of the current generation attempt.
    generation: LiveGeneration
    #: The current tool round in call order.
    tools: list[ToolSlot]
    #: Nested calls of running tool calls, in creation order (task IDs ascend).
    nestedTools: list[NestedToolSlot]
    #: Live compaction tasks in task ID order; absent when none.
    compactions: list[CompactionStatus]
    #: The newest compaction failure, so a client can show it after the task ended.
    compactionFailure: CompactionFailure


def _checkpoint_when(value: Any, *_args: Any) -> bool:
    # REMINDER: a complete base whenever nothing runs (spec §8.2): no generation and no running tool slot. A
    # slot holds output only while running, so every base is small. Do not add a delta-count bound.
    return (
        "generation" not in value
        and not any(slot["status"] == "running" for slot in value.get("tools", ()))
        and not any(slot["status"] == "running" for slot in value.get("nestedTools", ()))
    )


LiveDoc: ConversationDocToken[LiveState] = define_doc(
    kind="pi.live",
    version=1,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda: {},
    checkpoint_when=_checkpoint_when,
)

#: Built-in task kinds that can own ``pi.live.run``.
_RUN_TASK_KINDS: Final = frozenset({"pi.generation"})
_TOOL_TASK_KIND: Final = "pi.tool"
_COMPACTION_TASK_KIND: Final = "pi.compaction"


def end_run(tx: Tx, live: Any, task_id: TaskId[Any], settlement: SubmissionSettlement) -> None:
    """End the run owned by ``task_id``: settle each of its inputs and remove ``run``.

    Always removes ``generation``, ``tools``, and ``nestedTools``, whose presentation belongs to the ending run.
    """
    run = live.get("run")
    if run is not None and run["taskId"] == task_id:
        for id in run["inputs"]:
            tx.settle_submission(id, settlement)
        del live["run"]
    for key in ("generation", "tools", "nestedTools"):
        live.pop(key, None)


def add_compaction_status(live: Any, status: CompactionStatus) -> None:
    """Add the status of a compaction task created in this commit; statuses stay in task ID order."""
    if "compactions" not in live:
        live["compactions"] = []
    live["compactions"].append(status)


def compaction_status(live: Any, task_id: TaskId[Any]) -> Any:
    """The status of compaction task ``task_id``, if listed."""
    for status in live.get("compactions", ()):
        if status["taskId"] == task_id:
            return status
    return None


def remove_compaction_status(live: Any, task_id: TaskId[Any]) -> None:
    """Remove the status of compaction task ``task_id``, and the list once empty."""
    statuses = live.get("compactions")
    if statuses is None:
        return
    for index, status in enumerate(statuses):
        if status["taskId"] == task_id:
            del statuses[index]
            break
    if len(statuses) == 0:
        del live["compactions"]


def fail_compaction_status(live: Any, task_id: TaskId[Any], error: str) -> None:
    """Remove the status of compaction task ``task_id`` and record that it failed with ``error``."""
    status = compaction_status(live, task_id)
    if status is not None:
        live["compactionFailure"] = {"taskId": status["taskId"], "reason": status["reason"], "error": error}
    remove_compaction_status(live, task_id)


def tool_slot(live: Any, task_id: TaskId[Any]) -> Any:
    """The slot of tool task ``task_id``, model-issued or nested, if ``pi.live`` still lists it.

    Nested slots are found by binary search: their task IDs ascend, and every element a search reads costs a
    node in the draft's change tracker.
    """
    for slot in live.get("tools", ()):
        if slot.get("taskId") == task_id:
            return slot
    nested = live.get("nestedTools")
    if nested is None:
        return None
    low = 0
    high = len(nested) - 1
    while low <= high:
        middle = (low + high) // 2
        candidate = nested[middle]
        candidate_id = candidate["taskId"]
        if candidate_id == task_id:
            return candidate
        if candidate_id < task_id:
            low = middle + 1
        else:
            high = middle - 1
    return None


def finish_slot(slot: Any) -> None:
    """Mark a slot done and clear its running output; the caller sets a model-issued call's ``entry``, or a
    nested call's ``summary``. Neither is set when the task faulted or was orphaned."""
    slot["status"] = "done"
    clear_progress(slot)


def remove_nested_slots(live: Any, task_id: TaskId[Any]) -> None:
    """Remove the nested slots below tool task ``task_id``, transitively; the list holds parents before children."""
    nested = live.get("nestedTools")
    if nested is None:
        return
    removed = {task_id}
    for slot in nested:
        if slot["parentTaskId"] in removed:
            removed.add(slot["taskId"])
    for index in range(len(nested) - 1, -1, -1):
        if nested[index]["parentTaskId"] in removed:
            del nested[index]
    if len(nested) == 0:
        del live["nestedTools"]


def clear_progress(slot: Any) -> None:
    """Remove what a tool published while running; its result entry or a rerun replaces it."""
    for key in ("output", "droppedBytes", "droppedLines", "details", "diagnostics"):
        slot.pop(key, None)


async def settle_scheduler_outcome(tx: Transaction, record: AnyTaskRecord, outcome: SchedulerOutcome) -> None:
    """Harness cleanup for a terminal outcome the scheduler writes itself (``faulted`` or ``orphaned``).

    A run task ends its run; a tool task's slot is marked done without an entry, and context derivation
    synthesizes the missing result; a compaction task's status is removed and its failure recorded. Ignores
    other kinds so it never creates ``pi.live`` elsewhere.
    REMINDER: a committed generation partial becomes an aborted assistant entry here, exactly as in the
    generation abort handler. The scheduler's commit has no task scope, so that entry has no ``byTaskId``.
    """
    from .generation import convert_partial

    kind = record["kind"]
    conversation_id = record["conversationId"]
    if kind == _TOOL_TASK_KIND:
        live = await tx.doc(LiveDoc, conversation_id)
        slot = tool_slot(live, record["id"])
        if slot is not None:
            finish_slot(slot)
        remove_nested_slots(live, record["id"])
        return
    if kind == _COMPACTION_TASK_KIND:
        error = outcome["error"]["message"] if outcome["status"] == "faulted" else outcome["reason"]  # type: ignore[typeddict-item]
        fail_compaction_status(await tx.doc(LiveDoc, conversation_id), record["id"], error)
        return
    if kind not in _RUN_TASK_KINDS:
        return
    live = await tx.doc(LiveDoc, conversation_id)
    run = live.get("run")
    if run is None or run["taskId"] != record["id"]:
        return
    await convert_partial(tx, live, conversation_id)
    end_run(
        tx,
        live,
        record["id"],
        {"status": "unanswered", "reason": "faulted", "detail": outcome["error"]["message"]}  # type: ignore[typeddict-item]
        if outcome["status"] == "faulted"
        else {"status": "unanswered", "reason": outcome["reason"]},  # type: ignore[typeddict-item]
    )
