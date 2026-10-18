"""Static typing cases for ``test_types.py`` (the compile-time half of upstream ``test/types.test.ts``).

``test_types.py`` runs pyright on this file with ``reportUnnecessaryTypeIgnoreComment``: every line
marked ``# pyright: ignore`` must be a type error (otherwise the ignore is reported as unnecessary),
and every ``assert_type`` must hold. Nothing here runs at import time except definitions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict, assert_type

from pi_chord.context import Context
from pi_durable import (
    Conversation,
    Harness,
    HookApi,
    MemoryStorage,
    SettledTask,
    Storage,
    Task,
    TaskRuntime,
    define_extension,
    define_task,
    hook,
    id_from_number,
)
from pi_durable.harness.types import CompactionResult, SubmissionDraft
from pi_durable.types import (
    ContextEdit,
    ConversationId,
    DocumentContent,
    DocumentCreate,
    DocumentId,
    DocumentRecord,
    EntryId,
    RunningTask,
    Seq,
    StorageWrite,
    SubmissionCreate,
    SubmissionId,
    SubmissionRecord,
    TaskId,
    TaskOutcome,
    TaskState,
)

conversation_id: ConversationId = id_from_number(1)
entry_id: EntryId = id_from_number(2)
answer_id: EntryId = id_from_number(3)
task_id: TaskId[int] = id_from_number(4)
submission_id: SubmissionId = id_from_number(5)
document_id: DocumentId = id_from_number(6)
seq = Seq(1)


def brands_ids() -> None:
    as_int: int = conversation_id
    widened: TaskId[Any] = task_id
    wrong_task: TaskId[Any] = conversation_id  # pyright: ignore
    wrong_conversation: ConversationId = task_id  # pyright: ignore
    wrong_document: DocumentId = entry_id  # pyright: ignore
    wrong_sequence: Seq = entry_id  # pyright: ignore
    narrowed: TaskId[str] = task_id  # pyright: ignore
    del as_int, widened, wrong_task, wrong_conversation, wrong_document, wrong_sequence, narrowed


class Checkpoint(TypedDict):
    phase: str


def encodes_discriminator_dependent_fields() -> None:
    omit: ContextEdit = {"target": entry_id, "action": "omit"}
    replace: ContextEdit = {"target": entry_id, "action": "replace", "messages": []}
    pending: TaskState[Checkpoint, int] = {"status": "pending", "checkpoint": {"phase": "ready"}}
    terminal: TaskState[Checkpoint, int] = {"status": "terminal", "outcome": {"status": "completed", "result": 1}}
    completed_input: SubmissionRecord = {
        "id": submission_id,
        "conversationId": conversation_id,
        "type": "input",
        "status": "done",
        "entry": entry_id,
        "answer": answer_id,
    }
    completed_write: SubmissionRecord = {
        "id": submission_id,
        "conversationId": conversation_id,
        "type": "write",
        "status": "done",
        "entry": entry_id,
    }
    queued_write_create: SubmissionCreate = {"conversationId": conversation_id, "type": "write", "status": "queued"}
    base_content: DocumentContent = {"kind": "base", "version": 1, "value": {"count": 1}}
    delta_content: DocumentContent = {"kind": "delta", "version": 1, "ops": [("s", ("count",), 2)]}
    conversation_document: DocumentCreate = {
        "id": document_id,
        "kind": "test",
        "scope": {"kind": "conversation", "conversationId": conversation_id},
        "history": "rewindable",
        "fork": "asOf",
    }
    if pending["status"] == "pending":
        assert_type(pending["checkpoint"], Checkpoint)
    if terminal["status"] == "terminal" and terminal["outcome"]["status"] == "completed":
        assert_type(terminal["outcome"]["result"], int)
    del omit, replace, completed_input, completed_write, queued_write_create, base_content, delta_content
    del conversation_document


def rejects_invalid_records() -> None:
    missing_replacement: ContextEdit = {"target": entry_id, "action": "replace"}  # pyright: ignore
    omission_with_messages: ContextEdit = {"target": entry_id, "action": "omit", "messages": []}  # pyright: ignore
    pending_with_outcome: TaskState[Checkpoint, int] = {
        "status": "pending",
        "checkpoint": {"phase": "ready"},
        "outcome": {"status": "completed", "result": 1},
    }  # pyright: ignore
    terminal_with_checkpoint: TaskState[Checkpoint, int] = {
        "status": "terminal",
        "checkpoint": {"phase": "ready"},
        "outcome": {"status": "completed", "result": 1},
    }  # pyright: ignore
    session_with_history: DocumentRecord = {
        "id": document_id,
        "kind": "test",
        "createdAt": seq,
        "scope": {"kind": "session"},
        "history": "latest",
        "fork": "current",
    }  # pyright: ignore
    conversation_without_policy: DocumentCreate = {
        "id": document_id,
        "kind": "test",
        "scope": {"kind": "conversation", "conversationId": conversation_id},
    }  # pyright: ignore
    task_create_with_policy: DocumentCreate = {
        "id": document_id,
        "kind": "test",
        "scope": {"kind": "task", "taskId": task_id},
        "history": "latest",
        "fork": "initial",
    }  # pyright: ignore
    create_with_sequence: DocumentCreate = {
        "id": document_id,
        "kind": "test",
        "scope": {"kind": "session"},
        "createdAt": seq,
    }  # pyright: ignore
    base_with_ops: DocumentContent = {"kind": "base", "version": 1, "value": {}, "ops": []}  # pyright: ignore
    delta_with_value: DocumentContent = {"kind": "delta", "version": 1, "ops": [], "value": {}}  # pyright: ignore
    create_with_delta: StorageWrite = {
        "type": "document.create",
        "record": {"id": document_id, "kind": "test", "scope": {"kind": "session"}},
        "content": {"kind": "delta", "version": 1, "ops": []},
    }  # pyright: ignore
    completed_with_error: TaskOutcome[int] = {
        "status": "completed",
        "result": 1,
        "error": {"message": "impossible"},
    }  # pyright: ignore
    queued_with_entry: SubmissionRecord = {
        "id": submission_id,
        "conversationId": conversation_id,
        "type": "input",
        "status": "queued",
        "entry": entry_id,
    }  # pyright: ignore
    input_without_answer: SubmissionRecord = {
        "id": submission_id,
        "conversationId": conversation_id,
        "type": "input",
        "status": "done",
        "entry": entry_id,
    }  # pyright: ignore
    write_with_answer: SubmissionRecord = {
        "id": submission_id,
        "conversationId": conversation_id,
        "type": "write",
        "status": "done",
        "entry": entry_id,
        "answer": answer_id,
    }  # pyright: ignore
    submission_create_with_id: SubmissionCreate = {
        "id": submission_id,
        "conversationId": conversation_id,
        "type": "write",
        "status": "queued",
    }  # pyright: ignore
    del missing_replacement, omission_with_messages, pending_with_outcome, terminal_with_checkpoint
    del session_with_history, conversation_without_policy, task_create_with_policy, create_with_sequence
    del base_with_ops, delta_with_value, create_with_delta, completed_with_error, queued_with_entry
    del input_without_answer, write_with_answer, submission_create_with_id


# A task with narrowed input, several phases, and custom hooks.


class StepperInput(TypedDict):
    steps: int


class StepperResult(TypedDict):
    ran: int


class SkipDecision(TypedDict):
    skip: bool


def before_step(step: int, api: HookApi, context: Context) -> SkipDecision | None:
    return {"skip": True} if step > 1 else None


def before_step_wrong(step: str, api: HookApi, context: Context) -> None:
    return None


class StepperHooks(TypedDict, total=False):
    before_step: Callable[[int, HookApi, Context], SkipDecision | None]


async def plan(
    task: RunningTask[StepperInput, dict[str, Any], StepperResult],
    runtime: TaskRuntime[StepperInput, dict[str, Any], StepperResult, StepperHooks],
    context: Context,
) -> None:
    assert_type(task["input"]["steps"], int)
    await runtime.commit(lambda tx, current: {"status": "running", "checkpoint": {"phase": "run", "step": 0}}, context)


async def run(
    task: RunningTask[StepperInput, dict[str, Any], StepperResult],
    runtime: TaskRuntime[StepperInput, dict[str, Any], StepperResult, StepperHooks],
    context: Context,
) -> None:
    await runtime.hooks.each("before_step", lambda handler: None)
    await runtime.commit(
        lambda tx, current: {"status": "terminal", "outcome": {"status": "completed", "result": {"ran": 1}}},
        context,
    )


async def abort(
    task: RunningTask[StepperInput, dict[str, Any], StepperResult],
    runtime: TaskRuntime[StepperInput, dict[str, Any], StepperResult, StepperHooks],
    context: Context,
) -> None:
    return None


Stepper: Task[StepperInput, dict[str, Any], StepperResult, StepperHooks] = define_task(
    name="test.stepper",
    version=1,
    initial=lambda input: {"phase": "plan", "steps": input["steps"]},
    phases={"plan": plan, "run": run},
    abort=abort,
)


def types_submissions_waits_and_extensions() -> None:
    input_draft: SubmissionDraft = {"type": "input", "content": "hi", "whenBusy": "steer"}
    write_draft: SubmissionDraft = {"type": "write", "entry": {"kind": "note"}}
    registration = hook(Stepper, {"before_step": before_step})
    extension = define_extension("stepper", tasks=[Stepper], hooks=[registration])
    input_with_model: SubmissionDraft = {
        "type": "input",
        "content": "hi",
        "entry": {"kind": "note", "model": []},
    }  # pyright: ignore
    write_with_content: SubmissionDraft = {"type": "write", "entry": {"kind": "note"}, "content": "hi"}  # pyright: ignore
    write_when_busy: SubmissionDraft = {"type": "write", "entry": {"kind": "note"}, "whenBusy": "steer"}  # pyright: ignore
    wrong_hook = hook(Stepper, {"before_step": before_step_wrong})  # pyright: ignore
    unknown_hook = hook(Stepper, {"after_step": before_step})  # pyright: ignore
    del input_draft, write_draft, extension, input_with_model, write_with_content, write_when_busy
    del wrong_hook, unknown_hook


async def typed_waits(
    harness: Harness,
    conversation: Conversation,
    runtime: TaskRuntime[None, dict[str, Any], None, StepperHooks],
    context: Context,
) -> None:
    id = await conversation.commit(
        lambda tx: tx.create_task(Stepper, {"steps": 2}, {"ownership": {"kind": "conversation"}}), context
    )
    assert_type(id, TaskId[StepperResult])
    settled = await harness.wait_for_task(id, context)
    assert_type(settled, SettledTask[StepperResult])
    outcome = settled["state"]["outcome"]
    if outcome["status"] == "completed":
        assert_type(outcome["result"], StepperResult)
    assert_type(await runtime.wait_for_task(id, context), SettledTask[StepperResult])
    compaction = await conversation.compact(None, context)
    assert_type(compaction, TaskId[CompactionResult])
    assert_type(await harness.wait_for_task(compaction, context), SettledTask[CompactionResult])
    await conversation.commit(
        lambda tx: tx.create_task(Stepper, {"steps": "two"}, {"ownership": {"kind": "conversation"}}),  # pyright: ignore
        context,
    )


def memory_storage_is_a_storage() -> None:
    storage: Storage = MemoryStorage()
    del storage
