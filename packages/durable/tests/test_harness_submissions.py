"""Port of ``test/harness-submissions.test.ts``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import all_entries, chat_setup, open_chat, unanswered
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, resolve

from pi_ai import AbortController, faux_assistant_message
from pi_ai.types import UserMessage
from pi_chord.context import Context, with_abort_signal
from pi_durable import AssistantEntry, ConversationBusy, GenerationTask, LiveDoc, MemoryStorage, UserEntry, define_entry
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


def controlled() -> ControlledStorage:
    return ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])


async def test_appends_an_idle_write_and_settles_it_done_without_a_turn() -> None:
    opened = await open_chat(controlled(), chat_setup())
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "write", "entry": {"kind": "note", "data": {"text": "x"}}}, context)
    settled = await submission.wait(context)
    assert settled == {
        "id": submission.id,
        "conversationId": root.id,
        "type": "write",
        "status": "done",
        "entry": settled["entry"],  # type: ignore[typeddict-item]
    }
    assert isinstance(settled["entry"], int)  # type: ignore[typeddict-item]
    entries = await all_entries(root)
    assert entries == [{"id": settled["entry"], "conversationId": root.id, "kind": "note", "data": {"text": "x"}}]  # type: ignore[typeddict-item]
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    tasks = await harness.commit(lambda tx: tx.scan_tasks({"conversationId": root.id}, 10), context)
    assert tasks["items"] == []
    await harness.close(context)


async def test_places_idle_input_and_rejects_busy_input_with_when_busy_reject_without_writing() -> None:
    storage = controlled()
    setup = chat_setup()
    setup.now = lambda: 42
    busy = unanswered()
    setup.faux.set_responses([busy.step])
    opened = await open_chat(storage, setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    await busy.reached
    record = await submission.status(context)
    assert record["status"] == "placed"
    entry = await root.commit(lambda tx: tx.entry(UserEntry, record["entry"]), context)  # type: ignore[typeddict-item]
    assert entry["model"] == [UserMessage("hi", timestamp=42)]
    live = await harness.snapshot(LiveDoc, root.id, context)
    assert live is not None
    assert live["run"]["inputs"] == [submission.id]
    task = await harness.get_task(live["run"]["taskId"], context)
    assert task is not None and task["kind"] == "pi.generation"

    commits = len(storage.commits)
    with pytest.raises(ConversationBusy) as caught:
        await root.submit({"type": "input", "content": "again", "whenBusy": "reject"}, context)
    assert caught.value.conversation_id == root.id
    assert len(storage.commits) == commits
    await harness.close(context)


async def test_deduplicates_request_ids_per_conversation_before_any_write() -> None:
    storage = controlled()
    setup = chat_setup()
    busy = unanswered()
    setup.faux.set_responses([busy.step])
    opened = await open_chat(storage, setup)
    harness, root = opened.harness, opened.root
    first = await root.submit({"type": "input", "content": "hi", "requestId": "r1"}, context)
    await busy.reached
    commits = len(storage.commits)
    # Deduplication runs before the busy check.
    again = await root.submit({"type": "input", "content": "different", "requestId": "r1"}, context)
    assert again.id == first.id
    assert len(storage.commits) == commits
    with pytest.raises(RuntimeError, match="Request r1 already identifies a submission of type input"):
        await root.submit({"type": "write", "entry": {"kind": "note"}, "requestId": "r1"}, context)
    status = await first.status(context)
    assert status["requestId"] == "r1" and status["status"] == "placed"

    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    write = await other.submit({"type": "write", "entry": {"kind": "note"}, "requestId": "r1"}, context)
    assert write.id != first.id
    assert (await other.submit({"type": "write", "entry": {"kind": "note"}, "requestId": "r1"}, context)).id == write.id
    await harness.close(context)


async def test_reports_abort_results_and_looks_submissions_up_by_conversation() -> None:
    setup = chat_setup()
    release = deferred()

    async def step(*_args: Any) -> Any:
        await release
        return faux_assistant_message("answer")

    setup.faux.set_responses([step])
    opened = await open_chat(controlled(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    assert await submission.abort(context) == "already_placed"
    assert await harness.abort_submission(submission.id, context, root.id) == "already_placed"
    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    assert await harness.abort_submission(submission.id, context, other.id) == "not_found"
    assert await harness.abort_submission(999_999, context) == "not_found"  # type: ignore[arg-type]
    assert await harness.submission(999_999, context) is None  # type: ignore[arg-type]

    resolve(release)
    await submission.wait(context)
    assert await submission.abort(context) == "settled"
    assert await harness.abort_submission(submission.id, context) == "settled"
    await harness.close(context)


async def test_cancels_only_a_wait_and_rejects_pending_waits_on_close() -> None:
    setup = chat_setup()
    setup.faux.set_responses([unanswered().step])
    opened = await open_chat(controlled(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    controller = AbortController()
    cancelled = submission.wait(with_abort_signal(controller.signal, context))
    pending = submission.wait(context)
    controller.abort(RuntimeError("stop waiting"))
    with pytest.raises(Exception, match="stop waiting"):
        await cancelled
    assert (await submission.status(context))["status"] == "placed"
    await harness.close(context)
    with pytest.raises(Exception, match="Harness is closed"):
        await pending


async def test_rejects_a_wait_whose_submission_read_spans_the_start_of_close() -> None:
    entered = deferred()
    release = deferred()
    hold = [False]

    class HeldReads(ControlledStorage):
        async def submission(self, id: Any, call_context: Context) -> Any:
            if hold[0]:
                resolve(entered)
                await release
            return await super().submission(id, call_context)

    setup = chat_setup()
    setup.faux.set_responses([unanswered().step])
    opened = await open_chat(HeldReads(MemoryStorage(), STORAGE_BACKENDS["memory"]), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    hold[0] = True
    waiting = submission.wait(context)
    await entered
    closing = harness.close(context)
    resolve(release)
    with pytest.raises(Exception, match="Harness is closed"):
        await waiting
    await closing


async def test_reacquires_a_submission_after_reopen_and_settles_it_durably(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    setup = chat_setup()
    # The first process never answers; the reopened one does.
    busy = unanswered()
    setup.faux.set_responses([busy.step, faux_assistant_message("after reopen")])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    id = (await opened.root.submit({"type": "input", "content": "hi", "requestId": "print"}, context)).id
    await busy.reached
    await opened.harness.close(context)

    opened = await open_chat(await open_sqlite_storage(path), setup)
    submission = await opened.harness.submission(id, context)
    assert submission is not None
    assert (await submission.status(context))["status"] == "placed"
    opened.harness.resume()
    settled = await submission.wait(context)
    assert settled["status"] == "done" and settled["type"] == "input"
    await opened.harness.close(context)

    opened = await open_chat(await open_sqlite_storage(path), setup)
    reacquired = await opened.harness.submission(id, context)
    assert reacquired is not None
    assert await reacquired.wait(context) == settled
    again = await opened.root.submit({"type": "input", "content": "hi", "requestId": "print"}, context)
    assert again.id == id
    await opened.harness.close(context)


async def test_enables_scheduling_when_a_caller_submits_or_waits() -> None:
    setup = chat_setup()
    setup.faux.set_responses([faux_assistant_message("answer")])
    opened = await open_chat(controlled(), setup)
    # No resume(): submitting asks for progress.
    submission = await opened.root.submit({"type": "input", "content": "hi"}, context)
    assert (await submission.wait(context))["status"] == "done"
    await opened.harness.close(context)

    passive = await open_chat(controlled(), chat_setup())
    task_id = await passive.root.commit(
        lambda tx: tx.create_task(GenerationTask, {}, {"ownership": {"kind": "conversation"}}), context
    )
    # A committed task alone does not start scheduling; waiting for it does.
    record = await passive.harness.get_task(task_id, context)
    assert record is not None and record["state"]["status"] == "pending"
    assert (await passive.harness.wait_for_task(task_id, context))["state"]["status"] == "terminal"
    await passive.harness.close(context)


async def test_settles_submissions_by_their_current_record_in_the_transaction() -> None:
    opened = await open_chat(controlled(), chat_setup())
    harness, root = opened.harness, opened.root

    async def note(tx: Any) -> Any:
        return (await tx.append_entry(root.id, {"kind": "note"}))["id"]

    entry = await root.commit(note, context)

    def create(record: Any) -> Any:
        async def change(tx: Any) -> Any:
            return (await tx.create_submission(record))["id"]

        return harness.commit_with(change, context)

    queued = await create({"conversationId": root.id, "type": "input", "status": "queued"})
    write = await create({"conversationId": root.id, "type": "write", "status": "queued"})
    answer = {"status": "done", "answer": entry}
    with pytest.raises(Exception, match="is not a placed input"):
        await root.commit(lambda tx: tx.settle_submission(queued, answer), context)
    with pytest.raises(Exception, match="is not a placed input"):
        await root.commit(lambda tx: tx.settle_submission(write, answer), context)
    with pytest.raises(Exception, match="does not exist"):
        await root.commit(lambda tx: tx.settle_submission(999_999, {"status": "unanswered", "reason": "x"}), context)

    # A submission created earlier in the same commit settles; a second settlement leaves the first.
    async def settle_twice(tx: Any) -> Any:
        id = (
            await tx.create_submission({"conversationId": root.id, "type": "input", "status": "placed", "entry": entry})
        )["id"]
        tx.settle_submission(id, answer)
        tx.settle_submission(id, {"status": "unanswered", "reason": "late"})
        return id

    placed = await harness.commit_with(settle_twice, context)
    found = await harness.submission(placed, context)
    assert found is not None
    status = await found.status(context)
    assert status["status"] == "done" and status["entry"] == entry and status["answer"] == entry  # type: ignore[typeddict-item]
    await harness.close(context)


async def test_appends_and_reads_typed_entries_through_tokens() -> None:
    Counter = define_entry("app.counter")
    Marker = define_entry("app.marker")
    opened = await open_chat(controlled(), chat_setup())
    harness, root = opened.harness, opened.root
    counter = await root.commit(lambda tx: tx.append_entry(Counter, root.id, {"data": {"n": 1}}), context)
    assert counter["data"]["n"] == 1
    assert counter == {"id": counter["id"], "conversationId": root.id, "kind": "app.counter", "data": {"n": 1}}
    marker = await root.commit(lambda tx: tx.append_entry(Marker, root.id, {}), context)
    assert marker["kind"] == "app.marker"
    assert await root.commit(lambda tx: tx.entry(Counter, counter["id"]), context) == counter
    assert await root.commit(lambda tx: tx.entry(Marker, counter["id"]), context) is None
    assert await root.commit(lambda tx: tx.entry(Counter, 999_999), context) is None
    assert Counter.is_(counter) is True
    assert AssistantEntry.is_(counter) is False
    assert [UserEntry.kind, AssistantEntry.kind] == ["pi.user", "pi.assistant"]
    await harness.close(context)
