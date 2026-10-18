"""Port of ``test/session-tables.test.ts``."""

from __future__ import annotations

from typing import Any

import pytest
from durable_session_support import (
    StorageBackend,
    context,
    create_conversation,
    document_changes,
    flush,
    open_test_session,
)

from pi_chord.json import UNDEFINED
from pi_durable import ReadAfterWrite, define_doc, define_doc_family, define_task
from pi_durable.ids import id_from_number


async def _noop(*_args: Any) -> None:
    return None


WorkTask = define_task(
    name="test.work",
    version=1,
    initial=lambda _input: {"phase": "start"},
    phases={"start": _noop, "next": _noop},
    abort=_noop,
)
ProgressDoc = define_doc(kind="test.progress", version=1, scope="task", initial=lambda: {"lines": []})
StepDoc = define_doc_family(kind="test.step", version=1, scope="task", initial=lambda _seed: {"lines": []})
NotesDoc = define_doc(
    kind="test.notes", version=1, scope="conversation", history="rewindable", fork="asOf", initial=lambda: {"text": ""}
)


def terminal(task: Any) -> Any:
    return {
        "id": task["id"],
        "conversationId": task["conversationId"],
        "kind": task["kind"],
        "version": task["version"],
        "input": task["input"],
        "background": task["background"],
        "abortRequested": task["abortRequested"],
        "state": {"status": "terminal", "outcome": {"status": "completed", "result": {"ok": True}}},
    }


async def create_task(session: Any, conversation_id: Any, with_document: bool = False) -> Any:
    async def change(tx: Any) -> Any:
        task_id = await tx.create_task(
            WorkTask, {"path": "a"}, {"ownership": {"kind": "conversation"}, "conversationId": conversation_id}
        )
        if with_document:
            (await tx.doc(ProgressDoc, task_id))["lines"].append("started")
        return task_id

    return await session.commit(change, context)


async def test_allows_table_reads_only_before_the_first_table_write(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)

    async def change(tx: Any) -> None:
        assert await tx.conversation(conversation_id) == {"id": conversation_id}
        assert await tx.scan_conversations({}, 1) == {"items": [{"id": conversation_id}]}
        assert await tx.scan_tasks({"conversationId": conversation_id}, 10) == {"items": []}
        assert await tx.scan_entries({"conversationId": conversation_id}, 10) == {"items": []}
        await tx.append_entry(conversation_id, {"kind": "note"})
        with pytest.raises(ReadAfterWrite):
            await tx.conversation(conversation_id)
        with pytest.raises(ReadAfterWrite, match=r"Tx\.task\(\) cannot read tables after the first table write"):
            await tx.task(id_from_number(1))
        with pytest.raises(ReadAfterWrite):
            await tx.entry(id_from_number(1))
        with pytest.raises(ReadAfterWrite):
            await tx.scan_conversations({}, 10)
        with pytest.raises(ReadAfterWrite):
            await tx.scan_entries({"conversationId": conversation_id}, 10)
        # Document access remains available after table writes.
        (await tx.doc(NotesDoc, conversation_id))["text"] = "after write"

    await session.commit(change, context)
    assert await session.snapshot(NotesDoc, conversation_id, context) == {"text": "after write"}


async def test_passes_caller_selected_limits_and_cursors_through_table_scans(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)
    ids = [await create_conversation(session) for _ in range(3)]

    async def change(tx: Any) -> None:
        first = await tx.scan_conversations({}, 2)
        assert [item["id"] for item in first["items"]] == ids[:2]
        assert first.get("next") is not None
        second = await tx.scan_conversations({}, 2, first["next"])
        assert [item["id"] for item in second["items"]] == ids[2:]
        assert second.get("next") is None

    await session.commit(change, context)


async def test_treats_synchronous_set_task_as_the_first_table_write(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    task_id = await create_task(session, conversation_id)

    async def change(tx: Any) -> None:
        task = await tx.task(task_id)
        tx.set_task(task)
        with pytest.raises(ReadAfterWrite):
            await tx.task(task_id)

    await session.commit_with(change, context)


async def test_creates_conversations_entries_and_tasks_with_minted_ids(storage_backend: StorageBackend) -> None:
    storage, session, publications = await open_test_session(storage_backend)

    async def change(tx: Any) -> Any:
        conversation = await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        first = await tx.append_entry(conversation["id"], {"kind": "note", "data": "one"})
        headed = await tx.append_entry(conversation["id"], {"kind": "summary", "head": "self"})
        task = await tx.create_task(
            WorkTask,
            {"path": "x"},
            {"ownership": {"kind": "conversation"}, "conversationId": conversation["id"], "background": True},
        )
        return {"conversation": conversation, "first": first, "headed": headed, "task": task}

    created = await session.commit(change, context)
    ids = [created["conversation"]["id"], created["first"]["id"], created["headed"]["id"], created["task"]]
    assert len(set(ids)) == 4
    assert created["headed"]["head"] == created["headed"]["id"]
    assert created["first"] == {
        "id": created["first"]["id"],
        "conversationId": created["conversation"]["id"],
        "kind": "note",
        "data": "one",
    }
    assert (await storage.entry(created["headed"]["id"], context))["entry"] == created["headed"]
    assert await storage.task(created["task"], context) == {
        "id": created["task"],
        "conversationId": created["conversation"]["id"],
        "kind": "test.work",
        "version": 1,
        "input": {"path": "x"},
        "background": True,
        "abortRequested": False,
        "state": {"status": "pending", "checkpoint": {"phase": "start"}},
    }
    await flush()
    changes = publications[-1]["changes"]
    assert len(changes) == 4
    admitted = storage.admitted_commits[-1]
    for published in changes:
        assert any(write is published for write in admitted)
    assert sorted(published["type"] for published in changes) == ["conversation", "entry", "entry", "task"]
    assert next(c for c in changes if c["type"] == "conversation")["value"] is created["conversation"]
    entries = [c["value"] for c in changes if c["type"] == "entry"]
    assert any(entry is created["first"] for entry in entries)
    assert any(entry is created["headed"] for entry in entries)

    with pytest.raises(TypeError, match="requires options.conversationId"):
        await session.commit(
            lambda tx: tx.create_task(WorkTask, {"path": "x"}, {"ownership": {"kind": "conversation"}}), context
        )
    with pytest.raises(RuntimeError, match="Conversation 12345 does not exist"):
        await session.commit(lambda tx: tx.append_entry(id_from_number(12345), {"kind": "note"}), context)


async def test_creates_a_conversation_with_an_explicitly_staged_task_owner(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)

    async def change(tx: Any) -> Any:
        supervisor_id = await tx.create_task(
            WorkTask,
            {"path": "background"},
            {"ownership": {"kind": "conversation"}, "conversationId": parent_id, "background": True},
        )
        child = await tx.create_conversation({"ownership": {"kind": "task", "taskId": supervisor_id}})
        return supervisor_id, child

    supervisor_id, child = await session.commit(change, context)
    assert child["owner"] == {"conversationId": parent_id, "taskId": supervisor_id}
    assert await storage.conversation(child["id"], context) == child

    async def move(tx: Any) -> None:
        supervisor = await tx.task(supervisor_id)
        tx.set_task({**supervisor, "conversationId": child["id"]})

    with pytest.raises(RuntimeError, match=f"Task {supervisor_id} cannot change conversations"):
        await session.commit_with(move, context)
    assert (await storage.task(supervisor_id, context))["conversationId"] == parent_id


async def test_rejects_missing_terminal_and_abort_marked_conversation_owners_atomically(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    moved: list[Any] = []

    async def move_staged(tx: Any) -> None:
        task_id = await tx.create_task(
            WorkTask, {"path": "move"}, {"ownership": {"kind": "conversation"}, "conversationId": parent_id}
        )
        moved.append(task_id)
        tx.set_task(
            {
                "id": task_id,
                "conversationId": id_from_number(998),
                "kind": WorkTask.definition.name,
                "version": WorkTask.definition.version,
                "input": {"path": "move"},
                "background": False,
                "abortRequested": False,
                "state": {"status": "pending", "checkpoint": {"phase": "start"}},
            }
        )

    with pytest.raises(RuntimeError, match="cannot change conversations"):
        await session.commit_with(move_staged, context)
    assert await storage.task(moved[0], context) is None

    missing_task_id = id_from_number(999)
    with pytest.raises(RuntimeError, match="Conversation owner task 999 does not exist"):
        await session.commit(
            lambda tx: tx.create_conversation({"ownership": {"kind": "task", "taskId": missing_task_id}}), context
        )

    rejected_child: list[Any] = []

    async def aborting(tx: Any) -> None:
        supervisor_id = await tx.create_task(
            WorkTask, {"path": "aborting"}, {"ownership": {"kind": "conversation"}, "conversationId": parent_id}
        )
        rejected_child.append(
            (await tx.create_conversation({"ownership": {"kind": "task", "taskId": supervisor_id}}))["id"]
        )
        tx.set_task(
            {
                "id": supervisor_id,
                "conversationId": parent_id,
                "kind": WorkTask.definition.name,
                "version": WorkTask.definition.version,
                "input": {"path": "aborting"},
                "background": False,
                "abortRequested": True,
                "state": {"status": "pending", "checkpoint": {"phase": "start"}},
            }
        )

    with pytest.raises(RuntimeError, match="is abort-marked"):
        await session.commit_with(aborting, context)
    assert len(rejected_child) == 1
    assert await storage.conversation(rejected_child[0], context) is None

    async def terminal_owner(tx: Any) -> None:
        supervisor_id = await tx.create_task(
            WorkTask, {"path": "terminal"}, {"ownership": {"kind": "conversation"}, "conversationId": parent_id}
        )
        await tx.create_conversation({"ownership": {"kind": "task", "taskId": supervisor_id}})
        tx.set_task(
            {
                "id": supervisor_id,
                "conversationId": parent_id,
                "kind": WorkTask.definition.name,
                "version": WorkTask.definition.version,
                "input": {"path": "terminal"},
                "background": False,
                "abortRequested": False,
                "state": {"status": "terminal", "outcome": {"status": "completed", "result": {"ok": True}}},
            }
        )

    with pytest.raises(RuntimeError, match="is terminal"):
        await session.commit_with(terminal_owner, context)

    terminal_owner_id = await create_task(session, parent_id)

    async def finish(tx: Any) -> None:
        tx.set_task(terminal(await tx.task(terminal_owner_id)))

    await session.commit_with(finish, context)
    with pytest.raises(RuntimeError, match="is terminal"):
        await session.commit(
            lambda tx: tx.create_conversation({"ownership": {"kind": "task", "taskId": terminal_owner_id}}), context
        )


async def test_takes_ownership_of_table_json_and_rejects_non_strict_values(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    payload = {"nested": {"value": 1}}

    async def change(tx: Any) -> Any:
        created = await tx.append_entry(conversation_id, {"kind": "data", "data": payload})
        payload["nested"]["value"] = 2
        return created

    entry = await session.commit(change, context)
    assert (await storage.entry(entry["id"], context))["entry"]["data"] == {"nested": {"value": 1}}
    omitted = await session.commit(
        lambda tx: tx.append_entry(conversation_id, {"kind": "omitted", "data": UNDEFINED}), context
    )
    assert "data" not in omitted
    assert "data" not in (await storage.entry(omitted["id"], context))["entry"]

    commits = len(storage.commits)
    with pytest.raises(TypeError, match="strict JSON"):
        await session.commit(
            lambda tx: tx.append_entry(conversation_id, {"kind": "invalid", "data": float("nan")}), context
        )
    assert len(storage.commits) == commits


async def test_replaces_task_records_completely(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    task_id = await create_task(session, conversation_id)

    async def change(tx: Any) -> None:
        task = await tx.task(task_id)
        tx.set_task(
            {
                "id": task["id"],
                "conversationId": task["conversationId"],
                "kind": task["kind"],
                "version": task["version"],
                "input": task["input"],
                "background": task["background"],
                "abortRequested": task["abortRequested"],
                "state": {"status": "running", "checkpoint": {"phase": "next", "step": 2}},
                "memos": {"choice": "b"},
            }
        )

    await session.commit_with(change, context)
    stored = await storage.task(task_id, context)
    assert stored["state"] == {"status": "running", "checkpoint": {"phase": "next", "step": 2}}
    assert stored["memos"] == {"choice": "b"}


async def test_creates_a_task_and_then_its_document_in_one_transaction_without_read_after_write(
    storage_backend: StorageBackend,
) -> None:
    _, session, publications = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)

    async def change(tx: Any) -> Any:
        created_task_id = await tx.create_task(
            WorkTask, {"path": "a"}, {"ownership": {"kind": "conversation"}, "conversationId": conversation_id}
        )
        # Validation uses the candidate task record, not a caller table read.
        (await tx.doc(ProgressDoc, created_task_id))["lines"].append("created")
        (await tx.doc(StepDoc, created_task_id, "one", None))["lines"].append("step")
        return created_task_id

    task_id = await session.commit(change, context)
    assert await session.snapshot(ProgressDoc, task_id, context) == {"lines": ["created"]}
    await flush()
    publication = publications[-1]
    documents = document_changes(publication)
    assert any(c["type"] == "task" and c["value"]["id"] == task_id for c in publication["changes"])
    assert len(documents) == 2
    # Task documents derive their conversation from the task record.
    for document in documents:
        assert document["conversationId"] == conversation_id

    async def later(tx: Any) -> None:
        await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        (await tx.doc(ProgressDoc, task_id))["lines"].append("committed task")

    await session.commit(later, context)
    await flush()
    assert document_changes(publications[-1])[0]["conversationId"] == conversation_id


async def test_rejects_task_documents_after_a_terminal_candidate(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    task_id = await create_task(session, conversation_id, True)

    async def change(tx: Any) -> None:
        task = await tx.task(task_id)
        progress = await tx.doc(ProgressDoc, task_id)
        tx.set_task(terminal(task))
        with pytest.raises(RuntimeError, match=f"Task {task_id} is terminal"):
            await tx.doc(ProgressDoc, task_id)
        with pytest.raises(RuntimeError, match=f"Task {task_id} is terminal"):
            await tx.doc(StepDoc, task_id, "late", None)
        with pytest.raises(RuntimeError, match="terminal candidate"):
            tx.set_task(task)
        progress["lines"].append("final")

    await session.commit_with(change, context)

    async def acquire(tx: Any) -> None:
        await tx.doc(ProgressDoc, task_id)

    with pytest.raises(RuntimeError, match=f"Task {task_id} is terminal"):
        await session.commit(acquire, context)


async def test_retires_task_documents_at_terminal_settlement_including_documents_created_in_the_same_transaction(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    task_id = await create_task(session, conversation_id, True)

    async def step(tx: Any) -> None:
        (await tx.doc(StepDoc, task_id, "committed", None))["lines"].append("x")

    await session.commit(step, context)
    await flush()
    published = len(publications)

    async def finish(tx: Any) -> None:
        task = await tx.task(task_id)
        (await tx.doc(StepDoc, task_id, "new", None))["lines"].append("created then retired")
        tx.set_task(terminal(task))

    await session.commit_with(finish, context)
    writes: list[Any] = storage.commits[-1]
    assert len(writes) == 5
    assert len([w for w in writes if w["type"] == "task"]) == 1
    creations = [w for w in writes if w["type"] == "document.create"]
    retirements = [w for w in writes if w["type"] == "document.retire"]
    assert len(creations) == 1
    assert len(retirements) == 3
    assert creations[0]["record"]["id"] in [w["id"] for w in retirements]
    await flush()
    assert len(publications) == published + 1
    publication = publications[-1]
    documents = document_changes(publication)
    assert len([c for c in publication["changes"] if c["type"] == "task"]) == 1
    assert [document["value"] for document in documents] == [None, None, None]
    for document in documents:
        assert document["ops"] == []
        assert document["conversationId"] == conversation_id
    assert await session.snapshot(ProgressDoc, task_id, context) is None
    assert await session.snapshot(StepDoc, task_id, "committed", context) is None
    alive = await storage.scan_documents(
        {"scope": {"kind": "task", "taskId": task_id}, "at": "current"}, 10, None, context
    )
    assert list(alive["items"]) == []

    async def again(tx: Any) -> None:
        tx.set_task(terminal(await tx.task(task_id)))

    with pytest.raises(RuntimeError, match=f"Task {task_id} is already terminal"):
        await session.commit_with(again, context)


async def test_validates_document_owners(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)

    async def task_document(tx: Any) -> None:
        await tx.doc(ProgressDoc, id_from_number(4242))

    async def conversation_document(tx: Any) -> None:
        await tx.doc(NotesDoc, id_from_number(4242))

    with pytest.raises(RuntimeError, match="Task 4242 does not exist"):
        await session.commit(task_document, context)
    with pytest.raises(RuntimeError, match="Conversation 4242 does not exist"):
        await session.commit(conversation_document, context)
