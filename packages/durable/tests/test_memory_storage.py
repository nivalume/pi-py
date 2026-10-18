"""Port of ``test/memory-storage.test.ts``, plus smoke tests of the memory storage.

The storage conformance suite (``testing/storage-conformance.ts``) that upstream registers here is
ported separately and runs against :class:`MemoryStorage` too; the smoke tests below cover the
basics until then.
"""

from __future__ import annotations

from typing import Any

import pytest

from pi_ai.types import TextContent, UserMessage
from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable import ROOT_CONVERSATION_ID, MemoryStorage, StorageRequestError, id_from_number, seq_from_number
from pi_durable.types import (
    AnyTaskRecord,
    ConversationId,
    DocumentId,
    EntryId,
    EntryRecord,
    StorageWrite,
    SubmissionRecord,
    TaskId,
)

context = BACKGROUND_CONTEXT


async def test_does_not_expose_retained_state_through_a_prepared_commit() -> None:
    storage = MemoryStorage()
    await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)
    entry_id: EntryId = id_from_number(2)
    data = {"nested": [1]}
    prepared = storage.prepare_commit(
        [
            {
                "type": "entry",
                "value": {"id": entry_id, "conversationId": ROOT_CONVERSATION_ID, "kind": "test", "data": data},
            }
        ]
    )
    # Upstream freezes the prepared writes; Python cannot, so the writes are detached from the caller's
    # values instead (and documented as never to be mutated).
    exposed = prepared.writes[0]
    assert exposed["type"] == "entry"
    assert exposed["value"]["data"] is not data
    data["nested"].append(2)

    assert prepared.apply() == seq_from_number(2)
    assert prepared.apply() == seq_from_number(2)
    stored = await storage.entry(entry_id, context)
    assert stored is not None
    assert stored["entry"]["data"] == {"nested": [1]}


def entry(id: int, conversation_id: int, kind: str = "test", **fields: Any) -> EntryRecord:
    return {"id": id_from_number(id), "conversationId": id_from_number(conversation_id), "kind": kind, **fields}


def pending_task(id: int, conversation_id: int, **fields: Any) -> AnyTaskRecord:
    return {
        "id": id_from_number(id),
        "conversationId": id_from_number(conversation_id),
        "kind": "test.task",
        "version": 1,
        "input": None,
        "background": False,
        "abortRequested": False,
        "state": {"status": "pending", "checkpoint": {"phase": "start"}},
        **fields,
    }


async def create_root(storage: MemoryStorage) -> ConversationId:
    await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)
    return ROOT_CONVERSATION_ID


async def test_mints_ids_above_every_stored_id_and_owns_one_namespace() -> None:
    storage = MemoryStorage()
    root = await create_root(storage)
    assert await storage.mint_id() == 2
    await storage.commit([{"type": "entry", "value": entry(100, root)}], context)
    assert await storage.mint_id() == 101
    with pytest.raises(RuntimeError, match="ID 100 already belongs to entry"):
        await storage.commit([{"type": "task", "value": pending_task(100, root)}], context)
    with pytest.raises(RuntimeError, match=f"ID {root} already belongs to conversation"):
        await storage.commit([{"type": "conversation", "value": {"id": root}}], context)
    await storage.commit([{"type": "entry", "value": entry(2**53 - 1, root, "last-id")}], context)
    with pytest.raises(RuntimeError, match="ID space is exhausted"):
        await storage.mint_id()


async def test_scans_fork_aware_history_in_both_orders_with_cursors() -> None:
    storage = MemoryStorage()
    root = await create_root(storage)
    await storage.commit([{"type": "entry", "value": entry(id, root)} for id in (2, 3, 4)], context)
    child: ConversationId = id_from_number(5)
    await storage.commit(
        [
            {
                "type": "conversation",
                "value": {"id": child, "parent": {"conversationId": root, "at": id_from_number(3)}},
            },
            {"type": "entry", "value": entry(6, child, head=id_from_number(6))},
            {"type": "entry", "value": entry(7, child)},
        ],
        context,
    )

    first = await storage.scan_entries({"conversationId": child}, 2, None, context)
    assert [item["id"] for item in first["items"]] == [7, 6]
    second = await storage.scan_entries({"conversationId": child}, 2, first.get("next"), context)
    assert [item["id"] for item in second["items"]] == [3, 2]
    assert "next" not in second

    ascending = await storage.scan_entries({"conversationId": child, "order": "ascending"}, 3, None, context)
    assert [item["id"] for item in ascending["items"]] == [2, 3, 6]
    rest = await storage.scan_entries({"conversationId": child}, 3, ascending.get("next"), context)
    assert [item["id"] for item in rest["items"]] == [7]
    with pytest.raises(StorageRequestError, match="cursor"):
        await storage.scan_entries({"conversationId": child, "order": "descending"}, 3, ascending.get("next"), context)

    assert await storage.visible_entry(child, id_from_number(4), context) is None
    visible = await storage.visible_entry(child, id_from_number(3), context)
    assert visible is not None and visible["entry"]["id"] == 3
    assert (await storage.entry(id_from_number(4), context)) is not None
    with pytest.raises(StorageRequestError, match="Unknown conversation"):
        await storage.visible_entry(id_from_number(999), id_from_number(2), context)

    marker = await storage.find_latest_head_marker(child, None, context)
    assert marker is not None and marker["id"] == 6
    assert await storage.find_latest_head_marker(child, id_from_number(5), context) is None


async def test_indexes_tasks_by_status_and_submissions_by_request() -> None:
    storage = MemoryStorage()
    root = await create_root(storage)
    await storage.commit([{"type": "task", "value": pending_task(2, root)}], context)
    running = pending_task(2, root, state={"status": "running", "checkpoint": {"phase": "start"}})
    await storage.commit([{"type": "task", "value": running}], context)
    assert (await storage.scan_tasks({"status": "pending"}, 10, None, context))["items"] == []
    assert [task["id"] for task in (await storage.scan_tasks({"status": "running"}, 10, None, context))["items"]] == [2]

    submission: SubmissionRecord = {
        "id": id_from_number(3),
        "conversationId": root,
        "type": "input",
        "status": "queued",
        "requestId": "r1",
    }
    await storage.commit([{"type": "submission", "value": submission}], context)
    found = await storage.submission_by_request(root, "r1", context)
    assert found == submission and found is not submission
    placed: SubmissionRecord = {**submission, "status": "placed", "entry": id_from_number(4)}  # type: ignore[typeddict-item]
    await storage.commit([{"type": "submission", "value": placed}], context)
    assert (await storage.scan_submissions({"status": "queued"}, 10, None, context))["items"] == []
    assert (await storage.submission_by_request(root, "r1", context)) == placed


async def test_round_trips_messages_through_their_stored_json() -> None:
    storage = MemoryStorage()
    root = await create_root(storage)
    message = UserMessage([TextContent("hi")], timestamp=1)
    await storage.commit([{"type": "entry", "value": entry(2, root, "pi.user", model=[message])}], context)
    stored = await storage.entry(id_from_number(2), context)
    assert stored is not None
    model = stored["entry"]["model"]
    assert model == [message]
    assert model[0] is not message


async def test_materializes_documents_from_bases_and_deltas() -> None:
    storage = MemoryStorage()
    root = await create_root(storage)
    doc: DocumentId = id_from_number(2)
    created = await storage.commit(
        [
            {
                "type": "document.create",
                "record": {
                    "id": doc,
                    "kind": "test.doc",
                    "scope": {"kind": "conversation", "conversationId": root},
                    "history": "rewindable",
                    "fork": "asOf",
                },
                "content": {"kind": "base", "version": 1, "value": {"count": 1}},
            }
        ],
        context,
    )
    changed = await storage.commit(
        [
            {
                "type": "document.change",
                "id": doc,
                "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 2)]},
            }
        ],
        context,
    )
    current = await storage.document(doc, "current", context)
    assert current is not None
    assert current["value"] == {"count": 2}
    assert current["deltasSinceBase"] == 1
    assert current["record"]["createdAt"] == created
    historical = await storage.document(doc, created, context)
    assert historical is not None and historical["value"] == {"count": 1}

    address: Any = {"kind": "test.doc", "scope": {"kind": "conversation", "conversationId": root}}
    assert (await storage.find_document(address, "current", context)) == current["record"]

    child: ConversationId = id_from_number(3)
    copy: list[StorageWrite] = [
        {"type": "conversation", "value": {"id": child}},
        {
            "type": "document.copy",
            "record": {
                "id": id_from_number(4),
                "kind": "test.doc",
                "scope": {"kind": "conversation", "conversationId": child},
                "history": "rewindable",
                "fork": "asOf",
            },
            "source": {"id": doc, "at": created},
        },
    ]
    await storage.commit(copy, context)
    copied = await storage.document(id_from_number(4), "current", context)
    assert copied is not None and copied["value"] == {"count": 1}

    retired = await storage.commit([{"type": "document.retire", "id": doc}], context)
    assert await storage.document(doc, "current", context) is None
    assert await storage.find_document(address, "current", context) is None
    at_change = await storage.find_document(address, changed, context)
    assert at_change is not None and at_change.get("retiredAt") == retired
    with pytest.raises(RuntimeError, match="is retired"):
        await storage.commit([{"type": "document.retire", "id": doc}], context)


async def test_rejects_historical_reads_of_current_only_documents() -> None:
    storage = MemoryStorage()
    await create_root(storage)
    doc: DocumentId = id_from_number(2)
    seq = await storage.commit(
        [
            {
                "type": "document.create",
                "record": {"id": doc, "kind": "test.session", "scope": {"kind": "session"}},
                "content": {"kind": "base", "version": 1, "value": {}},
            }
        ],
        context,
    )
    with pytest.raises(StorageRequestError, match="does not retain historical content"):
        await storage.document(doc, seq, context)
    with pytest.raises(RuntimeError, match="version transition requires a base"):
        await storage.commit(
            [{"type": "document.change", "id": doc, "content": {"kind": "delta", "version": 2, "ops": []}}], context
        )
    page = await storage.scan_documents({"scope": {"kind": "session"}, "at": "current"}, 10, None, context)
    assert [record["id"] for record in page["items"]] == [doc]


async def test_rejects_every_operation_after_close() -> None:
    storage = MemoryStorage()
    await create_root(storage)
    await storage.close(context)
    with pytest.raises(RuntimeError, match="closed"):
        await storage.conversation(ROOT_CONVERSATION_ID, context)
    with pytest.raises(RuntimeError, match="closed"):
        await storage.commit([], context)
    with pytest.raises(RuntimeError, match="closed"):
        await storage.mint_id()


def test_ids_are_plain_ints() -> None:
    task: TaskId[int] = TaskId(4)
    assert type(task) is int
