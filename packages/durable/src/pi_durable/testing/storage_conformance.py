"""Runner-independent conformance cases for ``Storage`` backends (port of ``testing/storage-conformance.ts``).

Third-party backends run these to check they behave like the built-in storages: register them with
:func:`~pi_durable.testing.register_storage_conformance` in a pytest module, or iterate
:func:`create_storage_conformance` with any runner.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from pi_chord.context import BACKGROUND_CONTEXT

from ..ids import MAX_SAFE_INTEGER, id_from_number
from ..types import (
    ROOT_CONVERSATION_ID,
    AnyTaskRecord,
    ConversationId,
    Cursor,
    DocumentCreate,
    DocumentId,
    EntryId,
    EntryRecord,
    Page,
    ScanOrder,
    Storage,
    StorageWrite,
    SubmissionQuery,
    SubmissionRecord,
)
from .types import StorageConformanceAssertions, StorageConformanceCase, StorageConformanceOptions

__all__ = ["create_storage_conformance"]

context = BACKGROUND_CONTEXT

_ConformanceTest = Callable[[Storage], Awaitable[None]]


async def _create_root(storage: Storage) -> ConversationId:
    await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)
    return ROOT_CONVERSATION_ID


def _pending_task(id: int, conversation_id: int, phase: str = "ready") -> AnyTaskRecord:
    return {
        "id": id_from_number(id),
        "conversationId": id_from_number(conversation_id),
        "kind": "test.task",
        "version": 1,
        "input": {"value": id},
        "state": {"status": "pending", "checkpoint": {"phase": phase}},
        "background": False,
        "abortRequested": False,
    }


def _entry(id: int, conversation_id: int, kind: str = "message", **extra: Any) -> EntryRecord:
    return cast(EntryRecord, {"id": id, "conversationId": conversation_id, "kind": kind, **extra})


def _ids(page: Page[Any]) -> list[int]:
    return [item["id"] for item in page["items"]]


def _round_trip(cursor: Cursor | None) -> Cursor | None:
    return None if cursor is None else json.loads(json.dumps(cursor))


def create_storage_conformance(options: StorageConformanceOptions) -> Sequence[StorageConformanceCase]:
    """Create the runner-independent cases. ``withStorage`` must call and await its callback exactly once per case."""
    a: StorageConformanceAssertions = options["assertions"]
    with_storage = options["withStorage"]
    cases: list[StorageConformanceCase] = []

    def case(name: str) -> Callable[[_ConformanceTest], _ConformanceTest]:
        def register(test: _ConformanceTest) -> _ConformanceTest:
            cases.append(StorageConformanceCase(name, lambda: with_storage(test)))
            return test

        return register

    @case("reserves ID 1 for the immutable root conversation")
    async def _(storage: Storage) -> None:
        a.strict_equal(await storage.mint_id(), 2)
        a.strict_equal(await _create_root(storage), ROOT_CONVERSATION_ID)
        a.deep_equal(await storage.conversation(ROOT_CONVERSATION_ID, context), {"id": ROOT_CONVERSATION_ID})
        await a.rejects(
            storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context),
            f"ID {ROOT_CONVERSATION_ID} already belongs to conversation",
        )

    @case("commits mixed table writes atomically and rolls all of them back on failure")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        entry_id = await storage.mint_id()
        task_id = await storage.mint_id()
        submission_id = await storage.mint_id()
        task = _pending_task(task_id, root_id)
        input: SubmissionRecord = {
            "id": id_from_number(submission_id),
            "conversationId": root_id,
            "requestId": "request-1",
            "type": "input",
            "status": "placed",
            "entry": id_from_number(entry_id),
        }
        initial_seq = await storage.commit(
            [
                {"type": "entry", "value": _entry(entry_id, root_id, "user", data={"text": "hello"})},
                {"type": "task", "value": task},
                {"type": "submission", "value": input},
            ],
            context,
        )
        a.deep_equal(
            await storage.entry(id_from_number(entry_id), context),
            {"entry": _entry(entry_id, root_id, "user", data={"text": "hello"}), "commitSeq": initial_seq},
        )
        a.deep_equal(await storage.task(id_from_number(task_id), context), task)
        a.deep_equal(await storage.submission(id_from_number(submission_id), context), input)

        transient_entry_id = await storage.mint_id()
        running_task: AnyTaskRecord = {**task, "state": {"status": "running", "checkpoint": {"phase": "effect"}}}
        done_input = cast(SubmissionRecord, {**input, "status": "done", "answer": transient_entry_id})
        await a.rejects(
            storage.commit(
                [
                    {"type": "task", "value": running_task},
                    {"type": "submission", "value": done_input},
                    {"type": "entry", "value": _entry(transient_entry_id, root_id, "assistant")},
                    {"type": "conversation", "value": {"id": root_id}},
                ],
                context,
            ),
            f"ID {root_id} already belongs to conversation",
        )
        a.deep_equal(await storage.task(id_from_number(task_id), context), task)
        a.deep_equal(await storage.submission(id_from_number(submission_id), context), input)
        a.strict_equal(await storage.entry(id_from_number(transient_entry_id), context), None)
        after_rollback_seq = await storage.commit(
            [{"type": "entry", "value": _entry(await storage.mint_id(), root_id, "after-rollback")}], context
        )
        a.greater_than(after_rollback_seq, initial_seq)

    @case("detaches retained writes and every returned record")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        entry_id = id_from_number(await storage.mint_id())
        task_id = id_from_number(await storage.mint_id())
        submission_id = id_from_number(await storage.mint_id())
        entry_data: dict[str, Any] = {"nested": [1, 2]}
        checkpoint: dict[str, Any] = {"phase": "ready", "nested": {"count": 1}}
        detail: dict[str, Any] = {"codes": ["initial"]}
        stored_task: AnyTaskRecord = {
            **_pending_task(task_id, root_id),
            "state": {"status": "pending", "checkpoint": checkpoint},
        }
        stored_input: SubmissionRecord = {
            "id": submission_id,
            "conversationId": root_id,
            "type": "input",
            "status": "unanswered",
            "reason": "failed",
            "detail": detail,
        }
        await storage.commit(
            [
                {"type": "entry", "value": _entry(entry_id, root_id, "note", data=entry_data)},
                {"type": "task", "value": stored_task},
                {"type": "submission", "value": stored_input},
            ],
            context,
        )
        entry_data["nested"].append(3)
        checkpoint["nested"]["count"] = 2
        detail["codes"].append("mutated")

        async def check() -> None:
            stored_entry = await storage.entry(entry_id, context)
            a.deep_equal(None if stored_entry is None else stored_entry["entry"].get("data"), {"nested": [1, 2]})
            task = await storage.task(task_id, context)
            a.deep_equal(
                None if task is None else task["state"],
                {"status": "pending", "checkpoint": {"phase": "ready", "nested": {"count": 1}}},
            )
            submission = await storage.submission(submission_id, context)
            a.deep_equal(None if submission is None else submission.get("detail"), {"codes": ["initial"]})

        await check()
        read_entry = await storage.entry(entry_id, context)
        assert read_entry is not None
        cast(Any, read_entry["entry"])["data"]["nested"].append(9)
        read_task = await storage.task(task_id, context)
        assert read_task is not None
        if read_task["state"]["status"] != "terminal":
            read_task["state"]["checkpoint"]["nested"]["count"] = 9  # type: ignore[typeddict-item]
        read_input = await storage.submission(submission_id, context)
        assert read_input is not None
        read_input["detail"]["codes"].append("read mutation")  # type: ignore[typeddict-item]
        await check()

    @case("detaches prototype-like JSON keys without changing object prototypes")
    async def _(storage: Storage) -> None:
        # Upstream guards JavaScript prototypes; here the same keys must stay ordinary, detached data.
        root_id = await _create_root(storage)
        entry_id = id_from_number(await storage.mint_id())
        data: dict[str, Any] = json.loads(
            '{"__proto__":{"polluted":false},"constructor":{"label":"stored"},"toString":"value"}'
        )
        await storage.commit([{"type": "entry", "value": _entry(entry_id, root_id, "note", data=data)}], context)
        data["__proto__"]["polluted"] = True
        data["constructor"]["label"] = "mutated"
        first = await storage.entry(entry_id, context)
        assert first is not None
        first_read = cast(Any, first["entry"])["data"]
        a.strict_equal(type(first_read), dict)
        a.ok("__proto__" in first_read, "the __proto__ key is kept")
        a.deep_equal(first_read["__proto__"], {"polluted": False})
        a.deep_equal(first_read["constructor"], {"label": "stored"})
        a.strict_equal(first_read["toString"], "value")
        first_read["__proto__"]["polluted"] = True
        second = await storage.entry(entry_id, context)
        assert second is not None
        second_read = cast(Any, second["entry"])["data"]
        a.deep_equal(second_read["__proto__"], {"polluted": False})
        a.deep_equal(second_read["constructor"], {"label": "stored"})
        a.strict_equal(second_read["toString"], "value")

    @case("indexes entries committed out of ID order")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        await storage.commit(
            [
                {"type": "entry", "value": _entry(30, root_id)},
                {"type": "entry", "value": _entry(10, root_id)},
                {"type": "entry", "value": _entry(20, root_id, "marker", head=10)},
            ],
            context,
        )
        a.deep_equal(_ids(await storage.scan_entries({"conversationId": root_id}, 10, None, context)), [30, 20, 10])
        marker = await storage.find_latest_head_marker(root_id, None, context)
        a.strict_equal(None if marker is None else marker["id"], 20)

    @case("continues an entry cursor below its last item after a newer commit")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        oldest_id = await storage.mint_id()
        middle_id = await storage.mint_id()
        newest_id = await storage.mint_id()
        await storage.commit(
            [
                {"type": "entry", "value": _entry(oldest_id, root_id)},
                {"type": "entry", "value": _entry(middle_id, root_id)},
                {"type": "entry", "value": _entry(newest_id, root_id)},
            ],
            context,
        )
        first = await storage.scan_entries({"conversationId": root_id}, 2, None, context)
        a.deep_equal(_ids(first), [newest_id, middle_id])
        appended_id = await storage.mint_id()
        await storage.commit([{"type": "entry", "value": _entry(appended_id, root_id)}], context)
        second = await storage.scan_entries({"conversationId": root_id}, 2, first.get("next"), context)
        a.deep_equal(_ids(second), [oldest_id])
        a.strict_equal(second.get("next"), None)

    @case("paginates conversations by opaque cursor in ascending ID order")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        second_id = await storage.mint_id()
        third_id = await storage.mint_id()
        await storage.commit(
            [
                {"type": "conversation", "value": {"id": id_from_number(third_id)}},
                {"type": "conversation", "value": {"id": id_from_number(second_id)}},
            ],
            context,
        )
        first = await storage.scan_conversations({}, 2, None, context)
        a.deep_equal(_ids(first), [root_id, second_id])
        a.ok(first.get("next") is not None, "Expected value to be defined")
        second = await storage.scan_conversations({}, 2, _round_trip(first.get("next")), context)
        a.deep_equal(_ids(second), [third_id])
        a.strict_equal(second.get("next"), None)

    @case("filters and pages conversations by durable owner edges")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        other_owner_id = id_from_number(await storage.mint_id())
        first_task_id = id_from_number(await storage.mint_id())
        second_task_id = id_from_number(await storage.mint_id())
        first_id = id_from_number(await storage.mint_id())
        second_id = id_from_number(await storage.mint_id())
        third_id = id_from_number(await storage.mint_id())
        await storage.commit(
            [
                {"type": "conversation", "value": {"id": other_owner_id}},
                {
                    "type": "conversation",
                    "value": {"id": first_id, "owner": {"conversationId": root_id, "taskId": first_task_id}},
                },
                {
                    "type": "conversation",
                    "value": {"id": second_id, "owner": {"conversationId": root_id, "taskId": second_task_id}},
                },
                {
                    "type": "conversation",
                    "value": {"id": third_id, "owner": {"conversationId": other_owner_id, "taskId": first_task_id}},
                },
            ],
            context,
        )
        first = await storage.scan_conversations({"ownerConversationId": root_id}, 1, None, context)
        a.deep_equal(_ids(first), [first_id])
        a.ok(first.get("next") is not None, "Expected value to be defined")
        second = await storage.scan_conversations({"ownerConversationId": root_id}, 1, first.get("next"), context)
        a.deep_equal(_ids(second), [second_id])
        a.strict_equal(second.get("next"), None)
        a.deep_equal(
            _ids(await storage.scan_conversations({"ownerTaskId": first_task_id}, 10, None, context)),
            [first_id, third_id],
        )
        a.deep_equal(
            _ids(
                await storage.scan_conversations(
                    {"ownerConversationId": root_id, "ownerTaskId": first_task_id}, 10, None, context
                )
            ),
            [first_id],
        )

    @case("scans deep fork history newest-first through every ancestor cap")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        root_first = id_from_number(await storage.mint_id())
        root_fork_point = id_from_number(await storage.mint_id())
        root_excluded_same_commit = id_from_number(await storage.mint_id())
        root_entries_seq = await storage.commit(
            [
                {"type": "entry", "value": _entry(root_first, root_id)},
                {"type": "entry", "value": _entry(root_fork_point, root_id, "marker", head=root_first)},
                {"type": "entry", "value": _entry(root_excluded_same_commit, root_id)},
            ],
            context,
        )
        child_id = id_from_number(await storage.mint_id())
        await storage.commit(
            [
                {
                    "type": "conversation",
                    "value": {"id": child_id, "parent": {"conversationId": root_id, "at": root_fork_point}},
                }
            ],
            context,
        )
        child_fork_point = id_from_number(await storage.mint_id())
        child_excluded = id_from_number(await storage.mint_id())
        await storage.commit(
            [
                {"type": "entry", "value": _entry(child_fork_point, child_id, "note")},
                {"type": "entry", "value": _entry(child_excluded, child_id)},
            ],
            context,
        )
        root_excluded_later = id_from_number(await storage.mint_id())
        await storage.commit([{"type": "entry", "value": _entry(root_excluded_later, root_id)}], context)
        grandchild_id = id_from_number(await storage.mint_id())
        await storage.commit(
            [
                {
                    "type": "conversation",
                    "value": {"id": grandchild_id, "parent": {"conversationId": child_id, "at": child_fork_point}},
                }
            ],
            context,
        )
        grandchild_head = id_from_number(await storage.mint_id())
        grandchild_tail = id_from_number(await storage.mint_id())
        grandchild_entries_seq = await storage.commit(
            [
                {"type": "entry", "value": _entry(grandchild_head, grandchild_id, "marker", head=grandchild_head)},
                {"type": "entry", "value": _entry(grandchild_tail, grandchild_id)},
            ],
            context,
        )
        child_excluded_later = id_from_number(await storage.mint_id())
        await storage.commit([{"type": "entry", "value": _entry(child_excluded_later, child_id)}], context)

        first = await storage.scan_entries({"conversationId": grandchild_id}, 2, None, context)
        a.deep_equal(_ids(first), [grandchild_tail, grandchild_head])
        second = await storage.scan_entries({"conversationId": grandchild_id}, 2, first.get("next"), context)
        a.deep_equal(_ids(second), [child_fork_point, root_fork_point])
        third = await storage.scan_entries({"conversationId": grandchild_id}, 2, second.get("next"), context)
        a.deep_equal(_ids(third), [root_first])
        a.strict_equal(third.get("next"), None)

        # Oldest first: the root's segment, then each fork's, each capped at the next fork point.
        first_up = await storage.scan_entries({"conversationId": grandchild_id, "order": "ascending"}, 2, None, context)
        a.deep_equal(_ids(first_up), [root_first, root_fork_point])
        second_up = await storage.scan_entries(
            {"conversationId": grandchild_id, "order": "ascending"}, 2, first_up.get("next"), context
        )
        a.deep_equal(_ids(second_up), [child_fork_point, grandchild_head])
        third_up = await storage.scan_entries({"conversationId": grandchild_id}, 2, second_up.get("next"), context)
        a.deep_equal(_ids(third_up), [grandchild_tail])
        a.strict_equal(third_up.get("next"), None)
        a.deep_equal(
            _ids(
                await storage.scan_entries(
                    {
                        "conversationId": grandchild_id,
                        "order": "ascending",
                        "minEntryId": root_fork_point,
                        "maxEntryId": child_fork_point,
                    },
                    10,
                    None,
                    context,
                )
            ),
            [root_fork_point, child_fork_point],
        )
        await a.rejects(
            storage.scan_entries(
                {"conversationId": grandchild_id, "order": "descending"}, 2, first_up.get("next"), context
            ),
            "cursor",
        )

        current_marker = await storage.find_latest_head_marker(grandchild_id, None, context)
        assert current_marker is not None
        a.strict_equal(current_marker["id"], grandchild_head)
        a.strict_equal(current_marker["head"], grandchild_head)
        historical_marker = await storage.find_latest_head_marker(grandchild_id, child_fork_point, context)
        assert historical_marker is not None
        a.strict_equal(historical_marker["id"], root_fork_point)
        a.strict_equal(historical_marker["head"], root_first)
        a.strict_equal(await storage.find_latest_head_marker(grandchild_id, root_first, context), None)

        active_first = await storage.scan_entries(
            {"conversationId": grandchild_id, "minEntryId": current_marker["head"]}, 1, None, context
        )
        a.deep_equal(_ids(active_first), [grandchild_tail])
        a.ok(active_first.get("next") is not None, "Expected value to be defined")
        active_second = await storage.scan_entries(
            {"conversationId": grandchild_id, "minEntryId": current_marker["head"]},
            1,
            active_first.get("next"),
            context,
        )
        a.deep_equal(_ids(active_second), [grandchild_head])
        a.strict_equal(active_second.get("next"), None)
        a.deep_equal(
            _ids(
                await storage.scan_entries(
                    {
                        "conversationId": grandchild_id,
                        "minEntryId": historical_marker["head"],
                        "maxEntryId": child_fork_point,
                    },
                    10,
                    None,
                    context,
                )
            ),
            [child_fork_point, root_fork_point, root_first],
        )

        a.deep_equal(
            await storage.entry(root_first, context),
            {"entry": _entry(root_first, root_id), "commitSeq": root_entries_seq},
        )

        async def commit_seq(id: EntryId) -> object:
            stored = await storage.entry(id, context)
            return None if stored is None else stored["commitSeq"]

        a.strict_equal(await commit_seq(root_fork_point), root_entries_seq)
        a.strict_equal(await commit_seq(grandchild_head), grandchild_entries_seq)
        a.strict_equal(await commit_seq(grandchild_tail), grandchild_entries_seq)
        a.strict_equal(await storage.entry(id_from_number(999_999), context), None)

        a.deep_equal(
            await storage.visible_entry(grandchild_id, root_first, context),
            {"entry": _entry(root_first, root_id), "commitSeq": root_entries_seq},
        )
        visible = await storage.visible_entry(grandchild_id, child_fork_point, context)
        a.strict_equal(None if visible is None else visible["entry"]["conversationId"], child_id)
        visible = await storage.visible_entry(grandchild_id, grandchild_tail, context)
        a.strict_equal(None if visible is None else visible["commitSeq"], grandchild_entries_seq)
        for conversation_id, hidden in (
            (grandchild_id, root_excluded_same_commit),
            (grandchild_id, root_excluded_later),
            (grandchild_id, child_excluded),
            (grandchild_id, child_excluded_later),
            (root_id, grandchild_head),
            (grandchild_id, id_from_number(999_999)),
        ):
            a.strict_equal(await storage.visible_entry(conversation_id, hidden, context), None)
        await a.rejects(storage.visible_entry(id_from_number(999_999), root_first, context), "Unknown conversation")
        await a.rejects(
            storage.scan_entries({"conversationId": id_from_number(999_999)}, 10, None, context), "Unknown conversation"
        )

    @case("scans tables in either ID order and continues a cursor in its order")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        conversation_ids: list[int] = [root_id]
        task_ids: list[int] = []
        submission_ids: list[int] = []
        for _index in range(3):
            conversation_id = await storage.mint_id()
            task_id = await storage.mint_id()
            submission_id = await storage.mint_id()
            await storage.commit(
                [
                    {"type": "conversation", "value": {"id": id_from_number(conversation_id)}},
                    {"type": "task", "value": _pending_task(task_id, root_id)},
                    {
                        "type": "submission",
                        "value": {
                            "id": id_from_number(submission_id),
                            "conversationId": root_id,
                            "type": "input",
                            "status": "queued",
                        },
                    },
                ],
                context,
            )
            conversation_ids.append(conversation_id)
            task_ids.append(task_id)
            submission_ids.append(submission_id)

        def query(order: ScanOrder | None) -> dict[str, Any]:
            return {} if order is None else {"order": order}

        scans: list[tuple[Callable[[ScanOrder | None, Cursor | None], Awaitable[Page[Any]]], list[int]]] = [
            (lambda order, cursor: storage.scan_conversations(query(order), 2, cursor, context), conversation_ids),  # type: ignore[arg-type]
            (lambda order, cursor: storage.scan_tasks(query(order), 2, cursor, context), task_ids),  # type: ignore[arg-type]
            (lambda order, cursor: storage.scan_submissions(query(order), 2, cursor, context), submission_ids),  # type: ignore[arg-type]
        ]

        async def all_ids(
            scan: Callable[[ScanOrder | None, Cursor | None], Awaitable[Page[Any]]], order: ScanOrder | None
        ) -> list[int]:
            page = await scan(order, None)
            found = _ids(page)
            while page.get("next") is not None:
                # A cursor carries its order; the query may omit it.
                page = await scan(None, _round_trip(page.get("next")))
                found.extend(_ids(page))
            return found

        for scan, ids in scans:
            reversed_ids = list(reversed(ids))
            a.deep_equal(await all_ids(scan, None), ids)
            a.deep_equal(await all_ids(scan, "ascending"), ids)
            a.deep_equal(await all_ids(scan, "descending"), reversed_ids)
            descending = await scan("descending", None)
            a.deep_equal(_ids(await scan("descending", descending.get("next"))), reversed_ids[2:4])
            await a.rejects(scan("ascending", descending.get("next")), "cursor")

    @case("replaces complete task records and pages filtered task scans")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        first_id = await storage.mint_id()
        second_id = await storage.mint_id()
        third_id = await storage.mint_id()
        first: AnyTaskRecord = {**_pending_task(first_id, root_id), "memos": {"winner": "first"}}
        second: AnyTaskRecord = {**_pending_task(second_id, root_id), "background": True}
        third: AnyTaskRecord = {**_pending_task(third_id, root_id), "abortRequested": True}
        await storage.commit(
            [{"type": "task", "value": first}, {"type": "task", "value": second}, {"type": "task", "value": third}],
            context,
        )
        running: AnyTaskRecord = {
            **first,
            "state": {"status": "running", "checkpoint": {"phase": "effect", "attempt": 1}},
            "abortRequested": True,
        }
        await storage.commit([{"type": "task", "value": running}], context)
        a.deep_equal(await storage.task(id_from_number(first_id), context), running)
        terminal: AnyTaskRecord = {
            "id": id_from_number(first_id),
            "conversationId": root_id,
            "kind": first["kind"],
            "version": first["version"],
            "input": first["input"],
            "state": {"status": "terminal", "outcome": {"status": "completed", "result": {"entryId": 99}}},
            "background": False,
            "abortRequested": True,
        }
        await storage.commit([{"type": "task", "value": terminal}], context)
        a.deep_equal(await storage.task(id_from_number(first_id), context), terminal)

        pending_page = await storage.scan_tasks({"status": "pending"}, 1, None, context)
        a.deep_equal(_ids(pending_page), [second_id])
        a.ok(pending_page.get("next") is not None, "Expected value to be defined")
        a.deep_equal(
            _ids(await storage.scan_tasks({"status": "pending"}, 1, pending_page.get("next"), context)), [third_id]
        )
        a.deep_equal(
            (await storage.scan_tasks({"status": "terminal", "abortRequested": True}, 10, None, context))["items"],
            [terminal],
        )
        a.deep_equal(_ids(await storage.scan_tasks({"background": True}, 10, None, context)), [second_id])

    @case("stores owners and scans waiting and completing tasks by status")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        owner_id = id_from_number(await storage.mint_id())
        waiting_id = await storage.mint_id()
        completing_id = await storage.mint_id()
        owner = _pending_task(owner_id, root_id)
        waiting: AnyTaskRecord = {
            **_pending_task(waiting_id, root_id),
            "owner": owner_id,
            "state": {"status": "waiting", "checkpoint": {"phase": "next"}, "on": [owner_id], "policy": "allSettled"},
            "memos": {"kept": True},
        }
        base = {key: value for key, value in _pending_task(completing_id, root_id).items() if key != "state"}
        completing = cast(
            AnyTaskRecord,
            {
                **base,
                "owner": owner_id,
                "state": {"status": "completing", "outcome": {"status": "failed", "error": {"message": "held"}}},
            },
        )
        await storage.commit([{"type": "task", "value": value} for value in (owner, waiting, completing)], context)
        a.deep_equal(await storage.task(id_from_number(waiting_id), context), waiting)
        a.deep_equal(await storage.task(id_from_number(completing_id), context), completing)

        async def scan(status: str) -> Sequence[AnyTaskRecord]:
            return (await storage.scan_tasks({"status": status}, 10, None, context))["items"]  # type: ignore[typeddict-item]

        a.deep_equal(await scan("waiting"), [waiting])
        a.deep_equal(await scan("completing"), [completing])
        a.deep_equal([task["id"] for task in await scan("pending")], [owner_id])
        terminal = cast(
            AnyTaskRecord,
            {**completing, "state": {"status": "terminal", "outcome": completing["state"]["outcome"]}},  # type: ignore[typeddict-item]
        )
        await storage.commit([{"type": "task", "value": terminal}], context)
        a.deep_equal(await scan("completing"), [])
        a.deep_equal(await scan("terminal"), [terminal])

    @case("stores and scans sleeping tasks by status")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        sleeping_id = await storage.mint_id()
        sleeping: AnyTaskRecord = {
            **_pending_task(sleeping_id, root_id),
            "state": {"status": "sleeping", "checkpoint": {"phase": "next"}, "until": 61_000},
            "memos": {"kept": True},
        }
        await storage.commit([{"type": "task", "value": sleeping}], context)
        a.deep_equal(await storage.task(id_from_number(sleeping_id), context), sleeping)

        async def scan(status: str) -> Sequence[AnyTaskRecord]:
            return (await storage.scan_tasks({"status": status}, 10, None, context))["items"]  # type: ignore[typeddict-item]

        a.deep_equal(await scan("sleeping"), [sleeping])
        running: AnyTaskRecord = {**sleeping, "state": {"status": "running", "checkpoint": {"phase": "next"}}}
        await storage.commit([{"type": "task", "value": running}], context)
        a.deep_equal(await scan("sleeping"), [])
        a.deep_equal(await scan("running"), [running])

    @case("indexes request IDs per conversation and replaces complete submission records")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        second_conversation_id = id_from_number(await storage.mint_id())
        await storage.commit([{"type": "conversation", "value": {"id": second_conversation_id}}], context)
        first_id = id_from_number(await storage.mint_id())
        second_id = id_from_number(await storage.mint_id())
        other_conversation_id = id_from_number(await storage.mint_id())
        first: SubmissionRecord = {
            "id": first_id,
            "conversationId": root_id,
            "requestId": "same",
            "type": "input",
            "status": "queued",
        }
        second: SubmissionRecord = {
            "id": second_id,
            "conversationId": root_id,
            "requestId": "other",
            "type": "input",
            "status": "queued",
        }
        other_conversation: SubmissionRecord = {
            "id": other_conversation_id,
            "conversationId": second_conversation_id,
            "requestId": "same",
            "type": "input",
            "status": "queued",
        }
        await storage.commit(
            [
                {"type": "submission", "value": first},
                {"type": "submission", "value": second},
                {"type": "submission", "value": other_conversation},
            ],
            context,
        )
        a.deep_equal(await storage.submission_by_request(root_id, "same", context), first)
        a.deep_equal(await storage.submission_by_request(second_conversation_id, "same", context), other_conversation)

        placed_second = cast(SubmissionRecord, {**second, "status": "placed", "entry": await storage.mint_id()})
        await storage.commit([{"type": "submission", "value": placed_second}], context)
        a.deep_equal(await storage.submission(second_id, context), placed_second)
        a.deep_equal(await storage.submission_by_request(root_id, "other", context), placed_second)

        async def ids(query: SubmissionQuery) -> list[int]:
            found: list[int] = []
            cursor: Cursor | None = None
            while True:
                page = await storage.scan_submissions(query, 1, cursor, context)
                found.extend(_ids(page))
                cursor = page.get("next")
                if cursor is None:
                    return found

        a.deep_equal(await ids({}), [first_id, second_id, other_conversation_id])
        a.deep_equal(await ids({"conversationId": root_id}), [first_id, second_id])
        # A status change moves the record between status scans.
        a.deep_equal(await ids({"status": "queued"}), [first_id, other_conversation_id])
        a.deep_equal(await ids({"status": "placed"}), [second_id])
        a.deep_equal(await ids({"conversationId": second_conversation_id, "status": "queued"}), [other_conversation_id])
        a.deep_equal(await ids({"conversationId": second_conversation_id, "status": "placed"}), [])
        a.deep_equal(
            (await storage.scan_submissions({"status": "placed"}, 10, None, context))["items"], [placed_second]
        )

    @case("stores passive write submissions without input-only lifecycle states")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        done_id = id_from_number(await storage.mint_id())
        failed_id = id_from_number(await storage.mint_id())
        queued_done: SubmissionRecord = {
            "id": done_id,
            "conversationId": root_id,
            "requestId": "passive-done",
            "type": "write",
            "status": "queued",
        }
        queued_failed: SubmissionRecord = {
            "id": failed_id,
            "conversationId": root_id,
            "requestId": "passive-failed",
            "type": "write",
            "status": "queued",
        }
        await storage.commit(
            [{"type": "submission", "value": queued_done}, {"type": "submission", "value": queued_failed}], context
        )
        done = cast(SubmissionRecord, {**queued_done, "status": "done", "entry": await storage.mint_id()})
        unanswered = cast(
            SubmissionRecord,
            {**queued_failed, "status": "unanswered", "reason": "closed", "detail": {"retryable": False}},
        )
        await storage.commit(
            [{"type": "submission", "value": done}, {"type": "submission", "value": unanswered}], context
        )
        a.deep_equal(await storage.submission(done_id, context), done)
        a.deep_equal(await storage.submission_by_request(root_id, "passive-done", context), done)
        a.deep_equal(await storage.submission(failed_id, context), unanswered)
        a.deep_equal(await storage.submission_by_request(root_id, "passive-failed", context), unanswered)

    @case("reconstructs rewindable documents and preserves half-open incarnations")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        first_id = id_from_number(await storage.mint_id())
        first_record: DocumentCreate = {
            "id": first_id,
            "kind": "conversation.notes",
            "scope": {"kind": "conversation", "conversationId": root_id},
            "history": "rewindable",
            "fork": "asOf",
        }
        initial: dict[str, Any] = {"items": ["a"], "nested": {"count": 1}}
        created_at = await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": first_record,
                    "content": {"kind": "base", "version": 1, "value": initial},
                }
            ],
            context,
        )
        appended: list[Any] = ["b"]
        changed_at = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": first_id,
                    "content": {
                        "kind": "delta",
                        "version": 1,
                        "ops": [("p", ("items",), 1, 0, appended), ("s", ("nested", "count"), 2)],
                    },
                }
            ],
            context,
        )
        initial["items"].append("caller mutation")
        appended.append("caller mutation")
        a.partial_deep_equal(
            await storage.document(first_id, created_at, context),
            {"version": 1, "value": {"items": ["a"], "nested": {"count": 1}}, "deltasSinceBase": 0},
        )
        changed = await storage.document(first_id, changed_at, context)
        assert changed is not None
        a.deep_equal(changed["value"], {"items": ["a", "b"], "nested": {"count": 2}})
        a.strict_equal(changed["deltasSinceBase"], 1)
        changed["value"]["items"].append("read mutation")  # type: ignore[union-attr]
        current = await storage.document(first_id, "current", context)
        a.deep_equal(None if current is None else current["value"], {"items": ["a", "b"], "nested": {"count": 2}})

        checkpoint_at = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": first_id,
                    "content": {
                        "kind": "base",
                        "version": 2,
                        "value": {"items": ["checkpoint"], "nested": {"count": 3}},
                    },
                }
            ],
            context,
        )
        replaced_at = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": first_id,
                    "content": {
                        "kind": "delta",
                        "version": 2,
                        "ops": [("r", {"items": ["replacement"], "nested": {"count": 4}})],
                    },
                }
            ],
            context,
        )
        a.partial_deep_equal(
            await storage.document(first_id, changed_at, context),
            {"version": 1, "value": {"items": ["a", "b"], "nested": {"count": 2}}},
        )
        a.partial_deep_equal(
            await storage.document(first_id, checkpoint_at, context),
            {"version": 2, "value": {"items": ["checkpoint"], "nested": {"count": 3}}, "deltasSinceBase": 0},
        )
        a.partial_deep_equal(
            await storage.document(first_id, replaced_at, context),
            {"value": {"items": ["replacement"], "nested": {"count": 4}}, "deltasSinceBase": 1},
        )
        current = await storage.document(first_id, "current", context)
        a.strict_equal(None if current is None else current["deltasSinceBase"], 1)

        second_id = id_from_number(await storage.mint_id())
        retired_at = await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": {**first_record, "id": second_id},
                    "content": {"kind": "base", "version": 1, "value": {"items": ["new"]}},
                },
                {"type": "document.retire", "id": first_id},
                {
                    "type": "document.change",
                    "id": first_id,
                    "content": {"kind": "delta", "version": 2, "ops": [("s", ("retiring",), True)]},
                },
            ],
            context,
        )
        address: Any = {"kind": first_record["kind"], "scope": first_record["scope"]}
        found = await storage.find_document(address, changed_at, context)
        a.strict_equal(None if found is None else found["id"], first_id)
        a.partial_deep_equal(
            await storage.find_document(address, retired_at, context), {"id": second_id, "createdAt": retired_at}
        )
        a.deep_equal(
            _ids(await storage.scan_documents({"scope": first_record["scope"], "at": changed_at}, 10, None, context)),
            [first_id],
        )
        a.deep_equal(
            _ids(await storage.scan_documents({"scope": first_record["scope"], "at": retired_at}, 10, None, context)),
            [second_id],
        )
        a.strict_equal(await storage.document(first_id, retired_at, context), None)
        current = await storage.document(second_id, "current", context)
        a.deep_equal(None if current is None else current["value"], {"items": ["new"]})

    @case("streams long document tails across root replacement deltas")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        id = id_from_number(await storage.mint_id())
        record: DocumentCreate = {
            "id": id,
            "kind": "conversation.long-tail",
            "scope": {"kind": "conversation", "conversationId": root_id},
            "history": "rewindable",
            "fork": "asOf",
        }
        initial: dict[str, Any] = {
            "revision": 0,
            "rows": [{"value": value, "stable": f"row-{value}"} for value in range(512)],
        }
        created_at = await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": record,
                    "content": {"kind": "base", "version": 1, "value": initial},
                }
            ],
            context,
        )
        before_replacement = copy.deepcopy(initial)
        before_replacement_at = created_at
        for revision in range(1, 25):
            index = (revision * 17) % len(before_replacement["rows"])
            before_replacement["rows"][index]["value"] = -revision
            before_replacement["revision"] = revision
            before_replacement_at = await storage.commit(
                [
                    {
                        "type": "document.change",
                        "id": id,
                        "content": {
                            "kind": "delta",
                            "version": 1,
                            "ops": [("s", ("rows", index, "value"), -revision), ("s", ("revision",), revision)],
                        },
                    }
                ],
                context,
            )
        replacement: dict[str, Any] = {
            "revision": 100,
            "rows": [{"value": 10_000 + value, "stable": f"new-{value}"} for value in range(512)],
        }
        replacement_snapshot = copy.deepcopy(replacement)
        replacement_at = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": id,
                    "content": {"kind": "delta", "version": 1, "ops": [("r", replacement)]},
                }
            ],
            context,
        )
        replacement["rows"][0]["value"] = -999
        current_value = copy.deepcopy(replacement_snapshot)
        for revision in range(101, 125):
            index = (revision * 19) % len(current_value["rows"])
            current_value["rows"][index]["value"] = -revision
            current_value["revision"] = revision
            await storage.commit(
                [
                    {
                        "type": "document.change",
                        "id": id,
                        "content": {
                            "kind": "delta",
                            "version": 1,
                            "ops": [("s", ("rows", index, "value"), -revision), ("s", ("revision",), revision)],
                        },
                    }
                ],
                context,
            )

        async def value_at(at: Any) -> Any:
            stored = await storage.document(id, at, context)
            return None if stored is None else stored["value"]

        a.deep_equal(await value_at(created_at), initial)
        a.deep_equal(await value_at(before_replacement_at), before_replacement)
        a.deep_equal(await value_at(replacement_at), replacement_snapshot)
        read = await storage.document(id, "current", context)
        assert read is not None
        a.deep_equal(read["value"], current_value)
        read["value"]["rows"][0]["value"] = -1_000  # type: ignore[index]
        a.deep_equal(await value_at("current"), current_value)

    @case("copies stored document bases independently and rejects ambiguous sources")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        child_id = id_from_number(await storage.mint_id())
        second_child_id = id_from_number(await storage.mint_id())
        await storage.commit(
            [
                {"type": "conversation", "value": {"id": child_id}},
                {"type": "conversation", "value": {"id": second_child_id}},
            ],
            context,
        )
        source_id = id_from_number(await storage.mint_id())
        source_record: DocumentCreate = {
            "id": source_id,
            "kind": "copy.source",
            "scope": {"kind": "conversation", "conversationId": root_id},
            "history": "rewindable",
            "fork": "asOf",
        }
        created_at = await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": source_record,
                    "content": {"kind": "base", "version": 2, "value": {"count": 1, "rows": [{"value": "base"}]}},
                }
            ],
            context,
        )
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": source_id,
                    "content": {
                        "kind": "delta",
                        "version": 2,
                        "ops": [("s", ("count",), 2), ("p", ("rows",), 1, 0, [{"value": "current"}])],
                    },
                }
            ],
            context,
        )
        historical_copy_id = id_from_number(await storage.mint_id())
        current_copy_id = id_from_number(await storage.mint_id())
        retired_copy_id = id_from_number(await storage.mint_id())

        def child_record(id: DocumentId, conversation_id: ConversationId) -> DocumentCreate:
            return {
                "id": id,
                "kind": source_record["kind"],
                "scope": {"kind": "conversation", "conversationId": conversation_id},
                "history": "rewindable",
                "fork": "asOf",
            }

        await storage.commit(
            [
                {
                    "type": "document.copy",
                    "record": child_record(historical_copy_id, child_id),
                    "source": {"id": source_id, "at": created_at},
                },
                {
                    "type": "document.copy",
                    "record": child_record(current_copy_id, second_child_id),
                    "source": {"id": source_id, "at": "current"},
                },
                {
                    "type": "document.copy",
                    "record": child_record(retired_copy_id, root_id),
                    "source": {"id": source_id, "at": "current"},
                },
                {"type": "document.retire", "id": retired_copy_id},
            ],
            context,
        )
        a.partial_deep_equal(
            await storage.document(historical_copy_id, "current", context),
            {"version": 2, "value": {"count": 1, "rows": [{"value": "base"}]}},
        )
        a.partial_deep_equal(
            await storage.document(current_copy_id, "current", context),
            {"version": 2, "value": {"count": 2, "rows": [{"value": "base"}, {"value": "current"}]}},
        )
        a.strict_equal(await storage.document(retired_copy_id, "current", context), None)

        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": source_id,
                    "content": {"kind": "base", "version": 2, "value": {"count": 99, "rows": []}},
                },
                {"type": "document.retire", "id": source_id},
            ],
            context,
        )
        current = await storage.document(current_copy_id, "current", context)
        a.deep_equal(
            None if current is None else current["value"],
            {"count": 2, "rows": [{"value": "base"}, {"value": "current"}]},
        )

        latest_source_id = id_from_number(await storage.mint_id())
        latest_copy_id = id_from_number(await storage.mint_id())
        latest_source: DocumentCreate = {
            "id": latest_source_id,
            "kind": "copy.latest",
            "scope": {"kind": "conversation", "conversationId": root_id},
            "history": "latest",
            "fork": "current",
        }
        await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": latest_source,
                    "content": {"kind": "base", "version": 4, "value": {"retained": "copy"}},
                }
            ],
            context,
        )
        await storage.commit(
            [
                {
                    "type": "document.copy",
                    "record": {
                        **latest_source,
                        "id": latest_copy_id,
                        "scope": {"kind": "conversation", "conversationId": child_id},
                    },
                    "source": {"id": latest_source_id, "at": "current"},
                }
            ],
            context,
        )
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": latest_source_id,
                    "content": {"kind": "base", "version": 4, "value": {"retained": "source-only"}},
                },
                {"type": "document.retire", "id": latest_source_id},
            ],
            context,
        )
        a.partial_deep_equal(
            await storage.document(latest_copy_id, "current", context), {"version": 4, "value": {"retained": "copy"}}
        )

        conflict_id = id_from_number(await storage.mint_id())
        conflict_error: BaseException | None = None
        try:
            await storage.commit(
                [
                    {
                        "type": "document.copy",
                        "record": child_record(conflict_id, child_id),
                        "source": {"id": current_copy_id, "at": "current"},
                    },
                    {"type": "document.retire", "id": current_copy_id},
                ],
                context,
            )
        except Exception as error:
            conflict_error = error
        # Rejected without effect: the copy and the retirement in the same batch are both absent.
        a.strict_equal(isinstance(conflict_error, Exception), True)
        a.strict_equal(await storage.document(conflict_id, "current", context), None)
        current = await storage.document(current_copy_id, "current", context)
        a.deep_equal(
            None if current is None else current["value"],
            {"count": 2, "rows": [{"value": "base"}, {"value": "current"}]},
        )

        mismatch_id = id_from_number(await storage.mint_id())
        mismatch_error: BaseException | None = None
        try:
            await storage.commit(
                [
                    {
                        "type": "document.copy",
                        "record": cast(
                            DocumentCreate, {**child_record(mismatch_id, child_id), "kind": "copy.mismatch"}
                        ),
                        "source": {"id": current_copy_id, "at": "current"},
                    }
                ],
                context,
            )
        except Exception as error:
            mismatch_error = error
        a.strict_equal(isinstance(mismatch_error, Exception), True)
        a.strict_equal(await storage.document(mismatch_id, "current", context), None)

    @case("uses bases for version transitions and rejects historical reads of current-only documents")
    async def _(storage: Storage) -> None:
        await _create_root(storage)
        id = id_from_number(await storage.mint_id())
        record: DocumentCreate = {"id": id, "kind": "session.settings", "scope": {"kind": "session"}}
        await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": record,
                    "content": {"kind": "base", "version": 1, "value": {"count": 1}},
                }
            ],
            context,
        )
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": id,
                    "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 2)]},
                }
            ],
            context,
        )
        migrated_at = await storage.commit(
            [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 2, "value": {"count": 3}}}],
            context,
        )
        a.partial_deep_equal(await storage.document(id, "current", context), {"version": 2, "value": {"count": 3}})
        await a.rejects(storage.document(id, migrated_at, context), "does not retain historical content")
        await a.rejects(
            storage.commit(
                [
                    {
                        "type": "document.change",
                        "id": id,
                        "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 4)]},
                    }
                ],
                context,
            ),
            "version transition requires a base",
        )
        current = await storage.document(id, "current", context)
        a.deep_equal(None if current is None else current["value"], {"count": 3})
        await storage.commit([{"type": "document.retire", "id": id}], context)
        a.strict_equal(await storage.document(id, "current", context), None)

    @case("indexes logical addresses and exact-scope scans independently")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        first_id = id_from_number(await storage.mint_id())
        second_id = id_from_number(await storage.mint_id())
        conversation_id = id_from_number(await storage.mint_id())
        task_id = id_from_number(await storage.mint_id())
        task_singleton_id = id_from_number(await storage.mint_id())
        task_family_id = id_from_number(await storage.mint_id())
        task_other_kind_id = id_from_number(await storage.mint_id())

        def create(record: Any, owner: str) -> StorageWrite:
            return {
                "type": "document.create",
                "record": record,
                "content": {"kind": "base", "version": 1, "value": {"owner": owner}},
            }

        created_at = await storage.commit(
            [
                {"type": "task", "value": _pending_task(task_id, root_id)},
                create({"id": first_id, "kind": "cache", "scope": {"kind": "session"}, "key": "__proto__"}, "first"),
                create(
                    {"id": second_id, "kind": "cache", "scope": {"kind": "session"}, "key": "constructor"}, "second"
                ),
                create(
                    {
                        "id": conversation_id,
                        "kind": "cache",
                        "scope": {"kind": "conversation", "conversationId": root_id},
                        "history": "latest",
                        "fork": "current",
                        "key": "__proto__",
                    },
                    "conversation",
                ),
                create(
                    {"id": task_singleton_id, "kind": "task.cache", "scope": {"kind": "task", "taskId": task_id}},
                    "singleton",
                ),
                create(
                    {
                        "id": task_family_id,
                        "kind": "task.cache",
                        "scope": {"kind": "task", "taskId": task_id},
                        "key": "member",
                    },
                    "family",
                ),
                create(
                    {"id": task_other_kind_id, "kind": "task.other", "scope": {"kind": "task", "taskId": task_id}},
                    "other",
                ),
            ],
            context,
        )

        async def found_id(address: Any) -> object:
            record = await storage.find_document(address, "current", context)
            return None if record is None else record["id"]

        a.strict_equal(await found_id({"kind": "cache", "scope": {"kind": "session"}, "key": "__proto__"}), first_id)
        session_query: Any = {"scope": {"kind": "session"}, "at": "current"}
        a.strict_equal(len((await storage.scan_documents(session_query, 1, None, context))["items"]), 1)
        first = await storage.scan_documents(session_query, 1, None, context)
        second = await storage.scan_documents(session_query, 1, first.get("next"), context)
        a.deep_equal(_ids(first) + _ids(second), [first_id, second_id])
        a.deep_equal(
            _ids(
                await storage.scan_documents(
                    {"scope": {"kind": "conversation", "conversationId": root_id}, "at": "current"}, 10, None, context
                )
            ),
            [conversation_id],
        )
        a.strict_equal(
            await found_id({"kind": "task.cache", "scope": {"kind": "task", "taskId": task_id}}), task_singleton_id
        )
        a.strict_equal(
            await found_id({"kind": "task.cache", "scope": {"kind": "task", "taskId": task_id}, "key": "member"}),
            task_family_id,
        )
        a.deep_equal(
            _ids(
                await storage.scan_documents(
                    {"scope": {"kind": "task", "taskId": task_id}, "at": "current", "kind": "task.cache"},
                    10,
                    None,
                    context,
                )
            ),
            [task_singleton_id, task_family_id],
        )
        await a.rejects(storage.document(task_singleton_id, created_at, context), "does not retain historical content")

    @case("keeps document lifecycle failures atomic and gives create-plus-retire an empty lifetime")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        first_id = id_from_number(await storage.mint_id())
        second_id = id_from_number(await storage.mint_id())
        record: DocumentCreate = {"id": first_id, "kind": "singleton", "scope": {"kind": "session"}}
        await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": record,
                    "content": {"kind": "base", "version": 1, "value": {"value": 1}},
                }
            ],
            context,
        )
        await a.rejects(
            storage.commit(
                [
                    {
                        "type": "document.create",
                        "record": {**record, "id": second_id},
                        "content": {"kind": "base", "version": 1, "value": {"value": 2}},
                    },
                    {"type": "document.change", "id": first_id, "content": {"kind": "delta", "version": 1, "ops": []}},
                ],
                context,
            ),
            "already has a current incarnation",
        )
        current = await storage.document(first_id, "current", context)
        a.deep_equal(None if current is None else current["value"], {"value": 1})
        a.strict_equal(await storage.document(second_id, "current", context), None)

        empty_id = id_from_number(await storage.mint_id())
        empty_at = await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": {
                        "id": empty_id,
                        "kind": record["kind"],
                        "key": "empty",
                        "scope": {"kind": "conversation", "conversationId": root_id},
                        "history": "rewindable",
                        "fork": "initial",
                    },
                    "content": {"kind": "base", "version": 1, "value": {}},
                },
                {"type": "document.retire", "id": empty_id},
            ],
            context,
        )
        a.strict_equal(await storage.document(empty_id, "current", context), None)
        a.strict_equal(await storage.document(empty_id, empty_at, context), None)
        a.strict_equal(
            await storage.find_document(
                {"kind": record["kind"], "scope": {"kind": "conversation", "conversationId": root_id}, "key": "empty"},
                empty_at,
                context,
            ),
            None,
        )

    @case("rolls back record tables and secondary indexes when a document command fails")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        task_id = await storage.mint_id()
        submission_id = id_from_number(await storage.mint_id())
        document_id = id_from_number(await storage.mint_id())
        task = _pending_task(task_id, root_id)
        submission: SubmissionRecord = {
            "id": submission_id,
            "conversationId": root_id,
            "requestId": "atomic",
            "type": "input",
            "status": "queued",
        }
        record: DocumentCreate = {"id": document_id, "kind": "atomic", "scope": {"kind": "session"}}
        baseline_seq = await storage.commit(
            [
                {"type": "task", "value": task},
                {"type": "submission", "value": submission},
                {
                    "type": "document.create",
                    "record": record,
                    "content": {"kind": "base", "version": 1, "value": {"count": 1}},
                },
            ],
            context,
        )
        entry_id = await storage.mint_id()
        conflicting_document_id = id_from_number(await storage.mint_id())
        await a.rejects(
            storage.commit(
                [
                    {
                        "type": "task",
                        "value": {**task, "state": {"status": "running", "checkpoint": {"phase": "effect"}}},
                    },
                    {
                        "type": "submission",
                        "value": cast(SubmissionRecord, {**submission, "status": "unanswered", "reason": "failed"}),
                    },
                    {"type": "entry", "value": _entry(entry_id, root_id, "transient")},
                    {
                        "type": "document.create",
                        "record": {**record, "id": conflicting_document_id},
                        "content": {"kind": "base", "version": 1, "value": {"count": 2}},
                    },
                ],
                context,
            ),
            "already has a current incarnation",
        )
        a.deep_equal(await storage.task(id_from_number(task_id), context), task)
        a.deep_equal((await storage.scan_tasks({"status": "pending"}, 10, None, context))["items"], [task])
        a.deep_equal(await storage.submission_by_request(root_id, "atomic", context), submission)
        a.strict_equal(await storage.entry(id_from_number(entry_id), context), None)
        a.strict_equal(await storage.document(conflicting_document_id, "current", context), None)
        found = await storage.find_document({"kind": record["kind"], "scope": record["scope"]}, "current", context)
        a.strict_equal(None if found is None else found["id"], document_id)
        after_rollback_seq = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": document_id,
                    "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 3)]},
                }
            ],
            context,
        )
        a.greater_than(after_rollback_seq, baseline_seq)

    @case("keeps indexed string identities lossless")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        first = "\ud800"
        second = "\ud801"
        first_task_id = await storage.mint_id()
        second_task_id = await storage.mint_id()
        first_submission_id = id_from_number(await storage.mint_id())
        second_submission_id = id_from_number(await storage.mint_id())
        first_kind_document_id = id_from_number(await storage.mint_id())
        second_kind_document_id = id_from_number(await storage.mint_id())
        first_key_document_id = id_from_number(await storage.mint_id())
        second_key_document_id = id_from_number(await storage.mint_id())

        def create(record: Any, identity: str) -> StorageWrite:
            return {
                "type": "document.create",
                "record": record,
                "content": {"kind": "base", "version": 1, "value": {"identity": identity}},
            }

        await storage.commit(
            [
                {"type": "task", "value": {**_pending_task(first_task_id, root_id), "kind": first}},
                {"type": "task", "value": {**_pending_task(second_task_id, root_id), "kind": second}},
                {
                    "type": "submission",
                    "value": {
                        "id": first_submission_id,
                        "conversationId": root_id,
                        "requestId": first,
                        "type": "input",
                        "status": "queued",
                    },
                },
                {
                    "type": "submission",
                    "value": {
                        "id": second_submission_id,
                        "conversationId": root_id,
                        "requestId": second,
                        "type": "input",
                        "status": "queued",
                    },
                },
                create({"id": first_kind_document_id, "kind": first, "scope": {"kind": "session"}}, "first kind"),
                create({"id": second_kind_document_id, "kind": second, "scope": {"kind": "session"}}, "second kind"),
                create(
                    {"id": first_key_document_id, "kind": "family", "key": first, "scope": {"kind": "session"}},
                    "first key",
                ),
                create(
                    {"id": second_key_document_id, "kind": "family", "key": second, "scope": {"kind": "session"}},
                    "second key",
                ),
            ],
            context,
        )
        a.deep_equal(_ids(await storage.scan_tasks({"kind": first}, 10, None, context)), [first_task_id])
        a.deep_equal(_ids(await storage.scan_tasks({"kind": second}, 10, None, context)), [second_task_id])
        task = await storage.task(id_from_number(first_task_id), context)
        a.strict_equal(None if task is None else task["kind"], first)
        task = await storage.task(id_from_number(second_task_id), context)
        a.strict_equal(None if task is None else task["kind"], second)
        submission = await storage.submission_by_request(root_id, first, context)
        a.strict_equal(None if submission is None else submission.get("requestId"), first)
        a.strict_equal(None if submission is None else submission["id"], first_submission_id)
        submission = await storage.submission_by_request(root_id, second, context)
        a.strict_equal(None if submission is None else submission["id"], second_submission_id)

        async def found_id(address: Any) -> object:
            record = await storage.find_document(address, "current", context)
            return None if record is None else record["id"]

        a.strict_equal(await found_id({"kind": first, "scope": {"kind": "session"}}), first_kind_document_id)
        a.strict_equal(await found_id({"kind": second, "scope": {"kind": "session"}}), second_kind_document_id)
        a.strict_equal(
            await found_id({"kind": "family", "key": first, "scope": {"kind": "session"}}), first_key_document_id
        )
        a.strict_equal(
            await found_id({"kind": "family", "key": second, "scope": {"kind": "session"}}), second_key_document_id
        )
        a.deep_equal(
            _ids(
                await storage.scan_documents(
                    {"scope": {"kind": "session"}, "at": "current", "kind": first}, 10, None, context
                )
            ),
            [first_kind_document_id],
        )

    @case("keeps one global record ID namespace and rejects exhausted ID minting")
    async def _(storage: Storage) -> None:
        root_id = await _create_root(storage)
        explicit_entry_id = 100
        await storage.commit([{"type": "entry", "value": _entry(explicit_entry_id, root_id)}], context)
        a.strict_equal(await storage.mint_id(), 101)
        await a.rejects(
            storage.commit([{"type": "task", "value": _pending_task(explicit_entry_id, root_id)}], context),
            f"ID {explicit_entry_id} already belongs to entry",
        )
        await storage.commit([{"type": "entry", "value": _entry(MAX_SAFE_INTEGER, root_id, "last-id")}], context)
        await a.rejects(storage.mint_id(), "ID space is exhausted")
        await a.rejects(storage.mint_id(), "ID space is exhausted")

    @case("rejects every operation after close")
    async def _(storage: Storage) -> None:
        await _create_root(storage)
        await storage.close(context)
        await a.rejects(storage.conversation(ROOT_CONVERSATION_ID, context), "closed")
        await a.rejects(storage.commit([], context), "closed")
        await a.rejects(storage.mint_id(), "closed")

    return cases
