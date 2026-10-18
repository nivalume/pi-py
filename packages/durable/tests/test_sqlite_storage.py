"""Port of ``test/sqlite-storage.test.ts``."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from pi_chord.context import BACKGROUND_CONTEXT, Context
from pi_durable import ROOT_CONVERSATION_ID
from pi_durable.ids import id_from_number
from pi_durable.storage.sqlite import SqliteStorage
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.testing import register_storage_conformance
from pi_durable.types import AnyTaskRecord, DocumentCreate, EntryRecord, Seq, Storage, StorageWrite

context = BACKGROUND_CONTEXT


async def _with_storage(use: Any) -> None:
    import tempfile

    directory = tempfile.mkdtemp(prefix="pi-durable-sqlite-")
    storage = await open_sqlite_storage(os.path.join(directory, "storage.sqlite"))
    try:
        await use(storage)
    finally:
        await storage.close(context)


class ReopeningStorage:
    """Closes and reopens the database after every commit, so every read sees only what was persisted."""

    def __init__(self, current: SqliteStorage, path: str) -> None:
        self._current = current
        self._path = path
        self._closed = False

    async def commit(self, writes: Sequence[StorageWrite], commit_context: Context) -> Seq:
        if self._closed:
            return await self._current.commit(writes, commit_context)
        try:
            return await self._current.commit(writes, commit_context)
        finally:
            await self._current.close(context)
            self._current = await open_sqlite_storage(self._path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current, name)

    async def close(self, close_context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        await self._current.close(close_context)


async def _with_reopening_storage(use: Any) -> None:
    import tempfile

    directory = tempfile.mkdtemp(prefix="pi-durable-sqlite-conformance-")
    path = os.path.join(directory, "storage.sqlite")
    created = await open_sqlite_storage(path)
    await created.close(context)
    storage = ReopeningStorage(await open_sqlite_storage(path), path)
    try:
        await use(storage)
    finally:
        await storage.close(context)


register_storage_conformance(globals(), "SqliteStorage", _with_storage)
register_storage_conformance(globals(), "SqliteStorage across reopen", _with_reopening_storage)


@pytest.fixture
async def sqlite_storage(tmp_path: Path) -> AsyncIterator[tuple[SqliteStorage, str]]:
    path = str(tmp_path / "storage.sqlite")
    storage = await open_sqlite_storage(path)
    try:
        yield storage, path
    finally:
        await storage.close(context)


def entry(id: int, conversation_id: int, data: Any = None) -> EntryRecord:
    record: dict[str, Any] = {"id": id, "conversationId": conversation_id, "kind": "message"}
    if data is not None:
        record["data"] = data
    return record  # type: ignore[return-value]


async def create_root(storage: Storage) -> Seq:
    return await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)


def pending_task(id: int) -> AnyTaskRecord:
    return {
        "id": id_from_number(id),
        "conversationId": ROOT_CONVERSATION_ID,
        "kind": "test.task",
        "version": 1,
        "input": None,
        "state": {"status": "pending", "checkpoint": {"phase": "ready"}},
        "background": False,
        "abortRequested": False,
    }


def scalar(path: str, sql: str) -> Any:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return db.execute(sql).fetchone()[0]
    finally:
        db.close()


def revision_count(path: str, document_id: int) -> int:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return db.execute("SELECT count(*) FROM document_revisions WHERE document_id = ?", (document_id,)).fetchone()[0]
    finally:
        db.close()


async def test_persists_records_sequence_allocation_and_global_id_allocation_across_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "storage.sqlite")
    storage = await open_sqlite_storage(path)
    assert await create_root(storage) == 1
    entry_id = await storage.mint_id()
    assert await storage.commit([{"type": "entry", "value": entry(entry_id, ROOT_CONVERSATION_ID)}], context) == 2
    await storage.close(context)

    reopened = await open_sqlite_storage(path)
    try:
        assert await reopened.entry(id_from_number(entry_id), context) == {
            "entry": entry(entry_id, ROOT_CONVERSATION_ID),
            "commitSeq": 2,
        }
        assert await reopened.mint_id() == entry_id + 1
        assert await reopened.commit([{"type": "task", "value": pending_task(await reopened.mint_id())}], context) == 3
    finally:
        await reopened.close(context)


async def test_namespaces_tables_and_indexes_with_a_prefix_across_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "storage.sqlite")
    host = sqlite3.connect(path)
    host.execute("CREATE TABLE tasks (text TEXT)")
    host.close()
    storage = await open_sqlite_storage(path, prefix="pi_")
    assert await create_root(storage) == 1
    await storage.close(context)

    reopened = await open_sqlite_storage(path, prefix="pi_")
    try:
        assert await reopened.conversation(ROOT_CONVERSATION_ID, context) == {"id": ROOT_CONVERSATION_ID}
        database = sqlite3.connect(path)
        try:
            names = [
                row[0]
                for row in database.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%'"
                )
            ]
        finally:
            database.close()
        assert [name for name in names if not name.startswith("pi_")] == ["tasks"]
        assert "pi_tasks" in names
    finally:
        await reopened.close(context)


async def test_rejects_an_invalid_prefix_before_creating_the_database(tmp_path: Path) -> None:
    database_directory = tmp_path / "nested"
    with pytest.raises(ValueError, match="Invalid SQLite schema prefix"):
        await open_sqlite_storage(str(database_directory / "storage.sqlite"), prefix="bad-prefix")
    assert not database_directory.exists()


async def test_rejects_persisted_metadata_corruption_on_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "storage.sqlite")
    storage = await open_sqlite_storage(path)
    await storage.close(context)
    database = sqlite3.connect(path)
    database.execute("DELETE FROM durable_metadata")
    database.commit()
    database.close()
    with pytest.raises(RuntimeError, match="Durable SQLite metadata is missing"):
        await open_sqlite_storage(path)


async def test_rejects_a_document_whose_required_base_is_missing(sqlite_storage: tuple[SqliteStorage, str]) -> None:
    storage, path = sqlite_storage
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": {"id": id, "kind": "corrupt", "scope": {"kind": "session"}},
                "content": {"kind": "base", "version": 1, "value": {"retained": True}},
            }
        ],
        context,
    )
    database = sqlite3.connect(path)
    database.execute("DELETE FROM document_revisions WHERE document_id = ?", (id,))
    database.commit()
    database.close()
    with pytest.raises(RuntimeError, match=f"Document {id} is missing a required base"):
        await storage.document(id, "current", context)


async def test_replays_detached_root_replacements_and_follow_up_edits_while_rejecting_corrupt_operations(
    sqlite_storage: tuple[SqliteStorage, str],
) -> None:
    storage, path = sqlite_storage
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": {"id": id, "kind": "replay", "scope": {"kind": "session"}},
                "content": {"kind": "base", "version": 1, "value": {"nested": {"value": 1}, "rows": []}},
            }
        ],
        context,
    )
    for ops in (
        [("r", {"nested": {"value": 2}, "rows": [{"id": 1}]})],
        [("s", ("nested", "value"), 3), ("p", ("rows",), 1, 0, [{"id": 2}]), ("m", ("rows",), [1, 0])],
        [("s", ("nested", "value"), 4)],
    ):
        await storage.commit(
            [{"type": "document.change", "id": id, "content": {"kind": "delta", "version": 1, "ops": ops}}], context
        )

    expected = {"nested": {"value": 4}, "rows": [{"id": 2}, {"id": 1}]}
    first = await storage.document(id, "current", context)
    assert first is not None
    assert first["value"] == expected
    first["value"]["nested"]["value"] = 99  # type: ignore[index]
    first["value"]["rows"][0]["id"] = 99  # type: ignore[index]
    again = await storage.document(id, "current", context)
    assert again is not None and again["value"] == expected

    database = sqlite3.connect(path)
    database.execute(
        "UPDATE document_revisions SET content = ? WHERE document_id = ? AND seq = "
        "(SELECT max(seq) FROM document_revisions WHERE document_id = ?)",
        ('[["unknown"]]', id, id),
    )
    database.commit()
    database.close()
    with pytest.raises(TypeError, match="unknown op verb"):
        await storage.document(id, "current", context)


async def test_rolls_sql_rows_and_sequence_allocation_back_as_one_transaction(
    sqlite_storage: tuple[SqliteStorage, str],
) -> None:
    storage, path = sqlite_storage
    await create_root(storage)
    transient_id = await storage.mint_id()
    transient_task_id = await storage.mint_id()
    circular: dict[str, Any] = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="Circular reference"):
        await storage.commit(
            [
                {"type": "entry", "value": entry(transient_id, ROOT_CONVERSATION_ID)},
                {"type": "task", "value": {**pending_task(transient_task_id), "input": circular}},
            ],
            context,
        )
    assert await storage.entry(id_from_number(transient_id), context) is None
    assert await storage.task(id_from_number(transient_task_id), context) is None
    committed_id = await storage.mint_id()
    assert await storage.commit([{"type": "entry", "value": entry(committed_id, ROOT_CONVERSATION_ID)}], context) == 2
    assert scalar(path, "SELECT count(*) FROM entries") == 1
    assert scalar(path, "SELECT next_seq FROM durable_metadata WHERE singleton = 1") == 3


async def test_reconstructs_recent_and_ancient_rewindable_points_after_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "storage.sqlite")
    storage = await open_sqlite_storage(path)
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    record: DocumentCreate = {
        "id": id,
        "kind": "history",
        "scope": {"kind": "conversation", "conversationId": ROOT_CONVERSATION_ID},
        "history": "rewindable",
        "fork": "asOf",
    }
    created_at = await storage.commit(
        [
            {
                "type": "document.create",
                "record": record,
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    ancient_at = created_at
    recent_at = created_at
    for count in range(1, 41):
        content: Any = (
            {"kind": "base", "version": 1, "value": {"count": count}}
            if count == 20
            else {"kind": "delta", "version": 1, "ops": [("s", ("count",), count)]}
        )
        recent_at = await storage.commit([{"type": "document.change", "id": id, "content": content}], context)
        if count == 5:
            ancient_at = recent_at
    await storage.close(context)
    reopened = await open_sqlite_storage(path)
    try:
        ancient = await reopened.document(id, ancient_at, context)
        recent = await reopened.document(id, recent_at, context)
        assert ancient is not None and ancient["value"] == {"count": 5}
        assert recent is not None and recent["value"] == {"count": 40}
    finally:
        await reopened.close(context)


async def test_uses_indexes_for_exact_addresses_exact_scopes_entry_history_and_document_revision_tails(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "storage.sqlite")
    storage = await open_sqlite_storage(path)
    await storage.close(context)
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        plans = [
            (
                """SELECT record FROM documents
                WHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?
                AND retired_at IS NULL ORDER BY created_at DESC LIMIT 1""",
                ("kind", "session", 0, 0, ""),
            ),
            (
                """SELECT record FROM documents
                WHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?
                AND created_at <= ? AND (retired_at IS NULL OR retired_at > ?)
                ORDER BY created_at DESC LIMIT 1""",
                ("kind", "conversation", 1, 0, "", 10, 10),
            ),
            (
                """SELECT record FROM documents
                WHERE scope_kind = ? AND owner_id = ? AND kind = ? AND id > ? ORDER BY id LIMIT ?""",
                ("task", 1, "kind", 0, 10),
            ),
            ("SELECT record FROM entries WHERE conversation_id = ? AND id <= ? ORDER BY id DESC LIMIT ?", (1, 10, 10)),
            (
                "SELECT record FROM entries WHERE conversation_id = ? AND head IS NOT NULL AND id <= ? "
                "ORDER BY id DESC LIMIT 1",
                (1, 10),
            ),
            ("SELECT record FROM tasks WHERE status = ? AND id > ? ORDER BY id LIMIT ?", ("pending", 0, 10)),
            (
                """SELECT seq, kind, version, content FROM document_revisions
                WHERE document_id = ? AND kind = 'base' AND seq <= ? ORDER BY seq DESC LIMIT 1""",
                (1, 10),
            ),
            (
                """SELECT seq, kind, version, content FROM document_revisions
                WHERE document_id = ? AND seq > ? AND seq <= ? ORDER BY seq""",
                (1, 5, 10),
            ),
        ]
        details = [
            "\n".join(row[3] for row in db.execute(f"EXPLAIN QUERY PLAN {sql}", params)) for sql, params in plans
        ]
    finally:
        db.close()
    assert "documents_by_address" in details[0]
    assert "documents_by_address" in details[1]
    assert "documents_by_scope_kind" in details[2]
    assert "entries_by_conversation" in details[3]
    assert "entry_heads_by_conversation" in details[4]
    assert "tasks_by_status" in details[5]
    assert "document_revisions_by_kind" in details[6]
    assert "sqlite_autoindex_document_revisions_1" in details[7]
    for detail in details[:3]:
        assert "SCAN documents" not in detail
    assert "SCAN document_revisions" not in details[7]


async def test_reclaims_current_only_revisions_only_after_a_base_or_retirement(
    sqlite_storage: tuple[SqliteStorage, str],
) -> None:
    storage, path = sqlite_storage
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": {"id": id, "kind": "latest", "scope": {"kind": "session"}},
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    for count in range(1, 11):
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": id,
                    "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), count)]},
                }
            ],
            context,
        )
    assert revision_count(path, id) == 11
    import json

    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        content = db.execute(
            "SELECT content FROM document_revisions WHERE document_id = ? AND kind = 'delta' ORDER BY seq DESC LIMIT 1",
            (id,),
        ).fetchone()[0]
    finally:
        db.close()
    assert json.loads(content) == [["s", ["count"], 10]]
    await storage.commit(
        [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"count": 11}}}],
        context,
    )
    assert revision_count(path, id) == 1
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": id,
                "content": {"kind": "delta", "version": 1, "ops": [("r", {"count": 12})]},
            }
        ],
        context,
    )
    assert revision_count(path, id) == 2
    await storage.commit([{"type": "document.retire", "id": id}], context)
    assert revision_count(path, id) == 0


async def test_auto_checkpoints_wal_frames_and_truncates_the_wal_on_close(tmp_path: Path) -> None:
    path = str(tmp_path / "storage.sqlite")
    storage = await open_sqlite_storage(path, wal_auto_checkpoint_pages=1)
    await create_root(storage)
    for index in range(20):
        await storage.commit(
            [
                {
                    "type": "entry",
                    "value": entry(
                        await storage.mint_id(), ROOT_CONVERSATION_ID, {"text": "x" * (32 * 1024), "index": index}
                    ),
                }
            ],
            context,
        )
    wal_path = f"{path}-wal"
    assert os.stat(wal_path).st_size < 512 * 1024
    observer = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert observer.execute("SELECT count(*) FROM entries").fetchone()[0] == 20
        await storage.close(context)
        assert os.stat(wal_path).st_size == 0
    finally:
        observer.close()


async def test_reuses_pages_released_by_current_only_checkpoints(sqlite_storage: tuple[SqliteStorage, str]) -> None:
    storage, path = sqlite_storage
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    large = "x" * (512 * 1024)
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": {"id": id, "kind": "reuse", "scope": {"kind": "session"}},
                "content": {"kind": "base", "version": 1, "value": {"text": large}},
            }
        ],
        context,
    )
    await storage.commit(
        [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"text": "small"}}}],
        context,
    )
    pages_after_delete = scalar(path, "SELECT page_count FROM pragma_page_count()")
    free_after_delete = scalar(path, "SELECT freelist_count FROM pragma_freelist_count()")
    assert free_after_delete > 0
    await storage.commit(
        [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"text": large}}}],
        context,
    )
    assert scalar(path, "SELECT page_count FROM pragma_page_count()") <= pages_after_delete + 2
    assert scalar(path, "SELECT freelist_count FROM pragma_freelist_count()") < free_after_delete


async def test_keeps_representative_row_and_document_storage_bounded(tmp_path: Path) -> None:
    path = str(tmp_path / "storage.sqlite")
    storage = await open_sqlite_storage(path)
    await create_root(storage)
    for index in range(100):
        await storage.commit(
            [
                {
                    "type": "entry",
                    "value": entry(
                        await storage.mint_id(), ROOT_CONVERSATION_ID, {"index": index, "text": "x" * 1_024}
                    ),
                }
            ],
            context,
        )
    document_id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": {
                    "id": document_id,
                    "kind": "size.history",
                    "scope": {"kind": "conversation", "conversationId": ROOT_CONVERSATION_ID},
                    "history": "rewindable",
                    "fork": "asOf",
                },
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    for count in range(1, 101):
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": document_id,
                    "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), count)]},
                }
            ],
            context,
        )
    await storage.close(context)
    assert os.stat(path).st_size < 1024 * 1024


async def test_stores_pi_ai_messages_in_entries_as_their_camel_case_json(
    sqlite_storage: tuple[SqliteStorage, str],
) -> None:
    import json

    from pi_ai.types import TextContent, UserMessage

    storage, path = sqlite_storage
    await create_root(storage)
    id = await storage.mint_id()
    message = UserMessage([TextContent("hi")], timestamp=5)
    await storage.commit(
        [{"type": "entry", "value": {**entry(id, ROOT_CONVERSATION_ID), "kind": "pi.user", "model": [message]}}],
        context,
    )
    stored = await storage.entry(id_from_number(id), context)
    assert stored is not None and stored["entry"]["model"] == [message]  # type: ignore[typeddict-item]
    record = json.loads(scalar(path, f"SELECT record FROM entries WHERE id = {id}"))
    assert record["model"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}], "timestamp": 5}]
