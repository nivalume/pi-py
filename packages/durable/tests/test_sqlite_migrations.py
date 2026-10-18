"""Port of ``test/sqlite-migrations.test.ts``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable import ROOT_CONVERSATION_ID
from pi_durable.storage.sqlite import (
    CURRENT_SQLITE_SCHEMA_VERSION,
    SQLITE_MIGRATIONS,
    SqliteMigration,
    apply_sqlite_migrations,
)
from pi_durable.storage.sqlite.stdlib import open_sqlite_database, open_sqlite_storage

context = BACKGROUND_CONTEXT


def _js(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


async def test_creates_the_current_schema_and_can_be_applied_repeatedly(tmp_path: Path) -> None:
    database = await open_sqlite_database(tmp_path / "storage.sqlite")
    try:
        await apply_sqlite_migrations(database)
        await apply_sqlite_migrations(database)
        assert await database.get("SELECT version FROM durable_schema WHERE singleton = 1") == {
            "version": CURRENT_SQLITE_SCHEMA_VERSION
        }
        assert await database.get("SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1") == {
            "next_id": "2",
            "next_seq": 1,
        }
    finally:
        await database.close()


async def test_migrates_a_version_1_database_to_store_sleeping_tasks_keeping_its_tasks_and_indexes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "storage.sqlite"
    database = await open_sqlite_database(path)
    await apply_sqlite_migrations(database, SQLITE_MIGRATIONS[:1])
    waiting = {
        "id": 5,
        "conversationId": 1,
        "kind": "test.task",
        "version": 1,
        "input": None,
        "background": False,
        "abortRequested": False,
        "state": {"status": "waiting", "checkpoint": {"phase": "run"}, "on": [], "policy": "allSettled"},
    }
    await database.run(
        "INSERT INTO tasks (id, conversation_id, kind, status, abort_requested, background, record) "
        "VALUES (5, 1, 'test.task', 'waiting', 0, 0, ?)",
        _js(waiting),
    )
    with pytest.raises(Exception):  # noqa: B017 - any constraint error
        await database.run(
            "INSERT INTO tasks (id, conversation_id, kind, status, abort_requested, background, record) "
            "VALUES (6, 1, 'test.task', 'sleeping', 0, 0, '{}')"
        )
    await database.close()

    storage = await open_sqlite_storage(path)
    await storage.close(context)
    migrated = await open_sqlite_database(path)
    try:
        assert await migrated.get("SELECT version FROM durable_schema WHERE singleton = 1") == {"version": 2}
        assert await migrated.get("SELECT status, record FROM tasks WHERE id = 5") == {
            "status": "waiting",
            "record": _js(waiting),
        }
        await migrated.run(
            "INSERT INTO tasks (id, conversation_id, kind, status, abort_requested, background, record) "
            "VALUES (6, 1, 'test.task', 'sleeping', 0, 0, '{}')"
        )
        indexes = await migrated.all(
            "SELECT name FROM sqlite_schema WHERE type = 'index' AND tbl_name = 'tasks' ORDER BY name"
        )
        assert [row["name"] for row in indexes] == [
            "tasks_by_abort_requested",
            "tasks_by_background",
            "tasks_by_conversation",
            "tasks_by_kind",
            "tasks_by_status",
        ]
        assert await migrated.get("SELECT count(*) AS count FROM sqlite_schema WHERE name = 'tasks_v2'") == {"count": 0}
    finally:
        await migrated.close()


async def test_rejects_a_database_newer_than_the_portable_core(tmp_path: Path) -> None:
    path = tmp_path / "storage.sqlite"
    database = await open_sqlite_database(path)
    await apply_sqlite_migrations(database)
    await database.run("UPDATE durable_schema SET version = ? WHERE singleton = 1", CURRENT_SQLITE_SCHEMA_VERSION + 1)
    await database.close()
    with pytest.raises(RuntimeError, match="is newer than supported version"):
        await open_sqlite_storage(path)


async def test_rolls_a_failed_migration_back_and_preserves_stored_data_for_a_successful_retry(tmp_path: Path) -> None:
    path = tmp_path / "storage.sqlite"
    storage = await open_sqlite_storage(path)
    await storage.commit(
        [
            {"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}},
            {
                "type": "entry",
                "value": {
                    "id": 2,
                    "conversationId": ROOT_CONVERSATION_ID,
                    "kind": "retained",
                    "data": {"retained": True},
                },  # type: ignore[typeddict-item]
            },
        ],
        context,
    )
    await storage.close(context)

    database = await open_sqlite_database(path)
    next_version = CURRENT_SQLITE_SCHEMA_VERSION + 1
    failed = [
        *SQLITE_MIGRATIONS,
        SqliteMigration(next_version, ("CREATE TABLE migration_probe (value TEXT) STRICT", "THIS IS NOT SQL")),
    ]
    with pytest.raises(Exception):  # noqa: B017 - a SQL syntax error
        await apply_sqlite_migrations(database, failed)
    assert await database.get("SELECT version FROM durable_schema WHERE singleton = 1") == {
        "version": CURRENT_SQLITE_SCHEMA_VERSION
    }
    assert await database.get(
        "SELECT count(*) AS count FROM sqlite_schema WHERE type = 'table' AND name = 'migration_probe'"
    ) == {"count": 0}

    successful = [
        *SQLITE_MIGRATIONS,
        SqliteMigration(next_version, ("CREATE TABLE migration_probe (value TEXT) STRICT",)),
    ]
    await apply_sqlite_migrations(database, successful)
    assert await database.get("SELECT version FROM durable_schema WHERE singleton = 1") == {"version": next_version}
    assert await database.get("SELECT record, commit_seq FROM entries WHERE id = 2") == {
        "record": _js(
            {"id": 2, "conversationId": ROOT_CONVERSATION_ID, "kind": "retained", "data": {"retained": True}}
        ),
        "commit_seq": 1,
    }
    assert await database.get("SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1") == {
        "next_id": "3",
        "next_seq": 2,
    }
    await database.close()
