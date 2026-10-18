"""Versioned schema history of the durable SQLite storage (port of ``storage/sqlite/migrations.ts``).

The statements are upstream's, character for character (including their indentation, which SQLite keeps
in ``sqlite_schema``), so a database migrated by TypeScript pi-durable and one migrated here are
interchangeable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pi_chord.sqlite import SqliteDatabase, SqliteMigration
from pi_chord.sqlite import apply_sqlite_migrations as _apply_migrations

from .schema import DEFAULT_SQLITE_SCHEMA, SqliteSchema

__all__ = [
    "CURRENT_SQLITE_SCHEMA_VERSION",
    "SQLITE_MIGRATIONS",
    "SqliteMigration",
    "apply_sqlite_migrations",
    "apply_sqlite_migrations_for_schema",
    "create_sqlite_migrations",
]


def create_sqlite_migrations(schema: SqliteSchema) -> tuple[SqliteMigration, ...]:
    """Build the immutable, ordered schema history for a set of SQLite identifiers."""
    t = schema.tables
    i = schema.indexes
    # next_id is TEXT because node:sqlite rejects INTEGER results outside JavaScript's safe integer range.
    initial = (
        f"CREATE TABLE {t.durable_metadata} (\n\t\t\tsingleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n\t\t\tnext_id TEXT NOT NULL,\n\t\t\tnext_seq INTEGER NOT NULL\n\t\t) STRICT",
        f"INSERT INTO {t.durable_metadata} (singleton, next_id, next_seq) VALUES (1, '2', 1)",
        f"CREATE TABLE {t.record_ids} (\n\t\t\tid INTEGER PRIMARY KEY,\n\t\t\trecord_type TEXT NOT NULL CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))\n\t\t) STRICT",
        f"CREATE TABLE {t.conversations} (\n\t\t\tid INTEGER PRIMARY KEY,\n\t\t\towner_conversation_id INTEGER,\n\t\t\towner_task_id INTEGER,\n\t\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t\t) STRICT",
        f"CREATE INDEX {i.conversations_by_owner_conversation} ON {t.conversations} (owner_conversation_id, id)",
        f"CREATE INDEX {i.conversations_by_owner_task} ON {t.conversations} (owner_task_id, id)",
        f"CREATE TABLE {t.entries} (\n\t\t\tid INTEGER PRIMARY KEY,\n\t\t\tconversation_id INTEGER NOT NULL,\n\t\t\thead INTEGER,\n\t\t\tcommit_seq INTEGER NOT NULL,\n\t\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t\t) STRICT",
        f"CREATE INDEX {i.entries_by_conversation} ON {t.entries} (conversation_id, id DESC)",
        f"CREATE INDEX {i.entry_heads_by_conversation} ON {t.entries} (conversation_id, id DESC) WHERE head IS NOT NULL",
        f"CREATE TABLE {t.tasks} (\n\t\t\tid INTEGER PRIMARY KEY,\n\t\t\tconversation_id INTEGER NOT NULL,\n\t\t\tkind TEXT NOT NULL,\n\t\t\tstatus TEXT NOT NULL CHECK (status IN ('pending', 'running', 'waiting', 'completing', 'terminal')),\n\t\t\tabort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),\n\t\t\tbackground INTEGER NOT NULL CHECK (background IN (0, 1)),\n\t\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t\t) STRICT",
        f"CREATE INDEX {i.tasks_by_status} ON {t.tasks} (status, id)",
        f"CREATE INDEX {i.tasks_by_conversation} ON {t.tasks} (conversation_id, id)",
        f"CREATE INDEX {i.tasks_by_kind} ON {t.tasks} (kind, id)",
        f"CREATE INDEX {i.tasks_by_abort_requested} ON {t.tasks} (abort_requested, id)",
        f"CREATE INDEX {i.tasks_by_background} ON {t.tasks} (background, id)",
        f"CREATE TABLE {t.submissions} (\n\t\t\tid INTEGER PRIMARY KEY,\n\t\t\tconversation_id INTEGER NOT NULL,\n\t\t\trequest_id TEXT,\n\t\t\tstatus TEXT NOT NULL CHECK (status IN ('queued', 'placed', 'done', 'unanswered')),\n\t\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t\t) STRICT",
        f"CREATE INDEX {i.submissions_by_request} ON {t.submissions} (conversation_id, request_id)",
        f"CREATE INDEX {i.submissions_by_conversation} ON {t.submissions} (conversation_id, id)",
        f"CREATE INDEX {i.submissions_by_status} ON {t.submissions} (status, id)",
        f"CREATE TABLE {t.documents} (\n\t\t\tid INTEGER PRIMARY KEY,\n\t\t\tkind TEXT NOT NULL,\n\t\t\tfamily INTEGER NOT NULL CHECK (family IN (0, 1)),\n\t\t\tkey_value TEXT NOT NULL,\n\t\t\tscope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),\n\t\t\towner_id INTEGER NOT NULL,\n\t\t\tcreated_at INTEGER NOT NULL,\n\t\t\tretired_at INTEGER,\n\t\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t\t) STRICT",
        f"CREATE INDEX {i.documents_by_address}\n\t\t\tON {t.documents} (kind, scope_kind, owner_id, family, key_value, created_at DESC, retired_at)",
        f"CREATE INDEX {i.documents_by_scope} ON {t.documents} (scope_kind, owner_id, id)",
        f"CREATE INDEX {i.documents_by_scope_kind} ON {t.documents} (scope_kind, owner_id, kind, id)",
        f"CREATE TABLE {t.document_revisions} (\n\t\t\tdocument_id INTEGER NOT NULL,\n\t\t\tseq INTEGER NOT NULL,\n\t\t\tkind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),\n\t\t\tversion INTEGER NOT NULL,\n\t\t\tcontent TEXT NOT NULL CHECK (json_valid(content)),\n\t\t\tPRIMARY KEY (document_id, seq)\n\t\t) STRICT",
        f"CREATE INDEX {i.document_revisions_by_kind} ON {t.document_revisions} (document_id, kind, seq DESC)",
    )
    # Version 2 adds the `sleeping` task status. SQLite cannot change a CHECK constraint in place, so the table is
    # rebuilt.
    rebuilt = f"{t.tasks}_v2"
    sleeping = (
        f"CREATE TABLE {rebuilt} (\n\t\t\tid INTEGER PRIMARY KEY,\n\t\t\tconversation_id INTEGER NOT NULL,\n\t\t\tkind TEXT NOT NULL,\n\t\t\tstatus TEXT NOT NULL CHECK (status IN ('pending', 'running', 'waiting', 'sleeping', 'completing', 'terminal')),\n\t\t\tabort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),\n\t\t\tbackground INTEGER NOT NULL CHECK (background IN (0, 1)),\n\t\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t\t) STRICT",
        f"INSERT INTO {rebuilt} (id, conversation_id, kind, status, abort_requested, background, record)\n\t\t\tSELECT id, conversation_id, kind, status, abort_requested, background, record FROM {t.tasks}",
        f"DROP TABLE {t.tasks}",
        f"ALTER TABLE {rebuilt} RENAME TO {t.tasks}",
        f"CREATE INDEX {i.tasks_by_status} ON {t.tasks} (status, id)",
        f"CREATE INDEX {i.tasks_by_conversation} ON {t.tasks} (conversation_id, id)",
        f"CREATE INDEX {i.tasks_by_kind} ON {t.tasks} (kind, id)",
        f"CREATE INDEX {i.tasks_by_abort_requested} ON {t.tasks} (abort_requested, id)",
        f"CREATE INDEX {i.tasks_by_background} ON {t.tasks} (background, id)",
    )
    return (SqliteMigration(1, initial), SqliteMigration(2, sleeping))


SQLITE_MIGRATIONS: Final = create_sqlite_migrations(DEFAULT_SQLITE_SCHEMA)

CURRENT_SQLITE_SCHEMA_VERSION: Final = SQLITE_MIGRATIONS[-1].version if SQLITE_MIGRATIONS else 0


async def apply_sqlite_migrations(
    database: SqliteDatabase, migrations: Sequence[SqliteMigration] = SQLITE_MIGRATIONS
) -> None:
    """Apply all pending schema migrations atomically."""
    await apply_sqlite_migrations_for_schema(database, DEFAULT_SQLITE_SCHEMA, migrations)


async def apply_sqlite_migrations_for_schema(
    database: SqliteDatabase, schema: SqliteSchema, migrations: Sequence[SqliteMigration] | None = None
) -> None:
    """Apply migrations rendered for a particular schema (by default, the whole history for it)."""
    if migrations is None:
        migrations = create_sqlite_migrations(schema)
    await _apply_migrations(database, migrations, table=schema.tables.durable_schema)
