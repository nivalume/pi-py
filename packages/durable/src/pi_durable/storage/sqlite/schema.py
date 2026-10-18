"""Identifiers of the durable SQLite schema (port of ``storage/sqlite/schema.ts``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pi_chord.sqlite import validate_sqlite_schema_prefix

__all__ = [
    "DEFAULT_SQLITE_SCHEMA",
    "SqliteIndexNames",
    "SqliteSchema",
    "SqliteTableNames",
    "create_sqlite_schema",
    "validate_sqlite_schema_prefix",
]


@dataclass(frozen=True, slots=True)
class SqliteTableNames:
    """Table names (upstream ``TABLE_NAMES``, keys in snake_case).

    Immutable identifier history for migrations and runtime queries: never change or remove an existing
    value; a rename adds a new entry so historical migrations keep rendering their original object names.
    """

    durable_schema: str = "durable_schema"
    durable_metadata: str = "durable_metadata"
    record_ids: str = "record_ids"
    conversations: str = "conversations"
    entries: str = "entries"
    tasks: str = "tasks"
    submissions: str = "submissions"
    documents: str = "documents"
    document_revisions: str = "document_revisions"


@dataclass(frozen=True, slots=True)
class SqliteIndexNames:
    """Index names (upstream ``INDEX_NAMES``); the same immutability rule as :class:`SqliteTableNames`."""

    conversations_by_owner_conversation: str = "conversations_by_owner_conversation"
    conversations_by_owner_task: str = "conversations_by_owner_task"
    entries_by_conversation: str = "entries_by_conversation"
    entry_heads_by_conversation: str = "entry_heads_by_conversation"
    tasks_by_status: str = "tasks_by_status"
    tasks_by_conversation: str = "tasks_by_conversation"
    tasks_by_kind: str = "tasks_by_kind"
    tasks_by_abort_requested: str = "tasks_by_abort_requested"
    tasks_by_background: str = "tasks_by_background"
    submissions_by_request: str = "submissions_by_request"
    submissions_by_conversation: str = "submissions_by_conversation"
    submissions_by_status: str = "submissions_by_status"
    documents_by_address: str = "documents_by_address"
    documents_by_scope: str = "documents_by_scope"
    documents_by_scope_kind: str = "documents_by_scope_kind"
    document_revisions_by_kind: str = "document_revisions_by_kind"


@dataclass(frozen=True, slots=True)
class SqliteSchema:
    """The identifiers used by every migration and storage query."""

    tables: SqliteTableNames
    indexes: SqliteIndexNames


def _prefixed(names: SqliteTableNames | SqliteIndexNames, prefix: str) -> dict[str, str]:
    return {field: f"{prefix}{getattr(names, field)}" for field in names.__dataclass_fields__}


def create_sqlite_schema(prefix: str | None = None) -> SqliteSchema:
    """Build the identifiers used by every migration and storage query, optionally under a table prefix."""
    if prefix is not None:
        validate_sqlite_schema_prefix(prefix)
    actual = prefix or ""
    return SqliteSchema(
        SqliteTableNames(**_prefixed(SqliteTableNames(), actual)),
        SqliteIndexNames(**_prefixed(SqliteIndexNames(), actual)),
    )


DEFAULT_SQLITE_SCHEMA: Final = create_sqlite_schema()
