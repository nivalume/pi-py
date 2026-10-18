"""Durable SQLite storage on Python's ``sqlite3`` (replaces upstream ``storage/sqlite/node.ts``).

Re-exports the :mod:`pi_chord.sqlite.stdlib` adapter, as upstream re-exports chord's Node adapter.
"""

from __future__ import annotations

import os

from pi_chord.sqlite import validate_sqlite_schema_prefix
from pi_chord.sqlite.stdlib import (
    SqliteDatabaseOptions,
    StdlibSqliteDatabase,
    StdlibSqliteTransaction,
    open_sqlite_database,
)

from .storage import SqliteStorage

__all__ = [
    "SqliteDatabaseOptions",
    "StdlibSqliteDatabase",
    "StdlibSqliteTransaction",
    "open_sqlite_database",
    "open_sqlite_storage",
]


async def open_sqlite_storage(
    path: str | os.PathLike[str],
    *,
    prefix: str | None = None,
    wal_auto_checkpoint_pages: int | None = None,
    busy_timeout_ms: int | None = None,
) -> SqliteStorage:
    """Open or create file-backed durable storage on ``sqlite3``, optionally under a table prefix.

    Upstream ``openNodeSqliteStorage(path, options)``: the connection settings are those of
    :class:`~pi_chord.sqlite.stdlib.SqliteDatabaseOptions` (their defaults when omitted), ``prefix`` is
    :meth:`SqliteStorage.open`'s. An invalid prefix raises before the database file is created.
    """
    if prefix is not None:
        validate_sqlite_schema_prefix(prefix)
    settings: dict[str, int] = {}
    if wal_auto_checkpoint_pages is not None:
        settings["wal_auto_checkpoint_pages"] = wal_auto_checkpoint_pages
    if busy_timeout_ms is not None:
        settings["busy_timeout_ms"] = busy_timeout_ms
    database = await open_sqlite_database(path, SqliteDatabaseOptions(**settings))
    return await SqliteStorage.open(database, prefix=prefix)
