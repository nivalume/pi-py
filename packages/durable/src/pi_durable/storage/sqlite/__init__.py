"""Portable SQLite storage (port of ``storage/sqlite/index.ts``).

:class:`SqliteStorage` runs over any :class:`~pi_chord.sqlite.SqliteDatabase` facade. This module does not
load a driver; :mod:`pi_durable.storage.sqlite.stdlib` opens storage on Python's ``sqlite3``
(upstream ``storage/sqlite/node.ts``).
"""

from __future__ import annotations

from pi_chord.sqlite import SqliteDatabase, SqliteExecutor, SqliteValue

from .migrations import (
    CURRENT_SQLITE_SCHEMA_VERSION,
    SQLITE_MIGRATIONS,
    SqliteMigration,
    apply_sqlite_migrations,
)
from .storage import SqliteStorage, SqliteStorageOptions

__all__ = [
    "CURRENT_SQLITE_SCHEMA_VERSION",
    "SQLITE_MIGRATIONS",
    "SqliteDatabase",
    "SqliteExecutor",
    "SqliteMigration",
    "SqliteStorage",
    "SqliteStorageOptions",
    "SqliteValue",
    "apply_sqlite_migrations",
]
