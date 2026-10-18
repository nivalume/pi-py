"""Versioned schema migrations (port of chord ``sqlite/migrations.ts``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .database import SqliteDatabase, SqliteExecutor
from .schema import validate_sqlite_schema_prefix

__all__ = ["SqliteMigration", "apply_sqlite_migrations"]


@dataclass(frozen=True, slots=True)
class SqliteMigration:
    """One schema version: the statements that upgrade the previous version to ``version``."""

    version: int
    statements: Sequence[str]


async def apply_sqlite_migrations(
    database: SqliteDatabase,
    migrations: Sequence[SqliteMigration],
    *,
    table: str,
) -> None:
    """Apply all pending migrations atomically, tracking their version in the named metadata table.

    Versions must be contiguous from 1. Every pending migration and the metadata bootstrap run in
    one transaction, so a failure leaves the database as it was.
    """
    validate_sqlite_schema_prefix(table)
    for index, migration in enumerate(migrations):
        version = migration.version
        if version.__class__ is not int or version != index + 1:
            raise ValueError("SQLite migrations must have contiguous versions starting at 1")
    current_version = migrations[-1].version if migrations else 0

    async def migrate(transaction: SqliteExecutor) -> None:
        await transaction.exec(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
            "version INTEGER NOT NULL CHECK (version >= 0)"
            ") STRICT"
        )
        await transaction.run(f"INSERT OR IGNORE INTO {table} (singleton, version) VALUES (1, 0)")
        row = await transaction.get(f"SELECT version FROM {table} WHERE singleton = 1")
        if row is None:
            raise RuntimeError("SQLite schema metadata is missing")
        stored_version: int = row["version"]
        if stored_version > current_version:
            raise RuntimeError(
                f"SQLite schema version {stored_version} is newer than supported version {current_version}"
            )
        for migration in migrations:
            if migration.version <= stored_version:
                continue
            for statement in migration.statements:
                await transaction.exec(statement)
            await transaction.run(f"UPDATE {table} SET version = ? WHERE singleton = 1", migration.version)

    await database.transaction(migrate)
