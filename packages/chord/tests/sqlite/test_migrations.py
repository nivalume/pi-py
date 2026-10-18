"""Port of chord ``test/sqlite/migrations.test.ts``."""

from __future__ import annotations

import pytest

from pi_chord.sqlite import SqliteMigration, apply_sqlite_migrations, validate_sqlite_schema_prefix
from pi_chord.sqlite.stdlib import open_sqlite_database


async def test_tracks_independent_histories_and_skips_migrations_already_applied() -> None:
    database = await open_sqlite_database(":memory:")
    try:
        first = [SqliteMigration(1, ["CREATE TABLE notes (body TEXT)", "INSERT INTO notes VALUES ('kept')"])]
        await apply_sqlite_migrations(database, first, table="notes_schema")
        await apply_sqlite_migrations(database, first, table="notes_schema")
        await apply_sqlite_migrations(database, [], table="other_schema")
        await apply_sqlite_migrations(
            database,
            [*first, SqliteMigration(2, ["ALTER TABLE notes ADD COLUMN title TEXT"])],
            table="notes_schema",
        )
        assert await database.all("SELECT body, title FROM notes") == [{"body": "kept", "title": None}]
        assert await database.get("SELECT version FROM notes_schema") == {"version": 2}
        assert await database.get("SELECT version FROM other_schema") == {"version": 0}
        with pytest.raises(RuntimeError, match="SQLite schema version 2 is newer than supported version 1"):
            await apply_sqlite_migrations(database, first, table="notes_schema")
    finally:
        await database.close()


async def test_rejects_invalid_migration_versions_before_touching_the_database() -> None:
    database = await open_sqlite_database(":memory:")
    try:
        for versions in [[0], [2], [1, 3], [1, 1], [True]]:
            with pytest.raises(ValueError, match="contiguous versions starting at 1"):
                await apply_sqlite_migrations(
                    database,
                    [SqliteMigration(version, []) for version in versions],
                    table="app_schema",
                )
        assert await database.all("SELECT name FROM sqlite_schema") == []
    finally:
        await database.close()


async def test_validates_prefixes_and_migration_table_identifiers_before_executing_sql() -> None:
    database = await open_sqlite_database(":memory:")
    try:
        for prefix in ["", "1bad", "bad-prefix", "bad; DROP TABLE notes", "sqlite_app", "SQLITE_app", "bad\n", "é"]:
            with pytest.raises(ValueError):
                validate_sqlite_schema_prefix(prefix)
            with pytest.raises(ValueError):
                await apply_sqlite_migrations(database, [], table=prefix)
        for prefix in ["pi_", "_private", "app2"]:
            validate_sqlite_schema_prefix(prefix)
        assert await database.all("SELECT name FROM sqlite_schema") == []
    finally:
        await database.close()


async def test_rolls_initial_bootstrap_and_every_pending_migration_back_together() -> None:
    database = await open_sqlite_database(":memory:")
    try:
        failed = [
            SqliteMigration(
                1,
                [
                    "CREATE TABLE migration_first (value TEXT) STRICT",
                    "INSERT INTO migration_first (value) VALUES ('retained')",
                ],
            ),
            SqliteMigration(2, ["CREATE TABLE migration_second (value TEXT) STRICT", "THIS IS NOT SQL"]),
        ]
        with pytest.raises(Exception):  # noqa: B017 - any SQLite error
            await apply_sqlite_migrations(database, failed, table="app_schema")
        assert await database.get(
            "SELECT count(*) AS count FROM sqlite_schema"
            " WHERE name IN ('app_schema', 'migration_first', 'migration_second')"
        ) == {"count": 0}

        await apply_sqlite_migrations(
            database,
            [failed[0], SqliteMigration(2, ["CREATE TABLE migration_second (value TEXT) STRICT"])],
            table="app_schema",
        )
        assert await database.get("SELECT version FROM app_schema WHERE singleton = 1") == {"version": 2}
        assert await database.get("SELECT value FROM migration_first") == {"value": "retained"}
    finally:
        await database.close()
