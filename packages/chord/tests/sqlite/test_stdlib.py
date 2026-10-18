"""Port of chord ``test/sqlite/node.test.ts`` for the stdlib ``sqlite3`` adapter, plus adapter specifics."""

from __future__ import annotations

import asyncio
import gc
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from pi_chord.sqlite import SqliteDatabase, SqliteExecutor
from pi_chord.sqlite.stdlib import SqliteDatabaseOptions, StdlibSqliteDatabase, open_sqlite_database


async def until(predicate: object, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():  # type: ignore[operator]
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


async def test_configures_connection_defaults_and_accepts_explicit_settings() -> None:
    for options in [None, SqliteDatabaseOptions(busy_timeout_ms=123, wal_auto_checkpoint_pages=0)]:
        database = await open_sqlite_database(":memory:", options)
        try:
            assert await database.get("PRAGMA busy_timeout") == {
                "timeout": 5_000 if options is None else options.busy_timeout_ms
            }
            assert await database.get("PRAGMA wal_autocheckpoint") == {
                "wal_autocheckpoint": 1_000 if options is None else options.wal_auto_checkpoint_pages
            }
            assert await database.get("PRAGMA synchronous") == {"synchronous": 1}
        finally:
            await database.close()


async def test_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="busy_timeout_ms"):
        await open_sqlite_database(":memory:", SqliteDatabaseOptions(busy_timeout_ms=-1))
    with pytest.raises(ValueError, match="wal_auto_checkpoint_pages"):
        await open_sqlite_database(":memory:", SqliteDatabaseOptions(wal_auto_checkpoint_pages="1; DROP"))  # type: ignore[arg-type]


async def test_commits_work_done_through_the_transaction_handle_and_closes_idempotently() -> None:
    database = await open_sqlite_database(":memory:")

    async def work(transaction: SqliteExecutor) -> None:
        await transaction.exec("CREATE TABLE async_probe (value INTEGER)")
        await transaction.run("INSERT INTO async_probe (value) VALUES (?)", 1)

    await database.transaction(work)
    assert await database.get("SELECT value FROM async_probe") == {"value": 1}
    await database.close()
    await database.close()


async def test_returns_the_callback_result() -> None:
    database = await open_sqlite_database(":memory:")

    async def work(transaction: SqliteExecutor) -> str:
        row = await transaction.get("SELECT 'done' AS value")
        assert row is not None
        return row["value"]

    assert await database.transaction(work) == "done"
    await database.close()


async def test_serializes_concurrent_transactions() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE transaction_queue (value INTEGER)")
    first_started = asyncio.Event()
    first_gate = asyncio.Event()

    async def first_work(transaction: SqliteExecutor) -> None:
        await transaction.exec("INSERT INTO transaction_queue (value) VALUES (1)")
        first_started.set()
        await first_gate.wait()

    first = database.transaction(first_work)
    await first_started.wait()

    second_started = False

    async def second_work(transaction: SqliteExecutor) -> None:
        nonlocal second_started
        second_started = True
        await transaction.exec("INSERT INTO transaction_queue (value) VALUES (2)")

    second = database.transaction(second_work)
    await asyncio.sleep(0.02)
    assert not second_started

    first_gate.set()
    await asyncio.gather(first, second)
    assert await database.all("SELECT value FROM transaction_queue ORDER BY value") == [{"value": 1}, {"value": 2}]
    await database.close()


async def test_runs_operations_in_call_order_whether_they_start_immediately_or_wait() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE call_order (value INTEGER)")

    # Operations called during a transaction must neither see its uncommitted rows nor join its rollback.
    async def work(handle: SqliteExecutor) -> None:
        await handle.run("INSERT INTO call_order (value) VALUES (?)", 1)
        await asyncio.sleep(0)
        raise RuntimeError("roll back")

    transaction = database.transaction(work)
    before_write = database.all("SELECT value FROM call_order ORDER BY value")
    write = database.run("INSERT INTO call_order (value) VALUES (?)", 2)
    after_write = database.all("SELECT value FROM call_order ORDER BY value")
    with pytest.raises(RuntimeError, match="roll back"):
        await transaction
    await write
    assert await before_write == []
    assert await after_write == [{"value": 2}]

    await database.close()


async def test_queues_ordinary_operations_behind_an_active_transaction() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE operation_queue (value INTEGER)")
    transaction_started = asyncio.Event()
    transaction_gate = asyncio.Event()

    async def work(transaction: SqliteExecutor) -> None:
        await transaction.exec("INSERT INTO operation_queue (value) VALUES (1)")
        transaction_started.set()
        await transaction_gate.wait()

    pending = database.transaction(work)
    await transaction_started.wait()

    write = database.exec("INSERT INTO operation_queue (value) VALUES (2)")
    read = database.all("SELECT value FROM operation_queue ORDER BY value")
    await asyncio.sleep(0.02)
    assert not write.done()
    assert not read.done()

    transaction_gate.set()
    await pending
    await write
    assert await read == [{"value": 1}, {"value": 2}]
    await database.close()


async def test_queues_database_calls_made_synchronously_by_a_transaction_that_started_immediately() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE barrier_probe (value INTEGER)")
    outside: list[asyncio.Future[None]] = []

    async def work(handle: SqliteExecutor) -> None:
        # Misuse: this call must wait for the transaction instead of joining it.
        outside.append(database.run("INSERT INTO barrier_probe (value) VALUES (?)", 2))
        await handle.run("INSERT INTO barrier_probe (value) VALUES (?)", 1)
        await asyncio.sleep(0)
        raise RuntimeError("roll back")

    with pytest.raises(RuntimeError, match="roll back"):
        await database.transaction(work)
    await outside[0]
    assert await database.all("SELECT value FROM barrier_probe") == [{"value": 2}]
    await database.close()


async def test_rejects_a_transaction_handle_used_after_its_transaction_settles() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE stale_probe (value INTEGER)")
    handles: list[SqliteExecutor] = []

    async def work(transaction: SqliteExecutor) -> None:
        handles.append(transaction)
        await transaction.run("INSERT INTO stale_probe (value) VALUES (?)", 1)

    await database.transaction(work)
    stale = "SQLite transaction handle is no longer active"
    with pytest.raises(RuntimeError, match=stale):
        await handles[0].exec("INSERT INTO stale_probe (value) VALUES (2)")
    with pytest.raises(RuntimeError, match=stale):
        await handles[0].run("INSERT INTO stale_probe (value) VALUES (?)", 3)
    assert await database.all("SELECT value FROM stale_probe") == [{"value": 1}]
    await database.close()


async def test_does_not_preserve_the_callback_error_when_rollback_itself_fails() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE rollback_probe (value INTEGER)")

    async def work(transaction: SqliteExecutor) -> None:
        await transaction.exec("INSERT INTO rollback_probe (value) VALUES (1)")
        await transaction.exec("COMMIT")
        raise RuntimeError("rejected after an escaped commit")

    with pytest.raises(ExceptionGroup, match="rollback failed") as raised:
        await database.transaction(work)
    callback_error, rollback_error = raised.value.exceptions
    assert str(callback_error) == "rejected after an escaped commit"
    assert isinstance(rollback_error, sqlite3.OperationalError)
    assert await database.get("SELECT value FROM rollback_probe") == {"value": 1}
    await database.close()


# --- stdlib adapter specifics ---


async def test_reports_a_failed_commit_after_rolling_back() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec(
        "CREATE TABLE parent (id INTEGER PRIMARY KEY);"
        "CREATE TABLE child (parent_id INTEGER REFERENCES parent (id) DEFERRABLE INITIALLY DEFERRED);"
        "PRAGMA foreign_keys = ON;"
    )

    async def work(transaction: SqliteExecutor) -> None:
        await transaction.run("INSERT INTO child (parent_id) VALUES (?)", 7)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        await database.transaction(work)
    assert await database.all("SELECT * FROM child") == []
    # The worker left the transaction: ordinary operations and new transactions proceed.
    await database.run("INSERT INTO parent (id) VALUES (?)", 7)
    await database.transaction(work)
    assert await database.all("SELECT * FROM child") == [{"parent_id": 7}]
    await database.close()


async def test_exec_runs_several_statements_inside_a_transaction_without_committing() -> None:
    database = await open_sqlite_database(":memory:")

    async def work(transaction: SqliteExecutor) -> None:
        await transaction.exec(
            "CREATE TABLE script (value TEXT); INSERT INTO script VALUES ('a;b'); -- trailing; comment\n"
            "CREATE TRIGGER script_copy AFTER INSERT ON script BEGIN INSERT INTO script_log VALUES (new.value); END;"
            "CREATE TABLE script_log (value TEXT);\n"
            "INSERT INTO script VALUES ('c')"
        )
        assert await transaction.all("SELECT value FROM script ORDER BY value") == [{"value": "a;b"}, {"value": "c"}]
        assert await transaction.all("SELECT value FROM script_log") == [{"value": "c"}]
        raise RuntimeError("roll back")

    with pytest.raises(RuntimeError, match="roll back"):
        await database.transaction(work)
    assert await database.all("SELECT name FROM sqlite_schema") == []
    await database.exec("")
    await database.exec("  -- only a comment")
    with pytest.raises(sqlite3.OperationalError):
        await database.exec("CREATE TABLE ok (value); THIS IS NOT SQL;")
    assert await database.all("SELECT name FROM sqlite_schema") == [{"name": "ok"}]
    await database.close()


async def test_maps_values_like_upstream() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE value_probe (value ANY) STRICT")
    values = [None, 0, -1, 2**63 - 1, -(2**63), 1.5, "text", "", b"\x00\xff", bytearray(b"ab"), memoryview(b"cd")]
    for value in values:
        await database.run("INSERT INTO value_probe (value) VALUES (?)", value)  # type: ignore[arg-type]
    rows = await database.all("SELECT value FROM value_probe ORDER BY rowid")
    assert [row["value"] for row in rows] == [
        None,
        0,
        -1,
        2**63 - 1,
        -(2**63),
        1.5,
        "text",
        "",
        b"\x00\xff",
        b"ab",
        b"cd",
    ]
    # Integers past JavaScript's safe range read back exactly.
    assert await database.get("SELECT ? + 1 AS value", 2**60) == {"value": 2**60 + 1}
    with pytest.raises(TypeError, match="parameter 2"):
        await database.run("INSERT INTO value_probe (value) VALUES (?), (?)", 1, True)
    with pytest.raises(TypeError, match="parameter 1"):
        await database.get("SELECT ?", object())  # type: ignore[arg-type]
    with pytest.raises(OverflowError):
        await database.run("INSERT INTO value_probe (value) VALUES (?)", 2**63)
    assert await database.get("SELECT count(*) AS count FROM value_probe") == {"count": len(values)}
    assert await database.get("SELECT value FROM value_probe WHERE value = ?", "missing") is None
    assert await database.all("SELECT value FROM value_probe WHERE value = ?", "missing") == []
    assert await database.get("UPDATE value_probe SET value = value WHERE 0") is None
    await database.close()


async def test_get_resets_its_statement_so_an_unfinished_read_holds_no_lock(tmp_path: Path) -> None:
    path = tmp_path / "lock.db"
    database = await open_sqlite_database(path)
    await database.exec("CREATE TABLE many (value INTEGER); INSERT INTO many VALUES (1), (2), (3)")
    assert await database.get("SELECT value FROM many ORDER BY value") == {"value": 1}
    other = sqlite3.connect(path, timeout=0)
    try:
        assert other.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0  # not busy
    finally:
        other.close()
    await database.close()


async def test_rejects_operations_after_close() -> None:
    database = await open_sqlite_database(":memory:")
    await database.close()
    with pytest.raises(sqlite3.ProgrammingError):
        await database.get("SELECT 1")

    async def work(transaction: SqliteExecutor) -> None:
        pass

    with pytest.raises(sqlite3.ProgrammingError):
        await database.transaction(work)
    await database.close()


async def test_operations_called_before_close_run_first() -> None:
    database = await open_sqlite_database(":memory:")
    read = database.get("SELECT 42 AS value")
    closing = database.close()
    second_close = database.close()
    assert await read == {"value": 42}
    await closing
    await second_close


async def test_creates_the_parent_directory_and_uses_wal_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "store.db"
    database = await open_sqlite_database(path)
    assert await database.get("PRAGMA journal_mode") == {"journal_mode": "wal"}
    await database.exec("CREATE TABLE disk (value TEXT)")
    await database.run("INSERT INTO disk VALUES (?)", "kept")
    await database.close()
    wal = path.with_name("store.db-wal")
    # Close checkpoints with TRUNCATE; SQLite then removes the WAL of the last connection.
    assert not wal.exists() or wal.stat().st_size == 0
    reopened = await open_sqlite_database(str(path))
    assert await reopened.all("SELECT value FROM disk") == [{"value": "kept"}]
    await reopened.close()


async def test_reports_open_failures(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(OSError):
        await open_sqlite_database(blocker / "store.db")
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        await open_sqlite_database(garbage)


async def test_reports_a_busy_begin_and_stays_usable(tmp_path: Path) -> None:
    path = tmp_path / "busy.db"
    database = await open_sqlite_database(path, SqliteDatabaseOptions(busy_timeout_ms=10))
    await database.exec("CREATE TABLE busy (value INTEGER)")
    other = sqlite3.connect(path, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    ran = False

    async def work(transaction: SqliteExecutor) -> None:
        nonlocal ran
        ran = True

    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            await database.transaction(work)
        assert not ran
    finally:
        other.execute("ROLLBACK")
        other.close()
    await database.transaction(work)
    assert ran
    await database.close()


async def test_does_not_block_the_event_loop_while_sqlite_works() -> None:
    database = await open_sqlite_database(":memory:")
    ticks = 0
    stop = False

    async def ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.001)

    task = asyncio.ensure_future(ticker())
    await asyncio.sleep(0)
    row = await database.get(
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 2000000) SELECT count(*) AS c FROM n"
    )
    stop = True
    await task
    assert row == {"c": 2_000_000}
    assert ticks > 5
    await database.close()


async def test_a_cancelled_transaction_rolls_back_and_the_database_stays_usable() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE cancel_probe (value INTEGER)")
    inside = asyncio.Event()

    async def work(transaction: SqliteExecutor) -> None:
        await transaction.run("INSERT INTO cancel_probe VALUES (?)", 1)
        inside.set()
        await asyncio.Event().wait()

    task = database.transaction(work)
    await inside.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await database.all("SELECT value FROM cancel_probe") == []

    # Cancelled before its callback starts: the queued transaction never runs it and releases the worker.
    gate = asyncio.Event()

    async def hold(transaction: SqliteExecutor) -> None:
        await gate.wait()

    ran = False

    async def never(transaction: SqliteExecutor) -> None:
        nonlocal ran
        ran = True

    holding = database.transaction(hold)
    queued = database.transaction(never)
    await asyncio.sleep(0.01)
    queued.cancel()
    gate.set()
    await holding
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert not ran
    assert await database.get("SELECT 1 AS one") == {"one": 1}
    await database.close()


async def test_skips_a_queued_statement_whose_caller_was_cancelled() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE skip_probe (value INTEGER)")
    gate = asyncio.Event()

    async def hold(transaction: SqliteExecutor) -> None:
        await gate.wait()

    holding = database.transaction(hold)
    write = database.run("INSERT INTO skip_probe VALUES (?)", 1)
    await asyncio.sleep(0.01)
    write.cancel()
    gate.set()
    await holding
    assert await database.all("SELECT value FROM skip_probe") == []
    await database.close()


async def test_concurrent_handle_operations_run_in_call_order() -> None:
    database = await open_sqlite_database(":memory:")
    await database.exec("CREATE TABLE concurrent (value INTEGER)")

    async def work(transaction: SqliteExecutor) -> list[int]:
        writes = [transaction.run("INSERT INTO concurrent VALUES (?)", index) for index in range(100)]
        rows = transaction.all("SELECT value FROM concurrent ORDER BY rowid")
        await asyncio.gather(*writes)
        return [row["value"] for row in await rows]

    assert await database.transaction(work) == list(range(100))
    reads = await asyncio.gather(*(database.get("SELECT ? AS value", index) for index in range(500)))
    assert reads == [{"value": index} for index in range(500)]
    await database.close()


async def test_a_dropped_database_stops_its_worker() -> None:
    def workers() -> int:
        return sum(thread.name == "pi-chord-sqlite" for thread in threading.enumerate())

    before = workers()
    database = await open_sqlite_database(":memory:")
    assert workers() == before + 1
    del database
    gc.collect()
    await until(lambda: workers() == before)


async def test_satisfies_the_portable_protocol() -> None:
    database: SqliteDatabase = await open_sqlite_database(":memory:")
    assert isinstance(database, StdlibSqliteDatabase)
    await database.close()


async def test_stays_consistent_under_concurrent_and_cancelled_transactions(tmp_path: Path) -> None:
    database = await open_sqlite_database(tmp_path / "stress.db")
    await database.exec("CREATE TABLE counter (value INTEGER NOT NULL); INSERT INTO counter VALUES (0)")
    committed = 0

    async def bump(transaction: SqliteExecutor) -> None:
        row = await transaction.get("SELECT value FROM counter")
        assert row is not None
        await asyncio.sleep(0)
        await transaction.run("UPDATE counter SET value = ?", row["value"] + 1)

    async def client(index: int) -> None:
        nonlocal committed
        for step in range(20):
            task = database.transaction(bump)
            if (index + step) % 7 == 0:
                await asyncio.sleep(0)
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                continue
            committed += 1

    await asyncio.wait_for(asyncio.gather(*(client(index) for index in range(16))), 30)
    row = await database.get("SELECT value FROM counter")
    assert row is not None
    # A cancellation can land after COMMIT ran but before the caller resumed; every other one rolled back.
    assert committed <= row["value"] <= 16 * 20
    assert committed > 200
    await database.close()
