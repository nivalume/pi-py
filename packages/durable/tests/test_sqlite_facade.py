"""Port of ``test/sqlite-facade.test.ts``.

Dropped: the Cloudflare Durable Object adapter re-export (no Cloudflare runtime in Python). Upstream counts
``node:sqlite`` prepares; ``sqlite3`` caches prepared statements by SQL text itself, so the Python test checks
the precondition that makes the cache hit: storage statements are constant texts with bound values.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

import pytest

import pi_chord.sqlite.stdlib as chord_stdlib
from pi_chord.context import BACKGROUND_CONTEXT
from pi_chord.sqlite import SqliteExecutor, SqliteRow, SqliteValue
from pi_durable import ROOT_CONVERSATION_ID
from pi_durable.ids import id_from_number
from pi_durable.storage.sqlite import SqliteStorage
from pi_durable.storage.sqlite.stdlib import StdlibSqliteDatabase, open_sqlite_database

T = TypeVar("T")
context = BACKGROUND_CONTEXT


class RecordingExecutor:
    def __init__(self, delegate: SqliteExecutor, statements: list[str]) -> None:
        self._delegate = delegate
        self._statements = statements

    def exec(self, sql: str) -> Awaitable[None]:
        self._statements.append(sql)
        return self._delegate.exec(sql)

    def run(self, sql: str, *params: SqliteValue) -> Awaitable[None]:
        self._statements.append(sql)
        return self._delegate.run(sql, *params)

    def get(self, sql: str, *params: SqliteValue) -> Awaitable[SqliteRow | None]:
        self._statements.append(sql)
        return self._delegate.get(sql, *params)

    def all(self, sql: str, *params: SqliteValue) -> Awaitable[list[SqliteRow]]:
        self._statements.append(sql)
        return self._delegate.all(sql, *params)


class RecordingDatabase(RecordingExecutor):
    def __init__(self, delegate: StdlibSqliteDatabase) -> None:
        self.statements: list[str] = []
        super().__init__(delegate, self.statements)
        self._database = delegate

    def transaction(self, callback: Callable[[SqliteExecutor], Awaitable[T]]) -> Awaitable[T]:
        return self._database.transaction(lambda handle: callback(RecordingExecutor(handle, self.statements)))

    def close(self) -> Awaitable[None]:
        return self._database.close()


class ControlledSettlementDatabase:
    def __init__(self, delegate: StdlibSqliteDatabase) -> None:
        self._delegate = delegate
        self._mode: Literal["immediate", "delay", "reject"] = "immediate"
        self._pending: Callable[[], None] | None = None

    def exec(self, sql: str) -> Awaitable[None]:
        return self._delegate.exec(sql)

    def run(self, sql: str, *params: SqliteValue) -> Awaitable[None]:
        return self._delegate.run(sql, *params)

    def get(self, sql: str, *params: SqliteValue) -> Awaitable[SqliteRow | None]:
        return self._delegate.get(sql, *params)

    def all(self, sql: str, *params: SqliteValue) -> Awaitable[list[SqliteRow]]:
        return self._delegate.all(sql, *params)

    def transaction(self, callback: Callable[[SqliteExecutor], Awaitable[T]]) -> Awaitable[T]:
        mode = self._mode
        self._mode = "immediate"
        if mode == "immediate":
            return self._delegate.transaction(callback)

        async def controlled(transaction: SqliteExecutor) -> T:
            value = await callback(transaction)
            if mode == "reject":
                raise RuntimeError("controlled settlement rejection")
            return value

        settlement = self._delegate.transaction(controlled)
        result: asyncio.Future[T] = asyncio.get_running_loop().create_future()

        def settle() -> None:
            def relay(done: asyncio.Future[T]) -> None:
                if done.exception() is not None:
                    result.set_exception(done.exception())  # type: ignore[arg-type]
                else:
                    result.set_result(done.result())

            settlement.add_done_callback(relay)

        self._pending = settle
        return result

    def close(self) -> Awaitable[None]:
        return self._delegate.close()

    def control_next_settlement(self, mode: Literal["delay", "reject"]) -> None:
        if self._pending is not None:
            raise RuntimeError("A settlement is already pending")
        self._mode = mode

    def settle(self) -> None:
        settle = self._pending
        if settle is None:
            raise RuntimeError("No settlement is pending")
        self._pending = None
        settle()


async def _yield(times: int = 1) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


def test_re_exports_the_chord_driver_classes_and_opener_without_wrapping_them() -> None:
    assert StdlibSqliteDatabase is chord_stdlib.StdlibSqliteDatabase
    assert open_sqlite_database is chord_stdlib.open_sqlite_database


async def test_issues_storage_statements_as_constant_texts_reused_across_transactions() -> None:
    database = RecordingDatabase(await open_sqlite_database(":memory:"))
    storage = await SqliteStorage.open(database)  # type: ignore[arg-type]
    await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)
    before = len(database.statements)
    await storage.commit(
        [
            {
                "type": "entry",
                "value": {"id": id_from_number(index + 2), "conversationId": ROOT_CONVERSATION_ID, "kind": "cached"},
            }
            for index in range(100)
        ],
        context,
    )
    hundred = database.statements[before:]
    assert len(hundred) > 300
    # Every value is bound: 300 statements of a 100-entry commit share a handful of SQL texts.
    assert len(set(hundred)) <= 6
    first = await storage.entry(id_from_number(2), context)
    assert first is not None and first["entry"]["kind"] == "cached"
    known = set(database.statements)
    await storage.commit(
        [
            {
                "type": "entry",
                "value": {"id": id_from_number(102), "conversationId": ROOT_CONVERSATION_ID, "kind": "cached-again"},
            }
        ],
        context,
    )
    again = await storage.entry(id_from_number(102), context)
    assert again is not None and again["entry"]["kind"] == "cached-again"
    with pytest.raises(RuntimeError, match="ID 1 already belongs to conversation"):
        await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)
    first = await storage.entry(id_from_number(2), context)
    assert first is not None and first["entry"]["kind"] == "cached"
    assert set(database.statements) <= known
    await storage.close(context)


async def test_reads_storage_commits_in_call_order() -> None:
    storage = await SqliteStorage.open(await open_sqlite_database(":memory:"))
    commit = asyncio.create_task(
        storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)
    )
    read = asyncio.create_task(storage.conversation(ROOT_CONVERSATION_ID, context))
    await commit
    assert await read == {"id": ROOT_CONVERSATION_ID}
    await storage.close(context)


async def test_lets_admitted_multi_query_reads_finish_before_storage_closes() -> None:
    storage = await SqliteStorage.open(await open_sqlite_database(":memory:"))
    entry_id = id_from_number(2)
    await storage.commit(
        [
            {"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}},
            {"type": "entry", "value": {"id": entry_id, "conversationId": ROOT_CONVERSATION_ID, "kind": "probe"}},
        ],
        context,
    )
    scan = asyncio.create_task(storage.scan_entries({"conversationId": ROOT_CONVERSATION_ID}, 10, None, context))
    entry = asyncio.create_task(storage.visible_entry(ROOT_CONVERSATION_ID, entry_id, context))
    head = asyncio.create_task(storage.find_latest_head_marker(ROOT_CONVERSATION_ID, None, context))
    closed = asyncio.create_task(storage.close(context))
    # A repeated close settles only when the database is closed.
    closed_again = asyncio.create_task(storage.close(context))
    assert [item["id"] for item in (await scan)["items"]] == [entry_id]
    stored = await entry
    assert stored is not None and stored["entry"]["kind"] == "probe"
    assert await head is None
    await closed
    await closed_again
    with pytest.raises(RuntimeError, match="SqliteStorage is closed"):
        await storage.scan_entries({"conversationId": ROOT_CONVERSATION_ID}, 10, None, context)


async def test_awaits_async_transaction_settlement_and_adopts_ids_only_after_success() -> None:
    database = ControlledSettlementDatabase(await open_sqlite_database(":memory:"))
    database.control_next_settlement("delay")
    opening = asyncio.create_task(SqliteStorage.open(database))  # type: ignore[arg-type]
    await _yield(20)
    assert not opening.done()
    database.settle()
    storage = await opening

    database.control_next_settlement("delay")
    committing = asyncio.create_task(
        storage.commit(
            [
                {
                    "type": "entry",
                    "value": {"id": id_from_number(100), "conversationId": ROOT_CONVERSATION_ID, "kind": "settled"},
                }
            ],
            context,
        )
    )
    await _yield(20)
    assert not committing.done()
    assert await storage.mint_id() == 2
    database.settle()
    assert await committing == 1
    assert await storage.mint_id() == 101

    database.control_next_settlement("reject")
    rejected = asyncio.create_task(
        storage.commit(
            [
                {
                    "type": "entry",
                    "value": {"id": id_from_number(200), "conversationId": ROOT_CONVERSATION_ID, "kind": "rejected"},
                }
            ],
            context,
        )
    )
    await _yield()
    assert await storage.mint_id() == 102
    database.settle()
    with pytest.raises(RuntimeError, match="controlled settlement rejection"):
        await rejected
    assert await storage.mint_id() == 103
    assert await storage.entry(id_from_number(200), context) is None
    await storage.close(context)


async def test_reads_a_document_from_one_committed_state_while_a_commit_replaces_its_base() -> None:
    id = id_from_number(5)
    # Each yield count starts the commit at a different point of the read's record and revision queries.
    for yields in range(16):
        storage = await SqliteStorage.open(await open_sqlite_database(":memory:"))
        await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": {"id": id, "kind": "replaced", "scope": {"kind": "session"}},
                    "content": {"kind": "base", "version": 1, "value": {"value": 1}},
                }
            ],
            context,
        )
        read = asyncio.create_task(storage.document(id, "current", context))
        await _yield(yields)
        replace = asyncio.create_task(
            storage.commit(
                [
                    {
                        "type": "document.change",
                        "id": id,
                        "content": {"kind": "base", "version": 1, "value": {"value": 2}},
                    }
                ],
                context,
            )
        )
        stored, _ = await asyncio.gather(read, replace)
        assert stored is not None and stored["value"] in ({"value": 1}, {"value": 2})
        await storage.close(context)
