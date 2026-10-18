"""SQLite adapter on Python's standard ``sqlite3`` (replaces chord ``sqlite/node.ts``).

Every ``sqlite3`` call of a database runs on one dedicated worker thread that opens and owns its
connection, so the connection is never shared between threads and the event loop never blocks
on SQLite. Operations are handed to the worker through a FIFO queue at call time, which gives the
call-order guarantee upstream's ``SerialOperationQueue`` provides: an operation runs after
everything called before it, and calls made while a transaction is open wait until it commits or
rolls back.

A transaction occupies the worker from ``BEGIN IMMEDIATE`` to ``COMMIT``/``ROLLBACK``: while its
callback runs on the event loop, the worker executes only the statements of that transaction's
handle and leaves everything else queued. Results are handed back to the event loop in batches:
the worker keeps executing queued work and wakes the loop once when it runs out (or after a bounded
batch), so concurrent operations share thread hops.

Prepared statements are cached per connection by SQL text (``sqlite3``'s statement cache), shared by
the database and its transaction handles.

Values: ``None``, ``int``, ``float``, ``str`` and bytes-like values bind; ``bool`` and other types
are rejected as upstream's ``node:sqlite`` rejects them, and integers outside the signed 64-bit range
raise :class:`OverflowError` (upstream rejects such ``bigint`` bindings). Integer results are
exact Python ints, without ``node:sqlite``'s ``Number.MAX_SAFE_INTEGER`` limit. BLOB results are
``bytes``. Rows are ``dict`` objects keyed by column name.

Cancellation: upstream's SQLite API takes no ``Context``, and neither does this one. Cancelling
the task awaiting an operation is honored: a statement whose awaiting future was cancelled before
the worker reached it is skipped, and a cancelled transaction rolls back. A statement already
executing runs to completion. ``COMMIT``, ``ROLLBACK`` and ``close`` are never skipped.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Any, TypeVar

from .database import SqliteExecutor, SqliteRow, SqliteValue

__all__ = [
    "SqliteDatabaseOptions",
    "StdlibSqliteDatabase",
    "StdlibSqliteTransaction",
    "open_sqlite_database",
]

T = TypeVar("T")

_DEFAULT_WAL_AUTO_CHECKPOINT_PAGES = 1_000
_DEFAULT_BUSY_TIMEOUT_MS = 5_000
#: Distinct SQL texts whose prepared statements stay cached per connection.
_CACHED_STATEMENTS = 512
#: Completions the worker collects before it wakes the event loop while more work is queued.
_BATCH_LIMIT = 64

_STALE_HANDLE = "SQLite transaction handle is no longer active"
_ROLLBACK_FAILED = "SQLite transaction failed and rollback failed"
_CLOSED = "Cannot operate on a closed database."


@dataclass(frozen=True, slots=True)
class SqliteDatabaseOptions:
    """SQLite connection settings for a database file (upstream ``NodeSqliteDatabaseOptions``)."""

    #: SQLite WAL auto-checkpoint threshold. SQLite and this adapter default to 1,000 pages; 0 disables it.
    wal_auto_checkpoint_pages: int = _DEFAULT_WAL_AUTO_CHECKPOINT_PAGES
    #: Time SQLite waits for a competing file lock. SQLite defaults to 0; this adapter defaults to 5,000 ms.
    busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS


# Job kinds. A job is ``(kind, sql, params, future)``; a transaction job carries its channel in place of params.
_EXEC = 0
_RUN = 1
_GET = 2
_ALL = 3
_TRANSACTION = 4
_END = 5
_CLOSE = 6
_SHUTDOWN = 7

_Job = tuple[int, str, Any, "asyncio.Future[Any] | None"]
_Completion = tuple["asyncio.Future[Any]", bool, Any]

_SHUTDOWN_JOB: _Job = (_SHUTDOWN, "", (), None)
#: Sent by a transaction whose task was cancelled before its callback started.
_ABANDON_JOB: _Job = (_END, "ROLLBACK", (), None)

_EXACT_VALUE_TYPES = frozenset({int, float, str, bytes, type(None)})
_VALUE_TYPES = (int, float, str, bytes, bytearray, memoryview)


def _check_params(params: tuple[Any, ...]) -> None:
    for index, value in enumerate(params, 1):
        cls = value.__class__
        if cls in _EXACT_VALUE_TYPES:
            continue
        if cls is bool or not isinstance(value, _VALUE_TYPES):
            raise TypeError(f"Provided value cannot be bound to SQLite parameter {index}.")


def _exec_script(connection: sqlite3.Connection, sql: str) -> None:
    """Run SQL text that may hold several statements, without ``executescript``'s implicit COMMIT."""
    if ";" not in sql:
        connection.execute(sql).close()
        return
    start = 0
    search = 0
    while True:
        index = sql.find(";", search)
        if index == -1:
            break
        search = index + 1
        statement = sql[start:search]
        # A semicolon inside a literal, a comment or a trigger body does not end the statement.
        if sqlite3.complete_statement(statement):
            connection.execute(statement).close()
            start = search
    rest = sql[start:]
    if rest and not rest.isspace():
        connection.execute(rest).close()


def _execute(connection: sqlite3.Connection, kind: int, sql: str, params: tuple[Any, ...]) -> Any:
    if kind == _EXEC:
        _exec_script(connection, sql)
        return None
    if params:
        _check_params(params)
    cursor = connection.execute(sql, params)
    try:
        if kind == _RUN:
            return None
        if kind == _GET:
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(zip([column[0] for column in cursor.description], row, strict=True))
        rows = cursor.fetchall()
        if not rows:
            return []
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in rows]
    finally:
        # Resets the statement, so an unfinished read does not hold a read transaction open.
        cursor.close()


def _deliver(batch: list[_Completion]) -> None:
    for future, ok, value in batch:
        if future.done():
            continue
        if ok:
            future.set_result(value)
        else:
            future.set_exception(value)


class _Worker:
    """The thread that owns one connection. It references the queue, never the database object."""

    __slots__ = ("_completions", "_jobs", "_options", "_opened", "_path")

    def __init__(
        self,
        jobs: SimpleQueue[_Job],
        path: str,
        options: SqliteDatabaseOptions,
        opened: asyncio.Future[None],
    ) -> None:
        self._jobs = jobs
        self._path = path
        self._options = options
        self._opened = opened
        self._completions: list[_Completion] = []

    # --- completions ---

    def _complete(self, future: asyncio.Future[Any], ok: bool, value: Any) -> None:
        completions = self._completions
        completions.append((future, ok, value))
        if len(completions) >= _BATCH_LIMIT:
            self._flush()

    def _flush(self) -> None:
        batch = self._completions
        if not batch:
            return
        self._completions = []
        loop = batch[0][0].get_loop()
        if len(batch) > 1 and any(future.get_loop() is not loop for future, _, _ in batch):
            groups: dict[asyncio.AbstractEventLoop, list[_Completion]] = {}
            for completion in batch:
                groups.setdefault(completion[0].get_loop(), []).append(completion)
            for group_loop, group in groups.items():
                _post(group_loop, group)
        else:
            _post(loop, batch)

    def _next(self, jobs: SimpleQueue[_Job]) -> _Job:
        if jobs.empty():
            # About to sleep: hand back everything finished so far first.
            self._flush()
        return jobs.get()

    # --- lifecycle ---

    def run(self) -> None:
        try:
            connection = self._open()
        except Exception as error:
            _post(self._opened.get_loop(), [(self._opened, False, error)])
            return
        self._complete(self._opened, True, None)
        try:
            self._serve(connection)
        finally:
            self._flush()

    def _open(self) -> sqlite3.Connection:
        path = self._path
        options = self._options
        if path != ":memory:":
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        # Created on this thread, used only by it: `check_same_thread` stays on.
        connection = sqlite3.connect(
            path,
            timeout=options.busy_timeout_ms / 1000,
            isolation_level=None,
            cached_statements=_CACHED_STATEMENTS,
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {options.busy_timeout_ms}").close()
            connection.execute("PRAGMA journal_mode = WAL").close()
            connection.execute("PRAGMA synchronous = NORMAL").close()
            connection.execute(f"PRAGMA wal_autocheckpoint = {options.wal_auto_checkpoint_pages}").close()
        except BaseException:
            # Preserve the configuration failure.
            try:
                connection.close()
            except Exception:
                pass
            raise
        return connection

    def _serve(self, connection: sqlite3.Connection) -> None:
        jobs = self._jobs
        complete = self._complete
        while True:
            kind, sql, params, future = self._next(jobs)
            if kind <= _ALL:
                assert future is not None
                if future.cancelled():
                    continue
                try:
                    value = _execute(connection, kind, sql, params)
                except Exception as error:
                    complete(future, False, error)
                else:
                    complete(future, True, value)
            elif kind == _TRANSACTION:
                assert future is not None
                self._transaction(connection, params, future)
            elif kind == _CLOSE:
                assert future is not None
                failure: Exception | None = None
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").close()
                except Exception as error:
                    failure = error
                try:
                    connection.close()
                except Exception as error:
                    failure = failure or error
                complete(future, failure is None, failure)
                return
            else:  # _SHUTDOWN: the database was garbage collected without close().
                try:
                    connection.close()
                except Exception:
                    pass
                return

    def _transaction(
        self,
        connection: sqlite3.Connection,
        channel: SimpleQueue[_Job],
        started: asyncio.Future[Any],
    ) -> None:
        complete = self._complete
        if started.cancelled():
            return
        try:
            connection.execute("BEGIN IMMEDIATE").close()
        except Exception as error:
            complete(started, False, error)
            return
        complete(started, True, None)
        while True:
            kind, sql, params, future = self._next(channel)
            if kind == _END:
                self._end(connection, sql, future)
                return
            assert future is not None
            if future.cancelled():
                continue
            try:
                value = _execute(connection, kind, sql, params)
            except Exception as error:
                complete(future, False, error)
            else:
                complete(future, True, value)

    def _end(self, connection: sqlite3.Connection, sql: str, future: asyncio.Future[Any] | None) -> None:
        try:
            connection.execute(sql).close()
        except Exception as error:
            failure: BaseException = error
            if sql == "COMMIT":
                # The transaction may still be open: roll it back, as upstream does after a failed COMMIT.
                try:
                    connection.execute("ROLLBACK").close()
                except Exception as rollback_error:
                    failure = ExceptionGroup(_ROLLBACK_FAILED, [error, rollback_error])
            if future is not None:
                self._complete(future, False, failure)
        else:
            if future is not None:
                self._complete(future, True, None)


def _post(loop: asyncio.AbstractEventLoop, batch: list[_Completion]) -> None:
    try:
        loop.call_soon_threadsafe(_deliver, batch)
    except RuntimeError:
        # The loop is closed: nobody is left to receive the results.
        pass


class _Executor:
    """``exec``/``run``/``get``/``all`` as futures created at call time, so call order is queue order."""

    __slots__ = ()

    def _submit(self, kind: int, sql: str, params: tuple[Any, ...]) -> asyncio.Future[Any]:
        raise NotImplementedError

    def exec(self, sql: str) -> asyncio.Future[None]:
        """Run SQL text without bindings; it may contain several statements."""
        return self._submit(_EXEC, sql, ())

    def run(self, sql: str, *params: SqliteValue) -> asyncio.Future[None]:
        """Execute one statement with positional bindings, discarding any result rows."""
        return self._submit(_RUN, sql, params)

    def get(self, sql: str, *params: SqliteValue) -> asyncio.Future[SqliteRow | None]:
        """Execute one statement and return its first row, or ``None``."""
        return self._submit(_GET, sql, params)

    def all(self, sql: str, *params: SqliteValue) -> asyncio.Future[list[SqliteRow]]:
        """Execute one statement and return all its rows."""
        return self._submit(_ALL, sql, params)


class StdlibSqliteTransaction(_Executor):
    """The handle a transaction callback receives; it expires when the callback settles."""

    __slots__ = ("_active", "_channel")

    def __init__(self, channel: SimpleQueue[_Job]) -> None:
        self._channel = channel
        self._active = True

    def _submit(self, kind: int, sql: str, params: tuple[Any, ...]) -> asyncio.Future[Any]:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        if self._active:
            self._channel.put((kind, sql, params, future))
        else:
            future.set_exception(RuntimeError(_STALE_HANDLE))
        return future


class StdlibSqliteDatabase(_Executor):
    """:class:`~pi_chord.sqlite.SqliteDatabase` adapter backed by Python's ``sqlite3``.

    Create one with :func:`open_sqlite_database`. Operations must be called from a running event
    loop; each returns a future that is already queued, so un-awaited calls still run in call order.
    """

    __slots__ = ("__weakref__", "_closing", "_finalizer", "_jobs")

    def __init__(self, jobs: SimpleQueue[_Job]) -> None:
        """Wrap the worker that serves ``jobs``. Internal: use :func:`open_sqlite_database`."""
        self._jobs = jobs
        self._closing: asyncio.Future[None] | None = None
        # A database dropped without close() still stops its worker and closes the connection.
        self._finalizer = weakref.finalize(self, jobs.put, _SHUTDOWN_JOB)

    def _submit(self, kind: int, sql: str, params: tuple[Any, ...]) -> asyncio.Future[Any]:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        if self._closing is None:
            self._jobs.put((kind, sql, params, future))
        else:
            future.set_exception(sqlite3.ProgrammingError(_CLOSED))
        return future

    def transaction(self, callback: Callable[[SqliteExecutor], Awaitable[T]]) -> asyncio.Future[T]:
        """Run ``callback`` in a ``BEGIN IMMEDIATE`` transaction and commit, or roll back if it raises.

        The transaction is queued at call time, behind everything called before it. The callback
        must use the handle it receives; the handle expires when the callback settles. Calling this
        database (including :meth:`transaction` or :meth:`close`) from inside the callback waits for
        the transaction and so never settles. If rollback fails after the callback raised, the
        result raises a :class:`BaseExceptionGroup` of both errors instead of the callback error.
        """
        loop = asyncio.get_running_loop()
        if self._closing is not None:
            failed: asyncio.Future[T] = loop.create_future()
            failed.set_exception(sqlite3.ProgrammingError(_CLOSED))
            return failed
        started: asyncio.Future[None] = loop.create_future()
        channel: SimpleQueue[_Job] = SimpleQueue()
        self._jobs.put((_TRANSACTION, "", channel, started))
        return loop.create_task(_transaction(started, channel, callback))

    def close(self) -> asyncio.Future[None]:
        """Checkpoint the WAL and close the connection after everything queued before; idempotent.

        A second call settles when the first close finishes, without its error.
        """
        loop = asyncio.get_running_loop()
        first = self._closing is None
        if self._closing is None:
            self._closing = loop.create_future()
            self._jobs.put((_CLOSE, "", (), self._closing))
            self._finalizer.detach()
        result: asyncio.Future[None] = loop.create_future()

        def relay(closing: asyncio.Future[None]) -> None:
            error = closing.exception()
            if result.done():
                return
            if first and error is not None:
                result.set_exception(error)
            else:
                result.set_result(None)

        self._closing.add_done_callback(relay)
        return result


async def _transaction(
    started: asyncio.Future[None],
    channel: SimpleQueue[_Job],
    callback: Callable[[SqliteExecutor], Awaitable[T]],
) -> T:
    try:
        await started
    except asyncio.CancelledError:
        # The worker may have begun already; it rolls back when it reads this (or never begins).
        channel.put(_ABANDON_JOB)
        raise
    handle = StdlibSqliteTransaction(channel)
    try:
        result = await callback(handle)
        handle._active = False
    except BaseException as error:
        handle._active = False
        try:
            await _end(channel, "ROLLBACK")
        except Exception as rollback_error:
            raise BaseExceptionGroup(_ROLLBACK_FAILED, [error, rollback_error]) from None
        raise
    # A failed COMMIT is rolled back by the worker in the same hop; its error (or a group with the
    # rollback error) is raised here.
    await _end(channel, "COMMIT")
    return result


def _end(channel: SimpleQueue[_Job], sql: str) -> asyncio.Future[None]:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    channel.put((_END, sql, (), future))
    return future


async def open_sqlite_database(
    path: str | os.PathLike[str],
    options: SqliteDatabaseOptions | None = None,
) -> StdlibSqliteDatabase:
    """Open and configure a ``sqlite3``-backed database facade (upstream ``openNodeSqliteDatabase``).

    Creates the parent directory of a file database, then sets the busy timeout,
    ``journal_mode = WAL``, ``synchronous = NORMAL`` and ``wal_autocheckpoint``. ``":memory:"``
    opens a private in-memory database. The connection lives on a new daemon worker thread.
    """
    if options is None:
        options = SqliteDatabaseOptions()
    for name in ("wal_auto_checkpoint_pages", "busy_timeout_ms"):
        value = getattr(options, name)
        if value.__class__ is not int or value < 0:
            raise ValueError(f"SqliteDatabaseOptions.{name} must be a non-negative int, got {value!r}")
    loop = asyncio.get_running_loop()
    opened: asyncio.Future[None] = loop.create_future()
    jobs: SimpleQueue[_Job] = SimpleQueue()
    worker = _Worker(jobs, os.fspath(path), options, opened)
    threading.Thread(target=worker.run, name="pi-chord-sqlite", daemon=True).start()
    try:
        await opened
    except asyncio.CancelledError:
        jobs.put(_SHUTDOWN_JOB)
        raise
    return StdlibSqliteDatabase(jobs)
