"""The portable asynchronous SQLite facade (port of chord ``sqlite/database.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias, TypeVar

__all__ = ["SqliteDatabase", "SqliteExecutor", "SqliteRow", "SqliteValue"]

T = TypeVar("T")

#: Values supported by the portable SQLite drivers. ``int`` covers both of JavaScript's ``number``
#: integers and ``bigint``; ``bool`` is not a SQLite value (bind ``0``/``1``).
SqliteValue: TypeAlias = int | float | str | bytes | None

#: One result row: column name to value. Values are :data:`SqliteValue`; the mapping is typed
#: loosely so callers can read columns into their own row types, as upstream's ``get<T>`` does.
SqliteRow: TypeAlias = dict[str, Any]


class SqliteExecutor(Protocol):
    """Asynchronous SQL operations shared by a database and its transaction handles.

    ``exec`` runs SQL text without bindings and may contain several statements. ``run``, ``get``
    and ``all`` execute one statement with positional bindings. Adapters may cache prepared
    statements by SQL text, so callers pass values as bindings instead of interpolating them.

    Methods return awaitables: an adapter may implement them as ``async def`` or, like
    :class:`pi_chord.sqlite.stdlib.StdlibSqliteDatabase`, as plain methods that start the
    operation at call time and return a future.
    """

    def exec(self, sql: str) -> Awaitable[None]: ...

    def run(self, sql: str, *params: SqliteValue) -> Awaitable[None]: ...

    def get(self, sql: str, *params: SqliteValue) -> Awaitable[SqliteRow | None]: ...

    def all(self, sql: str, *params: SqliteValue) -> Awaitable[list[SqliteRow]]: ...


class SqliteDatabase(SqliteExecutor, Protocol):
    """Minimal portable SQLite database facade.

    All operations are asynchronous so adapters may execute outside the caller's thread.

    ``transaction`` passes the callback a transaction handle. All work in the transaction must use
    that handle; the handle is invalid after the callback settles. Adapters must queue unrelated
    operations and other transactions until the transaction finishes. The returned awaitable
    settles after commit or rollback. Calling the database itself (including ``transaction`` or
    ``close``) from inside a callback therefore waits for that transaction and never settles.

    When the callback raises, the adapter must roll the transaction back before raising that same
    error. If rollback fails, it must raise a different error (a :class:`BaseExceptionGroup`
    holding both) so callers cannot mistake the callback error for a guaranteed rollback.
    """

    def transaction(self, callback: Callable[[SqliteExecutor], Awaitable[T]]) -> Awaitable[T]: ...

    def close(self) -> Awaitable[None]: ...
