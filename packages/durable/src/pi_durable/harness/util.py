"""Small helpers shared by the Harness components (port of ``harness/util.ts``)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pi_ai.abort import AbortError
from pi_chord.context import Context

from ..errors import SessionFailed
from ..types import Cursor, Page

__all__ = [
    "Waiters",
    "chain",
    "closed_error",
    "eager",
    "js_same",
    "json_equal",
    "optional_extension",
    "scan_all",
    "settle",
]

K = TypeVar("K")
T = TypeVar("T")


class Waiters(Generic[K, T]):
    """Pending waits by key. Each settles once: through ``resolve``, ``reject_all``, or cancellation of its context."""

    __slots__ = ("_sets",)

    def __init__(self) -> None:
        self._sets: dict[K, dict[asyncio.Future[T], None]] = {}

    def add(self, key: K, context: Context) -> asyncio.Future[T]:
        """A future for ``key``; it fails with the signal's reason when ``context`` is cancelled first."""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[T] = loop.create_future()
        signal = context.abort_signal
        if signal is not None and signal.aborted:
            waiter.set_exception(_reason(signal.reason))
            return waiter
        waiters = self._sets.get(key)
        if waiters is None:
            waiters = {}
            self._sets[key] = waiters
        waiters[waiter] = None
        if signal is not None:
            current = waiters

            def on_abort() -> None:
                def run() -> None:
                    current.pop(waiter, None)
                    if not current and self._sets.get(key) is current:
                        del self._sets[key]
                    if not waiter.done():
                        waiter.set_exception(_reason(signal.reason))

                loop.call_soon_threadsafe(run)

            remove = signal.add_listener(on_abort)
            waiter.add_done_callback(lambda _future: remove())
        return waiter

    def pending(self) -> list[K]:
        """The keys with a pending wait, as a list (resolving while iterating is safe)."""
        return list(self._sets)

    def resolve(self, key: K, value: T) -> None:
        waiters = self._sets.pop(key, None)
        for waiter in waiters or ():
            if not waiter.done():
                waiter.set_result(value)

    def reject_all(self, error: BaseException) -> None:
        sets = list(self._sets.values())
        self._sets.clear()
        for waiters in sets:
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_exception(error)


def _reason(reason: Any) -> BaseException:
    return reason if isinstance(reason, BaseException) else AbortError(reason)


async def scan_all(scan: Callable[[Cursor | None], Awaitable[Page[T]]]) -> list[T]:
    """Every item of a paginated scan, in page order."""
    items: list[T] = []
    cursor: Cursor | None = None
    while True:
        page = await scan(cursor)
        items.extend(page["items"])
        cursor = page.get("next")
        if cursor is None:
            return items


def closed_error(session: Any) -> BaseException:
    """What a closed Harness raises: ``SessionFailed`` with the cause when it closed because the Session failed."""
    failure = session.failure
    return RuntimeError("Harness is closed") if failure is None else SessionFailed(failure.error)


def optional_extension(extension: str | None) -> dict[str, str]:
    """``{"extension": name}`` when the extension is known."""
    return {} if extension is None else {"extension": extension}


def eager(awaitable: Awaitable[T], then: Callable[[T], Any] | None = None) -> asyncio.Future[Any]:
    """Keep an already started operation as a future, optionally mapped by ``then``.

    The operation's synchronous part (admission to the Session line) ran when it was called; mapping
    its result happens once it settles, as ``promise.then(then)`` would.
    """
    if then is None:
        return asyncio.ensure_future(awaitable)

    async def mapped() -> Any:
        return then(await awaitable)

    return asyncio.ensure_future(mapped())


def chain(
    future: asyncio.Future[Any], then: Callable[[Any], Any], context: Context | None = None
) -> asyncio.Future[Any]:
    """``future.then(then)`` settled through done callbacks, without a task, so waiters resume promptly.

    ``then`` gets the result and returns the mapped value, or a future to follow. With ``context``, a
    cancellation of its signal rejects the result (only the wait; the operation keeps running).
    """
    result: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    _follow(future, then, result)
    signal = None if context is None else context.abort_signal
    if signal is not None:
        loop = asyncio.get_running_loop()

        def on_abort() -> None:
            def reject() -> None:
                if not result.done():
                    result.set_exception(_reason(signal.reason))

            loop.call_soon_threadsafe(reject)

        remove = signal.add_listener(on_abort)
        result.add_done_callback(lambda _future: remove())
    return result


def _follow(future: asyncio.Future[Any], then: Callable[[Any], Any] | None, result: asyncio.Future[Any]) -> None:
    def settled(done: asyncio.Future[Any]) -> None:
        if result.done():
            if not done.cancelled():
                done.exception()
            return
        if done.cancelled():
            result.cancel()
            return
        error = done.exception()
        if error is not None:
            result.set_exception(error)
            return
        if then is None:
            result.set_result(done.result())
            return
        try:
            mapped = then(done.result())
        except BaseException as raised:
            result.set_exception(raised)
            return
        if isinstance(mapped, asyncio.Future):
            _follow(mapped, None, result)
        else:
            result.set_result(mapped)

    future.add_done_callback(settled)


def settle(future: asyncio.Future[Any]) -> None:
    """Done callback that marks a future's failure as observed (upstream ``.catch(() => {})``)."""
    if not future.cancelled():
        future.exception()


def js_same(left: Any, right: Any) -> bool:
    """JavaScript ``left === right`` for JSON values: identity for containers, value for primitives."""
    if left is right:
        return True
    if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return bool(left == right)


def json_equal(left: Any, right: Any) -> bool:
    """Structural equality of two JSON values; object key order is ignored, ``True`` never equals ``1``."""
    if left is right:
        return True
    if isinstance(left, dict) and isinstance(right, dict):
        if len(left) != len(right):
            return False
        return all(key in right and json_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right
