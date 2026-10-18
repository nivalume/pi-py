"""Future helpers for operations that start eagerly, like the JavaScript functions they port.

A JavaScript ``async`` function runs synchronously until its first ``await``, and a Promise-returning
method does its admission work before it returns. A Python coroutine does nothing until awaited, so the
Session and its transactions return ``asyncio`` futures from plain methods instead: the synchronous part
(admission, staging, cache lookups) runs at the call and the rest runs as a task.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

__all__ = ["consume", "rejected", "resolved", "start"]

T = TypeVar("T")


def resolved(value: T) -> asyncio.Future[T]:
    """A future already resolved with ``value`` (``Promise.resolve(value)``)."""
    future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


def rejected(error: BaseException) -> asyncio.Future[Any]:
    """A future already failed with ``error`` (``Promise.reject(error)``)."""
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    future.set_exception(error)
    return future


def start(awaitable: Awaitable[T]) -> asyncio.Future[T]:
    """Run ``awaitable`` as a task (or return it when it already is a future)."""
    return asyncio.ensure_future(awaitable)


def consume(future: asyncio.Future[Any]) -> None:
    """Done callback marking a future's exception as retrieved (upstream ``.catch(() => {})``)."""
    if not future.cancelled():
        future.exception()
