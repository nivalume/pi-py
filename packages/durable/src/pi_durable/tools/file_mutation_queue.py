"""Per-file serialization of ``edit`` and ``write`` (port of ``tools/file-mutation-queue.ts``)."""

from __future__ import annotations

import asyncio
import re
import weakref
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from ..env.types import get_or_throw

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..env.types import ExecutionEnv

__all__ = ["with_file_mutation_queue"]

T = TypeVar("T")

#: Tail of the mutation chain of each file, keyed by file system id and canonical path, per event loop.
_queues: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Future[None]]] = (
    weakref.WeakKeyDictionary()
)

_TRAILING_SEPARATOR = re.compile(r"[/\\]$")


async def _mutation_key(env: ExecutionEnv, path: str, context: Context) -> str:
    absolute_path = get_or_throw(await env.absolute_path(path, context))
    return f"{env.id}\0{await _canonical(env, absolute_path, context)}"


async def _canonical(env: ExecutionEnv, absolute_path: str, context: Context) -> str:
    """The canonical path; for a file that does not exist yet, its canonical parent joined with its name.

    So a ``write`` that creates a file and a later mutation of it share one key even under a symlinked
    directory.
    """
    result = await env.canonical_path(absolute_path, context)
    if result["ok"] is True:
        return result["value"]
    error = result["error"]
    if error.code == "not_supported":
        return absolute_path
    if error.code != "not_found":
        raise error
    # The file system splits the path, so a name may contain characters that are separators elsewhere.
    parent = get_or_throw(await env.join_path([absolute_path, ".."], context))
    if parent == absolute_path or not absolute_path.startswith(parent):
        return absolute_path
    name = absolute_path[len(parent) + (0 if _TRAILING_SEPARATOR.search(parent) else 1) :]
    return get_or_throw(await env.join_path([await _canonical(env, parent, context), name], context))


def _settle(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


async def with_file_mutation_queue(env: ExecutionEnv, path: str, fn: Callable[[], Awaitable[T]], context: Context) -> T:
    """Serialize ``edit`` and ``write`` mutations of one file within this process.

    One file is the same file system id and canonical path, whichever environment object the call got.
    Other files, and other file systems, never wait. Concurrent calls on one file run in the order their
    keys resolve. Not a lock against ``bash`` or other processes. A call waiting for its turn is not
    interrupted by its context; cancelling its task still lets later calls run in order.
    """
    key = await _mutation_key(env, path, context)
    loop = asyncio.get_running_loop()
    queues = _queues.setdefault(loop, {})
    # Take the slot without awaiting, so no other call can take it in between.
    previous = queues.get(key)
    done: asyncio.Future[None] = loop.create_future()
    tail: asyncio.Future[None] = loop.create_future()
    # `tail` is `previous.then(() => done)`: it settles once both have.
    if previous is None:
        done.add_done_callback(lambda _: _settle(tail))
    else:
        previous.add_done_callback(lambda _: done.add_done_callback(lambda _: _settle(tail)))
    queues[key] = tail
    try:
        if previous is not None:
            await asyncio.shield(previous)
        return await fn()
    finally:
        _settle(done)
        if queues.get(key) is tail:
            del queues[key]
