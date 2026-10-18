"""Explicit invocation context: cancellation and scoped values passed through operations.

Port of ``@earendil-works/chord/context``. A :class:`Context` is immutable; deriving one
(:func:`with_context_value`, :func:`with_abort_signal`, ...) never changes its parent.
Cancellation uses :class:`pi_ai.AbortSignal`, so a context's signal can be handed to
``pi_ai`` requests directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, Generic, NamedTuple, TypeVar

from pi_ai.abort import AbortController, AbortError, AbortSignal

__all__ = [
    "Context",
    "ContextKey",
    "BACKGROUND_CONTEXT",
    "TODO_CONTEXT",
    "create_context_key",
    "with_context_value",
    "with_abort_signal",
    "without_abort_signal",
    "with_cancel",
    "Cancellable",
    "await_with_context",
]

T = TypeVar("T")


class ContextKey(Generic[T]):
    """Typed identity of one value carried by a :class:`Context`. Keys compare by identity."""

    __slots__ = ("description",)

    def __init__(self, description: str) -> None:
        self.description = description

    def __repr__(self) -> str:
        return f"ContextKey({self.description!r})"


_ABORT_SIGNAL_KEY: ContextKey[AbortSignal | None] = ContextKey("chord.abortSignal")


class Context:
    """Immutable invocation-scoped values. Use the module functions to derive contexts."""

    __slots__ = ()

    def value(self, key: ContextKey[T]) -> T | None:
        raise NotImplementedError

    @property
    def abort_signal(self) -> AbortSignal | None:
        return self.value(_ABORT_SIGNAL_KEY)

    def throw_if_aborted(self) -> None:
        """Raise the cancellation of this context, if any (``abortSignal?.throwIfAborted()``)."""
        signal = self.abort_signal
        if signal is not None:
            signal.throw_if_aborted()


class _EmptyContext(Context):
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def value(self, key: ContextKey[T]) -> T | None:
        return None

    def __str__(self) -> str:
        return self._name

    __repr__ = __str__


class _ValueContext(Context):
    __slots__ = ("_key", "_parent", "_value")

    def __init__(self, parent: Context, key: ContextKey[Any], value: Any) -> None:
        self._parent = parent
        self._key = key
        self._value = value

    def value(self, key: ContextKey[T]) -> T | None:
        context: Context = self
        # Walk the chain iteratively: derived contexts can be deep.
        while isinstance(context, _ValueContext):
            if context._key is key:
                return context._value
            context = context._parent
        return context.value(key)

    def __str__(self) -> str:
        return f"{self._parent}.WithValue({self._key.description})"

    __repr__ = __str__


#: A context that is never cancelled and carries no values.
BACKGROUND_CONTEXT: Context = _EmptyContext("[Context BACKGROUND_CONTEXT]")
#: Placeholder root for code that does not have a context yet.
TODO_CONTEXT: Context = _EmptyContext("[Context TODO_CONTEXT]")


def create_context_key(description: str) -> ContextKey[Any]:
    return ContextKey(description)


def with_context_value(key: ContextKey[T], value: T, parent: Context) -> Context:
    """Derive a context containing one additional or replaced value."""
    return _ValueContext(parent, key, value)


def with_abort_signal(signal: AbortSignal, context: Context) -> Context:
    """Derive a context cancelled by either the parent's signal or ``signal``."""
    parent = context.abort_signal
    combined = signal if parent is None else AbortSignal.any([parent, signal])
    return _ValueContext(context, _ABORT_SIGNAL_KEY, combined)


def without_abort_signal(context: Context) -> Context:
    """Derive a context keeping all values except cancellation. Intended for mandatory cleanup."""
    return _ValueContext(context, _ABORT_SIGNAL_KEY, None)


class Cancellable(NamedTuple):
    context: Context
    cancel: Any  # (reason: Any = None) -> None


def with_cancel(context: Context) -> Cancellable:
    """Derive an independently cancellable child context."""
    controller = AbortController()
    return Cancellable(with_abort_signal(controller.signal, context), controller.abort)


async def await_with_context(awaitable: Awaitable[T], context: Context) -> T:
    """Wait for ``awaitable`` until it finishes or ``context`` is cancelled.

    Cancellation stops only this wait: a coroutine or future keeps running in its own
    task. The cancellation raises the signal's reason when it is an exception,
    otherwise :class:`pi_ai.AbortError`.
    """
    signal = context.abort_signal
    if signal is None:
        return await awaitable
    if signal.aborted:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise _abort_error(signal)
    future = asyncio.ensure_future(awaitable)
    if future.done():
        return future.result()
    loop = asyncio.get_running_loop()
    aborted: asyncio.Future[None] = loop.create_future()

    def on_abort() -> None:
        loop.call_soon_threadsafe(_resolve, aborted)

    remove = signal.add_listener(on_abort)
    try:
        await asyncio.wait({future, aborted}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        remove()
        if not aborted.done():
            aborted.cancel()
    if future.done():
        return future.result()
    future.add_done_callback(_consume)
    raise _abort_error(signal)


def _abort_error(signal: AbortSignal) -> BaseException:
    reason = signal.reason
    if isinstance(reason, BaseException):
        return reason
    return AbortError(reason)


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()
