"""Cancellation primitives modeled on the web ``AbortController``/``AbortSignal`` pair.

pi passes an ``AbortSignal`` through every layer (``options.signal``). The Python
port keeps that contract so the same cancellation semantics apply: aborting a
request ends its stream with ``stop_reason == "aborted"`` instead of raising.

Inside async code, :func:`abort_scope` turns an abort into prompt task
cancellation (any pending ``await`` such as a network read is interrupted) and
re-raises it as :class:`AbortError`, the same way :func:`asyncio.timeout` turns a
deadline into :class:`TimeoutError`.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterable
from typing import Any

__all__ = [
    "AbortError",
    "AbortController",
    "AbortSignal",
    "abort_scope",
    "operation_signal",
    "race_with_abort_signal",
    "sleep",
]


class AbortError(Exception):
    """Raised when an operation is cancelled through an :class:`AbortSignal`."""

    def __init__(self, reason: Any = None) -> None:
        if isinstance(reason, BaseException):
            message = str(reason) or "The operation was aborted"
        elif reason is None:
            message = "The operation was aborted"
        else:
            message = str(reason)
        super().__init__(message)
        self.reason = reason


_Listener = Callable[[], Any]


class AbortSignal:
    """Read side of an :class:`AbortController`.

    Listeners run synchronously when the signal aborts. ``abort()`` may be called
    from any thread; listeners that touch an event loop should hand work to it with
    ``loop.call_soon_threadsafe`` (the helpers in this module already do).
    """

    __slots__ = ("__weakref__", "_aborted", "_listeners", "_lock", "_reason")

    def __init__(self) -> None:
        self._aborted = False
        self._reason: Any = None
        self._listeners: list[_Listener] | None = None
        self._lock = threading.Lock()

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> Any:
        return self._reason

    def throw_if_aborted(self) -> None:
        if self._aborted:
            raise self._reason if isinstance(self._reason, AbortError) else AbortError(self._reason)

    def add_listener(self, listener: _Listener) -> Callable[[], None]:
        """Register ``listener`` to run on abort. Returns a function that removes it.

        If the signal is already aborted the listener runs immediately.
        """
        with self._lock:
            if not self._aborted:
                if self._listeners is None:
                    self._listeners = []
                self._listeners.append(listener)
                return lambda: self._remove(listener)
        listener()
        return _noop

    def _remove(self, listener: _Listener) -> None:
        with self._lock:
            if self._listeners:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass

    def _abort(self, reason: Any) -> None:
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
            self._reason = AbortError(reason) if reason is None or not isinstance(reason, BaseException) else reason
            listeners, self._listeners = self._listeners, None
        for listener in listeners or ():
            try:
                listener()
            except Exception:  # listeners must not break abort propagation
                pass

    async def wait(self) -> Any:
        """Wait until the signal aborts and return its reason."""
        if self._aborted:
            return self._reason
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        remove = self.add_listener(lambda: _resolve_in_loop(loop, future))
        try:
            await future
        finally:
            remove()
        return self._reason

    @staticmethod
    def abort(reason: Any = None) -> AbortSignal:
        """An already aborted signal."""
        controller = AbortController()
        controller.abort(reason)
        return controller.signal

    @staticmethod
    def timeout(seconds: float) -> AbortSignal:
        """A signal that aborts after ``seconds`` (requires a running event loop)."""
        controller = AbortController()
        loop = asyncio.get_running_loop()
        loop.call_later(seconds, controller.abort, TimeoutError("The operation timed out"))
        return controller.signal

    @staticmethod
    def any(signals: Iterable[AbortSignal | None]) -> AbortSignal:
        """A signal that aborts when any of ``signals`` aborts."""
        active = [signal for signal in signals if signal is not None]
        controller = AbortController()
        for signal in active:
            if signal.aborted:
                controller.abort(signal.reason)
                return controller.signal
        for signal in active:
            signal.add_listener(lambda signal=signal: controller.abort(signal.reason))
        return controller.signal

    def __repr__(self) -> str:
        return f"AbortSignal(aborted={self._aborted})"


def _noop() -> None:
    return None


def _resolve_future(future: asyncio.Future[Any]) -> None:
    if not future.done():
        future.set_result(None)


class AbortController:
    """Write side: ``controller.abort()`` aborts ``controller.signal``."""

    __slots__ = ("signal",)

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: Any = None) -> None:
        self.signal._abort(reason)


def operation_signal(signal: AbortSignal | None) -> AbortSignal:
    """Return ``signal`` or a fresh never-aborting signal for APIs whose signal is optional."""
    return signal if signal is not None else AbortSignal()


class abort_scope:
    """Cancel the current task when ``signal`` aborts, raising :class:`AbortError`.

    ::

        async with abort_scope(options.get("signal")):
            response = await client.send(request, stream=True)
            async for line in response.aiter_lines():
                ...

    A ``None`` signal makes the scope a no-op. Cancellation that does not come from
    the signal propagates unchanged.
    """

    __slots__ = ("_cancelling", "_exited", "_fired", "_loop", "_remove", "_signal", "_task")

    def __init__(self, signal: AbortSignal | None) -> None:
        self._signal = signal
        self._task: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._remove: Callable[[], None] | None = None
        self._fired = False
        self._exited = False
        self._cancelling = 0

    async def __aenter__(self) -> abort_scope:
        signal = self._signal
        if signal is None:
            return self
        signal.throw_if_aborted()
        self._task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        if self._task is not None:
            self._cancelling = self._task.cancelling()
            self._remove = signal.add_listener(self._on_abort)
        return self

    def _on_abort(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._cancel()
        else:
            loop.call_soon_threadsafe(self._cancel)

    def _cancel(self) -> None:
        task = self._task
        if task is not None and not task.done() and not self._fired and not self._exited:
            self._fired = True
            task.cancel()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        self._exited = True
        if self._remove is not None:
            self._remove()
            self._remove = None
        if self._fired and self._task is not None:
            if self._task.uncancel() <= self._cancelling and exc_type is asyncio.CancelledError:
                assert self._signal is not None
                raise AbortError(self._signal.reason) from exc
        return None


async def race_with_abort_signal(awaitable: Any, signal: AbortSignal | None) -> Any:
    """Await ``awaitable`` but stop waiting (raising :class:`AbortError`) when ``signal`` aborts.

    Unlike :class:`abort_scope`, the abandoned operation keeps running in its own task
    and its eventual exception is consumed, matching pi's ``raceWithAbortSignal``.
    """
    if signal is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    if signal.aborted:
        task.add_done_callback(_consume_task_result)
        signal.throw_if_aborted()
    loop = asyncio.get_running_loop()
    abort_future: asyncio.Future[None] = loop.create_future()
    remove = signal.add_listener(lambda: _resolve_in_loop(loop, abort_future))
    try:
        await asyncio.wait({task, abort_future}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        remove()
        if not abort_future.done():
            abort_future.cancel()
    # An abort that fired while the operation ran wins, even if the operation also finished
    # in the same loop iteration (JavaScript abort listeners run synchronously).
    if signal.aborted:
        task.add_done_callback(_consume_task_result)
        signal.throw_if_aborted()
    return task.result()


def _resolve_in_loop(loop: asyncio.AbstractEventLoop, future: asyncio.Future[Any]) -> None:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        _resolve_future(future)
    elif not loop.is_closed():
        loop.call_soon_threadsafe(_resolve_future, future)


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


async def sleep(seconds: float, signal: AbortSignal | None = None) -> None:
    """``asyncio.sleep`` that raises :class:`AbortError` as soon as ``signal`` aborts."""
    if signal is None:
        await asyncio.sleep(seconds)
        return
    async with abort_scope(signal):
        await asyncio.sleep(max(0.0, seconds))
