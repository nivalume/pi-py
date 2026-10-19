"""Committed-state observers: the Chord state source and the exact-frame watch (port of ``session/observation.ts``).

Both are bound one-to-one to one attached state, a document incarnation or a conversation view, and
are advanced by the Session's commit publication. A ``None`` value retires them. Deliveries are
scheduled on the running event loop (``loop.call_soon``, upstream ``queueMicrotask``), never inline.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Final, Generic, TypeAlias, TypeVar

from pi_ai.abort import AbortSignal
from pi_chord._callbacks import trim_arguments
from pi_chord.context import Context, without_abort_signal
from pi_chord.delta import Op
from pi_chord.json import JsonObject
from pi_chord.state import ReplicatedStateSnapshot, ReplicatedStateSourceFrame

from ..errors import error_message
from ..types import WatchEnd

__all__ = [
    "MAX_PENDING_WATCH_FRAMES",
    "RETIREMENT_OPERATIONS",
    "CommittedStateSource",
    "CommittedWatch",
    "ObservedDocumentValue",
]

T = TypeVar("T")

#: What a document observer holds: the committed value, or ``None`` once the incarnation retired.
ObservedDocumentValue: TypeAlias = "JsonObject | None"

#: Maximum exact committed frames retained behind one unavailable watch listener.
MAX_PENDING_WATCH_FRAMES: Final = 100

#: Canonical terminal update for a retired document incarnation (immutable by contract).
RETIREMENT_OPERATIONS: Final[list[Op]] = [("r", None)]


class CommittedStateSource(Generic[T]):
    """Session-to-Chord bridge owned one-to-one by one attached state; a ``None`` value retires it.

    Implements :class:`pi_chord.state.ReplicatedStateSource`. Internal.
    """

    __slots__ = ("_attachments", "_closed", "_cursor", "_release", "_retired", "_value")

    def __init__(self, value: T, release: Callable[[], None]) -> None:
        self._attachments: dict[_SessionSourceAttachment[T], None] = {}
        self._release: Callable[[], None] | None = release
        #: Released on disposal.
        self._value: T | None = value
        self._cursor = 0
        self._retired = False
        self._closed = False

    def attach(self) -> _SessionSourceAttachment[T]:
        if self._closed:
            raise RuntimeError("State source is closed")
        attachment: _SessionSourceAttachment[T]

        def release() -> None:
            self._attachments.pop(attachment, None)
            if not self._attachments:
                self._finish_disposal()

        attachment = _SessionSourceAttachment(ReplicatedStateSnapshot(self._value, self._cursor), release)  # type: ignore[arg-type]
        self._attachments[attachment] = None
        return attachment

    def advance(self, value: T, ops: Sequence[Op], context: Context) -> None:
        if self._closed or self._retired:
            return
        self._value = value
        self._cursor += 1
        if value is None:
            self._retired = True
        frame = ReplicatedStateSourceFrame(cursor=self._cursor, value=value, ops=ops, context=context)
        for attachment in list(self._attachments):
            attachment.publish(frame)

    def close_session(self, failure: object = None) -> None:
        """Stop publishing; the attached states keep their last value. A failure ends them the same way."""
        if self._closed:
            return
        for attachment in list(self._attachments):
            attachment.dispose()
        self._finish_disposal()

    def _finish_disposal(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._value = None
        release = self._release
        self._release = None
        if release is not None:
            release()


class _SessionSourceAttachment(Generic[T]):
    """One attachment of a :class:`CommittedStateSource` (a ``ReplicatedStateSourceAttachment``)."""

    __slots__ = ("_activated", "_delivering", "_disposed", "_frames", "_listener", "_release", "_scheduled", "snapshot")

    def __init__(self, snapshot: ReplicatedStateSnapshot[T], release: Callable[[], None]) -> None:
        self.snapshot = snapshot
        self._release: Callable[[], None] | None = release
        self._frames: deque[ReplicatedStateSourceFrame[T]] = deque()
        self._listener: Callable[[ReplicatedStateSourceFrame[T]], None] | None = None
        self._activated = False
        self._disposed = False
        self._scheduled = False
        self._delivering = False

    def activate(self, listener: Callable[[ReplicatedStateSourceFrame[T]], None]) -> None:
        if self._activated:
            raise RuntimeError("State attachment is already active")
        if self._disposed:
            raise RuntimeError("State attachment is disposed")
        self._activated = True
        self._listener = listener
        self._drain()

    def publish(self, frame: ReplicatedStateSourceFrame[T]) -> None:
        if self._disposed:
            return
        self._frames.append(frame)
        if not self._activated or self._delivering or self._scheduled:
            return
        self._scheduled = True
        asyncio.get_running_loop().call_soon(self._run_scheduled)

    def _run_scheduled(self) -> None:
        self._scheduled = False
        if self._disposed:
            return
        try:
            self._drain()
        except Exception:
            # Chord's frame listener contains source-contract failures. Isolate an unexpected direct
            # listener failure to this attachment as well.
            self.dispose()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._frames.clear()
        self._listener = None
        release = self._release
        self._release = None
        if release is not None:
            release()

    def _drain(self) -> None:
        listener = self._listener
        if listener is None or self._delivering or self._disposed:
            return
        self._delivering = True
        try:
            while not self._disposed and self._frames:
                listener(self._frames.popleft())
        finally:
            self._delivering = False


class _WatchFrame(Generic[T]):
    __slots__ = ("context", "ops", "value")

    def __init__(self, value: T, ops: Sequence[Op], context: Context) -> None:
        self.value = value
        self.ops = ops
        self.context = context


class CommittedWatch(Generic[T]):
    """Serialized exact-frame watch bound to one document incarnation or conversation view.

    Implements :class:`pi_durable.types.WatchHandle`. A ``None`` value retires it. Internal: the
    Session and the Harness create it.
    """

    __slots__ = (
        "_cancellation_remove",
        "_closed",
        "_detach",
        "_end",
        "_listener",
        "_loop",
        "_pending",
        "_replace",
        "_report",
        "_retired",
        "_running",
        "_scheduled",
        "_started",
        "_value",
    )

    def __init__(
        self,
        value: T,
        detach: Callable[[], None],
        report: Callable[[BaseException], None],
        replace: Callable[[], T] | None = None,
    ) -> None:
        """``report`` gets a listener's raise, which also ends the watch. ``replace`` gives the value an
        overflow delivers; by default the newest value. Must be created on the running event loop."""
        self._value = value
        self._detach = detach
        self._report = report
        self._replace = replace
        self._loop = asyncio.get_running_loop()
        self._closed: asyncio.Future[WatchEnd] = self._loop.create_future()
        self._pending: deque[_WatchFrame[T]] = deque()
        self._listener: Callable[[T, Sequence[Op], Context], Any] | None = None
        self._started = False
        self._scheduled = False
        self._running = False
        self._retired = False
        #: Set once, by ``_terminate()``.
        self._end: WatchEnd | None = None
        self._cancellation_remove: Callable[[], None] | None = None

    @property
    def value(self) -> T:
        """Acquisition revision before start; latest delivered immutable revision afterward."""
        return self._value

    @property
    def closed(self) -> asyncio.Future[WatchEnd]:
        """Settles with the watch's terminal result; awaitable any number of times."""
        return self._closed

    def start(self, listener: Callable[[T, Sequence[Op], Context], Any]) -> None:
        """Install the sole listener (sync or async). Never invokes it inline."""
        if self._started:
            raise RuntimeError("Watch is already started")
        if self._end is not None:
            raise RuntimeError("Watch is stopped")
        self._started = True
        # Called with (value, ops, context); a listener may declare only the leading ones.
        self._listener = trim_arguments(listener, 3)
        if self._pending:
            self._schedule()

    def stop(self) -> asyncio.Future[WatchEnd]:
        """Idempotently stop future callbacks at once and return the terminal result (``closed``)."""
        self._terminate({"reason": "stopped"})
        return self._closed

    def observe_cancellation(self, signal: AbortSignal) -> None:
        """End the watch with ``cancelled`` when ``signal`` aborts. Installed once."""
        if self._cancellation_remove is not None:
            raise RuntimeError("Watch cancellation is already installed")
        if self._end is not None:
            return
        loop = self._loop
        thread = threading.get_ident()

        def on_abort() -> None:
            if threading.get_ident() == thread:
                self.cancel()
            else:
                loop.call_soon_threadsafe(self.cancel)

        self._cancellation_remove = signal.add_listener(on_abort)
        if signal.aborted:
            self.cancel()

    def cancel(self) -> None:
        self._terminate({"reason": "cancelled"})

    def close_session(self, failure: Any = None) -> None:
        """End with ``session_closed``, or with ``session_failed`` and the error when the Session failed.

        ``failure`` is the Session's failure (an object with ``error``), or ``None``.
        """
        self._terminate(
            {"reason": "session_closed"} if failure is None else {"reason": "session_failed", "error": failure.error}
        )

    def advance(self, value: T, ops: Sequence[Op], context: Context) -> None:
        if self._end is not None or self._retired:
            return
        if value is None:
            self._retired = True
        if len(self._pending) >= MAX_PENDING_WATCH_FRAMES:
            self._pending.clear()
            replacement = self._replace() if self._replace is not None else None
            if replacement is None:
                replacement = value
            self._pending.append(_WatchFrame(replacement, [("r", replacement)], context))  # type: ignore[list-item]
        else:
            self._pending.append(_WatchFrame(value, ops, context))
        if self._started:
            self._schedule()

    def _schedule(self) -> None:
        if self._scheduled or self._running or self._end is not None:
            return
        self._scheduled = True
        self._loop.call_soon(self._run_scheduled)

    def _run_scheduled(self) -> None:
        self._scheduled = False
        self._loop.create_task(self._drain())

    async def _drain(self) -> None:
        if self._running or self._end is not None or not self._started:
            return
        self._running = True
        try:
            while self._end is None and self._pending:
                frame = self._pending.popleft()
                self._value = frame.value
                delivery_context = without_abort_signal(frame.context)
                listener = self._listener
                assert listener is not None
                try:
                    result = listener(frame.value, frame.ops, delivery_context)
                    if inspect.isawaitable(result):
                        await result
                except Exception as error:
                    if self._end is None:
                        self._report(error)
                        self._terminate({"reason": "listener_error", "error": _to_error(error)})
                    break
                if frame.value is None:
                    self._terminate({"reason": "retired"})
                    break
        finally:
            self._running = False
            if self._end is None and self._pending:
                self._schedule()

    def _terminate(self, end: WatchEnd) -> None:
        """End the watch, once: detach, drop pending frames, and settle ``closed``. A running callback keeps running."""
        if self._end is not None:
            return
        self._end = end
        self._detach()
        self._pending.clear()
        remove = self._cancellation_remove
        if remove is not None:
            remove()
        if not self._closed.done():
            self._closed.set_result(end)


def _to_error(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else RuntimeError(error_message(error))
