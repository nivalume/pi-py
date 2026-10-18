"""Async event streams with a final result.

:class:`EventStream` is a single-producer queue that consumers read with
``async for`` while ``await stream.result()`` yields the final value. It mirrors
pi's ``EventStream`` with a few Python-specific improvements:

* Streams can be created outside a running event loop; the producer starts on the
  first ``async for``/``result()`` call.
* The producer task is owned by the stream, so it cannot be garbage collected
  mid-flight, and ``await stream.aclose()`` (or ``async with stream:``) cancels it.
* An unexpected producer crash ends the stream and surfaces from ``result()``
  instead of leaving consumers waiting forever.
* Forwarding one stream into another (:meth:`EventStream.forward_to`) wires the
  producer directly into the target, so wrapped streams cost no extra queue hop
  per event.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Generic, TypeVar

from .types import AssistantMessage, AssistantMessageEvent, DoneEvent, ErrorEvent, now_ms

__all__ = [
    "EventStream",
    "AssistantMessageEventStream",
    "create_assistant_message_event_stream",
]

T = TypeVar("T")
R = TypeVar("R")

_UNSET: Any = object()


class EventStream(Generic[T, R]):
    """Generic event stream for async iteration with a final result."""

    __slots__ = (
        "__weakref__",
        "_closing",
        "_done",
        "_exception",
        "_extract_result",
        "_is_complete",
        "_pending",
        "_queue",
        "_result",
        "_result_waiters",
        "_sink",
        "_task",
        "_upstream",
        "_waiters",
    )

    def __init__(self, is_complete: Callable[[T], bool], extract_result: Callable[[T], R]) -> None:
        self._queue: deque[T] = deque()
        self._waiters: deque[asyncio.Future[Any]] = deque()
        self._done = False
        self._result: Any = _UNSET
        self._result_waiters: list[asyncio.Future[Any]] | None = None
        self._exception: BaseException | None = None
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._task: asyncio.Task[Any] | None = None
        self._pending: Coroutine[Any, Any, Any] | None = None
        self._sink: EventStream[T, R] | None = None
        self._upstream: EventStream[Any, Any] | None = None
        self._closing = False

    # -- producer side -----------------------------------------------------

    @property
    def done(self) -> bool:
        """True once a terminal event was pushed or :meth:`end` was called."""
        return self._done

    def push(self, event: T) -> None:
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            self._set_result(self._extract_result(event))
        sink = self._sink
        if sink is not None:
            sink.push(event)
            return
        waiters = self._waiters
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(event)
                return
        self._queue.append(event)

    def end(self, result: R | None = None) -> None:
        self._done = True
        if result is not None:
            self._set_result(result)
        sink = self._sink
        if sink is not None:
            sink.end(result if result is not None else (None if self._result is _UNSET else self._result))
            return
        self._wake_all()

    def start(self, producer: Coroutine[Any, Any, Any]) -> None:
        """Run ``producer`` as this stream's background task.

        The task starts immediately when an event loop is running; otherwise it
        starts on first consumption.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._pending = producer
            return
        self._launch(loop, producer)

    def forward_to(self, target: EventStream[T, R]) -> None:
        """Deliver this stream's events and result into ``target``.

        Already queued events are moved immediately; later events go straight to
        ``target.push`` without an intermediate task.
        """
        target._upstream = self
        self._ensure_started()
        queue = self._queue
        while queue:
            target.push(queue.popleft())
        if self._exception is not None:
            target._fail(self._exception)
            return
        if self._done:
            target.end(None if self._result is _UNSET else self._result)
            return
        self._sink = target

    async def aclose(self) -> None:
        """Cancel the producer (and any forwarded upstream producer) and end the stream."""
        self._closing = True
        upstream = self._upstream
        if upstream is not None:
            await upstream.aclose()
        pending = self._pending
        if pending is not None:
            self._pending = None
            pending.close()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        if not self._done:
            self._done = True
            self._wake_all()

    async def __aenter__(self) -> EventStream[T, R]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- consumer side -----------------------------------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        self._ensure_started()
        return self

    async def __anext__(self) -> T:
        queue = self._queue
        if queue:
            return queue.popleft()
        if self._done:
            if self._exception is not None:
                raise self._exception
            raise StopAsyncIteration
        self._ensure_started()
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        event = await waiter
        if event is _UNSET:
            if self._exception is not None:
                raise self._exception
            raise StopAsyncIteration
        return event

    async def result(self) -> R:
        """Wait for and return the stream's final result."""
        self._ensure_started()
        if self._result is not _UNSET:
            return self._result
        if self._exception is not None:
            raise self._exception
        if self._done:
            raise RuntimeError("Stream ended without a result")
        waiter = asyncio.get_running_loop().create_future()
        if self._result_waiters is None:
            self._result_waiters = []
        self._result_waiters.append(waiter)
        return await waiter

    # -- internals ---------------------------------------------------------

    def _set_result(self, result: R) -> None:
        if self._result is not _UNSET:
            return
        self._result = result
        waiters, self._result_waiters = self._result_waiters, None
        for waiter in waiters or ():
            if not waiter.done():
                waiter.set_result(result)

    def _wake_all(self) -> None:
        waiters = self._waiters
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(_UNSET)
        if self._result is _UNSET and self._result_waiters:
            waiters_r, self._result_waiters = self._result_waiters, None
            error = self._exception or RuntimeError("Stream ended without a result")
            for waiter in waiters_r:
                if not waiter.done():
                    waiter.set_exception(error)

    def _fail(self, error: BaseException) -> None:
        if self._done and self._result is not _UNSET:
            return
        self._exception = error
        self._done = True
        sink = self._sink
        if sink is not None:
            sink._fail(error)
            return
        self._wake_all()

    def __del__(self) -> None:
        # A stream created outside an event loop and never consumed: close its producer
        # instead of leaving a "coroutine was never awaited" warning behind.
        pending = getattr(self, "_pending", None)
        if pending is not None:
            self._pending = None
            pending.close()

    def _ensure_started(self) -> None:
        pending = self._pending
        if pending is not None:
            self._pending = None
            self._launch(asyncio.get_running_loop(), pending)

    def _launch(self, loop: asyncio.AbstractEventLoop, producer: Coroutine[Any, Any, Any]) -> None:
        task = loop.create_task(producer)
        self._task = task
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            if self._closing:
                if not self._done:
                    self.end()
            elif not self._done:
                # Never hand a CancelledError to consumers running in other tasks.
                self._fail(RuntimeError("Stream producer was cancelled"))
            return
        error = task.exception()
        if error is not None and not (self._done and self._result is not _UNSET):
            self._fail(error)
        elif not self._done and self._upstream is None:
            # The producer returned without a terminal event: never leave consumers waiting.
            self.end()


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Event stream of one assistant response.

    It also times the response: the final message (``done``/``error`` event, or the
    result passed to :meth:`end`) gets ``duration_ms`` measured with a monotonic
    clock from the stream's creation, unless the message already has one or its
    ``timestamp`` predates the stream (a forwarded response that started elsewhere).
    """

    __slots__ = ("_started_at", "_started_at_monotonic")

    def __init__(self) -> None:
        super().__init__(_is_terminal_event, _extract_terminal_message)
        self._started_at = now_ms()
        self._started_at_monotonic = time.perf_counter()

    def push(self, event: AssistantMessageEvent) -> None:
        kind = event.type
        if kind == "done":
            self._time(event.message)  # type: ignore[union-attr]
        elif kind == "error":
            self._time(event.error)  # type: ignore[union-attr]
        EventStream.push(self, event)

    def end(self, result: AssistantMessage | None = None) -> None:
        if result is not None:
            self._time(result)
        EventStream.end(self, result)

    def _time(self, message: AssistantMessage) -> None:
        if self._done or message.duration_ms is not None or message.timestamp < self._started_at:
            return
        message.duration_ms = max(0, round((time.perf_counter() - self._started_at_monotonic) * 1000))


def _is_terminal_event(event: AssistantMessageEvent) -> bool:
    kind = event.type
    return kind == "done" or kind == "error"


def _extract_terminal_message(event: AssistantMessageEvent) -> AssistantMessage:
    if isinstance(event, DoneEvent):
        return event.message
    if isinstance(event, ErrorEvent):
        return event.error
    raise ValueError("Unexpected event type for final result")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory for :class:`AssistantMessageEventStream` (for use in extensions)."""
    return AssistantMessageEventStream()
