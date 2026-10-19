"""Replicated state: an immutable JSON value with ordered, serialized change notifications (port of chord ``state.ts``).

Either authoritative and mutable (``replicated_state(initial)``), or a publication-only view
of an external revision source (``replicated_state(source)``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar, overload

from ._callbacks import trim_arguments
from .context import BACKGROUND_CONTEXT, Context
from .delta import Draft, Op, Prepared, Tracker, track

__all__ = [
    "ReplicatedStateDelivery",
    "ReplicatedState",
    "MutableReplicatedState",
    "AttachedReplicatedState",
    "ReplicatedStateSnapshot",
    "ReplicatedStateSourceFrame",
    "ReplicatedStateSourceAttachment",
    "ReplicatedStateSource",
    "ReplicatedStateSourceOptions",
    "StateListener",
    "replicated_state",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

_logger = logging.getLogger("pi_chord.state")

#: At most this many deliveries wait behind a subscriber's running callback.
_MAX_PENDING = 100


@dataclass(frozen=True, slots=True)
class ReplicatedStateDelivery:
    """What a subscriber callback is receiving: the hydration, or the update with this sequence."""

    kind: Literal["hydrate", "update"]
    sequence: int


#: A subscriber callback. It may return an awaitable; the next delivery waits for it.
StateListener = Callable[[T, Context, ReplicatedStateDelivery], "Awaitable[object] | object"]


@dataclass(frozen=True, slots=True)
class ReplicatedStateSnapshot(Generic[T]):
    """The fixed immutable value and cursor captured at an attachment boundary."""

    value: T
    cursor: int


@dataclass(frozen=True, slots=True)
class ReplicatedStateSourceFrame(Generic[T]):
    """One immutable authoritative revision committed after an attachment snapshot."""

    #: Monotonic source cursor. The first frame after a snapshot is ``snapshot.cursor + 1``.
    cursor: int
    #: The exact immutable value produced by this commit.
    value: T
    #: The exact immutable batch that produced ``value`` from the preceding source revision.
    ops: Sequence[Op]
    context: Context


class ReplicatedStateSourceAttachment(Protocol[T]):
    """One attachment to a :class:`ReplicatedStateSource`."""

    @property
    def snapshot(self) -> ReplicatedStateSnapshot[T]:
        """Fixed immutable snapshot captured at the atomic attachment boundary."""
        ...

    def activate(self, listener: Callable[[ReplicatedStateSourceFrame[T]], None]) -> None:
        """Install the sole listener and synchronously drain every buffered frame in commit order.

        Single-use. From then on every committed frame is delivered in order until disposal,
        including commits made reentrantly while a prior frame is being delivered.
        """
        ...

    def dispose(self) -> None:
        """Stop delivery and release source resources. Idempotent."""
        ...


class ReplicatedStateSource(Protocol[T]):
    """An authoritative immutable revision source.

    ``attach()`` must synchronously and atomically capture one snapshot and register the
    returned attachment to buffer every later committed frame: the snapshot includes every
    commit before that boundary, buffered frames every commit after it, with no overlap or
    gap. Snapshot values, frame values and batches are immutable and stay valid after
    delivery; chord only publishes these references, never applies or re-diffs them.
    """

    def attach(self) -> ReplicatedStateSourceAttachment[T]: ...


@dataclass(frozen=True, slots=True)
class ReplicatedStateSourceOptions:
    #: Receives source-contract and listener failures without raising them into the source.
    on_error: Callable[[Exception], None] | None = None


class ReplicatedState(Protocol[T_co]):
    """A contract-immutable value with serialized change notifications."""

    @property
    def value(self) -> T_co | None:
        """The current value (immutable by contract and possibly shared with the provider)."""
        ...

    def subscribe(self, listener: StateListener[Any]) -> Callable[[], None]:
        """Subscribe; returns the unsubscribe function.

        Each subscription serializes its callbacks, awaiting the hydration before updates.
        At most 100 deliveries wait behind the running callback; overflow keeps only the
        newest, so update sequences may skip. Failures are reported in isolation and
        delivery continues. Unsubscribing discards pending work without cancelling or
        joining a running callback.
        """
        ...


@dataclass(slots=True)
class _Delivery(Generic[T]):
    value: T
    context: Context
    delivery: ReplicatedStateDelivery


def _report_error_async(error: Exception) -> None:
    """Default reporter: hand the error to the running loop's exception handler, else log it."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _logger.error("Unhandled replicated state error", exc_info=error)
        return
    loop.call_exception_handler({"message": "Unhandled replicated state error", "exception": error})


def _report_default(error: Exception) -> None:
    # Looked up at call time so tests can replace the module-level reporter.
    _report_error_async(error)


def _to_error(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else RuntimeError(str(error))


def _discard_awaitable(result: object) -> None:
    if inspect.iscoroutine(result):
        result.close()
    elif isinstance(result, asyncio.Future):
        result.add_done_callback(_consume)


def _start(result: Awaitable[Any], loop: asyncio.AbstractEventLoop) -> asyncio.Future[Any]:
    """Run an awaitable listener result; a coroutine starts eagerly, like a JavaScript async function."""
    if inspect.iscoroutine(result):
        try:
            return asyncio.Task(result, loop=loop, eager_start=True)  # type: ignore[call-arg]
        except TypeError:  # Python 3.11: no eager start
            return loop.create_task(result)
    return asyncio.ensure_future(result, loop=loop)


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


class _StateSubscriber(Generic[T]):
    """One subscription, independent of producer and other subscriber progress."""

    __slots__ = ("_closed", "_listener", "_pending", "_report_error", "_running", "_started")

    def __init__(self, listener: StateListener[T], report_error: Callable[[Exception], None]) -> None:
        # Called with (value, context, delivery); a listener may declare only the leading ones.
        self._listener = trim_arguments(listener, 3)
        self._report_error = report_error
        self._pending: deque[_Delivery[T]] = deque()
        self._running = False
        self._started = False
        self._closed = False

    def push(self, frame: _Delivery[T]) -> None:
        if self._closed:
            return
        pending = self._pending
        if len(pending) == _MAX_PENDING:
            # A cold replica can queue updates reentrantly before the first hydration starts.
            hydration = None if self._started else pending[0]
            pending.clear()
            if hydration is not None:
                pending.append(hydration)
        pending.append(frame)

    def drain(self) -> None:
        if self._running or self._closed:
            return
        self._running = True
        pending = self._pending
        while pending:
            frame = pending.popleft()
            self._started = True
            try:
                result = self._listener(frame.value, frame.context, frame.delivery)
            except Exception as error:
                self._report(error)
                continue
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    _discard_awaitable(result)
                    self._report(TypeError("An asynchronous replicated state listener needs a running event loop"))
                    continue
                future = _start(result, loop)
                future.add_done_callback(self._settled)
                return
        self._running = False

    def _settled(self, future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                self._report(error)
        self._running = False
        self.drain()

    def clear(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self._closed = True
        self.clear()

    def _report(self, error: BaseException) -> None:
        try:
            self._report_error(_to_error(error))
        except Exception as report_error:
            _report_default(report_error)


@dataclass(slots=True)
class _Publication(Generic[T]):
    value: T
    sequence: int
    context: Context


class _ReplicatedStatePublisher(Generic[T]):
    """Maintains local publication order independently of how revisions are produced."""

    __slots__ = ("_delivering", "_listeners", "_publications", "_report_error", "_sequence", "_value")

    def __init__(self, initial: T, report_error: Callable[[Exception], None] = _report_default) -> None:
        self._listeners: dict[_StateSubscriber[T], int] = {}
        self._report_error = report_error
        self._publications: deque[_Publication[T]] = deque()
        self._value = initial
        self._sequence = 0
        self._delivering = False

    @property
    def value(self) -> T:
        return self._value

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        value = self._value
        sequence = self._sequence
        subscriber = _StateSubscriber(listener, self._report_error)
        self._listeners[subscriber] = sequence
        # Hydration has no caller; it is delivered with the background context.
        subscriber.push(_Delivery(value, BACKGROUND_CONTEXT, ReplicatedStateDelivery("hydrate", sequence)))
        subscriber.drain()

        def unsubscribe() -> None:
            subscriber.close()
            self._listeners.pop(subscriber, None)

        return unsubscribe

    def publish(self, value: T, context: Context) -> None:
        """Publish an already-prepared immutable revision. Subscriber failures are reported, never raised."""
        self._value = value
        self._sequence += 1
        self._publications.append(_Publication(value, self._sequence, context))
        if self._delivering:
            return
        self._delivering = True
        try:
            publications = self._publications
            while publications:
                publication = publications.popleft()
                delivery = ReplicatedStateDelivery("update", publication.sequence)
                for subscriber, hydrated in list(self._listeners.items()):
                    if publication.sequence <= hydrated:
                        continue
                    subscriber.push(_Delivery(publication.value, publication.context, delivery))
                    subscriber.drain()
        finally:
            self._delivering = False


class MutableReplicatedState(Generic[T]):
    """Authoritative replicated state: atomic draft changes and whole-root replacements."""

    __slots__ = ("_changing", "_publisher", "_tracker")

    def __init__(self, initial: T) -> None:
        self._tracker: Tracker[T] = track(initial)
        self._publisher: _ReplicatedStatePublisher[T] = _ReplicatedStatePublisher(self._tracker.value)
        self._changing = False

    @property
    def value(self) -> T:
        """The current revision. Immutable by contract; later updates never mutate it."""
        return self._tracker.value

    def change(self, context: Context, mutate: Callable[[Draft[T]], object]) -> None:
        """Atomically publish one synchronous draft mutation.

        Draft handles are unusable after ``mutate`` returns. Placed values are copied;
        assigning :data:`~pi_chord.json.UNDEFINED` to a key deletes it. If ``mutate`` raises,
        nothing changes.
        """
        if self._changing:
            raise RuntimeError("Replicated state cannot be changed reentrantly from a change callback")
        self._changing = True
        try:
            change = self._tracker.begin_change()
            try:
                outcome = mutate(change.state)
                if inspect.isawaitable(outcome):
                    _discard_awaitable(outcome)
                    raise TypeError("Replicated state change callbacks must be synchronous")
                prepared: Prepared[T] = change.prepare()
            except BaseException:
                change.abort()
                raise
        finally:
            self._changing = False
        self._tracker.adopt(prepared)
        if prepared.ops:
            self._publisher.publish(prepared.value, context)

    def replace(self, context: Context, value: T) -> None:
        """Atomically take immutable ownership of an alias-free strict-JSON replacement.

        The caller must not mutate ``value`` afterwards. A deeply equal replacement publishes nothing.
        """
        if self._changing:
            raise RuntimeError("Replicated state cannot be replaced from a change callback")
        prepared = self._tracker.prepare_replace(value)
        self._tracker.adopt(prepared)
        if prepared.ops:
            self._publisher.publish(prepared.value, context)

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        """See :meth:`ReplicatedState.subscribe`."""
        return self._publisher.subscribe(listener)


class AttachedReplicatedState(Generic[T]):
    """A synchronously hydrated, publication-only state backed by one source attachment."""

    __slots__ = ("_attachment", "_cursor", "_disposed", "_on_error", "_publisher")

    def __init__(self, attachment: ReplicatedStateSourceAttachment[T], options: ReplicatedStateSourceOptions) -> None:
        snapshot = attachment.snapshot
        _assert_cursor(snapshot.cursor, "snapshot")
        self._attachment = attachment
        self._cursor = snapshot.cursor
        self._on_error = options.on_error
        self._disposed = False
        self._publisher: _ReplicatedStatePublisher[T] = _ReplicatedStatePublisher(snapshot.value, self._report)

    @property
    def value(self) -> T:
        """The last published value; still readable after :meth:`dispose`."""
        return self._publisher.value

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        """See :meth:`ReplicatedState.subscribe`."""
        return self._publisher.subscribe(listener)

    def _activate(self) -> None:
        self._attachment.activate(self._receive)

    def dispose(self) -> None:
        """Release the source attachment. Idempotent; the last published value stays readable."""
        if self._disposed:
            return
        self._disposed = True
        self._attachment.dispose()

    def _receive(self, frame: ReplicatedStateSourceFrame[T]) -> None:
        if self._disposed:
            return
        try:
            _assert_cursor(frame.cursor, "frame")
            expected = self._cursor + 1
            if frame.cursor != expected:
                raise RuntimeError(
                    f"Replicated state source cursor has a gap: expected {expected}, received {frame.cursor}"
                )
            self._cursor = frame.cursor
            self._publisher.publish(frame.value, frame.context)
        except Exception as error:
            self._fail(error)

    def _fail(self, error: Exception) -> None:
        if self._disposed:
            return
        self._disposed = True
        try:
            self._attachment.dispose()
        except Exception as dispose_error:
            self._report(ExceptionGroup("Replicated state source contract failed", [error, dispose_error]))
            return
        self._report(error)

    def _report(self, error: Exception) -> None:
        reporter = self._on_error
        try:
            if reporter is None:
                _report_default(error)
            else:
                reporter(error)
        except Exception as report_error:
            _report_default(report_error)


def _assert_cursor(cursor: object, kind: str) -> None:
    if type(cursor) is not int or abs(cursor) > 2**53 - 1:
        raise TypeError(f"Replicated state source {kind} cursor must be a safe integer")


def _attach_source(
    source: ReplicatedStateSource[T], options: ReplicatedStateSourceOptions
) -> AttachedReplicatedState[T]:
    attachment = source.attach()
    try:
        state = AttachedReplicatedState(attachment, options)
        state._activate()
        return state
    except Exception as error:
        try:
            attachment.dispose()
        except Exception as dispose_error:
            raise ExceptionGroup("Failed to attach replicated state source", [error, dispose_error]) from None
        raise


@overload
def replicated_state(
    initial_or_source: ReplicatedStateSource[T], options: ReplicatedStateSourceOptions | None = None
) -> AttachedReplicatedState[T]: ...
@overload
def replicated_state(initial_or_source: T) -> MutableReplicatedState[T]: ...
def replicated_state(initial_or_source: Any, options: ReplicatedStateSourceOptions | None = None) -> Any:
    """Create replicated state.

    With a :class:`ReplicatedStateSource` (anything with a callable ``attach``), attach a
    publication-only state to its authoritative stream. Otherwise take immutable ownership
    of an alias-free strict-JSON ``dict`` or ``list`` root, which the caller must not mutate
    afterwards, and return authoritative :class:`MutableReplicatedState`.
    """
    if callable(getattr(initial_or_source, "attach", None)) and not isinstance(initial_or_source, (dict, list)):
        return _attach_source(initial_or_source, options or ReplicatedStateSourceOptions())
    return MutableReplicatedState(initial_or_source)
