"""The Session kernel: one mutation line, the document tracker cache, and committed publication (port of ``session/session.ts``).

Upstream's Promise-returning methods are plain methods here that return ``asyncio`` futures: their
synchronous part (admission to the line, cache lookups, sealing on close) runs at the call, as in
JavaScript, and callers ``await`` the result. Methods must be called on the running event loop.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeVar

from pi_ai.abort import AbortError, AbortSignal
from pi_chord.context import BACKGROUND_CONTEXT, Context, await_with_context, without_abort_signal
from pi_chord.delta import Op, track
from pi_chord.json import JsonObject
from pi_chord.state import ReplicatedStateSourceOptions, replicated_state

from ..documents import (
    AnyDocDefinition,
    AnyDocToken,
    check_record_scope,
    check_record_version,
    materialize_document,
    resolve_address,
)
from ..documents import (
    address_id as record_address_id,
)
from ..errors import SessionFailed, StorageRequestError
from ..ids import MAX_SAFE_INTEGER, id_from_number
from ..timers import Timers, host_timers
from ..types import (
    AnyTaskRecord,
    CommitChange,
    CommitListener,
    CommitPublication,
    ConversationDocToken,
    ConversationId,
    ConversationQuery,
    ConversationRecord,
    Cursor,
    DocumentAddress,
    DocumentCommitChange,
    DocumentCommitChangeDocument,
    DocumentId,
    DocumentPoint,
    DocumentQuery,
    DocumentRecord,
    EntryId,
    EntryQuery,
    EntryRecord,
    HeadMarker,
    Page,
    ReportInfo,
    Seq,
    Session,
    SessionEnd,
    SetTimer,
    Storage,
    StorageWrite,
    StoredDocument,
    StoredEntry,
    SubmissionId,
    SubmissionQuery,
    SubmissionRecord,
    TaskId,
    TaskQuery,
    Tx,
)
from ._futures import consume, rejected, resolved
from .observation import RETIREMENT_OPERATIONS, CommittedStateSource, CommittedWatch
from .transaction import LoadedDocument, Transaction, TransactionScope

__all__ = [
    "DocumentCacheSizes",
    "SessionFailure",
    "SessionImpl",
    "contained",
    "create_session",
    "date_now",
    "document_retention",
    "set_default_document_retention",
]

T = TypeVar("T")

#: The default source of a report: a listener, which never fails what ran it.
_LISTENER: Final[ReportInfo] = {"type": "listener_error"}


def date_now() -> int:
    """``Date.now()``: wall-clock epoch milliseconds as an ``int``."""
    return int(time.time() * 1000)


def create_session(
    storage: Storage,
    *,
    now: Callable[[], float] | None = None,
    document_retention_ms: float | None = None,
    set_timer: SetTimer | None = None,
) -> Session:
    """Open a Session kernel over one storage backend.

    ``now`` is the wall clock for task times (default :func:`date_now`). ``document_retention_ms`` is how long
    an unobserved document stays cached after its last use (default ten minutes). ``set_timer`` arms the
    timers that evict idle documents (see ``HarnessOptions.setTimer``). The first error a Storage method
    raises fails the Session (``SessionFailed``); see :class:`SessionImpl`.
    """
    retention = document_retention_ms
    return SessionImpl(storage, now, None if retention is None else (lambda: retention), set_timer)


#: Default document retention: ten minutes, as for kept contexts.
_default_retention: float = 600_000


def set_default_document_retention(ms: float) -> None:
    """Internal, for tests: the retention of every Session that keeps the default."""
    global _default_retention
    _default_retention = ms


def document_retention(ms: object) -> float:
    """A retention as given, when it is a number of milliseconds, ``0`` or more (``math.inf`` keeps
    documents); otherwise the default."""
    if isinstance(ms, int | float) and not isinstance(ms, bool) and ms >= 0:
        return ms
    return _default_retention


@dataclass(frozen=True, slots=True)
class SessionFailure:
    """The error that failed a Session (upstream ``{ error }``)."""

    error: BaseException


class DocumentCacheSizes(NamedTuple):
    """How many documents the cache holds, how many are idle, how many incarnations are pinned, and sweeps queued."""

    cached: int
    idle: int
    attached: int
    sweeps: int


class _Sweep:
    __slots__ = ("at", "disarm")

    def __init__(self, at: float) -> None:
        self.at = at
        self.disarm: Callable[[], None] | None = None


class _Host:
    """The Session services a transaction uses while it holds the line (``TransactionHost``)."""

    __slots__ = ("_session",)

    def __init__(self, session: SessionImpl) -> None:
        self._session = session

    @property
    def storage(self) -> Storage:
        return self._session.storage

    def now(self) -> float:
        return self._session._now()

    def cached(self, address_id: str) -> LoadedDocument | None:
        session = self._session
        cached = session._documents.get(address_id)
        if cached is not None:
            session._touch_document(address_id)
        return cached

    async def load(
        self, definition: AnyDocDefinition, address_id: str, address: DocumentAddress, context: Context
    ) -> LoadedDocument | None:
        return await self._session._load_document(definition, address_id, address, context)

    def install(self, document: LoadedDocument) -> None:
        session = self._session
        session._documents[document.address_id] = document
        session._touch_document(document.address_id)

    def evict(self, address_id: str, record_id: DocumentId) -> None:
        session = self._session
        cached = session._documents.get(address_id)
        if cached is not None and cached.record["id"] == record_id:
            session._drop_document(address_id)

    async def conversation_created(self, tx: Transaction, record: ConversationRecord) -> None:
        await self._session.conversation_created(tx, record)


class SessionImpl:
    """Session kernel: one mutation line, the loaded document tracker cache, and committed publication.

    The cache is bounded by use: a document with no watch or document state attached leaves it once it
    was not read or written for the retention period, and the next access loads it from Storage.
    Eviction runs as a job on the mutation line, so no transaction ever holds a tracker that leaves.

    Only committed state is observable. Every commit callback, preparation, Storage settlement, adoption,
    and publication enqueue runs while the line is held; listeners run later.

    Storage failures are final. Every Storage call goes through one guard: the first error a Storage
    method raises, other than for a cancelled caller, fails the Session, as does a commit it cannot adopt.
    That call gets the error; every later call and every Storage call still underway gets
    ``SessionFailed``; close listeners run at once with the error, and the Session then closes itself,
    Storage included, and ``closed`` settles. Nothing retries: transient errors are the Storage's to
    retry. Reopening recovers from what was committed.

    Members documented as *internal* serve the Harness and tests; *protected* ones are for subclasses.
    """

    def __init__(
        self,
        storage: Storage,
        now: Callable[[], float] | None = None,
        document_retention_ms: Callable[[], float | None] | None = None,
        set_timer: SetTimer | None = None,
    ) -> None:
        self._storage = GuardedStorage(storage, self)
        self._documents: dict[str, LoadedDocument] = {}
        #: Cached documents with no observer attached, by address, oldest use first, with the time of that use.
        self._idle_documents: dict[str, float] = {}
        #: Observers attached per document incarnation (record ID); a cached incarnation with any is never evicted.
        self._attached_documents: dict[DocumentId, int] = {}
        clock = now if now is not None else date_now
        retention = document_retention_ms if document_retention_ms is not None else (lambda: None)
        self._now: Callable[[], float] = clock
        self._timers: Timers = host_timers(
            set_timer, lambda error: self.report(error, {"type": "host_error", "callback": "setTimer"})
        )

        # Cache bookkeeping must never change the outcome of the operation it follows: a clock or a
        # retention setting that raises counts as now, and as the default.
        def cache_clock() -> float:
            try:
                return clock()
            except Exception:
                return date_now()

        def cache_retention() -> float:
            try:
                return document_retention(retention())
            except Exception:
                return document_retention(None)

        self._clock = cache_clock
        self._document_retention_ms = cache_retention
        #: The pending eviction: a timer for the oldest idle document, or a line job when it is due.
        self._document_sweep: _Sweep | None = None
        self._sweep_jobs = 0
        #: Set once the host declined a background timer: then none is kept, and each commit evicts what expired.
        self._document_timers_unavailable = False
        self._commit_listeners: dict[CommitListener, None] = {}
        #: The Session's own listeners, such as its scheduler's: they keep memory in step with storage, so a
        #: raise fails it.
        self._internal_listeners: dict[CommitListener, None] = {}
        self._close_listeners: dict[Callable[[], None], None] = {}
        self._host = _Host(self)
        self._tail: asyncio.Future[Any] | None = None
        self._closing: asyncio.Future[None] | None = None
        self._failure: SessionFailure | None = None
        self._failed_future: asyncio.Future[Any] | None = None
        self._end: SessionEnd | None = None
        self._closed_future: asyncio.Future[SessionEnd] | None = None

    @property
    def closed(self) -> asyncio.Future[SessionEnd]:
        """Settles once the Session has closed, Storage included; awaitable any number of times."""
        future = self._closed_future
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._closed_future = future
            if self._end is not None:
                future.set_result(self._end)
        return future

    @property
    def failure(self) -> SessionFailure | None:
        """Internal: the error that failed the Session, if one did."""
        return self._failure

    @property
    def failed(self) -> asyncio.Future[Any]:
        """Internal: fails with ``SessionFailed`` the moment the Session fails; never resolves."""
        future = self._failed_future
        if future is None:
            future = asyncio.get_running_loop().create_future()
            # Waits race it; none has to.
            future.add_done_callback(consume)
            self._failed_future = future
            if self._failure is not None:
                future.set_exception(SessionFailed(self._failure.error))
        return future

    @property
    def storage(self) -> Storage:
        """Protected: the Storage behind the failure guard; every component reads and writes through it."""
        return self._storage

    def commit(self, change: Callable[[Tx], T | Awaitable[T]], context: Context) -> asyncio.Future[T]:
        """Run one atomic transaction on the mutation line; admission happens at the call."""
        return self.commit_with(change, context)  # type: ignore[arg-type]

    def fail(self, error: BaseException) -> None:
        """Internal: fail the Session with ``error`` (the first wins) and report it once.

        Admission ends, and close listeners run now; they read the error from ``failure``. The Storage
        guard calls this; the Harness also calls it for a raise in its scheduler's own commits.
        """
        if self._failure is not None:
            return
        self._failure = SessionFailure(error)
        failed = self._failed_future
        if failed is not None and not failed.done():
            failed.set_exception(SessionFailed(error))
        # Seal first, so a report handler that calls back finds the Session failed. Closing runs the close
        # listeners synchronously; a failing backend close is the caller's to see, and here nobody waits.
        self.close(BACKGROUND_CONTEXT).add_done_callback(consume)
        self.report(error, {"type": "session_failed"})

    def commit_with(
        self,
        change: Callable[[Transaction], T | Awaitable[T]],
        context: Context,
        scope: TransactionScope | None = None,
    ) -> asyncio.Future[T]:
        """Internal commit exposing the concrete transaction and its internal operations.

        ``scope`` sets the default ``tx.create_task()`` conversation and the task attributed to appended entries.
        """
        try:
            self.assert_usable()
        except Exception as error:
            return rejected(error)
        return self._enqueue(lambda: self._run_commit(change, context, scope))

    def read_on_line(self, job: Callable[[], Awaitable[T]]) -> asyncio.Future[T]:
        """Internal: run a read-only job on the mutation line so multi-read derivations observe one committed state."""
        try:
            self.assert_usable()
        except Exception as error:
            return rejected(error)

        async def run() -> T:
            self._assert_healthy()
            return await job()

        return self._enqueue(run)

    async def conversation_document_on_line(
        self, token: ConversationDocToken[Any], conversation_id: ConversationId, context: Context
    ) -> tuple[DocumentRecord, int, JsonObject] | None:
        """Internal: a conversation document's current incarnation, value version, and value, for a job already
        running on the line (see ``read_on_line()``). Absent documents are ``None``."""
        definition = token.definition
        resolved_address = resolve_address(definition, [conversation_id, context])
        loaded = await self._load_document(definition, resolved_address.id, resolved_address.address, context)
        if loaded is None:
            return None
        check_record_scope(definition, loaded.record)
        check_record_version(definition, loaded.record, loaded.stored_version)
        return loaded.record, loaded.value_version, loaded.tracker.value

    def snapshot(self, token: AnyDocToken, *args: Any) -> asyncio.Future[Any]:
        """The committed value of a document, or ``None`` when absent; never creates it.

        Arguments after the token are the owner ID and the family key, as the token's scope requires, then
        the context.
        """
        try:
            self.assert_usable()
            definition = token.definition
            resolved_address = resolve_address(definition, args)
            context: Context = args[resolved_address.next_argument]
            cached = self._documents.get(resolved_address.id)
            if cached is not None and cached.value_version == definition.version:
                self._touch_document(resolved_address.id)
                check_record_scope(definition, cached.record)
                check_record_version(definition, cached.record, cached.stored_version)
                return resolved(cached.tracker.value)
        except Exception as error:
            return rejected(error)

        async def job() -> Any:
            self._assert_healthy()
            loaded = await self._load_document(definition, resolved_address.id, resolved_address.address, context)
            if loaded is None:
                return None
            check_record_scope(definition, loaded.record)
            check_record_version(definition, loaded.record, loaded.stored_version)
            return loaded.tracker.value

        return self._enqueue(job)

    def document_state(self, token: AnyDocToken, *args: Any) -> asyncio.Future[Any]:
        """A read-only Chord state of a document's committed incarnation, or ``None`` when absent."""
        try:
            self.assert_usable()
            definition = token.definition
            resolved_address = resolve_address(definition, args)
            context: Context = args[resolved_address.next_argument]
        except Exception as error:
            return rejected(error)

        async def job() -> Any:
            self._assert_healthy()
            loaded = await self._load_document(definition, resolved_address.id, resolved_address.address, context)
            if loaded is None:
                return None
            source, detach = self._attach_document(
                definition, loaded, lambda value, release: CommittedStateSource(value, release)
            )
            try:
                return replicated_state(source, ReplicatedStateSourceOptions(on_error=lambda error: self.report(error)))
            except BaseException:
                detach()
                raise

        return self._enqueue(job)

    def watch_doc(self, token: AnyDocToken, *args: Any) -> asyncio.Future[Any]:
        """A serialized exact-frame watch of a document's committed incarnation, or ``None`` when absent.

        Cancelling the context cancels an acquisition underway, and later ends the watch (``cancelled``).
        """
        try:
            self.assert_usable()
            definition = token.definition
            resolved_address = resolve_address(definition, args)
            context: Context = args[resolved_address.next_argument]
        except Exception as error:
            return rejected(error)
        signal = context.abort_signal
        cancelled = [signal is not None and signal.aborted]

        def mark_cancelled() -> None:
            cancelled[0] = True

        remove = signal.add_listener(mark_cancelled) if signal is not None else None

        async def job() -> CommittedWatch[Any] | None:
            self._assert_healthy()
            if cancelled[0]:
                raise _cancellation_error(signal)
            loaded = await self._load_document(definition, resolved_address.id, resolved_address.address, context)
            if cancelled[0]:
                raise _cancellation_error(signal)
            if loaded is None:
                return None
            watch, _detach = self._attach_document(
                definition, loaded, lambda value, release: CommittedWatch(value, release, lambda e: self.report(e))
            )
            return watch

        acquisition = self._enqueue(job)

        async def finish() -> CommittedWatch[Any] | None:
            try:
                watch = await acquisition
                if watch is None:
                    return None
                if cancelled[0]:
                    watch.cancel()
                    raise _cancellation_error(signal)
                if signal is not None:
                    watch.observe_cancellation(signal)
                return watch
            finally:
                if remove is not None:
                    remove()

        return asyncio.ensure_future(finish())

    def snapshot_as_of(self, token: AnyDocToken, *args: Any) -> asyncio.Future[Any]:
        """A rewindable conversation document's value as of a visible entry's commit, or ``None`` when absent then.

        Arguments after the token: the conversation ID, the family key for a family, the entry ID, the context.
        """
        try:
            self.assert_usable()
            definition = token.definition
            resolved_address = resolve_address(definition, args)
            scope = resolved_address.address["scope"]
            if scope["kind"] != "conversation":
                raise TypeError("Session.snapshot_as_of() requires a conversation document")
            conversation_id = scope["conversationId"]
            at_value = args[resolved_address.next_argument] if resolved_address.next_argument < len(args) else None
            if type(at_value) is not int or abs(at_value) > MAX_SAFE_INTEGER:
                raise TypeError("Session.snapshot_as_of() requires an entry ID")
            at: EntryId = id_from_number(at_value)
            context: Context = args[resolved_address.next_argument + 1]
        except Exception as error:
            return rejected(error)

        async def job() -> Any:
            self._assert_healthy()
            stored_entry = await self._storage.visible_entry(conversation_id, at, context)
            if stored_entry is None:
                raise RuntimeError(f"Entry {at} is not visible from conversation {conversation_id}")
            address: DocumentAddress = {
                **resolved_address.address,
                "scope": {"kind": "conversation", "conversationId": stored_entry["entry"]["conversationId"]},
            }
            record = await self._storage.find_document(address, stored_entry["commitSeq"], context)
            if record is None:
                return None
            stored = await self._storage.document(record["id"], stored_entry["commitSeq"], context)
            if stored is None:
                raise RuntimeError(f"Historical document {record['id']} ({record['kind']}) cannot be read")
            return materialize_document(definition, stored)

        return self._enqueue(job)

    def close(self, context: Context) -> asyncio.Future[None]:
        """Seal admission at once, settle admitted commits, then close Storage.

        Cancelling ``context`` stops only this wait.
        """
        closing = self._closing
        if closing is None:
            cleanup = without_abort_signal(context)
            closing = asyncio.ensure_future(self._close_sequence(cleanup))
            closing.add_done_callback(consume)
            self._closing = closing
            self._notify_close()
        return asyncio.ensure_future(await_with_context(asyncio.shield(closing), context))

    async def _close_sequence(self, cleanup: Context) -> None:
        # Seal admission before anything else runs, then stop observers; admitted work settles before Storage closes.
        try:
            await self.before_close()

            async def job() -> None:
                self._commit_listeners.clear()
                self._internal_listeners.clear()
                self._documents.clear()
                self._idle_documents.clear()
                self._attached_documents.clear()
                try:
                    await self._storage.close(cleanup)
                except Exception as error:
                    # A Storage that cannot close is a failed one; an earlier failure stays the cause.
                    if self._failure is None:
                        self._failure = SessionFailure(error)
                        self.report(error, {"type": "session_failed"})
                    raise

            await self._enqueue(job)
        finally:
            failure = self._failure
            self._settle_end({"reason": "closed"} if failure is None else {"reason": "failed", "error": failure.error})

    def _settle_end(self, end: SessionEnd) -> None:
        if self._end is not None:
            return
        self._end = end
        future = self._closed_future
        if future is not None and not future.done():
            future.set_result(end)

    def _notify_close(self) -> None:
        """Run the close listeners once, each on its own: one that fails is reported, and the rest still run."""
        sweep = self._document_sweep
        if sweep is not None and sweep.disarm is not None:
            sweep.disarm()
        self._document_sweep = None
        listeners = list(self._close_listeners)
        self._close_listeners.clear()
        for listener in listeners:
            contained(listener, lambda error: self.report(error))

    def report(self, error: BaseException, info: ReportInfo = _LISTENER) -> None:
        """Internal: the Session's failure, and errors of listeners, which never fail what ran them.

        ``info`` defaults to a listener. A plain Session drops them.
        """

    async def conversation_created(self, tx: Transaction, record: ConversationRecord) -> None:
        """Protected: runs inside every transaction that creates or forks a conversation, after the record is staged.

        A plain Session stages nothing; a Harness stages its built-in documents.
        """

    def write_at_close(self, writes: Callable[[], Sequence[StorageWrite]]) -> asyncio.Future[None]:
        """Protected, for ``before_close()``: on the line, after every commit admitted before the seal settled,
        write what ``writes()`` returns then straight to Storage.

        Nothing is published, since the Session is closing; a failed Session writes nothing.
        """

        async def job() -> None:
            if self._failure is not None:
                return
            batch = writes()
            if len(batch) > 0:
                await self._storage.commit(batch, BACKGROUND_CONTEXT)

        return self._enqueue(job)

    async def before_close(self) -> None:
        """Protected: runs after close seals admission and before the line closes Storage; must not raise."""

    def subscribe_commits(self, listener: CommitListener) -> Callable[[], None]:
        """Register a synchronous post-adoption listener; returns the unsubscribe function.

        It must not block or call Session operations; a raise is reported.
        """
        self.assert_usable()
        self._commit_listeners[listener] = None
        return lambda: self._commit_listeners.pop(listener, None) and None

    def observe_commits(self, listener: CommitListener) -> Callable[[], None]:
        """Internal: ``subscribe_commits()`` for the Session's own components, whose state follows each commit.

        A raise leaves that state behind storage, so it fails the Session instead of being reported. They
        run before host listeners.
        """
        self.assert_usable()
        self._internal_listeners[listener] = None
        return lambda: self._internal_listeners.pop(listener, None) and None

    def subscribe_close(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Register a listener called synchronously when close begins, also when a failure closes the Session.

        ``failure`` is then set. It must not block or call Session operations; a raise is reported.
        """
        self.assert_usable()
        self._close_listeners[listener] = None
        return lambda: self._close_listeners.pop(listener, None) and None

    def unload_documents(self) -> asyncio.Future[None]:
        """Drop every loaded tracker on the mutation line; later access cold-loads from Storage."""

        async def job() -> None:
            self._documents.clear()
            self._idle_documents.clear()

        return self._enqueue(job)

    @property
    def document_cache_sizes(self) -> DocumentCacheSizes:
        """Internal, for tests: cached and idle documents, pinned incarnations, and sweep jobs queued so far."""
        return DocumentCacheSizes(
            len(self._documents), len(self._idle_documents), len(self._attached_documents), self._sweep_jobs
        )

    def _touch_document(self, address_id: str) -> None:
        """Note a use of a cached document: an unattached one becomes the newest idle one, and an eviction is scheduled."""
        cached = self._documents.get(address_id)
        if cached is None or cached.record["id"] in self._attached_documents:
            return
        now = self._clock()
        since = self._idle_documents.get(address_id)
        # A hot document moves to the back at most once per granule, a tenth of the retention up to 1 s: it may
        # then leave up to a granule early, which costs a reload, while a commit-heavy workload skips the reordering.
        if since is not None and now - since < min(1_000, self._document_retention_ms() / 10):
            return
        self._idle_documents.pop(address_id, None)
        self._idle_documents[address_id] = now
        # A pending sweep for the oldest document comes first unless the retention was shortened since; then it
        # moves up.
        sweep = self._document_sweep
        if sweep is None or sweep.at > now + self._document_retention_ms():
            self._schedule_document_sweep()

    def _drop_document(self, address_id: str) -> None:
        self._documents.pop(address_id, None)
        self._idle_documents.pop(address_id, None)

    def _schedule_document_sweep(self) -> None:
        """Schedule the eviction of the oldest idle document.

        On the line at once when it is due, otherwise when it expires, by a background timer
        (``HarnessOptions.setTimer``). Where the host declines it, none is kept: the end of each commit
        evicts what expired.
        """
        if self._closing is not None:
            return
        oldest = next(iter(self._idle_documents.values()), None)
        if oldest is None:
            return
        at = oldest + self._document_retention_ms()
        if not math.isfinite(at):
            return
        current = self._document_sweep
        if current is not None and current.at <= at:
            return
        if current is not None and current.disarm is not None:
            current.disarm()
        # The pending sweep stays set until its job runs, so touches while the line is busy queue no more.
        sweep = _Sweep(at)

        def run() -> None:
            self._sweep_jobs += 1

            async def job() -> None:
                if self._document_sweep is sweep:
                    self._document_sweep = None
                if self._closing is None:
                    self._sweep_documents()

            self._enqueue(job).add_done_callback(consume)

        delay = at - self._clock()
        self._document_sweep = sweep
        if delay <= 0:
            asyncio.get_running_loop().call_soon(run)
            return
        if self._document_timers_unavailable:
            self._document_sweep = None
            return
        # A timer that fires early sweeps nothing and arms the next one.
        disarm = self._timers.background(run, delay)
        if disarm is None:
            self._document_timers_unavailable = True
            self._document_sweep = None
            return
        sweep.disarm = disarm

    def _sweep_documents(self) -> None:
        """On the line: evict the idle documents whose retention passed, oldest first, then schedule the next eviction."""
        expired = self._clock() - self._document_retention_ms()
        for address_id, since in list(self._idle_documents.items()):
            if since > expired:
                break
            self._drop_document(address_id)
        self._schedule_document_sweep()

    def _pin_document(self, address_id: str, record_id: DocumentId) -> Callable[[], None]:
        """Pin a cached incarnation for one observer; the returned release unpins it once, idle again after the last."""
        self._attached_documents[record_id] = self._attached_documents.get(record_id, 0) + 1
        self._idle_documents.pop(address_id, None)
        pinned = [True]

        def release() -> None:
            if not pinned[0]:
                return
            pinned[0] = False
            count = self._attached_documents.get(record_id, 1) - 1
            if count > 0:
                self._attached_documents[record_id] = count
                return
            self._attached_documents.pop(record_id, None)
            cached = self._documents.get(address_id)
            if cached is not None and cached.record["id"] == record_id:
                self._touch_document(address_id)

        return release

    async def _run_commit(
        self,
        change: Callable[[Transaction], T | Awaitable[T]],
        context: Context,
        scope: TransactionScope | None,
    ) -> T:
        self._assert_healthy()
        context.throw_if_aborted()
        tx = Transaction(self._host, context, scope)
        try:
            outcome = change(tx)
            result: T = await outcome if inspect.isawaitable(outcome) else outcome  # type: ignore[assignment]
            # A callback that caught a failed read must not commit, nor succeed as if it had.
            self._assert_healthy()
        except BaseException:
            await tx.settle_failure()
            self._sweep_documents()
            raise
        try:
            writes = await tx.settle_success()
        except BaseException:
            # Settlement aborted every change already.
            self._sweep_documents()
            raise
        if len(writes) == 0:
            tx.discard()
            self._sweep_documents()
            return result
        try:
            # Once admitted, caller cancellation does not interrupt Storage settlement.
            seq = await self._storage.commit(writes, without_abort_signal(context))
        except BaseException:
            # The guard has failed the Session.
            tx.discard()
            raise
        try:
            documents = tx.adopt(seq)
        except Exception as error:
            # Storage already committed; a failed adoption leaves memory behind durable state.
            self.fail(error)
            raise
        for document in documents:
            self._touch_document(record_address_id(document["record"]))  # type: ignore[arg-type]
        self._publish(seq, writes, documents, context)
        self._sweep_documents()
        return result

    def _publish(
        self,
        seq: Seq,
        writes: Sequence[StorageWrite],
        documents: Sequence[DocumentCommitChange],
        context: Context,
    ) -> None:
        if not self._commit_listeners and not self._internal_listeners:
            return
        changes: list[CommitChange] = []
        for write in writes:
            if write["type"] in ("conversation", "entry", "task", "submission"):
                changes.append(write)  # type: ignore[arg-type]
        changes.extend(documents)
        publication: CommitPublication = {"seq": seq, "changes": changes}
        # The commit is durable: a failing listener neither fails the commit nor skips the others.
        for listener in list(self._internal_listeners):
            contained(lambda listener=listener: listener(publication, context), self.fail)
        for listener in list(self._commit_listeners):
            contained(lambda listener=listener: listener(publication, context), lambda error: self.report(error))

    def _attach_document(
        self,
        definition: AnyDocDefinition,
        loaded: LoadedDocument,
        create: Callable[[JsonObject, Callable[[], None]], Any],
    ) -> tuple[Any, Callable[[], None]]:
        """Attach an observer to one committed incarnation: check the definition, then forward this
        incarnation's committed changes and close. The returned ``detach`` removes both subscriptions."""
        check_record_scope(definition, loaded.record)
        check_record_version(definition, loaded.record, loaded.stored_version)
        subscriptions: list[Callable[[], None]] = []
        # Attached, the incarnation stays cached; once its last observer detaches, or it retires, it is idle again.
        unpin = self._pin_document(loaded.address_id, loaded.record["id"])

        def detach() -> None:
            for unsubscribe in subscriptions:
                unsubscribe()
            unpin()

        try:
            observer = create(loaded.tracker.value, detach)
        except BaseException:
            unpin()
            raise
        observed = [loaded.value_version]
        record_id = loaded.record["id"]
        is_state = isinstance(observer, CommittedStateSource)

        def on_commit(publication: CommitPublication, context: Context) -> None:
            for change in publication["changes"]:
                if change["type"] != "document" or change["record"]["id"] != record_id:
                    continue
                # A document state's frames carry no caller cancellation; a watch observes its own cancellation.
                frame_context = without_abort_signal(context) if is_state else context
                ops = _observed_operations(observed, change)
                # A migration-only base changes nothing for an observer of the new version.
                if len(ops) == 0:
                    continue
                # A retired incarnation never changes again: its observers keep their last value but pin nothing.
                if change["value"] is None:
                    unpin()
                observer.advance(change["value"], ops, frame_context)

        subscriptions.append(self.observe_commits(on_commit))
        subscriptions.append(self.subscribe_close(lambda: observer.close_session(self._failure)))
        return observer, detach

    async def _load_document(
        self, definition: AnyDocDefinition, address_id: str, address: DocumentAddress, context: Context
    ) -> LoadedDocument | None:
        cached = self._documents.get(address_id)
        # A tracker serves only tokens of the version its value was materialized for; others reload from Storage.
        if cached is not None and cached.value_version == definition.version:
            self._touch_document(address_id)
            return cached
        if cached is not None:
            self._drop_document(address_id)
        record = await self._storage.find_document(address, "current", context)
        if record is None:
            return None
        stored = await self._storage.document(record["id"], "current", context)
        if stored is None:
            raise RuntimeError(f"Current document {record['id']} ({record['kind']}) cannot be read")
        value = materialize_document(definition, stored)
        loaded = LoadedDocument(
            address_id, stored["record"], stored["version"], definition.version, stored["deltasSinceBase"], track(value)
        )
        self._documents[address_id] = loaded
        self._touch_document(address_id)
        return loaded

    def _enqueue(self, job: Callable[[], Awaitable[T]]) -> asyncio.Future[T]:
        """Run ``job`` on the mutation line, after every job enqueued before it settled."""
        previous = self._tail

        async def run() -> T:
            if previous is not None and not previous.done():
                await asyncio.wait((previous,))
            return await job()

        task = asyncio.ensure_future(run())
        self._tail = task
        return task

    def assert_usable(self) -> None:
        """Internal: raise ``SessionFailed`` when failed, else an error when closing."""
        self._assert_healthy()
        if self._closing is not None:
            raise RuntimeError("Session is closed")

    def _assert_healthy(self) -> None:
        if self._failure is not None:
            raise SessionFailed(self._failure.error)


def contained(callback: Callable[[], object], on_error: Callable[[BaseException], None]) -> None:
    """Run a host callback that must not fail its caller.

    A raise, or an awaitable it returns that fails, goes to ``on_error``, which is not awaited.
    """
    try:
        result = callback()
        if inspect.isawaitable(result):
            future = asyncio.ensure_future(result)

            def settled(future: asyncio.Future[Any]) -> None:
                if future.cancelled():
                    return
                error = future.exception()
                if error is not None:
                    on_error(error)

            future.add_done_callback(settled)
    except Exception as error:
        on_error(error)


def _observed_operations(observed: list[int], change: DocumentCommitChangeDocument) -> Sequence[Op]:
    """Operations an observer applies for one committed change.

    An observer hydrated under another definition version holds a differently shaped value, so it receives
    the new value as a root replacement instead of operations for that shape.
    """
    if change["value"] is None:
        return RETIREMENT_OPERATIONS
    if change["version"] == observed[0]:
        return change["ops"]
    assert change["version"] is not None
    observed[0] = change["version"]
    return [("r", change["value"])]


def _cancellation_error(signal: AbortSignal | None) -> BaseException:
    reason = None if signal is None else signal.reason
    return reason if isinstance(reason, BaseException) else AbortError("The operation was aborted")


class GuardedStorage:
    """The Session's Storage behind its failure guard; every component of the Session reads and writes through it.

    Before a call, a failed Session gets ``SessionFailed`` without reaching the backend. A call that raises
    fails the Session and re-raises the backend's error, unless it is a ``StorageRequestError`` or a read
    whose caller's context was aborted: neither says the Storage is broken. ``close()`` always reaches the
    backend, so a failed Session still closes it.
    """

    __slots__ = ("_closing", "_idle", "_latch", "_storage", "_underway")

    def __init__(self, storage: Storage, latch: SessionImpl) -> None:
        self._storage = storage
        self._latch = latch
        #: Backend calls underway, also those off the Session line, which close waits for.
        self._underway = 0
        self._idle: asyncio.Future[None] | None = None
        self._closing = False

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        """Never exempt, whatever it raises: once admitted, whether a failed batch committed is unknown."""
        return await self._call(None, lambda: self._storage.commit(writes, context))

    async def mint_id(self) -> int:
        return await self._call(None, self._storage.mint_id)

    async def conversation(self, id: ConversationId, context: Context) -> ConversationRecord | None:
        return await self._call(context, lambda: self._storage.conversation(id, context))

    async def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[ConversationRecord]:
        return await self._call(context, lambda: self._storage.scan_conversations(query, limit, cursor, context))

    async def entry(self, id: EntryId, context: Context) -> StoredEntry | None:
        return await self._call(context, lambda: self._storage.entry(id, context))

    async def visible_entry(self, conversation_id: ConversationId, id: EntryId, context: Context) -> StoredEntry | None:
        return await self._call(context, lambda: self._storage.visible_entry(conversation_id, id, context))

    async def find_latest_head_marker(
        self, conversation_id: ConversationId, at_or_before_entry_id: EntryId | None, context: Context
    ) -> HeadMarker | None:
        return await self._call(
            context, lambda: self._storage.find_latest_head_marker(conversation_id, at_or_before_entry_id, context)
        )

    async def scan_entries(
        self, query: EntryQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[EntryRecord]:
        return await self._call(context, lambda: self._storage.scan_entries(query, limit, cursor, context))

    async def task(self, id: TaskId[Any], context: Context) -> AnyTaskRecord | None:
        return await self._call(context, lambda: self._storage.task(id, context))

    async def scan_tasks(
        self, query: TaskQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[AnyTaskRecord]:
        return await self._call(context, lambda: self._storage.scan_tasks(query, limit, cursor, context))

    async def submission(self, id: SubmissionId, context: Context) -> SubmissionRecord | None:
        return await self._call(context, lambda: self._storage.submission(id, context))

    async def scan_submissions(
        self, query: SubmissionQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[SubmissionRecord]:
        return await self._call(context, lambda: self._storage.scan_submissions(query, limit, cursor, context))

    async def submission_by_request(
        self, conversation_id: ConversationId, request_id: str, context: Context
    ) -> SubmissionRecord | None:
        return await self._call(
            context, lambda: self._storage.submission_by_request(conversation_id, request_id, context)
        )

    async def find_document(
        self, address: DocumentAddress, at: DocumentPoint, context: Context
    ) -> DocumentRecord | None:
        return await self._call(context, lambda: self._storage.find_document(address, at, context))

    async def document(self, id: DocumentId, at: DocumentPoint, context: Context) -> StoredDocument | None:
        return await self._call(context, lambda: self._storage.document(id, at, context))

    async def scan_documents(
        self, query: DocumentQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[DocumentRecord]:
        return await self._call(context, lambda: self._storage.scan_documents(query, limit, cursor, context))

    async def close(self, context: Context) -> None:
        """Close the backend once every call underway has settled, so none outlives it or fails after ``closed``.

        New calls are refused from here on: a paged read must not start its next page while the backend closes.
        """
        self._closing = True
        while self._underway > 0:
            idle = self._idle
            if idle is None:
                idle = asyncio.get_running_loop().create_future()
                self._idle = idle
            await idle
        await self._storage.close(context)

    async def _call(self, read: Context | None, run: Callable[[], Awaitable[T]]) -> T:
        """Run one backend call under the guard.

        ``read`` is a read's context; ``commit()`` and ``mint_id()`` pass none, so nothing exempts their
        errors. A call still underway when another fails the Session ends with ``SessionFailed`` too,
        whatever it returns: nothing it read or wrote is used.
        """
        self._assert_healthy()
        if self._closing:
            raise RuntimeError("Session is closed")
        self._underway += 1
        try:
            result = await run()
        except Exception as error:
            self._assert_healthy()
            # An invalid read, or one its caller cancelled, says nothing about the Storage; it fails that call only.
            signal = None if read is None else read.abort_signal
            exempt = read is not None and (
                isinstance(error, StorageRequestError) or (signal is not None and signal.aborted)
            )
            if not exempt:
                self._latch.fail(error)
            raise
        finally:
            self._underway -= 1
            if self._underway == 0 and self._idle is not None:
                idle = self._idle
                self._idle = None
                if not idle.done():
                    idle.set_result(None)
        self._assert_healthy()
        return result

    def _assert_healthy(self) -> None:
        failure = self._latch.failure
        if failure is not None:
            raise SessionFailed(failure.error)


if TYPE_CHECKING:

    def _implements_session(session: SessionImpl) -> Session:
        return session

    def _implements_storage(storage: GuardedStorage) -> Storage:
        return storage

    def _implements_watch(watch: CommittedWatch[Any]) -> Any:
        from ..types import WatchHandle

        handle: WatchHandle[Any] = watch
        return handle
