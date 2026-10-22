"""The durable task scheduler of one Harness (port of ``harness/scheduler.ts``).

``queueMicrotask`` callbacks run as ``loop.call_soon`` callbacks; detached async work (``void promise``) runs
as tasks the scheduler keeps referenced until they finish.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict, cast

from pi_ai.abort import AbortController, AbortSignal
from pi_chord.context import Context, await_with_context, with_abort_signal
from pi_chord.json import copy_json
from pi_telemetry import (
    NOOP_TELEMETRY_CONTEXT,
    SchemaTelemetrySpan,
    TelemetryContext,
    bind_typed_span_starter,
    is_telemetry_enabled,
)

from ..errors import error_message
from ..session._futures import consume, rejected
from ..telemetry import GENERATION_SPAN, TASK_SPAN
from ..telemetry import STARTER as TELEMETRY_STARTER
from ..timers import Timers
from ..types import (
    AnyTask,
    AnyTaskRecord,
    CommitPublication,
    ConversationId,
    EntryId,
    ReportInfo,
    ReportTask,
    Storage,
    StorageWrite,
    TaskId,
)
from .agent import DEFAULT_RESTART_LIMIT, agent_hooks
from .context import read_context_from
from .types import Agent, RegistryReader, RegistrySnapshot, TaskInspection, Wake
from .util import Waiters, chain, closed_error, json_equal, optional_extension, scan_all

__all__ = [
    "BlockedReason",
    "BlockedSubject",
    "InvocationBinding",
    "SchedulerOutcome",
    "TaskScheduler",
    "TaskSchedulerOptions",
    "blocked_subject",
]

_SCAN_PAGE_SIZE: Final = 256
_LIVE_STATUSES: Final = ("pending", "running", "waiting", "sleeping", "completing")

#: Why a task cannot be reserved: no registered definition can take it, or it was cut off ``restartLimit`` times.
BlockedReason: TypeAlias = Literal["missing_task", "task_too_old", "migration_failed", "restart_limit"]


class BlockedSubject(TypedDict):
    """What ``blocked_reason()`` reads of a task: the task graph keeps only this of each record."""

    id: TaskId[Any]
    kind: str
    version: int
    abortRequested: bool
    abortReason: NotRequired[str]
    restarts: NotRequired[int]
    status: str


def blocked_subject(record: AnyTaskRecord) -> BlockedSubject:
    """The part of a record ``blocked_reason()`` reads."""
    subject: dict[str, Any] = {
        "id": record["id"],
        "kind": record["kind"],
        "version": record["version"],
        "abortRequested": record["abortRequested"],
    }
    if "abortReason" in record:
        subject["abortReason"] = record["abortReason"]
    if "restarts" in record:
        subject["restarts"] = record["restarts"]
    subject["status"] = record["state"]["status"]
    return subject  # type: ignore[return-value]


#: Terminal outcomes the scheduler writes without running task code (``faulted`` or ``orphaned``).
SchedulerOutcome: TypeAlias = dict[str, Any]


@dataclass(slots=True, eq=False)
class InvocationBinding:
    """An invocation a conversation handle is bound to: its signal, and a check that raises once it ended."""

    signal: AbortSignal
    check: Callable[[], None]


@dataclass(slots=True, eq=False)
class _Invocation:
    """One in-memory execution of a task in run or abort mode."""

    task_id: TaskId[Any]
    kind: str
    #: The extension that installed the definition this invocation runs, as of its reservation; none for a built-in.
    extension: str | None
    conversation_id: ConversationId
    mode: Literal["run", "abort"]
    controller: AbortController
    #: Context passed to handlers; cancelled by ``controller``.
    context: Context
    done: asyncio.Future[None]
    #: Watches acquired through the runtime; stopped at invocation end.
    watches: dict[Any, None] = field(default_factory=dict)
    #: Timers armed through the runtime; disarmed at invocation end.
    timers: dict[Callable[[], None], None] = field(default_factory=dict)
    ended: bool = False

    def finish(self) -> None:
        if not self.done.done():
            self.done.set_result(None)


@dataclass(slots=True, eq=False)
class _Phase:
    """What a runtime reads for the phase handler it serves: the phase's snapshot and task, and its agent."""

    snapshot: Callable[[], RegistrySnapshot]
    task: Callable[[], AnyTask]
    agent: asyncio.Future[Agent] | None = None
    #: The running handler's telemetry parent (``runtime.telemetry_context``): its span while one is open.
    telemetry: Any = NOOP_TELEMETRY_CONTEXT


@dataclass(slots=True, eq=False)
class _Reservation:
    invocation: _Invocation
    task: AnyTask
    snapshot: RegistrySnapshot


@dataclass(slots=True, eq=False)
class _RunState:
    task: AnyTask
    snapshot: RegistrySnapshot
    #: Replacement definition already reported as unable to take over: ``(task,)`` once reported.
    reported: tuple[AnyTask | None] | None = None


@dataclass(frozen=True, slots=True)
class _PhaseResult:
    """Outcome of the phase that just returned, judged by the next step."""

    checkpoint: Any
    failure: BaseException | None = None
    failed: bool = False


@dataclass(frozen=True, slots=True)
class _Fault:
    error: BaseException


#: Step decision: continue with the next phase, end the invocation, or end it by writing ``faulted``.
_Decision: TypeAlias = "bool | _Fault"


class _TaskNode(TypedDict):
    """The immutable ownership fields of a task."""

    conversationId: ConversationId
    owner: NotRequired[TaskId[Any]]
    background: bool


#: A node of the downward index: a task (an ``int``), or a conversation as ``"c<id>"``.
_Node: TypeAlias = "TaskId[Any] | str"


def _conversation_node(id: ConversationId) -> str:
    return f"c{id}"


def _conversation_of(node: str) -> ConversationId:
    return ConversationId(int(node[1:]))


def _parent_node(node: Any) -> _Node:
    """The node a task hangs from: its owner task, or its conversation."""
    owner = node.get("owner")
    return owner if owner is not None else _conversation_node(node["conversationId"])


#: Where a walk up the ownership tree continues: an owner task, or a conversation (``("task", id)``,
#: ``("conversation", id)``).
_Up: TypeAlias = tuple[str, int]


def _parent_of(node: Any) -> _Up:
    owner = node.get("owner")
    return ("task", owner) if owner is not None else ("conversation", node["conversationId"])


@dataclass(slots=True, eq=False)
class _Overlay:
    """Candidate records a commit staged; they override committed records in ownership walks."""

    tasks: dict[TaskId[Any], AnyTaskRecord]
    edges: dict[ConversationId, TaskId[Any] | None]
    #: Links to the staged tasks ``_below`` does not list, by parent node, for walks down; built at first use.
    links: dict[_Node, dict[_Node, None]] | None = None


@dataclass(frozen=True, slots=True)
class _Fit:
    """A definition that can take a record (``task``, ``migrates``), or why none can (``reason``, ``error``)."""

    task: AnyTask | None = None
    migrates: bool = False
    reason: BlockedReason | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class TaskSchedulerOptions:
    """What the Harness gives its scheduler."""

    session: Any
    storage: Storage
    registry: RegistryReader
    #: The host's current model access (``HarnessOptions.models``); read at each access.
    models: Callable[[], Any]
    #: Resolve a conversation's agent against a snapshot; the runtime calls it at most once per phase.
    agent: Callable[[ConversationId, RegistrySnapshot, Context], Awaitable[Agent]]
    #: Resolve the settings; read at each access.
    settings: Callable[[], Any]
    #: Build a conversation's environment with ``HarnessOptions.env``.
    env: Callable[[ConversationId, Context], Awaitable[Any]]
    now: Callable[[], float]
    #: ``HarnessOptions.onWake``: replaces the timer for sleeping tasks.
    on_wake: Callable[[Wake], None] | None
    #: The host's timers (``HarnessOptions.setTimer``).
    timers: Timers
    #: While closing, on the line after admitted commits settle, write what the callback returns straight to Storage.
    write_at_close: Callable[[Callable[[], Sequence[StorageWrite]]], Awaitable[None]]
    report: Callable[[BaseException, ReportInfo], None]
    #: Harness cleanup staged in the commit that makes an outcome the scheduler wrote itself terminal.
    settle_outcome: Callable[[Any, AnyTaskRecord, SchedulerOutcome], Awaitable[None]]
    #: Withdraw a conversation's queued inputs, for conversation abort and abort cascades.
    withdraw_inputs: Callable[[Any, ConversationId], Awaitable[None]]
    #: Invocation-bound handle of an existing conversation, for task runtimes and tools.
    conversation: Callable[[ConversationId, InvocationBinding, Context], Awaitable[Any]]
    #: Context for scheduler commits and invocations; carries no caller cancellation.
    context: Context
    #: ``HarnessOptions.telemetryContext``: parent of phase spans. Process-local, never persisted.
    telemetry_context: TelemetryContext | None = None
    #: The Harness's approval services (``pi.tool``'s policy and waiters); process-local.
    approvals: Any = None


_REGISTRY: Final[ReportInfo] = {"type": "host_error", "callback": "registry"}
_SETTINGS: Final[ReportInfo] = {"type": "host_error", "callback": "settings"}
_ON_WAKE: Final[ReportInfo] = {"type": "host_error", "callback": "onWake"}


class _IndexSizes(TypedDict):
    live: int
    settled: int
    edges: int
    below: int
    roots: int
    unloaded: int
    dropQueue: int


class TaskScheduler:
    """Durable task scheduler of one Harness.

    ``_live`` mirrors every committed non-terminal task record. The synchronous commit listener updates it on
    the Session line, so code running on the line reads exactly the committed state from it.

    Tasks and conversations form one ownership tree (spec §5.5): a task's parent is its owner task, or its
    conversation; a conversation's parent is its owner task, if any. ``_below`` lists the tree downward around
    live work. The ownership indexes are bounded by live work: an owner edge and an ended task's ownership
    fields are kept only while something is listed below them.

    Invariant: every task transition is decided and written by one callback serialized on the Session line.
    Handlers and joins run off the line. An invocation ends inside the step that decides its end, so a runtime
    commit it queued either lands before that decision or is rejected.
    """

    def __init__(self, options: TaskSchedulerOptions) -> None:
        self._session = options.session
        self._storage = options.storage
        self._registry = options.registry
        self._models = options.models
        self._agent = options.agent
        self._settings = options.settings
        self._env = options.env
        self._now = options.now
        self._on_wake = options.on_wake
        self._timers = options.timers
        self._write_at_close = options.write_at_close
        self._report = options.report
        self._settle_outcome = options.settle_outcome
        self._withdraw_inputs = options.withdraw_inputs
        self._conversation = options.conversation
        self._context = options.context
        if options.approvals is not None:
            self._approvals = options.approvals
        else:
            from .approvals import HarnessApprovals

            self._approvals = HarnessApprovals(None, lambda id: self.abort(id, self._context))
        telemetry = options.telemetry_context
        self._telemetry: TelemetryContext | None = telemetry if is_telemetry_enabled(telemetry) else None
        #: Open ``pi.harness.tool`` spans by task, so a nested call's span becomes a child of its caller's. Memory only.
        self._tool_spans: dict[TaskId[Any], Any] = {}
        self._live: dict[TaskId[Any], AnyTaskRecord] = {}
        #: Context range last read through a task runtime, per conversation, and when it was first seen idle.
        self._contexts: dict[ConversationId, list[Any]] = {}
        #: Background timer for the earliest expiry of an idle context: ``(at, disarm)``.
        self._expiry: tuple[float, Callable[[], None]] | None = None
        self._invocations: dict[TaskId[Any], _Invocation] = {}
        self._task_waiters: Waiters[TaskId[Any], Any] = Waiters()
        #: Idle waiters by conversation; ``None`` waits for the whole Harness.
        self._idle_waiters: Waiters[ConversationId | None, None] = Waiters()
        #: Definition whose migration failed per task; retried only once the registry resolves another definition.
        self._failed_migrations: dict[TaskId[Any], tuple[AnyTask, BaseException]] = {}
        #: Owner task of each known conversation, ``None`` when ownerless.
        self._edges: dict[ConversationId, TaskId[Any] | None] = {}
        #: Ownership fields of the known terminal tasks, which walks pass through.
        self._settled: dict[TaskId[Any], _TaskNode] = {}
        #: Ended tasks and conversations found unneeded, dropped by ``_sweep()`` unless needed again by then.
        self._drop_queue: dict[_Node, None] = {}
        #: The tree downward around live work: each node's child nodes that are live tasks or have nodes below them.
        self._below: dict[_Node, dict[_Node, None]] = {}
        self._roots: dict[ConversationId, None] = {}
        #: Known tasks whose chain may not be loaded.
        self._unloaded: dict[TaskId[Any], None] = {}
        #: Live tasks with cancellation intent, the sources of abort cascades (spec §5.4).
        self._intent: dict[TaskId[Any], None] = {}
        #: Live tasks that are not ``completing``, the candidates of a reservation pass.
        self._runnable: dict[TaskId[Any], None] = {}
        #: ``completing`` tasks the next finalize pass checks.
        self._finalize_checks: dict[TaskId[Any], None] = {}
        #: Live waiting tasks by each task in their ``on``.
        self._waiters: dict[TaskId[Any], dict[TaskId[Any], None]] = {}
        #: ``failFast`` waiters the next reconcile checks for a failed task in ``on``.
        self._fail_fast_checks: dict[TaskId[Any], None] = {}
        #: Live tasks with ``abandonOnRestart`` found at open, from an earlier Harness.
        self._abandoned: dict[TaskId[Any], None] = {}
        #: Sleeping live tasks, due or not: the candidates of ``_wake_due()``.
        self._sleeping: dict[TaskId[Any], None] = {}
        #: Timer for the earliest sleep, without ``onWake`` while scheduling runs: ``(at, disarm)``.
        self._timer: tuple[float, Callable[[], None]] | None = None
        #: Last value given to ``onWake``, so an unchanged one is not repeated.
        self._last_wake: Wake | None = None
        self._wake_scheduled = False
        self._blocked_listeners: dict[Callable[[], None], None] = {}
        #: A registry snapshot that raised for the task graph was reported; it is read per record, so only once.
        self._snapshot_reported = False
        #: Tasks a pass left out as parked; each pass checks them again, as ``settings.restartLimit`` may have risen.
        self._parked_tasks: dict[TaskId[Any], None] = {}
        self._reconcile_scheduled = False
        #: Reconcile passes underway with work (they may overlap).
        self._reconciling_work = 0
        self._sweep_scheduled = False
        self._cascade_pending = False
        self._unsubscribe_registry: Callable[[], None] = lambda: None
        self._enabled = False
        self._closing = False
        self._dirty = False
        self._draining = False
        #: Detached work, referenced until it finishes.
        self._tasks: set[asyncio.Future[Any]] = set()

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _spawn(self, awaitable: Awaitable[Any]) -> asyncio.Future[Any]:
        future = asyncio.ensure_future(awaitable)
        self._tasks.add(future)

        def done(finished: asyncio.Future[Any]) -> None:
            self._tasks.discard(finished)
            consume(finished)

        future.add_done_callback(done)
        return future

    @staticmethod
    def _soon(callback: Callable[[], None]) -> None:
        asyncio.get_running_loop().call_soon(callback)

    # ─── Lifecycle ─────────────────────────────────────────────────────────

    async def open(self, context: Context) -> None:
        """Load live tasks and change surviving ``running`` tasks back to ``pending``, counting the restart.

        Dispatches nothing. A task that reaches the restart limit this way is parked and reported.
        """
        self._session.observe_commits(lambda publication, _context: self._observe(publication))
        self._session.subscribe_close(self._seal)

        def on_registry() -> None:
            self._notify_blocked()
            self._kick()

        self._unsubscribe_registry = self._registry.subscribe(on_registry)
        limit = self._restart_limit()
        parked: list[AnyTaskRecord] = []

        async def change(tx: Any) -> None:
            # Every table read before the first write.
            scans: list[list[AnyTaskRecord]] = []
            for status in _LIVE_STATUSES:
                scans.append(
                    await scan_all(
                        lambda cursor, status=status: tx.scan_tasks({"status": status}, _SCAN_PAGE_SIZE, cursor)
                    )
                )
            for records in scans:
                for record in records:
                    self._track(None, record)
                    if record.get("abandonOnRestart") is True:
                        self._abandoned[record["id"]] = None
                    state = record["state"]
                    if state["status"] == "running":
                        restarts = record.get("restarts", 0) + 1
                        pending = _with_state(record, {"status": "pending", "checkpoint": state["checkpoint"]})
                        tx.set_task({**pending, "restarts": restarts})
                        if restarts >= limit and not record["abortRequested"]:
                            parked.append(record)
                    if state["status"] == "waiting" and state["policy"] == "failFast":
                        self._fail_fast_checks[record["id"]] = None

        await self._session.commit_with(change, context)
        for record in parked:
            restarts = record.get("restarts", 0) + 1
            error = RuntimeError(
                f"Task {record['id']} ({record['kind']}) was cut off {restarts} times without progress; "
                "it is parked until retry_task()"
            )
            self._report(error, {"type": "task_parked", **self._about_task(_record_task(record))})  # type: ignore[typeddict-item]
        # Derive abort marks a crash left unapplied below cancelled owners, and finalize held outcomes.
        self._cascade_pending = True
        self._schedule_reconcile()
        self._schedule_wake()

    def resume(self) -> None:
        """Enable scheduling, and make sleeping tasks that are due runnable. Idempotent."""
        self._enabled = True
        self._wake_due()
        self._kick()

    async def retry(self, id: TaskId[Any], context: Context) -> None:
        """Run a task parked by the restart limit again: clears its restart count and enables scheduling."""

        async def change(tx: Any) -> None:
            current = await tx.task(id)
            if current is None:
                raise RuntimeError(f"Task {id} does not exist")
            status = current["state"]["status"]
            if status in ("terminal", "completing") or not self._parked(current):
                return
            tx.set_task(_progressed(current))

        await self._session.commit_with(change, context)
        self.resume()

    def blocked_reason(self, record: BlockedSubject) -> BlockedReason | None:
        """Why a live task cannot run here, if it cannot: its definition (missing, too old, failed migration) or
        the restart limit. A running task, an abort-marked one, and one holding its outcome are never blocked."""
        status = record["status"]
        if status in ("running", "completing", "terminal"):
            return None
        # An abort request runs its handler or orphans the task; a ``restart`` mark waits for the definition.
        if record["abortRequested"] and "abortReason" not in record:
            return None
        if self._parked(record):
            return "restart_limit"
        try:
            snapshot = self._registry.snapshot()
        except Exception as error:
            if not self._snapshot_reported:
                self._report(error, _REGISTRY)
            self._snapshot_reported = True
            return None
        fit = self._fit(record, snapshot.task(record["kind"]))
        if fit.reason is not None:
            return fit.reason
        return "migration_failed" if fit.migrates and fit.task.definition.migrate is None else None  # type: ignore[union-attr]

    def subscribe_blocked(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` when blocked reasons may have changed without a commit."""
        self._blocked_listeners[listener] = None
        return lambda: self._blocked_listeners.pop(listener, None) and None

    def wake(self) -> Wake:
        """When the Session next needs a process: ``active`` while some task can run now or is running, ``at``
        for the earliest sleep not yet due. Synchronous, from memory."""
        now = self._now()
        # A reconcile pass with work to do is work too: it finalizes held outcomes and marks cascades.
        active = len(self._invocations) > 0 or self._reconciling_work > 0 or self._reconcile_work()
        at: float | None = None
        snapshot: RegistrySnapshot | None = None
        snapshot_failed = False
        for record in self._live.values():
            state: Any = record["state"]
            status = state["status"]
            if status == "completing":
                continue
            if status == "waiting" and not record["abortRequested"] and any(id in self._live for id in state["on"]):
                continue
            if status == "sleeping" and not record["abortRequested"] and state["until"] > now:
                at = state["until"] if at is None else min(at, state["until"])
                continue
            if active:
                continue
            if self._parked(record):
                continue
            # An abort invocation waits for the live work below it, which counts itself.
            if record["abortRequested"] and len(self._waiting_on(record, True)) > 0:
                continue
            # A task no definition can take waits for a registry change, except an abort-marked one, orphaned.
            if not record["abortRequested"] or "abortReason" in record:
                if snapshot is None and not snapshot_failed:
                    try:
                        snapshot = self._registry.snapshot()
                    except Exception as error:
                        snapshot_failed = True
                        self._report(error, _REGISTRY)
                if snapshot is not None and self._fit(record, snapshot.task(record["kind"])).reason is not None:
                    continue
            active = True
        return {"active": active} if at is None else {"active": active, "at": at}

    async def join(self) -> None:
        """Signal every invocation and wait for them, then write the tasks they ran back as ``pending``.

        A close is not a crash. Close calls this after every close listener has run, so waits, watches, and
        streams a task holds end with the Session's reason, not as cancelled by its signal.
        """
        stopped = list(self._invocations.values())
        for invocation in stopped:
            invocation.controller.abort()
        if stopped:
            await asyncio.wait([invocation.done for invocation in stopped])
        if len(stopped) == 0:
            return

        def writes() -> list[StorageWrite]:
            batch: list[StorageWrite] = []
            for invocation in stopped:
                record = self._live.get(invocation.task_id)
                if record is None or record["state"]["status"] != "running":
                    continue
                value = _with_state(record, {"status": "pending", "checkpoint": record["state"]["checkpoint"]})
                batch.append({"type": "task", "value": value})
            return batch

        # A close is not a crash: a task it stopped is ``pending`` again, so the next open counts no restart for it.
        # A failure leaves them ``running``, counted as if cut off; the Storage guard fails the Session.
        try:
            await self._write_at_close(writes)
        except Exception:
            pass

    def abort(
        self, id: TaskId[Any], context: Context, keep_restart: bool = False
    ) -> asyncio.Future[Literal["marked", "terminal"]]:
        """Commit the abort mark, or settle a task that no registered definition can take as ``orphaned`` when
        nothing it owns is live, then join the run invocation seen on the line."""

        async def change(tx: Any) -> tuple[Literal["marked", "terminal"], _Invocation | None]:
            current = await tx.task(id)
            if current is None:
                raise RuntimeError(f"Task {id} does not exist")
            if current["state"]["status"] == "terminal":
                return "terminal", None
            invocation = self._invocations.get(id)
            # A restart-marked task has no run invocation; it only waits for its abort.
            if keep_restart and current.get("abortReason") == "restart":
                return "marked", None
            if invocation is None and current["state"]["status"] != "completing":
                await self._load_scopes(False)
                if not self._has_owned_live(id):
                    blocked = self._resolve(current, self._registry.snapshot())
                    if isinstance(blocked, str):
                        await self._terminate(tx, current, {"status": "orphaned", "reason": blocked})
                        return "marked", None
            if not current["abortRequested"] or "abortReason" in current:
                tx.set_task(_with_abort_mark(current))
            return "marked", invocation if invocation is not None and invocation.mode == "run" else None

        def join(marked: tuple[Literal["marked", "terminal"], _Invocation | None]) -> Any:
            result, run = marked
            # The commit listener signalled the run; join it. A run that ignores its signal can outlive a failed
            # Session, which ends the wait instead.
            if run is None:
                return result
            return chain(_race_failed(run.done, self._session.failed), lambda _done: result, context)

        # The mark joins the line at the call; the join settles through callbacks, as upstream's awaits do.
        return chain(self._session.commit_with(change, context), join)

    def wait_for_task(self, id: TaskId[Any], context: Context) -> asyncio.Future[Any]:
        """Resolve with the terminal receipt; check and register on the line so no terminal publication falls between."""

        async def check() -> Awaitable[Any]:
            if self._closing:
                raise closed_error(self._session)
            if id in self._live:
                return self._task_waiters.add(id, context)
            record = await self._storage.task(id, context)
            if record is None:
                raise RuntimeError(f"Task {id} does not exist")
            done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            done.set_result(record)
            return done

        found = self._session.read_on_line(check)

        async def wait() -> Any:
            return await (await found)

        return asyncio.ensure_future(wait())

    def wait_for_idle(self, conversation_id: ConversationId | None, context: Context) -> asyncio.Future[None]:
        """Resolve when ordinary traversal from the conversation, or from every ownerless conversation, reaches no
        live non-background task."""
        if self._closing:
            return rejected(closed_error(self._session))
        if self._idle(conversation_id):
            done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done
        self._schedule_reconcile()
        return self._idle_waiters.add(conversation_id, context)

    async def abort_conversation(self, conversation_id: ConversationId, background: bool, context: Context) -> None:
        """``Conversation.abort()``: in one commit, withdraw the queued inputs and mark every live non-background
        task that ordinary traversal from the conversation reaches; resolves once the scope is idle. With
        ``background``, traversal crosses background boundaries, and the wait also covers every task it reached."""

        async def change(tx: Any) -> list[TaskId[Any]]:
            queued = await self._load_scopes(True)
            scope = ("conversation", conversation_id)
            reached: list[TaskId[Any]] = []
            for _id, node, live in self._walk_down(_conversation_node(conversation_id), cross_background=background):
                if live is None or (node["background"] and not background):
                    continue
                reached.append(live["id"])
                if not live["abortRequested"] or "abortReason" in live:
                    tx.set_task(_with_abort_mark(live))
            for id in queued:
                if self._in_scope(("conversation", id), scope, background) is True:
                    await self._withdraw_inputs(tx, id)
            return reached

        reached = await self._session.commit_with(change, context)
        if background:
            for id in reached:
                await self.wait_for_task(id, context)
        await self.wait_for_idle(conversation_id, context)

    # ─── Scheduling ────────────────────────────────────────────────────────

    def _observe(self, publication: CommitPublication) -> None:
        updated: list[AnyTaskRecord] = []
        failed: list[TaskId[Any]] = []
        changed = False
        # Edges first, so the tasks a commit creates in a new conversation hang from it.
        for change in publication["changes"]:
            if change["type"] == "conversation" and change["value"]["id"] not in self._edges:
                owner = change["value"].get("owner")
                self._set_edge(change["value"]["id"], None if owner is None else owner["taskId"])
        for change in publication["changes"]:
            if change["type"] != "task":
                continue
            changed = True
            record = change["value"]
            previous = self._live.get(record["id"])
            if _failed_outcome(record) and (previous is None or not _failed_outcome(previous)):
                failed.append(record["id"])
            self._track(previous, record)
            status = record["state"]["status"]
            if status == "terminal":
                self._failed_migrations.pop(record["id"], None)
                self._fail_fast_checks.pop(record["id"], None)
                self._task_waiters.resolve(record["id"], record)
                # Its owner may finalize now.
                self._schedule_reconcile()
                continue
            if record["abortRequested"] and (previous is None or previous["abortRequested"] is not True):
                self._cascade_pending = True
                # Signal a run invocation of the newly marked task; its next step ends it.
                invocation = self._invocations.get(record["id"])
                if invocation is not None and invocation.mode == "run":
                    invocation.controller.abort()
            # A request replacing a ``restart`` mark upgrades the marks of the work below.
            if previous is not None and "abortReason" in previous and "abortReason" not in record:
                self._cascade_pending = True
            if status == "completing" and (previous is None or previous["state"]["status"] != "completing"):
                if _cancellation_intent(record):
                    self._cascade_pending = True
                self._schedule_reconcile()
            if (
                status == "waiting"
                and cast(Any, record["state"])["policy"] == "failFast"
                and (previous is None or previous["state"]["status"] != "waiting")
            ):
                self._fail_fast_checks[record["id"]] = None
                self._schedule_reconcile()
            # Its owners and place never change: what is above it is checked once, when it becomes live.
            if previous is None:
                updated.append(record)
        for id in failed:
            for waiter in list(self._waiters.get(id, ())):
                waiting = self._live.get(waiter)
                state = None if waiting is None else waiting["state"]
                if state is not None and state["status"] == "waiting" and state["policy"] == "failFast":
                    self._fail_fast_checks[waiter] = None
                    self._schedule_reconcile()
        for change in publication["changes"]:
            # A queued input below a cancelled owner is withdrawn, even after its cascade.
            if (
                change["type"] != "submission"
                or change["value"]["status"] != "queued"
                or change["value"]["type"] != "input"
            ):
                continue
            # Only an owner with cancellation intent withdraws it; a later intent's mark runs the cascade then.
            if len(self._intent) == 0:
                continue
            conversation = change["value"]["conversationId"]
            if not self._known(_conversation_node(conversation)) or self._below_cancelled(
                ("conversation", conversation)
            ):
                self._cascade_pending = True
        for record in updated:
            # Work created below a cancelled owner, even after its cascade, is aborted too.
            if record["id"] in self._unloaded:
                self._schedule_reconcile()
            elif (
                not record["background"] and not record["abortRequested"] and self._below_cancelled(_parent_of(record))
            ):
                self._cascade_pending = True
        # A cascade that a reservation or step found pending runs with the next commit of any kind.
        if self._cascade_pending:
            self._schedule_reconcile()
        self._sweep()
        if not changed:
            return
        self._settle_idle()
        self._kick()

    def _context_retention_ms(self) -> float:
        """``settings.contextRetentionMs``, or 0 when the host's settings raise: kept contexts are only a cache."""
        try:
            return self._settings()["contextRetentionMs"]
        except Exception as error:
            self._report(error, _SETTINGS)
            return 0

    def _settle_idle(self) -> None:
        """Resolve idle waiters, and drop each kept context whose conversation has been idle for the retention."""
        for conversation_id in self._idle_waiters.pending():
            if self._idle(conversation_id):
                self._idle_waiters.resolve(conversation_id, None)
        now = self._now()
        retention = self._context_retention_ms()
        for conversation_id, kept in list(self._contexts.items()):
            idle_since = kept[1]
            if idle_since is not None and now - idle_since >= retention:
                del self._contexts[conversation_id]
            elif not self._idle(conversation_id):
                kept[1] = None
            elif retention > 0:
                if kept[1] is None:
                    kept[1] = now
            else:
                del self._contexts[conversation_id]
        self._schedule_expiry()

    def _schedule_expiry(self) -> None:
        """Run ``_settle_idle()`` when the earliest idle context expires, on a background timer.

        Where the host declines it, none is kept: task changes alone check expiry then.
        """
        retention = self._context_retention_ms()
        at: float | None = None
        for _range, idle_since in self._contexts.values():
            if idle_since is not None and (at is None or idle_since + retention < at):
                at = idle_since + retention
        if self._expiry is not None and self._expiry[0] == at:
            return
        if self._expiry is not None:
            self._expiry[1]()
        self._expiry = None
        if at is None or self._closing:
            return

        def fire() -> None:
            self._expiry = None
            self._settle_idle()

        disarm = self._timers.background(fire, at - self._now())
        if disarm is not None:
            self._expiry = (at, disarm)

    # ─── Ownership ─────────────────────────────────────────────────────────

    def _schedule_reconcile(self) -> None:
        if self._reconcile_scheduled or self._closing:
            return
        self._reconcile_scheduled = True
        # A task's first step runs on the next loop iteration, as a microtask would.
        self._spawn(self._reconcile())

    async def _reconcile(self) -> None:
        """One commit that applies what committed records imply: abort marks below live owners with cancellation
        intent (spec §5.4), ``failFast`` marks (spec §5.5), withdrawn queued inputs below cancelled owners, and the
        final terminal record of every ``completing`` task whose ordinary owned work is gone."""
        self._reconcile_scheduled = False
        work = self._reconcile_work()
        if work:
            self._reconciling_work += 1
        cascade = self._cascade_pending
        self._cascade_pending = False
        checks = list(self._fail_fast_checks)
        self._fail_fast_checks.clear()

        async def change(tx: Any) -> None:
            if self._closing:
                return
            queued = await self._load_scopes(cascade)
            # A cascade from an abandoned owner passes its ``restart`` reason on; any other intent marks, or upgrades
            # a ``restart`` mark, as a request. Collected first, so a request wins over a ``restart`` mark.
            marks: dict[TaskId[Any], tuple[AnyTaskRecord, str | None]] = {}

            def mark(record: AnyTaskRecord, reason: str | None) -> None:
                staged = marks.get(record["id"])
                if staged is not None and staged[1] is None:
                    return
                if record["abortRequested"] and ("abortReason" not in record or reason is not None):
                    return
                marks[record["id"]] = (record, reason)

            # Every live foreground task below a live owner with cancellation intent, which is its nearest one.
            for id in list(self._intent):
                owner = self._live[id]
                # Only an owner abandoned after a restart, and nothing else, passes its reason on.
                restart = owner.get("abortReason") == "restart" and not _failed_outcome(owner)
                for _below, node, live in self._walk_down(id, enter=lambda below: below not in self._intent):
                    if live is not None and not node["background"]:
                        mark(live, "restart" if restart else None)
            for id in checks:
                waiter = self._live.get(id)
                if waiter is None or waiter["state"]["status"] != "waiting":
                    continue
                if not await self._any_failed(waiter["state"]["on"]):
                    continue
                # Every other live task: the failed one keeps its own outcome.
                for member in waiter["state"]["on"]:
                    record = self._live.get(member)
                    if record is not None and not _failed_outcome(record):
                        mark(record, None)
            for record, reason in marks.values():
                tx.set_task(_with_abort_mark(record, reason))
            for id in queued:
                if self._below_cancelled(("conversation", id)):
                    await self._withdraw_inputs(tx, id)
            await self._finalize(tx)

        try:
            await self._session.commit_with(change, self._context)
        except Exception as error:
            # No extension code runs in this commit: a raise is a storage failure, a host callback, or a bug.
            self._fail_session(error)
        if work:
            self._reconciling_work -= 1
        self._schedule_wake()
        self._settle_idle()

    def _reconcile_work(self) -> bool:
        """Whether a reconcile pass would have work: a cascade to derive, held outcomes, or ``failFast`` waits."""
        return self._cascade_pending or len(self._finalize_checks) > 0 or len(self._fail_fast_checks) > 0

    async def _any_failed(self, ids: Sequence[TaskId[Any]]) -> bool:
        """Whether any of ``ids`` holds or ended with an outcome other than ``completed``."""
        for id in ids:
            record = self._live.get(id)
            if record is None:
                record = await self._storage.task(id, self._context)
            if record is not None and _failed_outcome(record):
                return True
        return False

    async def _finalize(self, tx: Any) -> None:
        """Write the terminal record of every ``completing`` task without live ordinary owned work, in rounds."""
        round_ids = list(self._finalize_checks)
        self._finalize_checks.clear()
        done: set[TaskId[Any]] = set()
        overlay = _overlay_of(tx)
        while len(round_ids) > 0:
            following: dict[TaskId[Any], None] = {}
            for id in round_ids:
                record = overlay.tasks.get(id) or self._live.get(id)
                if record is None or record["state"]["status"] != "completing" or id in done:
                    continue
                if self._has_owned_live(id, overlay, done):
                    continue
                outcome = record["state"]["outcome"]
                terminal = _with_state(record, {"status": "terminal", "outcome": outcome})
                tx.set_task(terminal)
                overlay.tasks[id] = terminal
                done.add(id)
                # REMINDER: only the scheduler writes ``faulted`` and ``orphaned`` (spec §5.4).
                if outcome["status"] in ("faulted", "orphaned"):
                    await self._settle_outcome(tx, record, cast(Any, outcome))
                above = self._nearest_live_above(record, overlay)
                if above is not None and above["state"]["status"] == "completing":
                    following[above["id"]] = None
            round_ids = list(following)

    async def _load_scopes(self, queued: bool) -> list[ConversationId]:
        """Load the owner chains of the tasks in ``_unloaded`` and, with ``queued``, of every conversation with
        queued inputs; returns the latter. Reads committed Storage directly."""
        for id in list(self._unloaded):
            # An earlier load in this loop may have passed it.
            if id in self._unloaded:
                await self._load_chain(("task", id))
        if not queued:
            return []
        submissions = await scan_all(
            lambda cursor: self._storage.scan_submissions({"status": "queued"}, _SCAN_PAGE_SIZE, cursor, self._context)
        )
        conversations = list(
            dict.fromkeys(submission["conversationId"] for submission in submissions if submission["type"] == "input")
        )
        for id in conversations:
            await self._load_chain(("conversation", id))
        return conversations

    async def _load_chain(self, start: _Up, overlay: _Overlay | None = None) -> None:
        """Load the owner edges and task nodes from ``start`` up to its ownerless root."""
        passed: list[TaskId[Any]] = []
        at: _Up | None = start
        while at is not None:
            if at[0] == "task":
                id: TaskId[Any] = at[1]  # type: ignore[assignment]
                node: Any = self._live.get(id) or self._settled.get(id)
                # Created in this commit: the load from its own record covers what is above it.
                if node is None and overlay is not None and id in overlay.tasks:
                    break
                if node is not None and id not in self._unloaded:
                    break
                if node is None:
                    record = await self._storage.task(id, self._context)
                    if record is None:
                        return
                    node = _node_of(record)
                    if record["state"]["status"] == "terminal":
                        self._settle(id, node)
                    # A loaded node can reveal a cancelling owner: the next reconcile derives marks again.
                    self._schedule_reconcile()
                passed.append(id)
                at = _parent_of(node)
            else:
                conversation: ConversationId = at[1]  # type: ignore[assignment]
                edge = self._edge(conversation, overlay)
                if edge is _UNKNOWN:
                    record = await self._storage.conversation(conversation, self._context)
                    owner = None if record is None else record.get("owner")
                    edge = None if owner is None else owner["taskId"]
                    self._set_edge(conversation, edge)
                    # A loaded node can reveal a cancelling owner: the next reconcile derives marks again.
                    self._schedule_reconcile()
                at = None if edge is None else ("task", edge)  # type: ignore[assignment]
        for id in passed:
            self._unloaded.pop(id, None)

    def _set_edge(self, conversation_id: ConversationId, owner: TaskId[Any] | None) -> None:
        self._edges[conversation_id] = owner
        # The conversation now hangs from its owner, or is a root; with nothing below it, it waits for the sweep.
        self._relist(_conversation_node(conversation_id))
        self._drop_if_unneeded(_conversation_node(conversation_id))

    def _settle(self, id: TaskId[Any], node: _TaskNode) -> None:
        """Know a terminal task a chain load read, so walks pass through it; its own chain is not loaded yet."""
        self._settled[id] = node
        self._unloaded[id] = None
        self._relist(id)
        self._drop_if_unneeded(id)

    def _drop_if_unneeded(self, node: _Node) -> None:
        """Queue an ended task or a conversation for the sweep when nothing is listed below it."""
        if not isinstance(node, str) and node in self._live:
            return
        if node in self._below:
            return
        self._drop_queue[node] = None
        self._schedule_sweep()

    def _schedule_sweep(self) -> None:
        """Sweep in a job of its own on the Session line, after the operation underway."""
        if self._sweep_scheduled or self._closing:
            return
        self._sweep_scheduled = True

        async def job() -> None:
            self._sweep_scheduled = False
            self._sweep()

        try:
            # Rejected only once the Session is closing or failed, when nothing is kept anyway.
            self._session.read_on_line(job).add_done_callback(consume)
        except Exception:
            pass

    def _sweep(self) -> None:
        """Drop the queued nodes that are still unneeded: not live, nothing listed below."""
        for node in self._drop_queue:
            if node in self._below:
                continue
            if isinstance(node, str):
                self._edges.pop(_conversation_of(node), None)
                self._roots.pop(_conversation_of(node), None)
            elif node not in self._live:
                self._settled.pop(node, None)  # type: ignore[call-overload]
                self._unloaded.pop(node, None)  # type: ignore[call-overload]
        self._drop_queue.clear()

    def _track(self, previous: AnyTaskRecord | None, record: AnyTaskRecord) -> None:
        """Bring the scheduler's sets in line with a task's committed record."""
        id = record["id"]
        if previous is not None and previous["state"]["status"] == "waiting":
            for on in previous["state"]["on"]:
                _delete_from(self._waiters, on, id)
        state = record["state"]
        if state["status"] == "terminal":
            node = _node_of(record)
            self._live.pop(id, None)
            self._sleeping.pop(id, None)
            self._intent.pop(id, None)
            self._runnable.pop(id, None)
            self._finalize_checks.pop(id, None)
            self._settled[id] = node
            # Before anything is forgotten: the walk up to the task it held passes the ended tasks between them.
            self._release(node)
            for waiter in self._waiters.get(id, ()):
                self._runnable[waiter] = None
            self._relist(id)
            self._drop_if_unneeded(id)
            return
        self._live[id] = record
        if previous is None:
            if not self._known(_parent_node(record)):
                self._unloaded[id] = None
            self._relist(id)
        if _cancellation_intent(record):
            self._intent[id] = None
        else:
            self._intent.pop(id, None)
        if state["status"] == "completing":
            self._runnable.pop(id, None)
            self._finalize_checks[id] = None
        else:
            self._runnable[id] = None
        if state["status"] == "waiting":
            for on in state["on"]:
                _add_to(self._waiters, on, id)
        if state["status"] == "sleeping":
            self._sleeping[id] = None
        else:
            self._sleeping.pop(id, None)

    def _release(self, node: Any, overlay: _Overlay | None = None) -> AnyTaskRecord | None:
        """The nearest live task above ended ordinary work may be free now: one holding its outcome, or aborting."""
        above = self._nearest_live_above(node, overlay)
        if above is not None and above["state"]["status"] == "completing":
            self._finalize_checks[above["id"]] = None
        elif above is not None and above["abortRequested"] is True:
            self._runnable[above["id"]] = None
        return above

    def _relist(self, start: _Node) -> None:
        """List ``start`` under its parent node in ``_below`` while it is a live task or has nodes listed below it,
        and unlist it otherwise; then its parent, as far up as that changes."""
        node: _Node = start
        while True:
            needed = (not isinstance(node, str) and node in self._live) or node in self._below
            parent: Any
            if isinstance(node, str):
                parent = self._edges.get(_conversation_of(node), _UNKNOWN)
            else:
                known = self._live.get(node) or self._settled.get(node)
                if known is None:
                    return
                parent = _parent_node(known)
            if parent is _UNKNOWN:
                return
            if parent is None:
                root = _conversation_of(node)  # type: ignore[arg-type]
                if needed:
                    self._roots[root] = None
                else:
                    self._roots.pop(root, None)
                return
            listed = node in self._below.get(parent, ())
            if listed == needed:
                return
            if needed:
                first = parent not in self._below
                _add_to(self._below, parent, node)
                if not first:
                    return
            else:
                _delete_from(self._below, parent, node)
                if parent in self._below:
                    return
                self._drop_if_unneeded(parent)
            node = parent

    def _nearest_live_above(self, node: Any, overlay: _Overlay | None = None) -> AnyTaskRecord | None:
        """The nearest live task above ended ordinary work ``node``, in the overlay's view."""
        current = node
        while not current["background"]:
            owner = current.get("owner")
            if owner is None:
                owner = self._edge(current["conversationId"], overlay)
            if owner is None or owner is _UNKNOWN:
                return None
            record = (overlay.tasks.get(owner) if overlay is not None else None) or self._live.get(owner)
            if record is not None and record["state"]["status"] != "terminal":
                return record
            above = record if record is not None else self._settled.get(owner)
            if above is None:
                return None
            current = above
        return None

    def _known(self, node: _Node) -> bool:
        """Whether the chain above ``node`` is loaded, without walking."""
        if not isinstance(node, str):
            return (node in self._live or node in self._settled) and node not in self._unloaded
        owner = self._edges.get(_conversation_of(node), _UNKNOWN)
        return owner is None or (owner is not _UNKNOWN and self._known(owner))

    def _edge(self, id: ConversationId, overlay: _Overlay | None = None) -> Any:
        """Owner task of a conversation, ``None`` when ownerless, ``_UNKNOWN`` while not loaded."""
        if overlay is not None and id in overlay.edges:
            return overlay.edges[id]
        return self._edges.get(id, _UNKNOWN)

    def _node(self, id: TaskId[Any], overlay: _Overlay | None = None) -> Any:
        if overlay is not None and id in overlay.tasks:
            return overlay.tasks[id]
        return self._live.get(id) or self._settled.get(id)

    def _above(self, start: _Up, overlay: _Overlay | None = None) -> Iterator[tuple[str, Any, Any]]:
        """Walk up from ``start``: ``("task", id, node)``, ``("conversation", id, None)``, or ``("unknown", ...)``."""
        at: _Up | None = start
        while at is not None:
            if at[0] == "task":
                node = self._node(at[1], overlay)  # type: ignore[arg-type]
                if node is None:
                    yield ("unknown", None, None)
                    return
                yield ("task", at[1], node)
                at = _parent_of(node)
            else:
                yield ("conversation", at[1], None)
                edge = self._edge(at[1], overlay)  # type: ignore[arg-type]
                if edge is _UNKNOWN:
                    yield ("unknown", None, None)
                    return
                at = None if edge is None else ("task", edge)

    def _walk_down(
        self,
        start: _Node,
        *,
        overlay: _Overlay | None = None,
        cross_background: bool = False,
        enter: Callable[[TaskId[Any]], bool] | None = None,
    ) -> Iterator[tuple[TaskId[Any], Any, AnyTaskRecord | None]]:
        """The tasks below ``start`` that a walk down reaches: ``(id, node, live record or None)``."""
        links = None if overlay is None else self._links(overlay)
        pending: list[_Node] = [start]
        while pending:
            at = pending.pop()
            for children in (self._below.get(at), None if links is None else links.get(at)):
                for child in list(children or ()):
                    if isinstance(child, str):
                        pending.append(child)
                        continue
                    record = (overlay.tasks.get(child) if overlay is not None else None) or self._live.get(child)
                    node = record if record is not None else self._settled.get(child)
                    if node is None:
                        continue
                    live = record if record is not None and record["state"]["status"] != "terminal" else None
                    node = cast(Any, node)
                    yield child, node, live
                    if (cross_background or not node["background"]) and (enter is None or enter(child)):
                        pending.append(child)

    def _links(self, overlay: _Overlay) -> dict[_Node, dict[_Node, None]]:
        """The links walks down need for the commit's new tasks."""
        if overlay.links is not None:
            return overlay.links
        links: dict[_Node, dict[_Node, None]] = {}
        for record in overlay.tasks.values():
            if record["id"] in self._live or record["id"] in self._settled:
                continue
            child: _Node = record["id"]
            parent: Any = _parent_node(record)
            while parent is not None and parent is not _UNKNOWN and child not in links.get(parent, ()):
                _add_to(links, parent, child)
                if not isinstance(parent, str) and (parent in self._live or parent in overlay.tasks):
                    break
                above: Any
                if isinstance(parent, str):
                    above = self._edge(_conversation_of(parent), overlay)
                else:
                    fields = self._settled.get(parent)
                    above = _UNKNOWN if fields is None else _parent_node(fields)
                if above is not None and above is not _UNKNOWN and parent in self._below.get(above, ()):
                    break
                child = parent
                parent = above
        overlay.links = links
        return links

    def _owned_live(
        self,
        owner: TaskId[Any],
        *,
        overlay: _Overlay | None = None,
        first: bool = False,
        done: set[TaskId[Any]] | None = None,
    ) -> list[TaskId[Any]]:
        """The live ordinary owned work of ``owner`` (spec §5.5); with ``first``, only the first."""
        found: list[TaskId[Any]] = []
        enter = None if done is None else (lambda id: id not in done)
        for id, node, live in self._walk_down(owner, overlay=overlay, enter=enter):
            if live is None or node["background"]:
                continue
            found.append(id)
            if first:
                break
        return found

    def _has_owned_live(
        self, owner: TaskId[Any], overlay: _Overlay | None = None, done: set[TaskId[Any]] | None = None
    ) -> bool:
        return len(self._owned_live(owner, overlay=overlay, first=True, done=done)) > 0

    def _in_scope(self, start: _Up, scope: tuple[str, Any], cross_background: bool = False) -> bool | None:
        """Whether ordinary traversal from ``scope`` reaches ``start``; ``None`` while an edge is not loaded."""
        for kind, id, node in self._above(start):
            if kind == "unknown":
                return None
            if kind == "conversation":
                if scope[0] == "conversation" and id == scope[1]:
                    return True
            elif node["background"] and not cross_background:
                return False
        return scope[0] == "roots"

    def _below_cancelled(self, start: _Up) -> bool:
        """Whether a live owner's cancellation intent reaches ``start``."""
        return self._cancelling_owner(start) is not None

    def _cancelling_owner(self, start: _Up) -> AnyTaskRecord | None:
        """The nearest live owner above ``start`` whose cancellation intent reaches it, if any."""
        # Nothing cancels, the common case: no walk.
        if len(self._intent) == 0:
            return None
        for kind, id, node in self._above(start):
            if kind == "unknown":
                return None
            if kind != "task":
                continue
            live = self._live.get(id)
            if live is not None and _cancellation_intent(live):
                return live
            if node["background"]:
                return None
        return None

    def _fail_session(self, error: BaseException) -> None:
        """Fail the Session, which reports it, for a raise in one of the scheduler's own commits."""
        if not self._closing:
            self._session.fail(error)

    def _seal(self) -> None:
        """Close listener: runs synchronously once admission is sealed, before ``join()``, or when the Session fails."""
        self._closing = True
        try:
            self._unsubscribe_registry()
        except Exception as error:
            self._report(error, _REGISTRY)
        error = closed_error(self._session)
        self._task_waiters.reject_all(error)
        self._idle_waiters.reject_all(error)
        self._contexts.clear()
        if self._expiry is not None:
            self._expiry[1]()
        self._expiry = None
        if self._timer is not None:
            self._timer[1]()
        self._timer = None

    def _schedule_wake(self) -> None:
        """Recompute the wake soon, off the Session line: arm the built-in timer, or tell ``onWake`` of a change."""
        if self._wake_scheduled or self._closing:
            return
        self._wake_scheduled = True

        def run() -> None:
            self._wake_scheduled = False
            if self._closing:
                return
            wake = self.wake()
            if self._on_wake is None:
                self._arm_timer(wake.get("at") if self._enabled else None)
                return
            last = self._last_wake
            if last is not None and last["active"] == wake["active"] and last.get("at") == wake.get("at"):
                return
            try:
                self._on_wake(wake)
                # Only a delivered wake counts: after a raise, the next change or kick offers it again.
                self._last_wake = wake
            except Exception as error:
                self._report(error, _ON_WAKE)

        self._soon(run)

    def _arm_timer(self, at: float | None) -> None:
        """Keep the timer for the earliest sleep; it rechecks the clock when it fires."""
        if self._timer is not None and self._timer[0] == at:
            return
        if self._timer is not None:
            self._timer[1]()
        self._timer = None
        if at is None or self._closing:
            return

        def fire() -> None:
            self._timer = None
            self._wake_due()
            self._kick()

        disarm = self._timers.foreground(fire, at - self._now())
        self._timer = (at, disarm)

    def _wake_due(self) -> None:
        """Make every sleeping task whose time has come a reservation candidate."""
        for id in self._sleeping:
            record = self._live.get(id)
            if record is not None and not self._not_due(record):
                self._runnable[id] = None

    def _parked(self, record: Any) -> bool:
        """A task cut off ``restartLimit`` times in a row; an abort mark still runs its abort handler."""
        restarts = record.get("restarts")
        return not record["abortRequested"] and restarts is not None and restarts >= self._restart_limit()

    def _about_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """What a report about ``task``'s own code says of it: the task, and the extension that installed it."""
        if "extension" in task:
            extension = task["extension"]
        else:
            extension = None
            try:
                snapshot = self._registry.snapshot()
                extension = _installer_of(snapshot, snapshot.task(task["kind"]))
            except Exception:
                # A registry that raises is reported where it is read for scheduling.
                pass
        return {"task": _report_task(task), **optional_extension(extension)}

    def _invocation_about(self, invocation: _Invocation) -> dict[str, Any]:
        return self._about_task(
            {
                "taskId": invocation.task_id,
                "kind": invocation.kind,
                "conversationId": invocation.conversation_id,
                "extension": invocation.extension,
            }
        )

    def _unpark_raised(self) -> None:
        """Make parked tasks that a raised ``settings.restartLimit`` no longer parks candidates again."""
        unparked = False
        for id in list(self._parked_tasks):
            record = self._live.get(id)
            if record is not None and self._parked(record):
                continue
            del self._parked_tasks[id]
            if record is None:
                continue
            self._runnable[id] = None
            unparked = True
        if unparked:
            self._soon(self._notify_blocked)

    def _restart_limit(self) -> float:
        """``settings.restartLimit``, or the default when the host's settings raise."""
        try:
            return self._settings()["restartLimit"]
        except Exception as error:
            self._report(error, _SETTINGS)
            return DEFAULT_RESTART_LIMIT

    def _notify_blocked(self) -> None:
        for listener in list(self._blocked_listeners):
            try:
                listener()
            except Exception as error:
                self._report(error, {"type": "listener_error"})

    def _not_due(self, record: AnyTaskRecord) -> bool:
        """A sleep whose ``until`` the clock has not reached; an abort mark ends a sleep."""
        state = record["state"]
        return state["status"] == "sleeping" and not record["abortRequested"] and state["until"] > self._now()

    def _kick(self) -> None:
        self._schedule_wake()
        self._dirty = True
        if self._draining or not self._enabled or self._closing:
            return
        self._draining = True
        # Never commit synchronously from a commit or registry listener: the task's first step runs on the next
        # loop iteration, as upstream's microtask does.
        self._spawn(self._drain())

    async def _drain(self) -> None:
        try:
            while self._dirty and self._enabled and not self._closing:
                self._dirty = False
                for reservation in await self._reserve():
                    self._start(reservation)
        except Exception as error:
            # As in ``_reconcile()``: the reservation commit runs no extension code.
            self._fail_session(error)
        finally:
            self._draining = False
            # A wakeup that arrived after the loop's last check still needs its pass.
            if self._dirty:
                self._kick()

    async def _reserve(self) -> list[_Reservation]:
        """Reserve every eligible task in one commit; orphan abort-marked tasks no definition can take."""
        reservations: list[_Reservation] = []
        faulted: list[tuple[AnyTaskRecord, str]] = []
        abandoned: list[TaskId[Any]] = []

        async def change(tx: Any) -> None:
            if not self._enabled or self._closing:
                return
            for id in self._abandoned:
                record = self._live.get(id)
                if record is not None and not record["abortRequested"]:
                    tx.set_task(_with_abort_mark(record, "restart"))
            if len(self._abandoned) > 0:
                abandoned.extend(self._abandoned)
                return
            await self._load_scopes(False)
            # Every pass picks up sleeps whose time has come.
            self._wake_due()
            self._unpark_raised()
            # Taken once per pass, and only when some task is a candidate.
            snapshot: RegistrySnapshot | None = None
            for id in list(self._runnable):
                record = self._live.get(id)
                if record is None or record["state"]["status"] == "completing" or id in self._invocations:
                    self._runnable.pop(id, None)
                    continue
                # It returns once what it waits for ends, its record changes, or the work below it ends.
                if len(self._waiting_on(record, True)) > 0:
                    self._runnable.pop(id, None)
                    continue
                # It returns at the first pass after its time comes.
                if self._not_due(record):
                    self._runnable.pop(id, None)
                    continue
                # It returns when ``retry()`` clears its restart count, or after ``settings.restartLimit`` rose.
                if self._parked(record):
                    self._runnable.pop(id, None)
                    # Newly parked without a commit, as after a lowered limit: the graph must hear it.
                    if id not in self._parked_tasks:
                        self._soon(self._notify_blocked)
                    self._parked_tasks[id] = None
                    continue
                mode: Literal["run", "abort"] = "abort" if record["abortRequested"] else "run"
                # Work below an owner with cancellation intent waits for its cascade mark instead of running a phase.
                if mode == "run" and not record["background"] and self._below_cancelled(_parent_of(record)):
                    self._cascade_pending = True
                    self._schedule_reconcile()
                    continue
                # An abort handler cut off ``restartLimit`` times in a row would run forever.
                if mode == "abort" and record.get("restarts", 0) >= self._restart_limit():
                    message = (
                        f"Abort handler of task {id} ({record['kind']}) was cut off {record.get('restarts')} times"
                    )
                    await self._terminate(
                        tx,
                        record,
                        {"status": "faulted", "error": {"message": message, "detail": {"reason": "restart_limit"}}},
                    )
                    faulted.append((record, message))
                    continue
                if snapshot is None:
                    snapshot = self._registry.snapshot()
                resolution = self._resolve(record, snapshot)
                if isinstance(resolution, str):
                    # An abandoned task waits for its definition, so its abort handler can clean up.
                    if mode == "abort" and "abortReason" not in record:
                        await self._terminate(tx, record, {"status": "orphaned", "reason": resolution})
                    continue
                task, resolved = resolution
                if resolved is not record or record["state"]["status"] != "running":
                    tx.set_task(
                        _with_state(
                            resolved, {"status": "running", "checkpoint": cast(Any, resolved["state"])["checkpoint"]}
                        )
                    )
                # Registered on the line, so marks and later reservations see it and close joins it.
                self._runnable.pop(id, None)
                invocation = self._create_invocation(record, mode, _installer_of(snapshot, task))
                reservations.append(_Reservation(invocation, task, snapshot))

        try:
            await self._session.commit_with(change, self._context)
        except BaseException:
            for reservation in reservations:
                self._invocations.pop(reservation.invocation.task_id, None)
                self._runnable[reservation.invocation.task_id] = None
                reservation.invocation.finish()
            raise
        for record, message in faulted:
            self._report(
                RuntimeError(message),
                {"type": "task_faulted", **self._about_task(_record_task(record)), "reason": "restart_limit"},  # type: ignore[typeddict-item]
            )
        # Marked, or already marked or gone: done with them. Reserve again, now seeing the marks.
        for id in abandoned:
            self._abandoned.pop(id, None)
        if len(abandoned) > 0:
            self._dirty = True
        return reservations

    def _waiting_on(self, record: AnyTaskRecord, first: bool = False) -> list[TaskId[Any]]:
        """Live tasks a task waits for before its next invocation: its live ordinary owned work when abort-marked,
        otherwise the live part of the ``on`` of a wait."""
        if record["abortRequested"]:
            return self._owned_live(record["id"], first=first)
        state = record["state"]
        if state["status"] != "waiting":
            return []
        live = [id for id in state["on"] if id in self._live]
        return live[:1] if first else live

    def _resolve(
        self, record: AnyTaskRecord, snapshot: RegistrySnapshot
    ) -> BlockedReason | tuple[AnyTask, AnyTaskRecord]:
        """Resolve the record's definition by kind, migrating an older stored version: ``(task, record)`` or why not."""
        fit = self._fit(record, snapshot.task(record["kind"]))
        if fit.reason is not None:
            return fit.reason
        assert fit.task is not None
        if not fit.migrates:
            return fit.task, record
        definition = fit.task.definition
        try:
            if definition.migrate is None:
                raise _missing_migration(record, definition)
            migrated = definition.migrate(record["input"], cast(Any, record["state"])["checkpoint"], record["version"])
            state = {**record["state"], "checkpoint": copy_json(migrated["checkpoint"])}
            migrated_record: Any = {
                **record,
                "version": definition.version,
                "input": copy_json(migrated["input"]),
                "state": state,
            }
            return fit.task, migrated_record
        except Exception as error:
            self._failed_migrations[record["id"]] = (fit.task, error)
            # Blocked now without a commit: the graph and the host's wake must hear it.
            self._soon(self._notify_blocked)
            self._schedule_wake()
            extension = _installer_of(snapshot, fit.task)
            self._report(
                error,
                {"type": "migration_failed", **self._about_task({**_record_task(record), "extension": extension})},  # type: ignore[typeddict-item]
            )
            return "migration_failed"

    def _fit(self, record: Any, task: AnyTask | None) -> _Fit:
        if task is None:
            return _Fit(reason="missing_task")
        version = task.definition.version
        if version == record["version"]:
            return _Fit(task=task, migrates=False)
        if version < record["version"]:
            return _Fit(reason="task_too_old")
        failed = self._failed_migrations.get(record["id"])
        if failed is not None and failed[0] is task:
            return _Fit(reason="migration_failed", error=failed[1])
        return _Fit(task=task, migrates=True)

    @property
    def index_sizes(self) -> _IndexSizes:
        """Sizes of the ownership indexes, for tests that bound the scheduler's memory."""
        return {
            "live": len(self._live),
            "settled": len(self._settled),
            "edges": len(self._edges),
            "below": sum(len(children) for children in self._below.values()),
            "roots": len(self._roots),
            "unloaded": len(self._unloaded),
            "dropQueue": len(self._drop_queue),
        }

    async def inspect(self, snapshot: RegistrySnapshot) -> tuple[str, list[TaskInspection]]:
        """Scheduling state and every live task with its derived state, read on the Session line."""
        await self._load_scopes(False)
        tasks: list[TaskInspection] = [
            {"record": record, "state": self._inspect_task(record, snapshot)} for record in self._live.values()
        ]
        scheduling = "closing" if self._closing else "running" if self._enabled else "paused"
        return scheduling, tasks

    def _inspect_task(self, record: AnyTaskRecord, snapshot: RegistrySnapshot) -> Any:
        if record["id"] in self._invocations:
            return {"kind": "running"}
        state: Any = record["state"]
        if state["status"] == "completing":
            return {"kind": "completing"}
        on = self._waiting_on(record)
        if len(on) > 0:
            return {"kind": "waiting", "on": on}
        if self._not_due(record):
            return {"kind": "sleeping", "until": state["until"]}
        if self._parked(record):
            return {"kind": "blocked", "reason": "restart_limit"}
        fit = self._fit(record, snapshot.task(record["kind"]))
        if fit.reason is not None:
            blocked: dict[str, Any] = {"kind": "blocked", "reason": fit.reason}
            if fit.error is not None:
                blocked["error"] = fit.error
            return blocked
        assert fit.task is not None
        if fit.migrates and fit.task.definition.migrate is None:
            return {
                "kind": "blocked",
                "reason": "migration_failed",
                "error": _missing_migration(record, fit.task.definition),
            }
        return {"kind": "ready", "migrates": fit.migrates}

    def _create_invocation(
        self, record: AnyTaskRecord, mode: Literal["run", "abort"], extension: str | None
    ) -> _Invocation:
        controller = AbortController()
        invocation = _Invocation(
            task_id=record["id"],
            kind=record["kind"],
            extension=extension,
            conversation_id=record["conversationId"],
            mode=mode,
            controller=controller,
            context=with_abort_signal(controller.signal, self._context),
            done=asyncio.get_running_loop().create_future(),
        )
        self._invocations[record["id"]] = invocation
        return invocation

    def _start(self, reservation: _Reservation) -> None:
        invocation = reservation.invocation

        async def run() -> None:
            try:
                if invocation.mode == "run":
                    await self._run(reservation)
                else:
                    await self._run_abort(reservation)
            except Exception as error:
                self._report(error, {"type": "task_error", **self._invocation_about(invocation)})  # type: ignore[typeddict-item]
            finally:
                self._end(invocation)
                invocation.finish()
                self._kick()

        self._spawn(run())

    async def _run(self, reservation: _Reservation) -> None:
        """Run phase handlers, each preceded by a step that decides on the line whether the invocation continues."""
        invocation = reservation.invocation
        state = _RunState(reservation.task, reservation.snapshot)
        phase = _Phase(snapshot=lambda: state.snapshot, task=lambda: state.task)
        runtime = _Runtime(self, invocation, phase)
        previous: _PhaseResult | None = None
        while True:
            current = await self._step(
                invocation,
                lambda tx, current, previous=previous: self._decide(tx, current, previous, state, invocation.extension),
            )
            # Close may seal between the decision and dispatch.
            if current is None or self._closing:
                return
            checkpoint = cast(Any, current["state"])["checkpoint"]
            # Each phase handler resolves its agent afresh, at first use.
            phase.agent = None
            try:
                handler = state.task.definition.phases[checkpoint["phase"]]
                if self._telemetry is None:
                    await handler(current, runtime, invocation.context)  # type: ignore[arg-type]
                else:
                    await self._traced_phase(
                        invocation,
                        current,
                        checkpoint,
                        phase,
                        lambda handler=handler, current=current: handler(current, runtime, invocation.context),  # type: ignore[misc]
                    )
                previous = _PhaseResult(checkpoint)
            except Exception as error:
                previous = _PhaseResult(checkpoint, error, True)

    def _decide(
        self, tx: Any, current: AnyTaskRecord, previous: _PhaseResult | None, state: _RunState, extension: str | None
    ) -> Any:
        """Precedence rules for a run invocation, on the line. Rules 1 and 2 are applied by ``_step``."""
        # 3. abort mark: end; a fresh abort invocation starts once the task's ordinary owned work is gone.
        if current["abortRequested"]:
            return False
        # An owner's cancellation intent ends it too, before its cascade marks it; reservation then holds it back.
        if not current["background"] and self._below_cancelled(_parent_of(current)):
            self._cascade_pending = True
            self._schedule_reconcile()
            return False
        if previous is None:
            return True
        # 4. uncaught error.
        if previous.failed:
            return _Fault(previous.failure)  # type: ignore[arg-type]
        # 6. no durable progress.
        if json_equal(cast(Any, current["state"])["checkpoint"], previous.checkpoint):
            message = f"Task {current['kind']} phase {previous.checkpoint['phase']} returned without durable progress"
            return _Fault(RuntimeError(message))
        # 5. progress: refresh the snapshot; hand over to a replacement definition that can take the task.
        state.snapshot = self._registry.snapshot()
        replacement = state.snapshot.task(current["kind"])
        if replacement is not state.task:
            if replacement is not None and _can_reserve(replacement, current):
                tx.set_task(
                    _with_state(current, {"status": "pending", "checkpoint": cast(Any, current["state"])["checkpoint"]})
                )
                return False
            if state.reported is None or state.reported[0] is not replacement:
                state.reported = (replacement,)
                self._report(
                    RuntimeError(f"Task {current['id']} keeps running under its old {current['kind']} definition"),
                    {
                        "type": "task_outdated",
                        **self._about_task({**_record_task(current), "extension": extension}),
                        "reason": "missing_task" if replacement is None else "incompatible_task",
                    },  # type: ignore[typeddict-item]
                )
        return True

    async def _run_abort(self, reservation: _Reservation) -> None:
        """Run the abort handler once; rules 1, 2, and 4 apply, and returning without an outcome faults."""
        invocation = reservation.invocation
        current = self._live.get(invocation.task_id)
        if current is None or self._closing:
            return
        failure: BaseException | None = None
        try:
            phase = _Phase(snapshot=lambda: reservation.snapshot, task=lambda: reservation.task)
            runtime = _Runtime(self, invocation, phase)
            abort = reservation.task.definition.abort
            if self._telemetry is None:
                await abort(current, runtime, invocation.context)  # type: ignore[arg-type]
            else:
                checkpoint = {**cast(Any, current["state"]).get("checkpoint", {}), "phase": "abort"}
                await self._traced_phase(
                    invocation,
                    current,
                    checkpoint,
                    phase,
                    lambda: abort(current, runtime, invocation.context),  # type: ignore[arg-type]
                )
        except Exception as error:
            failure = error
        message = f"Abort handler of task {invocation.task_id} returned without a terminal outcome"
        await self._step(
            invocation, lambda _tx, _current: _Fault(failure if failure is not None else RuntimeError(message))
        )

    async def _traced_phase(
        self,
        invocation: _Invocation,
        record: AnyTaskRecord,
        checkpoint: Any,
        phase: _Phase,
        call: Callable[[], Awaitable[Any]],
    ) -> None:
        """Run one handler under its telemetry span; tool tasks open their own (``pi.harness.tool``)."""
        telemetry = self._telemetry
        assert telemetry is not None
        kind = invocation.kind
        if kind == "pi.tool":
            phase.telemetry = self._tool_parent(record, telemetry)
            try:
                await call()
            finally:
                phase.telemetry = NOOP_TELEMETRY_CONTEXT
            return
        attributes: dict[str, Any] = {
            "pi.harness.task_id": invocation.task_id,
            "pi.harness.conversation_id": invocation.conversation_id,
            "pi.harness.phase": checkpoint.get("phase") if isinstance(checkpoint, dict) else None,
            "pi.harness.restarts": record.get("restarts"),
        }
        if kind == "pi.generation":
            name = GENERATION_SPAN
            attributes["pi.harness.attempt"] = checkpoint.get("attempt")
            model = checkpoint.get("model")
            if isinstance(model, dict):
                attributes["pi.harness.model.provider"] = model.get("provider")
                attributes["pi.harness.model.id"] = model.get("modelId")
        else:
            name = TASK_SPAN
            attributes["pi.harness.task.kind"] = kind
            attributes["pi.harness.task.extension"] = invocation.extension
        signal = invocation.controller.signal

        async def body(span: SchemaTelemetrySpan, _child: Any) -> None:
            phase.telemetry = span
            started = asyncio.get_running_loop().time()
            outcome = "returned"
            try:
                await call()
            except BaseException as error:
                outcome = "aborted" if signal.aborted else "raised"
                span.set_status({"status": "error", "error": {"name": type(error).__name__, "message": ""}})
                raise
            finally:
                phase.telemetry = NOOP_TELEMETRY_CONTEXT
                span.set_attributes(
                    {
                        "pi.harness.phase_outcome": outcome,
                        "pi.harness.duration_ms": round((asyncio.get_running_loop().time() - started) * 1000, 3),
                    }
                )

        await bind_typed_span_starter(telemetry, TELEMETRY_STARTER)(name, attributes, body)

    def _tool_parent(self, record: AnyTaskRecord, root: TelemetryContext) -> TelemetryContext:
        """A nested call's span parents to its caller's span while that is open here; anything else to the root."""
        task_input = record.get("input")
        if isinstance(task_input, dict) and task_input.get("kind") == "nested":
            caller = self._tool_spans.get(task_input.get("parent"))  # type: ignore[arg-type]
            if caller is not None:
                return caller
        return root

    async def _step(self, invocation: _Invocation, decide: Callable[[Any, AnyTaskRecord], Any]) -> AnyTaskRecord | None:
        """One synchronous decision on the Session line."""
        fault: list[BaseException] = []

        async def change(tx: Any) -> AnyTaskRecord | None:
            found = self._live.get(invocation.task_id)
            current = found if found is not None and found["state"]["status"] == "running" else None
            decision = decide(tx, current) if current is not None and not self._closing else False
            if decision is True:
                return current
            self._end(invocation)
            if decision is not False:
                fault.append(decision.error)
                message = error_message(decision.error)
                await self._terminate(tx, current, {"status": "faulted", "error": {"message": message}})  # type: ignore[arg-type]
            return None

        try:
            continued = await self._session.commit_with(change, self._context)
        except Exception as error:
            # The step writes the scheduler's own decision; a raise there would otherwise leave the task running.
            self._end(invocation)
            self._fail_session(error)
            return None
        # Reported once the fault is durable: a host may count it, for example toward a breaker.
        if fault:
            self._report(fault[0], {"type": "task_faulted", **self._invocation_about(invocation), "reason": "threw"})  # type: ignore[typeddict-item]
        return continued

    async def _terminate(self, tx: Any, record: AnyTaskRecord, outcome: SchedulerOutcome) -> None:
        """Write an outcome the scheduler decided: held as ``completing`` while ordinary owned work is live."""
        await self._load_scopes(False)
        if self._has_owned_live(record["id"], _overlay_of(tx)):
            tx.set_task(_with_state(record, {"status": "completing", "outcome": outcome}))
            return
        tx.set_task(_with_state(record, {"status": "terminal", "outcome": outcome}))
        await self._settle_outcome(tx, record, outcome)

    async def _commit_state(self, tx: Any, invocation: _Invocation, current: AnyTaskRecord, following: Any) -> None:
        """Replace a running task's state with what it committed."""
        status = following["status"]
        if status == "waiting":
            await self._validate_wait(tx, invocation, current, following["on"], following["policy"])
        if status == "sleeping":
            _validate_sleep(invocation, current["id"], following["until"])
        if status == "terminal":
            overlay = _overlay_of(tx)
            await self._load_scopes(False)
            for record in list(overlay.tasks.values()):
                await self._load_chain(_parent_of(record), overlay)
            if self._has_owned_live(current["id"], overlay):
                tx.set_task(
                    _with_state(_progressed(current), {"status": "completing", "outcome": following["outcome"]})
                )
                return
        # A state the task committed that differs from its current one is progress: its restart count starts over.
        same = json_equal(current["state"], following)
        tx.set_task(_with_state(current if same else _progressed(current), following))

    async def _validate_wait(
        self, tx: Any, invocation: _Invocation, current: AnyTaskRecord, on: Sequence[TaskId[Any]], policy: str
    ) -> None:
        """A wait names existing tasks, and ``failFast`` only tasks the waiter owns. It may not close a cycle."""
        if invocation.mode == "abort":
            raise RuntimeError(f"Abort handler of task {current['id']} cannot wait")
        overlay = _overlay_of(tx)
        targets: list[TaskId[Any]] = []
        for id in on:
            member = overlay.tasks.get(id) or self._live.get(id)
            if member is None:
                member = await self._storage.task(id, self._context)
            if member is None:
                raise RuntimeError(f"Task {id} does not exist")
            if policy == "failFast" and member.get("owner") != current["id"]:
                raise RuntimeError(f"Task {current['id']} can wait failFast only on tasks it owns; {id} is not one")
            # A live child already holds the waiter: waiting on it adds no edge. A terminal task has no edges.
            if member["state"]["status"] != "terminal" and member.get("owner") != current["id"]:
                targets.append(id)
        if len(targets) == 0:
            return
        cycle = await self._wait_cycle(current["id"], targets, overlay)
        if cycle is not None:
            path = " -> ".join(str(step) for step in [current["id"], *cycle])
            raise RuntimeError(f"Task {current['id']} cannot wait on {cycle[0]}: {path} is a wait cycle")

    async def _wait_cycle(
        self, task: TaskId[Any], on: Sequence[TaskId[Any]], overlay: _Overlay
    ) -> list[TaskId[Any]] | None:
        """The path from a member of ``on`` back to ``task`` in the wait-for graph, if the wait would close a cycle."""
        # Owned work is found by walks down, which need the chains of every live task loaded.
        await self._load_scopes(False)
        origin: dict[TaskId[Any], TaskId[Any] | None] = {}
        walked: set[TaskId[Any]] = set()
        pending: list[TaskId[Any]] = []

        def reach(id: TaskId[Any], via: TaskId[Any] | None) -> None:
            if id in origin:
                return
            origin[id] = via
            pending.append(id)

        def path(end: TaskId[Any]) -> list[TaskId[Any]]:
            steps: list[TaskId[Any]] = []
            at: TaskId[Any] | None = end
            while at is not None:
                steps.insert(0, at)
                at = origin.get(at)
            return steps

        for id in on:
            reach(id, None)
        while pending:
            id = pending.pop()
            if id == task:
                return path(id)
            record = overlay.tasks.get(id) or self._live.get(id)
            if record is None or record["state"]["status"] == "terminal":
                continue
            # An abort-marked task waits only for its owned work, as abort runs bottom-up.
            if record["state"]["status"] == "waiting" and not record["abortRequested"]:
                for member in record["state"]["on"]:
                    reach(member, id)
            if id in walked:
                continue
            walked.add(id)
            # Below a walked task the walk does not go again: its own walk covered what lies below.
            found: list[TaskId[Any]] = []
            for below, node, live in self._walk_down(id, overlay=overlay, enter=lambda below: below not in walked):
                if live is None or node["background"]:
                    continue
                found.append(below)
                reach(below, id)
            walked.update(found)
        return None

    def _end(self, invocation: _Invocation) -> None:
        """End an invocation: its runtime operations raise from now on, its signal aborts, its watches stop."""
        if invocation.ended:
            return
        invocation.ended = True
        if self._invocations.get(invocation.task_id) is invocation:
            del self._invocations[invocation.task_id]
            if invocation.task_id in self._live:
                self._runnable[invocation.task_id] = None
        for watch in list(invocation.watches):
            watch.stop()
        for disarm in list(invocation.timers):
            disarm()
        invocation.timers.clear()
        # Pending waits bound to the invocation, such as a tool's wait_for_task(), raise with it.
        invocation.controller.abort(_ended_error(invocation))

    def _idle(self, conversation_id: ConversationId | None) -> bool:
        """No live non-background task in the scope; a task whose owner edges are not loaded counts as inside."""
        scope: tuple[str, Any] = ("roots", None) if conversation_id is None else ("conversation", conversation_id)
        for id in self._unloaded:
            record = self._live.get(id)
            if (
                record is not None
                and not record["background"]
                and self._in_scope(_parent_of(record), scope) is not False
            ):
                return False
        starts = list(self._roots) if conversation_id is None else [conversation_id]
        for start in starts:
            for _id, node, live in self._walk_down(_conversation_node(start)):
                if live is not None and not node["background"]:
                    return False
        return True

    # ─── Invocation runtime support ────────────────────────────────────────

    def _gated(
        self, invocation: _Invocation, change: Callable[[Any, AnyTaskRecord], Any], context: Context
    ) -> asyncio.Future[Any]:
        """Commit after rereading the task on the line and gating the invocation."""
        if invocation.ended:
            return rejected(_ended_error(invocation))

        async def gated(tx: Any) -> Any:
            if invocation.ended:
                raise _ended_error(invocation)
            if self._closing:
                raise closed_error(self._session)
            found = self._live.get(invocation.task_id)
            if found is None:
                raise RuntimeError(f"Task {invocation.task_id} is terminal")
            if found["state"]["status"] != "running":
                raise RuntimeError(f"Task {invocation.task_id} is {found['state']['status']}")
            if invocation.mode == "run" and found["abortRequested"]:
                raise RuntimeError(f"Task {invocation.task_id} has a durable abort mark")
            result = change(tx, found)
            return await result if inspect.isawaitable(result) else result

        return self._session.commit_with(
            gated, context, {"conversationId": invocation.conversation_id, "taskId": invocation.task_id}
        )


_UNKNOWN: Final[Any] = type("_Unknown", (), {"__repr__": lambda self: "UNKNOWN", "__slots__": ()})()


class _Hooks:
    """``runtime.hooks``: dispatches one hook of the task to every matching handler, in extension order."""

    __slots__ = ("_agent", "_invocation", "_phase", "_scheduler")

    def __init__(
        self, scheduler: TaskScheduler, invocation: _Invocation, phase: _Phase, agent: Callable[[], Any]
    ) -> None:
        self._scheduler = scheduler
        self._invocation = invocation
        self._phase = phase
        self._agent = agent

    async def each(self, name: str, invoke: Callable[[Any], Awaitable[None] | None]) -> None:
        agent = await self._agent()
        for extension, handlers in agent_hooks(agent, self._phase.task().definition.name):
            handler = handlers.get(name) if isinstance(handlers, dict) else getattr(handlers, name, None)
            if not callable(handler):
                continue
            try:
                result = invoke(handler)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                if self._invocation.controller.signal.aborted:
                    raise
                self._scheduler._report(
                    error,
                    {"type": "extension_error", "extension": extension, "task": _report_task_of(self._invocation)},
                )


class _Runtime:
    """Operations of one task invocation (``TaskRuntime``). Every operation raises after the invocation ends."""

    def __init__(self, scheduler: TaskScheduler, invocation: _Invocation, phase: _Phase) -> None:
        self._scheduler = scheduler
        self._invocation = invocation
        self._phase = phase
        self.task_id = invocation.task_id
        self.conversation_id = invocation.conversation_id
        self.signal = invocation.controller.signal
        self.hooks = _Hooks(scheduler, invocation, phase, self._shared_agent)

    def __repr__(self) -> str:
        return f"TaskRuntime(task={self.task_id})"

    def _ended(self) -> BaseException:
        return _ended_error(self._invocation)

    def _shared_agent(self) -> asyncio.Future[Agent]:
        """Resolved at most once per phase handler, at first use, with the invocation's context."""
        if self._invocation.ended:
            return rejected(self._ended())
        phase = self._phase
        if phase.agent is None:
            phase.agent = asyncio.ensure_future(
                self._scheduler._agent(self._invocation.conversation_id, phase.snapshot(), self._invocation.context)
            )
            # A caller that stops waiting must not leave the shared resolution's failure unobserved.
            phase.agent.add_done_callback(consume)
        return phase.agent

    @property
    def models(self) -> Any:
        return self._scheduler._models()

    @property
    def telemetry_context(self) -> Any:
        """The running phase's telemetry parent: its span, or ``NOOP_TELEMETRY_CONTEXT``."""
        return self._phase.telemetry

    @property
    def approvals(self) -> Any:
        """Internal (``pi.tool``): the Harness's approval services."""
        return self._scheduler._approvals

    def bind_tool_span(self, span: Any) -> Callable[[], None]:
        """Internal (``pi.tool``): make ``span`` the parent of this call's nested calls until the returned
        function runs."""
        spans = self._scheduler._tool_spans
        task_id = self.task_id
        spans[task_id] = span

        def release() -> None:
            if spans.get(task_id) is span:
                del spans[task_id]

        return release

    @property
    def settings(self) -> Any:
        return self._scheduler._settings()

    @property
    def registry(self) -> RegistrySnapshot:
        return self._phase.snapshot()

    async def agent(self, context: Context) -> Agent:
        if self._invocation.ended:
            raise self._ended()
        return await await_with_context(asyncio.shield(self._shared_agent()), context)

    async def env(self, context: Context) -> Any:
        if self._invocation.ended:
            raise self._ended()
        return await self._scheduler._env(self._invocation.conversation_id, context)

    def commit(self, change: Callable[[Any, Any], Any], context: Context) -> asyncio.Future[None]:
        scheduler = self._scheduler
        invocation = self._invocation

        async def committed(tx: Any, current: AnyTaskRecord) -> None:
            result = change(tx, current)
            following = await result if inspect.isawaitable(result) else result
            if following is not None:
                await scheduler._commit_state(tx, invocation, current, following)

        return scheduler._gated(invocation, committed, context)

    def memo(self, name: str, *rest: Any) -> asyncio.Future[Any]:
        scheduler = self._scheduler
        invocation = self._invocation
        if len(rest) == 1:
            if invocation.ended:
                return rejected(self._ended())
            done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            done.set_result(_memo_of(scheduler._live.get(invocation.task_id), name))
            return done
        candidate, context = rest

        def change(tx: Any, current: AnyTaskRecord) -> Any:
            winner = _memo_of(current, name)
            if winner is not None:
                return winner
            tx.set_task({**current, "memos": {**current.get("memos", {}), name: candidate}})
            return candidate

        return scheduler._gated(invocation, change, context)

    def sleep_until(self, until: float, checkpoint: Any, context: Context) -> asyncio.Future[None]:
        scheduler = self._scheduler
        invocation = self._invocation
        return scheduler._gated(
            invocation,
            lambda tx, current: scheduler._commit_state(
                tx, invocation, current, {"status": "sleeping", "checkpoint": checkpoint, "until": until}
            ),
            context,
        )

    async def watch_doc(self, *args: Any) -> Any:
        invocation = self._invocation
        if invocation.ended:
            raise self._ended()
        watch = await self._scheduler._session.watch_doc(*args)
        if watch is None:
            return None
        if invocation.ended:
            watch.stop()
            raise self._ended()
        invocation.watches[watch] = None
        watch.closed.add_done_callback(lambda _future: invocation.watches.pop(watch, None))
        return watch

    async def snapshot(self, *args: Any) -> Any:
        if self._invocation.ended:
            raise self._ended()
        return await self._scheduler._session.snapshot(*args)

    async def snapshot_as_of(self, *args: Any) -> Any:
        if self._invocation.ended:
            raise self._ended()
        return await self._scheduler._session.snapshot_as_of(*args)

    async def get_task(self, id: TaskId[Any], context: Context) -> Any:
        if self._invocation.ended:
            raise self._ended()
        scheduler = self._scheduler
        return await scheduler._session.read_on_line(lambda: scheduler._storage.task(id, context))

    async def wait_for_task(self, id: TaskId[Any], context: Context) -> Any:
        if self._invocation.ended:
            raise self._ended()
        return await self._scheduler.wait_for_task(id, with_abort_signal(self._invocation.controller.signal, context))

    async def owned_tasks(self, context: Context) -> list[AnyTaskRecord]:
        invocation = self._invocation
        scheduler = self._scheduler
        if invocation.ended:
            raise self._ended()

        async def read() -> list[AnyTaskRecord]:
            # The step that ends the invocation may have run on the line before this.
            if invocation.ended:
                raise self._ended()
            context.throw_if_aborted()
            # Every live task is listed under its parent node, its owner task when it has one.
            owned: list[AnyTaskRecord] = []
            for node in scheduler._below.get(invocation.task_id, ()):
                record = None if isinstance(node, str) else scheduler._live.get(node)
                if record is not None and record.get("owner") == invocation.task_id:
                    owned.append(record)
            owned.sort(key=lambda record: record["id"])
            # Copies, as from storage: the scheduler reads its own.
            return [copy_json(record) for record in owned]  # type: ignore[misc]

        return await scheduler._session.read_on_line(read)

    async def abort_owned(self, id: TaskId[Any], context: Context) -> None:
        invocation = self._invocation
        scheduler = self._scheduler
        if invocation.ended:
            raise self._ended()
        bound = with_abort_signal(invocation.controller.signal, context)
        record = await scheduler._session.read_on_line(lambda: scheduler._storage.task(id, bound))
        if record is None or record.get("owner") != invocation.task_id:
            raise RuntimeError(f"Task {id} is not owned by task {invocation.task_id}")
        if record["state"]["status"] == "terminal":
            return
        await scheduler.abort(id, bound, True)
        await scheduler.wait_for_task(id, bound)

    async def outcomes(self, ids: Sequence[TaskId[Any]], context: Context) -> list[Any]:
        if self._invocation.ended:
            raise self._ended()
        scheduler = self._scheduler

        async def read() -> list[Any]:
            outcomes: list[Any] = []
            for id in ids:
                record = await scheduler._storage.task(id, context)
                state = None if record is None else record["state"]
                if state is None or state["status"] != "terminal":
                    raise RuntimeError(f"Task {id} is not terminal")
                outcomes.append(state["outcome"])
            return outcomes

        return await scheduler._session.read_on_line(read)

    async def conversation(self, id: ConversationId, context: Context) -> Any:
        invocation = self._invocation
        if invocation.ended:
            raise self._ended()

        def check() -> None:
            if invocation.ended:
                raise self._ended()

        binding = InvocationBinding(invocation.controller.signal, check)
        return await self._scheduler._conversation(id, binding, context)

    async def entry(self, *args: Any) -> Any:
        if len(args) == 2:
            token, id, context = None, args[0], args[1]
        else:
            token, id, context = args
        if self._invocation.ended:
            raise self._ended()
        scheduler = self._scheduler
        found = await scheduler._session.read_on_line(
            lambda: scheduler._storage.visible_entry(self._invocation.conversation_id, id, context)
        )
        entry = None if found is None else found["entry"]
        return entry if token is None or (entry is not None and entry["kind"] == token.kind) else None

    async def context(self, conversation_id: ConversationId, context: Context, *, at: EntryId | None = None) -> Any:
        invocation = self._invocation
        scheduler = self._scheduler
        if invocation.ended:
            raise self._ended()
        kept = scheduler._contexts.get(conversation_id)
        view, range_ = await read_context_from(
            scheduler._session,
            scheduler._storage,
            conversation_id,
            context,
            at,
            None if kept is None else kept[0],
        )
        # Keep it unless the invocation ended or a concurrent read already kept a newer range.
        kept = scheduler._contexts.get(conversation_id)
        if (
            range_ is not None
            and not scheduler._closing
            and not invocation.ended
            and (kept is None or kept[0].bounds.tail <= range_.bounds.tail)
        ):
            # A read of another, idle conversation starts or continues its retention period.
            if not scheduler._idle(conversation_id):
                scheduler._contexts[conversation_id] = [range_, None]
            elif scheduler._context_retention_ms() > 0:
                idle_since = None if kept is None else kept[1]
                scheduler._contexts[conversation_id] = [
                    range_,
                    idle_since if idle_since is not None else scheduler._now(),
                ]
                scheduler._schedule_expiry()
        return view

    def now(self) -> float:
        if self._invocation.ended:
            raise self._ended()
        return self._scheduler._now()

    def set_timer(self, callback: Callable[[], None], delay_ms: float) -> Callable[[], None]:
        invocation = self._invocation
        # An ended invocation's timers never fire.
        if invocation.ended:
            return lambda: None
        holder: list[Callable[[], None]] = []

        def fire() -> None:
            if holder:
                invocation.timers.pop(holder[0], None)
            if not invocation.ended:
                callback()

        disarm = self._scheduler._timers.foreground(fire, delay_ms)
        holder.append(disarm)
        invocation.timers[disarm] = None

        def cancel() -> None:
            invocation.timers.pop(disarm, None)
            disarm()

        return cancel

    def report(self, error: BaseException, from_: Any = None) -> None:
        invocation = self._invocation
        scheduler = self._scheduler
        if invocation.ended:
            raise self._ended()
        # The task's own failure names the task's extension; any other names whose code failed, if anyone's.
        task = _report_task_of(invocation)
        if from_ is None:
            info: Any = {"type": "task_error", **scheduler._invocation_about(invocation)}
        elif from_["type"] == "extension":
            info = {"type": "extension_error", "task": task, **optional_extension(from_.get("extension"))}
        else:
            info = {"type": "host_error", "callback": from_["callback"], "task": task}
        scheduler._report(error, info)


def _race_failed(done: asyncio.Future[None], failed: asyncio.Future[Any]) -> asyncio.Future[None]:
    """``Promise.race([done, failed])``: resolves with ``done``, fails when the Session fails first.

    Settled through done callbacks, without a task, so a caller resumes as soon as the run ends.
    """
    result: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def on_done(_future: asyncio.Future[None]) -> None:
        if not result.done():
            result.set_result(None)

    def on_failed(future: asyncio.Future[Any]) -> None:
        if result.done() or future.cancelled():
            return
        error = future.exception()
        if error is not None:
            result.set_exception(error)

    done.add_done_callback(on_done)
    failed.add_done_callback(on_failed)
    return result


def _installer_of(snapshot: RegistrySnapshot, task: AnyTask | None) -> str | None:
    """The extension that installed ``task`` in ``snapshot``, the last one that lists it; none for a built-in."""
    if task is None:
        return None
    found: str | None = None
    for installed in snapshot.installed():
        if any(candidate is task for candidate in installed.tasks):
            found = installed.name
    return found


def _report_task(task: dict[str, Any]) -> ReportTask:
    """The report info of a failure of a task."""
    return {"id": task["taskId"], "kind": task["kind"], "conversationId": task["conversationId"]}


def _report_task_of(invocation: _Invocation) -> ReportTask:
    return {"id": invocation.task_id, "kind": invocation.kind, "conversationId": invocation.conversation_id}


def _record_task(record: AnyTaskRecord) -> dict[str, Any]:
    """A stored task's identity, as an invocation names it."""
    return {"taskId": record["id"], "kind": record["kind"], "conversationId": record["conversationId"]}


def _with_abort_mark(record: AnyTaskRecord, reason: str | None = None) -> AnyTaskRecord:
    """The record with an abort mark: ``restart`` for an abandonment; without a reason, a request.

    The restart count starts over: it now counts cut-offs of the abort handler.
    """
    rest = {key: value for key, value in record.items() if key not in ("abortReason", "restarts")}
    rest["abortRequested"] = True
    if reason is not None:
        rest["abortReason"] = reason
    return rest  # type: ignore[return-value]


def _cancellation_intent(record: AnyTaskRecord) -> bool:
    """A live owner's durable cancellation intent: its abort mark, or a held outcome other than ``completed``."""
    return record["state"]["status"] != "terminal" and (record["abortRequested"] or _failed_outcome(record))


def _failed_outcome(record: AnyTaskRecord) -> bool:
    """Whether the record holds or ends with an outcome other than ``completed``."""
    state = record["state"]
    return state["status"] in ("completing", "terminal") and state["outcome"]["status"] != "completed"  # type: ignore[typeddict-item]


def _node_of(record: AnyTaskRecord) -> _TaskNode:
    node: dict[str, Any] = {"conversationId": record["conversationId"]}
    if "owner" in record:
        node["owner"] = record["owner"]
    node["background"] = record["background"]
    return node  # type: ignore[return-value]


def _overlay_of(tx: Any) -> _Overlay:
    tasks = {record["id"]: record for record in tx.staged_tasks()}
    edges: dict[ConversationId, Any] = {}
    for record in tx.staged_conversations():
        owner = record.get("owner")
        edges[record["id"]] = None if owner is None else owner["taskId"]
    return _Overlay(tasks, edges)


def _add_to(index: dict[Any, dict[Any, None]], key: Any, value: Any) -> None:
    values = index.get(key)
    if values is None:
        index[key] = {value: None}
    else:
        values[value] = None


def _delete_from(index: dict[Any, dict[Any, None]], key: Any, value: Any) -> None:
    values = index.get(key)
    if values is None:
        return
    values.pop(value, None)
    if len(values) == 0:
        del index[key]


def _memo_of(record: AnyTaskRecord | None, name: str) -> Any:
    memos = None if record is None else record.get("memos")
    return None if memos is None else memos.get(name)


def _missing_migration(record: AnyTaskRecord, definition: Any) -> BaseException:
    return RuntimeError(f"Task {record['kind']} version {definition.version} has no migration from {record['version']}")


def _ended_error(invocation: _Invocation) -> BaseException:
    return RuntimeError(f"Task {invocation.task_id} invocation has ended")


def _progressed(record: AnyTaskRecord) -> AnyTaskRecord:
    """The record without its restart count, as after a state the task committed."""
    if "restarts" not in record:
        return record
    return {key: value for key, value in record.items() if key != "restarts"}  # type: ignore[return-value]


def _with_state(record: AnyTaskRecord, state: Any) -> AnyTaskRecord:
    """Replace a live record's state; memos and the restart count disappear once an outcome is decided."""
    if state["status"] not in ("terminal", "completing"):
        return {**record, "state": state}
    rest = {key: value for key, value in record.items() if key not in ("memos", "restarts")}
    return {**rest, "state": state}  # type: ignore[return-value]


def _can_reserve(task: AnyTask, record: AnyTaskRecord) -> bool:
    """Whether a definition can take the task at reservation: same version, or newer with a migration."""
    definition = task.definition
    return definition.version == record["version"] or (
        definition.version > record["version"] and definition.migrate is not None
    )


def _validate_sleep(invocation: _Invocation, task: TaskId[Any], until: Any) -> None:
    """A sleep's ``until`` is a time on the Harness clock; an abort handler cannot sleep."""
    if invocation.mode == "abort":
        raise RuntimeError(f"Abort handler of task {task} cannot sleep")
    if not isinstance(until, int | float) or isinstance(until, bool) or not math.isfinite(until) or until < 0:
        raise RuntimeError(f"Task {task} cannot sleep until {_js_string(until)}: not a time in milliseconds")


def _js_string(value: Any) -> str:
    """``String(value)`` for the values a sleep may be given."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_integer():
            return str(int(value))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
