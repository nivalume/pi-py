"""The Harness: a Session kernel extended with conversations, a registry, and a task scheduler (port of ``harness/harness.ts``).

Methods that admit work to the Session line (commits, line reads) run that part at the call and return an
``asyncio`` future, as upstream's Promise-returning methods do (DESIGN.md §5).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeVar, cast

from pi_chord.context import Context, with_abort_signal, without_abort_signal
from pi_chord.state import AttachedReplicatedState

from ..entries import ResetEntry
from ..session import SessionImpl, Transaction, contained, date_now
from ..session._futures import rejected
from ..timers import host_timers
from ..types import (
    ROOT_CONVERSATION_ID,
    ConversationId,
    ConversationOwnership,
    ConversationRecord,
    Cursor,
    EntryId,
    EntryRecord,
    Page,
    ReportInfo,
    Storage,
    SubmissionId,
    TaskId,
    TaskRecord,
    Tx,
)
from .agent import AgentDoc, configure, create_agent, resolve_agent, resolve_settings
from .compaction import create_compaction
from .context import read_context
from .inbox import InboxDoc, withdraw_queued_inputs
from .live import LiveDoc, settle_scheduler_outcome
from .provider import ProviderDoc
from .registry import BUILTIN_TASKS
from .scheduler import InvocationBinding, TaskScheduler, TaskSchedulerOptions
from .submissions import Submissions
from .task_graph import TaskGraph, TaskGraphBlocking, TaskGraphView, TaskGraphWatch
from .types import (
    Agent,
    AgentChange,
    CompactionResult,
    ContextView,
    ConversationCreateOptions,
    ConversationInit,
    ConversationWatch,
    EntriesQuery,
    HarnessInspection,
    HarnessOptions,
    HarnessSettings,
    RegistrySnapshot,
    SettledTask,
    Submission,
    SubmissionDraft,
    Wake,
)
from .usage import UsageDoc, UsageState, add_usage_state
from .util import eager, scan_all
from .view import ConversationView, ConversationViews

__all__ = ["BoundConversation", "ConversationImpl", "Harness", "safe_now"]

T = TypeVar("T")

_SCAN_PAGE_SIZE: Final = 256
_LISTENER: Final[ReportInfo] = {"type": "listener_error"}


@dataclass(frozen=True, slots=True)
class _CreateTarget:
    kind: Literal["root", "independent", "fork"]
    ownership: ConversationOwnership | None = None
    parent_id: ConversationId | None = None
    at: EntryId | None = None


class ConversationImpl:
    """Stateless handle for one conversation, bound to the Harness that returned it. Compare handles by ``id``."""

    __slots__ = ("_harness", "id")

    def __init__(self, id: ConversationId, harness: Harness) -> None:
        self.id = id
        self._harness = harness

    def __repr__(self) -> str:
        return f"Conversation({self.id})"

    def agent(self, context: Context) -> asyncio.Future[Agent]:
        """Resolved with the current registry snapshot and settings."""
        return asyncio.ensure_future(self._harness.resolve_agent(self.id, None, context))

    def configure(self, change: AgentChange, context: Context) -> asyncio.Future[None]:
        """``configure()`` in its own commit."""
        return self._harness.commit_with(lambda tx: configure(tx, self.id, change), context)

    def submit(self, submission: SubmissionDraft, context: Context) -> asyncio.Future[Submission]:
        """Durably admit user input or a passive entry write."""
        return self._harness.submissions.submit(self.id, submission, context)

    def compact(self, instructions: str | None, context: Context) -> asyncio.Future[TaskId[CompactionResult]]:
        """Admit a manual compaction task and return its ID."""
        try:
            self._harness.tasks.resume()
        except Exception as error:
            return rejected(error)
        task_input: Any = (
            {"reason": "manual"} if instructions is None else {"reason": "manual", "instructions": instructions}
        )
        return self._harness.commit_with(lambda tx: create_compaction(tx, self.id, task_input), context)

    def reset(self, handoff: str | None, context: Context) -> asyncio.Future[None]:
        """Admit a write of a ``pi.reset`` entry that starts a new context."""
        from pi_ai.types import UserMessage

        entry: Any = {"kind": ResetEntry.kind, "head": "self"}
        if handoff is not None:
            entry["model"] = [UserMessage(handoff, timestamp=self._harness.now())]  # type: ignore[arg-type]
        return eager(
            self._harness.submissions.submit(self.id, {"type": "write", "entry": entry}, context), lambda _: None
        )

    def commit(self, change: Callable[[Tx], T | Awaitable[T]], context: Context) -> asyncio.Future[T]:
        """Session commit whose ``tx.create_task()`` defaults to this conversation."""
        return self._harness.commit_with(change, context, {"conversationId": self.id})  # type: ignore[arg-type]

    def context(self, context: Context, *, at: EntryId | None = None) -> asyncio.Future[ContextView]:
        """Committed raw active transcript and model context; with ``at``, as of that visible entry."""
        try:
            self._harness.assert_usable()
        except Exception as error:
            return rejected(error)
        return read_context(self._harness, self._harness.storage, self.id, context, at)

    def entries(
        self, query: EntriesQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> asyncio.Future[Page[EntryRecord]]:
        """Fork-aware history of this conversation, newest first unless ``query["order"]`` is ``ascending``."""
        bounded: Any = {"conversationId": self.id}
        for key in ("minEntryId", "maxEntryId", "order"):
            if query.get(key) is not None:
                bounded[key] = query[key]  # type: ignore[literal-required]
        storage = self._harness.storage
        return self._harness.read_on_line(lambda: storage.scan_entries(bounded, limit, cursor, context))

    def fork(
        self, at: EntryId, options: ConversationCreateOptions, context: Context
    ) -> asyncio.Future[ConversationImpl]:
        return self._harness.create(
            _CreateTarget("fork", options["ownership"], self.id, at), options.get("agent"), options.get("init"), context
        )

    def abort(self, context: Context, *, background: bool = False) -> asyncio.Future[None]:
        """Withdraw queued inputs, mark every live non-background task of the scope, and resolve once idle."""
        try:
            self._harness.tasks.resume()
        except Exception as error:
            return rejected(error)
        return asyncio.ensure_future(self._harness.tasks.abort_conversation(self.id, background, context))

    def wait_for_idle(self, context: Context) -> asyncio.Future[None]:
        """Resolve when this conversation's ordinary ownership scope has no live non-background task."""
        try:
            self._harness.tasks.resume()
        except Exception as error:
            return rejected(error)
        return self._harness.tasks.wait_for_idle(self.id, context)

    def view_state(self, context: Context) -> asyncio.Future[AttachedReplicatedState[ConversationView]]:
        """The structural view (spec §9.3) as a disposable read-only Chord state."""
        return asyncio.ensure_future(self._harness.views.state(self.id, context))

    def watch(self, context: Context) -> asyncio.Future[ConversationWatch]:
        """The structural view as a serialized exact-frame watch with bounded pending frames."""
        return asyncio.ensure_future(self._harness.views.watch(self.id, context))


def _current_models(models: Any) -> Callable[[], Any]:
    """``HarnessOptions.models`` as a function: a plain ``Models`` is returned at every call."""
    return models if callable(models) else lambda: models


def _current_settings(settings: Any) -> Callable[[], HarnessSettings | None]:
    """``HarnessOptions.settings`` as a function: plain settings, or none, are returned at every call."""
    return cast("Callable[[], HarnessSettings | None]", settings if callable(settings) else lambda: settings)


class Harness(SessionImpl):
    """Durable agent harness over one Session (``Harness.open()``).

    A Session kernel extended with conversation handles, a registry, and a task scheduler. The registry may keep
    changing while the Harness runs.
    """

    def __init__(self, raw_storage: Storage, options: HarnessOptions, context: Context) -> None:
        report = _safe_report(options.get("onReport"))
        now = safe_now(options.get("now"), report)
        host_settings = _current_settings(options.get("settings"))

        def retention() -> Any:
            settings = host_settings()
            return None if settings is None else settings.get("documentRetentionMs")

        # Read at each use, as all settings are, but only this field: it is read on every commit.
        super().__init__(raw_storage, now, retention, options.get("setTimer"))
        # Every component reads and writes through the Session's failure guard: ``raw_storage`` serves only ``super()``.
        storage = self.storage
        self._options = options
        self._harness_report = report
        self._host_settings = host_settings
        self.now: Callable[[], float] = now

        def settings() -> Any:
            return resolve_settings(host_settings())

        async def conversation(id: ConversationId, binding: InvocationBinding, call_context: Context) -> Any:
            record = await self.read_on_line(lambda: storage.conversation(id, call_context))
            return None if record is None else BoundConversation(id, binding, self.submissions, self.tasks)

        set_timer = options.get("setTimer")
        self.tasks = TaskScheduler(
            TaskSchedulerOptions(
                session=self,
                storage=storage,
                registry=options["registry"],
                models=_current_models(options["models"]),
                agent=lambda id, snapshot, call_context: self.resolve_agent(id, snapshot, call_context),
                settings=settings,
                env=lambda id, call_context: self.build_env(id, call_context),
                now=now,
                on_wake=options.get("onWake"),
                timers=host_timers(
                    set_timer, lambda error: report(error, {"type": "host_error", "callback": "setTimer"})
                ),
                write_at_close=self.write_at_close,
                report=report,
                settle_outcome=settle_scheduler_outcome,
                withdraw_inputs=withdraw_queued_inputs,
                conversation=conversation,
                context=without_abort_signal(context),
            )
        )
        self.submissions = Submissions(self, storage, now, settings, self.tasks.resume)
        self._task_graph = TaskGraphView(
            self, storage, TaskGraphBlocking(self.tasks.blocked_reason, self.tasks.subscribe_blocked)
        )
        self.views = ConversationViews(self, storage)

    @classmethod
    async def open(cls, storage: Storage, options: HarnessOptions, context: Context) -> Harness:
        """Open a Harness over ``storage``. The registry may keep changing while the Harness runs.

        ``options`` is the ``HarnessOptions`` dict (``models``, ``registry``, ``settings``, ``env``,
        ``conversationCreated``, ``now``, ``setTimer``, ``onWake``, ``onReport``).
        """
        context.throw_if_aborted()
        snapshot = options["registry"].snapshot()
        missing = [task for task in BUILTIN_TASKS if snapshot.task(task.definition.name) is None]
        if len(missing) > 0:
            names = ", ".join(task.definition.name for task in missing)
            raise RuntimeError(f"Registry lacks built-in tasks {names}; create it with create_registry()")
        harness = cls(storage, options, context)
        try:
            await harness.tasks.open(context)
        except BaseException:
            # The caller's context may be what failed open: close without it, and raise the open error.
            # A failing Storage close fails the Session, which reports it.
            try:
                await harness.close(without_abort_signal(context))
            except Exception:
                pass
            raise
        return harness

    # ─── Internal services ───────────────────────────────────────────────

    async def resolve_agent(self, id: ConversationId, snapshot: RegistrySnapshot | None, context: Context) -> Agent:
        """Resolve a conversation's committed ``pi.agent`` against ``snapshot``, or the current one, and the settings."""
        registry = snapshot if snapshot is not None else self._options["registry"].snapshot()
        state = await self.snapshot(AgentDoc, id, context)
        return resolve_agent(
            state,
            registry,
            resolve_settings(self._host_settings()),
            lambda error, extension: self._harness_report(
                error, {"type": "extension_error", "conversationId": id, "extension": extension}
            ),
        )

    async def build_env(self, id: ConversationId, context: Context) -> Any:
        """Build a conversation's environment from its current ``cwd``; ``None`` without an ``env`` option."""
        build = self._options.get("env")
        if build is None:
            return None
        state = await self.snapshot(AgentDoc, id, context)
        cwd = None if state is None else state.get("cwd")
        target: Any = {"conversationId": id, "read": self}
        if cwd is not None:
            target["cwd"] = cwd
        result = build(target, context)
        return await result if inspect.isawaitable(result) else result

    @property
    def scheduler_index_sizes(self) -> Any:
        """Internal, for tests: sizes of the scheduler's ownership indexes."""
        return self.tasks.index_sizes

    # ─── Harness surface ─────────────────────────────────────────────────

    def resume(self) -> None:
        """Enable task scheduling. Idempotent; raises after close."""
        self.assert_usable()
        self.tasks.resume()

    def wake(self) -> Wake:
        """When this Session next needs a running process, from the scheduler's memory."""
        self.assert_usable()
        return self.tasks.wake()

    def get_task(self, id: TaskId[T], context: Context) -> asyncio.Future[TaskRecord[Any, Any, T] | None]:
        storage = self.storage
        return self.read_on_line(lambda: storage.task(id, context))

    def inspect(self, context: Context) -> asyncio.Future[HarnessInspection]:
        """Live tasks and unsettled submissions. Writes nothing and runs no task code."""
        storage = self.storage

        async def job() -> HarnessInspection:
            scheduling, tasks = await self.tasks.inspect(self._options["registry"].snapshot())

            async def scan(status: str) -> list[Any]:
                return await scan_all(
                    lambda cursor: storage.scan_submissions({"status": status}, _SCAN_PAGE_SIZE, cursor, context)  # type: ignore[typeddict-item]
                )

            submissions = sorted([*(await scan("queued")), *(await scan("placed"))], key=lambda record: record["id"])
            return {"scheduling": scheduling, "tasks": tasks, "submissions": submissions}  # type: ignore[typeddict-item]

        return self.read_on_line(job)

    def submission(self, id: SubmissionId, context: Context) -> asyncio.Future[Submission | None]:
        """Reacquire a submission, for example after reopen."""
        return self.submissions.get(id, context)

    def abort_submission(
        self, id: SubmissionId, context: Context, conversation_id: ConversationId | None = None
    ) -> asyncio.Future[Literal["aborted", "already_placed", "settled", "not_found"]]:
        return self.submissions.abort(id, context, conversation_id)

    def abort_task(self, id: TaskId[Any], context: Context) -> asyncio.Future[Literal["marked", "terminal"]]:
        return self.tasks.abort(id, context)

    def retry_task(self, id: TaskId[Any], context: Context) -> asyncio.Future[None]:
        """Run a task parked by ``settings.restartLimit`` again."""
        try:
            self.assert_usable()
        except Exception as error:
            return rejected(error)
        return asyncio.ensure_future(self.tasks.retry(id, context))

    def wait_for_task(self, id: TaskId[T], context: Context) -> asyncio.Future[SettledTask[T]]:
        """Resolve with the terminal receipt; cancelling ``context`` cancels only this wait."""
        self.tasks.resume()
        return self.tasks.wait_for_task(id, context)

    def wait_for_idle(self, context: Context) -> asyncio.Future[None]:
        self.tasks.resume()
        return self.tasks.wait_for_idle(None, context)

    async def usage(self, context: Context) -> UsageState:
        """Sum every conversation's committed ``pi.usage``. Each document is read at its own point; totals only grow."""
        storage = self.storage
        conversations = await self.read_on_line(
            lambda: scan_all(lambda cursor: storage.scan_conversations({}, _SCAN_PAGE_SIZE, cursor, context))
        )
        total = UsageDoc.definition.initial()
        for conversation in conversations:
            state = await self.snapshot(UsageDoc, conversation["id"], context)
            if state is not None:
                add_usage_state(total, state)
        return total

    def task_graph(self, context: Context) -> asyncio.Future[AttachedReplicatedState[TaskGraph]]:
        return asyncio.ensure_future(self._task_graph.state(context))

    def watch_task_graph(self, context: Context) -> asyncio.Future[TaskGraphWatch]:
        return asyncio.ensure_future(self._task_graph.watch(context))

    def root(
        self, context: Context, *, agent: AgentChange | None = None, init: ConversationInit | None = None
    ) -> asyncio.Future[ConversationImpl]:
        """The reserved root conversation, created with ``agent`` and ``init`` in one commit when absent."""
        return self.create(_CreateTarget("root"), agent, init, context)

    def conversation(self, id: ConversationId, context: Context) -> asyncio.Future[ConversationImpl | None]:
        try:
            self.assert_usable()
        except Exception as error:
            return rejected(error)
        storage = self.storage
        record = self.read_on_line(lambda: storage.conversation(id, context))
        return eager(record, lambda found: None if found is None else ConversationImpl(found["id"], self))

    def create_conversation(
        self, options: ConversationCreateOptions, context: Context
    ) -> asyncio.Future[ConversationImpl]:
        return self.create(
            _CreateTarget("independent", options["ownership"]), options.get("agent"), options.get("init"), context
        )

    def report(self, error: BaseException, info: ReportInfo = _LISTENER) -> None:
        self._harness_report(error, info)

    async def before_close(self) -> None:
        """Signal and join task invocations after the close listeners and before Storage closes."""
        await self.tasks.join()

    def create(
        self,
        target: _CreateTarget,
        agent: AgentChange | None,
        init: ConversationInit | None,
        context: Context,
    ) -> asyncio.Future[ConversationImpl]:
        """Internal: create the root, an independent conversation, or a fork, with its conveniences, in one commit."""
        try:
            self.assert_usable()
        except Exception as error:
            return rejected(error)

        async def change(tx: Transaction) -> ConversationId:
            if target.kind == "root" and await tx.conversation(ROOT_CONVERSATION_ID) is not None:
                return ROOT_CONVERSATION_ID
            if target.kind == "root":
                record = await tx.create_root_conversation()
            elif target.kind == "fork":
                record = await tx.fork_conversation(target.parent_id, target.at, {"ownership": target.ownership})  # type: ignore[arg-type,typeddict-item]
            else:
                record = await tx.create_conversation({"ownership": target.ownership})  # type: ignore[typeddict-item]
            if agent is not None:
                await configure(tx, record["id"], agent)
            if init is not None:
                result = init(tx, record["id"])
                if inspect.isawaitable(result):
                    await result
            return record["id"]

        return eager(self.commit_with(change, context), lambda id: ConversationImpl(id, self))

    async def conversation_created(self, tx: Transaction, record: ConversationRecord) -> None:
        """The built-in creation hook, in every commit that creates or forks a conversation.

        Empty ``pi.live``, ``pi.inbox``, and ``pi.usage``, a fresh ``pi.provider``, the conversation's ``pi.agent``
        (see ``create_agent()``), then ``HarnessOptions.conversationCreated``.
        """
        await tx.doc(LiveDoc, record["id"])
        await tx.doc(InboxDoc, record["id"])
        await tx.doc(UsageDoc, record["id"])
        await tx.doc(ProviderDoc, record["id"])
        await create_agent(tx, record)
        created = self._options.get("conversationCreated")
        if created is not None:
            result = created(tx, record)
            if inspect.isawaitable(result):
                await result


def _safe_report(
    on_report: Callable[[BaseException, ReportInfo], Any] | None,
) -> Callable[[BaseException, ReportInfo], None]:
    """``onReport`` that cannot fail: it is the last place an error goes, so its own raise is dropped."""

    def report(error: BaseException, info: ReportInfo) -> None:
        if on_report is None:
            return
        callback = on_report
        contained(lambda: callback(error, info), lambda _error: None)

    return report


def safe_now(
    now: Callable[[], float] | None, report: Callable[[BaseException, ReportInfo], None]
) -> Callable[[], float]:
    """The host clock, falling back to ``date_now`` when it raises: a broken clock must not stop commits.

    The first raise is reported; the clock is read in every task change, so later ones are not.
    """
    if now is None:
        return date_now
    reported = [False]

    def safe() -> float:
        try:
            return now()
        except Exception as error:
            # Set first: a report handler that reads the clock again must not report again.
            if not reported[0]:
                reported[0] = True
                report(error, {"type": "host_error", "callback": "now"})
            return date_now()

    return safe


class BoundConversation:
    """Invocation-bound handle for tasks and tools (``ConversationHandle``).

    Every operation, and every operation of a submission it returns, first checks the invocation and runs under
    its signal, so it raises once the invocation ends; admitted work stays durable.
    """

    __slots__ = ("_binding", "_submissions", "_tasks", "id")

    def __init__(
        self, id: ConversationId, binding: InvocationBinding, submissions: Submissions, tasks: TaskScheduler
    ) -> None:
        self.id = id
        self._binding = binding
        self._submissions = submissions
        self._tasks = tasks

    def _bind(self, context: Context) -> Context:
        return with_abort_signal(self._binding.signal, context)

    async def submit(self, draft: Any, context: Context) -> Any:
        self._binding.check()
        submission = await self._submissions.submit(self.id, draft, self._bind(context))
        return _BoundSubmission(submission, self._binding)

    async def abort(self, context: Context, *, background: bool = False) -> None:
        self._binding.check()
        await self._tasks.abort_conversation(self.id, background, self._bind(context))

    async def wait_for_idle(self, context: Context) -> None:
        self._binding.check()
        await self._tasks.wait_for_idle(self.id, self._bind(context))


class _BoundSubmission:
    """A submission whose operations check the invocation and run under its signal."""

    __slots__ = ("_binding", "_submission", "id")

    def __init__(self, submission: Any, binding: InvocationBinding) -> None:
        self.id = submission.id
        self._submission = submission
        self._binding = binding

    async def status(self, context: Context) -> Any:
        self._binding.check()
        return await self._submission.status(with_abort_signal(self._binding.signal, context))

    async def wait(self, context: Context) -> Any:
        self._binding.check()
        return await self._submission.wait(with_abort_signal(self._binding.signal, context))

    async def abort(self, context: Context) -> Any:
        self._binding.check()
        return await self._submission.abort(with_abort_signal(self._binding.signal, context))
