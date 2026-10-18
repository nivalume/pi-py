"""The task graph mount: every live task of the Session (port of ``harness/task-graph.ts``, spec §9.5)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict

from typing_extensions import ReadOnly

from pi_chord.context import BACKGROUND_CONTEXT, Context, without_abort_signal
from pi_chord.delta import Op, apply_immutable
from pi_chord.state import AttachedReplicatedState, ReplicatedStateSourceOptions, replicated_state

from ..session import CommittedStateSource, CommittedWatch, contained
from ..session._futures import consume
from ..types import AnyTaskRecord, CommitPublication, ConversationId, JoinPolicy, Storage, TaskId, WatchHandle
from .scheduler import BlockedReason, BlockedSubject, blocked_subject
from .util import closed_error, scan_all

__all__ = [
    "TaskGraph",
    "TaskGraphBlocking",
    "TaskGraphNode",
    "TaskGraphState",
    "TaskGraphView",
    "TaskGraphWatch",
]


class TaskGraphStateActive(TypedDict):
    status: ReadOnly[Literal["pending", "running"]]
    phase: ReadOnly[str]


class TaskGraphStateWaiting(TypedDict):
    status: ReadOnly[Literal["waiting"]]
    phase: ReadOnly[str]
    on: ReadOnly[Sequence[TaskId[Any]]]
    policy: ReadOnly[JoinPolicy]


class TaskGraphStateSleeping(TypedDict):
    """Asleep until the Harness clock reaches ``until``."""

    status: ReadOnly[Literal["sleeping"]]
    phase: ReadOnly[str]
    until: ReadOnly[float]


class TaskGraphStateCompleting(TypedDict):
    """Outcome held until its ordinary owned work drains."""

    status: ReadOnly[Literal["completing"]]
    outcome: ReadOnly[str]


#: A live task's durable status without its checkpoint and outcome payloads (spec §9.5).
TaskGraphState: TypeAlias = (
    TaskGraphStateActive | TaskGraphStateWaiting | TaskGraphStateSleeping | TaskGraphStateCompleting
)


class TaskGraphNode(TypedDict):
    id: ReadOnly[TaskId[Any]]
    kind: ReadOnly[str]
    conversationId: ReadOnly[ConversationId]
    #: Owner task; absent for a conversation-owned task.
    owner: ReadOnly[NotRequired[TaskId[Any]]]
    background: ReadOnly[bool]
    abortRequested: ReadOnly[bool]
    state: ReadOnly[TaskGraphState]
    #: Why it cannot run here, when it cannot. Follows registry changes.
    blocked: ReadOnly[NotRequired[BlockedReason]]
    #: Conversations this task owns, in ID order.
    conversations: ReadOnly[Sequence[ConversationId]]


class TaskGraph(TypedDict):
    """Every live task of the Session (spec §9.5), keyed by its decimal ID."""

    tasks: ReadOnly[dict[str, TaskGraphNode]]


TaskGraphWatch: TypeAlias = WatchHandle[TaskGraph]


@dataclass(frozen=True, slots=True)
class TaskGraphBlocking:
    """What the graph asks the scheduler: a task's blocked reason, and when reasons may change without a commit."""

    blocked: Callable[[BlockedSubject], BlockedReason | None]
    subscribe_blocked: Callable[[Callable[[], None]], Callable[[], None]]


@dataclass(slots=True, eq=False)
class _Mount:
    value: TaskGraph
    #: What the blocked reason reads of each live task, so a registry change can derive it again without payloads.
    records: dict[str, BlockedSubject]
    observers: dict[Any, None] = field(default_factory=dict)


_LIVE_STATUSES: Final = ("pending", "running", "waiting", "sleeping", "completing")
_SCAN_PAGE_SIZE: Final = 256


class TaskGraphView:
    """The Harness's task graph mount: built on the Session line by its first observer and dropped with its last.

    It advances from the Session's commit publications, which are durable.
    """

    __slots__ = ("_blocking", "_closed", "_mount", "_session", "_storage")

    def __init__(self, session: Any, storage: Storage, blocking: TaskGraphBlocking) -> None:
        self._session = session
        self._storage = storage
        self._blocking = blocking
        self._mount: _Mount | None = None
        self._closed = False

        def on_commit(publication: CommitPublication, context: Context) -> None:
            if self._mount is None:
                return
            _advance(self._mount, publication, context, blocking.blocked, session.report)

        session.observe_commits(on_commit)
        blocking.subscribe_blocked(self._refresh_blocked)

        def on_close() -> None:
            self._closed = True
            mount = self._mount
            for observer in list(mount.observers if mount is not None else ()):
                contained(lambda observer=observer: observer.close_session(self._session.failure), session.report)
            self._mount = None

        session.subscribe_close(on_close)

    async def state(self, context: Context) -> AttachedReplicatedState[TaskGraph]:
        """A disposable read-only Chord state of the graph."""
        observer, detach = await self._attach(lambda value, release: CommittedStateSource(value, release), context)
        try:
            return replicated_state(observer, ReplicatedStateSourceOptions(on_error=self._session.report))
        except BaseException:
            detach()
            raise

    async def watch(self, context: Context) -> TaskGraphWatch:
        """A serialized exact-frame watch of the graph; cancelling ``context`` stops it."""
        observer, _detach = await self._attach(
            lambda value, release: CommittedWatch(value, release, self._session.report), context
        )
        signal = context.abort_signal
        if signal is not None and signal.aborted:
            observer.cancel()
            context.throw_if_aborted()
        if signal is not None:
            observer.observe_cancellation(signal)
        return observer

    def _attach(self, create: Callable[[TaskGraph, Callable[[], None]], Any], context: Context) -> Any:
        """Register an observer created from the current revision, atomically on the Session line."""

        async def job() -> tuple[Any, Callable[[], None]]:
            mount = self._mount
            if mount is None:
                value, records = await self._build(context)
                mount = _Mount(value, records)
            holder: list[Any] = []

            def detach() -> None:
                if holder:
                    mount.observers.pop(holder[0], None)
                if len(mount.observers) == 0 and self._mount is mount:
                    self._mount = None

            observer = create(mount.value, detach)
            holder.append(observer)
            # Close or cancellation may begin while the mount builds; register nothing then.
            if self._closed:
                raise closed_error(self._session)
            context.throw_if_aborted()
            self._mount = mount
            mount.observers[observer] = None
            return observer, detach

        return self._session.read_on_line(job)

    def _refresh_blocked(self) -> None:
        """On the Session line: derive every node's blocked reason again, and publish the ones that changed."""
        if self._mount is None or self._closed:
            return

        async def job() -> None:
            mount = self._mount
            if mount is None:
                return
            ops: list[Op] = []
            for key, record in mount.records.items():
                blocked = self._blocking.blocked(record)
                node = mount.value["tasks"].get(key)
                if blocked == (None if node is None else node.get("blocked")):
                    continue
                ops.append(
                    ("d", ("tasks", key, "blocked")) if blocked is None else ("s", ("tasks", key, "blocked"), blocked)
                )
            _publish(mount, ops, BACKGROUND_CONTEXT, self._session.report)

        try:
            # Rejected only once the Session is closing or failed.
            self._session.read_on_line(job).add_done_callback(consume)
        except Exception:
            pass

    async def _build(self, context: Context) -> tuple[TaskGraph, dict[str, BlockedSubject]]:
        tasks: dict[str, TaskGraphNode] = {}
        live: dict[str, BlockedSubject] = {}
        records: list[AnyTaskRecord] = []
        for status in _LIVE_STATUSES:
            records.extend(
                await scan_all(
                    lambda cursor, status=status: self._storage.scan_tasks(
                        {"status": status}, _SCAN_PAGE_SIZE, cursor, context
                    )
                )
            )
        records.sort(key=lambda record: record["id"])
        for record in records:
            owned = await scan_all(
                lambda cursor, record=record: self._storage.scan_conversations(
                    {"ownerTaskId": record["id"]}, _SCAN_PAGE_SIZE, cursor, context
                )
            )
            conversations = sorted(conversation["id"] for conversation in owned)
            subject = blocked_subject(record)
            tasks[str(record["id"])] = _node_of(record, conversations, self._blocking.blocked(subject))
            live[str(record["id"])] = subject
        return {"tasks": tasks}, live


_ABSENT: Final[Any] = object()


def _advance(
    mount: _Mount,
    publication: CommitPublication,
    context: Context,
    blocked: Callable[[BlockedSubject], BlockedReason | None],
    report: Callable[[BaseException], None],
) -> None:
    """Derive the mount's operations from one publication, apply them, and hand the revision to every observer."""
    ops: list[Op] = []
    # Nodes this publication set or deleted, over the mount's value.
    changed: dict[str, Any] = {}

    def node(key: str) -> TaskGraphNode | None:
        if key in changed:
            return changed[key]
        return mount.value["tasks"].get(key)

    for change in publication["changes"]:
        if change["type"] != "task":
            continue
        record = change["value"]
        key = str(record["id"])
        previous = node(key)
        if record["state"]["status"] == "terminal":
            mount.records.pop(key, None)
            if previous is None:
                continue
            ops.append(("d", ("tasks", key)))
            changed[key] = None
            continue
        subject = blocked_subject(record)
        mount.records[key] = subject
        nxt = _node_of(record, [] if previous is None else previous["conversations"], blocked(subject))
        if previous is not None and _stringify(nxt) == _stringify(previous):
            continue
        ops.append(("s", ("tasks", key), nxt))  # type: ignore[arg-type]
        changed[key] = nxt
    # After the tasks, so a conversation created with its owner task in one commit finds the owner's node. Change
    # order within a publication is unspecified, so each owner's list is sorted again.
    created: dict[str, list[ConversationId]] = {}
    for change in publication["changes"]:
        if change["type"] != "conversation" or "owner" not in change["value"]:
            continue
        key = str(change["value"]["owner"]["taskId"])
        if node(key) is not None:
            created.setdefault(key, []).append(change["value"]["id"])
    for key, ids in created.items():
        owner = node(key)
        assert owner is not None
        conversations = sorted([*owner["conversations"], *ids])
        ops.append(("s", ("tasks", key, "conversations"), conversations))  # type: ignore[arg-type]
    _publish(mount, ops, context, report)


def _stringify(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _publish(mount: _Mount, ops: list[Op], context: Context, report: Callable[[BaseException], None]) -> None:
    """Apply ``ops`` to the mount and hand the revision to every observer; one that raises is reported."""
    if len(ops) == 0:
        return
    mount.value = apply_immutable(mount.value, ops)
    frame_context = without_abort_signal(context)
    for observer in list(mount.observers):
        contained(lambda observer=observer: observer.advance(mount.value, ops, frame_context), report)


def _node_of(record: AnyTaskRecord, conversations: Sequence[ConversationId], blocked: BlockedReason | None) -> Any:
    node: dict[str, Any] = {
        "id": record["id"],
        "kind": record["kind"],
        "conversationId": record["conversationId"],
    }
    if "owner" in record:
        node["owner"] = record["owner"]
    node["background"] = record["background"]
    node["abortRequested"] = record["abortRequested"]
    node["state"] = _state_of(record)
    node["conversations"] = list(conversations)
    if blocked is not None:
        node["blocked"] = blocked
    return node


def _state_of(record: AnyTaskRecord) -> TaskGraphState:
    state: Any = record["state"]
    status = state["status"]
    if status in ("pending", "running"):
        return {"status": status, "phase": state["checkpoint"]["phase"]}
    if status == "waiting":
        return {
            "status": "waiting",
            "phase": state["checkpoint"]["phase"],
            "on": list(state["on"]),
            "policy": state["policy"],
        }
    if status == "sleeping":
        return {"status": "sleeping", "phase": state["checkpoint"]["phase"], "until": state["until"]}
    # Terminal records never reach here: they leave the graph.
    return {"status": "completing", "outcome": state["outcome"]["status"]}
