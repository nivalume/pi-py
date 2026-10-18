"""Core durable records, definitions and the Session/Storage interfaces (port of ``types.ts``).

How the TypeScript types map to Python (see ``packages/durable/DESIGN.md`` for the full contract):

- **Records and other plain data** (conversation, entry, task, submission and document records,
  storage writes, queries, pages, outcomes, option objects) are ``TypedDict`` s whose keys are
  exactly upstream's property names, camelCase included, so a record *is* its persisted JSON:
  ``{"id": 2, "conversationId": 1, "kind": "pi.user"}``. Fields are ``ReadOnly`` like upstream's
  ``readonly``; build changed records with ``{**record, "state": ...}``. Optional properties are
  ``NotRequired`` keys: absent means JS ``undefined``, ``None`` means JSON ``null``.
- **Discriminated unions** are unions of ``TypedDict`` s that share a ``Literal`` key (``type``,
  ``status``, ``kind``, ``action``, ``reason``). An anonymous upstream member is named
  ``<Union><Variant>``: ``TaskStateRunning``, ``SubmissionRecordInputDone``, ``StorageWriteEntry``.
  Narrow with ``if state["status"] == "running":``.
- **pi-ai messages** inside records (``EntryRecord["model"]``, ``ContextEdit["messages"]``) are
  ``pi_ai`` message dataclasses in memory. Storage backends encode them with
  :mod:`pi_ai.serialization` (see :mod:`pi_durable.storage.codec`). Every other payload
  (entry ``data``, task input/checkpoint/result, memos, document values) is strict JSON.
- **IDs and sequences** are plain ``int`` at runtime. ``ConversationId``, ``EntryId``,
  ``SubmissionId``, ``DocumentId`` and ``Seq`` are ``NewType`` brands; ``TaskId[R]`` also carries the
  task's result type for type checkers. Apply a brand at a trusted boundary: ``EntryId(n)`` or
  :func:`pi_durable.ids.id_from_number`.
- **Definitions** (document, entry and task tokens) are slotted classes with snake_case
  attributes, built by ``define_doc``/``define_doc_family``/``define_entry``/``define_task``.
- **Interfaces with behavior** (``Storage``, ``Session``, ``Tx``, ``TaskRuntime``, ``WatchHandle``)
  are ``Protocol`` s with snake_case methods. Methods that return a Promise upstream are
  ``async def``; synchronous ones stay synchronous. ``context`` keeps its upstream position.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    NewType,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    overload,
)

from pi_ai.types import Message
from pi_chord.context import Context
from pi_chord.delta import Draft, Op
from pi_chord.json import JsonObject, JsonValue
from pi_chord.state import AttachedReplicatedState

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

    from pi_ai.abort import AbortSignal
    from pi_ai.models import Models

    from .env import ExecutionEnv
    from .harness.types import Agent, ContextView, ConversationHandle, RegistrySnapshot, Settings, SettledTask

__all__ = [
    "JsonValue",
    "JsonObject",
    # ids
    "Id",
    "ConversationId",
    "EntryId",
    "TaskId",
    "SubmissionId",
    "DocumentId",
    "Seq",
    "ROOT_CONVERSATION_ID",
    # documents: definitions and tokens
    "DocumentScopeName",
    "DocumentHistory",
    "DocumentFork",
    "CheckpointInfo",
    "DocDefinition",
    "DocToken",
    "DocFamilyToken",
    "SessionDocToken",
    "ConversationDocToken",
    "RewindableConversationDocToken",
    "TaskDocToken",
    "SessionDocFamilyToken",
    "ConversationDocFamilyToken",
    "RewindableConversationDocFamilyToken",
    "TaskDocFamilyToken",
    # reports and timers
    "ReportTask",
    "HostCallback",
    "ReportInfo",
    "ReportInfoSessionFailed",
    "ReportInfoTaskFaulted",
    "ReportInfoTaskParked",
    "ReportInfoMigrationFailed",
    "ReportInfoTaskOutdated",
    "ReportInfoTaskError",
    "ReportInfoExtensionError",
    "ReportInfoHostError",
    "ReportInfoListenerError",
    "ReportFrom",
    "ReportFromExtension",
    "ReportFromHost",
    "TimerOptions",
    "SetTimer",
    # tasks
    "RunningTask",
    "NextTaskState",
    "PhaseHandler",
    "AbortHandler",
    "HookRunner",
    "TaskRuntime",
    "TaskMigration",
    "TaskDefinition",
    "Task",
    "AnyTask",
    "TaskOwnership",
    "TaskOwnershipConversation",
    "TaskOwnershipTask",
    "JoinPolicy",
    "TaskOptions",
    "TaskOutcomeError",
    "TaskOutcome",
    "TaskOutcomeCompleted",
    "TaskOutcomeFailed",
    "TaskOutcomeAborted",
    "TaskOutcomeOrphaned",
    "TaskOutcomeFaulted",
    "TaskStatus",
    "TaskState",
    "TaskStatePending",
    "TaskStateRunning",
    "TaskStateWaiting",
    "TaskStateSleeping",
    "TaskStateCompleting",
    "TaskStateTerminal",
    "TaskRecord",
    "AnyTaskRecord",
    # conversations and entries
    "ConversationOwnership",
    "ConversationOwnershipOwnerless",
    "ConversationOwnershipTask",
    "OwnershipOptions",
    "ConversationParent",
    "ConversationOwner",
    "ConversationRecord",
    "ContextEdit",
    "ContextEditOmit",
    "ContextEditReplace",
    "EntryRecord",
    "HeadMarker",
    "EntryDraft",
    "TypedEntry",
    "TypedEntryDraft",
    "Entry",
    # submissions
    "SubmissionStatus",
    "SubmissionRecord",
    "SubmissionRecordInputQueued",
    "SubmissionRecordInputPlaced",
    "SubmissionRecordInputDone",
    "SubmissionRecordInputUnanswered",
    "SubmissionRecordWriteQueued",
    "SubmissionRecordWriteDone",
    "SubmissionRecordWriteUnanswered",
    "SubmissionCreate",
    "SubmissionCreateInputQueued",
    "SubmissionCreateInputPlaced",
    "SubmissionCreateInputDone",
    "SubmissionCreateInputUnanswered",
    "SubmissionCreateWriteQueued",
    "SubmissionCreateWriteDone",
    "SubmissionCreateWriteUnanswered",
    "SubmissionSettlement",
    "SubmissionSettlementDone",
    "SubmissionSettlementUnanswered",
    # document records
    "DocumentScope",
    "DocumentScopeSession",
    "DocumentScopeConversation",
    "DocumentScopeTask",
    "DocumentCreate",
    "DocumentCreateSession",
    "DocumentCreateConversation",
    "DocumentCreateTask",
    "DocumentRecord",
    "DocumentRecordSession",
    "DocumentRecordConversation",
    "DocumentRecordTask",
    # storage
    "Page",
    "Cursor",
    "ScanOrder",
    "ConversationQuery",
    "EntryQuery",
    "TaskQuery",
    "SubmissionQuery",
    "DocumentPoint",
    "DocumentAddress",
    "DocumentQuery",
    "DocumentContent",
    "DocumentContentBase",
    "DocumentContentDelta",
    "DocumentCopySource",
    "StoredDocument",
    "StoredEntry",
    "StorageWrite",
    "StorageWriteConversation",
    "StorageWriteEntry",
    "StorageWriteTask",
    "StorageWriteSubmission",
    "StorageWriteDocumentCreate",
    "StorageWriteDocumentCopy",
    "StorageWriteDocumentChange",
    "StorageWriteDocumentRetire",
    "DocumentCommitChange",
    "DocumentCommitChangeDocument",
    "DocumentCommitChangeCopy",
    "TableCommitChange",
    "CommitChange",
    "CommitPublication",
    # session
    "Tx",
    "DocumentState",
    "WatchEnd",
    "WatchEndStopped",
    "WatchEndListenerError",
    "WatchEndSessionFailed",
    "WatchListener",
    "WatchHandle",
    "DocumentWatch",
    "DocumentReader",
    "DocumentObserver",
    "SessionEnd",
    "SessionEndClosed",
    "SessionEndFailed",
    "CommitListener",
    "Session",
    "Storage",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
D = TypeVar("D")
I = TypeVar("I")  # noqa: E741 - upstream's name for task inputs and family seeds
S = TypeVar("S")
R = TypeVar("R")
H = TypeVar("H")
R_co = TypeVar("R_co", covariant=True)
H_co = TypeVar("H_co", covariant=True)

# ─── IDs ─────────────────────────────────────────────────────────────────────

#: Any durable record ID: a plain ``int`` from the Session-global namespace.
Id: TypeAlias = int
ConversationId = NewType("ConversationId", int)
EntryId = NewType("EntryId", int)
SubmissionId = NewType("SubmissionId", int)
DocumentId = NewType("DocumentId", int)
#: Strictly increasing sequence assigned to one atomic storage commit; gaps are permitted.
Seq = NewType("Seq", int)

if TYPE_CHECKING:

    class TaskId(int, Generic[R_co]):
        """Task ID carrying the task's result type ``R`` for type checkers; a plain ``int`` at runtime."""

else:

    class TaskId(int):
        """Task ID carrying the task's result type ``R`` for type checkers; a plain ``int`` at runtime.

        ``TaskId(4)`` returns the ``int`` 4 and ``TaskId[R]`` is ``TaskId``, like a ``NewType``.
        """

        __slots__ = ()

        def __new__(cls, value: int) -> int:  # type: ignore[misc]
            return int(value)

        def __class_getitem__(cls, item: object) -> type:
            return cls


#: The root conversation always uses this reserved ID.
ROOT_CONVERSATION_ID: Final = ConversationId(1)

# ─── Document definitions and tokens ─────────────────────────────────────────

DocumentScopeName: TypeAlias = Literal["session", "conversation", "task"]
#: ``latest`` retains only current state; ``rewindable`` keeps history for as-of reads.
DocumentHistory: TypeAlias = Literal["latest", "rewindable"]
#: What a fork starts with: the source state at the fork point, its current state, or ``initial``.
DocumentFork: TypeAlias = Literal["asOf", "current", "initial"]


class CheckpointInfo(TypedDict):
    """Stored replay state supplied to a document's checkpoint predicate."""

    #: Deltas already stored after the newest base, excluding the change being evaluated.
    deltasSinceBase: ReadOnly[int]


@dataclass(frozen=True, slots=True, eq=False)
class DocDefinition:
    """Erased definition of a singleton document or a document family (``AnyDocDefinition``).

    Built by :func:`pi_durable.documents.define_doc` and ``define_doc_family``; the Session reads it
    after overload resolution. ``history`` and ``fork`` are set exactly for conversation documents.
    """

    #: Stable persisted kind; part of the public protocol.
    kind: str
    #: Positive integer version of the stored value shape.
    version: int
    scope: DocumentScopeName
    #: ``initial()`` for a singleton, ``initial(seed)`` for a family member.
    initial: Callable[..., Any]
    history: DocumentHistory | None = None
    fork: DocumentFork | None = None
    #: True for a keyed document family.
    family: bool = False
    #: Convert a value stored by an older version: ``migrate(value, from_version)``.
    migrate: Callable[[JsonObject, int], Any] | None = None
    #: Return True to store this ordinary change as a complete base instead of a delta.
    checkpoint_when: Callable[[Any, Sequence[Op], CheckpointInfo], bool] | None = None


class DocToken(Generic[T]):
    """Typed singleton document token passed explicitly to typed access; ``T`` is the value type."""

    __slots__ = ("definition",)

    def __init__(self, definition: DocDefinition) -> None:
        self.definition = definition

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.definition.kind!r})"


class DocFamilyToken(Generic[T, I]):
    """Typed document family token; ``T`` is the value type and ``I`` the ``initial(seed)`` seed type."""

    __slots__ = ("definition",)

    def __init__(self, definition: DocDefinition) -> None:
        self.definition = definition

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.definition.kind!r})"


class SessionDocToken(DocToken[T]):
    __slots__ = ()


class ConversationDocToken(DocToken[T]):
    __slots__ = ()


class RewindableConversationDocToken(ConversationDocToken[T]):
    __slots__ = ()


class TaskDocToken(DocToken[T]):
    __slots__ = ()


class SessionDocFamilyToken(DocFamilyToken[T, I]):
    __slots__ = ()


class ConversationDocFamilyToken(DocFamilyToken[T, I]):
    __slots__ = ()


class RewindableConversationDocFamilyToken(ConversationDocFamilyToken[T, I]):
    __slots__ = ()


class TaskDocFamilyToken(DocFamilyToken[T, I]):
    __slots__ = ()


# ─── Reports and timers ──────────────────────────────────────────────────────


class ReportTask(TypedDict):
    """The task a report is about."""

    id: ReadOnly[TaskId[Any]]
    #: The task's registered name.
    kind: ReadOnly[str]
    conversationId: ReadOnly[ConversationId]


#: A host callback, by its ``HarnessOptions`` name; ``env`` is the environment builder, ``models`` a provider call.
HostCallback: TypeAlias = Literal["now", "settings", "registry", "env", "models", "setTimer", "onWake"]


class ReportInfoSessionFailed(TypedDict):
    type: ReadOnly[Literal["session_failed"]]


class ReportInfoTaskFaulted(TypedDict):
    type: ReadOnly[Literal["task_faulted"]]
    task: ReadOnly[ReportTask]
    extension: ReadOnly[NotRequired[str]]
    reason: ReadOnly[Literal["threw", "restart_limit"]]


class ReportInfoTaskParked(TypedDict):
    type: ReadOnly[Literal["task_parked"]]
    task: ReadOnly[ReportTask]
    extension: ReadOnly[NotRequired[str]]


class ReportInfoMigrationFailed(TypedDict):
    type: ReadOnly[Literal["migration_failed"]]
    task: ReadOnly[ReportTask]
    extension: ReadOnly[NotRequired[str]]


class ReportInfoTaskOutdated(TypedDict):
    type: ReadOnly[Literal["task_outdated"]]
    task: ReadOnly[ReportTask]
    extension: ReadOnly[NotRequired[str]]
    reason: ReadOnly[Literal["missing_task", "incompatible_task"]]


class ReportInfoTaskError(TypedDict):
    type: ReadOnly[Literal["task_error"]]
    task: ReadOnly[ReportTask]
    extension: ReadOnly[NotRequired[str]]


class ReportInfoExtensionError(TypedDict):
    type: ReadOnly[Literal["extension_error"]]
    extension: ReadOnly[NotRequired[str]]
    task: ReadOnly[NotRequired[ReportTask]]
    conversationId: ReadOnly[NotRequired[ConversationId]]


class ReportInfoHostError(TypedDict):
    type: ReadOnly[Literal["host_error"]]
    callback: ReadOnly[HostCallback]
    task: ReadOnly[NotRequired[ReportTask]]


class ReportInfoListenerError(TypedDict):
    type: ReadOnly[Literal["listener_error"]]


#: What a report to ``HarnessOptions.onReport`` is about (see upstream ``ReportInfo`` for each type).
ReportInfo: TypeAlias = (
    ReportInfoSessionFailed
    | ReportInfoTaskFaulted
    | ReportInfoTaskParked
    | ReportInfoMigrationFailed
    | ReportInfoTaskOutdated
    | ReportInfoTaskError
    | ReportInfoExtensionError
    | ReportInfoHostError
    | ReportInfoListenerError
)


class ReportFromExtension(TypedDict):
    type: ReadOnly[Literal["extension"]]
    extension: ReadOnly[NotRequired[str]]


class ReportFromHost(TypedDict):
    type: ReadOnly[Literal["host"]]
    callback: ReadOnly[HostCallback]


#: Whose code a task's ``runtime.report()`` is about when not the task's own.
ReportFrom: TypeAlias = ReportFromExtension | ReportFromHost


class TimerOptions(TypedDict):
    #: Background timers only free memory; a host may decline one by returning ``None``.
    background: ReadOnly[bool]


#: A host timer (``HarnessOptions.setTimer``): call ``callback`` once, about ``delay_ms`` from now, and
#: return a function that disarms it, or ``None`` to decline a background timer. Called on the event loop.
SetTimer: TypeAlias = Callable[[Callable[[], None], float, TimerOptions], "Callable[[], None] | None"]

# ─── Task outcomes, states and records ───────────────────────────────────────


class TaskOutcomeError(TypedDict):
    """JSON-safe error snapshot persisted instead of a runtime exception."""

    message: ReadOnly[str]
    #: Optional structured diagnostic data for inspection or recovery.
    detail: ReadOnly[NotRequired[Any]]


class TaskOutcomeCompleted(TypedDict, Generic[R]):
    status: ReadOnly[Literal["completed"]]
    result: ReadOnly[R]


class TaskOutcomeFailed(TypedDict, Generic[R]):
    """Expected task or domain failure explicitly committed by its implementation."""

    status: ReadOnly[Literal["failed"]]
    error: ReadOnly[TaskOutcomeError]
    result: ReadOnly[NotRequired[R]]


class TaskOutcomeAborted(TypedDict, Generic[R]):
    """Explicit cancellation handled by the task's abort protocol."""

    status: ReadOnly[Literal["aborted"]]
    reason: ReadOnly[NotRequired[str]]
    result: ReadOnly[NotRequired[R]]


class TaskOutcomeOrphaned(TypedDict):
    """Task that cannot resume because its definition or migration is unavailable."""

    status: ReadOnly[Literal["orphaned"]]
    reason: ReadOnly[str]


class TaskOutcomeFaulted(TypedDict):
    """Runtime-detected contract failure, such as an uncaught throw or no durable progress."""

    status: ReadOnly[Literal["faulted"]]
    error: ReadOnly[TaskOutcomeError]


#: Durable reason and optional result recorded when a task becomes terminal.
TaskOutcome: TypeAlias = (
    "TaskOutcomeCompleted[R] | TaskOutcomeFailed[R] | TaskOutcomeAborted[R] | TaskOutcomeOrphaned | TaskOutcomeFaulted"
)

#: How a waiting task treats the tasks it waits on (spec §5.5).
JoinPolicy: TypeAlias = Literal["failFast", "allSettled"]
TaskStatus: TypeAlias = Literal["pending", "running", "waiting", "sleeping", "completing", "terminal"]


class TaskStatePending(TypedDict, Generic[S]):
    """Eligible for scheduling."""

    status: ReadOnly[Literal["pending"]]
    #: Complete durable state from which execution resumes.
    checkpoint: ReadOnly[S]


class TaskStateRunning(TypedDict, Generic[S]):
    """Reserved by one in-memory task invocation."""

    status: ReadOnly[Literal["running"]]
    checkpoint: ReadOnly[S]


class TaskStateWaiting(TypedDict, Generic[S]):
    """Parked without an invocation until every task in ``on`` is terminal; then resumes at ``checkpoint``."""

    status: ReadOnly[Literal["waiting"]]
    checkpoint: ReadOnly[S]
    on: ReadOnly[Sequence[TaskId[Any]]]
    policy: ReadOnly[JoinPolicy]


class TaskStateSleeping(TypedDict, Generic[S]):
    """Parked until the Harness clock reaches ``until`` (epoch milliseconds); then resumes at ``checkpoint``."""

    status: ReadOnly[Literal["sleeping"]]
    checkpoint: ReadOnly[S]
    until: ReadOnly[float]


class TaskStateCompleting(TypedDict, Generic[R]):
    """Outcome decided; becomes terminal once no ordinary owned work below is live. Runs no more code."""

    status: ReadOnly[Literal["completing"]]
    outcome: ReadOnly[TaskOutcome[R]]


class TaskStateTerminal(TypedDict, Generic[R]):
    """Permanently settled durable result receipt."""

    status: ReadOnly[Literal["terminal"]]
    outcome: ReadOnly[TaskOutcome[R]]


#: Complete durable execution state of a task.
TaskState: TypeAlias = (
    TaskStatePending[S]
    | TaskStateRunning[S]
    | TaskStateWaiting[S]
    | TaskStateSleeping[S]
    | TaskStateCompleting[R]
    | TaskStateTerminal[R]
)

#: Next state a task commits for itself: a replacement checkpoint, a wait, a sleep, or its outcome.
NextTaskState: TypeAlias = "TaskStateRunning[S] | TaskStateWaiting[S] | TaskStateSleeping[S] | TaskStateTerminal[R]"


class TaskRecord(TypedDict, Generic[I, S, R]):
    """Complete replacement record for one durable task state machine.

    ``memos`` exists only while the task can run (pending, running, waiting, sleeping); the Session
    drops it when the task completes. Python does not encode that dependency in the type.
    """

    id: ReadOnly[TaskId[R]]
    conversationId: ReadOnly[ConversationId]
    #: Registered task definition name.
    kind: ReadOnly[str]
    #: Definition version used to migrate live input and checkpoints.
    version: ReadOnly[int]
    #: Original task input retained while the task is live or terminal.
    input: ReadOnly[I]
    #: Owning task of a child task; absent for a task its conversation owns. Immutable.
    owner: ReadOnly[NotRequired[TaskId[Any]]]
    #: Whether this conversation-owned task is excluded from ordinary idle waits, aborts, and cascades.
    background: ReadOnly[bool]
    #: Durable abort mark checked before run-mode progress is committed.
    abortRequested: ReadOnly[bool]
    #: ``restart`` for a task abandoned after a restart, or below one.
    abortReason: ReadOnly[NotRequired[Literal["restart"]]]
    #: Set from ``TaskOptions.abandonOnRestart``. Immutable.
    abandonOnRestart: ReadOnly[NotRequired[Literal[True]]]
    #: Wall-clock milliseconds of the first change to ``running``, stamped by the Session.
    startedAt: ReadOnly[NotRequired[float]]
    #: Wall-clock milliseconds of the change to ``terminal``, stamped by the Session.
    endedAt: ReadOnly[NotRequired[float]]
    #: Consecutive restarts that cut the task off while it ran without it changing its state.
    restarts: ReadOnly[NotRequired[int]]
    state: ReadOnly[TaskState[S, R]]
    #: Small first-writer-wins values retained while the task can run.
    memos: ReadOnly[NotRequired[Mapping[str, Any]]]


#: The erased task record storage reads and writes (``TaskRecord<JsonValue, JsonValue, JsonValue>``).
AnyTaskRecord: TypeAlias = "TaskRecord[Any, Any, Any]"


class RunningTask(TaskRecord[I, S, R]):
    """Live task record reserved by one invocation."""

    state: ReadOnly[TaskStateRunning[S]]


class TaskOwnershipConversation(TypedDict):
    kind: ReadOnly[Literal["conversation"]]


class TaskOwnershipTask(TypedDict):
    kind: ReadOnly[Literal["task"]]
    taskId: ReadOnly[TaskId[Any]]


#: Who owns a task: its conversation (a top-level task) or another task of the same conversation.
TaskOwnership: TypeAlias = TaskOwnershipConversation | TaskOwnershipTask


class TaskOptions(TypedDict):
    """Creation options for a durable task."""

    #: Required: a task always names its owner (spec §5.5).
    ownership: ReadOnly[TaskOwnership]
    #: Default: the owner task's conversation, or the transaction's bound conversation.
    conversationId: ReadOnly[NotRequired[ConversationId]]
    #: Conversation-owned tasks only: excluded from ordinary idle waits, conversation aborts, and cascades.
    background: ReadOnly[NotRequired[bool]]
    #: Its creator awaits it only in memory; a later Harness abort-marks it with reason ``restart``.
    abandonOnRestart: ReadOnly[NotRequired[bool]]


class TaskMigration(TypedDict, Generic[I, S]):
    """What ``TaskDefinition.migrate`` returns."""

    input: I
    checkpoint: S


#: Runs one checkpoint phase: ``handler(task, runtime, context)``. It must commit a changed checkpoint or a
#: terminal outcome through ``runtime.commit()``; returning without durable progress faults the task.
PhaseHandler: TypeAlias = "Callable[[RunningTask[I, S, R], TaskRuntime[I, S, R, H], Context], Awaitable[None]]"
#: Runs in a fresh invocation after an abort mark and must commit a terminal outcome.
AbortHandler: TypeAlias = PhaseHandler


@dataclass(frozen=True, slots=True, eq=False)
class TaskDefinition(Generic[I, S, R, H]):
    """Executable durable state machine definition, registered in the registry by ``name``.

    The checkpoint ``S`` is a JSON object with a string ``phase`` key; ``phases`` maps every phase to its
    handler (Python cannot check exhaustiveness). ``H`` is the task's hook table: a ``TypedDict``
    (``total=False``) of handlers keyed by snake_case hook name.
    """

    #: Registered task kind persisted in ``TaskRecord["kind"]``.
    name: str
    #: Definition version persisted with live input and checkpoints.
    version: int
    #: First durable checkpoint for a newly created task.
    initial: Callable[[I], S]
    phases: Mapping[str, PhaseHandler[I, S, R, H]]
    abort: PhaseHandler[I, S, R, H]
    #: Convert a record stored by any older supported version: ``migrate(input, checkpoint, from_version)``.
    migrate: Callable[[Any, Any, int], TaskMigration[I, S]] | None = None
    hooks: H | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Task(Generic[I, S, R, H]):
    """Typed executable task definition (``define_task()``)."""

    definition: TaskDefinition[I, S, R, H]


#: Erased executable task definition stored in the registry.
AnyTask: TypeAlias = "Task[Any, Any, Any, Any]"


class HookRunner(Protocol):
    """Dispatches one hook of a task to every matching registered handler, in registry order."""

    async def each(self, name: str, invoke: Callable[[Any], Awaitable[None] | None]) -> None:
        """Call ``invoke`` with each matching handler named ``name`` (snake_case).

        An ordinary raise from ``invoke`` is reported and the next handler runs; once the invocation is
        signalled, the error propagates.
        """
        ...


# ─── Conversations and entries ───────────────────────────────────────────────


class ConversationOwnershipOwnerless(TypedDict):
    kind: ReadOnly[Literal["ownerless"]]


class ConversationOwnershipTask(TypedDict):
    kind: ReadOnly[Literal["task"]]
    taskId: ReadOnly[TaskId[Any]]


#: Ownership selected explicitly whenever a conversation is created.
ConversationOwnership: TypeAlias = ConversationOwnershipOwnerless | ConversationOwnershipTask


class OwnershipOptions(TypedDict):
    """``{ ownership }``: the options of ``Tx.create_conversation()`` and ``Tx.fork_conversation()``."""

    ownership: ReadOnly[ConversationOwnership]


class ConversationParent(TypedDict):
    """Fork source and inclusive parent entry through which history is inherited."""

    conversationId: ReadOnly[ConversationId]
    at: ReadOnly[EntryId]


class ConversationOwner(TypedDict):
    """Creator edge used for attribution, subtree abort, and subtree idle waits."""

    conversationId: ReadOnly[ConversationId]
    taskId: ReadOnly[TaskId[Any]]


class ConversationRecord(TypedDict):
    """Immutable identity, history ancestry, and task ownership of a transcript scope."""

    id: ReadOnly[ConversationId]
    parent: ReadOnly[NotRequired[ConversationParent]]
    owner: ReadOnly[NotRequired[ConversationOwner]]


class ContextEditOmit(TypedDict):
    #: Entry whose model messages are omitted.
    target: ReadOnly[EntryId]
    action: ReadOnly[Literal["omit"]]


class ContextEditReplace(TypedDict):
    #: Entry whose model messages are replaced.
    target: ReadOnly[EntryId]
    action: ReadOnly[Literal["replace"]]
    #: Messages contributed instead of the target entry's model messages.
    messages: ReadOnly[Sequence[Message]]


#: An immutable override of one visible entry's contribution to model context.
ContextEdit: TypeAlias = ContextEditOmit | ContextEditReplace


class EntryRecord(TypedDict):
    """Immutable transcript event with separate model-facing and application-facing payloads."""

    id: ReadOnly[EntryId]
    conversationId: ReadOnly[ConversationId]
    #: Application-defined entry discriminator.
    kind: ReadOnly[str]
    #: Messages contributed to model context; absent for display or bookkeeping entries.
    model: ReadOnly[NotRequired[Sequence[Message]]]
    #: JSON payload consumed by views, extensions, or bookkeeping logic.
    data: ReadOnly[NotRequired[Any]]
    #: First entry in the active context selected by this entry.
    head: ReadOnly[NotRequired[EntryId]]
    #: Context-only overrides of earlier visible entries.
    edits: ReadOnly[NotRequired[Sequence[ContextEdit]]]
    #: Task that appended this entry, when it was produced by durable work.
    byTaskId: ReadOnly[NotRequired[TaskId[Any]]]


class HeadMarker(EntryRecord):
    """An entry carrying ``head``, as ``find_latest_head_marker()`` returns it."""

    head: ReadOnly[EntryId]


class EntryDraft(TypedDict):
    """Entry content supplied before the Session assigns identity and task attribution."""

    kind: ReadOnly[str]
    model: ReadOnly[NotRequired[Sequence[Message]]]
    data: ReadOnly[NotRequired[Any]]
    #: ``"self"`` starts active context at the newly assigned entry ID.
    head: ReadOnly[NotRequired[EntryId | Literal["self"]]]
    edits: ReadOnly[NotRequired[Sequence[ContextEdit]]]


class TypedEntry(EntryRecord, Generic[D]):
    """Entry whose ``data`` has type ``D``. Kinds without data still type ``data`` as ``D``: do not read it."""

    data: ReadOnly[D]


class TypedEntryDraft(TypedDict, Generic[D]):
    """Entry content of a typed kind; the token supplies ``kind``."""

    model: ReadOnly[NotRequired[Sequence[Message]]]
    data: ReadOnly[NotRequired[D]]
    head: ReadOnly[NotRequired[EntryId | Literal["self"]]]
    edits: ReadOnly[NotRequired[Sequence[ContextEdit]]]


class Entry(Generic[D]):
    """Typed entry kind with a narrowing guard (``define_entry()``).

    ``define_entry()`` returns ``Entry[Any]``; annotate the token to type its data:
    ``Notes: Entry[NoteData] = define_entry("app.note")``.
    """

    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def is_(self, entry: EntryRecord | None) -> TypeGuard[TypedEntry[D]]:
        """Whether ``entry`` exists and has this kind (upstream ``is()``)."""
        return entry is not None and entry["kind"] == self.kind

    def __repr__(self) -> str:
        return f"Entry({self.kind!r})"


# ─── Submissions ─────────────────────────────────────────────────────────────

SubmissionStatus: TypeAlias = Literal["queued", "placed", "done", "unanswered"]


class _SubmissionCreateBase(TypedDict):
    conversationId: ReadOnly[ConversationId]
    #: Host-provided deduplication key, scoped to the conversation.
    requestId: ReadOnly[NotRequired[str]]


class SubmissionCreateInputQueued(_SubmissionCreateBase):
    """Admitted but not yet represented in the transcript."""

    type: ReadOnly[Literal["input"]]
    status: ReadOnly[Literal["queued"]]


class SubmissionCreateInputPlaced(_SubmissionCreateBase):
    """Added to the transcript and owned by an active run."""

    type: ReadOnly[Literal["input"]]
    status: ReadOnly[Literal["placed"]]
    entry: ReadOnly[EntryId]


class SubmissionCreateInputDone(_SubmissionCreateBase):
    """Successfully answered user input."""

    type: ReadOnly[Literal["input"]]
    status: ReadOnly[Literal["done"]]
    entry: ReadOnly[EntryId]
    answer: ReadOnly[EntryId]


class SubmissionCreateInputUnanswered(_SubmissionCreateBase):
    """Terminal input that can no longer receive an answer."""

    type: ReadOnly[Literal["input"]]
    status: ReadOnly[Literal["unanswered"]]
    entry: ReadOnly[NotRequired[EntryId]]
    reason: ReadOnly[str]
    detail: ReadOnly[NotRequired[Any]]


class SubmissionCreateWriteQueued(_SubmissionCreateBase):
    """Admitted but not yet appended to the transcript."""

    type: ReadOnly[Literal["write"]]
    status: ReadOnly[Literal["queued"]]


class SubmissionCreateWriteDone(_SubmissionCreateBase):
    """Successfully appended passive entry."""

    type: ReadOnly[Literal["write"]]
    status: ReadOnly[Literal["done"]]
    entry: ReadOnly[EntryId]


class SubmissionCreateWriteUnanswered(_SubmissionCreateBase):
    """Terminal passive write that could not be placed."""

    type: ReadOnly[Literal["write"]]
    status: ReadOnly[Literal["unanswered"]]
    reason: ReadOnly[str]
    detail: ReadOnly[NotRequired[Any]]


class SubmissionRecordInputQueued(SubmissionCreateInputQueued):
    id: ReadOnly[SubmissionId]


class SubmissionRecordInputPlaced(SubmissionCreateInputPlaced):
    id: ReadOnly[SubmissionId]


class SubmissionRecordInputDone(SubmissionCreateInputDone):
    id: ReadOnly[SubmissionId]


class SubmissionRecordInputUnanswered(SubmissionCreateInputUnanswered):
    id: ReadOnly[SubmissionId]


class SubmissionRecordWriteQueued(SubmissionCreateWriteQueued):
    id: ReadOnly[SubmissionId]


class SubmissionRecordWriteDone(SubmissionCreateWriteDone):
    id: ReadOnly[SubmissionId]


class SubmissionRecordWriteUnanswered(SubmissionCreateWriteUnanswered):
    id: ReadOnly[SubmissionId]


#: Durable lifecycle of one admitted user input or passive entry write.
SubmissionRecord: TypeAlias = (
    SubmissionRecordInputQueued
    | SubmissionRecordInputPlaced
    | SubmissionRecordInputDone
    | SubmissionRecordInputUnanswered
    | SubmissionRecordWriteQueued
    | SubmissionRecordWriteDone
    | SubmissionRecordWriteUnanswered
)

#: Submission fields supplied before the Session assigns an ID.
SubmissionCreate: TypeAlias = (
    SubmissionCreateInputQueued
    | SubmissionCreateInputPlaced
    | SubmissionCreateInputDone
    | SubmissionCreateInputUnanswered
    | SubmissionCreateWriteQueued
    | SubmissionCreateWriteDone
    | SubmissionCreateWriteUnanswered
)


class SubmissionSettlementDone(TypedDict):
    status: ReadOnly[Literal["done"]]
    answer: ReadOnly[EntryId]


class SubmissionSettlementUnanswered(TypedDict):
    status: ReadOnly[Literal["unanswered"]]
    reason: ReadOnly[str]
    detail: ReadOnly[NotRequired[Any]]


#: Terminal status staged for a submission; identity, type, and entry come from its current record.
SubmissionSettlement: TypeAlias = SubmissionSettlementDone | SubmissionSettlementUnanswered

# ─── Document records ────────────────────────────────────────────────────────


class DocumentScopeSession(TypedDict):
    kind: ReadOnly[Literal["session"]]


class DocumentScopeConversation(TypedDict):
    kind: ReadOnly[Literal["conversation"]]
    conversationId: ReadOnly[ConversationId]


class DocumentScopeTask(TypedDict):
    kind: ReadOnly[Literal["task"]]
    taskId: ReadOnly[TaskId[Any]]


DocumentScope: TypeAlias = DocumentScopeSession | DocumentScopeConversation | DocumentScopeTask


class DocumentCreateSession(TypedDict):
    #: Unique incarnation ID; never reused when the same logical document is recreated.
    id: ReadOnly[DocumentId]
    #: Stable document definition kind.
    kind: ReadOnly[str]
    #: Family member key; absent for singleton documents.
    key: ReadOnly[NotRequired[str]]
    scope: ReadOnly[DocumentScopeSession]


class DocumentCreateConversation(TypedDict):
    id: ReadOnly[DocumentId]
    kind: ReadOnly[str]
    key: ReadOnly[NotRequired[str]]
    scope: ReadOnly[DocumentScopeConversation]
    #: ``latest`` pairs only with fork ``current`` or ``initial``.
    history: ReadOnly[DocumentHistory]
    fork: ReadOnly[DocumentFork]


class DocumentCreateTask(TypedDict):
    id: ReadOnly[DocumentId]
    kind: ReadOnly[str]
    key: ReadOnly[NotRequired[str]]
    scope: ReadOnly[DocumentScopeTask]


#: Fields supplied when storage creates and stamps a new ``DocumentRecord``.
DocumentCreate: TypeAlias = DocumentCreateSession | DocumentCreateConversation | DocumentCreateTask


class DocumentRecordSession(DocumentCreateSession):
    #: Commit that created the incarnation, stamped by storage.
    createdAt: ReadOnly[Seq]
    #: Commit that retired the incarnation; absent while it is current.
    retiredAt: ReadOnly[NotRequired[Seq]]


class DocumentRecordConversation(DocumentCreateConversation):
    createdAt: ReadOnly[Seq]
    retiredAt: ReadOnly[NotRequired[Seq]]


class DocumentRecordTask(DocumentCreateTask):
    createdAt: ReadOnly[Seq]
    retiredAt: ReadOnly[NotRequired[Seq]]


#: Persisted lifecycle record for one create-to-retire document incarnation.
DocumentRecord: TypeAlias = DocumentRecordSession | DocumentRecordConversation | DocumentRecordTask

# ─── Storage values ──────────────────────────────────────────────────────────


#: Backend-owned JSON continuation state that callers only round-trip to the same scan.
Cursor: TypeAlias = Mapping[str, JsonValue]


class Page(TypedDict, Generic[T]):
    """One ordered scan result and its optional continuation state (upstream ``Page<T, Cursor>``)."""

    items: ReadOnly[Sequence[T]]
    next: ReadOnly[NotRequired[Cursor]]


#: ID order of a scan: ``ascending`` is oldest first, ``descending`` newest first.
ScanOrder: TypeAlias = Literal["ascending", "descending"]


class ConversationQuery(TypedDict, total=False):
    ownerConversationId: ReadOnly[ConversationId]
    ownerTaskId: ReadOnly[TaskId[Any]]
    #: Default ``ascending``; with a cursor, the cursor's order.
    order: ReadOnly[ScanOrder]


class EntryQuery(TypedDict):
    """Inclusive ID bounds for a scan of one conversation's fork-aware history."""

    conversationId: ReadOnly[ConversationId]
    minEntryId: ReadOnly[NotRequired[EntryId]]
    maxEntryId: ReadOnly[NotRequired[EntryId]]
    #: Default ``descending``; with a cursor, the cursor's order.
    order: ReadOnly[NotRequired[ScanOrder]]


class TaskQuery(TypedDict, total=False):
    conversationId: ReadOnly[ConversationId]
    kind: ReadOnly[str]
    status: ReadOnly[TaskStatus]
    abortRequested: ReadOnly[bool]
    background: ReadOnly[bool]
    #: Default ``ascending``; with a cursor, the cursor's order.
    order: ReadOnly[ScanOrder]


class SubmissionQuery(TypedDict, total=False):
    conversationId: ReadOnly[ConversationId]
    status: ReadOnly[SubmissionStatus]
    #: Default ``ascending``; with a cursor, the cursor's order.
    order: ReadOnly[ScanOrder]


#: Current state or one historical commit sequence used for document membership and content reads.
DocumentPoint: TypeAlias = "Seq | Literal['current']"


class DocumentAddress(TypedDict):
    """Exact logical identity of a singleton or one keyed family member."""

    kind: ReadOnly[str]
    scope: ReadOnly[DocumentScope]
    #: Absent selects the singleton; present selects one family member.
    key: ReadOnly[NotRequired[str]]


class DocumentQuery(TypedDict):
    """Ordered scan of document incarnations alive in one exact scope at one point."""

    scope: ReadOnly[DocumentScope]
    at: ReadOnly[DocumentPoint]
    kind: ReadOnly[NotRequired[str]]


class DocumentContentBase(TypedDict):
    version: ReadOnly[int]
    kind: ReadOnly[Literal["base"]]
    value: ReadOnly[JsonObject]


class DocumentContentDelta(TypedDict):
    version: ReadOnly[int]
    kind: ReadOnly[Literal["delta"]]
    #: Chord ops (tuples in memory, JSON arrays when stored).
    ops: ReadOnly[Sequence[Op]]


#: Complete checkpoint or Chord operation batch selected by the owning Session.
DocumentContent: TypeAlias = DocumentContentBase | DocumentContentDelta


class DocumentCopySource(TypedDict):
    """Exact persisted source selected for a definition-free document copy."""

    id: ReadOnly[DocumentId]
    at: ReadOnly[DocumentPoint]


class StoredDocument(TypedDict):
    """Detached materialized value and stored definition version at a selected point."""

    record: ReadOnly[DocumentRecord]
    version: ReadOnly[int]
    value: ReadOnly[JsonObject]
    #: Deltas replayed after the selected base to materialize ``value``.
    deltasSinceBase: ReadOnly[int]


class StoredEntry(TypedDict):
    """An entry and the sequence of the commit that persisted it (``Storage.entry()``)."""

    entry: ReadOnly[EntryRecord]
    commitSeq: ReadOnly[Seq]


class StorageWriteConversation(TypedDict):
    type: ReadOnly[Literal["conversation"]]
    value: ReadOnly[ConversationRecord]


class StorageWriteEntry(TypedDict):
    type: ReadOnly[Literal["entry"]]
    value: ReadOnly[EntryRecord]


class StorageWriteTask(TypedDict):
    type: ReadOnly[Literal["task"]]
    value: ReadOnly[AnyTaskRecord]


class StorageWriteSubmission(TypedDict):
    type: ReadOnly[Literal["submission"]]
    value: ReadOnly[SubmissionRecord]


class StorageWriteDocumentCreate(TypedDict):
    type: ReadOnly[Literal["document.create"]]
    record: ReadOnly[DocumentCreate]
    content: ReadOnly[DocumentContentBase]


class StorageWriteDocumentCopy(TypedDict):
    type: ReadOnly[Literal["document.copy"]]
    record: ReadOnly[DocumentCreate]
    source: ReadOnly[DocumentCopySource]


class StorageWriteDocumentChange(TypedDict):
    type: ReadOnly[Literal["document.change"]]
    id: ReadOnly[DocumentId]
    content: ReadOnly[DocumentContent]


class StorageWriteDocumentRetire(TypedDict):
    type: ReadOnly[Literal["document.retire"]]
    id: ReadOnly[DocumentId]


#: One record or document mutation in an atomic storage commit.
StorageWrite: TypeAlias = (
    StorageWriteConversation
    | StorageWriteEntry
    | StorageWriteTask
    | StorageWriteSubmission
    | StorageWriteDocumentCreate
    | StorageWriteDocumentCopy
    | StorageWriteDocumentChange
    | StorageWriteDocumentRetire
)


class DocumentCommitChangeDocument(TypedDict):
    """Committed change of one document incarnation."""

    type: ReadOnly[Literal["document"]]
    record: ReadOnly[DocumentRecord]
    #: Conversation owning the document; ``None`` only for Session documents.
    conversationId: ReadOnly[ConversationId | None]
    #: Definition version of ``value``; ``None`` when this commit retired the incarnation.
    version: ReadOnly[int | None]
    #: Exact adopted immutable revision, or ``None`` when this commit retired the incarnation.
    value: ReadOnly[JsonObject | None]
    #: Exact adopted operations for an ordinary update; empty for creation and retirement.
    ops: ReadOnly[Sequence[Op]]


class DocumentCommitChangeCopy(TypedDict):
    """Definition-free child initialization; consumers hydrate through state or watch acquisition."""

    type: ReadOnly[Literal["document.copy"]]
    record: ReadOnly[DocumentRecord]
    conversationId: ReadOnly[ConversationId]
    source: ReadOnly[DocumentCopySource]


DocumentCommitChange: TypeAlias = DocumentCommitChangeDocument | DocumentCommitChangeCopy
#: Complete table record committed without another publication copy.
TableCommitChange: TypeAlias = StorageWriteConversation | StorageWriteEntry | StorageWriteTask | StorageWriteSubmission
CommitChange: TypeAlias = TableCommitChange | DocumentCommitChange


class CommitPublication(TypedDict):
    """Every immutable change from one successful Session commit. Change order is unspecified."""

    seq: ReadOnly[Seq]
    changes: ReadOnly[Sequence[CommitChange]]


# ─── Session surface ─────────────────────────────────────────────────────────


class Tx(Protocol):
    """Transaction surface of one Session commit callback.

    Table reads and creation results are trusted immutable values and may be shared with internal
    commit state. Overloaded methods take positional arguments only, as upstream passes them.
    """

    async def conversation(self, id: ConversationId, /) -> ConversationRecord | None: ...

    @overload
    async def entry(self, id: EntryId, /) -> EntryRecord | None: ...
    @overload
    async def entry(self, token: Entry[D], id: EntryId, /) -> TypedEntry[D] | None:
        """``None`` when the entry is absent or has another kind."""
        ...

    async def task(self, id: TaskId[Any], /) -> AnyTaskRecord | None: ...

    async def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None = None, /
    ) -> Page[ConversationRecord]: ...

    async def scan_entries(
        self, query: EntryQuery, limit: int, cursor: Cursor | None = None, /
    ) -> Page[EntryRecord]: ...

    async def latest_head_marker(self, conversation_id: ConversationId, /) -> HeadMarker | None:
        """Newest visible entry of the conversation that carries a ``head``."""
        ...

    async def scan_tasks(
        self, query: TaskQuery, limit: int, cursor: Cursor | None = None, /
    ) -> Page[AnyTaskRecord]: ...

    async def submission_by_request(
        self, conversation_id: ConversationId, request_id: str, /
    ) -> SubmissionRecord | None:
        """Committed submission with a conversation-scoped request ID."""
        ...

    async def create_conversation(self, options: OwnershipOptions, /) -> ConversationRecord:
        """Create a conversation with explicitly selected ownership: ``{"ownership": ...}``."""
        ...

    async def fork_conversation(
        self, parent_conversation_id: ConversationId, at: EntryId, options: OwnershipOptions, /
    ) -> ConversationRecord:
        """Create a history fork at one concrete visible entry with explicitly selected ownership."""
        ...

    @overload
    async def append_entry(self, conversation_id: ConversationId, value: EntryDraft, /) -> EntryRecord: ...
    @overload
    async def append_entry(
        self, token: Entry[D], conversation_id: ConversationId, value: TypedEntryDraft[D], /
    ) -> TypedEntry[D]:
        """The token supplies ``kind`` and types ``data``."""
        ...

    async def create_task(self, task: Task[I, S, R, H], input: I, options: TaskOptions, /) -> TaskId[R]: ...

    async def create_submission(self, create: SubmissionCreate, /) -> SubmissionRecord:
        """Create a raw submission record with a fresh ID. No admission rules apply."""
        ...

    def settle_submission(self, id: SubmissionId, settlement: SubmissionSettlement, /) -> None:
        """Settle a queued or placed submission against this transaction's latest record of it."""
        ...

    def place_submission(self, id: SubmissionId, entry: EntryId, /) -> None:
        """Place a queued submission at ``entry``: an input becomes ``placed``, a write ``done``."""
        ...

    @overload
    async def doc(self, token: SessionDocToken[T], /) -> Draft[T]: ...
    @overload
    async def doc(self, token: ConversationDocToken[T], conversation_id: ConversationId, /) -> Draft[T]: ...
    @overload
    async def doc(self, token: TaskDocToken[T], task_id: TaskId[Any], /) -> Draft[T]: ...
    @overload
    async def doc(self, token: SessionDocFamilyToken[T, I], key: str, seed: I, /) -> Draft[T]: ...
    @overload
    async def doc(
        self, token: ConversationDocFamilyToken[T, I], conversation_id: ConversationId, key: str, seed: I, /
    ) -> Draft[T]: ...
    @overload
    async def doc(self, token: TaskDocFamilyToken[T, I], task_id: TaskId[Any], key: str, seed: I, /) -> Draft[T]: ...

    @overload
    async def retire_doc(self, token: SessionDocToken[Any], /) -> None: ...
    @overload
    async def retire_doc(self, token: ConversationDocToken[Any], conversation_id: ConversationId, /) -> None: ...
    @overload
    async def retire_doc(self, token: TaskDocToken[Any], task_id: TaskId[Any], /) -> None: ...
    @overload
    async def retire_doc(self, token: SessionDocFamilyToken[Any, Any], key: str, /) -> None: ...
    @overload
    async def retire_doc(
        self, token: ConversationDocFamilyToken[Any, Any], conversation_id: ConversationId, key: str, /
    ) -> None: ...
    @overload
    async def retire_doc(self, token: TaskDocFamilyToken[Any, Any], task_id: TaskId[Any], key: str, /) -> None: ...


#: Disposable, read-only Chord state bound to one committed document incarnation.
DocumentState: TypeAlias = "AttachedReplicatedState[T | None]"


class WatchEndStopped(TypedDict):
    reason: ReadOnly[Literal["stopped", "cancelled", "session_closed", "retired"]]


class WatchEndListenerError(TypedDict):
    reason: ReadOnly[Literal["listener_error"]]
    error: ReadOnly[Exception]


class WatchEndSessionFailed(TypedDict):
    """The Session failed; ``error`` is what failed it, usually a storage error (see ``SessionFailed``)."""

    reason: ReadOnly[Literal["session_failed"]]
    error: ReadOnly[BaseException]


#: Terminal result of one document watch.
WatchEnd: TypeAlias = WatchEndStopped | WatchEndListenerError | WatchEndSessionFailed

#: A watch listener: ``listener(value, ops, context)``.
WatchListener: TypeAlias = "Callable[[T, Sequence[Op], Context], Awaitable[None]]"


class WatchHandle(Protocol[T_co]):
    """Serialized exact-frame observation of an immutable value with bounded pending delivery."""

    @property
    def value(self) -> T_co:
        """Acquisition revision before start; latest delivered immutable revision afterward."""
        ...

    def start(self, listener: Callable[[Any, Sequence[Op], Context], Awaitable[None]]) -> None:
        """Install the sole asynchronous listener. Never invokes it inline."""
        ...

    async def stop(self) -> WatchEnd:
        """Idempotently stop future callbacks and return this watch's terminal result."""
        ...

    @property
    def closed(self) -> Awaitable[WatchEnd]:
        """Settles when the watch terminates (an ``asyncio.Future``, awaitable any number of times)."""
        ...


DocumentWatch: TypeAlias = "WatchHandle[T | None]"


class DocumentReader(Protocol):
    """Committed document reads (``Pick<Session, "snapshot" | "snapshotAsOf">``)."""

    @overload
    async def snapshot(self, token: SessionDocToken[T], context: Context, /) -> T | None: ...
    @overload
    async def snapshot(
        self, token: ConversationDocToken[T], conversation_id: ConversationId, context: Context, /
    ) -> T | None: ...
    @overload
    async def snapshot(self, token: TaskDocToken[T], task_id: TaskId[Any], context: Context, /) -> T | None: ...
    @overload
    async def snapshot(self, token: SessionDocFamilyToken[T, Any], key: str, context: Context, /) -> T | None: ...
    @overload
    async def snapshot(
        self, token: ConversationDocFamilyToken[T, Any], conversation_id: ConversationId, key: str, context: Context, /
    ) -> T | None: ...
    @overload
    async def snapshot(
        self, token: TaskDocFamilyToken[T, Any], task_id: TaskId[Any], key: str, context: Context, /
    ) -> T | None: ...

    @overload
    async def snapshot_as_of(
        self,
        token: RewindableConversationDocToken[T],
        conversation_id: ConversationId,
        at: EntryId,
        context: Context,
        /,
    ) -> T | None: ...
    @overload
    async def snapshot_as_of(
        self,
        token: RewindableConversationDocFamilyToken[T, Any],
        conversation_id: ConversationId,
        key: str,
        at: EntryId,
        context: Context,
        /,
    ) -> T | None: ...


class DocumentObserver(Protocol):
    """Non-creating document watch acquisition shared by Session and later invocation APIs."""

    @overload
    async def watch_doc(self, token: SessionDocToken[T], context: Context, /) -> WatchHandle[T | None] | None: ...
    @overload
    async def watch_doc(
        self, token: ConversationDocToken[T], conversation_id: ConversationId, context: Context, /
    ) -> WatchHandle[T | None] | None: ...
    @overload
    async def watch_doc(
        self, token: TaskDocToken[T], task_id: TaskId[Any], context: Context, /
    ) -> WatchHandle[T | None] | None: ...
    @overload
    async def watch_doc(
        self, token: SessionDocFamilyToken[T, Any], key: str, context: Context, /
    ) -> WatchHandle[T | None] | None: ...
    @overload
    async def watch_doc(
        self, token: ConversationDocFamilyToken[T, Any], conversation_id: ConversationId, key: str, context: Context, /
    ) -> WatchHandle[T | None] | None: ...
    @overload
    async def watch_doc(
        self, token: TaskDocFamilyToken[T, Any], task_id: TaskId[Any], key: str, context: Context, /
    ) -> WatchHandle[T | None] | None: ...


class SessionEndClosed(TypedDict):
    reason: ReadOnly[Literal["closed"]]


class SessionEndFailed(TypedDict):
    reason: ReadOnly[Literal["failed"]]
    error: ReadOnly[BaseException]


#: Why a Session ended: ``close()``, or a failed Storage call, whose error it carries.
SessionEnd: TypeAlias = SessionEndClosed | SessionEndFailed

#: A commit listener: ``listener(publication, context)``; synchronous, must not block or call Session APIs.
CommitListener: TypeAlias = Callable[[CommitPublication, Context], None]


class Session(DocumentObserver, DocumentReader, Protocol):
    """Owner of one mutation line, its records, and its tracked documents."""

    async def commit(self, change: Callable[[Tx], T | Awaitable[T]], context: Context) -> T:
        """Run one atomic transaction on the Session mutation line.

        Inside the callback read only through ``tx``: other Session reads may queue on the line the
        callback holds and then never resolve. ``change`` may be sync or async.
        """
        ...

    async def close(self, context: Context) -> None:
        """Seal admission, settle admitted commits, then close storage."""
        ...

    @property
    def closed(self) -> Awaitable[SessionEnd]:
        """Settles once the Session has closed, Storage included (an ``asyncio.Future``)."""
        ...

    def subscribe_commits(self, listener: CommitListener) -> Callable[[], None]:
        """Observe complete commits synchronously after adoption; returns the unsubscribe function."""
        ...

    def subscribe_close(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Observe close synchronously when it begins, also when a failed Storage call closes the Session."""
        ...

    @overload
    async def document_state(
        self, token: SessionDocToken[T], context: Context, /
    ) -> AttachedReplicatedState[T | None] | None: ...
    @overload
    async def document_state(
        self, token: ConversationDocToken[T], conversation_id: ConversationId, context: Context, /
    ) -> AttachedReplicatedState[T | None] | None: ...
    @overload
    async def document_state(
        self, token: TaskDocToken[T], task_id: TaskId[Any], context: Context, /
    ) -> AttachedReplicatedState[T | None] | None: ...
    @overload
    async def document_state(
        self, token: SessionDocFamilyToken[T, Any], key: str, context: Context, /
    ) -> AttachedReplicatedState[T | None] | None: ...
    @overload
    async def document_state(
        self, token: ConversationDocFamilyToken[T, Any], conversation_id: ConversationId, key: str, context: Context, /
    ) -> AttachedReplicatedState[T | None] | None: ...
    @overload
    async def document_state(
        self, token: TaskDocFamilyToken[T, Any], task_id: TaskId[Any], key: str, context: Context, /
    ) -> AttachedReplicatedState[T | None] | None: ...


class TaskRuntime(DocumentObserver, DocumentReader, Protocol[I, S, R, H_co]):
    """Operations of one task invocation. Every operation raises after the invocation ends."""

    @property
    def task_id(self) -> TaskId[R]: ...
    @property
    def conversation_id(self) -> ConversationId: ...
    @property
    def signal(self) -> AbortSignal:
        """Aborted when the run is signalled by ``abort_task()``, the Harness closes, or the invocation ends."""
        ...

    @property
    def registry(self) -> RegistrySnapshot:
        """Registry snapshot of the current phase; refreshed at every phase boundary."""
        ...

    async def agent(self, context: Context) -> Agent:
        """The task's conversation's agent, resolved at most once per phase, at first use."""
        ...

    @property
    def settings(self) -> Settings:
        """``HarnessOptions.settings``, resolved at each access."""
        ...

    @property
    def models(self) -> Models:
        """``HarnessOptions.models``, resolved at each access."""
        ...

    async def env(self, context: Context) -> ExecutionEnv | None:
        """Calls ``HarnessOptions.env`` for the task's conversation; raises its error."""
        ...

    @property
    def hooks(self) -> HookRunner:
        """Handlers of this task's name from the extensions its conversation selects, in extension order."""
        ...

    async def commit(
        self,
        change: Callable[
            [Tx, RunningTask[I, S, R]],
            NextTaskState[S, R] | Awaitable[NextTaskState[S, R] | None] | None,
        ],
        context: Context,
    ) -> None:
        """Commit on the Session line after rereading the task; a returned state replaces the task's state."""
        ...

    @overload
    async def memo(self, name: str, context: Context, /) -> Any:
        """Read a durable memo of this task (``None`` when absent)."""
        ...

    @overload
    async def memo(self, name: str, candidate: T, context: Context, /) -> T:
        """Store ``candidate`` unless a memo already exists; return the durable winner."""
        ...

    async def get_task(self, id: TaskId[T], context: Context) -> TaskRecord[Any, Any, T] | None: ...

    async def wait_for_task(self, id: TaskId[T], context: Context) -> SettledTask[T]:
        """Resolve with the task's terminal receipt; raises when the invocation ends."""
        ...

    async def abort_owned(self, id: TaskId[Any], context: Context) -> None:
        """Abort a task this task owns and resolve once it is terminal."""
        ...

    async def owned_tasks(self, context: Context) -> list[AnyTaskRecord]:
        """Live tasks this task owns directly, in ID order."""
        ...

    async def outcomes(self, ids: Sequence[TaskId[T]], context: Context) -> list[TaskOutcome[T]]:
        """Outcomes of terminal tasks, in order; raises when one is missing or not terminal."""
        ...

    async def conversation(self, id: ConversationId, context: Context) -> ConversationHandle | None:
        """Invocation-bound handle of an existing conversation; ``None`` when absent."""
        ...

    @overload
    async def entry(self, id: EntryId, context: Context, /) -> EntryRecord | None: ...
    @overload
    async def entry(self, token: Entry[D], id: EntryId, context: Context, /) -> TypedEntry[D] | None: ...

    async def context(
        self, conversation_id: ConversationId, context: Context, *, at: EntryId | None = None
    ) -> ContextView:
        """Committed raw active transcript and model context, optionally cut off at the visible entry ``at``."""
        ...

    def now(self) -> float:
        """The Harness clock, epoch milliseconds."""
        ...

    def set_timer(self, callback: Callable[[], None], delay_ms: float) -> Callable[[], None]:
        """Call ``callback`` once after about ``delay_ms`` on the Harness's timer; not durable."""
        ...

    def report(self, error: BaseException, from_: ReportFrom | None = None) -> None:
        """Forward a non-fatal failure to ``HarnessOptions.onReport`` with this task in its info."""
        ...

    async def sleep_until(self, until: float, checkpoint: S, context: Context) -> None:
        """Commit a sleep until the Harness clock reaches ``until``, then resume at ``checkpoint``."""
        ...


class Storage(Protocol):
    """Atomic persistence boundary for Session records.

    Storage trusts the owning Session to supply semantically valid records, references, ancestry,
    and transitions. Implementations enforce atomicity, global ID ownership, immutable
    conversation/entry creation, document record consistency, and detached values; the Session
    serializes commits.

    An error a method raises is final: it fails the Session, which nothing retries, so retry
    transient failures inside the method. Two errors of a read fail only that read: a
    :class:`~pi_durable.errors.StorageRequestError` for an invalid request (with no durable effect),
    and a cancellation because its caller's context was aborted. Any error from ``commit()`` or
    ``mint_id()`` is fatal.
    """

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        """Atomically persist one batch and return its sequence; later reads observe it."""
        ...

    async def mint_id(self) -> int:
        """A fresh ID from the Session-global namespace, greater than every ID minted or stored before."""
        ...

    async def conversation(self, id: ConversationId, context: Context) -> ConversationRecord | None:
        """Look up one conversation by exact ID."""
        ...

    async def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[ConversationRecord]:
        """Scan conversations in ``query["order"]`` (default ascending) by ID."""
        ...

    async def entry(self, id: EntryId, context: Context) -> StoredEntry | None:
        """Look up one global entry and the sequence of the commit that persisted it (``entry(id)``)."""
        ...

    async def visible_entry(self, conversation_id: ConversationId, id: EntryId, context: Context) -> StoredEntry | None:
        """Upstream ``entry(conversationId, id)``: the entry only when visible through the conversation's ancestry."""
        ...

    async def find_latest_head_marker(
        self, conversation_id: ConversationId, at_or_before_entry_id: EntryId | None, context: Context
    ) -> HeadMarker | None:
        """The newest visible entry with a ``head`` at or below the optional inclusive cutoff."""
        ...

    async def scan_entries(
        self, query: EntryQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[EntryRecord]:
        """Scan the inclusive visible range in ``query["order"]`` (default descending)."""
        ...

    async def task(self, id: TaskId[Any], context: Context) -> AnyTaskRecord | None:
        """Look up the latest complete record for one task."""
        ...

    async def scan_tasks(
        self, query: TaskQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[AnyTaskRecord]:
        """Scan task records matching every supplied filter in ``query["order"]`` (default ascending)."""
        ...

    async def submission(self, id: SubmissionId, context: Context) -> SubmissionRecord | None:
        """Look up the latest complete record for one admitted submission."""
        ...

    async def scan_submissions(
        self, query: SubmissionQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[SubmissionRecord]:
        """Scan submissions matching every supplied filter in ``query["order"]`` (default ascending)."""
        ...

    async def submission_by_request(
        self, conversation_id: ConversationId, request_id: str, context: Context
    ) -> SubmissionRecord | None:
        """Find a submission by its conversation-scoped host deduplication key."""
        ...

    async def find_document(
        self, address: DocumentAddress, at: DocumentPoint, context: Context
    ) -> DocumentRecord | None:
        """Resolve the incarnation occupying one exact logical address at the selected point."""
        ...

    async def document(self, id: DocumentId, at: DocumentPoint, context: Context) -> StoredDocument | None:
        """Materialize one specific incarnation by ID at the selected point."""
        ...

    async def scan_documents(
        self, query: DocumentQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[DocumentRecord]:
        """Scan incarnations alive in one exact scope at the selected point, by ascending ID."""
        ...

    async def close(self, context: Context) -> None:
        """Release backend resources; all later operations must raise."""
        ...
