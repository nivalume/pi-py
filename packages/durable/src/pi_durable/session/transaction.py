"""The transaction of one Session commit callback (port of ``session/transaction.ts``).

Every ``Tx`` operation that upstream returns as a Promise starts at the call and returns an
``asyncio`` future: its synchronous part (sealing checks, the read-after-write rule, document entry
registration) runs before it returns, so call order, not await order, decides what a commit stages,
exactly as upstream. Every such operation is tracked so callback settlement can reject and drain
unfinished work.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypedDict, TypeVar

from pi_chord.context import Context
from pi_chord.delta import Change, Op, Prepared, Tracker, track
from pi_chord.json import UNDEFINED, JsonObject, copy_json

from ..documents import (
    AnyDocDefinition,
    AnyDocToken,
    ResolvedAddress,
    address_id,
    check_record_scope,
    check_record_version,
    document_create,
    materialize_document_value,
    resolve_address,
)
from ..errors import ReadAfterWrite
from ..storage.codec import clone_value
from ..types import (
    ROOT_CONVERSATION_ID,
    AnyTaskRecord,
    ConversationId,
    ConversationOwner,
    ConversationOwnership,
    ConversationParent,
    ConversationQuery,
    ConversationRecord,
    Cursor,
    DocumentAddress,
    DocumentCommitChange,
    DocumentCopySource,
    DocumentCreate,
    DocumentId,
    DocumentRecord,
    Entry,
    EntryDraft,
    EntryId,
    EntryQuery,
    EntryRecord,
    HeadMarker,
    OwnershipOptions,
    Page,
    Seq,
    Storage,
    StorageWrite,
    StorageWriteDocumentChange,
    SubmissionCreate,
    SubmissionId,
    SubmissionRecord,
    SubmissionSettlement,
    Task,
    TaskId,
    TaskOptions,
    TaskQuery,
    Tx,
)
from ._futures import rejected, resolved
from .forks import ForkDocumentCopy, prepare_fork_document_copies

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

__all__ = ["LoadedDocument", "Transaction", "TransactionHost", "TransactionScope"]

T = TypeVar("T")

_INTERNAL_SCAN_PAGE_SIZE: Final = 256
#: Ops of a creation or retirement publication (immutable by contract).
_EMPTY_OPERATIONS: Final[list[Op]] = []


# ─── Submissions ─────────────────────────────────────────────────────────────


def _apply_submission_change(current: SubmissionRecord, change: Mapping[str, Any]) -> SubmissionRecord:
    """Complete record after one change: a settlement, or the placement ``{"status": "placed", "entry": id}``.

    Placement turns a queued input ``placed`` and a queued write ``done``; only a placed input can be
    answered. A settled record stays.
    """
    if current["status"] == "done" or current["status"] == "unanswered":
        return current
    if change["status"] == "placed":
        if current["status"] != "queued":
            raise RuntimeError(f"Submission {current['id']} is not queued")
        status = "placed" if current["type"] == "input" else "done"
        return {**current, "status": status, "entry": change["entry"]}  # type: ignore[return-value]
    if change["status"] == "done" and current["status"] != "placed":
        raise RuntimeError(f"Submission {current['id']} is not a placed input")
    # Queued and placed records carry no answer, reason, or detail; an unanswered input keeps its entry.
    return {**current, **change}  # type: ignore[return-value]


# ─── Session services and cached documents ───────────────────────────────────


class LoadedDocument:
    """One committed document incarnation owned by the Session tracker cache."""

    __slots__ = ("address_id", "deltas_since_base", "record", "stored_version", "tracker", "value_version")

    def __init__(
        self,
        address_id: str,
        record: DocumentRecord,
        stored_version: int,
        value_version: int,
        deltas_since_base: int,
        tracker: Tracker[JsonObject],
    ) -> None:
        self.address_id = address_id
        self.record = record
        #: Persisted definition version; older while the tracked value is migrated only in memory.
        self.stored_version = stored_version
        #: Definition version whose shape the tracked value has; access with another version reloads.
        self.value_version = value_version
        #: Stored deltas after the newest base; advanced by adoption so the next predicate call needs no read.
        self.deltas_since_base = deltas_since_base
        self.tracker = tracker


class TransactionHost(Protocol):
    """Session services used by a transaction while it holds the mutation line."""

    @property
    def storage(self) -> Storage: ...

    def now(self) -> float:
        """Wall clock for task lifecycle times."""
        ...

    def cached(self, address_id: str) -> LoadedDocument | None:
        """The cached current incarnation, without loading."""
        ...

    async def load(
        self, definition: AnyDocDefinition, address_id: str, address: DocumentAddress, context: Context
    ) -> LoadedDocument | None:
        """The cached current incarnation, cold-loading and migrating it when necessary."""
        ...

    def install(self, document: LoadedDocument) -> None:
        """Install a newly committed incarnation."""
        ...

    def evict(self, address_id: str, record_id: DocumentId) -> None:
        """Remove a retired incarnation if it is still the cached occupant of its address."""
        ...

    async def conversation_created(self, tx: Transaction, record: ConversationRecord) -> None:
        """Stage writes that belong to every newly created or forked conversation, in its creating transaction."""
        ...


class TransactionScope(TypedDict, total=False):
    """Defaults a commit binds to: the ``tx.create_task()`` conversation and the task attributed to appended entries."""

    conversationId: ReadOnly[ConversationId]
    #: Task whose runtime commit this is; stamped as ``byTaskId`` on appended entries.
    taskId: ReadOnly[TaskId[Any]]


class _TaskWrite:
    __slots__ = ("kind", "record")

    def __init__(self, kind: Literal["create", "replace"], record: AnyTaskRecord) -> None:
        self.kind = kind
        self.record = record


class _TransactionTask:
    """Committed and candidate state for one task touched by this transaction."""

    __slots__ = ("committed_read", "publication_conversation_id", "write")

    def __init__(self) -> None:
        self.committed_read: asyncio.Future[AnyTaskRecord | None] | None = None
        self.write: _TaskWrite | None = None
        self.publication_conversation_id: ConversationId | None = None


# Storage/cache provenance of one staged document incarnation.


class _LoadedTarget:
    __slots__ = ("document",)
    kind: Final = "loaded"

    def __init__(self, document: LoadedDocument) -> None:
        self.document = document


class _CreatedTarget:
    __slots__ = ("record", "tracker", "version")
    kind: Final = "created"

    def __init__(self, record: DocumentCreate, version: int, tracker: Tracker[JsonObject]) -> None:
        self.record = record
        self.version = version
        self.tracker = tracker


class _ForkCopyTarget:
    __slots__ = ("record", "source")
    kind: Final = "fork-copy"

    def __init__(self, record: DocumentCreate, source: DocumentCopySource) -> None:
        self.record = record
        self.source = source


class _RetireOnlyTarget:
    __slots__ = ("record",)
    kind: Final = "retire-only"

    def __init__(self, record: DocumentRecord) -> None:
        self.record = record


_DocumentTarget = _LoadedTarget | _CreatedTarget | _ForkCopyTarget | _RetireOnlyTarget


class _DocumentEntry:
    """One document incarnation acquired, created, or retired by this transaction."""

    __slots__ = ("address", "address_id", "change", "definition", "draft", "prepared", "retire_on_commit", "target")

    def __init__(
        self,
        address_id: str,
        address: DocumentAddress,
        definition: AnyDocDefinition | None,
        retire_on_commit: bool,
        target: _DocumentTarget | None = None,
    ) -> None:
        self.address_id = address_id
        self.address = address
        #: Absent for definition-free fork copies and retirement entries discovered by a terminal-task scan.
        self.definition = definition
        #: Memoized public acquisition; absent for metadata-only retirement.
        self.draft: asyncio.Future[Any] | None = None
        #: Set after acquisition or retirement lookup finds the affected incarnation.
        self.target = target
        self.change: Change[JsonObject] | None = None
        self.prepared: Prepared[JsonObject] | None = None
        self.retire_on_commit = retire_on_commit


class _PlanChange:
    """Prepared change of a tracked incarnation."""

    __slots__ = ("definition", "loaded", "prepared", "tracker", "version")

    def __init__(
        self,
        tracker: Tracker[JsonObject],
        prepared: Prepared[JsonObject],
        version: int,
        loaded: LoadedDocument | None = None,
        definition: AnyDocDefinition | None = None,
    ) -> None:
        self.tracker = tracker
        self.prepared = prepared
        self.version = version
        #: The cached incarnation this change updates; absent when it creates one.
        self.loaded = loaded
        self.definition = definition


class _DocumentPlan:
    """What one staged incarnation writes and publishes, decided before Storage admission.

    ``record`` is a ``DocumentRecord`` for an already committed incarnation and a ``DocumentCreate``
    for a new one.
    """

    __slots__ = ("address_id", "change", "content", "conversation_id", "record", "retire")

    def __init__(
        self,
        address_id: str,
        record: DocumentCreate | DocumentRecord,
        retire: bool,
        content: StorageWrite | None = None,
        change: _PlanChange | None = None,
    ) -> None:
        self.address_id = address_id
        self.record = record
        self.retire = retire
        #: Creation, copy, or change content; absent when only retirement is written.
        self.content = content
        #: Absent for fork copies and retirement-only entries.
        self.change = change
        #: Resolved before Storage admission so adoption performs no reads.
        self.conversation_id: ConversationId | None = None


# ─── Transaction ─────────────────────────────────────────────────────────────


class Transaction:
    """Transaction for one Session commit callback (implements :class:`pi_durable.types.Tx`).

    The Session calls one settlement method, then either discards prepared changes or adopts them once
    after Storage succeeds. Methods beyond ``Tx`` (``create_root_conversation``, ``submission``,
    ``set_task``, ``staged_tasks``, ``staged_conversations``) are internal to the Session and Harness.
    """

    __slots__ = (
        "_context",
        "_created_conversation_ids",
        "_documents",
        "_fork_source_conversation_ids",
        "_fork_source_document_ids",
        "_has_table_write",
        "_host",
        "_latest_document_by_address",
        "_pending",
        "_plans",
        "_scope",
        "_sealed",
        "_submission_changes",
        "_submissions",
        "_tasks_by_id",
        "_writes",
    )

    def __init__(self, host: TransactionHost, context: Context, scope: TransactionScope | None = None) -> None:
        self._host = host
        self._context = context
        self._scope: TransactionScope = {} if scope is None else scope
        self._pending: set[asyncio.Future[Any]] = set()
        self._sealed = False
        self._has_table_write = False
        #: Atomic batch; conversation and entry writes stage eagerly, task and document writes assemble later.
        self._writes: list[StorageWrite] = []
        self._created_conversation_ids: set[ConversationId] = set()
        self._fork_source_conversation_ids: set[ConversationId] = set()
        self._fork_source_document_ids: set[DocumentId] = set()
        #: One entry per task touched by a public read, candidate write, or document-owner lookup.
        self._tasks_by_id: dict[TaskId[Any], _TransactionTask] = {}
        #: Submissions created by this transaction, by ID.
        self._submissions: dict[SubmissionId, SubmissionRecord] = {}
        #: Submission settlements and placements in staging order, resolved against the latest record at assembly.
        self._submission_changes: list[tuple[SubmissionId, Mapping[str, Any]]] = []
        #: Write and publication plans of every staged incarnation, built during assembly.
        self._plans: list[_DocumentPlan] = []
        #: Every document acquisition or retirement marker in staging order.
        self._documents: list[_DocumentEntry] = []
        #: Latest transaction-local incarnation or retirement marker at each logical address.
        self._latest_document_by_address: dict[str, _DocumentEntry] = {}

    # ─── Table reads ────────────────────────────────────────────────────────

    def conversation(self, id: ConversationId, /) -> asyncio.Future[ConversationRecord | None]:
        return self._read("conversation", lambda: self._host.storage.conversation(id, self._context))

    def entry(self, first: Any, second: EntryId | None = None, /) -> asyncio.Future[Any]:
        """``entry(id)``, or ``entry(token, id)``: ``None`` when absent or of another kind."""
        kind = first.kind if isinstance(first, Entry) else None
        id: EntryId = second if isinstance(first, Entry) else first  # type: ignore[assignment]

        async def read() -> EntryRecord | None:
            stored = await self._host.storage.entry(id, self._context)
            entry = None if stored is None else stored["entry"]
            return entry if kind is None or (entry is not None and entry["kind"] == kind) else None

        return self._read("entry", read)

    def task(self, id: TaskId[Any], /) -> asyncio.Future[AnyTaskRecord | None]:
        return self._read("task", lambda: self._committed_task(id))

    def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None = None, /
    ) -> asyncio.Future[Page[ConversationRecord]]:
        return self._read(
            "scan_conversations", lambda: self._host.storage.scan_conversations(query, limit, cursor, self._context)
        )

    def scan_entries(
        self, query: EntryQuery, limit: int, cursor: Cursor | None = None, /
    ) -> asyncio.Future[Page[EntryRecord]]:
        return self._read("scan_entries", lambda: self._host.storage.scan_entries(query, limit, cursor, self._context))

    def latest_head_marker(self, conversation_id: ConversationId, /) -> asyncio.Future[HeadMarker | None]:
        return self._read(
            "latest_head_marker",
            lambda: self._host.storage.find_latest_head_marker(conversation_id, None, self._context),
        )

    def scan_tasks(
        self, query: TaskQuery, limit: int, cursor: Cursor | None = None, /
    ) -> asyncio.Future[Page[AnyTaskRecord]]:
        return self._read("scan_tasks", lambda: self._host.storage.scan_tasks(query, limit, cursor, self._context))

    def submission(self, id: SubmissionId, /) -> asyncio.Future[SubmissionRecord | None]:
        """Internal: committed submission record."""
        return self._read("submission", lambda: self._host.storage.submission(id, self._context))

    def submission_by_request(
        self, conversation_id: ConversationId, request_id: str, /
    ) -> asyncio.Future[SubmissionRecord | None]:
        """Committed submission with a conversation-scoped request ID."""
        return self._read(
            "submission_by_request",
            lambda: self._host.storage.submission_by_request(conversation_id, request_id, self._context),
        )

    # ─── Table writes ───────────────────────────────────────────────────────

    def create_conversation(self, options: OwnershipOptions, /) -> asyncio.Future[ConversationRecord]:
        return self._write(lambda: self._stage_conversation(None, options["ownership"]))

    def create_root_conversation(self) -> asyncio.Future[ConversationRecord]:
        """Internal final-form bootstrap path for the reserved root identity."""
        return self._write(lambda: self._stage_conversation(None, {"kind": "ownerless"}, ROOT_CONVERSATION_ID))

    def fork_conversation(
        self, parent_conversation_id: ConversationId, at: EntryId, options: OwnershipOptions, /
    ) -> asyncio.Future[ConversationRecord]:
        return self._write(
            lambda: self._stage_conversation({"conversationId": parent_conversation_id, "at": at}, options["ownership"])
        )

    async def _stage_conversation(
        self,
        parent: ConversationParent | None,
        ownership: ConversationOwnership,
        reserved_id: ConversationId | None = None,
    ) -> ConversationRecord:
        owner_task_id = ownership["taskId"] if ownership["kind"] == "task" else None
        id = reserved_id if reserved_id is not None else ConversationId(await self._host.storage.mint_id())
        self._assert_open()
        owner: ConversationOwner | None = None
        if owner_task_id is not None:
            task = await self._current_task(owner_task_id)
            self._assert_open()
            if task is None:
                raise RuntimeError(f"Conversation owner task {owner_task_id} does not exist")
            owner = {"conversationId": task["conversationId"], "taskId": owner_task_id}
        record: dict[str, Any] = {"id": id}
        if parent is not None:
            record["parent"] = parent
        if owner is not None:
            record["owner"] = owner
        conversation: ConversationRecord = record  # type: ignore[assignment]
        copies: list[ForkDocumentCopy] = (
            []
            if parent is None
            else await prepare_fork_document_copies(
                self._host.storage, parent["conversationId"], parent["at"], id, self._context
            )
        )
        self._assert_open()
        for copy in copies:
            self._fork_source_document_ids.add(copy.source["id"])
            entry = _DocumentEntry(
                address_id(copy.record),  # type: ignore[arg-type]
                copy.record,  # type: ignore[arg-type]
                None,
                False,
                _ForkCopyTarget(copy.record, copy.source),
            )
            self._documents.append(entry)
            self._latest_document_by_address[entry.address_id] = entry
        if parent is not None:
            self._fork_source_conversation_ids.add(parent["conversationId"])
        self._created_conversation_ids.add(id)
        self._writes.append({"type": "conversation", "value": conversation})
        await self._host.conversation_created(self, conversation)
        self._assert_open()
        return conversation

    def append_entry(self, first: Any, second: Any, third: Any = None, /) -> asyncio.Future[Any]:
        """``append_entry(conversation_id, draft)``, or ``append_entry(token, conversation_id, draft)``."""
        if isinstance(first, Entry):
            return self._append_entry(second, {**(third or {}), "kind": first.kind})  # type: ignore[typeddict-item]
        return self._append_entry(first, second)

    def _append_entry(self, conversation_id: ConversationId, value: EntryDraft) -> asyncio.Future[EntryRecord]:
        async def write() -> EntryRecord:
            await self._require_conversation(conversation_id)
            self._assert_open()
            id = EntryId(await self._host.storage.mint_id())
            self._assert_open()
            head = value.get("head", UNDEFINED)
            record: dict[str, Any] = {key: item for key, item in value.items() if key != "head"}
            record["id"] = id
            record["conversationId"] = conversation_id
            record["head"] = id if head == "self" else head
            record["byTaskId"] = self._scope.get("taskId", UNDEFINED)
            # Undefined `head` and `byTaskId` are omitted by the copy.
            entry: EntryRecord = _copy_table_json(record)
            self._writes.append({"type": "entry", "value": entry})
            return entry

        return self._write(write)

    def create_task(self, task: Task[Any, Any, Any, Any], input: Any, options: TaskOptions, /) -> asyncio.Future[Any]:
        async def write() -> TaskId[Any]:
            ownership = options["ownership"]
            owner: AnyTaskRecord | None = None
            if ownership["kind"] == "task":
                # Validated again against the owner's final candidate during assembly.
                owner = await self._current_task(ownership["taskId"])
                self._assert_open()
                if owner is None:
                    raise RuntimeError(f"Task owner {ownership['taskId']} does not exist")
                if options.get("background") is True:
                    raise TypeError("A child task cannot be background")
                requested = options.get("conversationId")
                if requested is not None and requested != owner["conversationId"]:
                    raise RuntimeError(f"A child task lives in its owner's conversation {owner['conversationId']}")
            conversation_id = _first_present(
                None if owner is None else owner["conversationId"],
                options.get("conversationId"),
                self._scope.get("conversationId"),
            )
            if conversation_id is None:
                raise TypeError("Tx.create_task() requires options.conversationId")
            await self._require_conversation(conversation_id)
            self._assert_open()
            definition = task.definition
            checkpoint = definition.initial(input)
            id: TaskId[Any] = TaskId(await self._host.storage.mint_id())
            self._assert_open()
            record: dict[str, Any] = {
                "id": id,
                "conversationId": conversation_id,
                "kind": definition.name,
                "version": definition.version,
                "input": input,
            }
            if owner is not None:
                record["owner"] = owner["id"]
            record["background"] = options.get("background", False)
            record["abortRequested"] = False
            if options.get("abandonOnRestart") is True:
                record["abandonOnRestart"] = True
            record["state"] = {"status": "pending", "checkpoint": checkpoint}
            entry = self._task_entry(id)
            entry.write = _TaskWrite("create", _copy_table_json(record))
            return id

        return self._write(write)

    def create_submission(self, create: SubmissionCreate, /) -> asyncio.Future[SubmissionRecord]:
        """Create a raw submission record with a fresh ID; no admission rules apply."""

        async def write() -> SubmissionRecord:
            await self._require_conversation(create["conversationId"])
            self._assert_open()
            id = SubmissionId(await self._host.storage.mint_id())
            self._assert_open()
            record: SubmissionRecord = _copy_table_json({**create, "id": id})
            self._submissions[id] = record
            return record

        return self._write(write)

    def settle_submission(self, id: SubmissionId, settlement: SubmissionSettlement, /) -> None:
        """Settle a submission, resolved at assembly against the latest candidate record, else committed state.

        Not a caller table read, so it works after the first table write.
        """
        self._assert_open()
        self._has_table_write = True
        self._submission_changes.append((id, copy_json(settlement)))  # type: ignore[arg-type]

    def place_submission(self, id: SubmissionId, entry: EntryId, /) -> None:
        """Place a queued submission at ``entry``; resolved at assembly like ``settle_submission()``."""
        self._assert_open()
        self._has_table_write = True
        self._submission_changes.append((id, {"status": "placed", "entry": entry}))

    def set_task(self, value: AnyTaskRecord) -> None:
        """Internal: replace one task record completely. Tasks change their own state through their runtime."""
        self._assert_open()
        self._has_table_write = True
        task = self._task_entry(value["id"])
        candidate = None if task.write is None else task.write.record
        if candidate is not None and candidate["state"]["status"] == "terminal":
            raise RuntimeError(f"Task {value['id']} already has a terminal candidate")
        if candidate is not None and candidate["conversationId"] != value["conversationId"]:
            raise RuntimeError(f"Task {value['id']} cannot change conversations")
        kind: Literal["create", "replace"] = (
            "create" if task.write is not None and task.write.kind == "create" else "replace"
        )
        task.write = _TaskWrite(kind, _copy_table_json(self._stamp_times(value, candidate)))

    def staged_tasks(self) -> list[AnyTaskRecord]:
        """Internal: candidate records of the tasks this transaction created or replaced so far."""
        return [task.write.record for task in self._tasks_by_id.values() if task.write is not None]

    def staged_conversations(self) -> list[ConversationRecord]:
        """Internal: conversations this transaction created or forked so far."""
        return [write["value"] for write in self._writes if write["type"] == "conversation"]

    # ─── Documents ──────────────────────────────────────────────────────────

    def doc(self, token: AnyDocToken, *args: Any) -> asyncio.Future[Any]:
        """The draft of a document, created from ``initial()`` (``initial(seed)`` for a family) when absent.

        Arguments after the token are the owner ID, the family key, and the family seed, as the
        token's scope requires. Concurrent acquisitions of one address share one draft.
        """
        try:
            self._assert_open()
            definition = token.definition
            resolved_address = resolve_address(definition, args)
            self._assert_task_documents_open(resolved_address)
            latest = self._latest_document_by_address.get(resolved_address.id)
            if latest is not None and not latest.retire_on_commit:
                if latest.draft is not None:
                    return latest.draft
                target = latest.target
                if isinstance(target, _ForkCopyTarget):
                    latest.draft = self._track(self._acquire_fork_copy(latest, definition, target))
                    return latest.draft
            seed = copy_json(args[resolved_address.next_argument]) if definition.family else None
            entry = _DocumentEntry(resolved_address.id, resolved_address.address, definition, False)
            self._documents.append(entry)
            self._latest_document_by_address[entry.address_id] = entry
            # Capture retirement before awaiting so a pending old acquisition and its replacement stay distinct.
            entry.draft = self._track(self._acquire(entry, seed, latest is not None and latest.retire_on_commit))
            return entry.draft
        except Exception as error:
            return rejected(error)

    def retire_doc(self, token: AnyDocToken, *args: Any) -> asyncio.Future[None]:
        """Retire the document at an address when the commit settles; a later ``doc()`` creates a new incarnation."""
        try:
            self._assert_open()
            definition = token.definition
            resolved_address = resolve_address(definition, args)
            latest = self._latest_document_by_address.get(resolved_address.id)
            if latest is not None and latest.retire_on_commit:
                return resolved(None)
            if latest is not None and isinstance(latest.target, _ForkCopyTarget):
                check_record_scope(definition, latest.target.record)
                latest.retire_on_commit = True
                return resolved(None)
            if latest is not None and latest.draft is not None:
                # Retirement of an acquired draft persists its final content before retirement.
                latest.retire_on_commit = True
                return self._track(_settled_to_none(latest.draft))
            entry = _DocumentEntry(resolved_address.id, resolved_address.address, definition, True)
            self._documents.append(entry)
            self._latest_document_by_address[entry.address_id] = entry
            return self._track(self._find_retirement(entry))
        except Exception as error:
            return rejected(error)

    async def _acquire(self, entry: _DocumentEntry, seed: Any, skip_load: bool) -> Any:
        definition = entry.definition
        assert definition is not None
        loaded = (
            None if skip_load else await self._host.load(definition, entry.address_id, entry.address, self._context)
        )
        self._assert_open()
        if loaded is not None:
            check_record_scope(definition, loaded.record)
            check_record_version(definition, loaded.record, loaded.stored_version)
            entry.target = _LoadedTarget(loaded)
            entry.change = loaded.tracker.begin_change()
            return entry.change.state
        scope = entry.address["scope"]
        if scope["kind"] == "conversation":
            await self._require_conversation(scope["conversationId"])
        if scope["kind"] == "task":
            task = await self._current_task(scope["taskId"])
            if task is None:
                raise RuntimeError(f"Task {scope['taskId']} does not exist")
            if task["state"]["status"] == "terminal":
                raise RuntimeError(f"Task {scope['taskId']} is terminal")
        self._assert_open()
        value: JsonObject = copy_json(definition.initial(seed) if definition.family else definition.initial())  # type: ignore[assignment]
        id = DocumentId(await self._host.storage.mint_id())
        self._assert_open()
        tracker = track(value)
        entry.target = _CreatedTarget(document_create(definition, entry.address, id), definition.version, tracker)
        entry.change = tracker.begin_change()
        return entry.change.state

    async def _acquire_fork_copy(
        self, entry: _DocumentEntry, definition: AnyDocDefinition, target: _ForkCopyTarget
    ) -> Any:
        stored = await self._host.storage.document(target.source["id"], target.source["at"], self._context)
        self._assert_open()
        if stored is None:
            raise RuntimeError(f"Fork source document {target.source['id']} cannot be read")
        source = stored["record"]
        record = target.record
        if (
            source["scope"]["kind"] != "conversation"
            or source["kind"] != record["kind"]
            or source.get("key") != record.get("key")
            or source.get("history") != record.get("history")
            or source.get("fork") != record.get("fork")
        ):
            raise RuntimeError(f"Fork source document {target.source['id']} does not match the copied record")
        value = materialize_document_value(definition, record, stored["version"], stored["value"])
        tracker = track(value)
        entry.definition = definition
        entry.target = _CreatedTarget(record, definition.version, tracker)
        entry.change = tracker.begin_change()
        return entry.change.state

    async def _find_retirement(self, entry: _DocumentEntry) -> None:
        cached = self._host.cached(entry.address_id)
        record = (
            cached.record
            if cached is not None
            else await self._host.storage.find_document(entry.address, "current", self._context)
        )
        self._assert_open()
        if record is None:
            return
        assert entry.definition is not None
        check_record_scope(entry.definition, record)
        entry.target = _RetireOnlyTarget(record)

    # ─── Settlement ─────────────────────────────────────────────────────────

    async def settle_failure(self) -> None:
        """Seal after callback failure: abort every change and observe every pending operation."""
        self._sealed = True
        self._abort_changes()
        await _all_settled(self._pending)

    async def settle_success(self) -> list[StorageWrite]:
        """Seal after callback success, prepare every open change, and assemble the atomic batch.

        Any failure aborts every change before Storage admission.
        """
        self._sealed = True
        if any(not operation.done() for operation in self._pending):
            self._abort_changes()
            await _all_settled(self._pending)
            raise RuntimeError("Session commit callback settled before its pending Tx operations")
        try:
            # Synchronously prepare every open change; this revokes every draft.
            for document in self._documents:
                if document.change is not None:
                    document.prepared = document.change.prepare()
            return await self._assemble()
        except BaseException:
            self._abort_changes()
            raise

    def discard(self) -> None:
        """Abort every prepared change after Storage failure or when no write is required."""
        self._abort_changes()

    def adopt(self, seq: Seq) -> list[DocumentCommitChange]:
        """Adopt every prepared change by pointer swap after Storage success and describe the publication."""
        publications: list[DocumentCommitChange] = []
        for plan in self._plans:
            committed = "createdAt" in plan.record
            record: DocumentRecord = plan.record if committed else {**plan.record, "createdAt": seq}  # type: ignore[assignment]
            if plan.retire:
                record = {**record, "retiredAt": seq}  # type: ignore[assignment]
            change = plan.change
            if change is not None:
                loaded = change.loaded
                prepared = change.prepared
                # A new incarnation is adopted unless it retires in the same commit; a loaded one only when it changed.
                if (not plan.retire) if loaded is None else len(prepared.ops) > 0:
                    change.tracker.adopt(prepared)
                else:
                    prepared.abort()
                if loaded is not None:
                    if loaded.stored_version < change.version:
                        loaded.stored_version = change.version
                    content = plan.content
                    if content is not None and content["type"] == "document.change":
                        if content["content"]["kind"] == "base":
                            loaded.deltas_since_base = 0
                        else:
                            loaded.deltas_since_base += 1
                elif not plan.retire:
                    self._host.install(
                        LoadedDocument(plan.address_id, record, change.version, change.version, 0, change.tracker)
                    )
            conversation_id = plan.conversation_id
            if plan.retire:
                if committed:
                    self._host.evict(plan.address_id, record["id"])
                publications.append(
                    {
                        "type": "document",
                        "record": record,
                        "conversationId": conversation_id,
                        "version": None,
                        "value": None,
                        "ops": _EMPTY_OPERATIONS,
                    }
                )
            elif plan.content is not None and plan.content["type"] == "document.copy":
                assert conversation_id is not None
                publications.append(
                    {
                        "type": "document.copy",
                        "record": record,
                        "conversationId": conversation_id,
                        "source": plan.content["source"],
                    }
                )
            elif change is not None and _publishes(plan):
                ops = _EMPTY_OPERATIONS if change.loaded is None else change.prepared.ops
                publications.append(
                    {
                        "type": "document",
                        "record": record,
                        "conversationId": conversation_id,
                        "version": change.version,
                        "value": change.prepared.value,
                        "ops": ops,
                    }
                )
        return publications

    async def _assemble(self) -> list[StorageWrite]:
        storage = self._host.storage
        plans = self._plans
        for document in self._documents:
            plan = _plan_document(document)
            if plan is not None:
                plans.append(plan)
        self._reject_fork_source_writes(plans)
        await self._validate_owners()
        for id, task in self._tasks_by_id.items():
            if task.write is None or task.write.kind != "replace":
                continue
            committed = await self._committed_task(id)
            if committed is None:
                raise RuntimeError(f"Task {id} does not exist")
            if committed["state"]["status"] == "terminal":
                raise RuntimeError(f"Task {id} is already terminal")
            if committed["conversationId"] != task.write.record["conversationId"]:
                raise RuntimeError(f"Task {id} cannot change conversations")

        # Terminal settlement retires every task document, including ones created by this transaction.
        terminal_task_ids: dict[TaskId[Any], None] = {}
        for task in self._tasks_by_id.values():
            if task.write is not None and task.write.record["state"]["status"] == "terminal":
                terminal_task_ids[task.write.record["id"]] = None
        if terminal_task_ids:
            retiring: set[DocumentId] = set()
            for plan in plans:
                scope = plan.record["scope"]
                if scope["kind"] != "task" or scope["taskId"] not in terminal_task_ids:
                    continue
                plan.retire = True
                retiring.add(plan.record["id"])
            for task_id in terminal_task_ids:
                task_entry = self._tasks_by_id.get(task_id)
                if task_entry is not None and task_entry.write is not None and task_entry.write.kind == "create":
                    continue
                cursor: Cursor | None = None
                while True:
                    page = await storage.scan_documents(
                        {"scope": {"kind": "task", "taskId": task_id}, "at": "current"},
                        _INTERNAL_SCAN_PAGE_SIZE,
                        cursor,
                        self._context,
                    )
                    for record in page["items"]:
                        if record["id"] in retiring:
                            continue
                        plans.append(_DocumentPlan(address_id(record), record, True))  # type: ignore[arg-type]
                        retiring.add(record["id"])
                    cursor = page.get("next")
                    if cursor is None:
                        break

        # Resolve publication ownership before Storage admission so adoption remains synchronous.
        for plan in plans:
            if not _publishes(plan):
                continue
            scope = plan.record["scope"]
            if scope["kind"] == "conversation":
                plan.conversation_id = scope["conversationId"]
            if scope["kind"] != "task":
                continue
            task_entry = self._task_entry(scope["taskId"])
            if task_entry.publication_conversation_id is None:
                current = await self._current_task(scope["taskId"])
                if current is not None:
                    task_entry.publication_conversation_id = current["conversationId"]
            plan.conversation_id = task_entry.publication_conversation_id

        for id, submission_change in self._submission_changes:
            current_submission = self._submissions.get(id)
            if current_submission is None:
                current_submission = await storage.submission(id, self._context)
            if current_submission is None:
                raise RuntimeError(f"Submission {id} does not exist")
            following = _apply_submission_change(current_submission, submission_change)
            if following is not current_submission:
                self._submissions[id] = following

        writes = self._writes
        for value in self._submissions.values():
            writes.append({"type": "submission", "value": value})
        for task in self._tasks_by_id.values():
            if task.write is not None:
                writes.append({"type": "task", "value": task.write.record})
        for plan in plans:
            change = plan.change
            content = plan.content
            # Checkpoint predicates run last, after every validation.
            if (
                content is not None
                and content["type"] == "document.change"
                and content["content"]["kind"] == "delta"
                and change is not None
                and change.loaded is not None
            ):
                prepared = change.prepared
                checkpoint_when = None if change.definition is None else change.definition.checkpoint_when
                if checkpoint_when is not None and checkpoint_when(
                    prepared.value, prepared.ops, {"deltasSinceBase": change.loaded.deltas_since_base}
                ):
                    base: StorageWriteDocumentChange = {
                        "type": "document.change",
                        "id": plan.record["id"],
                        "content": {"version": change.version, "kind": "base", "value": prepared.value},
                    }
                    plan.content = base
            if plan.content is not None:
                writes.append(plan.content)
            if plan.retire:
                writes.append({"type": "document.retire", "id": plan.record["id"]})
        return writes

    # ─── Helpers ────────────────────────────────────────────────────────────

    async def _validate_owners(self) -> None:
        """New owned work needs a live owner, judged on the owner's final candidate.

        Not ``completing``, terminal, or abort-marked: a task cannot create owned work in the commit that
        finishes it (spec §5.5).
        """
        owners: list[tuple[str, TaskId[Any]]] = []
        for write in self._writes:
            if write["type"] != "conversation":
                continue
            owner = write["value"].get("owner")
            if owner is not None:
                owners.append(("Conversation owner task", owner["taskId"]))
        for task in self._tasks_by_id.values():
            record = task.write.record if task.write is not None and task.write.kind == "create" else None
            owner_id = None if record is None else record.get("owner")
            if owner_id is not None:
                owners.append(("Task owner", owner_id))
        for what, task_id in owners:
            current = await self._current_task(task_id)
            if current is None:
                raise RuntimeError(f"{what} {task_id} does not exist")
            status = current["state"]["status"]
            if status == "terminal" or status == "completing":
                raise RuntimeError(f"{what} {task_id} is {status}")
            if current["abortRequested"]:
                raise RuntimeError(f"{what} {task_id} is abort-marked")

    def _reject_fork_source_writes(self, plans: Sequence[_DocumentPlan]) -> None:
        for plan in plans:
            if plan.content is None and not plan.retire:
                continue
            record = plan.record
            if record["id"] in self._fork_source_document_ids:
                raise RuntimeError(f"Cannot change fork source document {record['id']} in the fork transaction")
            scope = record["scope"]
            if (
                scope["kind"] == "conversation"
                and scope["conversationId"] in self._fork_source_conversation_ids
                and record.get("fork") == "current"
            ):
                raise RuntimeError(
                    f"Cannot fork conversation {scope['conversationId']} while changing its current-policy documents"
                )

    def _abort_changes(self) -> None:
        for document in self._documents:
            if document.change is not None:
                document.change.abort()

    def _assert_open(self) -> None:
        if self._sealed:
            raise RuntimeError("Transaction has settled")

    def _assert_task_documents_open(self, resolved_address: ResolvedAddress) -> None:
        scope = resolved_address.address["scope"]
        if scope["kind"] == "task":
            task = self._tasks_by_id.get(scope["taskId"])
            if task is not None and task.write is not None and task.write.record["state"]["status"] == "terminal":
                raise RuntimeError(f"Task {scope['taskId']} is terminal")

    def _track(self, operation: Awaitable[T]) -> asyncio.Future[T]:
        """Register an operation so callback settlement can reject and drain it."""
        future = asyncio.ensure_future(operation)
        self._pending.add(future)
        future.add_done_callback(self._settled)
        return future

    def _settled(self, future: asyncio.Future[Any]) -> None:
        # Tracked like upstream's `operation.then(settle, settle)`: a failure is observed here.
        self._pending.discard(future)
        if not future.cancelled():
            future.exception()

    def _read(self, method: str, read: Callable[[], Awaitable[T]]) -> asyncio.Future[T]:
        try:
            self._assert_open()
            if self._has_table_write:
                raise ReadAfterWrite(method)
            return self._track(read())
        except Exception as error:
            return rejected(error)

    def _write(self, write: Callable[[], Coroutine[Any, Any, T]]) -> asyncio.Future[T]:
        try:
            self._assert_open()
            self._has_table_write = True
            return self._track(write())
        except Exception as error:
            return rejected(error)

    async def _require_conversation(self, id: ConversationId) -> None:
        if id in self._created_conversation_ids:
            return
        if await self._host.storage.conversation(id, self._context) is None:
            raise RuntimeError(f"Conversation {id} does not exist")

    def _stamp_times(self, value: AnyTaskRecord, candidate: AnyTaskRecord | None) -> AnyTaskRecord:
        """Lifecycle times: ``startedAt`` on the first change to ``running``, ``endedAt`` on the change to ``terminal``.

        Once set, they carry over from the replaced record; records written before they existed lack them.
        """
        status = value["state"]["status"]
        started_at = _first_present(
            None if candidate is None else candidate.get("startedAt"),
            value.get("startedAt"),
            self._host.now() if status == "running" else None,
        )
        ended_at = _first_present(
            None if candidate is None else candidate.get("endedAt"),
            value.get("endedAt"),
            self._host.now() if status == "terminal" else None,
        )
        if started_at == value.get("startedAt") and ended_at == value.get("endedAt"):
            return value
        stamped: dict[str, Any] = dict(value)
        if started_at is not None:
            stamped["startedAt"] = started_at
        if ended_at is not None:
            stamped["endedAt"] = ended_at
        return stamped  # type: ignore[return-value]

    def _task_entry(self, id: TaskId[Any]) -> _TransactionTask:
        task = self._tasks_by_id.get(id)
        if task is None:
            task = _TransactionTask()
            self._tasks_by_id[id] = task
        return task

    async def _current_task(self, id: TaskId[Any]) -> AnyTaskRecord | None:
        """Latest candidate task record, falling back to committed state; not a caller table read."""
        task = self._tasks_by_id.get(id)
        if task is not None and task.write is not None:
            return task.write.record
        return await self._committed_task(id)

    def _committed_task(self, id: TaskId[Any]) -> asyncio.Future[AnyTaskRecord | None]:
        task = self._task_entry(id)
        if task.committed_read is None:
            task.committed_read = asyncio.ensure_future(self._host.storage.task(id, self._context))
        return task.committed_read


def _first_present(*values: Any) -> Any:
    """The first value that is not ``None`` (upstream ``a ?? b ?? c``)."""
    for value in values:
        if value is not None:
            return value
    return None


def _copy_table_json(value: Mapping[str, Any]) -> Any:
    """Take ownership of a table record: strict-JSON copy with ``UNDEFINED`` properties omitted.

    pi-ai messages (``model``, and ``messages`` of context edits) are detached through their stored JSON
    instead, since they are dataclasses in Python.
    """
    result: dict[str, Any] = {}
    for key, item in value.items():
        if item is UNDEFINED:
            continue
        if key == "model" and isinstance(item, list | tuple):
            result[key] = [clone_value(message) for message in item]
        elif key == "edits" and isinstance(item, list | tuple):
            result[key] = [_copy_edit(edit) for edit in item]
        else:
            result[key] = copy_json(item, omit_undefined_properties=True)
    return result


def _copy_edit(edit: Any) -> Any:
    if not isinstance(edit, dict):
        return copy_json(edit)
    result: dict[str, Any] = {}
    for key, item in edit.items():
        if item is UNDEFINED:
            continue
        if key == "messages" and isinstance(item, list | tuple):
            result[key] = [clone_value(message) for message in item]
        else:
            result[key] = copy_json(item, omit_undefined_properties=True)
    return result


async def _settled_to_none(future: asyncio.Future[Any]) -> None:
    await future


async def _all_settled(operations: set[asyncio.Future[Any]]) -> None:
    """Wait until every operation settled, whatever the outcome (``Promise.allSettled``)."""
    pending = [operation for operation in operations if not operation.done()]
    if pending:
        await asyncio.wait(pending)


def _plan_document(document: _DocumentEntry) -> _DocumentPlan | None:
    """Plan of one staged document: its record, content write, and prepared change. Retirement is decided later."""
    target = document.target
    if target is None:
        return None
    retire = document.retire_on_commit
    if isinstance(target, _CreatedTarget):
        prepared = document.prepared
        assert prepared is not None
        record = target.record
        return _DocumentPlan(
            document.address_id,
            record,
            retire,
            {
                "type": "document.create",
                "record": record,
                "content": {"version": target.version, "kind": "base", "value": prepared.value},
            },
            _PlanChange(target.tracker, prepared, target.version),
        )
    if isinstance(target, _ForkCopyTarget):
        return _DocumentPlan(
            document.address_id,
            target.record,
            retire,
            {"type": "document.copy", "record": target.record, "source": target.source},
        )
    if isinstance(target, _RetireOnlyTarget):
        return _DocumentPlan(document.address_id, target.record, retire)
    loaded = target.document
    definition = document.definition
    prepared = document.prepared
    assert definition is not None and prepared is not None
    version = definition.version
    id = loaded.record["id"]
    # A version change stores a base even without operations; otherwise only a change stores a delta.
    content: StorageWrite | None
    if loaded.stored_version < version:
        content = {
            "type": "document.change",
            "id": id,
            "content": {"version": version, "kind": "base", "value": prepared.value},
        }
    elif len(prepared.ops) > 0:
        content = {
            "type": "document.change",
            "id": id,
            "content": {"version": version, "kind": "delta", "ops": prepared.ops},
        }
    else:
        content = None
    change = _PlanChange(loaded.tracker, prepared, version, loaded, definition)
    return _DocumentPlan(document.address_id, loaded.record, retire, content, change)


def _publishes(plan: _DocumentPlan) -> bool:
    """Whether adoption publishes the plan: every creation, copy, and retirement, and a loaded incarnation
    that writes content, which includes a migration-only base so observers of the older shape receive the new value.
    """
    return plan.retire or plan.change is None or plan.change.loaded is None or plan.content is not None


if TYPE_CHECKING:

    def _implements_tx(tx: Transaction) -> Tx:
        return tx
