"""Detached in-memory reference implementation of ``Storage`` (port of ``storage/memory.ts``)."""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pi_chord.context import Context
from pi_chord.delta import Op, apply_immutable_batches
from pi_chord.json import JsonObject

from ..errors import StorageRequestError
from ..ids import MAX_SAFE_INTEGER, id_from_number, seq_from_number
from ..types import (
    AnyTaskRecord,
    ConversationId,
    ConversationQuery,
    ConversationRecord,
    Cursor,
    DocumentAddress,
    DocumentContent,
    DocumentCreate,
    DocumentId,
    DocumentPoint,
    DocumentQuery,
    DocumentRecord,
    DocumentScope,
    EntryId,
    EntryQuery,
    EntryRecord,
    HeadMarker,
    Page,
    ScanOrder,
    Seq,
    StorageWrite,
    StoredDocument,
    StoredEntry,
    SubmissionId,
    SubmissionQuery,
    SubmissionRecord,
    TaskId,
    TaskQuery,
)
from .codec import clone_value
from .scan import ScanStart, next_cursor, scan_start

__all__ = ["MemoryStorage", "PreparedMemoryCommit"]

_TableName = Literal["conversation", "entry", "task", "submission", "document"]
_INFINITY = float("inf")


@dataclass(slots=True)
class _DocumentRevision:
    seq: Seq
    content: DocumentContent


@dataclass(slots=True)
class _StoredDocumentState:
    record: DocumentRecord
    revisions: list[_DocumentRevision]


@dataclass(slots=True)
class _DocumentAction:
    create: DocumentCreate | None = None
    content: DocumentContent | None = None
    retire: bool = False


@dataclass(slots=True)
class _DocumentAddressIndex:
    ids: list[int] = field(default_factory=list)
    current_id: DocumentId | None = None


@dataclass(slots=True)
class _State:
    record_types: dict[int, _TableName] = field(default_factory=dict)
    conversations: dict[int, ConversationRecord] = field(default_factory=dict)
    conversation_ids: list[int] = field(default_factory=list)
    conversation_ids_by_owner_conversation: dict[int, list[int]] = field(default_factory=dict)
    conversation_ids_by_owner_task: dict[int, list[int]] = field(default_factory=dict)
    entries: dict[int, EntryRecord] = field(default_factory=dict)
    entry_ids: dict[int, list[int]] = field(default_factory=dict)
    head_entry_ids: dict[int, list[int]] = field(default_factory=dict)
    entry_commit_seqs: dict[int, Seq] = field(default_factory=dict)
    tasks: dict[int, AnyTaskRecord] = field(default_factory=dict)
    task_ids: list[int] = field(default_factory=list)
    task_ids_by_status: dict[str, list[int]] = field(
        default_factory=lambda: {
            "pending": [],
            "running": [],
            "waiting": [],
            "sleeping": [],
            "completing": [],
            "terminal": [],
        }
    )
    submissions: dict[int, SubmissionRecord] = field(default_factory=dict)
    submission_ids: list[int] = field(default_factory=list)
    submission_ids_by_status: dict[str, list[int]] = field(
        default_factory=lambda: {"queued": [], "placed": [], "done": [], "unanswered": []}
    )
    submission_ids_by_request: dict[int, dict[str, SubmissionId]] = field(default_factory=dict)
    documents: dict[int, _StoredDocumentState] = field(default_factory=dict)
    document_addresses: dict[str, _DocumentAddressIndex] = field(default_factory=dict)
    document_ids_by_scope: dict[str, list[int]] = field(default_factory=dict)


def _document_delta_batches(
    id: DocumentId, version: int, revisions: Sequence[_DocumentRevision], start: int
) -> Iterator[Sequence[Op]]:
    for index in range(start, len(revisions)):
        content = revisions[index].content
        if content["kind"] != "delta" or content["version"] != version:
            raise RuntimeError(f"Document {id} crosses a stored version boundary without a base")
        yield content["ops"]


def _cursor_id(cursor: Cursor | None) -> int | None:
    after = None if cursor is None else cursor.get("after")
    if after is None:
        return None
    if type(after) is not int or not -MAX_SAFE_INTEGER <= after <= MAX_SAFE_INTEGER:
        raise StorageRequestError("Invalid storage cursor")
    return after


def _insert_sorted(ids: list[int], id: int) -> None:
    if len(ids) == 0 or ids[-1] < id:
        ids.append(id)
    else:
        bisect.insort_left(ids, id)


def _remove_sorted(ids: list[int], id: int) -> None:
    index = bisect.bisect_left(ids, id)
    if index < len(ids) and ids[index] == id:
        del ids[index]


def _insert_map_id(index: dict[Any, list[int]], key: Any, id: int) -> None:
    ids = index.get(key)
    if ids is None:
        ids = []
        index[key] = ids
    _insert_sorted(ids, id)


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _scope_key(scope: DocumentScope) -> str:
    if scope["kind"] == "session":
        return _dumps(["session"])
    if scope["kind"] == "conversation":
        return _dumps(["conversation", scope["conversationId"]])
    return _dumps(["task", scope["taskId"]])


def _address_key(address: DocumentAddress) -> str:
    key = address.get("key")
    return _dumps([address["kind"], _scope_key(address["scope"]), ["singleton"] if key is None else ["family", key]])


def _record_address_key(record: DocumentRecord | DocumentCreate) -> str:
    address: dict[str, Any] = {"kind": record["kind"], "scope": record["scope"]}
    if "key" in record:
        address["key"] = record["key"]
    return _address_key(cast(DocumentAddress, address))


def _is_alive_at(record: DocumentRecord, at: DocumentPoint) -> bool:
    retired = record.get("retiredAt")
    if at == "current":
        return retired is None
    return record["createdAt"] <= at and (retired is None or at < retired)


def _is_current_only(record: DocumentRecord) -> bool:
    return record["scope"]["kind"] != "conversation" or record.get("history") == "latest"


def _page(values: list[Any], limit: int, order: ScanOrder = "ascending") -> Page[Any]:
    items = values[:limit]
    page: Page[Any] = {"items": clone_value(items)}
    if len(values) <= limit:
        return page
    return {**page, "next": next_cursor(items[-1]["id"], order)}


def _scan_indexes(ids: Sequence[int], start: ScanStart) -> Iterator[int]:
    """Indexes of sorted ``ids`` in scan order, after the cursor's ID when there is one."""
    if start.order == "ascending":
        first = 0 if start.after is None else bisect.bisect_right(ids, start.after)
        yield from range(first, len(ids))
    else:
        end = len(ids) if start.after is None else bisect.bisect_left(ids, start.after)
        yield from range(end - 1, -1, -1)


def _write_id(write: StorageWrite) -> int:
    if write["type"] in ("document.create", "document.copy"):
        return write["record"]["id"]  # type: ignore[typeddict-item]
    return write["value"]["id"]  # type: ignore[typeddict-item]


class PreparedMemoryCommit:
    """A fully validated, detached state mutation whose application performs no fallible preparation."""

    __slots__ = ("_actions", "_applied", "_storage", "seq", "writes")

    def __init__(
        self,
        storage: MemoryStorage,
        seq: Seq,
        writes: list[StorageWrite],
        actions: dict[int, _DocumentAction],
    ) -> None:
        self._storage = storage
        self._actions = actions
        self._applied = False
        self.seq = seq
        #: Detached writes for persistence. The memory storage retains them: never mutate them.
        self.writes: Sequence[StorageWrite] = writes

    def apply(self) -> Seq:
        """Adopt the commit; idempotent."""
        if not self._applied:
            self._applied = True
            self._storage._apply_prepared_commit(self.writes, self._actions, self.seq)
        return self.seq


class MemoryStorage:
    """Detached in-memory reference implementation of :class:`~pi_durable.types.Storage`.

    Reads and retained writes are cloned intentionally to match the ownership boundary of
    serialization-backed stores (messages round-trip through their stored JSON). This is backend
    conformance, not validation.
    """

    def __init__(self) -> None:
        self._state = _State()
        self._next_id = 2
        self._next_seq = 1
        self._closed = False

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        return self.prepare_commit(writes).apply()

    def prepare_commit(self, writes: Sequence[StorageWrite], seq: Seq | None = None) -> PreparedMemoryCommit:
        """Validate and detach one commit without changing observable state."""
        self._assert_open()
        if seq is None:
            seq = seq_from_number(self._next_seq)
        if type(seq) is not int or seq > MAX_SAFE_INTEGER or seq < self._next_seq:
            raise RuntimeError(f"Commit sequence {seq} does not strictly increase")
        detached = self._resolve_document_copies([clone_value(write) for write in writes])
        self._check_global_ids(detached)
        actions = self._prepare_document_actions(detached)
        self._check_document_actions(actions)
        return PreparedMemoryCommit(self, seq, detached, actions)

    def _resolve_document_copies(self, writes: list[StorageWrite]) -> list[StorageWrite]:
        if not any(write["type"] == "document.copy" for write in writes):
            return writes
        changed: set[int] = set()
        for write in writes:
            if write["type"] in ("document.create", "document.copy"):
                changed.add(write["record"]["id"])  # type: ignore[typeddict-item]
            elif write["type"] in ("document.change", "document.retire"):
                changed.add(write["id"])  # type: ignore[typeddict-item]
        resolved: list[StorageWrite] = []
        for write in writes:
            if write["type"] != "document.copy":
                resolved.append(write)
                continue
            record = write["record"]
            source = write["source"]
            try:
                if source["id"] in changed:
                    raise RuntimeError(f"Fork source document {source['id']} is changed in the copy batch")
                stored = self._materialize_document(source["id"], source["at"])
                if stored is None:
                    raise RuntimeError(f"Fork source document {source['id']} cannot be read")
                source_record = stored["record"]
                if (
                    source_record["scope"]["kind"] != "conversation"
                    or record["scope"]["kind"] != "conversation"
                    or source_record["kind"] != record["kind"]
                    or source_record.get("key") != record.get("key")
                    or source_record.get("history") != record.get("history")
                    or source_record.get("fork") != record.get("fork")
                ):
                    raise RuntimeError(f"Fork source document {source['id']} does not match the copied record")
                resolved.append(
                    {
                        "type": "document.create",
                        "record": record,
                        "content": {"kind": "base", "version": stored["version"], "value": stored["value"]},
                    }
                )
            except Exception as error:
                raise RuntimeError(f"Document copy {record['id']} was rejected") from error
        return resolved

    def _apply_prepared_commit(
        self, prepared: Sequence[StorageWrite], actions: Mapping[int, _DocumentAction], seq: Seq
    ) -> Seq:
        state = self._state
        for write in prepared:
            if write["type"] == "conversation":
                conversation = write["value"]
                id = conversation["id"]
                state.record_types[id] = "conversation"
                state.conversations[id] = conversation
                _insert_sorted(state.conversation_ids, id)
                owner = conversation.get("owner")
                if owner is not None:
                    _insert_map_id(state.conversation_ids_by_owner_conversation, owner["conversationId"], id)
                    _insert_map_id(state.conversation_ids_by_owner_task, owner["taskId"], id)
                self._next_id = max(self._next_id, id + 1)
            elif write["type"] == "entry":
                entry = write["value"]
                id = entry["id"]
                state.record_types[id] = "entry"
                state.entries[id] = entry
                state.entry_commit_seqs[id] = seq
                _insert_map_id(state.entry_ids, entry["conversationId"], id)
                if entry.get("head") is not None:
                    _insert_map_id(state.head_entry_ids, entry["conversationId"], id)
                self._next_id = max(self._next_id, id + 1)
            elif write["type"] == "task":
                task = write["value"]
                id = task["id"]
                state.record_types[id] = "task"
                previous = state.tasks.get(id)
                status = task["state"]["status"]
                if previous is None:
                    _insert_sorted(state.task_ids, id)
                    _insert_sorted(state.task_ids_by_status[status], id)
                elif previous["state"]["status"] != status:
                    _remove_sorted(state.task_ids_by_status[previous["state"]["status"]], id)
                    _insert_sorted(state.task_ids_by_status[status], id)
                state.tasks[id] = task
                self._next_id = max(self._next_id, id + 1)
            elif write["type"] == "submission":
                submission = write["value"]
                id = submission["id"]
                state.record_types[id] = "submission"
                previous_submission = state.submissions.get(id)
                if previous_submission is None:
                    _insert_sorted(state.submission_ids, id)
                    _insert_sorted(state.submission_ids_by_status[submission["status"]], id)
                elif previous_submission["status"] != submission["status"]:
                    _remove_sorted(state.submission_ids_by_status[previous_submission["status"]], id)
                    _insert_sorted(state.submission_ids_by_status[submission["status"]], id)
                previous_request = None if previous_submission is None else previous_submission.get("requestId")
                if previous_submission is not None and previous_request is not None:
                    previous_requests = state.submission_ids_by_request.get(previous_submission["conversationId"])
                    if previous_requests is not None and previous_requests.get(previous_request) == id:
                        del previous_requests[previous_request]
                        if len(previous_requests) == 0:
                            del state.submission_ids_by_request[previous_submission["conversationId"]]
                state.submissions[id] = submission
                request_id = submission.get("requestId")
                if request_id is not None:
                    requests = state.submission_ids_by_request.setdefault(submission["conversationId"], {})
                    requests[request_id] = submission["id"]
                self._next_id = max(self._next_id, id + 1)
            elif write["type"] == "document.copy":
                raise RuntimeError("Prepared document copy was not resolved")
        self._apply_document_actions(actions, seq)
        self._next_seq = seq + 1
        return seq

    async def mint_id(self) -> int:
        self._assert_open()
        if self._next_id > MAX_SAFE_INTEGER:
            raise RuntimeError("ID space is exhausted")
        id = self._next_id
        self._next_id += 1
        return id

    async def conversation(self, id: ConversationId, context: Context) -> ConversationRecord | None:
        self._assert_open()
        value = self._state.conversations.get(id)
        return None if value is None else clone_value(value)

    async def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[ConversationRecord]:
        self._assert_open()
        state = self._state
        owner_task = query.get("ownerTaskId")
        owner_conversation = query.get("ownerConversationId")
        if owner_task is not None:
            ids = state.conversation_ids_by_owner_task.get(owner_task, [])
        elif owner_conversation is not None:
            ids = state.conversation_ids_by_owner_conversation.get(owner_conversation, [])
        else:
            ids = state.conversation_ids
        start = scan_start(query.get("order"), cursor, "ascending")
        values: list[ConversationRecord] = []
        for index in _scan_indexes(ids, start):
            if len(values) > limit:
                break
            value = state.conversations[ids[index]]
            owner = value.get("owner")
            if owner_conversation is not None and (owner is None or owner["conversationId"] != owner_conversation):
                continue
            values.append(value)
        return _page(values, limit, start.order)

    async def entry(self, id: EntryId, context: Context) -> StoredEntry | None:
        self._assert_open()
        entry = self._state.entries.get(id)
        if entry is None:
            return None
        return {"entry": clone_value(entry), "commitSeq": self._state.entry_commit_seqs[id]}

    async def visible_entry(self, conversation_id: ConversationId, id: EntryId, context: Context) -> StoredEntry | None:
        self._assert_open()
        if type(id) is not int:
            raise StorageRequestError("Storage.entry() requires an entry ID")
        entry = next(self._visible_entries(conversation_id, id, id), None)
        if entry is None:
            return None
        return {"entry": clone_value(entry), "commitSeq": self._state.entry_commit_seqs[id]}

    async def find_latest_head_marker(
        self, conversation_id: ConversationId, at_or_before_entry_id: EntryId | None, context: Context
    ) -> HeadMarker | None:
        self._assert_open()
        state = self._state
        if conversation_id not in state.conversations:
            raise StorageRequestError(f"Unknown conversation: {conversation_id}")
        current_id = conversation_id
        upper: float = _INFINITY if at_or_before_entry_id is None else at_or_before_entry_id
        while True:
            ids = state.head_entry_ids.get(current_id, [])
            index = bisect.bisect_right(ids, upper) - 1
            if index >= 0:
                return clone_value(state.entries[ids[index]])
            conversation = state.conversations[current_id]
            parent = conversation.get("parent")
            if parent is None:
                return None
            upper = min(upper, parent["at"])
            current_id = parent["conversationId"]

    async def scan_entries(
        self, query: EntryQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[EntryRecord]:
        self._assert_open()
        order, after = scan_start(query.get("order"), cursor, "descending")
        # The cursor narrows the bound on the side the scan moves away from.
        min_entry_id: float | None = query.get("minEntryId")
        max_entry_id: float | None = query.get("maxEntryId")
        if after is not None and order == "descending":
            max_entry_id = min(after if max_entry_id is None else max_entry_id, after - 1)
        if after is not None and order == "ascending":
            min_entry_id = max(after if min_entry_id is None else min_entry_id, after + 1)
        visible: list[EntryRecord] = []
        entries = (
            self._visible_entries(query["conversationId"], min_entry_id, max_entry_id)
            if order == "descending"
            else self._visible_entries_ascending(query["conversationId"], min_entry_id, max_entry_id)
        )
        for entry in entries:
            visible.append(entry)
            if len(visible) > limit:
                break
        return _page(visible, limit, order)

    async def task(self, id: TaskId[Any], context: Context) -> AnyTaskRecord | None:
        self._assert_open()
        value = self._state.tasks.get(id)
        return None if value is None else clone_value(value)

    async def scan_tasks(
        self, query: TaskQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[AnyTaskRecord]:
        self._assert_open()
        state = self._state
        start = scan_start(query.get("order"), cursor, "ascending")
        status = query.get("status")
        ids = state.task_ids if status is None else state.task_ids_by_status[status]
        conversation_id = query.get("conversationId")
        kind = query.get("kind")
        abort_requested = query.get("abortRequested")
        background = query.get("background")
        values: list[AnyTaskRecord] = []
        for index in _scan_indexes(ids, start):
            if len(values) > limit:
                break
            value = state.tasks[ids[index]]
            if conversation_id is not None and value["conversationId"] != conversation_id:
                continue
            if kind is not None and value["kind"] != kind:
                continue
            if abort_requested is not None and value["abortRequested"] != abort_requested:
                continue
            if background is not None and value["background"] != background:
                continue
            values.append(value)
        return _page(values, limit, start.order)

    async def submission(self, id: SubmissionId, context: Context) -> SubmissionRecord | None:
        self._assert_open()
        value = self._state.submissions.get(id)
        return None if value is None else clone_value(value)

    async def scan_submissions(
        self, query: SubmissionQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[SubmissionRecord]:
        self._assert_open()
        state = self._state
        start = scan_start(query.get("order"), cursor, "ascending")
        status = query.get("status")
        ids = state.submission_ids if status is None else state.submission_ids_by_status[status]
        conversation_id = query.get("conversationId")
        values: list[SubmissionRecord] = []
        for index in _scan_indexes(ids, start):
            if len(values) > limit:
                break
            value = state.submissions[ids[index]]
            if conversation_id is not None and value["conversationId"] != conversation_id:
                continue
            values.append(value)
        return _page(values, limit, start.order)

    async def submission_by_request(
        self, conversation_id: ConversationId, request_id: str, context: Context
    ) -> SubmissionRecord | None:
        self._assert_open()
        id = self._state.submission_ids_by_request.get(conversation_id, {}).get(request_id)
        if id is None:
            return None
        return clone_value(self._state.submissions[id])

    async def find_document(
        self, address: DocumentAddress, at: DocumentPoint, context: Context
    ) -> DocumentRecord | None:
        self._assert_open()
        state = self._state
        index = state.document_addresses.get(_address_key(address))
        if at == "current":
            if index is None or index.current_id is None:
                return None
            return clone_value(state.documents[index.current_id].record)
        for id in [] if index is None else index.ids:
            record = state.documents[id].record
            if _is_alive_at(record, at):
                return clone_value(record)
        return None

    async def document(self, id: DocumentId, at: DocumentPoint, context: Context) -> StoredDocument | None:
        self._assert_open()
        stored = self._materialize_document(id, at)
        if stored is None:
            return None
        return {
            "record": clone_value(stored["record"]),
            "version": stored["version"],
            "value": clone_value(stored["value"]),
            "deltasSinceBase": stored["deltasSinceBase"],
        }

    async def scan_documents(
        self, query: DocumentQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[DocumentRecord]:
        self._assert_open()
        state = self._state
        ids = state.document_ids_by_scope.get(_scope_key(query["scope"]), [])
        after = _cursor_id(cursor)
        start = 0 if after is None else bisect.bisect_right(ids, after)
        kind = query.get("kind")
        at = query["at"]
        values: list[DocumentRecord] = []
        index = start
        while index < len(ids) and len(values) <= limit:
            record = state.documents[ids[index]].record
            index += 1
            if kind is not None and record["kind"] != kind:
                continue
            if _is_alive_at(record, at):
                values.append(record)
        return _page(values, limit)

    async def close(self, context: Context) -> None:
        self._closed = True

    def _materialize_document(self, id: DocumentId, at: DocumentPoint) -> StoredDocument | None:
        stored = self._state.documents.get(id)
        if stored is None:
            return None
        if at != "current" and _is_current_only(stored.record):
            raise StorageRequestError(f"Document {id} does not retain historical content")
        if not _is_alive_at(stored.record, at):
            return None
        revisions = (
            stored.revisions if at == "current" else [revision for revision in stored.revisions if revision.seq <= at]
        )
        base_index = len(revisions) - 1
        while base_index >= 0 and revisions[base_index].content["kind"] != "base":
            base_index -= 1
        base = revisions[base_index].content if base_index >= 0 else None
        if base is None or base["kind"] != "base":
            raise RuntimeError(f"Document {id} is missing a required base")
        value = cast(
            JsonObject,
            apply_immutable_batches(
                base["value"], _document_delta_batches(id, base["version"], revisions, base_index + 1)
            ),
        )
        return {
            "record": stored.record,
            "version": base["version"],
            "value": value,
            "deltasSinceBase": len(revisions) - base_index - 1,
        }

    def _visible_entries(
        self,
        conversation_id: ConversationId,
        min_entry_id: float | None = None,
        max_entry_id: float | None = None,
    ) -> Iterator[EntryRecord]:
        state = self._state
        if conversation_id not in state.conversations:
            raise StorageRequestError(f"Unknown conversation: {conversation_id}")
        lower = -_INFINITY if min_entry_id is None else min_entry_id
        current_id = conversation_id
        upper = _INFINITY if max_entry_id is None else max_entry_id
        while True:
            ids = state.entry_ids.get(current_id, [])
            for index in range(bisect.bisect_right(ids, upper) - 1, -1, -1):
                id = ids[index]
                if id < lower:
                    break
                yield state.entries[id]
            parent = state.conversations[current_id].get("parent")
            if parent is None:
                break
            upper = min(upper, parent["at"])
            if upper < lower:
                break
            current_id = parent["conversationId"]

    def _visible_entries_ascending(
        self,
        conversation_id: ConversationId,
        min_entry_id: float | None = None,
        max_entry_id: float | None = None,
    ) -> Iterator[EntryRecord]:
        """Visible entries oldest first: the fork chain's segments from the root conversation forward."""
        state = self._state
        if conversation_id not in state.conversations:
            raise StorageRequestError(f"Unknown conversation: {conversation_id}")
        lower = -_INFINITY if min_entry_id is None else min_entry_id
        segments: list[tuple[int, float]] = []
        current_id = conversation_id
        upper = _INFINITY if max_entry_id is None else max_entry_id
        while True:
            segments.append((current_id, upper))
            parent = state.conversations[current_id].get("parent")
            if parent is None:
                break
            upper = min(upper, parent["at"])
            if upper < lower:
                break
            current_id = parent["conversationId"]
        for segment_id, segment_upper in reversed(segments):
            ids = state.entry_ids.get(segment_id, [])
            for index in range(bisect.bisect_left(ids, lower), len(ids)):
                id = ids[index]
                if id > segment_upper:
                    break
                yield state.entries[id]

    def _check_global_ids(self, writes: Sequence[StorageWrite]) -> None:
        claimed: dict[int, _TableName] = {}
        for write in writes:
            kind = write["type"]
            if kind in ("document.change", "document.retire"):
                continue
            document = kind in ("document.create", "document.copy")
            table: _TableName = "document" if document else cast(_TableName, kind)
            id = _write_id(write)
            existing = self._state.record_types.get(id)
            earlier = claimed.get(id)
            if table in ("conversation", "entry", "document"):
                if existing is not None:
                    raise RuntimeError(f"ID {id} already belongs to {existing}")
                if earlier is not None:
                    raise RuntimeError(f"ID {id} is written more than once")
            else:
                if existing is not None and existing != table:
                    raise RuntimeError(f"ID {id} already belongs to {existing}")
                if earlier is not None and earlier != table:
                    raise RuntimeError(f"ID {id} is written as two record types")
            claimed[id] = table

    def _prepare_document_actions(self, writes: Sequence[StorageWrite]) -> dict[int, _DocumentAction]:
        actions: dict[int, _DocumentAction] = {}
        for write in writes:
            if write["type"] == "document.create":
                id = write["record"]["id"]
            elif write["type"] in ("document.change", "document.retire"):
                id = write["id"]  # type: ignore[typeddict-item]
            else:
                continue
            action = actions.get(id)
            if action is None:
                action = _DocumentAction()
                actions[id] = action
            if write["type"] == "document.create":
                if action.create is not None or action.content is not None:
                    raise RuntimeError(f"Document {id} has more than one content command")
                action.create = write["record"]
                action.content = write["content"]
            elif write["type"] == "document.change":
                if action.content is not None:
                    raise RuntimeError(f"Document {id} has more than one content command")
                action.content = write["content"]
            else:
                if action.retire:
                    raise RuntimeError(f"Document {id} is retired more than once")
                action.retire = True
        return actions

    def _check_document_actions(self, actions: Mapping[int, _DocumentAction]) -> None:
        state = self._state
        live_counts: dict[str, int] = {}
        for id, action in actions.items():
            existing = state.documents.get(id)
            if action.create is None and existing is None:
                raise RuntimeError(f"Unknown document: {id}")
            if action.create is not None and existing is not None:
                raise RuntimeError(f"Document {id} already exists")
            if existing is not None and existing.record.get("retiredAt") is not None:
                raise RuntimeError(f"Document {id} is retired")
            previous = existing.revisions[-1] if existing is not None and existing.revisions else None
            if action.content is not None and action.content["kind"] == "delta":
                if previous is None:
                    raise RuntimeError(f"Document {id} delta has no base")
                if previous.content["version"] != action.content["version"]:
                    raise RuntimeError(f"Document {id} version transition requires a base")

            if action.create is not None:
                key = _record_address_key(action.create)
            else:
                assert existing is not None
                key = _record_address_key(existing.record)
            address = state.document_addresses.get(key)
            current_id = None if address is None else address.current_id
            live = live_counts.get(key)
            if live is None:
                live = 0 if current_id is None else 1
            if action.retire and current_id == id:
                live -= 1
            if action.create is not None and not action.retire:
                live += 1
            live_counts[key] = live

        for live in live_counts.values():
            if live > 1:
                raise RuntimeError("Document address already has a current incarnation")

    def _apply_document_actions(self, actions: Mapping[int, _DocumentAction], seq: Seq) -> None:
        state = self._state
        for id, action in actions.items():
            stored = state.documents.get(id)
            if action.create is not None:
                assert action.content is not None
                record: dict[str, Any] = {**action.create, "createdAt": seq}
                if action.retire:
                    record["retiredAt"] = seq
                stored = _StoredDocumentState(cast(DocumentRecord, record), [_DocumentRevision(seq, action.content)])
                state.record_types[id] = "document"
                state.documents[id] = stored

                key = _record_address_key(stored.record)
                address = state.document_addresses.get(key)
                if address is None:
                    address = _DocumentAddressIndex()
                    state.document_addresses[key] = address
                _insert_sorted(address.ids, id)
                _insert_map_id(state.document_ids_by_scope, _scope_key(stored.record["scope"]), id)
                self._next_id = max(self._next_id, id + 1)
            elif action.content is not None:
                assert stored is not None
                revision = _DocumentRevision(seq, action.content)
                if action.content["kind"] == "base" and _is_current_only(stored.record):
                    stored.revisions = [revision]
                else:
                    stored.revisions.append(revision)

            assert stored is not None
            if action.retire and action.create is None:
                stored.record = cast(DocumentRecord, {**stored.record, "retiredAt": seq})
            if action.retire and _is_current_only(stored.record):
                stored.revisions = []
            if action.create is not None or action.retire:
                address = state.document_addresses[_record_address_key(stored.record)]
                if action.retire and address.current_id == id:
                    address.current_id = None
                if action.create is not None and not action.retire:
                    address.current_id = id_from_number(id)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryStorage is closed")
