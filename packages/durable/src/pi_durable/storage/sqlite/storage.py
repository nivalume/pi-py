"""Portable SQLite implementation of the storage contract (port of ``storage/sqlite/storage.ts``).

The tables, indexes, column values and record JSON are upstream's, so a database written by TypeScript
pi-durable opens here and the other way around: records are ``JSON.stringify`` text
(:func:`~pi_durable.storage.codec.json_stringify`; pi-ai messages in entries as their camelCase JSON),
indexed strings (task kinds, request IDs, document kinds and keys) are stored JSON-encoded so lone
UTF-16 surrogates stay lossless, and ``next_id`` is TEXT.

:class:`SqliteStorage` runs on any :class:`~pi_chord.sqlite.SqliteDatabase`;
:func:`~pi_durable.storage.sqlite.stdlib.open_sqlite_storage` opens one on Python's ``sqlite3``.
Statements of one transaction may be issued before the previous ones settle (in call order, through the
transaction handle), so an adapter that executes on another thread pays one hop per batch of
independent statements rather than one per statement.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, TypeVar, cast

from pi_chord.context import Context
from pi_chord.delta import apply
from pi_chord.sqlite import SqliteDatabase, SqliteExecutor, SqliteRow, SqliteValue

from ...errors import StorageRequestError
from ...ids import MAX_SAFE_INTEGER, seq_from_number
from ...types import (
    AnyTaskRecord,
    ConversationId,
    ConversationQuery,
    ConversationRecord,
    Cursor,
    DocumentAddress,
    DocumentContent,
    DocumentCopySource,
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
from ..codec import decode_entry, encode_entry, json_parse, json_stringify
from ..scan import ScanStart, next_cursor, scan_start
from .migrations import apply_sqlite_migrations_for_schema
from .schema import SqliteSchema, create_sqlite_schema

__all__ = ["SqliteStorage", "SqliteStorageOptions"]

T = TypeVar("T")
_TableName = Literal["conversation", "entry", "task", "submission", "document"]
_INFINITY = float("inf")
#: Stands for the commit sequence in an entry row's parameters.
_COMMIT_SEQ: Any = object()


class SqliteStorageOptions(TypedDict, total=False):
    """Keyword arguments of :meth:`SqliteStorage.open`."""

    #: Prefix for every table and index in the durable schema. It must be a SQLite identifier and must not
    #: use SQLite's reserved ``sqlite_`` prefix. Existing data is not migrated when this changes.
    prefix: str


@dataclass(slots=True)
class _DocumentAction:
    create: DocumentCreate | None = None
    copy: DocumentCopySource | None = None
    content: DocumentContent | None = None
    retire: bool = False


@dataclass(frozen=True, slots=True)
class _AddressParts:
    kind: str
    scope_kind: str
    owner_id: int
    family: int
    key_value: str

    @property
    def params(self) -> tuple[SqliteValue, ...]:
        return (self.kind, self.scope_kind, self.owner_id, self.family, self.key_value)


@dataclass(slots=True)
class _CheckedAction:
    id: DocumentId
    action: _DocumentAction
    existing: DocumentRecord | None = None
    delta_version: Awaitable[SqliteRow | None] | None = None
    delta_row: SqliteRow | None = None
    parts: _AddressParts | None = None
    key: tuple[SqliteValue, ...] = field(default=())


def _encode_json(value: Any) -> str:
    return json_stringify(value)


def _encode_indexed_string(value: str) -> str:
    # Some SQLite bindings replace lone UTF-16 surrogates. JSON encoding keeps indexed identities lossless.
    return json_stringify(value)


def _cursor_id(cursor: Cursor | None) -> int | None:
    after = None if cursor is None else cursor.get("after")
    if after is None:
        return None
    if type(after) is not int or not -MAX_SAFE_INTEGER <= after <= MAX_SAFE_INTEGER:
        raise StorageRequestError("Invalid storage cursor")
    return after


def _page(values: list[Any], limit: int, order: ScanOrder = "ascending") -> Page[Any]:
    items = values[:limit]
    if len(values) <= limit:
        return {"items": items}
    return {"items": items, "next": next_cursor(items[-1]["id"], order)}


def _scan_sql(start: ScanStart) -> tuple[str, int, str]:
    """The ID condition, its parameter, and the ``ORDER BY`` direction of a table scan."""
    if start.order == "ascending":
        return "id > ?", -1 if start.after is None else start.after, "ASC"
    return "id < ?", MAX_SAFE_INTEGER if start.after is None else start.after, "DESC"


def _scope_columns(scope: DocumentScope) -> tuple[str, int]:
    if scope["kind"] == "session":
        return "session", 0
    if scope["kind"] == "conversation":
        return "conversation", scope["conversationId"]
    return "task", scope["taskId"]


def _address_parts(address: DocumentAddress | DocumentCreate | DocumentRecord) -> _AddressParts:
    scope_kind, owner_id = _scope_columns(address["scope"])
    key = address.get("key")
    return _AddressParts(
        _encode_indexed_string(address["kind"]),
        scope_kind,
        owner_id,
        0 if key is None else 1,
        _encode_indexed_string("" if key is None else key),
    )


def _is_alive_at(record: DocumentRecord, at: DocumentPoint) -> bool:
    retired = record.get("retiredAt")
    if at == "current":
        return retired is None
    return record["createdAt"] <= at and (retired is None or at < retired)


def _is_current_only(record: DocumentRecord | DocumentCreate) -> bool:
    return record["scope"]["kind"] != "conversation" or record.get("history") == "latest"


def _write_id(write: StorageWrite) -> int | None:
    kind = write["type"]
    if kind in ("conversation", "entry", "task", "submission"):
        return write["value"]["id"]  # type: ignore[typeddict-item]
    if kind in ("document.create", "document.copy"):
        return write["record"]["id"]  # type: ignore[typeddict-item]
    return None


async def _settle(awaitables: Sequence[Awaitable[Any]]) -> list[Any]:
    """Await statements issued together; raise the first failure only after every one settled."""
    if not awaitables:
        return []
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


class SqliteStorage:
    """Portable SQLite implementation of :class:`~pi_durable.types.Storage`.

    Open one with :meth:`open` over an owned :class:`~pi_chord.sqlite.SqliteDatabase`; closing the storage
    closes the database.
    """

    __slots__ = ("_admitted_reads", "_closed", "_closing", "_db", "_next_id", "_reads_drained", "_schema")

    def __init__(self, db: SqliteDatabase, schema: SqliteSchema, next_id: int) -> None:
        """Internal: use :meth:`open`, which migrates the schema and reads the ID allocator first."""
        self._db = db
        self._schema = schema
        self._next_id = next_id
        self._closed = False
        self._closing: asyncio.Future[None] | None = None
        self._admitted_reads = 0
        self._reads_drained: asyncio.Future[None] | None = None

    @classmethod
    async def open(cls, db: SqliteDatabase, *, prefix: str | None = None) -> SqliteStorage:
        """Initialize storage over an owned SQLite database facade, optionally under a table prefix.

        On failure the database is closed and the initialization error raised.
        """
        try:
            schema = create_sqlite_schema(prefix)
            await apply_sqlite_migrations_for_schema(db, schema)
            metadata = await db.get(
                f"SELECT next_id, next_seq FROM {schema.tables.durable_metadata} WHERE singleton = 1"
            )
            if metadata is None:
                raise RuntimeError("Durable SQLite metadata is missing")
            return cls(db, schema, int(metadata["next_id"]))
        except BaseException:
            try:
                await db.close()
            except Exception:
                pass  # Preserve the initialization failure.
            raise

    # ─── commits ─────────────────────────────────────────────────────────────

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        self._assert_open()
        document_actions = self._prepare_document_actions(writes)
        candidate_next_id = self._candidate_next_id(writes)
        # Encode before the transaction: an unencodable record fails the commit without touching the database.
        table_rows = [self._table_row(write) for write in writes]
        tables = self._schema.tables

        async def run(transaction: SqliteExecutor) -> Seq:
            # Every read the checks need is independent of the others: issue them together.
            metadata_read = transaction.get(
                f"SELECT next_id, next_seq FROM {tables.durable_metadata} WHERE singleton = 1"
            )
            claims, claim_reads = self._global_id_reads(transaction, writes)
            documents = [_CheckedAction(id, action) for id, action in document_actions.items()]
            document_reads = [
                transaction.get(f"SELECT record FROM {tables.documents} WHERE id = ?", item.id) for item in documents
            ]
            metadata, *rows = await _settle([metadata_read, *claim_reads, *document_reads])
            if metadata is None:
                raise RuntimeError("Durable SQLite metadata is missing")
            committed_seq = seq_from_number(metadata["next_seq"])
            self._check_global_ids(claims, rows[: len(claim_reads)])
            await self._check_document_actions(transaction, document_actions, documents, rows[len(claim_reads) :])
            pending: list[Awaitable[Any]] = []
            for row in table_rows:
                if row is not None:
                    table, id, sql, params = row
                    pending.append(self._claim_id(transaction, id, table))
                    pending.append(
                        transaction.run(sql, *(committed_seq if param is _COMMIT_SEQ else param for param in params))
                    )
            await self._apply_document_actions(transaction, document_actions, committed_seq, pending)
            pending.append(
                transaction.run(
                    f"UPDATE {tables.durable_metadata} SET next_id = ?, next_seq = ? WHERE singleton = 1",
                    str(max(int(metadata["next_id"]), candidate_next_id)),
                    committed_seq + 1,
                )
            )
            await _settle(pending)
            return committed_seq

        seq = await self._db.transaction(run)
        self._next_id = max(self._next_id, candidate_next_id)
        return seq

    def _table_row(self, write: StorageWrite) -> tuple[_TableName, int, str, tuple[Any, ...]] | None:
        """The claim and the statement of a table write. An entry's ``_COMMIT_SEQ`` parameter stands for the commit
        sequence, known only inside the transaction."""
        tables = self._schema.tables
        if write["type"] == "conversation":
            value = write["value"]
            owner = value.get("owner")
            return (
                "conversation",
                value["id"],
                f"INSERT INTO {tables.conversations} (id, owner_conversation_id, owner_task_id, record) "
                "VALUES (?, ?, ?, ?)",
                (
                    value["id"],
                    None if owner is None else owner["conversationId"],
                    None if owner is None else owner["taskId"],
                    _encode_json(value),
                ),
            )
        if write["type"] == "entry":
            value = write["value"]
            return (
                "entry",
                value["id"],
                f"INSERT INTO {tables.entries} (id, conversation_id, head, commit_seq, record) VALUES (?, ?, ?, ?, ?)",
                (
                    value["id"],
                    value["conversationId"],
                    value.get("head"),
                    _COMMIT_SEQ,
                    _encode_json(encode_entry(value)),
                ),
            )
        if write["type"] == "task":
            value = write["value"]
            return (
                "task",
                value["id"],
                f"INSERT INTO {tables.tasks} (id, conversation_id, kind, status, abort_requested, background, record)\n"
                "\t\t\t\t\t\tVALUES (?, ?, ?, ?, ?, ?, ?)\n"
                "\t\t\t\t\t\tON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id, kind = excluded.kind,\n"
                "\t\t\t\t\t\tstatus = excluded.status, abort_requested = excluded.abort_requested,\n"
                "\t\t\t\t\t\tbackground = excluded.background, record = excluded.record",
                (
                    value["id"],
                    value["conversationId"],
                    _encode_indexed_string(value["kind"]),
                    value["state"]["status"],
                    1 if value["abortRequested"] else 0,
                    1 if value["background"] else 0,
                    _encode_json(value),
                ),
            )
        if write["type"] == "submission":
            value = write["value"]
            request_id = value.get("requestId")
            return (
                "submission",
                value["id"],
                f"INSERT INTO {tables.submissions} (id, conversation_id, request_id, status, record) "
                "VALUES (?, ?, ?, ?, ?)\n"
                "\t\t\t\t\t\tON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id,\n"
                "\t\t\t\t\t\trequest_id = excluded.request_id, status = excluded.status, record = excluded.record",
                (
                    value["id"],
                    value["conversationId"],
                    None if request_id is None else _encode_indexed_string(request_id),
                    value["status"],
                    _encode_json(value),
                ),
            )
        return None

    async def mint_id(self) -> int:
        self._assert_open()
        if self._next_id > MAX_SAFE_INTEGER:
            raise RuntimeError("ID space is exhausted")
        id = self._next_id
        self._next_id += 1
        return id

    # ─── table reads ─────────────────────────────────────────────────────────

    async def conversation(self, id: ConversationId, context: Context) -> ConversationRecord | None:
        self._assert_open()
        row = await self._db.get(f"SELECT record FROM {self._schema.tables.conversations} WHERE id = ?", id)
        return None if row is None else json_parse(row["record"])

    async def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[ConversationRecord]:
        self._assert_open()
        start = scan_start(query.get("order"), cursor, "ascending")
        clause, param, direction = _scan_sql(start)
        clauses = [clause]
        params: list[SqliteValue] = [param]
        owner_conversation = query.get("ownerConversationId")
        if owner_conversation is not None:
            clauses.append("owner_conversation_id = ?")
            params.append(owner_conversation)
        owner_task = query.get("ownerTaskId")
        if owner_task is not None:
            clauses.append("owner_task_id = ?")
            params.append(owner_task)
        params.append(limit + 1)
        rows = await self._db.all(
            f"SELECT record FROM {self._schema.tables.conversations} WHERE {' AND '.join(clauses)} "
            f"ORDER BY id {direction} LIMIT ?",
            *params,
        )
        return _page([json_parse(row["record"]) for row in rows], limit, start.order)

    async def entry(self, id: EntryId, context: Context) -> StoredEntry | None:
        self._assert_open()
        self._admitted_reads += 1
        try:
            return await self._read_entry(None, id)
        finally:
            self._release_read()

    async def visible_entry(self, conversation_id: ConversationId, id: EntryId, context: Context) -> StoredEntry | None:
        self._assert_open()
        self._admitted_reads += 1
        try:
            if type(id) is not int:
                raise StorageRequestError("Storage.entry() requires an entry ID")
            return await self._read_entry(conversation_id, id)
        finally:
            self._release_read()

    async def find_latest_head_marker(
        self, conversation_id: ConversationId, at_or_before_entry_id: EntryId | None, context: Context
    ) -> HeadMarker | None:
        self._assert_open()
        self._admitted_reads += 1
        try:
            return await self._read_latest_head_marker(conversation_id, at_or_before_entry_id)
        finally:
            self._release_read()

    async def scan_entries(
        self, query: EntryQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[EntryRecord]:
        self._assert_open()
        self._admitted_reads += 1
        try:
            return await self._read_entries(query, limit, cursor)
        finally:
            self._release_read()

    def _conversation_read(self, id: int) -> Awaitable[SqliteRow | None]:
        return self._db.get(f"SELECT record FROM {self._schema.tables.conversations} WHERE id = ?", id)

    async def _read_entry(self, conversation_id: ConversationId | None, id: EntryId) -> StoredEntry | None:
        entry_read = self._db.get(f"SELECT record, commit_seq FROM {self._schema.tables.entries} WHERE id = ?", id)
        conversation: ConversationRecord | None = None
        if conversation_id is not None:
            # The conversation and the entry are read together; the conversation is checked first.
            conversation_row, row = await _settle([self._conversation_read(conversation_id), entry_read])
            if conversation_row is None:
                raise StorageRequestError(f"Unknown conversation: {conversation_id}")
            conversation = json_parse(conversation_row["record"])
        else:
            row = await entry_read
        if row is None:
            return None
        entry = decode_entry(json_parse(row["record"]))
        if conversation is not None:
            upper_entry_id: float = _INFINITY
            while conversation["id"] != entry["conversationId"]:
                parent = conversation.get("parent")
                if parent is None:
                    return None
                upper_entry_id = min(upper_entry_id, parent["at"])
                conversation = await self._read_conversation(parent["conversationId"])
                assert conversation is not None
            if entry["id"] > upper_entry_id:
                return None
        return {"entry": entry, "commitSeq": seq_from_number(row["commit_seq"])}

    async def _read_latest_head_marker(
        self, conversation_id: ConversationId, at_or_before_entry_id: EntryId | None
    ) -> HeadMarker | None:
        entries = self._schema.tables.entries
        current_id: int = conversation_id
        upper: int | None = at_or_before_entry_id
        first = True
        while True:
            # Each fork level reads its conversation (for the parent link) and its newest marker together.
            if upper is None:
                marker_read = self._db.get(
                    f"SELECT record FROM {entries} WHERE conversation_id = ? AND head IS NOT NULL "
                    "ORDER BY id DESC LIMIT 1",
                    current_id,
                )
            else:
                marker_read = self._db.get(
                    f"SELECT record FROM {entries} WHERE conversation_id = ? AND head IS NOT NULL AND id <= ? "
                    "ORDER BY id DESC LIMIT 1",
                    current_id,
                    upper,
                )
            conversation_row, row = await _settle([self._conversation_read(current_id), marker_read])
            if conversation_row is None:
                if first:
                    raise StorageRequestError(f"Unknown conversation: {conversation_id}")
                raise RuntimeError(f"Conversation {current_id} is missing")
            first = False
            if row is not None:
                return cast(HeadMarker, decode_entry(json_parse(row["record"])))
            conversation: ConversationRecord = json_parse(conversation_row["record"])
            parent = conversation.get("parent")
            if parent is None:
                return None
            upper = parent["at"] if upper is None else min(upper, parent["at"])
            current_id = parent["conversationId"]

    async def _read_entries(self, query: EntryQuery, limit: int, cursor: Cursor | None) -> Page[EntryRecord]:
        order, after = scan_start(query.get("order"), cursor, "descending")
        if order == "ascending":
            return await self._read_entries_ascending(query, limit, after)
        entries = self._schema.tables.entries
        min_entry_id = query.get("minEntryId")
        upper: int | None = query.get("maxEntryId")
        if after is not None:
            upper = min(MAX_SAFE_INTEGER if upper is None else upper, after - 1)
        values: list[EntryRecord] = []
        current_id: int = query["conversationId"]
        first = True
        while True:
            clauses = ["conversation_id = ?"]
            params: list[SqliteValue] = [current_id]
            if min_entry_id is not None:
                clauses.append("id >= ?")
                params.append(min_entry_id)
            if upper is not None:
                clauses.append("id <= ?")
                params.append(upper)
            params.append(limit + 1 - len(values))
            # Each fork level reads its conversation (for the parent link) and its entries together.
            rows_read = self._db.all(
                f"SELECT record FROM {entries} WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?", *params
            )
            conversation_row, rows = await _settle([self._conversation_read(current_id), rows_read])
            if conversation_row is None:
                if first:
                    raise StorageRequestError(f"Unknown conversation: {query['conversationId']}")
                raise RuntimeError(f"Conversation {current_id} is missing")
            first = False
            values.extend(decode_entry(json_parse(row["record"])) for row in rows)
            conversation: ConversationRecord = json_parse(conversation_row["record"])
            parent = conversation.get("parent")
            if len(values) > limit or parent is None:
                break
            upper = parent["at"] if upper is None else min(upper, parent["at"])
            if min_entry_id is not None and upper < min_entry_id:
                break
            current_id = parent["conversationId"]
        return _page(values, limit, "descending")

    async def _read_entries_ascending(self, query: EntryQuery, limit: int, after: int | None) -> Page[EntryRecord]:
        """Oldest first: the fork chain's segments from the root conversation forward, each up to its fork point."""
        entries = self._schema.tables.entries
        segments: list[tuple[int, int | None]] = []
        conversation = await self._read_conversation(query["conversationId"])
        if conversation is None:
            raise StorageRequestError(f"Unknown conversation: {query['conversationId']}")
        min_entry_id = query.get("minEntryId")
        upper: int | None = query.get("maxEntryId")
        while True:
            segments.append((conversation["id"], upper))
            parent = conversation.get("parent")
            if parent is None:
                break
            upper = parent["at"] if upper is None else min(upper, parent["at"])
            if min_entry_id is not None and upper < min_entry_id:
                break
            conversation = await self._read_conversation(parent["conversationId"])
            assert conversation is not None
        lower: int | None = min_entry_id
        if after is not None:
            lower = max(after + 1 if lower is None else lower, after + 1)
        values: list[EntryRecord] = []
        for segment_id, segment_upper in reversed(segments):
            clauses = ["conversation_id = ?"]
            params: list[SqliteValue] = [segment_id]
            if lower is not None:
                clauses.append("id >= ?")
                params.append(lower)
            if segment_upper is not None:
                clauses.append("id <= ?")
                params.append(segment_upper)
            params.append(limit + 1 - len(values))
            rows = await self._db.all(
                f"SELECT record FROM {entries} WHERE {' AND '.join(clauses)} ORDER BY id ASC LIMIT ?", *params
            )
            values.extend(decode_entry(json_parse(row["record"])) for row in rows)
            if len(values) > limit:
                break
        return _page(values, limit, "ascending")

    async def task(self, id: TaskId[Any], context: Context) -> AnyTaskRecord | None:
        self._assert_open()
        row = await self._db.get(f"SELECT record FROM {self._schema.tables.tasks} WHERE id = ?", id)
        return None if row is None else json_parse(row["record"])

    async def scan_tasks(
        self, query: TaskQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[AnyTaskRecord]:
        self._assert_open()
        start = scan_start(query.get("order"), cursor, "ascending")
        clause, param, direction = _scan_sql(start)
        clauses = [clause]
        params: list[SqliteValue] = [param]
        conversation_id = query.get("conversationId")
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        kind = query.get("kind")
        if kind is not None:
            clauses.append("kind = ?")
            params.append(_encode_indexed_string(kind))
        status = query.get("status")
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        abort_requested = query.get("abortRequested")
        if abort_requested is not None:
            clauses.append("abort_requested = ?")
            params.append(1 if abort_requested else 0)
        background = query.get("background")
        if background is not None:
            clauses.append("background = ?")
            params.append(1 if background else 0)
        params.append(limit + 1)
        rows = await self._db.all(
            f"SELECT record FROM {self._schema.tables.tasks} WHERE {' AND '.join(clauses)} "
            f"ORDER BY id {direction} LIMIT ?",
            *params,
        )
        return _page([json_parse(row["record"]) for row in rows], limit, start.order)

    async def submission(self, id: SubmissionId, context: Context) -> SubmissionRecord | None:
        self._assert_open()
        row = await self._db.get(f"SELECT record FROM {self._schema.tables.submissions} WHERE id = ?", id)
        return None if row is None else json_parse(row["record"])

    async def scan_submissions(
        self, query: SubmissionQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[SubmissionRecord]:
        self._assert_open()
        start = scan_start(query.get("order"), cursor, "ascending")
        clause, param, direction = _scan_sql(start)
        clauses = [clause]
        params: list[SqliteValue] = [param]
        conversation_id = query.get("conversationId")
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        status = query.get("status")
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        params.append(limit + 1)
        rows = await self._db.all(
            f"SELECT record FROM {self._schema.tables.submissions} WHERE {' AND '.join(clauses)} "
            f"ORDER BY id {direction} LIMIT ?",
            *params,
        )
        return _page([json_parse(row["record"]) for row in rows], limit, start.order)

    async def submission_by_request(
        self, conversation_id: ConversationId, request_id: str, context: Context
    ) -> SubmissionRecord | None:
        self._assert_open()
        row = await self._db.get(
            f"SELECT record FROM {self._schema.tables.submissions} WHERE conversation_id = ? AND request_id = ?",
            conversation_id,
            _encode_indexed_string(request_id),
        )
        return None if row is None else json_parse(row["record"])

    # ─── documents ───────────────────────────────────────────────────────────

    async def find_document(
        self, address: DocumentAddress, at: DocumentPoint, context: Context
    ) -> DocumentRecord | None:
        self._assert_open()
        documents = self._schema.tables.documents
        parts = _address_parts(address)
        if at == "current":
            row = await self._db.get(
                f"SELECT record FROM {documents}\n"
                "\t\t\t\t\tWHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?\n"
                "\t\t\t\t\tAND retired_at IS NULL ORDER BY created_at DESC LIMIT 1",
                *parts.params,
            )
        else:
            row = await self._db.get(
                f"SELECT record FROM {documents}\n"
                "\t\t\t\t\tWHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?\n"
                "\t\t\t\t\tAND created_at <= ? AND (retired_at IS NULL OR retired_at > ?)\n"
                "\t\t\t\t\tORDER BY created_at DESC LIMIT 1",
                *parts.params,
                at,
                at,
            )
        return None if row is None else json_parse(row["record"])

    async def document(self, id: DocumentId, at: DocumentPoint, context: Context) -> StoredDocument | None:
        self._assert_open()
        # The record and revision queries must observe one committed state; a commit between them can replace the base.

        async def read(transaction: SqliteExecutor) -> StoredDocument | None:
            return await self._materialize_document(transaction, id, at)

        return await self._db.transaction(read)

    async def scan_documents(
        self, query: DocumentQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[DocumentRecord]:
        self._assert_open()
        scope_kind, owner_id = _scope_columns(query["scope"])
        after = _cursor_id(cursor)
        clauses = ["scope_kind = ?", "owner_id = ?", "id > ?"]
        params: list[SqliteValue] = [scope_kind, owner_id, -1 if after is None else after]
        kind = query.get("kind")
        if kind is not None:
            clauses.append("kind = ?")
            params.append(_encode_indexed_string(kind))
        at = query["at"]
        if at == "current":
            clauses.append("retired_at IS NULL")
        else:
            clauses.extend(("created_at <= ?", "(retired_at IS NULL OR retired_at > ?)"))
            params.extend((at, at))
        params.append(limit + 1)
        rows = await self._db.all(
            f"SELECT record FROM {self._schema.tables.documents} WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?",
            *params,
        )
        return _page([json_parse(row["record"]) for row in rows], limit)

    # ─── lifecycle ───────────────────────────────────────────────────────────

    async def close(self, context: Context) -> None:
        """Close the database once admitted reads finished; repeated calls wait for the same close."""
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.ensure_future(self._close_database())
        await asyncio.shield(self._closing)

    async def _close_database(self) -> None:
        if self._admitted_reads > 0:
            self._reads_drained = asyncio.get_running_loop().create_future()
            await self._reads_drained
        await self._db.close()

    def _release_read(self) -> None:
        """End a read that issues several queries. Close waits for admitted reads, so their later queries never reach
        a closed database. Single-query reads and transactions are already ordered before close by the database."""
        self._admitted_reads -= 1
        drained = self._reads_drained
        if self._admitted_reads == 0 and drained is not None and not drained.done():
            drained.set_result(None)

    async def _read_conversation(self, id: ConversationId) -> ConversationRecord | None:
        row = await self._db.get(f"SELECT record FROM {self._schema.tables.conversations} WHERE id = ?", id)
        return None if row is None else json_parse(row["record"])

    async def _materialize_document(
        self, executor: SqliteExecutor, id: DocumentId, at: DocumentPoint
    ) -> StoredDocument | None:
        tables = self._schema.tables
        upper = MAX_SAFE_INTEGER if at == "current" else at
        # The record and the newest base at or before `at` are read together; the record decides first.
        row, base = await _settle(
            [
                executor.get(f"SELECT record FROM {tables.documents} WHERE id = ?", id),
                executor.get(
                    f"SELECT seq, kind, version, content FROM {tables.document_revisions}\n"
                    "\t\t\t\tWHERE document_id = ? AND kind = 'base' AND seq <= ? ORDER BY seq DESC LIMIT 1",
                    id,
                    upper,
                ),
            ]
        )
        if row is None:
            return None
        record: DocumentRecord = json_parse(row["record"])
        if at != "current" and _is_current_only(record):
            raise StorageRequestError(f"Document {id} does not retain historical content")
        if not _is_alive_at(record, at):
            return None
        if base is None:
            raise RuntimeError(f"Document {id} is missing a required base")
        value = json_parse(base["content"])
        tail = await executor.all(
            f"SELECT seq, kind, version, content FROM {tables.document_revisions}\n"
            "\t\t\t\tWHERE document_id = ? AND seq > ? AND seq <= ? ORDER BY seq",
            id,
            base["seq"],
            upper,
        )
        for revision in tail:
            if revision["kind"] != "delta" or revision["version"] != base["version"]:
                raise RuntimeError(f"Document {id} crosses a stored version boundary without a base")
            value = apply(value, json_parse(revision["content"]))
        return {"record": record, "version": base["version"], "value": value, "deltasSinceBase": len(tail)}

    def _candidate_next_id(self, writes: Sequence[StorageWrite]) -> int:
        next_id = self._next_id
        for write in writes:
            id = _write_id(write)
            if id is not None:
                next_id = max(next_id, id + 1)
        return next_id

    def _global_id_reads(
        self, executor: SqliteExecutor, writes: Sequence[StorageWrite]
    ) -> tuple[list[tuple[_TableName, int]], list[Awaitable[SqliteRow | None]]]:
        """The IDs a commit claims, and the reads of their current owners."""
        checked: list[tuple[_TableName, int]] = []
        lookups: list[Awaitable[SqliteRow | None]] = []
        sql = f"SELECT record_type FROM {self._schema.tables.record_ids} WHERE id = ?"
        for write in writes:
            kind = write["type"]
            if kind in ("document.change", "document.retire"):
                continue
            table: _TableName = "document" if kind in ("document.create", "document.copy") else cast(_TableName, kind)
            id = _write_id(write)
            assert id is not None
            checked.append((table, id))
            lookups.append(executor.get(sql, id))
        return checked, lookups

    def _check_global_ids(self, checked: Sequence[tuple[_TableName, int]], rows: Sequence[SqliteRow | None]) -> None:
        claimed: dict[int, _TableName] = {}
        for (table, id), row in zip(checked, rows, strict=True):
            existing = None if row is None else row["record_type"]
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

    def _prepare_document_actions(self, writes: Sequence[StorageWrite]) -> dict[DocumentId, _DocumentAction]:
        actions: dict[DocumentId, _DocumentAction] = {}
        for write in writes:
            kind = write["type"]
            if kind == "document.create" or kind == "document.copy":
                id = write["record"]["id"]  # type: ignore[typeddict-item]
            elif kind == "document.change" or kind == "document.retire":
                id = write["id"]  # type: ignore[typeddict-item]
            else:
                continue
            action = actions.get(id)
            if action is None:
                action = _DocumentAction()
                actions[id] = action
            if write["type"] == "document.create":
                if action.create is not None or action.content is not None or action.copy is not None:
                    raise RuntimeError(f"Document {id} has more than one content command")
                action.create = write["record"]
                action.content = write["content"]
            elif write["type"] == "document.copy":
                if action.create is not None or action.content is not None or action.copy is not None:
                    raise RuntimeError(f"Document {id} has more than one content command")
                action.create = write["record"]
                action.copy = write["source"]
            elif write["type"] == "document.change":
                if action.content is not None or action.copy is not None:
                    raise RuntimeError(f"Document {id} has more than one content command")
                action.content = write["content"]
            else:
                if action.retire:
                    raise RuntimeError(f"Document {id} is retired more than once")
                action.retire = True
        return actions

    async def _check_document_actions(
        self,
        executor: SqliteExecutor,
        actions: Mapping[DocumentId, _DocumentAction],
        checked: Sequence[_CheckedAction],
        rows: Sequence[SqliteRow | None],
    ) -> None:
        """Check document commands against their current rows (``rows``, read for ``checked`` in action order)."""
        if not actions:
            return
        tables = self._schema.tables
        current_sql = (
            f"SELECT id FROM {tables.documents}\n"
            "\t\t\t\tWHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ? "
            "AND retired_at IS NULL\n\t\t\t\tLIMIT 1"
        )
        current: dict[tuple[SqliteValue, ...], Awaitable[SqliteRow | None]] = {}
        for item, row in zip(checked, rows, strict=True):
            item.existing = None if row is None else json_parse(row["record"])
            if item.action.content is not None and item.action.content["kind"] == "delta":
                item.delta_version = executor.get(
                    f"SELECT version FROM {tables.document_revisions} WHERE document_id = ? ORDER BY seq DESC LIMIT 1",
                    item.id,
                )
            record = item.action.create if item.action.create is not None else item.existing
            if record is not None:
                item.parts = _address_parts(record)
                item.key = item.parts.params
                if item.key not in current:
                    current[item.key] = executor.get(current_sql, *item.key)
        with_delta = [item for item in checked if item.delta_version is not None]
        delta_rows = await _settle([item.delta_version for item in with_delta if item.delta_version is not None])
        for item, delta_row in zip(with_delta, delta_rows, strict=True):
            item.delta_row = delta_row
        keys = list(current)
        resolved_current = dict(zip(keys, await _settle([current[key] for key in keys]), strict=True))

        live_counts: dict[tuple[SqliteValue, ...], int] = {}
        for item in checked:
            action = item.action
            copy = action.copy
            if copy is not None and copy["id"] in actions:
                raise RuntimeError(f"Document copy {item.id} source is changed in the copy batch")
            existing = item.existing
            if action.create is None and existing is None:
                raise RuntimeError(f"Unknown document: {item.id}")
            if action.create is not None and existing is not None:
                raise RuntimeError(f"Document {item.id} already exists")
            if existing is not None and existing.get("retiredAt") is not None:
                raise RuntimeError(f"Document {item.id} is retired")
            if item.delta_version is not None:
                assert action.content is not None
                previous = item.delta_row
                if previous is None:
                    raise RuntimeError(f"Document {item.id} delta has no base")
                if previous["version"] != action.content["version"]:
                    raise RuntimeError(f"Document {item.id} version transition requires a base")
            live = live_counts.get(item.key)
            if live is None:
                live = 0 if resolved_current[item.key] is None else 1
            if action.retire and existing is not None:
                live -= 1
            if action.create is not None and not action.retire:
                live += 1
            live_counts[item.key] = live
        for live in live_counts.values():
            if live > 1:
                raise RuntimeError("Document address already has a current incarnation")

    def _claim_id(self, executor: SqliteExecutor, id: int, table: _TableName) -> Awaitable[None]:
        return executor.run(
            f"INSERT OR IGNORE INTO {self._schema.tables.record_ids} (id, record_type) VALUES (?, ?)", id, table
        )

    async def _apply_document_actions(
        self,
        executor: SqliteExecutor,
        actions: Mapping[DocumentId, _DocumentAction],
        seq: Seq,
        pending: list[Awaitable[Any]],
    ) -> None:
        tables = self._schema.tables
        for id, action in actions.items():
            content: DocumentContent | None = action.content
            if action.copy is not None:
                source = action.copy
                try:
                    stored = await self._materialize_document(executor, source["id"], source["at"])
                    if stored is None:
                        raise RuntimeError(f"Fork source document {source['id']} cannot be read")
                    create = action.create
                    assert create is not None
                    source_record = stored["record"]
                    if (
                        source_record["scope"]["kind"] != "conversation"
                        or create["scope"]["kind"] != "conversation"
                        or source_record["kind"] != create["kind"]
                        or source_record.get("key") != create.get("key")
                        or source_record.get("history") != create.get("history")
                        or source_record.get("fork") != create.get("fork")
                    ):
                        raise RuntimeError(f"Fork source document {source['id']} does not match the copied record")
                    content = {"kind": "base", "version": stored["version"], "value": stored["value"]}
                except Exception as error:
                    raise RuntimeError(f"Document copy {id} was rejected") from error
            if action.create is not None:
                record_value: dict[str, Any] = {**action.create, "createdAt": seq}
                if action.retire:
                    record_value["retiredAt"] = seq
                record = cast(DocumentRecord, record_value)
                parts = _address_parts(record)
                pending.append(self._claim_id(executor, id, "document"))
                pending.append(
                    executor.run(
                        f"INSERT INTO {tables.documents}\n"
                        "\t\t\t\t\t\t(id, kind, family, key_value, scope_kind, owner_id, created_at, retired_at, record)\n"
                        "\t\t\t\t\t\tVALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        id,
                        parts.kind,
                        parts.family,
                        parts.key_value,
                        parts.scope_kind,
                        parts.owner_id,
                        seq,
                        seq if action.retire else None,
                        _encode_json(record),
                    )
                )
            else:
                row = await executor.get(f"SELECT record FROM {tables.documents} WHERE id = ?", id)
                assert row is not None
                record = json_parse(row["record"])

            if content is not None:
                if content["kind"] == "base" and _is_current_only(record):
                    pending.append(executor.run(f"DELETE FROM {tables.document_revisions} WHERE document_id = ?", id))
                if content["kind"] == "base":
                    encoded = _encode_json(content["value"])
                else:
                    encoded = _encode_json(content["ops"])
                pending.append(
                    executor.run(
                        f"INSERT INTO {tables.document_revisions} (document_id, seq, kind, version, content) "
                        "VALUES (?, ?, ?, ?, ?)",
                        id,
                        seq,
                        content["kind"],
                        content["version"],
                        encoded,
                    )
                )

            if action.retire:
                if action.create is None:
                    record = cast(DocumentRecord, {**record, "retiredAt": seq})
                    pending.append(
                        executor.run(
                            f"UPDATE {tables.documents} SET retired_at = ?, record = ? WHERE id = ?",
                            seq,
                            _encode_json(record),
                            id,
                        )
                    )
                if _is_current_only(record):
                    pending.append(executor.run(f"DELETE FROM {tables.document_revisions} WHERE document_id = ?", id))

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("SqliteStorage is closed")
