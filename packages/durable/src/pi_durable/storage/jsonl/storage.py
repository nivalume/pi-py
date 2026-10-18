"""Portable JSONL implementation of the storage contract (port of ``storage/jsonl/storage.ts``).

The directory holds ``main.jsonl``, one commit marker per line, and sidecars: ``task-<id>.jsonl`` for live
task records and ``doc-<id>.jsonl`` for document content. A commit appends its sidecar records first
(flushing them when ``fsync`` is set), then its marker; only records a marker confirms are state, so a
torn or unconfirmed tail is truncated on open. Current-only document history and terminal tasks' sidecars
are reclaimed once a marker published the state that makes them obsolete. The lines are upstream's
``JSON.stringify`` text (:func:`~pi_durable.storage.codec.json_stringify`), so TypeScript pi-durable and
this port read each other's directories.

State lives in a :class:`~pi_durable.storage.memory.MemoryStorage`, rebuilt from the files on open; the
files are reached only through a :class:`~pi_durable.env.FileSystem`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypedDict, TypeVar, cast

from pi_chord.context import Context

from ...env.types import FileError, FileSystem
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
    DocumentId,
    DocumentPoint,
    DocumentQuery,
    DocumentRecord,
    EntryId,
    EntryQuery,
    EntryRecord,
    HeadMarker,
    Page,
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
from ..codec import decode_entry, decode_ops, encode_entry, json_parse, json_stringify
from ..memory import MemoryStorage

__all__ = ["JsonlCorruptionError", "JsonlStorage", "JsonlStorageOptions", "JsonlStoragePoisonedError"]

T = TypeVar("T")

FORMAT_VERSION = 1
MAIN_FILE = "main.jsonl"
RECLAIM_SUFFIX = ".reclaim"

_SIDECAR_FILE = re.compile(r"(?:doc|task)-(?:0|[1-9]\d*)\.jsonl")
_RECLAIM_FILE = re.compile(r"(?:doc|task)-(?:0|[1-9]\d*)\.jsonl\.reclaim")


class JsonlStorageOptions(TypedDict, total=False):
    """Keyword arguments of :meth:`JsonlStorage.open`."""

    #: Flush every affected sidecar before appending the main marker. Defaults to false.
    fsync: bool


class JsonlCorruptionError(Exception):
    """The files hold a complete line that is not valid committed state."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class JsonlStoragePoisonedError(Exception):
    """A commit failed after its I/O began; the storage must be reopened, which recovers the confirmed state."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__("JSONL storage is poisoned and must be reopened")
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class _ParsedLine(Generic[T]):
    value: T
    start: int


@dataclass(frozen=True, slots=True)
class _ParsedFile(Generic[T]):
    path: str
    lines: list[_ParsedLine[T]]


@dataclass(frozen=True, slots=True)
class _EncodedCommit:
    marker: str
    sidecars: dict[str, str]


def _is_object(value: object) -> bool:
    return type(value) is dict


def _is_safe_integer(value: object) -> bool:
    return type(value) is int and -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER  # type: ignore[operator]


def _sidecar_file_name(kind: str, id: int) -> str:
    return f"{kind}-{id}.jsonl"


def _is_current_only(record: Mapping[str, Any]) -> bool:
    return record["scope"]["kind"] != "conversation" or record.get("history") == "latest"


def _sidecar_key(file: str, seq: int, ordinal: int) -> tuple[str, int, int]:
    return (file, seq, ordinal)


def _json_line(value: Any) -> str:
    return f"{json_stringify(value)}\n"


def _error_from_file(action: str, error: FileError) -> RuntimeError:
    failure = RuntimeError(f"JSONL {action} failed: {error.message}")
    failure.__cause__ = error
    return failure


def _parse_json(text: str, description: str) -> Any:
    try:
        return json_parse(text)
    except ValueError as error:
        raise JsonlCorruptionError(f"Malformed complete {description}", error) from None


def _validate_main_operation(value: Any, description: str) -> dict[str, Any]:
    if not _is_object(value) or type(value.get("type")) is not str:
        raise JsonlCorruptionError(f"Invalid write in {description}")
    kind = value["type"]
    if kind in ("conversation", "entry", "submission"):
        if not _is_object(value.get("value")) or not _is_safe_integer(value["value"].get("id")):
            raise JsonlCorruptionError(f"Invalid {kind} write in {description}")
        return value
    if kind == "task":
        task = value.get("value")
        if (
            not _is_object(task)
            or not _is_safe_integer(task.get("id"))
            or not _is_object(task.get("state"))
            or task["state"].get("status") != "terminal"
        ):
            raise JsonlCorruptionError(f"Invalid terminal task write in {description}")
        return value
    if kind == "document.retire":
        if not _is_safe_integer(value.get("id")):
            raise JsonlCorruptionError(f"Invalid document retirement in {description}")
        return value
    if kind == "task.sidecar":
        if not _is_safe_integer(value.get("id")) or not _is_safe_integer(value.get("ordinal")) or value["ordinal"] < 0:
            raise JsonlCorruptionError(f"Invalid task sidecar write in {description}")
        return value
    if kind == "document.create":
        record = value.get("record")
        if (
            not _is_object(record)
            or not _is_safe_integer(record.get("id"))
            or not _is_safe_integer(value.get("ordinal"))
            or value["ordinal"] < 0
        ):
            raise JsonlCorruptionError(f"Invalid document creation in {description}")
        return value
    if kind == "document.change":
        if not _is_safe_integer(value.get("id")) or not _is_safe_integer(value.get("ordinal")) or value["ordinal"] < 0:
            raise JsonlCorruptionError(f"Invalid document change in {description}")
        return value
    raise JsonlCorruptionError(f"Unknown write type in {description}")


def _parse_main_marker(text: str, line: int) -> dict[str, Any]:
    description = f"{MAIN_FILE} line {line}"
    value = _parse_json(text, description)
    if (
        not _is_object(value)
        or value.get("format") != FORMAT_VERSION
        or type(value.get("format")) is not int
        or value.get("type") != "commit"
        or not _is_safe_integer(value.get("seq"))
        or value["seq"] < 1
        or type(value.get("writes")) is not list
    ):
        raise JsonlCorruptionError(f"Invalid commit marker in {description}")
    return {
        "format": FORMAT_VERSION,
        "type": "commit",
        "seq": seq_from_number(value["seq"]),
        "writes": [_validate_main_operation(write, description) for write in value["writes"]],
    }


def _validate_document_content(value: Any, description: str) -> dict[str, Any]:
    if not _is_object(value) or not _is_safe_integer(value.get("version")) or value["version"] < 1:
        raise JsonlCorruptionError(f"Invalid document content in {description}")
    if value.get("kind") == "base" and _is_object(value.get("value")):
        return value
    if value.get("kind") == "delta" and type(value.get("ops")) is list:
        return value
    raise JsonlCorruptionError(f"Invalid document content in {description}")


def _parse_sidecar_record(text: str, file: str, line: int) -> dict[str, Any]:
    description = f"{file} line {line}"
    value = _parse_json(text, description)
    if (
        not _is_object(value)
        or value.get("format") != FORMAT_VERSION
        or type(value.get("format")) is not int
        or value.get("type") != "record"
        or not _is_safe_integer(value.get("seq"))
        or value["seq"] < 1
        or not _is_safe_integer(value.get("ordinal"))
        or value["ordinal"] < 0
        or not _is_object(value.get("payload"))
        or type(value["payload"].get("type")) is not str
    ):
        raise JsonlCorruptionError(f"Invalid sidecar record in {description}")
    payload = value["payload"]
    if payload["type"] == "task":
        task = payload.get("value")
        if (
            not _is_object(task)
            or not _is_safe_integer(task.get("id"))
            or not _is_object(task.get("state"))
            or task["state"].get("status") == "terminal"
        ):
            raise JsonlCorruptionError(f"Invalid live task record in {description}")
    elif payload["type"] == "document":
        if not _is_safe_integer(payload.get("id")):
            raise JsonlCorruptionError(f"Invalid document record in {description}")
        _validate_document_content(payload.get("content"), description)
    else:
        raise JsonlCorruptionError(f"Unknown sidecar record type in {description}")
    return value


def _decode_content(content: Mapping[str, Any]) -> DocumentContent:
    if content["kind"] == "delta":
        return {"kind": "delta", "version": content["version"], "ops": decode_ops(content["ops"])}
    return cast(DocumentContent, content)


def _encode_main_write(write: StorageWrite) -> Any:
    if write["type"] == "entry":
        return {"type": "entry", "value": encode_entry(write["value"])}
    return write


class JsonlStorage:
    """Portable JSONL implementation of :class:`~pi_durable.types.Storage`. Open one with :meth:`open`."""

    __slots__ = (
        "_closed",
        "_current_only_documents",
        "_directory",
        "_fs",
        "_fsync",
        "_live_task_sidecars",
        "_main_path",
        "_memory",
        "_poison_error",
    )

    def __init__(self, fs: FileSystem, directory: str, main_path: str, fsync: bool) -> None:
        """Internal: use :meth:`open`, which recovers the committed state first."""
        self._fs = fs
        self._directory = directory
        self._main_path = main_path
        self._fsync = fsync
        self._memory = MemoryStorage()
        self._current_only_documents: set[int] = set()
        self._live_task_sidecars: set[int] = set()
        self._closed = False
        self._poison_error: JsonlStoragePoisonedError | None = None

    @classmethod
    async def open(cls, directory: str, fs: FileSystem, context: Context, *, fsync: bool = False) -> JsonlStorage:
        """Open or create a JSONL storage directory using the supplied filesystem.

        With ``fsync``, every affected sidecar is flushed before the main marker is appended, and ``main.jsonl``
        is flushed before obsolete sidecars are reclaimed.
        """
        absolute = await fs.absolute_path(directory, context)
        if absolute["ok"] is False:
            raise _error_from_file("path resolution", absolute["error"])
        created = await fs.create_dir(absolute["value"], {"recursive": True}, context)
        if created["ok"] is False:
            raise _error_from_file("directory creation", created["error"])
        main_path = await fs.join_path([absolute["value"], MAIN_FILE], context)
        if main_path["ok"] is False:
            raise _error_from_file("path join", main_path["error"])
        storage = cls(fs, absolute["value"], main_path["value"], fsync)
        await storage._recover(context)
        return storage

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        self._assert_usable()
        prepared = self._memory.prepare_commit(writes)
        encoded = self._encode_commit(prepared.seq, prepared.writes)
        reclamations = self._plan_reclamations(prepared.writes, encoded)
        sidecars = [
            (file, content, await self._resolve_file(file, context)) for file, content in encoded.sidecars.items()
        ]
        for file, content, path in sidecars:
            appended = await self._fs.append_file(path, content, context)
            if appended["ok"] is False:
                raise self._poison(_error_from_file(f"append to {file}", appended["error"]))
        if self._fsync:
            for file, _content, path in sidecars:
                flushed = await self._fs.flush_file(path, context)
                if flushed["ok"] is False:
                    raise self._poison(_error_from_file(f"flush of {file}", flushed["error"]))
        marker = await self._fs.append_file(self._main_path, encoded.marker, context)
        if marker["ok"] is False:
            raise self._poison(_error_from_file(f"append to {MAIN_FILE}", marker["error"]))
        seq = prepared.apply()
        self._adopt_sidecar_state(prepared.writes)
        await self._reclaim_sidecars(reclamations, context)
        return seq

    async def mint_id(self) -> int:
        return await self._store().mint_id()

    async def conversation(self, id: ConversationId, context: Context) -> ConversationRecord | None:
        return await self._store().conversation(id, context)

    async def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[ConversationRecord]:
        return await self._store().scan_conversations(query, limit, cursor, context)

    async def entry(self, id: EntryId, context: Context) -> StoredEntry | None:
        return await self._store().entry(id, context)

    async def visible_entry(self, conversation_id: ConversationId, id: EntryId, context: Context) -> StoredEntry | None:
        store = self._store()
        if type(id) is not int:
            raise StorageRequestError("Storage.entry() requires an entry ID")
        return await store.visible_entry(conversation_id, id, context)

    async def find_latest_head_marker(
        self, conversation_id: ConversationId, at_or_before_entry_id: EntryId | None, context: Context
    ) -> HeadMarker | None:
        return await self._store().find_latest_head_marker(conversation_id, at_or_before_entry_id, context)

    async def scan_entries(
        self, query: EntryQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[EntryRecord]:
        return await self._store().scan_entries(query, limit, cursor, context)

    async def task(self, id: TaskId[Any], context: Context) -> AnyTaskRecord | None:
        return await self._store().task(id, context)

    async def scan_tasks(
        self, query: TaskQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[AnyTaskRecord]:
        return await self._store().scan_tasks(query, limit, cursor, context)

    async def submission(self, id: SubmissionId, context: Context) -> SubmissionRecord | None:
        return await self._store().submission(id, context)

    async def scan_submissions(
        self, query: SubmissionQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[SubmissionRecord]:
        return await self._store().scan_submissions(query, limit, cursor, context)

    async def submission_by_request(
        self, conversation_id: ConversationId, request_id: str, context: Context
    ) -> SubmissionRecord | None:
        return await self._store().submission_by_request(conversation_id, request_id, context)

    async def find_document(
        self, address: DocumentAddress, at: DocumentPoint, context: Context
    ) -> DocumentRecord | None:
        return await self._store().find_document(address, at, context)

    async def document(self, id: DocumentId, at: DocumentPoint, context: Context) -> StoredDocument | None:
        return await self._store().document(id, at, context)

    async def scan_documents(
        self, query: DocumentQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Page[DocumentRecord]:
        return await self._store().scan_documents(query, limit, cursor, context)

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        await self._memory.close(context)

    # ─── publication ─────────────────────────────────────────────────────────

    def _encode_commit(self, seq: Seq, writes: Sequence[StorageWrite]) -> _EncodedCommit:
        main_writes: list[Any] = []
        records: dict[str, list[dict[str, Any]]] = {}
        next_ordinal = 0

        def add_sidecar(file: str, payload: dict[str, Any]) -> int:
            nonlocal next_ordinal
            ordinal = next_ordinal
            next_ordinal += 1
            records.setdefault(file, []).append(
                {"format": FORMAT_VERSION, "type": "record", "seq": seq, "ordinal": ordinal, "payload": payload}
            )
            return ordinal

        for write in writes:
            if write["type"] in ("conversation", "entry", "submission", "document.retire"):
                main_writes.append(_encode_main_write(write))
            elif write["type"] == "task":
                value = write["value"]
                if value["state"]["status"] == "terminal":
                    main_writes.append(write)
                else:
                    ordinal = add_sidecar(_sidecar_file_name("task", value["id"]), {"type": "task", "value": value})
                    main_writes.append({"type": "task.sidecar", "id": value["id"], "ordinal": ordinal})
            elif write["type"] == "document.create":
                record = write["record"]
                ordinal = add_sidecar(
                    _sidecar_file_name("doc", record["id"]),
                    {"type": "document", "id": record["id"], "content": write["content"]},
                )
                main_writes.append({"type": "document.create", "record": record, "ordinal": ordinal})
            elif write["type"] == "document.change":
                ordinal = add_sidecar(
                    _sidecar_file_name("doc", write["id"]),
                    {"type": "document", "id": write["id"], "content": write["content"]},
                )
                main_writes.append({"type": "document.change", "id": write["id"], "ordinal": ordinal})

        sidecars = {
            file: "".join(_json_line(record) for record in file_records) for file, file_records in records.items()
        }
        marker = {"format": FORMAT_VERSION, "type": "commit", "seq": seq, "writes": main_writes}
        return _EncodedCommit(_json_line(marker), sidecars)

    def _plan_reclamations(self, writes: Sequence[StorageWrite], encoded: _EncodedCommit) -> dict[str, str]:
        created_current_only: set[int] = set()
        retired: set[int] = set()
        based: set[int] = set()
        final_tasks: dict[int, AnyTaskRecord] = {}
        for write in writes:
            if write["type"] == "document.create":
                if _is_current_only(write["record"]):
                    created_current_only.add(write["record"]["id"])
            elif write["type"] == "document.change":
                if write["content"]["kind"] == "base":
                    based.add(write["id"])
            elif write["type"] == "document.retire":
                retired.add(write["id"])
            elif write["type"] == "task":
                final_tasks[write["value"]["id"]] = write["value"]

        replacements: dict[str, str] = {}

        def is_current_only_document(id: int) -> bool:
            return id in self._current_only_documents or id in created_current_only

        for id in retired:
            if is_current_only_document(id):
                replacements[_sidecar_file_name("doc", id)] = ""
        for id in based:
            if not is_current_only_document(id) or id in retired:
                continue
            file = _sidecar_file_name("doc", id)
            content = encoded.sidecars.get(file)
            if content is not None:
                replacements[file] = content
        for id, task in final_tasks.items():
            if task["state"]["status"] == "terminal" and (
                id in self._live_task_sidecars or _sidecar_file_name("task", id) in encoded.sidecars
            ):
                replacements[_sidecar_file_name("task", id)] = ""
        return replacements

    def _adopt_sidecar_state(self, writes: Sequence[StorageWrite]) -> None:
        for write in writes:
            if write["type"] == "document.create":
                if _is_current_only(write["record"]):
                    self._current_only_documents.add(write["record"]["id"])
            elif write["type"] == "task":
                if write["value"]["state"]["status"] == "terminal":
                    self._live_task_sidecars.discard(write["value"]["id"])
                else:
                    self._live_task_sidecars.add(write["value"]["id"])

    async def _reclaim_sidecars(self, replacements: Mapping[str, str], context: Context) -> None:
        """The marker already published this state, so reclamation is retryable best-effort maintenance."""
        if not replacements:
            return
        if self._fsync:
            flushed = await self._fs.flush_file(self._main_path, context)
            if flushed["ok"] is False:
                return
        for file, content in replacements.items():
            await self._replace_sidecar(file, content, context)

    async def _replace_sidecar(self, file: str, content: str, context: Context) -> None:
        path = await self._fs.join_path([self._directory, file], context)
        if path["ok"] is False:
            return
        if content == "":
            await self._fs.remove(path["value"], {"force": True}, context)
            return
        temporary = await self._fs.join_path([self._directory, f"{file}{RECLAIM_SUFFIX}"], context)
        if temporary["ok"] is False:
            return
        written = await self._fs.write_file(temporary["value"], content, context)
        if written["ok"] is False:
            return
        if self._fsync:
            flushed = await self._fs.flush_file(temporary["value"], context)
            if flushed["ok"] is False:
                return
        await self._fs.rename_file(temporary["value"], path["value"], context)

    # ─── recovery ────────────────────────────────────────────────────────────

    async def _recover(self, context: Context) -> None:
        fs = self._fs
        directory = self._directory
        memory = self._memory
        main = await self._read_lines(fs, self._main_path, MAIN_FILE, context, _parse_main_marker)
        previous_seq = 0
        for line in main.lines:
            if line.value["seq"] <= previous_seq:
                raise JsonlCorruptionError(f"Commit sequence does not strictly increase in {MAIN_FILE}")
            previous_seq = line.value["seq"]

        listed = await fs.list_dir(directory, context)
        if listed["ok"] is False:
            raise _error_from_file("directory listing", listed["error"])
        for info in listed["value"]:
            if info["kind"] == "file" and _RECLAIM_FILE.fullmatch(info["name"]):
                await fs.remove(info["path"], {"force": True}, context)
        sidecar_files = sorted(
            info["name"] for info in listed["value"] if info["kind"] == "file" and _SIDECAR_FILE.fullmatch(info["name"])
        )
        parsed_files: dict[str, _ParsedFile[dict[str, Any]]] = {}
        record_by_key: dict[tuple[str, int, int], _ParsedLine[dict[str, Any]]] = {}
        for file in sidecar_files:
            path = await fs.join_path([directory, file], context)
            if path["ok"] is False:
                raise _error_from_file("path join", path["error"])

            def parse(text: str, line: int, file: str = file) -> dict[str, Any]:
                return _parse_sidecar_record(text, file, line)

            parsed = await self._read_lines(fs, path["value"], file, context, parse)
            parsed_files[file] = parsed
            previous: dict[str, Any] | None = None
            for line in parsed.lines:
                if previous is not None and (
                    line.value["seq"] < previous["seq"]
                    or (line.value["seq"] == previous["seq"] and line.value["ordinal"] <= previous["ordinal"])
                ):
                    raise JsonlCorruptionError(f"Sidecar records are out of order in {file}")
                previous = line.value
                record_by_key[_sidecar_key(file, line.value["seq"], line.value["ordinal"])] = line

        current_only: set[int] = set()
        retired: set[int] = set()
        final_task_is_live: dict[int, bool] = {}
        for line in main.lines:
            for operation in line.value["writes"]:
                kind = operation["type"]
                if kind == "document.create":
                    if _is_current_only(operation["record"]):
                        current_only.add(operation["record"]["id"])
                elif kind == "document.retire":
                    retired.add(operation["id"])
                elif kind == "task":
                    final_task_is_live[operation["value"]["id"]] = False
                elif kind == "task.sidecar":
                    final_task_is_live[operation["id"]] = True
        retired_current_only = {id for id in retired if id in current_only}

        latest_bases: dict[int, dict[str, Any]] = {}
        for line in main.lines:
            marker = line.value
            for operation in marker["writes"]:
                kind = operation["type"]
                if kind != "document.create" and kind != "document.change":
                    continue
                id = operation["record"]["id"] if kind == "document.create" else operation["id"]
                if id not in current_only:
                    continue
                found = record_by_key.get(
                    _sidecar_key(_sidecar_file_name("doc", id), marker["seq"], operation["ordinal"])
                )
                record = None if found is None else found.value
                if (
                    record is None
                    or record["payload"]["type"] != "document"
                    or record["payload"]["id"] != id
                    or record["payload"]["content"]["kind"] != "base"
                ):
                    continue
                previous_base = latest_bases.get(id)
                if (
                    previous_base is None
                    or record["seq"] > previous_base["seq"]
                    or (record["seq"] == previous_base["seq"] and record["ordinal"] > previous_base["ordinal"])
                ):
                    latest_bases[id] = record

        def is_before_latest_base(id: int, seq: int, ordinal: int) -> bool:
            base = latest_bases.get(id)
            return base is not None and (seq < base["seq"] or (seq == base["seq"] and ordinal < base["ordinal"]))

        terminal_tasks = {id for id, live in final_task_is_live.items() if not live}
        confirmed: set[tuple[str, int, int]] = set()
        for line in main.lines:
            marker = line.value
            writes: list[Any] = []
            for operation in marker["writes"]:
                kind = operation["type"]
                if kind == "entry":
                    writes.append({"type": "entry", "value": decode_entry(operation["value"])})
                elif kind in ("conversation", "submission", "task", "document.retire"):
                    writes.append(operation)
                elif kind == "task.sidecar":
                    optional = operation["id"] in terminal_tasks
                    record = self._confirm_record(
                        marker,
                        operation["ordinal"],
                        _sidecar_file_name("task", operation["id"]),
                        record_by_key,
                        confirmed,
                        optional,
                    )
                    if record is not None:
                        if record["payload"]["type"] != "task" or record["payload"]["value"]["id"] != operation["id"]:
                            raise JsonlCorruptionError(
                                f"Confirmed task sidecar data does not match commit {marker['seq']}"
                            )
                        if not optional:
                            writes.append({"type": "task", "value": record["payload"]["value"]})
                else:  # document.create, document.change
                    id = operation["record"]["id"] if kind == "document.create" else operation["id"]
                    reclaimed = id in retired_current_only or is_before_latest_base(
                        id, marker["seq"], operation["ordinal"]
                    )
                    record = self._confirm_record(
                        marker, operation["ordinal"], _sidecar_file_name("doc", id), record_by_key, confirmed, reclaimed
                    )
                    content: dict[str, Any] | None = None
                    if record is not None:
                        if record["payload"]["type"] != "document" or record["payload"]["id"] != id:
                            raise JsonlCorruptionError(
                                f"Confirmed document sidecar data does not match commit {marker['seq']}"
                            )
                        content = record["payload"]["content"]
                    if kind == "document.create":
                        if content is not None and content["kind"] != "base":
                            raise JsonlCorruptionError(
                                f"Document creation lacks a confirmed base in commit {marker['seq']}"
                            )
                        writes.append(
                            {
                                "type": "document.create",
                                "record": operation["record"],
                                "content": {"kind": "base", "version": 1, "value": {}}
                                if reclaimed or content is None
                                else content,
                            }
                        )
                    elif not reclaimed and content is not None:
                        writes.append({"type": "document.change", "id": id, "content": _decode_content(content)})
            try:
                memory.prepare_commit(writes, marker["seq"]).apply()
            except Exception as error:
                raise JsonlCorruptionError(f"Invalid committed state at sequence {marker['seq']}", error) from None

        reclamations: dict[str, str] = {}
        for file, parsed in parsed_files.items():
            unconfirmed_at: int | None = None
            for line in parsed.lines:
                key = _sidecar_key(file, line.value["seq"], line.value["ordinal"])
                if key in confirmed:
                    if unconfirmed_at is not None:
                        raise JsonlCorruptionError(f"Confirmed record follows an unconfirmed tail in {file}")
                elif unconfirmed_at is None:
                    unconfirmed_at = line.start
            if unconfirmed_at is not None:
                truncated = await fs.truncate_file(parsed.path, unconfirmed_at, context)
                if truncated["ok"] is False:
                    raise _error_from_file(f"tail truncation of {file}", truncated["error"])

            numeric_id = int(file[file.index("-") + 1 : -len(".jsonl")])
            confirmed_lines = [
                line
                for line in parsed.lines
                if _sidecar_key(file, line.value["seq"], line.value["ordinal"]) in confirmed
            ]
            retained: list[_ParsedLine[dict[str, Any]]] | None = None
            if file.startswith("task-") and numeric_id in terminal_tasks:
                retained = []
            elif file.startswith("doc-"):
                if numeric_id in retired_current_only:
                    retained = []
                elif numeric_id in latest_bases:
                    retained = [
                        line
                        for line in confirmed_lines
                        if not is_before_latest_base(numeric_id, line.value["seq"], line.value["ordinal"])
                    ]
            if retained is not None and (len(retained) < len(confirmed_lines) or len(retained) == 0):
                reclamations[file] = "".join(_json_line(line.value) for line in retained)
        await self._reclaim_sidecars(reclamations, context)

        self._current_only_documents.update(current_only)
        self._live_task_sidecars.update(id for id, live in final_task_is_live.items() if live)

    @staticmethod
    def _confirm_record(
        marker: Mapping[str, Any],
        ordinal: int,
        file: str,
        record_by_key: Mapping[tuple[str, int, int], _ParsedLine[dict[str, Any]]],
        confirmed: set[tuple[str, int, int]],
        optional: bool,
    ) -> dict[str, Any] | None:
        key = _sidecar_key(file, marker["seq"], ordinal)
        if key in confirmed:
            raise JsonlCorruptionError("Sidecar record is confirmed more than once")
        line = record_by_key.get(key)
        if line is None:
            if optional:
                return None
            raise JsonlCorruptionError(f"Missing confirmed sidecar record {file} at sequence {marker['seq']}")
        confirmed.add(key)
        return line.value

    @staticmethod
    async def _read_lines(
        fs: FileSystem, path: str, name: str, context: Context, parse: Callable[[str, int], T]
    ) -> _ParsedFile[T]:
        read = await fs.read_binary_file(path, context)
        if read["ok"] is False:
            if read["error"].code == "not_found":
                return _ParsedFile(path, [])
            raise _error_from_file(f"read of {name}", read["error"])
        data = read["value"]
        complete_size = len(data)
        if complete_size > 0 and data[-1] != 0x0A:
            complete_size = data.rfind(b"\n") + 1
            truncated = await fs.truncate_file(path, complete_size, context)
            if truncated["ok"] is False:
                raise _error_from_file(f"torn-line truncation of {name}", truncated["error"])
        lines: list[_ParsedLine[T]] = []
        start = 0
        line_number = 1
        while start < complete_size:
            end = data.index(b"\n", start)
            try:
                text = data[start:end].decode("utf-8")
            except UnicodeDecodeError as error:
                raise JsonlCorruptionError(f"Invalid UTF-8 in complete {name} line {line_number}", error) from None
            if text.startswith("﻿"):
                text = text[1:]  # TextDecoder drops a leading byte-order mark.
            lines.append(_ParsedLine(parse(text, line_number), start))
            start = end + 1
            line_number += 1
        return _ParsedFile(path, lines)

    async def _resolve_file(self, file: str, context: Context) -> str:
        path = await self._fs.join_path([self._directory, file], context)
        if path["ok"] is False:
            raise _error_from_file("path join", path["error"])
        return path["value"]

    def _store(self) -> MemoryStorage:
        self._assert_usable()
        return self._memory

    def _poison(self, cause: BaseException) -> JsonlStoragePoisonedError:
        if self._poison_error is None:
            self._poison_error = JsonlStoragePoisonedError(cause)
        return self._poison_error

    def _assert_usable(self) -> None:
        if self._closed:
            raise RuntimeError("JsonlStorage is closed")
        if self._poison_error is not None:
            raise self._poison_error
