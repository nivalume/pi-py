"""Port of ``test/jsonl-storage.test.ts``."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from pi_chord.context import BACKGROUND_CONTEXT, Context
from pi_durable import ROOT_CONVERSATION_ID
from pi_durable.env import FileError, FileSystem, RemoveOptions, Result, err
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.ids import id_from_number
from pi_durable.storage.jsonl import JsonlCorruptionError, JsonlStorage, JsonlStoragePoisonedError
from pi_durable.storage.jsonl.local import open_local_jsonl_storage
from pi_durable.testing import register_storage_conformance
from pi_durable.types import AnyTaskRecord, DocumentCreate, Seq, Storage, StorageWrite

context = BACKGROUND_CONTEXT


async def _with_storage(use: Any) -> None:
    directory = tempfile.mkdtemp(prefix="pi-durable-jsonl-")
    storage = await JsonlStorage.open(directory, LocalExecutionEnv(cwd=directory), context)
    try:
        await use(storage)
    finally:
        await storage.close(context)


class ReopeningStorage:
    """Closes and reopens the directory after every commit, so every read sees only what was persisted."""

    def __init__(self, current: JsonlStorage, directory: str) -> None:
        self._current = current
        self._directory = directory
        self._closed = False

    async def commit(self, writes: Sequence[StorageWrite], commit_context: Context) -> Seq:
        if self._closed:
            return await self._current.commit(writes, commit_context)
        try:
            return await self._current.commit(writes, commit_context)
        finally:
            await self._current.close(context)
            self._current = await JsonlStorage.open(self._directory, LocalExecutionEnv(cwd=self._directory), context)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current, name)

    async def close(self, close_context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        await self._current.close(close_context)


async def _with_reopening_storage(use: Any) -> None:
    directory = tempfile.mkdtemp(prefix="pi-durable-jsonl-conformance-")
    current = await JsonlStorage.open(directory, LocalExecutionEnv(cwd=directory), context)
    storage = ReopeningStorage(current, directory)
    try:
        await use(storage)
    finally:
        await storage.close(context)


register_storage_conformance(globals(), "JsonlStorage", _with_storage)
register_storage_conformance(globals(), "JsonlStorage across reopen", _with_reopening_storage)


@pytest.fixture
async def opened(tmp_path: Path) -> Any:
    storages: list[JsonlStorage] = []

    async def open_storage(directory: Path, fs: FileSystem | None = None, *, fsync: bool = False) -> JsonlStorage:
        storage = await JsonlStorage.open(
            str(directory), fs if fs is not None else LocalExecutionEnv(cwd=str(directory)), context, fsync=fsync
        )
        storages.append(storage)
        return storage

    yield open_storage
    for storage in storages:
        await storage.close(context)


def pending_task(id: int, phase: str = "ready") -> AnyTaskRecord:
    return {
        "id": id_from_number(id),
        "conversationId": ROOT_CONVERSATION_ID,
        "kind": "test.task",
        "version": 1,
        "input": None,
        "state": {"status": "pending", "checkpoint": {"phase": phase}},
        "background": False,
        "abortRequested": False,
    }


def terminal_task(id: int) -> AnyTaskRecord:
    return {
        "id": id_from_number(id),
        "conversationId": ROOT_CONVERSATION_ID,
        "kind": "test.task",
        "version": 1,
        "input": None,
        "state": {"status": "terminal", "outcome": {"status": "completed", "result": None}},
        "background": False,
        "abortRequested": False,
    }


async def create_root(storage: Storage) -> None:
    await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)


def session_document(id: int, kind: str = "test.document") -> DocumentCreate:
    return {"id": id_from_number(id), "kind": kind, "scope": {"kind": "session"}}


@dataclass(frozen=True)
class Failure:
    operation: Literal["append", "flush", "write", "rename", "remove"]
    call: int
    mode: Literal["before", "after", "short"]


class InstrumentedEnv(LocalExecutionEnv):
    def __init__(self, cwd: str) -> None:
        super().__init__(cwd=cwd)
        self.operations: list[str] = []
        self._failure: Failure | None = None
        self._calls: dict[str, int] = {}

    def fail(self, failure: Failure) -> None:
        self._failure = failure
        self._reset()

    def clear(self) -> None:
        self._failure = None
        self._reset()

    def _reset(self) -> None:
        self._calls = {}
        self.operations.clear()

    def _failing(self, operation: str) -> Failure | None:
        self._calls[operation] = self._calls.get(operation, 0) + 1
        failure = self._failure
        if failure is None or failure.operation != operation or failure.call != self._calls[operation]:
            return None
        return failure

    async def _write_like(self, operation: str, path: str, content: str | bytes, write: Any) -> Result[None, FileError]:
        self.operations.append(f"{operation}:{os.path.basename(path)}")
        failure = self._failing(operation)
        if failure is None:
            return await write(path, content, context)
        if failure.mode == "before":
            return err(FileError("unknown", f"injected {operation} failure", path))
        if failure.mode == "short":
            data = content.encode() if isinstance(content, str) else content
            written = await write(path, data[: max(1, len(data) // 2)], context)
            if written["ok"] is False:
                return written
            return err(FileError("unknown", f"injected short {operation}", path))
        written = await write(path, content, context)
        if written["ok"] is False:
            return written
        return err(FileError("unknown", f"injected post-{operation} failure", path))

    async def append_file(self, path: str, content: str | bytes, append_context: Context) -> Result[None, FileError]:
        return await self._write_like("append", path, content, super().append_file)

    async def write_file(self, path: str, content: str | bytes, write_context: Context) -> Result[None, FileError]:
        return await self._write_like("write", path, content, super().write_file)

    async def flush_file(self, path: str, flush_context: Context) -> Result[None, FileError]:
        self.operations.append(f"flush:{os.path.basename(path)}")
        failure = self._failing("flush")
        if failure is None:
            return await super().flush_file(path, flush_context)
        if failure.mode == "after":
            flushed = await super().flush_file(path, flush_context)
            if flushed["ok"] is False:
                return flushed
        return err(FileError("unknown", "injected flush failure", path))

    async def rename_file(
        self, source_path: str, destination_path: str, rename_context: Context
    ) -> Result[None, FileError]:
        self.operations.append(f"rename:{os.path.basename(source_path)}->{os.path.basename(destination_path)}")
        failure = self._failing("rename")
        if failure is None:
            return await super().rename_file(source_path, destination_path, rename_context)
        if failure.mode == "after":
            renamed = await super().rename_file(source_path, destination_path, rename_context)
            if renamed["ok"] is False:
                return renamed
        return err(FileError("unknown", "injected rename failure", source_path))

    async def remove(
        self, path: str, options: RemoveOptions | None, remove_context: Context
    ) -> Result[None, FileError]:
        self.operations.append(f"remove:{os.path.basename(path)}")
        failure = self._failing("remove")
        if failure is None:
            return await super().remove(path, options, remove_context)
        if failure.mode == "after":
            removed = await super().remove(path, options, remove_context)
            if removed["ok"] is False:
                return removed
        return err(FileError("unknown", "injected remove failure", path))


def append_to(path: Path, content: str | bytes) -> None:
    with open(path, "ab") as file:
        file.write(content.encode() if isinstance(content, str) else content)


def read_lines(path: Path) -> list[str]:
    text = path.read_text("utf-8")
    return [] if text == "" else text.rstrip().split("\n")


async def test_opens_through_the_local_adapter(tmp_path: Path) -> None:
    storage = await open_local_jsonl_storage(str(tmp_path), context)
    try:
        await create_root(storage)
        assert await storage.conversation(ROOT_CONVERSATION_ID, context) == {"id": ROOT_CONVERSATION_ID}
    finally:
        await storage.close(context)


async def test_publishes_every_commit_before_reclaiming_current_only_document_and_terminal_task_sidecars(
    tmp_path: Path, opened: Any
) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    task_id = await storage.mint_id()
    document_id = await storage.mint_id()
    await storage.commit([{"type": "task", "value": pending_task(task_id)}], context)
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(document_id),
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    await storage.commit(
        [{"type": "document.change", "id": document_id, "content": {"kind": "delta", "version": 1, "ops": []}}], context
    )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": document_id,
                "content": {"kind": "base", "version": 1, "value": {"count": 1}},
            }
        ],
        context,
    )
    assert len(read_lines(tmp_path / f"task-{task_id}.jsonl")) == 1
    assert len(read_lines(tmp_path / f"doc-{document_id}.jsonl")) == 1

    await storage.commit([{"type": "document.retire", "id": document_id}], context)
    await storage.commit([{"type": "task", "value": terminal_task(task_id)}], context)

    main = read_lines(tmp_path / "main.jsonl")
    assert len(main) == 7
    assert not (tmp_path / f"task-{task_id}.jsonl").exists()
    assert not (tmp_path / f"doc-{document_id}.jsonl").exists()
    assert [json.loads(line)["type"] for line in main] == ["commit"] * 7


async def test_orders_multiple_live_task_replacements_in_one_sidecar_by_commit_ordinal(
    tmp_path: Path, opened: Any
) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    task_id = await storage.mint_id()
    await storage.commit(
        [
            {"type": "task", "value": pending_task(task_id, "first")},
            {"type": "task", "value": pending_task(task_id, "second")},
        ],
        context,
    )
    assert len(read_lines(tmp_path / f"task-{task_id}.jsonl")) == 2
    reopened = await opened(tmp_path)
    assert await reopened.task(id_from_number(task_id), context) == pending_task(task_id, "second")


async def test_removes_a_complete_plus_torn_unconfirmed_multi_record_sidecar_append(
    tmp_path: Path, opened: Any
) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env)
    await create_root(storage)
    task_id = await storage.mint_id()
    env.fail(Failure("append", 1, "short"))
    with pytest.raises(JsonlStoragePoisonedError, match="poisoned"):
        await storage.commit(
            [
                {"type": "task", "value": pending_task(task_id, "first")},
                {"type": "task", "value": pending_task(task_id, f"second-{'x' * 512}")},
            ],
            context,
        )
    partial = (tmp_path / f"task-{task_id}.jsonl").read_bytes()
    assert partial[-1] != 0x0A
    assert partial.count(b"\n") == 1

    reopened = await opened(tmp_path)
    assert await reopened.task(id_from_number(task_id), context) is None
    assert (tmp_path / f"task-{task_id}.jsonl").stat().st_size == 0


async def test_serializes_the_complete_candidate_before_io_and_leaves_preparation_failures_usable(
    tmp_path: Path, opened: Any
) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env)
    await create_root(storage)
    env.clear()
    id = await storage.mint_id()
    with pytest.raises(TypeError):
        await storage.commit(
            [
                {
                    "type": "entry",
                    "value": {"id": id, "conversationId": ROOT_CONVERSATION_ID, "kind": "bad", "data": {1, 2}},
                }
            ],
            context,
        )
    assert env.operations == []
    assert await storage.entry(id_from_number(id), context) is None
    assert (
        await storage.commit(
            [{"type": "entry", "value": {"id": id, "conversationId": ROOT_CONVERSATION_ID, "kind": "good"}}], context
        )
        == 2
    )


@pytest.mark.parametrize(
    "failure",
    [
        Failure("append", 1, "before"),
        Failure("append", 1, "after"),
        Failure("append", 1, "short"),
        Failure("append", 2, "before"),
        Failure("append", 2, "after"),
        Failure("append", 3, "before"),
        Failure("append", 3, "short"),
        Failure("append", 3, "after"),
    ],
    ids=lambda failure: f"{failure.mode} failure at append {failure.call}",
)
async def test_poisons_after_an_append_failure_and_recovers_only_confirmed_state(
    tmp_path: Path, opened: Any, failure: Failure
) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env)
    await create_root(storage)
    first_id = id_from_number(await storage.mint_id())
    second_id = id_from_number(await storage.mint_id())
    env.fail(failure)
    with pytest.raises(JsonlStoragePoisonedError, match="poisoned"):
        await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": session_document(first_id, "first"),
                    "content": {"kind": "base", "version": 1, "value": {"text": "\u03b1"}},
                },
                {
                    "type": "document.create",
                    "record": session_document(second_id, "second"),
                    "content": {"kind": "base", "version": 1, "value": {"text": "\u03b2"}},
                },
            ],
            context,
        )
    with pytest.raises(JsonlStoragePoisonedError, match="poisoned"):
        await storage.document(first_id, "current", context)

    reopened = await opened(tmp_path)
    marker_survived = failure.call == 3 and failure.mode == "after"
    first = await reopened.document(first_id, "current", context)
    second = await reopened.document(second_id, "current", context)
    if marker_survived:
        assert first is not None and first["value"] == {"text": "\u03b1"}
        assert second is not None and second["value"] == {"text": "\u03b2"}
    else:
        assert first is None
        assert second is None


@pytest.mark.parametrize(
    "failure",
    [
        Failure("flush", 1, "before"),
        Failure("flush", 1, "after"),
        Failure("flush", 2, "before"),
        Failure("flush", 2, "after"),
    ],
    ids=lambda failure: f"{failure.mode} failure at flush {failure.call}",
)
async def test_poisons_after_a_flush_failure_and_never_writes_a_marker(
    tmp_path: Path, opened: Any, failure: Failure
) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env, fsync=True)
    await create_root(storage)
    first_id = id_from_number(await storage.mint_id())
    second_id = id_from_number(await storage.mint_id())
    env.fail(failure)
    with pytest.raises(JsonlStoragePoisonedError, match="poisoned"):
        await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": session_document(first_id, "flush.first"),
                    "content": {"kind": "base", "version": 1, "value": {}},
                },
                {
                    "type": "document.create",
                    "record": session_document(second_id, "flush.second"),
                    "content": {"kind": "base", "version": 1, "value": {}},
                },
            ],
            context,
        )
    reopened = await opened(tmp_path)
    assert await reopened.document(first_id, "current", context) is None
    assert await reopened.document(second_id, "current", context) is None


async def test_orders_publication_flushes_exactly_and_flushes_main_only_to_authorize_reclamation(
    tmp_path: Path, opened: Any
) -> None:
    for fsync in (False, True):
        directory = tmp_path / f"fsync-{fsync}"
        directory.mkdir()
        env = InstrumentedEnv(str(directory))
        storage = await opened(directory, env, fsync=fsync)
        await create_root(storage)
        first_id = await storage.mint_id()
        second_id = await storage.mint_id()
        env.clear()
        await storage.commit(
            [
                {
                    "type": "document.create",
                    "record": session_document(second_id, "second"),
                    "content": {"kind": "base", "version": 1, "value": {}},
                },
                {
                    "type": "document.create",
                    "record": session_document(first_id, "first"),
                    "content": {"kind": "base", "version": 1, "value": {}},
                },
            ],
            context,
        )
        assert env.operations == [
            f"append:doc-{second_id}.jsonl",
            f"append:doc-{first_id}.jsonl",
            *([f"flush:doc-{second_id}.jsonl", f"flush:doc-{first_id}.jsonl"] if fsync else []),
            "append:main.jsonl",
        ]

        env.clear()
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": first_id,
                    "content": {"kind": "base", "version": 1, "value": {"checkpoint": True}},
                }
            ],
            context,
        )
        assert env.operations == [
            f"append:doc-{first_id}.jsonl",
            *([f"flush:doc-{first_id}.jsonl"] if fsync else []),
            "append:main.jsonl",
            *(["flush:main.jsonl"] if fsync else []),
            f"write:doc-{first_id}.jsonl.reclaim",
            *([f"flush:doc-{first_id}.jsonl.reclaim"] if fsync else []),
            f"rename:doc-{first_id}.jsonl.reclaim->doc-{first_id}.jsonl",
        ]

        env.clear()
        await storage.commit(
            [
                {
                    "type": "entry",
                    "value": {
                        "id": await storage.mint_id(),
                        "conversationId": ROOT_CONVERSATION_ID,
                        "kind": "main-only",
                    },
                }
            ],
            context,
        )
        assert env.operations == ["append:main.jsonl"]

        task_id = await storage.mint_id()
        await storage.commit([{"type": "task", "value": pending_task(task_id)}], context)
        env.clear()
        await storage.commit([{"type": "task", "value": terminal_task(task_id)}], context)
        assert env.operations == [
            "append:main.jsonl",
            *(["flush:main.jsonl"] if fsync else []),
            f"remove:task-{task_id}.jsonl",
        ]


@pytest.mark.parametrize(
    "failure",
    [
        Failure("write", 1, "before"),
        Failure("write", 1, "short"),
        Failure("write", 1, "after"),
        Failure("rename", 1, "before"),
        Failure("rename", 1, "after"),
    ],
    ids=lambda failure: f"reclaim {failure.operation} {failure.mode}",
)
async def test_recovers_a_committed_base_across_a_reclaim_failure(
    tmp_path: Path, opened: Any, failure: Failure
) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env)
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(id),
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": id,
                "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 1)]},
            }
        ],
        context,
    )
    env.fail(failure)
    assert (
        await storage.commit(
            [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"count": 2}}}],
            context,
        )
        == 4
    )
    current = await storage.document(id, "current", context)
    assert current is not None and current["value"] == {"count": 2}
    await storage.close(context)

    reopened = await opened(tmp_path)
    current = await reopened.document(id, "current", context)
    assert current is not None and current["value"] == {"count": 2}
    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 1
    assert [name for name in os.listdir(tmp_path) if name.endswith(".reclaim")] == []


@pytest.mark.parametrize("mode", ["before", "after"])
async def test_recovers_document_retirement_reclamation_across_remove(tmp_path: Path, opened: Any, mode: Any) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env)
    await create_root(storage)
    task_id = id_from_number(await storage.mint_id())
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {"type": "task", "value": pending_task(task_id)},
            {
                "type": "document.create",
                "record": {"id": id, "kind": "task.document", "scope": {"kind": "task", "taskId": task_id}},
                "content": {"kind": "base", "version": 1, "value": {"count": 1}},
            },
        ],
        context,
    )
    env.fail(Failure("remove", 1, mode))
    assert await storage.commit([{"type": "document.retire", "id": id}], context) == 3
    assert await storage.document(id, "current", context) is None
    await storage.close(context)

    reopened = await opened(tmp_path)
    assert await reopened.document(id, "current", context) is None
    assert await reopened.task(task_id, context) == pending_task(task_id)
    assert not (tmp_path / f"doc-{id}.jsonl").exists()
    assert [name for name in os.listdir(tmp_path) if name.endswith(".reclaim")] == []


@pytest.mark.parametrize("mode", ["before", "after"])
async def test_defers_reclamation_after_an_authorizing_main_flush_failure(
    tmp_path: Path, opened: Any, mode: Any
) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env, fsync=True)
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(id),
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    env.fail(Failure("flush", 2, mode))
    assert (
        await storage.commit(
            [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"count": 2}}}],
            context,
        )
        == 3
    )
    assert env.operations == [
        f"append:doc-{id}.jsonl",
        f"flush:doc-{id}.jsonl",
        "append:main.jsonl",
        "flush:main.jsonl",
    ]
    current = await storage.document(id, "current", context)
    assert current is not None and current["value"] == {"count": 2}
    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 2
    await storage.close(context)

    recovery_env = InstrumentedEnv(str(tmp_path))
    recovery_env.fail(Failure("flush", 1, mode))
    deferred = await opened(tmp_path, recovery_env, fsync=True)
    current = await deferred.document(id, "current", context)
    assert current is not None and current["value"] == {"count": 2}
    assert recovery_env.operations == ["flush:main.jsonl"]
    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 2
    await deferred.close(context)

    reclaimed = await opened(tmp_path, LocalExecutionEnv(cwd=str(tmp_path)), fsync=True)
    current = await reclaimed.document(id, "current", context)
    assert current is not None and current["value"] == {"count": 2}
    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 1


@pytest.mark.parametrize("mode", ["before", "after"])
async def test_keeps_a_committed_base_usable_after_a_reclaim_temp_flush_failure(
    tmp_path: Path, opened: Any, mode: Any
) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env, fsync=True)
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(id),
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    env.fail(Failure("flush", 3, mode))
    assert (
        await storage.commit(
            [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"count": 2}}}],
            context,
        )
        == 3
    )
    env.clear()
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": id,
                "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 3)]},
            }
        ],
        context,
    )
    await storage.close(context)

    reopened = await opened(tmp_path, LocalExecutionEnv(cwd=str(tmp_path)), fsync=True)
    current = await reopened.document(id, "current", context)
    assert current is not None and current["value"] == {"count": 3}
    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 2


@pytest.mark.parametrize("mode", ["before", "after"])
async def test_recovers_terminal_task_reclamation_across_remove(tmp_path: Path, opened: Any, mode: Any) -> None:
    env = InstrumentedEnv(str(tmp_path))
    storage = await opened(tmp_path, env)
    await create_root(storage)
    id = await storage.mint_id()
    await storage.commit([{"type": "task", "value": pending_task(id)}], context)
    env.fail(Failure("remove", 1, mode))
    assert await storage.commit([{"type": "task", "value": terminal_task(id)}], context) == 3
    assert await storage.task(id_from_number(id), context) == terminal_task(id)
    await storage.close(context)

    reopened = await opened(tmp_path)
    assert await reopened.task(id_from_number(id), context) == terminal_task(id)
    assert not (tmp_path / f"task-{id}.jsonl").exists()
    assert [name for name in os.listdir(tmp_path) if name.endswith(".reclaim")] == []


async def test_appends_later_deltas_to_the_replacement_sidecar_after_a_current_only_base(
    tmp_path: Path, opened: Any
) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(id),
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    await storage.commit(
        [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"count": 10}}}],
        context,
    )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": id,
                "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 11)]},
            }
        ],
        context,
    )
    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 2
    reopened = await opened(tmp_path)
    current = await reopened.document(id, "current", context)
    assert current is not None and current["value"] == {"count": 11}


async def test_never_reclaims_rewindable_document_history_including_after_a_base_and_retirement(
    tmp_path: Path, opened: Any
) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    id = id_from_number(await storage.mint_id())
    record: DocumentCreate = {
        "id": id,
        "kind": "rewindable",
        "scope": {"kind": "conversation", "conversationId": ROOT_CONVERSATION_ID},
        "history": "rewindable",
        "fork": "asOf",
    }
    created_at = await storage.commit(
        [
            {
                "type": "document.create",
                "record": record,
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    changed_at = await storage.commit(
        [
            {
                "type": "document.change",
                "id": id,
                "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 1)]},
            }
        ],
        context,
    )
    await storage.commit(
        [{"type": "document.change", "id": id, "content": {"kind": "base", "version": 1, "value": {"count": 2}}}],
        context,
    )
    await storage.commit([{"type": "document.retire", "id": id}], context)

    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 3
    reopened = await opened(tmp_path)
    created = await reopened.document(id, created_at, context)
    changed = await reopened.document(id, changed_at, context)
    assert created is not None and created["value"] == {"count": 0}
    assert changed is not None and changed["value"] == {"count": 1}
    assert await reopened.document(id, "current", context) is None
    assert len(read_lines(tmp_path / f"doc-{id}.jsonl")) == 3


async def test_reclaims_retired_task_session_and_latest_conversation_document_sidecars(
    tmp_path: Path, opened: Any
) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    task_id = id_from_number(await storage.mint_id())
    session_id = id_from_number(await storage.mint_id())
    latest_id = id_from_number(await storage.mint_id())
    task_document_id = id_from_number(await storage.mint_id())
    created_at = await storage.commit(
        [
            {"type": "task", "value": pending_task(task_id)},
            {
                "type": "document.create",
                "record": session_document(session_id, "session"),
                "content": {"kind": "base", "version": 1, "value": {}},
            },
            {
                "type": "document.create",
                "record": {
                    "id": latest_id,
                    "kind": "latest",
                    "scope": {"kind": "conversation", "conversationId": ROOT_CONVERSATION_ID},
                    "history": "latest",
                    "fork": "current",
                },
                "content": {"kind": "base", "version": 1, "value": {}},
            },
            {
                "type": "document.create",
                "record": {"id": task_document_id, "kind": "task", "scope": {"kind": "task", "taskId": task_id}},
                "content": {"kind": "base", "version": 1, "value": {}},
            },
        ],
        context,
    )
    retired_at = await storage.commit(
        [
            {"type": "document.retire", "id": session_id},
            {"type": "document.retire", "id": latest_id},
            {"type": "document.retire", "id": task_document_id},
            {"type": "task", "value": terminal_task(task_id)},
        ],
        context,
    )
    for file in (
        f"doc-{session_id}.jsonl",
        f"doc-{latest_id}.jsonl",
        f"doc-{task_document_id}.jsonl",
        f"task-{task_id}.jsonl",
    ):
        assert not (tmp_path / file).exists()

    reopened = await opened(tmp_path)
    assert await reopened.task(task_id, context) == terminal_task(task_id)
    assert await reopened.document(session_id, "current", context) is None
    assert await reopened.document(latest_id, "current", context) is None
    assert await reopened.document(task_document_id, "current", context) is None
    found = await reopened.find_document({"kind": "session", "scope": {"kind": "session"}}, created_at, context)
    assert found is not None
    assert {key: found[key] for key in ("id", "createdAt", "retiredAt")} == {  # type: ignore[literal-required]
        "id": session_id,
        "createdAt": created_at,
        "retiredAt": retired_at,
    }
    assert await reopened.find_document({"kind": "session", "scope": {"kind": "session"}}, retired_at, context) is None


async def test_truncates_torn_utf8_tails_at_exact_byte_offsets_and_reuses_the_unconfirmed_sequence(
    tmp_path: Path, opened: Any
) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    document_id = id_from_number(await storage.mint_id())
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(document_id),
                "content": {"kind": "base", "version": 1, "value": {"text": "kept"}},
            }
        ],
        context,
    )
    sidecar_path = tmp_path / f"doc-{document_id}.jsonl"
    main_path = tmp_path / "main.jsonl"
    sidecar_size = sidecar_path.stat().st_size
    main_size = main_path.stat().st_size
    torn = '{"text":"€'.encode()
    append_to(sidecar_path, torn[:-1])
    append_to(main_path, torn[:-1])

    reopened = await opened(tmp_path)
    assert sidecar_path.stat().st_size == sidecar_size
    assert main_path.stat().st_size == main_size
    assert (
        await reopened.commit(
            [{"type": "document.change", "id": document_id, "content": {"kind": "delta", "version": 1, "ops": []}}],
            context,
        )
        == 3
    )


async def test_removes_complete_unconfirmed_sidecar_tails_without_resurrecting_terminal_tasks(
    tmp_path: Path, opened: Any
) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    task_id = await storage.mint_id()
    await storage.commit([{"type": "task", "value": pending_task(task_id)}], context)
    await storage.commit([{"type": "task", "value": terminal_task(task_id)}], context)
    sidecar_path = tmp_path / f"task-{task_id}.jsonl"
    assert not sidecar_path.exists()
    stale = {
        "format": 1,
        "type": "record",
        "seq": 4,
        "ordinal": 0,
        "payload": {"type": "task", "value": pending_task(task_id, "stale")},
    }
    append_to(sidecar_path, json.dumps(stale, separators=(",", ":")) + "\n")

    reopened = await opened(tmp_path)
    assert not sidecar_path.exists()
    assert await reopened.task(id_from_number(task_id), context) == terminal_task(task_id)
    assert await reopened.commit([], context) == 4


async def test_fails_open_when_confirmed_sidecar_data_is_missing(tmp_path: Path, opened: Any) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    document_id = await storage.mint_id()
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(document_id),
                "content": {"kind": "base", "version": 1, "value": {}},
            }
        ],
        context,
    )
    (tmp_path / f"doc-{document_id}.jsonl").write_text("")
    with pytest.raises(JsonlCorruptionError, match="Missing confirmed sidecar record"):
        await JsonlStorage.open(str(tmp_path), LocalExecutionEnv(cwd=str(tmp_path)), context)


async def test_rejects_a_confirmed_record_after_an_unconfirmed_sidecar_record(tmp_path: Path, opened: Any) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    document_id = await storage.mint_id()
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(document_id),
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": document_id,
                "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), 1)]},
            }
        ],
        context,
    )
    path = tmp_path / f"doc-{document_id}.jsonl"
    first, second = read_lines(path)
    unconfirmed = json.dumps(
        {
            "format": 1,
            "type": "record",
            "seq": 2,
            "ordinal": 999,
            "payload": {"type": "document", "id": document_id, "content": {"kind": "delta", "version": 1, "ops": []}},
        },
        separators=(",", ":"),
    )
    path.write_text(f"{first}\n{unconfirmed}\n{second}\n")
    with pytest.raises(JsonlCorruptionError, match="Confirmed record follows an unconfirmed tail"):
        await JsonlStorage.open(str(tmp_path), LocalExecutionEnv(cwd=str(tmp_path)), context)


async def test_rejects_non_increasing_main_commit_sequences(tmp_path: Path, opened: Any) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    path = tmp_path / "main.jsonl"
    append_to(path, path.read_bytes())
    with pytest.raises(JsonlCorruptionError, match="Commit sequence does not strictly increase"):
        await JsonlStorage.open(str(tmp_path), LocalExecutionEnv(cwd=str(tmp_path)), context)


async def test_rejects_structurally_invalid_confirmed_document_content(tmp_path: Path, opened: Any) -> None:
    storage = await opened(tmp_path)
    await create_root(storage)
    document_id = await storage.mint_id()
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": session_document(document_id),
                "content": {"kind": "base", "version": 1, "value": {}},
            }
        ],
        context,
    )
    path = tmp_path / f"doc-{document_id}.jsonl"
    record = json.loads(path.read_text().strip())
    del record["payload"]["content"]["value"]
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    with pytest.raises(JsonlCorruptionError, match="Invalid document content"):
        await JsonlStorage.open(str(tmp_path), LocalExecutionEnv(cwd=str(tmp_path)), context)


async def test_rejects_malformed_complete_main_and_sidecar_lines(tmp_path: Path, opened: Any) -> None:
    main_directory = tmp_path / "main"
    main_directory.mkdir()
    main_storage = await opened(main_directory)
    await create_root(main_storage)
    append_to(main_directory / "main.jsonl", "{bad}\n")
    with pytest.raises(JsonlCorruptionError, match="Malformed complete main.jsonl"):
        await JsonlStorage.open(str(main_directory), LocalExecutionEnv(cwd=str(main_directory)), context)

    sidecar_directory = tmp_path / "sidecar"
    sidecar_directory.mkdir()
    sidecar_storage = await opened(sidecar_directory)
    await create_root(sidecar_storage)
    (sidecar_directory / "doc-99.jsonl").write_text("{bad}\n")
    with pytest.raises(JsonlCorruptionError, match="Malformed complete doc-99.jsonl"):
        await JsonlStorage.open(str(sidecar_directory), LocalExecutionEnv(cwd=str(sidecar_directory)), context)


async def test_writes_upstream_json_lines_with_messages_as_camel_case_json(tmp_path: Path, opened: Any) -> None:
    from pi_ai.types import TextContent, UserMessage

    storage = await opened(tmp_path)
    await create_root(storage)
    id = await storage.mint_id()
    message = UserMessage([TextContent("hi \ud800")], timestamp=5)
    await storage.commit(
        [
            {
                "type": "entry",
                "value": {"id": id, "conversationId": ROOT_CONVERSATION_ID, "kind": "pi.user", "model": [message]},
            }
        ],
        context,
    )
    line = read_lines(tmp_path / "main.jsonl")[-1]
    assert line == (
        '{"format":1,"type":"commit","seq":2,"writes":[{"type":"entry","value":{"id":2,"conversationId":1,'
        '"kind":"pi.user","model":[{"role":"user","content":[{"type":"text","text":"hi \\ud800"}],"timestamp":5}]}}]}'
    )
    reopened = await opened(tmp_path)
    stored = await reopened.entry(id_from_number(id), context)
    assert stored is not None and stored["entry"]["model"] == [message]  # type: ignore[typeddict-item]
