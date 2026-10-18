"""Storages written by TypeScript pi-durable and by this port are interchangeable (Python-only).

``fixtures/storage_interop`` holds a SQLite database and a JSONL directory that upstream pi-durable 1.3.0
wrote from ``commits.json``, and ``expected.json``, what upstream reads back for ``reads.json`` (see
``fixtures/storage_interop/generate.py``). Python must read the same from those stores and from the stores
it writes itself; the reverse direction (TypeScript reading Python-written stores) was checked with the same
harness when the fixture was made.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable.storage.codec import decode_storage_write, encode_entry, json_parse, json_stringify
from pi_durable.storage.jsonl.local import open_local_jsonl_storage
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.types import Storage

FIXTURE = Path(__file__).parent / "fixtures" / "storage_interop"
context = BACKGROUND_CONTEXT


def load(name: str) -> Any:
    return json_parse((FIXTURE / name).read_text("utf-8"))


async def open_storage(backend: str, path: Path) -> Storage:
    if backend == "sqlite":
        return await open_sqlite_storage(path)
    return await open_local_jsonl_storage(str(path), context)


async def write_commits(storage: Storage) -> None:
    for writes in load("commits.json"):
        await storage.commit([decode_storage_write(write) for write in writes], context)


def encode(method: str, value: Any) -> Any:
    if value is None:
        return None
    if method in ("entry", "visibleEntry"):
        return {**value, "entry": encode_entry(value["entry"])}
    if method == "findLatestHeadMarker":
        return encode_entry(value)
    if method == "scanEntries":
        return {**value, "items": [encode_entry(item) for item in value["items"]]}
    return value


async def read_all(storage: Storage) -> Any:
    results: list[Any] = []
    for method, *args in load("reads.json"):
        try:
            if method == "conversation":
                value = await storage.conversation(args[0], context)
            elif method == "scanConversations":
                value = await storage.scan_conversations(args[0], args[1], None, context)
            elif method == "entry":
                value = await storage.entry(args[0], context)
            elif method == "visibleEntry":
                value = await storage.visible_entry(args[0], args[1], context)
            elif method == "findLatestHeadMarker":
                value = await storage.find_latest_head_marker(args[0], args[1], context)
            elif method == "scanEntries":
                value = await storage.scan_entries(args[0], args[1], None, context)
            elif method == "task":
                value = await storage.task(args[0], context)
            elif method == "scanTasks":
                value = await storage.scan_tasks(args[0], args[1], None, context)
            elif method == "submission":
                value = await storage.submission(args[0], context)
            elif method == "scanSubmissions":
                value = await storage.scan_submissions(args[0], args[1], None, context)
            elif method == "submissionByRequest":
                value = await storage.submission_by_request(args[0], args[1], context)
            elif method == "document":
                value = await storage.document(args[0], args[1], context)
            elif method == "findDocument":
                value = await storage.find_document(args[0], args[1], context)
            elif method == "scanDocuments":
                value = await storage.scan_documents(args[0], args[1], None, context)
            elif method == "mintId":
                value = await storage.mint_id()
            elif method == "commit":
                value = await storage.commit([decode_storage_write(write) for write in args[0]], context)
            else:
                raise AssertionError(f"Unknown read {method}")
            results.append({"ok": encode(method, value)})
        except Exception as error:
            results.append({"error": str(error)})
    # The comparison is of JSON, as upstream wrote it.
    return json_parse(json_stringify(results))


def typescript_store(backend: str, tmp_path: Path) -> Path:
    if backend == "sqlite":
        path = tmp_path / "ts.sqlite"
        path.write_bytes(gzip.decompress((FIXTURE / "ts.sqlite.gz").read_bytes()))
        return path
    path = tmp_path / "ts_jsonl"
    shutil.copytree(FIXTURE / "ts_jsonl", path)
    return path


@pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
async def test_reads_a_store_written_by_typescript_like_typescript(tmp_path: Path, backend: str) -> None:
    storage = await open_storage(backend, typescript_store(backend, tmp_path))
    try:
        assert await read_all(storage) == load("expected.json")
    finally:
        await storage.close(context)


@pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
async def test_writes_a_store_that_reads_like_the_typescript_one(tmp_path: Path, backend: str) -> None:
    path = tmp_path / ("py.sqlite" if backend == "sqlite" else "py_jsonl")
    storage = await open_storage(backend, path)
    await write_commits(storage)
    await storage.close(context)
    reopened = await open_storage(backend, path)
    try:
        assert await read_all(reopened) == load("expected.json")
    finally:
        await reopened.close(context)


def schema(path: Path) -> list[tuple[Any, ...]]:
    database = sqlite3.connect(path)
    try:
        return database.execute("SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY name").fetchall()
    finally:
        database.close()


async def test_creates_the_typescript_sqlite_schema_text_for_text(tmp_path: Path) -> None:
    storage = await open_sqlite_storage(tmp_path / "py.sqlite")
    await storage.close(context)
    assert schema(tmp_path / "py.sqlite") == schema(typescript_store("sqlite", tmp_path))


async def test_writes_the_typescript_jsonl_files_line_for_line(tmp_path: Path) -> None:
    path = tmp_path / "py_jsonl"
    storage = await open_storage("jsonl", path)
    await write_commits(storage)
    await storage.close(context)
    expected = FIXTURE / "ts_jsonl"
    assert sorted(file.name for file in path.iterdir()) == sorted(file.name for file in expected.iterdir())
    for file in expected.iterdir():
        written = (path / file.name).read_text("utf-8").splitlines()
        assert [json.loads(line) for line in written] == [
            json.loads(line) for line in file.read_text("utf-8").splitlines()
        ]
