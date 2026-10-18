"""Shared helpers of the Session tests (port of ``test/session-support.ts``).

The suite runs with ``--import-mode=importlib``; ``conftest.py`` puts this directory on ``sys.path``.

Storage backends: every Session test takes the ``storage_backend`` fixture, parametrized over
:data:`STORAGE_BACKENDS`. Upstream runs these tests on its memory storage only; to run them on
another backend too, add a :class:`StorageBackend` entry here (``open`` makes a fresh, empty storage;
``reopen`` returns a storage over the same data after the first was closed, as a fresh process would
reopen the same database).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NamedTuple

from pi_chord.context import BACKGROUND_CONTEXT, Context
from pi_durable.session import SessionImpl
from pi_durable.storage.codec import clone_value
from pi_durable.storage.memory import MemoryStorage
from pi_durable.types import (
    CommitPublication,
    ConversationId,
    ConversationQuery,
    Cursor,
    DocumentAddress,
    DocumentId,
    DocumentPoint,
    DocumentQuery,
    EntryId,
    EntryQuery,
    Seq,
    Storage,
    StorageWrite,
    SubmissionId,
    SubmissionQuery,
    TaskQuery,
)

context: Context = BACKGROUND_CONTEXT


class StorageBackend(NamedTuple):
    """A storage backend the Session tests run against."""

    name: str
    open: Callable[[], Awaitable[Storage]]
    reopen: Callable[[Storage], Awaitable[Storage]]


async def _open_memory() -> Storage:
    return MemoryStorage()


async def _reopen_memory(storage: Storage) -> Storage:
    assert isinstance(storage, MemoryStorage)
    storage._closed = False  # pyright: ignore[reportAttributeAccessIssue]
    return storage


#: The backends of the ``storage_backend`` fixture, by name. Extend it to run the Session tests elsewhere.
STORAGE_BACKENDS: dict[str, StorageBackend] = {"memory": StorageBackend("memory", _open_memory, _reopen_memory)}


class Gate:
    """A gate that holds calls until released and reports when the first held call arrives."""

    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self._gate: asyncio.Future[None] = loop.create_future()
        self._entered: asyncio.Future[None] = loop.create_future()

    @property
    def entered(self) -> asyncio.Future[None]:
        return self._entered

    def enter(self) -> None:
        if not self._entered.done():
            self._entered.set_result(None)

    def open(self) -> None:
        if not self._gate.done():
            self._gate.set_result(None)

    async def wait(self) -> None:
        await self._gate


class HeldGate(NamedTuple):
    entered: asyncio.Future[None]
    release: Callable[[], None]


class ControlledStorage:
    """A storage with observable commits, held calls, and injected commit failures.

    Wraps a backend's storage; upstream subclasses its memory storage.
    """

    def __init__(self, inner: Storage, backend: StorageBackend) -> None:
        self.inner = inner
        self.backend = backend
        #: Exact borrowed batches admitted by the Session.
        self.admitted_commits: list[Sequence[StorageWrite]] = []
        #: Detached batches for value assertions.
        self.commits: list[list[StorageWrite]] = []
        self.mint_count = 0
        self.document_read_count = 0
        self._commit_gate: Gate | None = None
        self._find_gate: Gate | None = None
        self._commit_failure: BaseException | None = None

    def hold_commits(self) -> HeldGate:
        held = Gate()
        self._commit_gate = held
        return HeldGate(held.entered, lambda: self._release("commit", held))

    def hold_find_document(self) -> HeldGate:
        held = Gate()
        self._find_gate = held
        return HeldGate(held.entered, lambda: self._release("find", held))

    def crash(self) -> None:
        """Simulate a crash during the held commit: it never reaches storage, and later commits proceed."""
        self._commit_gate = None

    def fail_next_commit(self, error: BaseException) -> None:
        self._commit_failure = error

    async def reopen(self) -> ControlledStorage:
        """Reopen after ``close()``, as a fresh process would reopen the same database."""
        self.inner = await self.backend.reopen(self.inner)
        return self

    def _release(self, kind: str, held: Gate) -> None:
        if kind == "commit" and self._commit_gate is held:
            self._commit_gate = None
        if kind == "find" and self._find_gate is held:
            self._find_gate = None
        held.open()

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        self.admitted_commits.append(writes)
        self.commits.append(clone_value(list(writes)))
        held = self._commit_gate
        if held is not None:
            held.enter()
            await held.wait()
        failure = self._commit_failure
        if failure is not None:
            self._commit_failure = None
            raise failure
        return await self.inner.commit(writes, context)

    async def mint_id(self) -> int:
        self.mint_count += 1
        return await self.inner.mint_id()

    async def document(self, id: DocumentId, at: DocumentPoint, context: Context) -> Any:
        self.document_read_count += 1
        return await self.inner.document(id, at, context)

    async def find_document(self, address: DocumentAddress, at: DocumentPoint, context: Context) -> Any:
        held = self._find_gate
        if held is not None:
            held.enter()
            await held.wait()
        return await self.inner.find_document(address, at, context)

    async def conversation(self, id: ConversationId, context: Context) -> Any:
        return await self.inner.conversation(id, context)

    async def scan_conversations(
        self, query: ConversationQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Any:
        return await self.inner.scan_conversations(query, limit, cursor, context)

    async def entry(self, id: EntryId, context: Context) -> Any:
        return await self.inner.entry(id, context)

    async def visible_entry(self, conversation_id: ConversationId, id: EntryId, context: Context) -> Any:
        return await self.inner.visible_entry(conversation_id, id, context)

    async def find_latest_head_marker(
        self, conversation_id: ConversationId, at_or_before_entry_id: EntryId | None, context: Context
    ) -> Any:
        return await self.inner.find_latest_head_marker(conversation_id, at_or_before_entry_id, context)

    async def scan_entries(self, query: EntryQuery, limit: int, cursor: Cursor | None, context: Context) -> Any:
        return await self.inner.scan_entries(query, limit, cursor, context)

    async def task(self, id: Any, context: Context) -> Any:
        return await self.inner.task(id, context)

    async def scan_tasks(self, query: TaskQuery, limit: int, cursor: Cursor | None, context: Context) -> Any:
        return await self.inner.scan_tasks(query, limit, cursor, context)

    async def submission(self, id: SubmissionId, context: Context) -> Any:
        return await self.inner.submission(id, context)

    async def scan_submissions(
        self, query: SubmissionQuery, limit: int, cursor: Cursor | None, context: Context
    ) -> Any:
        return await self.inner.scan_submissions(query, limit, cursor, context)

    async def submission_by_request(self, conversation_id: ConversationId, request_id: str, context: Context) -> Any:
        return await self.inner.submission_by_request(conversation_id, request_id, context)

    async def scan_documents(self, query: DocumentQuery, limit: int, cursor: Cursor | None, context: Context) -> Any:
        return await self.inner.scan_documents(query, limit, cursor, context)

    async def close(self, context: Context) -> None:
        await self.inner.close(context)


async def controlled_storage(backend: StorageBackend) -> ControlledStorage:
    return ControlledStorage(await backend.open(), backend)


class OpenedSession(NamedTuple):
    storage: ControlledStorage
    session: SessionImpl
    publications: list[CommitPublication]


async def open_test_session(backend: StorageBackend) -> OpenedSession:
    """Session kernel plus its controlled storage and every committed publication."""
    storage = await controlled_storage(backend)
    # These tests pin the cache's identities and loads, so documents stay as long as the default keeps them.
    session = SessionImpl(storage, None, lambda: 600_000)
    publications: list[CommitPublication] = []
    session.subscribe_commits(lambda publication, _context: publications.append(publication))
    return OpenedSession(storage, session, publications)


def document_changes(publication: CommitPublication) -> list[Any]:
    return [change for change in publication["changes"] if change["type"] == "document"]


def document_copy_changes(publication: CommitPublication) -> list[Any]:
    return [change for change in publication["changes"] if change["type"] == "document.copy"]


async def create_conversation(session: SessionImpl) -> ConversationId:
    """Create one conversation and return its ID."""

    async def change(tx: Any) -> ConversationId:
        return (await tx.create_conversation({"ownership": {"kind": "ownerless"}}))["id"]

    return await session.commit(change, context)


async def flush() -> None:
    """Let pending callbacks and the tasks they start run (upstream: one macrotask turn)."""
    for _ in range(25):
        await asyncio.sleep(0)


async def eventually(check: Callable[[], bool | Awaitable[bool]]) -> None:
    for _ in range(200):
        result = check()
        if not isinstance(result, bool):
            result = await result
        if result:
            return
        await flush()
    raise AssertionError("Condition was not reached")


async def settle_line(session: SessionImpl) -> None:
    """Let the line run what is queued on it, such as an eviction job."""

    async def nothing() -> None:
        return None

    await session.read_on_line(nothing)


def deferred() -> asyncio.Future[Any]:
    return asyncio.get_running_loop().create_future()


def resolve(future: asyncio.Future[Any], value: Any = None) -> None:
    if not future.done():
        future.set_result(value)
