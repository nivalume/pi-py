"""Representative storage workloads for benchmarks (port of ``testing/storage-benchmark.ts``).

:func:`seed_storage_benchmark` fills a storage through the public ``Storage`` contract only;
:data:`STORAGE_READ_BENCHMARKS` and :data:`STORAGE_WRITE_BENCHMARKS` are the timed scenarios, each with the
value its ``run`` must return. ``benchmarks/bench_durable_storage.py`` runs them for the built-in backends.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypedDict

from pi_chord.context import BACKGROUND_CONTEXT

from ..types import (
    ROOT_CONVERSATION_ID,
    AnyTaskRecord,
    ConversationId,
    DocumentCreate,
    DocumentId,
    EntryId,
    Seq,
    Storage,
    StorageWrite,
)

__all__ = [
    "STORAGE_MEMORY_SCALES",
    "STORAGE_READ_BENCHMARKS",
    "STORAGE_WRITE_BENCHMARKS",
    "TIMING_SCALE",
    "StorageBenchmarkDataset",
    "StorageBenchmarkScale",
    "StorageReadBenchmark",
    "StorageWriteBenchmark",
    "seed_storage_benchmark",
    "seed_storage_write_benchmark",
    "storage_benchmark_primary_record_count",
]

context = BACKGROUND_CONTEXT


class StorageBenchmarkScale(TypedDict):
    name: str
    entryCount: int
    taskCount: int
    documentCount: int


STORAGE_MEMORY_SCALES: Final[Sequence[StorageBenchmarkScale]] = (
    {"name": "1k", "entryCount": 1_000, "taskCount": 200, "documentCount": 200},
    {"name": "10k", "entryCount": 10_000, "taskCount": 2_000, "documentCount": 2_000},
)

TIMING_SCALE: Final[StorageBenchmarkScale] = {
    "name": "timing",
    "entryCount": 1_000,
    "taskCount": 300,
    "documentCount": 300,
}

REPLAY_TAILS: Final = (0, 16, 128, 1_024)
HISTORY_SEGMENT_LENGTH: Final = 128
FORK_DEPTH: Final = 8
ENTRIES_PER_FORK: Final = 32
BATCH_SIZE: Final = 100


def storage_benchmark_primary_record_count(scale: StorageBenchmarkScale) -> int:
    """The conversations, entries, tasks and documents a seed creates."""
    return (
        1
        + scale["entryCount"]
        + scale["taskCount"]
        + scale["documentCount"]
        + len(REPLAY_TAILS)
        + 1
        + FORK_DEPTH * (1 + ENTRIES_PER_FORK)
    )


class StorageBenchmarkDataset(TypedDict):
    firstEntryId: EntryId
    filteredTaskCount: int
    exactDocumentId: DocumentId
    exactDocumentKey: str
    #: Keyed by replay tail length (``0``, ``16``, ``128``, ``1024``).
    replayDocumentIds: Mapping[int, DocumentId]
    historicalDocumentId: DocumentId
    ancientAt: Seq
    recentAt: Seq
    deepestConversationId: ConversationId
    ancestorHeadEntryId: EntryId


def _task(id: int, index: int) -> AnyTaskRecord:
    statuses = ("pending", "running", "terminal")
    status = statuses[index % len(statuses)]
    common: dict[str, Any] = {
        "id": id,
        "conversationId": ROOT_CONVERSATION_ID,
        "kind": "benchmark.filtered" if index % 4 == 0 else "benchmark.other",
        "version": 1,
        "input": {"index": index},
        "background": index % 5 == 0,
        "abortRequested": index % 7 == 0,
    }
    if status == "terminal":
        return {**common, "state": {"status": status, "outcome": {"status": "completed", "result": {"index": index}}}}  # type: ignore[return-value,typeddict-item]
    return {**common, "state": {"status": status, "checkpoint": {"index": index, "payload": "x" * 64}}}  # type: ignore[return-value,typeddict-item]


async def seed_storage_benchmark(
    storage: Storage, scale: StorageBenchmarkScale = TIMING_SCALE
) -> StorageBenchmarkDataset:
    """Seed deterministic representative data through only the public ``Storage`` contract."""
    await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)

    first_entry_id: int | None = None
    for start in range(0, scale["entryCount"], BATCH_SIZE):
        writes: list[StorageWrite] = []
        for index in range(start, min(start + BATCH_SIZE, scale["entryCount"])):
            id = await storage.mint_id()
            if index == 0:
                first_entry_id = id
            value: dict[str, Any] = {"id": id, "conversationId": ROOT_CONVERSATION_ID, "kind": "benchmark.entry"}
            if index == 0:
                value["head"] = id
            value["data"] = {"index": index, "text": f"entry-{index}-{'x' * 96}"}
            writes.append({"type": "entry", "value": value})  # type: ignore[typeddict-item]
        await storage.commit(writes, context)

    for start in range(0, scale["taskCount"], BATCH_SIZE):
        writes = []
        for index in range(start, min(start + BATCH_SIZE, scale["taskCount"])):
            writes.append({"type": "task", "value": _task(await storage.mint_id(), index)})
        await storage.commit(writes, context)

    exact_document_id: int | None = None
    for start in range(0, scale["documentCount"], BATCH_SIZE):
        writes = []
        for index in range(start, min(start + BATCH_SIZE, scale["documentCount"])):
            id = await storage.mint_id()
            exact_document_id = id
            writes.append(
                {
                    "type": "document.create",
                    "record": {
                        "id": id,
                        "kind": "benchmark.family",
                        "key": f"key-{index}",
                        "scope": {"kind": "session"},
                    },  # type: ignore[typeddict-item]
                    "content": {"kind": "base", "version": 1, "value": {"index": index, "text": "x" * 128}},
                }
            )
        await storage.commit(writes, context)

    replay_entries = [(tail, await storage.mint_id()) for tail in REPLAY_TAILS]
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": {
                    "id": id,  # type: ignore[typeddict-item]
                    "kind": "benchmark.replay",
                    "key": str(tail),
                    "scope": {"kind": "conversation", "conversationId": ROOT_CONVERSATION_ID},
                    "history": "rewindable",
                    "fork": "asOf",
                },
                "content": {"kind": "base", "version": 1, "value": {"count": 0, "text": "x" * 64}},
            }
            for tail, id in replay_entries
        ],
        context,
    )
    for count in range(1, REPLAY_TAILS[-1] + 1):
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": id,  # type: ignore[typeddict-item]
                    "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), count)]},
                }
                for tail, id in replay_entries
                if count <= tail
            ],
            context,
        )

    historical_document_id = await storage.mint_id()
    historical_record: DocumentCreate = {
        "id": historical_document_id,  # type: ignore[typeddict-item]
        "kind": "benchmark.history",
        "scope": {"kind": "conversation", "conversationId": ROOT_CONVERSATION_ID},
        "history": "rewindable",
        "fork": "asOf",
    }
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": historical_record,
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ],
        context,
    )

    def change(count: int) -> StorageWrite:
        return {
            "type": "document.change",
            "id": historical_document_id,  # type: ignore[typeddict-item]
            "content": {"kind": "delta", "version": 1, "ops": [("s", ("count",), count)]},
        }

    ancient_at: Seq | None = None
    for count in range(1, HISTORY_SEGMENT_LENGTH + 1):
        ancient_at = await storage.commit([change(count)], context)
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": historical_document_id,  # type: ignore[typeddict-item]
                "content": {"kind": "base", "version": 1, "value": {"count": HISTORY_SEGMENT_LENGTH}},
            }
        ],
        context,
    )
    if ancient_at is None:
        raise RuntimeError("Benchmark history seed produced no commits")
    recent_at = ancient_at
    for count in range(HISTORY_SEGMENT_LENGTH + 1, HISTORY_SEGMENT_LENGTH * 2 + 1):
        recent_at = await storage.commit([change(count)], context)

    if first_entry_id is None:
        raise RuntimeError("Benchmark scale must create entries")
    parent_conversation_id: int = ROOT_CONVERSATION_ID
    parent_at = first_entry_id
    deepest_conversation_id: int = ROOT_CONVERSATION_ID
    for depth in range(FORK_DEPTH):
        conversation_id = await storage.mint_id()
        await storage.commit(
            [
                {
                    "type": "conversation",
                    "value": {
                        "id": conversation_id,
                        "parent": {"conversationId": parent_conversation_id, "at": parent_at},
                    },  # type: ignore[typeddict-item]
                }
            ],
            context,
        )
        ids = [await storage.mint_id() for _ in range(ENTRIES_PER_FORK)]
        await storage.commit(
            [
                {
                    "type": "entry",
                    "value": {
                        "id": id,
                        "conversationId": conversation_id,
                        "kind": "benchmark.fork",
                        "data": {"depth": depth, "index": index},
                    },  # type: ignore[typeddict-item]
                }
                for index, id in enumerate(ids)
            ],
            context,
        )
        parent_conversation_id = conversation_id
        parent_at = ids[-1]
        deepest_conversation_id = conversation_id

    if exact_document_id is None:
        raise RuntimeError("Benchmark scale must create documents")
    return {
        "firstEntryId": first_entry_id,  # type: ignore[typeddict-item]
        "filteredTaskCount": min(50, math.ceil(scale["taskCount"] / 60)),
        "exactDocumentId": exact_document_id,  # type: ignore[typeddict-item]
        "exactDocumentKey": f"key-{scale['documentCount'] - 1}",
        "replayDocumentIds": dict(replay_entries),  # type: ignore[typeddict-item]
        "historicalDocumentId": historical_document_id,  # type: ignore[typeddict-item]
        "ancientAt": ancient_at,
        "recentAt": recent_at,
        "deepestConversationId": deepest_conversation_id,  # type: ignore[typeddict-item]
        "ancestorHeadEntryId": first_entry_id,  # type: ignore[typeddict-item]
    }


@dataclass(frozen=True, slots=True)
class StorageReadBenchmark:
    """A timed read: ``await run(storage, dataset)`` returns ``expected(dataset)`` on a correct storage."""

    name: str
    run: Callable[[Storage, StorageBenchmarkDataset], Awaitable[float]]
    expected: Callable[[StorageBenchmarkDataset], float]


async def _exact_entry(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    stored = await storage.entry(dataset["firstEntryId"], context)
    return -1 if stored is None else stored["entry"]["id"]


async def _entry_page(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    return len((await storage.scan_entries({"conversationId": ROOT_CONVERSATION_ID}, 100, None, context))["items"])


async def _filtered_tasks(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    page = await storage.scan_tasks(
        {"kind": "benchmark.filtered", "status": "pending", "background": True}, 50, None, context
    )
    return len(page["items"])


async def _exact_document(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    record = await storage.find_document(
        {"kind": "benchmark.family", "key": dataset["exactDocumentKey"], "scope": {"kind": "session"}},
        "current",
        context,
    )
    return -1 if record is None else record["id"]


def _replay(tail: int) -> StorageReadBenchmark:
    async def run(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
        stored = await storage.document(dataset["replayDocumentIds"][tail], "current", context)
        return math.nan if stored is None else stored["value"]["count"]  # type: ignore[return-value]

    return StorageReadBenchmark(f"document replay tail ({tail})", run, lambda dataset: tail)


async def _ancient(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    stored = await storage.document(dataset["historicalDocumentId"], dataset["ancientAt"], context)
    return math.nan if stored is None else stored["value"]["count"]  # type: ignore[return-value]


async def _recent(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    stored = await storage.document(dataset["historicalDocumentId"], dataset["recentAt"], context)
    return math.nan if stored is None else stored["value"]["count"]  # type: ignore[return-value]


async def _fork_scan(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    page = await storage.scan_entries({"conversationId": dataset["deepestConversationId"]}, 100, None, context)
    return len(page["items"])


async def _fork_head(storage: Storage, dataset: StorageBenchmarkDataset) -> float:
    marker = await storage.find_latest_head_marker(dataset["deepestConversationId"], None, context)
    return -1 if marker is None else marker["id"]


STORAGE_READ_BENCHMARKS: Final[Sequence[StorageReadBenchmark]] = (
    StorageReadBenchmark("exact entry lookup", _exact_entry, lambda dataset: dataset["firstEntryId"]),
    StorageReadBenchmark("entry page scan (100)", _entry_page, lambda dataset: 100),
    StorageReadBenchmark("filtered task scan (50)", _filtered_tasks, lambda dataset: dataset["filteredTaskCount"]),
    StorageReadBenchmark(
        "exact document address among many", _exact_document, lambda dataset: dataset["exactDocumentId"]
    ),
    *(_replay(tail) for tail in REPLAY_TAILS),
    StorageReadBenchmark("ancient historical read before newer base", _ancient, lambda dataset: HISTORY_SEGMENT_LENGTH),
    StorageReadBenchmark(
        "recent historical read after newer base", _recent, lambda dataset: HISTORY_SEGMENT_LENGTH * 2
    ),
    StorageReadBenchmark("fork-depth history scan (100)", _fork_scan, lambda dataset: 100),
    StorageReadBenchmark("fork-depth head lookup", _fork_head, lambda dataset: dataset["ancestorHeadEntryId"]),
)


@dataclass(frozen=True, slots=True)
class StorageWriteBenchmark:
    """A timed write on a storage seeded by :func:`seed_storage_write_benchmark`; ``run`` returns ``expected``."""

    name: str
    expected: int
    run: Callable[[Storage], Awaitable[int]]


async def seed_storage_write_benchmark(storage: Storage) -> None:
    """Seed the common state expected by every write benchmark sample."""
    await storage.commit([{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}], context)
    writes: list[StorageWrite] = []
    for index in range(100):
        writes.append(
            {
                "type": "entry",
                "value": {
                    "id": await storage.mint_id(),
                    "conversationId": ROOT_CONVERSATION_ID,
                    "kind": "benchmark.baseline",
                    "data": {"index": index},
                },  # type: ignore[typeddict-item]
            }
        )
    await storage.commit(writes, context)


async def _commit_one(storage: Storage) -> int:
    id = await storage.mint_id()
    await storage.commit(
        [
            {
                "type": "entry",
                "value": {
                    "id": id,
                    "conversationId": ROOT_CONVERSATION_ID,
                    "kind": "benchmark.write",
                    "data": {"text": "x" * 128},
                },  # type: ignore[typeddict-item]
            }
        ],
        context,
    )
    return 1


async def _commit_hundred(storage: Storage) -> int:
    writes: list[StorageWrite] = []
    for index in range(100):
        writes.append(
            {
                "type": "entry",
                "value": {
                    "id": await storage.mint_id(),  # type: ignore[typeddict-item]
                    "conversationId": ROOT_CONVERSATION_ID,
                    "kind": "benchmark.write",
                    "data": {"index": index, "text": "x" * 128},
                },
            }
        )
    await storage.commit(writes, context)
    return len(writes)


async def _commit_mixed(storage: Storage) -> int:
    entry_id = await storage.mint_id()
    task_id = await storage.mint_id()
    submission_id = await storage.mint_id()
    document_id = await storage.mint_id()
    writes: list[StorageWrite] = [
        {"type": "entry", "value": {"id": entry_id, "conversationId": ROOT_CONVERSATION_ID, "kind": "benchmark.mixed"}},  # type: ignore[typeddict-item]
        {"type": "task", "value": _task(task_id, task_id)},
        {
            "type": "submission",
            "value": {
                "id": submission_id,  # type: ignore[typeddict-item]
                "conversationId": ROOT_CONVERSATION_ID,
                "requestId": f"benchmark-{submission_id}",
                "type": "write",
                "status": "done",
                "entry": entry_id,  # type: ignore[typeddict-item]
            },
        },
        {
            "type": "document.create",
            "record": {
                "id": document_id,
                "kind": "benchmark.mixed",
                "key": str(document_id),
                "scope": {"kind": "session"},
            },  # type: ignore[typeddict-item]
            "content": {"kind": "base", "version": 1, "value": {"entryId": entry_id, "taskId": task_id}},
        },
    ]
    await storage.commit(writes, context)
    return len(writes)


STORAGE_WRITE_BENCHMARKS: Final[Sequence[StorageWriteBenchmark]] = (
    StorageWriteBenchmark("commit one entry", 1, _commit_one),
    StorageWriteBenchmark("commit 100 entries", 100, _commit_hundred),
    StorageWriteBenchmark("commit mixed entry/task/submission/document", 4, _commit_mixed),
)
