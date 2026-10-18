"""pi_durable storage backends: the upstream storage benchmark scenarios on memory, SQLite and JSONL.

Port of upstream ``test/storage.bench.ts`` on :mod:`pi_durable.testing.storage_benchmark`:

- reads: each ``STORAGE_READ_BENCHMARKS`` scenario on a storage seeded with ``TIMING_SCALE`` (1,000 entries,
  300 tasks, 300 documents, replay tails, a rewindable history, an 8-deep fork chain), validated first;
- writes: each ``STORAGE_WRITE_BENCHMARKS`` scenario, one sample per freshly seeded storage;
- reopen: opening a seeded on-disk storage and its first exact entry read (SQLite and JSONL only).

Times are per operation (median and p90 of the samples). SQLite runs on a WAL database with
``synchronous = NORMAL`` and JSONL without ``fsync``, as upstream's defaults.

Run: python benchmarks/bench_durable_storage.py [--quick]
"""

from __future__ import annotations

import asyncio
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable

from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable import MemoryStorage
from pi_durable.storage.jsonl.local import open_local_jsonl_storage
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.testing import (
    STORAGE_READ_BENCHMARKS,
    STORAGE_WRITE_BENCHMARKS,
    seed_storage_benchmark,
    seed_storage_write_benchmark,
)
from pi_durable.types import Storage

BACKENDS = ("memory", "sqlite", "jsonl")
QUICK = "--quick" in sys.argv
READ_TIME_S = 0.1 if QUICK else 0.3
READ_MIN_ITERATIONS = 10
WRITE_ITERATIONS = 5 if QUICK else 20
WRITE_WARMUP = 2 if QUICK else 5
REOPEN_ITERATIONS = 5 if QUICK else 20
REOPEN_WARMUP = 1 if QUICK else 3

context = BACKGROUND_CONTEXT
directories: list[str] = []


def temporary_directory(prefix: str) -> str:
    directory = tempfile.mkdtemp(prefix=prefix)
    directories.append(directory)
    return directory


async def open_backend(backend: str, path: str | None = None) -> Storage:
    if backend == "memory":
        return MemoryStorage()
    if path is None:
        directory = temporary_directory("pi-durable-benchmark-")
        path = os.path.join(directory, "storage.sqlite" if backend == "sqlite" else "storage")
    if backend == "sqlite":
        return await open_sqlite_storage(path)
    return await open_local_jsonl_storage(path, context)


def summary(samples: list[float]) -> str:
    samples = sorted(samples)
    median = statistics.median(samples)
    p90 = samples[min(len(samples) - 1, int(len(samples) * 0.9))]
    return f"{median * 1e6:10.1f} µs  p90 {p90 * 1e6:10.1f} µs  ({len(samples)} samples)"


async def time_until(run: Callable[[], Awaitable[object]]) -> list[float]:
    for _ in range(3):
        await run()
    samples: list[float] = []
    deadline = time.perf_counter() + READ_TIME_S
    while time.perf_counter() < deadline or len(samples) < READ_MIN_ITERATIONS:
        started = time.perf_counter()
        await run()
        samples.append(time.perf_counter() - started)
    return samples


async def bench_reads() -> None:
    print("\n# reads (TIMING_SCALE)")
    fixtures: dict[str, Storage] = {}
    datasets = {}
    for backend in BACKENDS:
        storage = await open_backend(backend)
        started = time.perf_counter()
        datasets[backend] = await seed_storage_benchmark(storage)
        print(f"seed {backend:7s} {time.perf_counter() - started:8.2f} s")
        fixtures[backend] = storage
        for scenario in STORAGE_READ_BENCHMARKS:
            actual = await scenario.run(storage, datasets[backend])
            expected = scenario.expected(datasets[backend])
            if actual != expected:
                raise AssertionError(f"{backend} {scenario.name}: {actual} != {expected}")
    for scenario in STORAGE_READ_BENCHMARKS:
        print(f"\n{scenario.name}")
        for backend in BACKENDS:
            storage = fixtures[backend]
            dataset = datasets[backend]

            async def run(storage: Storage = storage, dataset=dataset, scenario=scenario) -> object:  # type: ignore[no-untyped-def]
                return await scenario.run(storage, dataset)

            print(f"  {backend:7s} {summary(await time_until(run))}")
    for storage in fixtures.values():
        await storage.close(context)


async def bench_writes() -> None:
    print("\n# writes (one sample per freshly seeded storage)")
    for scenario in STORAGE_WRITE_BENCHMARKS:
        print(f"\n{scenario.name}")
        for backend in BACKENDS:
            validation = await open_backend(backend)
            await seed_storage_write_benchmark(validation)
            if await scenario.run(validation) != scenario.expected:
                raise AssertionError(f"{backend} {scenario.name}")
            await validation.close(context)
            pool: list[Storage] = []
            for _ in range(WRITE_ITERATIONS + WRITE_WARMUP):
                storage = await open_backend(backend)
                await seed_storage_write_benchmark(storage)
                pool.append(storage)
            samples: list[float] = []
            for index, storage in enumerate(pool):
                started = time.perf_counter()
                await scenario.run(storage)
                elapsed = time.perf_counter() - started
                if index >= WRITE_WARMUP:
                    samples.append(elapsed)
                await storage.close(context)
            print(f"  {backend:7s} {summary(samples)}")


async def bench_reopen() -> None:
    print("\n# reopen and first exact read")
    for backend in ("sqlite", "jsonl"):
        seed_directory = temporary_directory(f"pi-durable-{backend}-reopen-benchmark-")
        name = "storage.sqlite" if backend == "sqlite" else "storage"
        seed_path = os.path.join(seed_directory, name)
        seed = await open_backend(backend, seed_path)
        dataset = await seed_storage_benchmark(seed)
        await seed.close(context)
        samples: list[float] = []
        for index in range(REOPEN_ITERATIONS + REOPEN_WARMUP):
            path = os.path.join(temporary_directory(f"pi-durable-{backend}-reopen-sample-"), name)
            if backend == "sqlite":
                shutil.copyfile(seed_path, path)
            else:
                shutil.copytree(seed_path, path)
            started = time.perf_counter()
            storage = await open_backend(backend, path)
            stored = await storage.entry(dataset["firstEntryId"], context)
            elapsed = time.perf_counter() - started
            if stored is None or stored["entry"]["id"] != dataset["firstEntryId"]:
                raise AssertionError(f"{backend} reopen read")
            await storage.close(context)
            if index >= REOPEN_WARMUP:
                samples.append(elapsed)
        print(f"  {backend:7s} {summary(samples)}")


async def main() -> None:
    try:
        await bench_reads()
        await bench_writes()
        await bench_reopen()
    finally:
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
