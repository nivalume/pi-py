"""pi_chord SQLite adapter: small-transaction commit latency and throughput.

Each transaction is the shape of a durable commit: read a metadata row, write a record, bump the
metadata row (``get`` + 2 ``run`` through the transaction handle, then COMMIT). Measured on an
in-memory database and on an on-disk WAL database (``synchronous = NORMAL``):

- latency: one transaction at a time, awaited (p50/p99 per transaction);
- throughput: 32 tasks issuing transactions concurrently (they serialize on the connection);
- round trip: one ``get`` at a time (one thread hop each way);
- reads: 32 tasks issuing single-row ``get`` calls concurrently (results share thread hops);
- baseline: the same transaction run synchronously on a plain ``sqlite3`` connection, i.e. the cost
  of SQLite itself without the worker thread and event loop.

Also benchmarks :class:`pi_chord.output.OutputBuffer` push + snapshot on tool-output sized streams.

Run: python benchmarks/bench_chord_sqlite.py
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import statistics
import tempfile
import time

from pi_chord.output import OutputBuffer, OutputLimits
from pi_chord.sqlite import SqliteExecutor
from pi_chord.sqlite.stdlib import StdlibSqliteDatabase, open_sqlite_database

SCHEMA = (
    "CREATE TABLE meta (singleton INTEGER PRIMARY KEY, next_id INTEGER NOT NULL) STRICT;"
    "INSERT INTO meta VALUES (1, 1);"
    "CREATE TABLE records (id INTEGER PRIMARY KEY, record TEXT NOT NULL) STRICT;"
)
RECORD = '{"kind":"entry","text":"' + "x" * 200 + '"}'
LATENCY_TRANSACTIONS = 3_000
THROUGHPUT_TRANSACTIONS = 6_000
READS = 20_000
CONCURRENCY = 32


async def commit(transaction: SqliteExecutor) -> None:
    row = await transaction.get("SELECT next_id FROM meta WHERE singleton = 1")
    assert row is not None
    next_id = row["next_id"]
    await transaction.run("INSERT INTO records (id, record) VALUES (?, ?)", next_id, RECORD)
    await transaction.run("UPDATE meta SET next_id = ? WHERE singleton = 1", next_id + 1)


async def bench_latency(database: StdlibSqliteDatabase) -> tuple[float, float, float]:
    samples: list[float] = []
    for _ in range(LATENCY_TRANSACTIONS):
        started = time.perf_counter()
        await database.transaction(commit)
        samples.append(time.perf_counter() - started)
    samples.sort()
    p50 = statistics.median(samples)
    p99 = samples[int(len(samples) * 0.99)]
    return p50 * 1e6, p99 * 1e6, len(samples) / sum(samples)


async def bench_roundtrip(database: StdlibSqliteDatabase) -> float:
    started = time.perf_counter()
    for _ in range(READS // 4):
        await database.get("SELECT 1 AS one")
    return (time.perf_counter() - started) / (READS // 4) * 1e6


async def bench_throughput(database: StdlibSqliteDatabase) -> float:
    per_task = THROUGHPUT_TRANSACTIONS // CONCURRENCY

    async def worker() -> None:
        for _ in range(per_task):
            await database.transaction(commit)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(CONCURRENCY)))
    return per_task * CONCURRENCY / (time.perf_counter() - started)


async def bench_reads(database: StdlibSqliteDatabase) -> float:
    per_task = READS // CONCURRENCY

    async def worker(offset: int) -> None:
        for index in range(per_task):
            await database.get("SELECT id, record FROM records WHERE id = ?", 1 + (offset + index) % 1000)

    started = time.perf_counter()
    await asyncio.gather(*(worker(offset) for offset in range(CONCURRENCY)))
    return per_task * CONCURRENCY / (time.perf_counter() - started)


def bench_baseline(path: str) -> float:
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA)
    started = time.perf_counter()
    for _ in range(LATENCY_TRANSACTIONS):
        connection.execute("BEGIN IMMEDIATE")
        next_id = connection.execute("SELECT next_id FROM meta WHERE singleton = 1").fetchone()[0]
        connection.execute("INSERT INTO records (id, record) VALUES (?, ?)", (next_id, RECORD))
        connection.execute("UPDATE meta SET next_id = ? WHERE singleton = 1", (next_id + 1,))
        connection.execute("COMMIT")
    elapsed = time.perf_counter() - started
    connection.close()
    return elapsed / LATENCY_TRANSACTIONS * 1e6


async def run(label: str, path: str) -> None:
    database = await open_sqlite_database(path)
    await database.exec(SCHEMA)
    roundtrip = await bench_roundtrip(database)
    p50, p99, sequential = await bench_latency(database)
    throughput = await bench_throughput(database)
    reads = await bench_reads(database)
    await database.close()
    print(f"{label}:")
    print(f"  sequential single get round trip {roundtrip:7.1f} us")
    print(f"  sequential transaction latency  p50 {p50:7.1f} us   p99 {p99:7.1f} us   ({sequential:,.0f} tx/s)")
    print(f"  concurrent transactions ({CONCURRENCY:>2})   {throughput:,.0f} tx/s")
    print(f"  concurrent single-row gets ({CONCURRENCY:>2}) {reads:,.0f} ops/s")


def bench_output() -> None:
    limits = OutputLimits(max_bytes=50 * 1024, max_lines=2000, retain="tail")
    chunk = "".join(f"build step {index}: compiling module_{index}.py ... ok é\n" for index in range(20))
    for label, snapshot_every in (("push only", 0), ("push + snapshot every 10", 10)):
        buffer = OutputBuffer(limits)
        pushes = 20_000
        started = time.perf_counter()
        for index in range(pushes):
            buffer.push(chunk)
            if snapshot_every and index % snapshot_every == 0:
                buffer.snapshot()
        buffer.snapshot()
        elapsed = time.perf_counter() - started
        size = len(chunk.encode())
        print(
            f"  OutputBuffer tail {label:<26} {pushes / elapsed:,.0f} chunks/s"
            f"  ({pushes * size / elapsed / 1e6:,.1f} MB/s, {size} B chunks)"
        )


async def main() -> None:
    await run("in-memory", ":memory:")
    with tempfile.TemporaryDirectory() as directory:
        await run("on-disk WAL (synchronous=NORMAL)", os.path.join(directory, "bench.db"))
        memory = bench_baseline(":memory:")
        disk = bench_baseline(os.path.join(directory, "baseline.db"))
    print("plain sqlite3 baseline (same transaction, synchronous, no worker thread):")
    print(f"  in-memory {memory:6.1f} us/tx   on-disk WAL {disk:6.1f} us/tx")
    print("output:")
    bench_output()


if __name__ == "__main__":
    asyncio.run(main())
