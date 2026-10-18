"""Hot paths of pi_chord.delta: track + commit of durable-sized documents, and batch replay.

Each commit scenario runs ``begin_change`` -> mutate the draft -> ``prepare`` -> ``adopt``
on documents shaped like pi-durable's (live tool slots, usage totals, inbox queues).
Replay scenarios apply the recorded batches with ``apply_immutable`` (one batch at a
time, like a view mount), ``apply_immutable_batches`` (a storage backlog) and the mutable
``apply`` (a replica restored from JSON).

Run: python benchmarks/bench_chord_delta.py [--quick]
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from typing import Any

from pi_chord.delta import apply, apply_immutable, apply_immutable_batches, track

QUICK = "--quick" in sys.argv


def usage() -> dict[str, Any]:
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0},
    }


def live_document(tools: int, nested: int) -> dict[str, Any]:
    return {
        "run": {"taskId": 7, "inputs": [1, 2, 3], "startedAt": 1_700_000_000_000},
        "generation": {"model": "claude", "text": "x" * 2_000, "thinking": "", "toolCalls": []},
        "tools": [
            {"taskId": 100 + index, "name": "bash", "status": "running", "output": "line\n" * 200, "droppedBytes": 0}
            for index in range(tools)
        ],
        "nestedTools": [{"taskId": 1_000 + index, "parentTaskId": 100, "status": "done"} for index in range(nested)],
        "compactions": [],
    }


def usage_document() -> dict[str, Any]:
    return {
        "models": {f"provider/model-{index}": usage() for index in range(8)},
        "tools": {f"tool-{index}": usage() for index in range(16)},
    }


def inbox_document(items: int) -> dict[str, Any]:
    return {
        "items": [
            {"id": index, "mode": "followUp", "content": [{"type": "text", "text": f"message {index}"}]}
            for index in range(items)
        ],
        "boundary": {"head": "entry-1", "steeringMode": "all", "followUpMode": "one-at-a-time"},
    }


def scalar_writes(draft: Any, step: int) -> None:
    draft["run"]["startedAt"] = step
    draft["generation"]["model"] = f"model-{step % 3}"
    draft["tools"][step % len(draft["tools"])]["status"] = "done" if step % 2 else "running"


def stream_append(draft: Any, step: int) -> None:
    slot = draft["tools"][3]
    slot["output"] += f"chunk {step}\n"
    slot["droppedBytes"] = step


def add_usage(draft: Any, step: int) -> None:
    total = draft["models"]["provider/model-3"]
    for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
        total[key] += step
    cost = total["cost"]
    for key in ("input", "output", "cacheRead", "cacheWrite", "total"):
        cost[key] += 0.001


def nested_lookup(draft: Any, step: int) -> None:
    nested = draft["nestedTools"]
    target = 1_000 + (step * 7919) % len(nested)
    low, high = 0, len(nested) - 1
    while low <= high:
        middle = (low + high) // 2
        candidate = nested[middle]
        if candidate["taskId"] == target:
            candidate["status"] = "running" if candidate["status"] == "done" else "done"
            return
        if candidate["taskId"] < target:
            low = middle + 1
        else:
            high = middle - 1


def queue_cycle(draft: Any, step: int) -> None:
    items = draft["items"]
    items.pop(0)
    items.append({"id": 10_000 + step, "mode": "followUp", "content": [{"type": "text", "text": "next"}]})
    draft["boundary"]["head"] = f"entry-{step}"


def append_entries(draft: Any, step: int) -> None:
    draft["entries"].append({"id": step, "kind": "pi.assistant", "text": "ok"})


def reorder(draft: Any, step: int) -> None:
    draft["tools"].sort(key=lambda slot: (slot["taskId"] * (step % 3 + 1)) % 17)


def read_only(draft: Any, step: int) -> None:
    for slot in draft["tools"]:
        _ = slot["status"]
    _ = draft["run"]["inputs"][0]


SCENARIOS: list[tuple[str, Callable[[], dict[str, Any]], Callable[[Any, int], None]]] = [
    ("scalar writes (live doc)", lambda: live_document(10, 200), scalar_writes),
    ("string append 1KB output", lambda: live_document(10, 200), stream_append),
    ("usage counters (10 increments)", usage_document, add_usage),
    ("binary search 5k nested + 1 write", lambda: live_document(10, 5_000), nested_lookup),
    ("queue shift + push (200 items)", lambda: inbox_document(200), queue_cycle),
    ("append to 2k-entry list", lambda: {"entries": [{"id": index} for index in range(2_000)]}, append_entries),
    ("sort 10 tool slots", lambda: live_document(10, 200), reorder),
    ("read-only traversal (no-op)", lambda: live_document(10, 200), read_only),
]


def timed(run: Callable[[], None], iterations: int, repeat: int = 5) -> float:
    """Best per-iteration time of ``repeat`` runs, in microseconds."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - start) / iterations)
    return best * 1e6


def commits(make: Callable[[], dict[str, Any]], mutate: Callable[[Any, int], None], iterations: int) -> list[Any]:
    """Run ``iterations`` commits on a fresh tracker; returns the batches."""
    tracker = track(make())
    batches: list[Any] = []
    for step in range(iterations):
        change = tracker.begin_change()
        mutate(change.state, step)
        prepared = change.prepare()
        tracker.adopt(prepared)
        batches.append(prepared.ops)
    return batches


def main() -> None:
    iterations = 500 if QUICK else 3_000
    print(f"{'scenario':38} {'commit µs':>10} {'ops':>5} {'apply_immutable µs':>19} {'apply µs':>9}")
    for name, make, mutate in SCENARIOS:
        commit_us = timed(lambda make=make, mutate=mutate: commits(make, mutate, iterations) and None, iterations)
        batches = commits(make, mutate, iterations)
        initial = make()
        ops = sum(len(batch) for batch in batches) / len(batches)

        def immutable_replay(initial: Any = initial, batches: list[Any] = batches) -> None:
            value = initial
            for batch in batches:
                value = apply_immutable(value, batch)

        immutable_us = timed(immutable_replay, iterations)
        wire = json.dumps(batches)
        base_json = json.dumps(initial)

        def mutable_replay(wire: str = wire, base_json: str = base_json) -> None:
            replica = json.loads(base_json)
            for batch in json.loads(wire):
                replica = apply(replica, batch)

        mutable_us = timed(mutable_replay, iterations)
        print(f"{name:38} {commit_us:10.1f} {ops:5.1f} {immutable_us:19.1f} {mutable_us:9.1f}")

    def streaming(draft: Any, step: int) -> None:
        stream_append(draft, step)
        scalar_writes(draft, step)

    batches = commits(lambda: live_document(10, 200), streaming, 1_000)
    base = live_document(10, 200)
    backlog_us = timed(lambda: apply_immutable_batches(base, batches) and None, 1)
    document = {"nested": {"rows": [{"value": index} for index in range(100)]}}
    draft = track(document).begin_change().state

    def draft_reads(draft: Any = draft) -> None:
        for index in range(20_000):
            _ = draft["nested"]["rows"][index % 100]["value"]

    def raw_reads(document: Any = document) -> None:
        for index in range(20_000):
            _ = document["nested"]["rows"][index % 100]["value"]

    draft_ms = timed(draft_reads, 1) / 1_000
    raw_ms = timed(raw_reads, 1) / 1_000
    print(f"\n20,000 hot nested reads: {draft_ms:.2f} ms through a draft, {raw_ms:.2f} ms on plain dicts")
    print(f"apply_immutable_batches, 1000-batch backlog: {backlog_us:.0f} µs ({backlog_us / 1_000:.1f} µs/batch)")


if __name__ == "__main__":
    main()
