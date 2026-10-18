"""Streaming tool-call argument parsing: incremental parser vs. re-parsing every delta.

Simulates a model writing a file through a tool call: ~40 KB of JSON arguments
arriving in small deltas. pi (TypeScript) re-parses the accumulated buffer on
every delta; pi-py's adapters use ``StreamingJsonParser``, which keeps the lexer
state between deltas.

Run: python benchmarks/bench_partial_json.py
"""

from __future__ import annotations

import json
import time

from pi_ai import StreamingJsonParser, parse_streaming_json


def make_deltas(size: int = 40_000, chunk: int = 24) -> list[str]:
    content = "".join(f'line {i}: value = compute({i}) \\ "quoted"\n' for i in range(size // 40))
    text = json.dumps({"path": "src/generated.py", "content": content, "mode": "overwrite"})
    return [text[i : i + chunk] for i in range(0, len(text), chunk)]


def naive(deltas: list[str]) -> object:
    buffer = ""
    value: object = {}
    for delta in deltas:
        buffer += delta
        value = parse_streaming_json(buffer)
    return value


def incremental(deltas: list[str]) -> object:
    parser = StreamingJsonParser()
    value: object = {}
    for delta in deltas:
        parser.feed(delta)
        value = parser.value()
    return value


def bench(name: str, fn, deltas: list[str], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(deltas)
        best = min(best, time.perf_counter() - start)
    assert isinstance(result, dict) and result["mode"] == "overwrite"
    print(f"{name:<12} {best * 1000:8.1f} ms  ({len(deltas)} deltas, {best / len(deltas) * 1e6:6.1f} µs/delta)")
    return best


if __name__ == "__main__":
    deltas = make_deltas()
    slow = bench("re-parse", naive, deltas)
    fast = bench("incremental", incremental, deltas)
    print(f"speed-up: {slow / fast:.1f}x")
