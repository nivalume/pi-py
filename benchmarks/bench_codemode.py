"""Hot paths of pi_codemode on Monty: per-execution overhead, tool-call round trips, gathered calls.

- ``execute``: a script that only returns, end to end through ``CodemodeSandbox.execute()`` (pool
  checkout, prelude feed, script feed, result), on a warm pool.
- ``start``: the same through a deterministic ``start()`` run.
- ``round trip``: one awaited tool call, measured over a loop of sequential calls in one script.
- ``gather N``: ``asyncio.gather`` over N calls of a tool that sleeps 10 ms, which shows the calls run
  concurrently on the host (wall time stays near 10 ms plus per-call overhead).
- ``output``: ``text()`` throughput through the record stream.

Run: python benchmarks/bench_codemode.py [--quick]
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pi_codemode import CodemodeSandbox, CodemodeSuccess, CodemodeTool

QUICK = "--quick" in sys.argv


async def timed(run: Callable[[], Awaitable[Any]], repeat: int) -> list[float]:
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = await run()
        samples.append((time.perf_counter() - started) * 1000)
        if isinstance(result, CodemodeSuccess) is False and hasattr(result, "ok"):
            raise RuntimeError(f"benchmark script failed: {result}")
    return samples


def report(name: str, samples: list[float], per: int = 1, unit: str = "ms") -> None:
    median = statistics.median(samples) / per
    best = min(samples) / per
    if unit == "us":
        print(f"{name:<34} median {median * 1000:9.1f} us   best {best * 1000:9.1f} us")
    else:
        print(f"{name:<34} median {median:9.2f} ms   best {best:9.2f} ms")


async def main() -> None:
    async def sleep_tool(args: Any, _context: Any) -> Any:
        await asyncio.sleep(0.01)
        return args

    sandbox = CodemodeSandbox(
        tools=[CodemodeTool("echo", lambda args, _context: args), CodemodeTool("sleep", sleep_tool)],
        timeout_ms=120_000,
    )
    try:
        started = time.perf_counter()
        await sandbox.execute("return 1")
        print(f"{'first execute (pool start)':<34} {(time.perf_counter() - started) * 1000:9.2f} ms")
        repeat = 20 if QUICK else 100

        report("execute (return 1)", await timed(lambda: sandbox.execute("return 1"), repeat))

        def start_run() -> Awaitable[Any]:
            return sandbox.start("return 1", started_at=0, seed=0, on_call=lambda _call: None).result

        report("start (deterministic, return 1)", await timed(start_run, repeat))

        calls = 200 if QUICK else 1000
        loop_code = f"for i in range({calls}):\n    await tools.echo(i)\nreturn 1"
        baseline = statistics.median(await timed(lambda: sandbox.execute("return 1"), 10))
        samples = await timed(lambda: sandbox.execute(loop_code), 3 if QUICK else 5)
        report("tool call round trip", [sample - baseline for sample in samples], per=calls, unit="us")

        for count in (10, 100) if QUICK else (10, 100, 1000):
            code = f"import asyncio\nreturn len(await asyncio.gather(*[tools.sleep(i) for i in range({count})]))"
            report(
                f"gather {count} x 10 ms sleeps",
                await timed(lambda code=code: sandbox.execute(code), 3 if QUICK else 5),
            )

        items = 10_000 if QUICK else 50_000
        output_code = f"for i in range({items}):\n    text('line ' + str(i))"
        samples = await timed(lambda: sandbox.execute(output_code), 3)
        report("text() item", [sample - baseline for sample in samples], per=items, unit="us")
    finally:
        await sandbox.close()


if __name__ == "__main__":
    asyncio.run(main())
