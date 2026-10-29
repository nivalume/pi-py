"""Hot paths of pi_codemode's two sandboxes: per-execution overhead, tool-call round trips, gathered calls.

Each case runs on ``CodemodeSandbox`` (JavaScript in QuickJS on wasmtime) and ``MontyCodemodeSandbox`` (Python in a
Monty worker pool), when their extras are installed:

- ``execute``: a script that only returns, end to end through ``execute()`` (a fresh VM per run for QuickJS, a pool
  checkout for Monty), warm.
- ``start``: the same through a deterministic ``start()`` run.
- ``round trip``: one awaited tool call, measured over a loop of sequential calls in one script.
- ``gather N``: N concurrent calls of a tool that sleeps 10 ms (``Promise.all``/``asyncio.gather``), which shows the
  calls run concurrently on the host (wall time stays near 10 ms plus per-call overhead).
- ``output``: ``text()`` throughput.

Run: python benchmarks/bench_codemode.py [--quick] [javascript|python]
"""

from __future__ import annotations

import asyncio
import importlib.util
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pi_codemode import CodemodeSandboxBase, CodemodeSuccess, CodemodeTool

QUICK = "--quick" in sys.argv

#: The scripts of each case, by language.
SCRIPTS = {
    "javascript": {
        "loop": "for (let i = 0; i < {n}; i++) await tools.echo(i);\nreturn 1",
        "gather": "return (await Promise.all(Array.from({{ length: {n} }}, (_, i) => tools.sleep(i)))).length",
        "output": "for (let i = 0; i < {n}; i++) text('line ' + i);",
    },
    "python": {
        "loop": "for i in range({n}):\n    await tools.echo(i)\nreturn 1",
        "gather": "import asyncio\nreturn len(await asyncio.gather(*[tools.sleep(i) for i in range({n})]))",
        "output": "for i in range({n}):\n    text('line ' + str(i))",
    },
}


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
        print(f"  {name:<32} median {median * 1000:9.1f} us   best {best * 1000:9.1f} us")
    else:
        print(f"  {name:<32} median {median:9.2f} ms   best {best:9.2f} ms")


def sandbox_of(language: str, tools: list[CodemodeTool]) -> CodemodeSandboxBase | None:
    if language == "javascript":
        if importlib.util.find_spec("wasmtime") is None:
            return None
        from pi_codemode import CodemodeSandbox

        return CodemodeSandbox(tools=tools, timeout_ms=120_000)
    if importlib.util.find_spec("pydantic_monty") is None:
        return None
    from pi_codemode import MontyCodemodeSandbox

    return MontyCodemodeSandbox(tools=tools, timeout_ms=120_000)


async def bench(language: str) -> None:
    async def sleep_tool(args: Any, _context: Any) -> Any:
        await asyncio.sleep(0.01)
        return args

    sandbox = sandbox_of(
        language, [CodemodeTool("echo", lambda args, _context: args), CodemodeTool("sleep", sleep_tool)]
    )
    if sandbox is None:
        print(f"{language}: skipped, its sandbox is not installed")
        return
    scripts = SCRIPTS[language]
    print(f"{language} ({type(sandbox).__name__})")
    try:
        started = time.perf_counter()
        await sandbox.execute("return 1")
        print(f"  {'first execute (runtime start)':<32} {(time.perf_counter() - started) * 1000:9.2f} ms")
        repeat = 20 if QUICK else 100

        report("execute (return 1)", await timed(lambda: sandbox.execute("return 1"), repeat))

        def start_run() -> Awaitable[Any]:
            return sandbox.start("return 1", started_at=0, seed=0, on_call=lambda _call: None).result

        report("start (deterministic, return 1)", await timed(start_run, repeat))

        calls = 200 if QUICK else 1000
        loop_code = scripts["loop"].format(n=calls)
        baseline = statistics.median(await timed(lambda: sandbox.execute("return 1"), 10))
        samples = await timed(lambda: sandbox.execute(loop_code), 3 if QUICK else 5)
        report("tool call round trip", [sample - baseline for sample in samples], per=calls, unit="us")

        for count in (10, 100) if QUICK else (10, 100, 1000):
            code = scripts["gather"].format(n=count)
            report(
                f"gather {count} x 10 ms sleeps",
                await timed(lambda code=code: sandbox.execute(code), 3 if QUICK else 5),
            )

        items = 10_000 if QUICK else 50_000
        output_code = scripts["output"].format(n=items)
        samples = await timed(lambda: sandbox.execute(output_code), 3)
        report("text() item", [sample - baseline for sample in samples], per=items, unit="us")
    finally:
        await sandbox.close()


async def main() -> None:
    languages = [language for language in SCRIPTS if language in sys.argv] or list(SCRIPTS)
    for language in languages:
        await bench(language)


if __name__ == "__main__":
    asyncio.run(main())
