"""Monty-pool equivalents of upstream ``test/worker-lifetime.test.ts``.

Upstream's tests cover Node/Bun worker threads: admission hooks (``acquireWorker``), exit accounting
(``onWorkerStarted``), ``terminate()``/``unref()`` ordering, and prompt process exit. Here each
execution checks a session (a ``monty`` worker subprocess) out of an ``AsyncMonty`` pool, so the
equivalents are: a pool that cannot start is a sandbox error and starts nothing; aborts, timeouts and
``close()`` settle promptly without waiting for the worker; the session goes back to the pool after
every ``execute()`` and ``start()``; ``max_processes`` bounds concurrency; a crashed worker fails only
its own execution; a shared pool outlives the sandboxes that use it; and the host process exits
promptly after a cancelled run.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Literal

import pytest

from pi_ai.abort import AbortController
from pi_codemode import CodemodeFailure, CodemodeSuccess, CodemodeTextOutput, CodemodeTool, MontyCodemodeSandbox


def monty_workers() -> list[int]:
    """Worker processes of this process's pools (Linux)."""
    pids: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as file:
                stat = file.read()
        except OSError:
            continue
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        parent = int(stat[stat.rindex(")") + 2 :].split()[1])
        if parent == os.getpid() and "monty" in name:
            pids.append(int(entry))
    return pids


linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")


async def test_reports_a_pool_that_cannot_start_as_a_sandbox_error() -> None:
    sandbox = MontyCodemodeSandbox(binary_path="/nonexistent/monty")
    try:
        result = await sandbox.execute("return 1")
        assert isinstance(result, CodemodeFailure)
        assert result.error.kind == "sandbox"
        assert result.error.message.startswith("Failed to start Monty:")
    finally:
        await sandbox.close()


@pytest.mark.parametrize("how", ["abort", "timeout", "close"])
@pytest.mark.parametrize("wait", ["tool", "spin"])
async def test_settles_without_waiting_for_the_worker(how: Literal["abort", "timeout", "close"], wait: str) -> None:
    tool_signal: list[Any] = []
    called = asyncio.Event()

    async def hang(_args: Any, context: Any) -> Any:
        tool_signal.append(context.signal)
        called.set()
        await asyncio.Event().wait()

    sandbox = MontyCodemodeSandbox(timeout_ms=300, tools=[CodemodeTool("hang", hang)])
    controller = AbortController()
    code = 'text("partial")\nawait tools.hang()' if wait == "tool" else 'text("partial")\nwhile True:\n    pass'
    pending = asyncio.ensure_future(sandbox.execute(code, signal=controller.signal))
    closing: asyncio.Future[None] | None = None
    try:
        if wait == "tool":
            await called.wait()
        else:
            await asyncio.sleep(0.15)
        triggered = time.perf_counter()
        if how == "abort":
            controller.abort(RuntimeError("cancelled"))
        if how == "close":
            closing = asyncio.ensure_future(sandbox.close())
        result = await pending
        assert time.perf_counter() - triggered < 1.0
        assert isinstance(result, CodemodeFailure)
        assert result.error.kind == ("timeout" if how == "timeout" else "aborted")
        # Output streamed before the end is kept, also while the worker was computing.
        assert result.output == [CodemodeTextOutput("partial")]
        if wait == "tool":
            assert [(call.name, call.status) for call in result.calls] == [("hang", "cancelled")]
            assert tool_signal[0].aborted is True
        # Neither close() nor anything the worker reports later replaces the decided outcome.
        await sandbox.close()
        assert await pending is result
    finally:
        if closing is not None:
            await closing
        await sandbox.close()


async def test_returns_the_session_to_the_pool_after_execute_and_start() -> None:
    # One worker: a session that were not released would block the next checkout.
    sandbox = MontyCodemodeSandbox(max_processes=1)
    try:
        for index in range(3):
            assert isinstance(await sandbox.execute(f"return {index}"), CodemodeSuccess)
            run = sandbox.start(f"return {index}", started_at=0, seed=0, on_call=lambda _call: None)
            result = await asyncio.wait_for(run.result, 10)
            assert isinstance(result, CodemodeSuccess) and result.value == index
            assert run.finished is True
        # exit() leaves its feed suspended; the session is discarded and the pool replaces its worker.
        assert isinstance(await asyncio.wait_for(sandbox.execute("exit()"), 10), CodemodeSuccess)
        assert isinstance(await asyncio.wait_for(sandbox.execute("return 1"), 10), CodemodeSuccess)
    finally:
        await sandbox.close()


async def test_max_processes_bounds_concurrent_executions() -> None:
    events: list[str] = []

    async def step(args: Any, _context: Any) -> Any:
        events.append(f"start {args}")
        await asyncio.sleep(0.05)
        events.append(f"end {args}")
        return args

    sandbox = MontyCodemodeSandbox(max_processes=1, tools=[CodemodeTool("step", step)])
    try:
        results = await asyncio.gather(
            sandbox.execute("return await tools.step('a')"), sandbox.execute("return await tools.step('b')")
        )
        assert [result.ok for result in results] == [True, True]
        assert events in (["start a", "end a", "start b", "end b"], ["start b", "end b", "start a", "end a"])
    finally:
        await sandbox.close()


@linux_only
async def test_a_crashed_worker_fails_only_its_own_execution() -> None:
    sandbox = MontyCodemodeSandbox(max_processes=2, timeout_ms=20_000)

    def kill_workers() -> None:
        for pid in monty_workers():
            os.kill(pid, signal.SIGKILL)

    async def crash(_args: Any, _context: Any) -> Any:
        kill_workers()
        await asyncio.sleep(0.05)
        return "late"

    sandbox.register_tool(CodemodeTool("crash", crash))
    try:
        # Killed while it waits on a tool: noticed when the result is handed to the dead worker.
        waiting = await sandbox.execute("text('before')\nreturn await tools.crash()")
        assert isinstance(waiting, CodemodeFailure)
        assert waiting.error.kind == "sandbox"
        assert waiting.error.message.startswith("The sandbox worker crashed")
        assert waiting.output == [CodemodeTextOutput("before")]
        # Killed while it computes: noticed at once.
        spinning = asyncio.ensure_future(sandbox.execute("while True:\n    pass"))
        await asyncio.sleep(0.2)
        kill_workers()
        crashed = await asyncio.wait_for(spinning, 10)
        assert isinstance(crashed, CodemodeFailure) and crashed.error.kind == "sandbox"
        # The pool replaces the workers; the host is unharmed.
        assert isinstance(await sandbox.execute("return 2"), CodemodeSuccess)
    finally:
        await sandbox.close()


async def test_a_shared_pool_outlives_the_sandboxes_that_use_it() -> None:
    from pydantic_monty import AsyncMonty

    async with AsyncMonty(max_processes=2) as pool:
        first = MontyCodemodeSandbox(pool=pool)
        second = MontyCodemodeSandbox(pool=pool)
        assert isinstance(await first.execute("return 1"), CodemodeSuccess)
        await first.close()
        result = await second.execute("return 2")
        assert isinstance(result, CodemodeSuccess) and result.value == 2
        await second.close()
        async with pool.checkout() as session:
            assert await session.feed_run("3") == 3


async def test_close_waits_for_sessions_and_closes_its_own_pool() -> None:
    sandbox = MontyCodemodeSandbox()
    running = asyncio.ensure_future(sandbox.execute("while True:\n    pass"))
    await asyncio.sleep(0.1)
    await asyncio.wait_for(sandbox.close(), 10)
    result = await running
    assert isinstance(result, CodemodeFailure) and result.error.message == "Sandbox closed"
    assert sandbox._own_pool is None
    assert sandbox._drivers == set()
    with pytest.raises(RuntimeError, match="closed"):
        await sandbox.execute("return 1")


_PROCESS_EXIT = """
import asyncio, json, sys, time
from pi_ai.abort import AbortController
from pi_codemode import MontyCodemodeSandbox

async def main(how):
    sandbox = MontyCodemodeSandbox(timeout_ms=200 if how == "timeout" else 60_000)
    controller = AbortController()
    started = time.perf_counter()
    pending = asyncio.ensure_future(sandbox.execute("while True:\\n    pass", signal=controller.signal))
    await asyncio.sleep(0.2)
    if how == "abort":
        controller.abort()
    if how == "close":
        await sandbox.close()
    result = await pending
    print(json.dumps({"type": "result", "kind": result.error.kind, "elapsed_ms": (time.perf_counter() - started) * 1000}), flush=True)
    await sandbox.close()

asyncio.run(main(sys.argv[1]))
"""


@pytest.mark.parametrize("how", ["abort", "timeout", "close"])
def test_the_process_exits_promptly_after_a_cancelled_run(how: str) -> None:
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, "-c", _PROCESS_EXIT, how], capture_output=True, text=True, timeout=30)
    elapsed = time.perf_counter() - started
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["type"] == "result"
    assert result["kind"] == ("timeout" if how == "timeout" else "aborted")
    assert result["elapsed_ms"] < 1500
    assert elapsed < 15
