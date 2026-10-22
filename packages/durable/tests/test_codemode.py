"""Port of ``test/codemode.test.ts``: durable code mode, its journal and replay, and its output.

Scripts here are Python run by Monty (``MontyCodemodeSandbox``); ``test_codemode_javascript.py`` runs upstream's
JavaScript cases on QuickJS. Adaptations for Python:

- Every restart case runs on MemoryStorage (reopened as a process would reopen a database) and on SQLite; upstream's
  run on SQLite only. Waits on storage poll in real time (``wait_for``), since SQLite runs on executor threads here.
- Script syntax: ``tools.echo(text="a")`` for ``tools.echo({ text: "a" })``, ``asyncio.gather`` for ``Promise.all``,
  ``time.time()``/``random.random()`` for ``Date.now()``/``Math.random()``, ``print`` for ``console.log``, ``str(e)``
  for ``e.message``, a ``raise Exception("boom")`` failure reported as ``Exception: boom`` (upstream ``Error: boom``),
  ``None``/``null`` for ``undefined``.
- "rejects a failed tool call with its error text and structured output": Monty exceptions carry only a message, so
  the script checks the message; the outcome's ``properties.structuredOutput`` is delivered but not visible to it.
- "times a script out across a restart": a 1000 ms deadline and an 1100 ms pause instead of 300/400 ms, so a Monty
  worker's start does not use up the deadline before the script reaches its first call.
- "drops a global's result that arrives after the script finished": the loop's exception handler (and a garbage
  collection) stands in for Node's ``unhandledRejection``.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import os
import re
import shutil
import tempfile
import textwrap
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pydantic_monty")

from durable_harness_support import (
    ChatSetup,
    aborted,
    add_hooks,
    add_tool,
    all_entries,
    chat_setup,
    open_chat,
    wait_for,
)
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, resolve

from pi_ai import ImageContent, TextContent, Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage
from pi_codemode import CodemodeRun, MontyCodemodeSandbox
from pi_durable import Harness, MemoryStorage, ToolResultEntry, ToolTask, define_tool
from pi_durable.codemode.codemode import (
    CodemodeGlobal,
    CodemodeJournalDoc,
    CodemodeRunDoc,
    create_codemode_tool,
)
from pi_durable.codemode.describe import CodemodeNamespace
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

Parameters = Type.Object({"text": Type.Optional(Type.String()), "id": Type.Optional(Type.Number())})

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
HEADER = re.compile(r"^Script (completed|failed)\nWall time [\d.]+ seconds\nOutput:\n")


def tool(name: str, execute: Any, replay: str = "unsafe") -> Any:
    return define_tool(name=name, description=name, parameters=Parameters, execute=execute, replay=replay)  # pyright: ignore[reportArgumentType]


def js(value: Any) -> str:
    """``String(value)`` of a JSON number or ``undefined``."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


DONE = faux_assistant_message([faux_text("done")])


def script(code: str) -> AssistantMessage:
    return faux_assistant_message(
        [faux_tool_call("codemode", {"code": textwrap.dedent(code).strip("\n")}, id="s1")], stop_reason="toolUse"
    )


_sandboxes: list[MontyCodemodeSandbox] = []
_directories: list[str] = []


@pytest.fixture(autouse=True)
async def _cleanup() -> AsyncIterator[None]:
    yield
    sandboxes = list(_sandboxes)
    _sandboxes.clear()
    await asyncio.gather(*(sandbox.close() for sandbox in sandboxes), return_exceptions=True)
    for directory in _directories:
        shutil.rmtree(directory, ignore_errors=True)
    _directories.clear()


def temp_dir(prefix: str) -> str:
    directory = tempfile.mkdtemp(prefix=prefix)
    _directories.append(directory)
    return directory


def new_sandbox() -> MontyCodemodeSandbox:
    sandbox = MontyCodemodeSandbox(memory_limit_bytes=64 * 1024 * 1024)
    _sandboxes.append(sandbox)
    return sandbox


def sqlite_path() -> Path:
    return Path(temp_dir("pi-durable-codemode-")) / "session.sqlite"


class Backend:
    """A storage a restart reopens: the same MemoryStorage, or the same SQLite file."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._memory: MemoryStorage | None = None
        self._path = sqlite_path() if name == "sqlite" else None

    async def open(self) -> Any:
        if self._path is not None:
            return await open_sqlite_storage(self._path)
        if self._memory is None:
            self._memory = MemoryStorage()
            return self._memory
        return await STORAGE_BACKENDS["memory"].reopen(self._memory)


@pytest.fixture(params=["memory", "sqlite"])
def backend(request: pytest.FixtureRequest) -> Backend:
    return Backend(request.param)


@dataclass
class Vm:
    starts: int = 0
    on_start: Callable[[CodemodeRun], None] | None = None


@dataclass
class Holds:
    reached: asyncio.Future[Any]
    block: bool = True
    runs: int = 0


@dataclass
class Slows:
    gates: dict[int, asyncio.Future[Any]] = field(default_factory=lambda: {})
    done: dict[int, asyncio.Future[Any]] = field(default_factory=lambda: {})
    finished: list[int] = field(default_factory=lambda: [])

    def _once(self, futures: dict[int, asyncio.Future[Any]], id: int) -> asyncio.Future[Any]:
        if id not in futures:
            futures[id] = deferred()
        return futures[id]

    def gate(self, id: int) -> asyncio.Future[Any]:
        return self._once(self.gates, id)

    def release(self, id: int) -> None:
        resolve(self.gate(id))

    def finished_call(self, id: int) -> asyncio.Future[Any]:
        """Resolves once call ``id`` has finished."""
        return self._once(self.done, id)


@dataclass
class Chat:
    setup: ChatSetup
    sandbox: MontyCodemodeSandbox
    vm: Vm
    echo: dict[str, int]
    holds: Holds
    pauses: Holds
    slows: Slows


def codemode_chat(
    globals: list[CodemodeGlobal] | None = None, namespaces: list[CodemodeNamespace] | None = None
) -> Chat:
    """A chat with the code mode tool and test tools:

    - ``echo`` returns ``echo <text>`` and counts its runs by text;
    - ``hold`` (replay-safe) waits for its abort while ``holds.block`` is set, then returns ``held <id>``;
    - ``slow`` returns ``slow <id>`` once ``slows.release(id)`` was called.
    """
    setup = chat_setup()
    sandbox = new_sandbox()
    vm = Vm()
    start = sandbox.start

    def counting_start(code: str, **options: Any) -> CodemodeRun:
        vm.starts += 1
        run = start(code, **options)
        if vm.on_start is not None:
            vm.on_start(run)
        return run

    sandbox.start = counting_start  # type: ignore[method-assign]
    add_tool(setup.registry, create_codemode_tool(sandbox=sandbox, globals=globals or [], namespaces=namespaces or []))
    echo: dict[str, int] = {}

    async def echo_execute(args: Any, _api: Any, _context: Any) -> Any:
        text = args.get("text", "")
        echo[text] = echo.get(text, 0) + 1
        return {"output": [TextContent(f"echo {text}")]}

    add_tool(setup.registry, tool("echo", echo_execute))
    holds = Holds(deferred())

    async def hold_execute(args: Any, _api: Any, call_context: Any) -> Any:
        holds.runs += 1
        if holds.block:
            resolve(holds.reached)
            await aborted(call_context.abort_signal)
        return {"output": [TextContent(f"held {js(args.get('id'))}")]}

    add_tool(setup.registry, tool("hold", hold_execute, "safe"))
    pauses = Holds(deferred())

    async def pause_execute(_args: Any, _api: Any, call_context: Any) -> Any:
        if pauses.block:
            resolve(pauses.reached)
            await aborted(call_context.abort_signal)
        return {"output": [TextContent("paused")]}

    add_tool(setup.registry, tool("pause", pause_execute, "safe"))
    slows = Slows()

    async def slow_execute(args: Any, _api: Any, _context: Any) -> Any:
        id = int(args.get("id", 0))
        await slows.gate(id)
        slows.finished.append(id)
        resolve(slows.finished_call(id))
        return {"output": [TextContent(f"slow {id}")]}

    add_tool(setup.registry, tool("slow", slow_execute))
    return Chat(setup, sandbox, vm, echo, holds, pauses, slows)


def tool_tasks(harness: Harness) -> Awaitable[Any]:
    return harness.commit(lambda tx: tx.scan_tasks({"kind": ToolTask.definition.name}, 100), context)


async def script_task(harness: Harness) -> Any:
    tasks = (await tool_tasks(harness))["items"]
    return next(task for task in tasks if task["input"]["kind"] == "model")["id"]


async def nested_count(harness: Harness) -> int:
    tasks = (await tool_tasks(harness))["items"]
    return len([task for task in tasks if task["input"]["kind"] == "nested"])


@dataclass
class ResultText:
    text: str
    is_error: bool


async def result_text(root: Any) -> ResultText:
    """The code mode call's result text, without the "Script completed" header, which tests of it check themselves."""
    entry = next(candidate for candidate in await all_entries(root) if ToolResultEntry.is_(candidate))
    message = entry["model"][0]
    text = "".join(item.text if isinstance(item, TextContent) else "" for item in message.content)
    return ResultText(HEADER.sub("", text, count=1), message.is_error)


def ok(text: str) -> ResultText:
    return ResultText(text, False)


async def journal_of(harness: Harness, task: Any) -> Any:
    return await harness.snapshot(CodemodeJournalDoc, task, context)


Change = Callable[[Harness, Any], Awaitable[None]]


async def nothing(_harness: Harness, _task: Any) -> None:
    return None


@dataclass
class Restarted:
    harness: Harness
    root: Any
    result: ResultText


async def restart(
    chat: Chat, code: str, backend: Backend, between: Change = nothing, before_close: Change = nothing
) -> Restarted:
    """Run ``code`` until ``holds.reached``, close the Harness (a restart), let ``between`` change the stored state,
    then reopen with ``hold`` no longer blocking and wait for the answer."""
    chat.setup.faux.set_responses([script(code), DONE])
    opened = await open_chat(await backend.open(), chat.setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await chat.holds.reached
    await before_close(opened.harness, await script_task(opened.harness))
    await opened.harness.close(context)
    chat.holds.block = False
    opened = await open_chat(await backend.open(), chat.setup)
    await between(opened.harness, await script_task(opened.harness))
    opened.harness.resume()
    submission = await opened.harness.submission(id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    return Restarted(opened.harness, opened.root, await result_text(opened.root))


def change_run(change: Callable[[Any], None]) -> Change:
    async def between(harness: Harness, task: Any) -> None:
        async def apply(tx: Any) -> None:
            change(await tx.doc(CodemodeRunDoc, task))

        await harness.commit(apply, context)

    return between


def change_journal(change: Callable[[Any], None]) -> Change:
    async def between(harness: Harness, task: Any) -> None:
        async def apply(tx: Any) -> None:
            change(await tx.doc(CodemodeJournalDoc, task))

        await harness.commit(apply, context)

    return between


def set_store(values: dict[str, Any], *, clear_journal: bool = False, forget_first_count: bool = False) -> Change:
    async def between(harness: Harness, task: Any) -> None:
        async def apply(tx: Any) -> None:
            run = await tx.doc(CodemodeRunDoc, task)
            run["start"]["store"] = values
            journal = await tx.doc(CodemodeJournalDoc, task)
            if clear_journal:
                # As if the restart came before the first journal commit: the rerun runs live from the start.
                journal["entries"] = []
            if forget_first_count:
                # As if the live run crashed before recording result 1's count.
                del journal["entries"][0]["issuedAfter"]

        await harness.commit(apply, context)

    return between


def journal_reaches(check: Callable[[Any], bool]) -> Change:
    async def before_close(harness: Harness, task: Any) -> None:
        async def reached() -> bool:
            journal = await journal_of(harness, task)
            return journal is not None and check(journal)

        await wait_for(reached)

    return before_close


async def submit_and_wait(root: Any) -> None:
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)


# ─── durable code mode ──────────────────────────────────────────────────────


async def test_runs_a_script_that_calls_tools_as_nested_calls_and_journals_each_result_before_delivering_it() -> None:
    chat = codemode_chat()
    chat.holds.block = False
    chat.setup.faux.set_responses(
        [script('a = await tools.echo(text="a")\nb = await tools.echo(text="b")\nreturn [a, b]'), DONE]
    )
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    await submit_and_wait(opened.root)
    assert await result_text(opened.root) == ok('["echo a","echo b"]')
    assert await nested_count(opened.harness) == 2
    await opened.harness.close(context)


async def test_resumes_a_script_after_a_restart_without_running_its_finished_calls_again(backend: Backend) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        """
        import time
        a = await tools.echo(text="a")
        held = await tools.hold(id=1)
        b = await tools.echo(text="b")
        return [a, held, b, time.time() > 0]
        """,
        backend,
    )
    assert restarted.result == ok('["echo a","held 1","echo b",true]')
    assert chat.echo == {"a": 1, "b": 1}
    # The sandbox ran twice; hold, replay-safe, reran once after the restart.
    assert chat.vm.starts == 2
    assert chat.holds.runs == 2
    await restarted.harness.close(context)


async def test_replays_results_in_the_order_they_arrived_not_the_order_the_calls_were_made(backend: Backend) -> None:
    chat = codemode_chat()
    code = """
        import asyncio
        order = []
        async def slow(id):
            await tools.slow(id=id)
            order.append(id)
        await asyncio.gather(slow(1), slow(2))
        await tools.echo(text=",".join(str(n) for n in order))
        await tools.hold()
        return order
    """
    # 2 finishes first, then 1.
    chat.slows.release(2)
    chat.slows.finished_call(2).add_done_callback(lambda _future: chat.slows.release(1))
    restarted = await restart(chat, code, backend)
    assert restarted.result == ok("[2,1]")
    assert chat.echo == {"2,1": 1}
    await restarted.harness.close(context)


async def test_journals_globals_with_their_results_and_hands_a_replay_the_recorded_ones(backend: Backend) -> None:
    counter = [0]

    async def next_(_args: Any, _api: Any, _context: Any) -> int:
        counter[0] += 1
        return counter[0]

    chat = codemode_chat([CodemodeGlobal("next", next_)])
    restarted = await restart(chat, "n = await next()\nawait tools.hold()\nreturn [n, await next()]", backend)
    # The first next() ran once and was replayed from the journal; the second ran after the restart.
    assert restarted.result == ok("[1,2]")
    assert counter[0] == 2
    await restarted.harness.close(context)


async def test_keeps_the_scripts_clock_and_randomness_across_a_restart(backend: Backend) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        """
        import random
        import time
        before = [time.time(), random.random()]
        await tools.hold()
        await tools.echo(text=str(before[0]) + "/" + str(before[1]))
        return "ok"
        """,
        backend,
    )
    assert restarted.result.text == "ok"
    # One echo: the same arguments, so the rerun admitted no second call.
    assert list(chat.echo.values()) == [1]
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_made_a_different_number_of_calls_than_before(
    backend: Backend,
) -> None:
    chat = codemode_chat()

    def change(journal: Any) -> None:
        journal["entries"][0]["issuedAfter"] = 7

    restarted = await restart(chat, 'await tools.echo(text="a")\nawait tools.hold()', backend, change_journal(change))
    assert restarted.result.is_error
    assert "after result 1 it had made 2 calls, where it had made 7" in restarted.result.text
    await restarted.harness.close(context)


async def test_ends_the_call_at_once_when_a_resumed_script_waits_without_making_a_call_it_made_before(
    backend: Backend,
) -> None:
    chat = codemode_chat()

    def change(journal: Any) -> None:
        journal["entries"][0]["id"] = 99

    restarted = await restart(chat, 'await tools.echo(text="a")\nawait tools.hold()', backend, change_journal(change))
    assert restarted.result.is_error
    assert "it waits without making call 99" in restarted.result.text
    # hold, admitted before the restart, resumed on its own; no call was admitted by the replay.
    assert await nested_count(restarted.harness) == 2
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_calls_a_global_with_other_arguments(backend: Backend) -> None:
    async def one(_args: Any, _api: Any, _context: Any) -> int:
        return 1

    chat = codemode_chat([CodemodeGlobal("next", one)])

    def change(journal: Any) -> None:
        journal["entries"][0]["global"]["args"] = "other"

    restarted = await restart(chat, "await next()\nawait tools.hold()", backend, change_journal(change))
    assert restarted.result.is_error
    assert "call 1 is next" in restarted.result.text
    await restarted.harness.close(context)


async def test_admits_no_call_of_a_diverging_replay_even_one_the_script_never_made_before(backend: Backend) -> None:
    chat = codemode_chat()
    # With k set, the replay makes three extra calls after result 1, which the live run did not make; the count in
    # result 1's acknowledgment shows the divergence only after they were issued.
    restarted = await restart(
        chat,
        """
        await tools.echo(text="a")
        extra = load("k") or 0
        for i in range(extra):
            tools.echo(text="extra" + str(i))
        await tools.hold()
        """,
        backend,
        set_store({"k": 3}),
        journal_reaches(lambda journal: journal["entries"][0].get("issuedAfter") == 2),
    )
    assert restarted.result.is_error
    assert "after result 1 it had made 5 calls, where it had made 2" in restarted.result.text
    assert chat.echo == {"a": 1}
    assert await nested_count(restarted.harness) == 2
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_ends_where_it_went_on_before(backend: Backend) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        """
        await tools.echo(text="a")
        if load("k"):
            return "early"
        await tools.hold()
        """,
        backend,
        set_store({"k": True}),
        # The live run acknowledged result 1 before the restart: it went on after it.
        journal_reaches(lambda journal: journal["entries"][0].get("issuedAfter") == 2),
    )
    assert restarted.result.is_error
    assert "it ended after result 1, where it went on before" in restarted.result.text
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_ends_before_calls_the_live_run_made_even_without_a_recorded_count(
    backend: Backend,
) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        """
        await tools.echo(text="a")
        if load("k"):
            return "early"
        await tools.echo(text="b")
        await tools.hold()
        """,
        backend,
        set_store({"k": True}, forget_first_count=True),
    )
    assert restarted.result.is_error
    assert "it ended after result 1, where it went on before" in restarted.result.text
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_makes_a_call_an_earlier_attempt_admitted_with_other_arguments(
    backend: Backend,
) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        'await tools.echo(text=load("k") or "a")\nawait tools.hold()',
        backend,
        set_store({"k": "other"}, clear_journal=True),
    )
    assert restarted.result.is_error
    assert "different path" in restarted.result.text
    assert "call 1 is tools.echo" in restarted.result.text
    assert chat.echo == {"a": 1}
    await restarted.harness.close(context)


async def test_records_a_count_the_live_run_never_recorded_as_soon_as_a_replay_acknowledges_it(
    backend: Backend,
) -> None:
    chat = codemode_chat()
    chat.setup.faux.set_responses([script('await tools.echo(text="a")\nawait tools.hold()'), DONE])
    opened = await open_chat(await backend.open(), chat.setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await chat.holds.reached
    await opened.harness.close(context)
    # As if result 1's count was never recorded; hold blocks again after the restart.
    chat.holds.reached = deferred()
    opened = await open_chat(await backend.open(), chat.setup)
    task = await script_task(opened.harness)

    async def forget(tx: Any) -> None:
        del (await tx.doc(CodemodeJournalDoc, task))["entries"][0]["issuedAfter"]

    await opened.harness.commit(forget, context)
    opened.harness.resume()
    # The replay acknowledges result 1 and goes live; the script waits on hold, and the count is already recorded.
    await chat.holds.reached
    harness = opened.harness

    async def recorded() -> bool:
        journal = await journal_of(harness, task)
        return journal is not None and journal["entries"][0].get("issuedAfter") == 2

    await wait_for(recorded)
    await opened.harness.close(context)
    chat.holds.block = False
    opened = await open_chat(await backend.open(), chat.setup)
    opened.harness.resume()
    submission = await opened.harness.submission(id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    await opened.harness.close(context)


async def test_lists_the_scripts_tool_calls_in_its_results_details() -> None:
    chat = codemode_chat()

    async def fails(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("permission denied")], "isError": True}

    add_tool(chat.setup.registry, tool("fails", fails))
    chat.setup.faux.set_responses(
        [
            script(
                """
                await tools.echo(text="a")
                try:
                    await tools.fails({})
                except Exception:
                    pass
                tools.hold(id=9)
                return 1
                """
            ),
            DONE,
        ]
    )
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    await submit_and_wait(opened.root)
    entry = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    calls = entry["model"][0].details["calls"]
    # A call still running, or not yet launched, when the script ended shows as cancelled.
    assert [{key: call[key] for key in ("id", "name", "args", "status")} for call in calls] == [
        {"id": 1, "name": "echo", "args": '{"text":"a"}', "status": "ok"},
        {"id": 2, "name": "fails", "args": "{}", "status": "error"},
        {"id": 3, "name": "hold", "args": '{"id":9}', "status": "cancelled"},
    ]
    assert calls[1]["error"] == "permission denied"
    assert calls[0]["durationMs"] >= 0
    await opened.harness.close(context)


async def test_stores_no_outcome_when_the_host_closes_the_sandbox_and_replays_the_script_after_a_restart(
    backend: Backend,
) -> None:
    chat = codemode_chat()
    chat.setup.faux.set_responses([script('await tools.echo(text="a")\nawait tools.hold()\nreturn "done"'), DONE])
    opened = await open_chat(await backend.open(), chat.setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await chat.holds.reached
    task = await script_task(opened.harness)
    # A host shutting down closes the sandbox before the Harness.
    await chat.sandbox.close()
    await asyncio.sleep(0.1)
    assert "result" not in ((await opened.harness.snapshot(CodemodeRunDoc, task, context)) or {})
    await opened.harness.close(context)
    # The next process has a new sandbox.
    chat.holds.block = False
    add_tool(chat.setup.registry, create_codemode_tool(sandbox=new_sandbox()))
    opened = await open_chat(await backend.open(), chat.setup)
    opened.harness.resume()
    submission = await opened.harness.submission(id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    assert await result_text(opened.root) == ok("done")
    assert chat.echo == {"a": 1}
    await opened.harness.close(context)


async def test_aborts_a_call_still_running_when_the_script_ends() -> None:
    chat = codemode_chat()
    chat.setup.faux.set_responses([script('tools.hold()\nawait tools.echo(text="a")\nreturn 1'), DONE])
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    await submit_and_wait(opened.root)
    assert await result_text(opened.root) == ok("1")
    tasks = (await tool_tasks(opened.harness))["items"]
    hold = next(task for task in tasks if task["input"]["kind"] == "nested" and task["input"]["call"]["name"] == "hold")
    assert chat.holds.runs == 1
    assert hold["state"]["status"] == "terminal"
    assert hold["state"]["outcome"]["status"] == "aborted"
    await opened.harness.close(context)


async def test_admits_no_call_once_the_script_has_ended_even_one_it_made_before_ending() -> None:
    chat = codemode_chat()
    chat.holds.block = False
    storage = ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])

    # Hold the fingerprint commit, and so the admission of the script's call, until the script has ended.
    def on_start(run: CodemodeRun) -> None:
        held = storage.hold_commits()
        run.result.add_done_callback(lambda _result: held.release())

    chat.vm.on_start = on_start
    chat.setup.faux.set_responses([script('tools.echo(text="late")\nreturn 1'), DONE])
    opened = await open_chat(storage, chat.setup)  # pyright: ignore[reportArgumentType]
    await submit_and_wait(opened.root)
    assert await result_text(opened.root) == ok("1")
    assert len(chat.echo) == 0
    assert await nested_count(opened.harness) == 0
    await opened.harness.close(context)


async def test_checks_a_reruns_calls_against_an_earlier_attempts_before_admitting_any_later_call(
    backend: Backend,
) -> None:
    chat = codemode_chat()
    # With k set, call 1 has other arguments and call 2 is another tool than before; call 3 would be new.
    restarted = await restart(
        chat,
        """
        k = load("k")
        tools.echo(text=k or "a")
        if k:
            tools.echo(text="new")
        await tools.hold()
        """,
        backend,
        set_store({"k": "x"}, clear_journal=True),
    )
    assert restarted.result.is_error
    # The first run already made more calls than before.
    assert "its first run made 3 calls, where it had made 2" in restarted.result.text
    assert await nested_count(restarted.harness) == 2
    assert chat.echo == {"a": 1}
    await restarted.harness.close(context)


async def _one(_args: Any, _api: Any, _context: Any) -> int:
    return 1


async def test_ends_the_call_when_a_rerun_makes_a_global_call_where_an_earlier_attempt_made_a_tool_call(
    backend: Backend,
) -> None:
    chat = codemode_chat([CodemodeGlobal("pick", _one)])
    restarted = await restart(
        chat,
        """
        if load("k"):
            await pick()
            await tools.echo(text="new")
        else:
            await tools.hold({})
        """,
        backend,
        set_store({"k": True}),
    )
    assert restarted.result.is_error
    assert "call 1 is pick(), but was tools.hold({})" in restarted.result.text
    assert len(chat.echo) == 0
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_reruns_first_run_makes_fewer_calls_than_an_earlier_attempt_admitted(
    backend: Backend,
) -> None:
    # Live, the first run admits hold 1 and hold 2. With k set, it makes only hold 1, and after its result a global
    # and a new tool call: the global's ID is hold 2's key, which no check after the release would see.
    chat = codemode_chat([CodemodeGlobal("pick", _one)])
    restarted = await restart(
        chat,
        """
        k = load("k")
        first = tools.hold(id=1)
        if not k:
            tools.hold(id=2)
        await first
        await pick()
        await tools.echo(text="new")
        """,
        backend,
        set_store({"k": True}),
        journal_reaches(lambda journal: journal.get("firstIssued") == 2),
    )
    assert restarted.result.is_error
    assert "its first run made 1 calls, where it had made 2" in restarted.result.text
    assert len(chat.echo) == 0
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_rerun_without_journal_entries_ends_in_its_first_run_where_it_went_on_before(
    backend: Backend,
) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        """
        if load("k"):
            store("x", 1)
            return "early"
        await tools.hold()
        """,
        backend,
        set_store({"k": True}),
        journal_reaches(lambda journal: journal.get("firstIssued") == 1),
    )
    assert restarted.result.is_error
    assert "it ended after result 0, where it went on before" in restarted.result.text
    await restarted.harness.close(context)


async def test_checks_every_call_a_rerun_holds_before_launching_any_of_them(backend: Backend) -> None:
    # Live, call 1 is a global that never finishes and call 2 is hold; with k set, call 1 is a new tool call and call
    # 2 holds with other arguments. Checking call 1 alone would launch it before call 2's conflict shows.
    async def forever(_args: Any, _api: Any, call_context: Any) -> None:
        await aborted(call_context.abort_signal)

    chat = codemode_chat([CodemodeGlobal("forever", forever)])
    restarted = await restart(
        chat,
        """
        k = load("k")
        if k:
            tools.echo(text="new")
        else:
            forever()
        await tools.hold(id=2 if k else 1)
        """,
        backend,
        set_store({"k": True}),
    )
    assert restarted.result.is_error
    assert 'call 2 is tools.hold({"id":2}), but was tools.hold({"id":1})' in restarted.result.text
    assert len(chat.echo) == 0
    assert await nested_count(restarted.harness) == 1
    await restarted.harness.close(context)


async def test_keeps_a_globals_none_result_none_live_and_replayed(backend: Backend) -> None:
    async def none(_args: Any, _api: Any, _context: Any) -> None:
        return None

    chat = codemode_chat([CodemodeGlobal("nothing", none)])
    restarted = await restart(
        chat,
        "before = await nothing()\nawait tools.hold()\nreturn [before is None, (await nothing()) is None]",
        backend,
    )
    assert restarted.result == ok("[true,true]")
    await restarted.harness.close(context)


async def test_drops_a_globals_result_that_arrives_after_the_script_finished() -> None:
    late = deferred()

    async def slowly(_args: Any, _api: Any, _context: Any) -> Any:
        return await late

    async def quickly(_args: Any, _api: Any, _context: Any) -> Any:
        return "quick"

    chat = codemode_chat([CodemodeGlobal("slowly", slowly), CodemodeGlobal("quickly", quickly)])
    chat.setup.faux.set_responses([script("pending = slowly()\nawait quickly()\nreturn 1"), DONE])
    loop = asyncio.get_running_loop()
    errors: list[Any] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, error: errors.append(error))
    try:
        opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
        await submit_and_wait(opened.root)
        assert await result_text(opened.root) == ok("1")
        # The script's tool call has ended; its pending global resolves now.
        resolve(late, "late")
        await asyncio.sleep(0.02)
        gc.collect()
        await asyncio.sleep(0)
        assert errors == []
        await opened.harness.close(context)
    finally:
        loop.set_exception_handler(previous)


async def test_raises_a_failed_tool_call_with_its_error_text() -> None:
    chat = codemode_chat()

    async def fails(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("permission denied")], "isError": True}

    add_tool(chat.setup.registry, tool("fails", fails))
    chat.holds.block = False
    chat.setup.faux.set_responses(
        [
            script(
                """
                try:
                    await tools.fails({})
                except Exception as error:
                    return [str(error), type(error).__name__]
                """
            ),
            DONE,
        ]
    )
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    await submit_and_wait(opened.root)
    assert await result_text(opened.root) == ok('["permission denied","Exception"]')
    await opened.harness.close(context)


async def test_resolves_a_schema_tools_error_result_that_carries_its_value_and_raises_for_one_without() -> None:
    chat = codemode_chat()
    exit_schema = Type.Object({"output": Type.String(), "exitCode": Type.Number()})

    async def run(_args: Any, _api: Any, _context: Any) -> Any:
        # Like bash on a nonzero exit: an error result with its value.
        return {
            "output": [TextContent("no matches")],
            "structuredOutput": {"output": "no matches", "exitCode": 1},
            "isError": True,
        }

    async def denied(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("permission denied")], "isError": True}

    for name, execute in (("run", run), ("denied", denied)):
        add_tool(
            chat.setup.registry,
            define_tool(
                name=name,
                description=name,
                parameters=Type.Object({}),
                structured_output_schema=exit_schema,
                execute=execute,
            ),
        )
    chat.holds.block = False
    chat.setup.faux.set_responses(
        [
            script(
                """
                r = await tools.run({})
                e = None
                try:
                    await tools.denied({})
                except Exception as error:
                    e = [str(error), None]
                return [r["exitCode"], r["output"], e]
                """
            ),
            DONE,
        ]
    )
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    await submit_and_wait(opened.root)
    assert await result_text(opened.root) == ok('[1,"no matches",["permission denied",null]]')
    await opened.harness.close(context)


async def test_resolves_a_schema_less_tool_with_its_text_as_its_declaration_says_also_when_it_returned_images() -> None:
    chat = codemode_chat()

    async def picture(args: Any, _api: Any, _context: Any) -> Any:
        output: list[Any] = [TextContent("a cat")] if args["caption"] else []
        return {"output": [*output, ImageContent(PNG, "image/png")]}

    add_tool(
        chat.setup.registry,
        define_tool(
            name="picture",
            description="picture",
            parameters=Type.Object({"caption": Type.Boolean()}),
            execute=picture,
        ),
    )
    chat.holds.block = False
    chat.setup.faux.set_responses(
        [script("return [await tools.picture(caption=True), await tools.picture(caption=False)]"), DONE]
    )
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    await submit_and_wait(opened.root)
    assert await result_text(opened.root) == ok('["a cat",""]')
    await opened.harness.close(context)


async def test_delivers_a_result_only_after_its_journal_commit() -> None:
    storage = ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])
    seen: list[str] = []

    async def first(_args: Any, _api: Any, _context: Any) -> int:
        seen.append("first")
        # The next commit is this result's journal commit: it never lands.
        storage.fail_next_commit(RuntimeError("lost"))
        return 1

    async def second(_args: Any, _api: Any, _context: Any) -> int:
        seen.append("second")
        return 2

    chat = codemode_chat([CodemodeGlobal("first", first), CodemodeGlobal("second", second)])
    chat.holds.block = False
    chat.setup.faux.set_responses([script("await first()\nawait second()"), DONE])
    opened = await open_chat(storage, chat.setup)  # pyright: ignore[reportArgumentType]
    await opened.root.submit({"type": "input", "content": "go"}, context)
    await wait_for(lambda: "first" in seen)
    await asyncio.sleep(0.1)
    assert seen == ["first"]
    try:
        await opened.harness.close(context)
    except Exception:
        pass


@pytest.mark.parametrize("field", ["fingerprint", "version"])
async def test_refuses_to_resume_when_the_sandboxs_fingerprint_or_the_code_mode_version_changed(
    backend: Backend, field: str
) -> None:
    chat = codemode_chat()

    def change(run: Any) -> None:
        if field == "fingerprint":
            run["start"]["fingerprint"] = "another sandbox"
        else:
            run["start"]["version"] = 0

    restarted = await restart(chat, 'await tools.hold()\nawait tools.echo(text="late")', backend, change_run(change))
    assert restarted.result.is_error
    assert "cannot resume" in restarted.result.text
    assert len(chat.echo) == 0
    await restarted.harness.close(context)


async def test_times_a_script_out_across_a_restart_without_admitting_new_calls(backend: Backend) -> None:
    chat = codemode_chat()

    async def pause(_harness: Harness, _task: Any) -> None:
        await asyncio.sleep(1.1)

    restarted = await restart(
        chat,
        '# @options: {"timeout_ms": 1000}\nawait tools.hold()\nawait tools.echo(text="late")',
        backend,
        pause,
    )
    assert restarted.result.is_error
    assert "deadline passed" in restarted.result.text
    assert len(chat.echo) == 0
    assert chat.vm.starts == 1
    await restarted.harness.close(context)


async def test_returns_a_decided_outcome_after_a_restart_without_running_the_script_again(backend: Backend) -> None:
    chat = codemode_chat()
    chat.holds.block = False
    chat.setup.faux.set_responses([script('return "first"'), DONE])
    # Stop after execute() returned, so after the outcome is stored, and before the result commits.
    stopped = Holds(deferred())

    async def after_tool(call: Any, _result: Any, _api: Any, hook_context: Any) -> None:
        if call.name != "codemode" or not stopped.block:
            return None
        resolve(stopped.reached)
        await aborted(hook_context.abort_signal)

    add_hooks(chat.setup.registry, ToolTask, {"after_tool": after_tool})
    opened = await open_chat(await backend.open(), chat.setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await stopped.reached
    task = await script_task(opened.harness)
    run = await opened.harness.snapshot(CodemodeRunDoc, task, context)
    assert run is not None and "result" in run
    await opened.harness.close(context)
    stopped.block = False
    opened = await open_chat(await backend.open(), chat.setup)
    opened.harness.resume()
    submission = await opened.harness.submission(id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    assert await result_text(opened.root) == ok("first")
    assert chat.vm.starts == 1
    await opened.harness.close(context)


async def test_keeps_store_values_per_conversation_as_of_a_forks_entry() -> None:
    chat = codemode_chat()
    chat.holds.block = False
    chat.setup.faux.set_responses([script('store("k", 1)\nreturn load("k")'), DONE, script('return [load("k")]'), DONE])
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    root = opened.root
    await (await root.submit({"type": "input", "content": "first"}, context)).wait(context)
    first_user = next(entry for entry in await all_entries(root) if entry["kind"] == "pi.user")
    await (await root.submit({"type": "input", "content": "second"}, context)).wait(context)
    results = [entry for entry in await all_entries(root) if ToolResultEntry.is_(entry)]
    assert [entry["model"][0].content[1].text for entry in results] == ["1", "[1]"]
    # A fork at the first input is from before the first script stored anything.
    chat.setup.faux.set_responses([script('return [load("k")]'), DONE])
    fork = await root.fork(first_user["id"], {"ownership": {"kind": "ownerless"}}, context)
    await (await fork.submit({"type": "input", "content": "fork"}, context)).wait(context)
    fork_result = [entry for entry in await all_entries(fork) if ToolResultEntry.is_(entry)][-1]
    assert fork_result["model"][0].content[1].text == "[null]"
    await opened.harness.close(context)


async def test_journals_results_that_finish_together_in_one_commit() -> None:
    release = deferred()
    waiting = [0]

    async def wait(_args: Any, _api: Any, _context: Any) -> None:
        waiting[0] += 1
        if waiting[0] == 50:
            resolve(release)
        await release

    chat = codemode_chat([CodemodeGlobal("wait", wait)])
    chat.holds.block = False
    chat.setup.faux.set_responses(
        [script("import asyncio\nreturn len(await asyncio.gather(*[wait() for _ in range(50)]))"), DONE]
    )
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    journal_commits = [0]

    def listener(publication: Any, _context: Any) -> None:
        for change in publication["changes"]:
            if change["type"] == "document" and change["record"]["kind"] == "pi.codemode.journal":
                journal_commits[0] += 1

    opened.harness.subscribe_commits(listener)
    await submit_and_wait(opened.root)
    assert await result_text(opened.root) == ok("50")
    # The journal's creation and a few groups, not one commit per result (51).
    assert journal_commits[0] < 10
    await opened.harness.close(context)


# ─── output and tool search ─────────────────────────────────────────────────


@dataclass
class ScriptRun:
    text: str
    message: Any


async def run_script(code: str, chat: Chat | None = None) -> ScriptRun:
    """Run ``code`` once in a chat with a local environment; the result's content and details."""
    chat = chat or codemode_chat()
    chat.holds.block = False
    cwd = temp_dir("pi-codemode-env-")
    chat.setup.faux.set_responses([script(code), DONE])
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup, env=LocalExecutionEnv(cwd=cwd))
    await submit_and_wait(opened.root)
    entry = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    message = entry["model"][0]
    await opened.harness.close(context)
    text = "".join(item.text if isinstance(item, TextContent) else f"<{item.type}>" for item in message.content)
    return ScriptRun(text, message)


def remember_output_dir(path: str | None) -> None:
    if path is not None:
        _directories.append(os.path.dirname(path))


async def test_lays_out_text_items_console_lines_and_the_returned_value_under_a_header() -> None:
    run = await run_script('text("one")\nprint("log", 1)\ntext({"two": 2})\nreturn "three"')
    assert re.match(r"^Script completed\nWall time [\d.]+ seconds\nOutput:\n", run.text)
    assert '==> text 1/3 <==\none\n==> text 2/3 <==\n{"two":2}\n==> text 3/3 <==\nthree' in run.text
    assert "<console_output>\nlog 1\n</console_output>" in run.text


async def test_cuts_text_past_the_token_budget_to_its_start_and_end_with_the_full_text_in_a_file() -> None:
    run = await run_script('# @options: {"max_output_tokens": 20}\ntext("BEGIN" + "x" * 1000 + "END")')
    assert "Warning: truncated output" in run.text
    assert "BEGIN" in run.text
    assert "END" in run.text
    path = run.message.details.get("fullOutputPath")
    remember_output_dir(path)
    assert f"[Full output: {path}" in run.text
    assert Path(path).read_text("utf-8") == "BEGIN" + "x" * 1000 + "END"


async def test_keeps_a_result_inside_the_tools_output_limits_whatever_max_output_tokens_says_spilling_the_rest() -> (
    None
):
    # 6 MiB of text: over the Harness's 4 MiB limit, which would cut it without a spill file.
    big = await run_script('# @options: {"max_output_tokens": 2000000}\nreturn "x" * (6 * 1024 * 1024)')
    assert "Warning: truncated output" in big.text
    assert len(big.text) <= 1024 * 1024 + 1000
    remember_output_dir(big.message.details["fullOutputPath"])
    assert len(Path(big.message.details["fullOutputPath"]).read_text("utf-8")) == 6 * 1024 * 1024
    # 200,000 short lines: within the byte limit, over the 100,000 lines one.
    long = await run_script('# @options: {"max_output_tokens": 2000000}\nreturn chr(10).join(["a"] * 200000)')
    assert "Warning: truncated output" in long.text
    assert len(long.text.split("\n")) < 100_000
    assert long.message.details.get("fullOutputPath") is not None
    remember_output_dir(long.message.details["fullOutputPath"])


async def test_writes_the_results_files_to_the_same_paths_when_it_reruns_after_a_crash() -> None:
    chat = codemode_chat()
    chat.holds.block = False
    cwd = temp_dir("pi-codemode-env-")
    written = deferred()

    # The process dies after the full output was written and before the result was committed.
    class CrashAfterWrite(LocalExecutionEnv):
        async def write_file(self, path: str, content: str | bytes, call_context: Any) -> Any:
            result = await super().write_file(path, content, call_context)
            resolve(written, path)
            await aborted(call_context.abort_signal)
            return result

    path = sqlite_path()
    chat.setup.faux.set_responses([script('# @options: {"max_output_tokens": 20}\nreturn "x" * 1000'), DONE])
    opened = await open_chat(await open_sqlite_storage(path), chat.setup, env=CrashAfterWrite(cwd=cwd))
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    first = await written
    remember_output_dir(first)
    await opened.harness.close(context)
    opened = await open_chat(await open_sqlite_storage(path), chat.setup, env=LocalExecutionEnv(cwd=cwd))
    opened.harness.resume()
    submission = await opened.harness.submission(id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    entry = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    assert entry["model"][0].details.get("fullOutputPath") == first
    assert os.listdir(os.path.dirname(first)) == ["output.txt"]
    await opened.harness.close(context)


async def test_saves_images_to_files_and_names_them_next_to_the_image() -> None:
    run = await run_script(f'image("data:image/png;base64,{PNG}")')
    saved = re.search(r"\[Image saved to (\S+) \(image/png, \d+ bytes\)\]", run.text)
    assert saved is not None
    remember_output_dir(saved.group(1))
    assert base64.b64encode(Path(saved.group(1)).read_bytes()).decode() == PNG
    assert any(isinstance(item, ImageContent) for item in run.message.content)


async def test_calls_a_deferred_tool_the_model_is_not_offered_and_finds_it_in_all_tools() -> None:
    chat = codemode_chat()

    async def later(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("later ran")]}

    add_tool(
        chat.setup.registry,
        define_tool(
            name="later", description="A deferred tool", parameters=Type.Object({}), deferred=True, execute=later
        ),
    )
    run = await run_script('return [await tools.later({}), any(tool["name"] == "later" for tool in ALL_TOOLS)]', chat)
    assert '["later ran",true]' in run.text


async def test_reports_a_failed_script_with_its_error_and_the_calls_already_made() -> None:
    run = await run_script('await tools.echo(text="a")\nraise Exception("boom")')
    assert run.message.is_error
    assert re.match(r"^Script failed\n", run.text)
    assert "Exception: boom" in run.text
    assert "Tool calls made before the failure (they are not undone): echo (ok)" in run.text


async def test_finds_tools_with_search_tools_describe_tool_and_describe_namespace_and_lists_them_in_all_tools() -> None:
    chat = codemode_chat([], [{"name": "github", "description": "GitHub MCP server", "prefix": "mcp__github__"}])

    async def list_issues(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": []}

    add_tool(chat.setup.registry, tool("mcp__github__list_issues", list_issues))
    run = await run_script(
        """
        found = await search_tools("issues")
        in_namespace = await search_tools("echo", namespace="github")
        return {
            "found": [tool["name"] for tool in found],
            "inNamespace": in_namespace,
            "described": "async def echo(" in await describe_tool("echo"),
            "missing": await describe_tool("nope"),
            "namespace": await describe_namespace("github"),
            "all": "mcp__github__list_issues" in [tool["name"] for tool in ALL_TOOLS],
        }
        """,
        chat,
    )
    import json

    assert json.loads(HEADER.sub("", run.text, count=1)) == {
        "found": ["mcp__github__list_issues"],
        "inNamespace": [],
        "described": True,
        "missing": None,
        "namespace": {"name": "github", "description": "GitHub MCP server", "tools": ["mcp__github__list_issues"]},
        "all": True,
    }
