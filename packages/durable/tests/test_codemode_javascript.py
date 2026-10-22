"""Port of ``test/codemode.test.ts`` for JavaScript scripts: durable code mode on QuickJS (``CodemodeSandbox``).

These are upstream's scripts, in upstream's language: ``await tools.echo({ text: "a" })``, ``Promise.all``,
``Date.now()``, ``console.log``, ``searchTools()``. ``test_codemode.py`` covers the same journal and replay logic with
Python scripts on Monty; this file checks what the language changes: script syntax, the camelCase discovery globals
and their messages, error properties scripts can read (``error.structuredOutput``), the console block, and that a
run started on one sandbox does not resume on the other. Restart cases run on MemoryStorage and on SQLite.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import tempfile
import textwrap
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import ChatSetup, aborted, add_tool, all_entries, chat_setup, open_chat, wait_for
from durable_session_support import STORAGE_BACKENDS, context, deferred, resolve

from pi_ai import ImageContent, TextContent, Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage
from pi_codemode import CodemodeRun, CodemodeSandbox, CodemodeSandboxBase
from pi_durable import Harness, MemoryStorage, ToolResultEntry, ToolTask, define_tool
from pi_durable.codemode.codemode import CodemodeGlobal, CodemodeJournalDoc, CodemodeRunDoc, create_codemode_tool
from pi_durable.codemode.describe import CodemodeNamespace
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

Parameters = Type.Object({"text": Type.Optional(Type.String()), "id": Type.Optional(Type.Number())})

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
HEADER = re.compile(r"^Script (completed|failed)\nWall time [\d.]+ seconds\nOutput:\n")
DONE = faux_assistant_message([faux_text("done")])

_sandboxes: list[CodemodeSandboxBase] = []
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


def remember_output_dir(path: str | None) -> None:
    if path is not None:
        _directories.append(os.path.dirname(path))


def new_sandbox() -> CodemodeSandbox:
    sandbox = CodemodeSandbox(memory_limit_bytes=64 * 1024 * 1024)
    _sandboxes.append(sandbox)
    return sandbox


def sqlite_path() -> Path:
    return Path(temp_dir("pi-durable-codemode-js-")) / "session.sqlite"


def tool(name: str, execute: Any, replay: str = "unsafe") -> Any:
    return define_tool(name=name, description=name, parameters=Parameters, execute=execute, replay=replay)  # pyright: ignore[reportArgumentType]


def script(code: str) -> AssistantMessage:
    return faux_assistant_message(
        [faux_tool_call("codemode", {"code": textwrap.dedent(code).strip("\n")}, id="s1")], stop_reason="toolUse"
    )


class Backend:
    """A storage a restart reopens: the same MemoryStorage, or the same SQLite file."""

    def __init__(self, name: str) -> None:
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
class Holds:
    reached: asyncio.Future[Any]
    block: bool = True
    runs: int = 0


@dataclass
class Chat:
    setup: ChatSetup
    sandbox: CodemodeSandboxBase
    starts: list[int]
    echo: dict[str, int]
    holds: Holds
    slows: dict[int, asyncio.Future[Any]]


def codemode_chat(
    globals: list[CodemodeGlobal] | None = None,
    namespaces: list[CodemodeNamespace] | None = None,
    sandbox: CodemodeSandboxBase | None = None,
) -> Chat:
    """A chat with the code mode tool on ``sandbox`` (default: a QuickJS one) and test tools: ``echo`` returns
    ``echo <text>`` and counts its runs; ``hold`` (replay-safe) waits for its abort while ``holds.block`` is set;
    ``slow`` returns ``slow <id>`` once ``slows[id]`` resolves."""
    setup = chat_setup()
    sandbox = sandbox or new_sandbox()
    starts = [0]
    start = sandbox.start

    def counting_start(code: str, **options: Any) -> CodemodeRun:
        starts[0] += 1
        return start(code, **options)

    sandbox.start = counting_start  # type: ignore[method-assign]
    add_tool(setup.registry, create_codemode_tool(sandbox=sandbox, globals=globals or [], namespaces=namespaces or []))
    echo: dict[str, int] = {}

    async def echo_execute(args: Any, _api: Any, _context: Any) -> Any:
        text = args.get("text", "")
        echo[text] = echo.get(text, 0) + 1
        return {"output": [TextContent(f"echo {text}")]}

    holds = Holds(deferred())

    async def hold_execute(args: Any, _api: Any, call_context: Any) -> Any:
        holds.runs += 1
        if holds.block:
            resolve(holds.reached)
            await aborted(call_context.abort_signal)
        held = args.get("id")
        return {"output": [TextContent(f"held {'' if held is None else int(held)}")]}

    slows: dict[int, asyncio.Future[Any]] = {}

    async def slow_execute(args: Any, _api: Any, _context: Any) -> Any:
        id = int(args.get("id", 0))
        await slows.setdefault(id, deferred())
        return {"output": [TextContent(f"slow {id}")]}

    add_tool(setup.registry, tool("echo", echo_execute))
    add_tool(setup.registry, tool("hold", hold_execute, "safe"))
    add_tool(setup.registry, tool("slow", slow_execute))
    return Chat(setup, sandbox, starts, echo, holds, slows)


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
    """The code mode call's result text without the "Script completed" header."""
    entry: Any = next(candidate for candidate in await all_entries(root) if ToolResultEntry.is_(candidate))
    message = entry["model"][0]
    text = "".join(item.text if isinstance(item, TextContent) else "" for item in message.content)
    return ResultText(HEADER.sub("", text, count=1), message.is_error)


def ok(text: str) -> ResultText:
    return ResultText(text, False)


Change = Callable[[Harness, Any], Awaitable[None]]


async def nothing(_harness: Harness, _task: Any) -> None:
    return None


@dataclass
class Restarted:
    harness: Harness
    root: Any
    result: ResultText


async def restart(
    chat: Chat,
    code: str,
    backend: Backend,
    between: Change = nothing,
    before_close: Change = nothing,
    resumed: Chat | None = None,
) -> Restarted:
    """Run ``code`` until ``holds.reached``, close the Harness (a restart), let ``between`` change the stored state,
    then reopen (with ``resumed``'s setup if given) with ``hold`` no longer blocking and wait for the answer."""
    chat.setup.faux.set_responses([script(code), DONE])
    opened = await open_chat(await backend.open(), chat.setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await chat.holds.reached
    await before_close(opened.harness, await script_task(opened.harness))
    await opened.harness.close(context)
    chat.holds.block = False
    after = resumed or chat
    after.holds.block = False
    if resumed is not None:
        # The model's answer to the script's result comes from the resumed process's model.
        resumed.setup.faux.set_responses([DONE])
    opened = await open_chat(await backend.open(), after.setup)
    await between(opened.harness, await script_task(opened.harness))
    opened.harness.resume()
    submission = await opened.harness.submission(id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    return Restarted(opened.harness, opened.root, await result_text(opened.root))


def change_doc(doc: Any, change: Callable[[Any], None]) -> Change:
    async def between(harness: Harness, task: Any) -> None:
        async def apply(tx: Any) -> None:
            change(await tx.doc(doc, task))

        await harness.commit(apply, context)

    return between


def set_store(values: dict[str, Any]) -> Change:
    def change(run: Any) -> None:
        run["start"]["store"] = values

    return change_doc(CodemodeRunDoc, change)


def journal_reaches(check: Callable[[Any], bool]) -> Change:
    async def before_close(harness: Harness, task: Any) -> None:
        async def reached() -> bool:
            journal = await harness.snapshot(CodemodeJournalDoc, task, context)
            return journal is not None and check(journal)

        await wait_for(reached)

    return before_close


@dataclass
class ScriptRun:
    text: str
    message: Any


async def run_script(code: str, chat: Chat | None = None) -> ScriptRun:
    """Run ``code`` once in a chat with a local environment; the result's content (with its header) and message."""
    chat = chat or codemode_chat()
    chat.holds.block = False
    chat.setup.faux.set_responses([script(code), DONE])
    env = LocalExecutionEnv(cwd=temp_dir("pi-codemode-js-env-"))
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup, env=env)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    entry: Any = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    message = entry["model"][0]
    await opened.harness.close(context)
    text = "".join(item.text if isinstance(item, TextContent) else f"<{item.type}>" for item in message.content)
    return ScriptRun(text, message)


def value_of(run: ScriptRun) -> Any:
    return json.loads(HEADER.sub("", run.text, count=1))


# ─── journal and replay ─────────────────────────────────────────────────────


async def test_runs_a_script_that_calls_tools_as_nested_calls() -> None:
    run = await run_script(
        'const a = await tools.echo({ text: "a" });\nconst b = await tools.echo({ text: "b" });\nreturn [a, b];'
    )
    assert value_of(run) == ["echo a", "echo b"]


async def test_resumes_a_script_after_a_restart_without_running_its_finished_calls_again(backend: Backend) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        """
        const a = await tools.echo({ text: "a" });
        const held = await tools.hold({ id: 1 });
        const b = await tools.echo({ text: "b" });
        return [a, held, b, Date.now() > 0];
        """,
        backend,
    )
    assert restarted.result == ok('["echo a","held 1","echo b",true]')
    assert chat.echo == {"a": 1, "b": 1}
    # The sandbox ran twice; hold, replay-safe, reran once after the restart.
    assert chat.starts[0] == 2
    assert chat.holds.runs == 2
    await restarted.harness.close(context)


async def test_replays_results_in_the_order_they_arrived_not_the_order_the_calls_were_made(backend: Backend) -> None:
    chat = codemode_chat()
    first, second = chat.slows.setdefault(1, deferred()), chat.slows.setdefault(2, deferred())
    # 2 finishes first, then 1.
    resolve(second)
    asyncio.get_running_loop().call_later(0.1, lambda: resolve(first))
    restarted = await restart(
        chat,
        """
        const order = [];
        const slow = async (id) => { await tools.slow({ id }); order.push(id); };
        await Promise.all([slow(1), slow(2)]);
        await tools.echo({ text: order.join(",") });
        await tools.hold();
        return order;
        """,
        backend,
    )
    assert restarted.result == ok("[2,1]")
    assert chat.echo == {"2,1": 1}
    await restarted.harness.close(context)


async def test_journals_globals_and_keeps_the_clock_and_randomness_across_a_restart(backend: Backend) -> None:
    counter = [0]

    async def next_(_args: Any, _api: Any, _context: Any) -> int:
        counter[0] += 1
        return counter[0]

    chat = codemode_chat([CodemodeGlobal("next", next_)])
    restarted = await restart(
        chat,
        """
        const n = await next();
        const before = [Date.now(), Math.random(), new Date().getTimezoneOffset()];
        await tools.hold();
        await tools.echo({ text: before.join("/") });
        return [n, await next(), typeof WeakRef];
        """,
        backend,
    )
    # The first next() ran once and was replayed from the journal; the echo had the same arguments both times.
    assert restarted.result == ok('[1,2,"undefined"]')
    assert counter[0] == 2
    assert list(chat.echo.values()) == [1]
    assert next(iter(chat.echo)).endswith("/0")
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_made_a_different_number_of_calls(backend: Backend) -> None:
    chat = codemode_chat()

    def change(journal: Any) -> None:
        journal["entries"][0]["issuedAfter"] = 7

    restarted = await restart(
        chat, 'await tools.echo({ text: "a" });\nawait tools.hold();', backend, change_doc(CodemodeJournalDoc, change)
    )
    assert restarted.result.is_error
    assert "after result 1 it had made 2 calls, where it had made 7" in restarted.result.text
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_ends_where_it_went_on_before(backend: Backend) -> None:
    chat = codemode_chat()
    restarted = await restart(
        chat,
        """
        await tools.echo({ text: "a" });
        if (load("k")) return "early";
        await tools.hold();
        """,
        backend,
        set_store({"k": True}),
        journal_reaches(lambda journal: journal["entries"][0].get("issuedAfter") == 2),
    )
    assert restarted.result.is_error
    assert "it ended after result 1, where it went on before" in restarted.result.text
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_resumed_script_makes_a_call_with_other_arguments(backend: Backend) -> None:
    chat = codemode_chat()

    def change(run: Any) -> None:
        run["start"]["store"] = {"k": "other"}

    async def clear_journal(harness: Harness, task: Any) -> None:
        await change_doc(CodemodeRunDoc, change)(harness, task)

        def clear(journal: Any) -> None:
            journal["entries"] = []

        await change_doc(CodemodeJournalDoc, clear)(harness, task)

    restarted = await restart(
        chat, 'await tools.echo({ text: load("k") ?? "a" });\nawait tools.hold();', backend, clear_journal
    )
    assert restarted.result.is_error
    assert "different path" in restarted.result.text
    assert 'call 1 is tools.echo({"text":"other"})' in restarted.result.text
    assert chat.echo == {"a": 1}
    await restarted.harness.close(context)


async def test_ends_the_call_when_a_rerun_makes_a_global_call_where_it_made_a_tool_call(backend: Backend) -> None:
    async def one(_args: Any, _api: Any, _context: Any) -> int:
        return 1

    chat = codemode_chat([CodemodeGlobal("pick", one)])
    restarted = await restart(
        chat,
        """
        if (load("k")) {
          await pick();
          await tools.echo({ text: "new" });
        } else {
          await tools.hold({});
        }
        """,
        backend,
        set_store({"k": True}),
    )
    assert restarted.result.is_error
    assert "call 1 is pick(), but was tools.hold({})" in restarted.result.text
    assert len(chat.echo) == 0
    await restarted.harness.close(context)


@pytest.mark.parametrize("field", ["fingerprint", "version"])
async def test_refuses_to_resume_when_the_fingerprint_or_the_code_mode_version_changed(
    backend: Backend, field: str
) -> None:
    chat = codemode_chat()

    def change(run: Any) -> None:
        if field == "fingerprint":
            run["start"]["fingerprint"] = "another sandbox"
        else:
            run["start"]["version"] = 0

    restarted = await restart(
        chat, 'await tools.hold();\nawait tools.echo({ text: "late" });', backend, change_doc(CodemodeRunDoc, change)
    )
    assert restarted.result.is_error
    assert "cannot resume" in restarted.result.text
    assert len(chat.echo) == 0
    await restarted.harness.close(context)


async def test_refuses_to_resume_a_python_run_on_the_javascript_sandbox(backend: Backend) -> None:
    pytest.importorskip("pydantic_monty")
    from pi_codemode import MontyCodemodeSandbox

    python = MontyCodemodeSandbox(memory_limit_bytes=64 * 1024 * 1024)
    _sandboxes.append(python)
    started = codemode_chat(sandbox=python)
    javascript = codemode_chat()
    # Valid in both languages, so only the sandbox tells the runs apart.
    restarted = await restart(started, "await tools.hold()\nawait tools.echo()", backend, resumed=javascript)
    assert restarted.result.is_error
    assert "The script cannot resume: the sandbox changed since it started (pydantic-monty " in restarted.result.text
    assert "-> quickjs-wasi 3.6.2, quickjs " in restarted.result.text
    assert len(started.echo) == 0
    assert len(javascript.echo) == 0
    await restarted.harness.close(context)


async def test_times_a_script_out_across_a_restart_without_admitting_new_calls(backend: Backend) -> None:
    chat = codemode_chat()

    async def pause(_harness: Harness, _task: Any) -> None:
        await asyncio.sleep(1.1)

    restarted = await restart(
        chat,
        '// @options: {"timeout_ms": 1000}\nawait tools.hold();\nawait tools.echo({ text: "late" });',
        backend,
        pause,
    )
    assert restarted.result.is_error
    assert "deadline passed" in restarted.result.text
    assert len(chat.echo) == 0
    assert chat.starts[0] == 1
    await restarted.harness.close(context)


# ─── results, errors and output ─────────────────────────────────────────────


async def test_rejects_a_failed_tool_call_with_its_error_text_and_structured_output() -> None:
    chat = codemode_chat()
    exit_schema = Type.Object({"output": Type.String(), "exitCode": Type.Number()})

    async def fails(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("permission denied")], "isError": True}

    async def run(_args: Any, _api: Any, _context: Any) -> Any:
        # Like bash on a nonzero exit: an error result with its value.
        return {
            "output": [TextContent("no matches")],
            "structuredOutput": {"output": "no matches", "exitCode": 1},
            "isError": True,
        }

    add_tool(chat.setup.registry, tool("fails", fails))
    for name, execute in (("run", run), ("denied", fails)):
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
    run_result = await run_script(
        """
        let failed;
        try { await tools.fails({}) } catch (error) { failed = [error.message, error.name, error.structuredOutput] }
        const r = await tools.run({});
        let denied;
        try { await tools.denied({}) } catch (error) { denied = [error.message, error.structuredOutput] }
        return [failed, r.exitCode, r.output, denied];
        """,
        chat,
    )
    # A schema-less tool's error carries its text as structuredOutput; a schema tool's error without a value none.
    assert value_of(run_result) == [
        ["permission denied", "Error", "permission denied"],
        1,
        "no matches",
        ["permission denied", None],
    ]


async def test_lists_the_scripts_tool_calls_in_its_results_details() -> None:
    chat = codemode_chat()

    async def fails(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("permission denied")], "isError": True}

    add_tool(chat.setup.registry, tool("fails", fails))
    run = await run_script(
        'await tools.echo({ text: "a" });\ntry { await tools.fails({}) } catch {}\ntools.hold({ id: 9 });\nreturn 1;',
        chat,
    )
    calls = run.message.details["calls"]
    assert [{key: call[key] for key in ("id", "name", "args", "status")} for call in calls] == [
        {"id": 1, "name": "echo", "args": '{"text":"a"}', "status": "ok"},
        {"id": 2, "name": "fails", "args": "{}", "status": "error"},
        {"id": 3, "name": "hold", "args": '{"id":9}', "status": "cancelled"},
    ]
    assert calls[1]["error"] == "permission denied"


async def test_lays_out_text_items_console_lines_and_the_returned_value_under_a_header() -> None:
    run = await run_script(
        'text("one");\nconsole.log("log", 1);\ntext({ two: 2 });\nconsole.error("bad");\nreturn "three";'
    )
    assert re.match(r"^Script completed\nWall time [\d.]+ seconds\nOutput:\n", run.text)
    assert '==> text 1/3 <==\none\n==> text 2/3 <==\n{"two":2}\n==> text 3/3 <==\nthree' in run.text
    assert "<console_output>\nlog 1\nbad\n</console_output>" in run.text


async def test_cuts_text_past_the_token_budget_with_the_full_text_in_a_file() -> None:
    run = await run_script('// @options: {"max_output_tokens": 20}\ntext("BEGIN" + "x".repeat(1000) + "END");')
    assert "Warning: truncated output" in run.text
    path = run.message.details.get("fullOutputPath")
    remember_output_dir(path)
    assert f"[Full output: {path}" in run.text
    assert Path(path).read_text("utf-8") == "BEGIN" + "x" * 1000 + "END"


async def test_saves_images_to_files_and_names_them_next_to_the_image() -> None:
    run = await run_script(f'image("data:image/png;base64,{PNG}");')
    saved = re.search(r"\[Image saved to (\S+) \(image/png, \d+ bytes\)\]", run.text)
    assert saved is not None
    remember_output_dir(saved.group(1))
    assert base64.b64encode(Path(saved.group(1)).read_bytes()).decode() == PNG
    assert any(isinstance(item, ImageContent) for item in run.message.content)


async def test_reports_a_failed_script_with_its_error_and_the_calls_already_made() -> None:
    run = await run_script('await tools.echo({ text: "a" });\nthrow new Error("boom");')
    assert run.message.is_error
    assert re.match(r"^Script failed\n", run.text)
    assert "Error: boom\n    at " in run.text
    assert "codemode.js:2" in run.text
    assert "Tool calls made before the failure (they are not undone): echo (ok)" in run.text


async def test_refuses_an_options_line_below_the_first_line_without_running_anything() -> None:
    chat = codemode_chat()
    run = await run_script('await tools.echo({ text: "a" });\n// @options: {"timeout_ms": 1}', chat)
    assert run.message.is_error
    assert "The // @options: line must be the first line of the input; found one on line 2" in run.text
    assert chat.echo == {}


async def test_keeps_store_values_across_calls_of_a_conversation() -> None:
    chat = codemode_chat()
    chat.holds.block = False
    chat.setup.faux.set_responses(
        [script('store("k", { n: 1 });\nreturn load("k").n;'), DONE, script('return [load("k"), load("nope")];'), DONE]
    )
    opened = await open_chat(await open_sqlite_storage(sqlite_path()), chat.setup)
    root = opened.root
    await (await root.submit({"type": "input", "content": "first"}, context)).wait(context)
    await (await root.submit({"type": "input", "content": "second"}, context)).wait(context)
    results: list[Any] = [entry for entry in await all_entries(root) if ToolResultEntry.is_(entry)]
    assert [entry["model"][0].content[1].text for entry in results] == ["1", '[{"n":1},null]']
    await opened.harness.close(context)


# ─── finding tools ──────────────────────────────────────────────────────────


async def test_finds_tools_with_search_tools_describe_tool_and_describe_namespace() -> None:
    chat = codemode_chat([], [{"name": "github", "description": "GitHub MCP server", "prefix": "mcp__github__"}])

    async def list_issues(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": []}

    add_tool(chat.setup.registry, tool("mcp__github__list_issues", list_issues))
    run = await run_script(
        """
        const found = await searchTools("issues");
        const inNamespace = await searchTools("echo", { namespace: "github" });
        return {
          found: found.map((tool) => tool.name),
          limited: (await searchTools("echo hold slow issues", { limit: 1 })).length,
          inNamespace,
          described: (await describeTool("echo")).includes("echo(args: {"),
          missing: (await describeTool("nope")) === undefined,
          namespace: await describeNamespace("github"),
          all: ALL_TOOLS.some((tool) => tool.name === "mcp__github__list_issues"),
          python: typeof search_tools,
        };
        """,
        chat,
    )
    assert value_of(run) == {
        "found": ["mcp__github__list_issues"],
        "limited": 1,
        "inNamespace": [],
        "described": True,
        "missing": True,
        "namespace": {"name": "github", "description": "GitHub MCP server", "tools": ["mcp__github__list_issues"]},
        "all": True,
        "python": "undefined",
    }


async def test_rejects_bad_discovery_arguments_with_upstreams_messages() -> None:
    run = await run_script(
        """
        const errors = [];
        for (const call of [
          () => searchTools(1),
          () => searchTools("x", { limit: 0 }),
          () => searchTools("x", { limit: 1.5 }),
          () => searchTools("x", { namespace: 1 }),
          () => describeTool(1),
          () => describeNamespace(),
        ]) {
          try { await call(); errors.push("no error"); } catch (error) { errors.push(error.message); }
        }
        return errors;
        """
    )
    assert value_of(run) == [
        "searchTools() expects a query string",
        "searchTools() limit must be a positive integer",
        "searchTools() limit must be a positive integer",
        "searchTools() namespace must be a string",
        "describeTool() expects a tool name",
        "describeNamespace() expects a namespace name",
    ]


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
    run = await run_script('return [await tools.later({}), ALL_TOOLS.some((tool) => tool.name === "later")];', chat)
    assert value_of(run) == ["later ran", True]
