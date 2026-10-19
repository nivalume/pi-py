"""Port of ``test/harness-tools-recovery.test.ts``."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from durable_harness_support import (
    ChatSetup,
    aborted,
    add_hooks,
    add_tool,
    all_entries,
    chat_setup,
    match_object,
    open_chat,
    wait_for,
)
from durable_session_support import context, deferred, resolve

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage, TextContent, ToolResultMessage
from pi_chord.context import Context
from pi_durable import GenerationTask, LiveDoc, ToolResultEntry, ToolTask, define_tool
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.tools import create_bash_tool


async def open(path: Path, setup: ChatSetup, env: Any = None) -> Any:
    opened = await open_chat(await open_sqlite_storage(path), setup, env=env)
    opened.harness.resume()
    return opened


def tool(name: str, execute: Any, **extra: Any) -> Any:
    return define_tool(name=name, description=name, parameters=Type.Object({}), execute=execute, **extra)


def call(name: str, id: str = "c1") -> AssistantMessage:
    return faux_assistant_message([faux_tool_call(name, {}, id=id)], stop_reason="toolUse")


DONE = faux_assistant_message([faux_text("done")])


def results(entries: list[Any]) -> list[ToolResultMessage]:
    return [entry["model"][0] for entry in entries if ToolResultEntry.is_(entry)]


def text(message: ToolResultMessage | None) -> str:
    return "|".join(item.text if isinstance(item, TextContent) else "" for item in (message.content if message else []))


async def tool_task_id(harness: Any) -> Any:
    found: list[Any] = []

    async def check() -> bool:
        live = await harness.snapshot(LiveDoc, 1, context)
        tools = (live or {}).get("tools") or []
        if tools and tools[0].get("taskId") is not None:
            found.append(tools[0]["taskId"])
            return True
        return False

    await wait_for(check)
    return found[0]


class Blocking(NamedTuple):
    registration: Any
    started: Any
    state: dict[str, int]


def blocking_tool(name: str, **extra: Any) -> Blocking:
    """A tool that writes output, then blocks until its invocation is cancelled the first ``blocking`` times it runs.

    ``started`` resolves once the output is durable.
    """
    started = deferred()
    state = {"runs": 0, "blocking": 1}

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        state["runs"] += 1
        api.output(f"run {state['runs']}\n")
        await api.details({"run": state["runs"]}, call_context)
        if state["runs"] <= state["blocking"]:
            resolve(started)
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        return {}

    return Blocking(tool(name, execute, **extra), started, state)


async def submission_status(harness: Any, id: Any) -> str:
    return (await (await harness.submission(id, context)).wait(context))["status"]


async def test_answers_an_unsafe_tool_interrupted_after_intent_with_its_durable_partial_output(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    registration, started, state = blocking_tool("work")
    add_tool(setup.registry, registration)
    setup.faux.set_responses([call("work"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await started
    task_id = await tool_task_id(opened.harness)
    await opened.harness.close(context)

    opened = await open(path, setup)
    assert match_object(
        ((await opened.harness.get_task(task_id, context)) or {}).get("state"),
        {"checkpoint": {"phase": "execute", "arguments": {}, "replay": "unsafe"}},
    )
    assert await submission_status(opened.harness, id) == "done"
    assert state["runs"] == 1
    [result] = results(await all_entries(opened.root))
    assert result.is_error is True
    assert result.details == {"run": 1}
    assert text(result) == "run 1\n|<harness>\n[error] Tool work was interrupted and may have partially run\n</harness>"
    assert await opened.harness.snapshot(LiveDoc, opened.root.id, context) == {}
    await opened.harness.close(context)


@pytest.mark.parametrize(
    ("stored", "current", "reruns"), [("safe", "safe", True), ("safe", "unsafe", False), ("unsafe", "safe", False)]
)
async def test_reruns_a_tool_only_when_both_the_stored_and_the_current_replay_policy_are_safe(
    tmp_path: Path, stored: str, current: str, reruns: bool
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    blocking = blocking_tool("work", replay=stored)
    registration = add_tool(setup.registry, blocking.registration)
    setup.faux.set_responses([call("work"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await blocking.started
    await opened.harness.close(context)

    registration.dispose()
    add_tool(setup.registry, dataclasses.replace(blocking.registration, replay=current))
    opened = await open(path, setup)
    assert await submission_status(opened.harness, id) == "done"
    [result] = results(await all_entries(opened.root))
    assert blocking.state["runs"] == (2 if reruns else 1)
    assert result.is_error is (not reruns)
    if reruns:
        assert text(result) == "run 2\n"
    await opened.harness.close(context)


@pytest.mark.parametrize("change", ["cwd", "deselect"])
async def test_reruns_a_safe_tool_with_the_environment_of_the_conversations_cwd_at_rerun_and_not_once_it_is_deselected(
    tmp_path: Path, change: str
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    cwds: list[Any] = []
    started = deferred()

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        cwds.append(None if api.env is None else api.env.cwd)
        if len(cwds) == 1:
            resolve(started)
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        return {}

    add_tool(setup.registry, tool("work", execute, replay="safe"))
    setup.faux.set_responses([call("work"), DONE])

    def env(target: Any, _context: Context) -> Any:
        return LocalExecutionEnv(cwd=target.get("cwd", "/"))

    opened = await open(path, setup, env)
    await opened.root.configure({"cwd": "/one"}, context)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await started
    await opened.root.configure({"cwd": "/two"} if change == "cwd" else {"extensions": []}, context)
    await opened.harness.close(context)

    opened = await open(path, setup, env)
    assert await submission_status(opened.harness, id) == "done"
    [result] = results(await all_entries(opened.root))
    if change == "cwd":
        assert cwds == ["/one", "/two"]
        assert result.is_error is False
    else:
        # A tool that no longer resolves is treated as unsafe: interrupted, not rerun.
        assert cwds == ["/one"]
        assert "was interrupted" in text(result)
    await opened.harness.close(context)


async def test_reruns_before_tool_when_interrupted_before_intent_and_executes_once(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    runs = [0]

    async def work(*_args: Any) -> Any:
        runs[0] += 1
        return {"output": []}

    add_tool(setup.registry, tool("work", work))
    reached = deferred()
    asked = [0]
    decisions: list[str] = []

    async def before_tool(_call: Any, api: Any, call_context: Context) -> Any:
        asked[0] += 1
        # A durable first-writer-wins decision survives the rerun.
        decisions.append(await api.memo("test:decision", f"attempt {asked[0]}", call_context))
        if asked[0] == 1:
            resolve(reached)
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        return None

    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})
    setup.faux.set_responses([call("work"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await reached
    await opened.harness.close(context)

    opened = await open(path, setup)
    assert await submission_status(opened.harness, id) == "done"
    assert [asked[0], runs[0]] == [2, 1]
    assert decisions == ["attempt 1", "attempt 1"]
    await opened.harness.close(context)


async def test_reruns_the_generation_tools_phase_interrupted_before_its_commit(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()

    async def work(*_args: Any) -> Any:
        return {"output": []}

    add_tool(setup.registry, tool("work", work))
    reached = deferred()
    observed = [0]

    async def after_tools(_assistant: Any, _results: Any, _api: Any, call_context: Context) -> None:
        observed[0] += 1
        if observed[0] == 1:
            resolve(reached)
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    add_hooks(setup.registry, GenerationTask, {"after_tools": after_tools})
    setup.faux.set_responses([call("work"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await reached
    await opened.harness.close(context)

    opened = await open(path, setup)
    assert await submission_status(opened.harness, id) == "done"
    assert observed[0] == 2
    assert [entry["kind"] for entry in await all_entries(opened.root)] == [
        "pi.user",
        "pi.system",
        "pi.assistant",
        "pi.tool-result",
        "pi.assistant",
    ]
    await opened.harness.close(context)


async def test_answers_an_aborted_tool_with_its_partial_output_and_continues_the_run(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    registration, started, _state = blocking_tool("work")
    add_tool(setup.registry, registration)
    setup.faux.set_responses([call("work"), DONE])
    opened = await open(path, setup)
    submission = await opened.root.submit({"type": "input", "content": "go"}, context)
    await started
    task_id = await tool_task_id(opened.harness)
    assert await opened.harness.abort_task(task_id, context) == "marked"
    outcome = (await opened.harness.wait_for_task(task_id, context))["state"]["outcome"]
    assert outcome["status"] == "aborted"
    assert isinstance(outcome["result"]["entryId"], int)
    assert (await submission.wait(context))["status"] == "done"
    [result] = results(await all_entries(opened.root))
    assert text(result) == "run 1\n|<harness>\n[error] Tool work was aborted\n</harness>"
    await opened.harness.close(context)


async def test_lets_context_derivation_answer_a_faulted_tool_and_continues_the_run(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()

    # A result that is not strict JSON makes the result commit throw, so the scheduler faults the task.
    async def bad(*_args: Any) -> Any:
        return {"output": [], "details": {"fn": lambda: 1}}

    add_tool(setup.registry, tool("bad", bad))
    requests: list[str] = []

    def answer(request: Any, *_rest: Any) -> Any:
        result = next((message for message in request.messages if message.role == "toolResult"), None)
        requests.append("none" if result is None else text(result))
        return DONE

    setup.faux.set_responses([call("bad"), answer])
    opened = await open(path, setup)
    settled = await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert settled["status"] == "done"
    assert results(await all_entries(opened.root)) == []
    assert requests == ["Tool result unavailable: history ends before this call completed."]
    tasks = await opened.harness.commit(lambda tx: tx.scan_tasks({"conversationId": opened.root.id}, 20), context)
    faulted = next(task for task in tasks["items"] if task["kind"] == "pi.tool")
    assert match_object(faulted["state"], {"status": "terminal", "outcome": {"status": "faulted"}})
    await opened.harness.close(context)


async def test_answers_a_real_bash_command_interrupted_by_close_and_reopen_then_finishes_the_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    env = LocalExecutionEnv(cwd=str(tmp_path))
    setup = chat_setup()
    add_tool(setup.registry, create_bash_tool())
    setup.faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("bash", {"command": "echo started; sleep 30"}, id="b")], stop_reason="toolUse"
            ),
            DONE,
        ]
    )
    opened = await open(path, setup, env)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id

    async def started() -> bool:
        live = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
        tools = (live or {}).get("tools") or []
        return bool(tools) and tools[0].get("output") == "started\n"

    await wait_for(started)
    await opened.harness.close(context)

    opened = await open(path, setup, env)
    assert await submission_status(opened.harness, id) == "done"
    [result] = results(await all_entries(opened.root))
    assert (
        text(result) == "started\n|<harness>\n[error] Tool bash was interrupted and may have partially run\n</harness>"
    )
    await opened.harness.close(context)


async def test_clears_the_interrupted_attempts_progress_before_a_safe_rerun(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    started = [deferred(), deferred()]
    runs = [0]

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        run = runs[0]
        runs[0] += 1
        if run == 0:
            api.diagnostic({"severity": "info", "message": "first a"})
            api.diagnostic({"severity": "info", "message": "first b"})
            await api.details({"run": 1, "extra": True}, call_context)
        else:
            api.diagnostic({"severity": "info", "message": "second"})
            await api.details({"run": 2}, call_context)
        resolve(started[run])
        await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        return {}

    add_tool(setup.registry, tool("work", execute, replay="safe"))
    setup.faux.set_responses([call("work"), DONE])
    opened = await open(path, setup)
    await opened.root.submit({"type": "input", "content": "go"}, context)
    await started[0]
    await opened.harness.close(context)

    opened = await open(path, setup)
    await started[1]
    slot = (((await opened.harness.snapshot(LiveDoc, opened.root.id, context)) or {}).get("tools") or [{}])[0]
    assert slot.get("diagnostics") == [{"severity": "info", "message": "second"}]
    assert slot.get("details") == {"run": 2}
    await opened.harness.close(context)
