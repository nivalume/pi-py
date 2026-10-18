"""Port of ``test/harness-nested-tools-restart.test.ts``.

Adaptation: SQLite runs on executor threads here, so waits on it poll in real time (``wait_for``); upstream's
``node:sqlite`` is synchronous, so a fixed number of turns (``eventually``) suffices there.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
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
from pi_ai.serialization import to_json_value
from pi_ai.types import AssistantMessage, TextContent, ToolResultMessage
from pi_chord.context import Context
from pi_durable import LiveDoc, ToolResultEntry, ToolTask, define_extension, define_task, define_tool, watch_events
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

Parameters = Type.Object({"text": Type.Optional(Type.String())})


def tool(name: str, execute: Any, **extra: Any) -> Any:
    return define_tool(name=name, description=name, parameters=Parameters, execute=execute, **extra)


def call(name: str, id: str = "c1") -> AssistantMessage:
    return faux_assistant_message([faux_tool_call(name, {}, id=id)], stop_reason="toolUse")


DONE = faux_assistant_message([faux_text("done")])


async def open(path: Path, setup: ChatSetup) -> Any:
    """Open the SQLite Harness without starting scheduling."""
    return await open_chat(await open_sqlite_storage(path), setup)


async def nested_tasks(harness: Any) -> list[Any]:
    page = await harness.commit(lambda tx: tx.scan_tasks({"kind": ToolTask.definition.name}, 100), context)
    return [task for task in page["items"] if task["input"]["kind"] == "nested"]


def results(entries: Sequence[Any]) -> list[ToolResultMessage]:
    return [entry["model"][0] for entry in entries if ToolResultEntry.is_(entry)]


class Ends(NamedTuple):
    ends: list[Any]
    stream: Any


async def collect_ends(harness: Any, conversation_id: Any) -> Ends:
    """Collect tool end events from now on."""
    ends: list[Any] = []
    stream = await watch_events(harness, conversation_id, context)

    async def listener(batch: Any, _context: Context) -> None:
        ends.extend(event for event in batch if event["type"] == "tool_execution_end")

    stream.start(listener)
    return Ends(ends, stream)


def codes(result: Any) -> list[str]:
    return [diagnostic["code"] for diagnostic in (result or {}).get("diagnostics", []) if "code" in diagnostic]


def output(value: str) -> Any:
    return {"output": [TextContent(value)]}


async def submission_status(harness: Any, id: Any) -> str:
    return (await (await harness.submission(id, context)).wait(context))["status"]


async def _abort_terminal(_task: Any, runtime: Any, abort_context: Context) -> None:
    await runtime.commit(lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, abort_context)


def spawn_worker(worker: Any, seen: dict[str, Any]) -> Any:
    """A tool that creates ``worker`` as a task it owns and waits for it."""

    async def spawn(_args: Any, api: Any, call_context: Context) -> Any:
        seen["worker"] = await api.create_task(
            worker, None, {"ownership": {"kind": "task", "taskId": api.task_id}}, call_context
        )
        await api.wait_for_task(seen["worker"], call_context)
        return {}

    return tool("spawn", spawn)


async def _batch_spawn(_args: Any, api: Any, call_context: Context) -> Any:
    await api.execute_tool("spawn", {}, call_context)
    return {}


async def test_abandons_a_replay_unsafe_callers_unfinished_nested_calls_before_they_run_again(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    runs = {"started": 0, "gated": 0}
    running = deferred()
    gated = deferred()

    # Replay-safe, so only the abandonment keeps it from rerunning.
    async def started(_args: Any, api: Any, call_context: Context) -> Any:
        runs["started"] += 1
        api.output("partial\n")
        await api.details({"step": 1}, call_context)
        resolve(running)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    async def gated_tool(*_args: Any) -> Any:
        runs["gated"] += 1
        return {}

    add_tool(setup.registry, tool("started", started, replay="safe"))
    add_tool(setup.registry, tool("gated", gated_tool))

    # Holds `gated` in its `call` phase, before intent, until the Harness closes.
    async def before_tool(hook_call: Any, _api: Any, hook_context: Context) -> Any:
        if hook_call.name != "gated":
            return None
        resolve(gated)
        return await aborted(hook_context.abort_signal)  # type: ignore[arg-type]

    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        first = api.execute_tool("started", {}, call_context)
        second = api.execute_tool("gated", {}, call_context)
        await asyncio.gather(first, second)
        return {}

    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await asyncio.gather(running, gated)
    await opened.harness.close(context)

    opened = await open(path, setup)
    ends, stream = await collect_ends(opened.harness, opened.root.id)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    await opened.harness.wait_for_idle(context)
    await stream.stop()
    assert runs == {"started": 1, "gated": 0}
    started_task, gate_task = await nested_tasks(opened.harness)
    for task in [started_task, gate_task]:
        assert match_object(task, {"abandonOnRestart": True, "abortRequested": True, "abortReason": "restart"})
        assert match_object(task["state"], {"status": "terminal", "outcome": {"status": "aborted"}})

    def end(call_id: str) -> Any:
        return next((event for event in ends if event["toolCallId"] == call_id), {}).get("result")

    assert codes(end("c1/1")) == ["interrupted"]
    assert match_object(end("c1/1"), {"isError": True, "details": {"step": 1}})
    assert "structuredOutput" not in end("c1/1")
    assert codes(end("c1/2")) == ["abandoned"]
    # The caller reports the interruption.
    [result] = results(await all_entries(opened.root))
    assert (result.tool_call_id, result.is_error) == ("c1", True)
    assert await opened.harness.snapshot(LiveDoc, opened.root.id, context) == {}
    await opened.harness.close(context)


async def test_abandons_the_work_an_abandoned_nested_call_owns_which_never_runs_a_phase_again(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    phases: list[str] = []
    aborts: list[Any] = []
    started = deferred()

    async def work(_task: Any, runtime: Any, _ctx: Context) -> None:
        phases.append("work")
        resolve(started)
        await aborted(runtime.signal)

    async def abort(task: Any, runtime: Any, abort_context: Context) -> None:
        aborts.append(task.get("abortReason"))
        await _abort_terminal(task, runtime, abort_context)

    Worker = define_task(
        name="test.worker", version=1, initial=lambda _input: {"phase": "work"}, phases={"work": work}, abort=abort
    )
    setup.registry.install(define_extension("worker", tasks=[Worker]))
    seen: dict[str, Any] = {}
    add_tool(setup.registry, spawn_worker(Worker, seen))
    add_tool(setup.registry, tool("batch", _batch_spawn))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await started
    await opened.harness.close(context)

    opened = await open(path, setup)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    await opened.harness.wait_for_idle(context)
    assert phases == ["work"]
    # The cascade passed the restart reason on.
    assert aborts == ["restart"]
    assert match_object(
        ((await opened.harness.get_task(seen["worker"], context)) or {}).get("state"),
        {"status": "terminal", "outcome": {"status": "aborted"}},
    )
    await opened.harness.close(context)


async def test_keeps_an_abandoned_task_whose_definition_is_missing_waiting_until_it_is_installed_then_runs_its_abort(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    aborts: list[Any] = []
    started = deferred()

    async def work(_task: Any, runtime: Any, _ctx: Context) -> None:
        resolve(started)
        await aborted(runtime.signal)

    async def abort(task: Any, runtime: Any, abort_context: Context) -> None:
        aborts.append(task.get("abortReason"))
        await _abort_terminal(task, runtime, abort_context)

    Worker = define_task(
        name="test.late", version=1, initial=lambda _input: {"phase": "work"}, phases={"work": work}, abort=abort
    )
    extension = define_extension("late", tasks=[Worker])
    setup.registry.install(extension)
    seen: dict[str, Any] = {}
    add_tool(setup.registry, spawn_worker(Worker, seen))
    add_tool(setup.registry, tool("batch", _batch_spawn))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await started
    await opened.harness.close(context)

    setup.registry.uninstall(extension)
    opened = await open(path, setup)
    opened.harness.resume()

    async def blocked() -> bool:
        inspected = next(
            (
                task
                for task in (await opened.harness.inspect(context))["tasks"]
                if task["record"]["id"] == seen["worker"]
            ),
            None,
        )
        return inspected is not None and inspected["state"]["kind"] == "blocked"

    await wait_for(blocked)
    assert ((await opened.harness.get_task(seen["worker"], context)) or {})["state"]["status"] != "terminal"
    assert aborts == []
    setup.registry.install(extension)
    assert await submission_status(opened.harness, id) == "done"
    assert aborts == ["restart"]
    assert match_object(
        ((await opened.harness.get_task(seen["worker"], context)) or {}).get("state"),
        {"status": "terminal", "outcome": {"status": "aborted"}},
    )
    await opened.harness.close(context)


async def test_orphans_an_abandoned_task_whose_definition_is_missing_once_an_abort_is_requested(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    started = deferred()

    async def work(_task: Any, runtime: Any, _ctx: Context) -> None:
        resolve(started)
        await aborted(runtime.signal)

    Worker = define_task(
        name="test.gone",
        version=1,
        initial=lambda _input: {"phase": "work"},
        phases={"work": work},
        abort=_abort_terminal,
    )
    extension = define_extension("gone", tasks=[Worker])
    setup.registry.install(extension)
    seen: dict[str, Any] = {}
    add_tool(setup.registry, spawn_worker(Worker, seen))
    add_tool(setup.registry, tool("batch", _batch_spawn))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await started
    await opened.harness.close(context)

    setup.registry.uninstall(extension)
    opened = await open(path, setup)
    opened.harness.resume()

    async def restart_marked() -> bool:
        return ((await opened.harness.get_task(seen["worker"], context)) or {}).get("abortReason") == "restart"

    await wait_for(restart_marked)
    await opened.root.abort(context)
    assert match_object(
        ((await opened.harness.get_task(seen["worker"], context)) or {}).get("state"),
        {"status": "terminal", "outcome": {"status": "orphaned", "reason": "missing_task"}},
    )
    assert await submission_status(opened.harness, id) == "unanswered"
    await opened.harness.close(context)


async def test_marks_nothing_when_the_reopened_harness_never_starts_scheduling(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    running = deferred()

    async def hang(_args: Any, _api: Any, call_context: Context) -> Any:
        resolve(running)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("hang", {}, call_context)
        return {}

    add_tool(setup.registry, tool("hang", hang))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    await opened.root.submit({"type": "input", "content": "go"}, context)
    await running
    await opened.harness.close(context)

    # An inspection-only open: no resume, no submission.
    opened = await open(path, setup)
    inspected = await opened.harness.inspect(context)
    assert inspected["scheduling"] == "paused"
    await opened.harness.close(context)

    opened = await open(path, setup)
    [nested] = await nested_tasks(opened.harness)
    assert match_object(nested, {"abandonOnRestart": True, "abortRequested": False})
    assert nested["state"]["status"] == "pending"
    await opened.harness.close(context)


async def test_lets_a_replay_safe_callers_unfinished_nested_calls_recover_and_reattaches_to_them_by_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    runs = {"safe": 0, "unsafe": 0, "caller": 0}
    running = {"safe": deferred(), "unsafe": deferred()}

    async def safe(_args: Any, _api: Any, call_context: Context) -> Any:
        runs["safe"] += 1
        if runs["safe"] == 1:
            resolve(running["safe"])
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        return output(f"safe run {runs['safe']}")

    async def unsafe(_args: Any, _api: Any, call_context: Context) -> Any:
        runs["unsafe"] += 1
        resolve(running["unsafe"])
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    collected: list[Any] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        runs["caller"] += 1
        both = await asyncio.gather(
            api.execute_tool("safe", {}, call_context), api.execute_tool("unsafe", {}, call_context)
        )
        collected[:] = both
        return {}

    add_tool(setup.registry, tool("safe", safe, replay="safe"))
    add_tool(setup.registry, tool("unsafe", unsafe))
    add_tool(setup.registry, tool("batch", batch, replay="safe"))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await asyncio.gather(running["safe"], running["unsafe"])
    await opened.harness.close(context)

    opened = await open(path, setup)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    assert runs == {"safe": 2, "unsafe": 1, "caller": 2}
    # Default keys follow call order, so the rerun found both calls.
    tasks = await nested_tasks(opened.harness)
    assert len(tasks) == 2
    for task in tasks:
        assert "abandonOnRestart" not in task
        assert "abortReason" not in task
    assert collected[0].get("structuredOutput") == "safe run 2"
    assert codes(collected[1]) == ["interrupted"]
    await opened.harness.close(context)


async def test_clears_the_slots_of_nested_calls_an_earlier_attempt_finished_when_the_rerun_makes_none(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    finished = deferred()
    attempts = [0]

    async def quick(*_args: Any) -> Any:
        return output("ok")

    async def caller(_args: Any, api: Any, call_context: Context) -> Any:
        attempts[0] += 1
        if attempts[0] == 1:
            await api.execute_tool("quick", {}, call_context)
            resolve(finished)
            return await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        return output("done")

    add_tool(setup.registry, tool("quick", quick))
    add_tool(setup.registry, tool("caller", caller, replay="safe"))
    setup.faux.set_responses([call("caller"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await finished
    # The finished nested call's slot is still listed, below its running caller.
    live: Any = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
    assert len(live["nestedTools"]) == 1
    await opened.harness.close(context)

    opened = await open(path, setup)
    slots_after_caller: list[int] = []

    def listener(publication: Any, _context: Context) -> None:
        for change in publication["changes"]:
            if change["type"] != "task" or change["value"]["kind"] != ToolTask.definition.name:
                continue
            if change["value"]["input"]["kind"] != "model" or change["value"]["state"]["status"] != "terminal":
                continue
            found = next(
                (
                    each
                    for each in publication["changes"]
                    if each["type"] == "document" and each["record"]["kind"] == "pi.live"
                ),
                None,
            )
            value = None if found is None else found["value"]
            slots_after_caller.append(len((value or {}).get("nestedTools") or []))

    opened.harness.subscribe_commits(listener)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    assert attempts[0] == 2
    # The caller's terminal commit removed the slot, though the rerun made no nested call.
    assert slots_after_caller[-1:] == [0]
    await opened.harness.close(context)


async def test_aborts_a_nested_call_an_earlier_attempt_of_a_replay_safe_caller_left_running_and_the_rerun_never_made_again(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    running = deferred()
    attempts = [0]

    async def slow(_args: Any, _api: Any, call_context: Context) -> Any:
        resolve(running)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    async def caller(_args: Any, api: Any, call_context: Context) -> Any:
        attempts[0] += 1
        # Only the first attempt calls `slow`; the rerun after the restart returns at once.
        if attempts[0] == 1:
            await api.execute_tool("slow", {}, call_context, key="slow")
        return output("done")

    add_tool(setup.registry, tool("slow", slow, replay="safe"))
    add_tool(setup.registry, tool("caller", caller, replay="safe"))
    setup.faux.set_responses([call("caller"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await running
    await opened.harness.close(context)

    opened = await open(path, setup)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    assert attempts[0] == 2
    # The rerun's settlement found the leftover call among the tasks it owns and aborted it.
    [leftover] = await nested_tasks(opened.harness)
    assert match_object(leftover["state"], {"status": "terminal", "outcome": {"status": "aborted"}})
    await opened.harness.close(context)


@pytest.mark.parametrize("order", ["same", "swapped"])
async def test_reattaches_a_replay_safe_caller_by_default_and_explicit_keys_and_rejects_a_rerun_that_calls_differently(
    tmp_path: Path, order: str
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    runs: list[str] = []

    async def echo(args: Any, *_rest: Any) -> Any:
        runs.append(args.get("text") or "")
        return output(f"echo {args.get('text')}")

    add_tool(setup.registry, tool("echo", echo))
    blocked = deferred()
    caller_runs = [0]
    seen: list[str] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        caller_runs[0] += 1
        first = "b" if caller_runs[0] == 2 and order == "swapped" else "a"
        one = await api.execute_tool("echo", {"text": first}, call_context)
        named = await api.execute_tool("echo", {"text": "named"}, call_context, key="named")
        if caller_runs[0] == 1:
            resolve(blocked)
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        seen[:] = [result["structuredOutput"] for result in (one, named)]
        return {}

    add_tool(setup.registry, tool("batch", batch, replay="safe"))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await blocked
    await opened.harness.close(context)

    opened = await open(path, setup)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    [result] = results(await all_entries(opened.root))
    if order == "same":
        assert runs == ["a", "named"]
        assert seen == ["echo a", "echo named"]
        assert result.is_error is False
    else:
        # Key 1 was made with other arguments: the rerun throws instead of reusing a different call's result.
        assert runs == ["a", "named"]
        assert result.is_error is True
        assert "Nested call c1/1 was already made with another tool" in json.dumps(to_json_value(result.content))
    await opened.harness.close(context)


async def test_abandons_the_child_tasks_a_replay_unsafe_tool_created_and_awaited_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    phases: list[str] = []
    started = deferred()

    async def work(_task: Any, runtime: Any, _ctx: Context) -> None:
        phases.append("work")
        resolve(started)
        await aborted(runtime.signal)

    async def abort(task: Any, runtime: Any, abort_context: Context) -> None:
        phases.append(f"abort {task.get('abortReason')}")
        await _abort_terminal(task, runtime, abort_context)

    Worker = define_task(
        name="test.direct-worker",
        version=1,
        initial=lambda _input: {"phase": "work"},
        phases={"work": work},
        abort=abort,
    )
    setup.registry.install(define_extension("direct-worker", tasks=[Worker]))

    async def work_tool(_args: Any, api: Any, call_context: Context) -> Any:
        id = await api.create_task(Worker, None, {"ownership": {"kind": "task", "taskId": api.task_id}}, call_context)
        await api.wait_for_task(id, call_context)
        return {}

    add_tool(setup.registry, tool("work", work_tool))
    setup.faux.set_responses([call("work"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await started
    await opened.harness.close(context)

    opened = await open(path, setup)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    assert phases == ["work", "abort restart"]
    await opened.harness.close(context)


async def test_leaves_background_work_under_an_abandoned_nested_call_running_and_settles_without_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    started = deferred()

    async def work(_task: Any, runtime: Any, _ctx: Context) -> None:
        resolve(started)
        await aborted(runtime.signal)

    Background = define_task(
        name="test.background",
        version=1,
        initial=lambda _input: {"phase": "work"},
        phases={"work": work},
        abort=_abort_terminal,
    )
    setup.registry.install(define_extension("background", tasks=[Background]))
    seen: dict[str, Any] = {}

    async def spawn(_args: Any, api: Any, call_context: Context) -> Any:
        # A conversation the call owns, with background work in it, as a persistent subagent has.
        async def change(tx: Any) -> Any:
            child = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
            return await tx.create_task(
                Background,
                None,
                {"conversationId": child["id"], "ownership": {"kind": "conversation"}, "background": True},
            )

        seen["background"] = await api.commit(change, call_context)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    add_tool(setup.registry, tool("spawn", spawn))
    add_tool(setup.registry, tool("batch", _batch_spawn))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open(path, setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await started
    await opened.harness.close(context)

    opened = await open(path, setup)
    opened.harness.resume()
    assert await submission_status(opened.harness, id) == "done"
    [spawn_task] = await nested_tasks(opened.harness)
    assert match_object(spawn_task, {"abortReason": "restart", "state": {"status": "terminal"}})
    assert match_object(
        await opened.harness.get_task(seen["background"], context),
        {"abortRequested": False, "state": {"status": "running"}},
    )
    await opened.harness.abort_task(seen["background"], context)
    await opened.harness.close(context)
