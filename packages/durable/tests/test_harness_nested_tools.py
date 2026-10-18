"""Port of ``test/harness-nested-tools.test.ts``.

Adaptations: upstream rebinds ``storage.commit`` on a storage instance, here a ``MemoryStorage`` subclass records
the writes; a value that is not JSON is ``object()`` (upstream a ``BigInt``); repaired arguments set a property to
``UNDEFINED`` (JavaScript ``undefined``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from durable_harness_support import (
    ChatSetup,
    aborted,
    add_hooks,
    add_tool,
    all_entries,
    chat_setup,
    eventually,
    match_object,
    open_chat,
)
from durable_session_support import context, deferred, resolve

from pi_ai import AbortController, Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage, ImageContent, TextContent, ToolResultMessage, Usage, UsageCost
from pi_chord.context import Context, with_abort_signal
from pi_chord.json import UNDEFINED
from pi_durable import (
    AssistantEntry,
    LiveDoc,
    MemoryStorage,
    NestedCallDoc,
    ToolResultEntry,
    ToolTask,
    UsageDoc,
    define_tool,
    watch_events,
)
from pi_durable.harness.usage import usage_json
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.types import Seq, StorageWrite

Parameters = Type.Object({"text": Type.Optional(Type.String())})


def tool(name: str, execute: Any, **extra: Any) -> Any:
    return define_tool(name=name, description=name, parameters=Parameters, execute=execute, **extra)


class Echo(NamedTuple):
    registration: Any
    state: dict[str, int]


def echo_tool(**extra: Any) -> Echo:
    """Returns ``echo <text>`` and counts its runs."""
    state = {"runs": 0}

    async def execute(args: Any, *_rest: Any) -> Any:
        state["runs"] += 1
        return {"output": [TextContent(f"echo {args.get('text') or ''}")]}

    return Echo(tool("echo", execute, **extra), state)


def call(name: str, args: dict[str, Any] | None = None, id: str = "c1") -> AssistantMessage:
    return faux_assistant_message([faux_tool_call(name, args or {}, id=id)], stop_reason="toolUse")


DONE = faux_assistant_message([faux_text("done")])


def results(entries: Sequence[Any]) -> list[ToolResultMessage]:
    return [entry["model"][0] for entry in entries if ToolResultEntry.is_(entry)]


def text(result: Any) -> str:
    """The text of a result message, or the value a schema-less nested call gives programs."""
    if result is None:
        return ""
    if not isinstance(result, ToolResultMessage):
        output = result.get("structuredOutput")
        return output if isinstance(output, str) else ""
    return "|".join(item.text if isinstance(item, TextContent) else "" for item in result.content)


def output(value: str) -> Any:
    return {"output": [TextContent(value)]}


async def tool_tasks(harness: Any) -> list[Any]:
    page = await harness.commit(lambda tx: tx.scan_tasks({"kind": ToolTask.definition.name}, 100), context)
    return list(page["items"])


class Ran(NamedTuple):
    harness: Any
    root: Any
    status: str
    entries: list[Any]


async def run_once(setup: ChatSetup, responses: list[Any]) -> Ran:
    setup.faux.set_responses(responses)
    opened = await open_chat(MemoryStorage(), setup)
    settled = await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    return Ran(opened.harness, opened.root, settled["status"], await all_entries(opened.root))


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def detached(awaitable: Any) -> asyncio.Future[Any]:
    """``void promise.catch(() => {})``."""
    future = asyncio.ensure_future(awaitable)
    future.add_done_callback(_consume)
    return future


USAGE = Usage(input=1, output=2, cache_read=0, cache_write=0, total_tokens=3, cost=UsageCost(0.1, 0.2, 0, 0, 0.3))


def ends_of(batch: Sequence[Any]) -> list[Any]:
    return [event for event in batch if event["type"] == "tool_execution_end"]


# ─── nested tool calls ──────────────────────────────────────────────────────


async def test_stores_one_document_per_nested_call_keyed_by_its_key_written_only_by_that_call() -> None:
    setup = chat_setup()
    echo = echo_tool()
    add_tool(setup.registry, echo.registration)

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await asyncio.gather(
            api.execute_tool("echo", {"text": "a"}, call_context),
            api.execute_tool("echo", {"text": "b"}, call_context, key="named"),
        )
        return {}

    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    # Per document written, the keys of the nested calls it holds.
    written: list[dict[str, Any]] = []

    class Recording(MemoryStorage):
        async def commit(self, writes: Sequence[StorageWrite], commit_context: Context) -> Seq:
            for write in writes:
                if write["type"] == "document.create":
                    record: Any = write["record"]  # type: ignore[typeddict-item]
                    written.append({"kind": record["kind"], "key": record.get("key")})
            return await super().commit(writes, commit_context)

    opened = await open_chat(Recording(), setup)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    nested = [write for write in written if write["kind"].startswith("pi.tool.")]
    assert nested == [
        {"kind": NestedCallDoc.definition.kind, "key": "1"},
        {"kind": NestedCallDoc.definition.kind, "key": "named"},
    ]
    await opened.harness.close(context)


async def test_rejects_a_nested_call_of_an_aborted_caller_and_leaves_no_document_for_it() -> None:
    setup = chat_setup()
    add_tool(setup.registry, echo_tool().registration)
    started = deferred()
    marked = deferred()
    seen: dict[str, Any] = {}

    async def late(_args: Any, api: Any, _call_context: Context) -> Any:
        seen["caller"] = api.task_id
        resolve(started)
        await marked
        # Its context is aborted too, so pass one that is not: the admission itself must refuse.
        try:
            await api.execute_tool("echo", {"text": "a"}, context)
        except Exception as error:
            seen["rejected"] = error
        return {}

    add_tool(setup.registry, tool("late", late))
    setup.faux.set_responses([call("late"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await started

    async def mark(tx: Any) -> None:
        record = await tx.task(seen["caller"])
        tx.set_task({**record, "abortRequested": True})

    await harness.commit(mark, context)
    resolve(marked)
    await submission.wait(context)
    message = str(seen.get("rejected"))
    assert "abort mark" in message or "aborted" in message or "has settled" in message
    assert await harness.snapshot(NestedCallDoc, seen["caller"], "1", context) is None
    await harness.close(context)


async def test_runs_a_nested_call_as_its_own_tool_task_and_returns_its_result_to_the_caller_not_the_transcript() -> (
    None
):
    setup = chat_setup()
    add_tool(setup.registry, echo_tool().registration)
    seen: dict[str, Any] = {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        seen["nested"] = await api.execute_tool("echo", {"text": "a"}, call_context)
        return output(f"got {text(seen['nested'])}")

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert ran.status == "done"
    parent, child = await tool_tasks(ran.harness)
    nested = seen["nested"]
    assert isinstance(nested.get("durationMs"), (int, float))
    assert {key: value for key, value in nested.items() if key != "durationMs"} == {
        "taskId": child["id"],
        "structuredOutput": "echo a",
        "isError": False,
        "diagnostics": [],
    }
    # One result entry: the model-issued call's, which records nothing of its nested calls.
    assert [(result.tool_call_id, text(result)) for result in results(ran.entries)] == [("c1", "got echo a")]
    assert results(ran.entries)[0].nested_calls is None
    assert match_object(parent["input"], {"kind": "model", "callId": "c1"})
    assert child["owner"] == parent["id"]
    assert child["input"] == {
        "kind": "nested",
        "parent": parent["id"],
        "parentCallId": "c1",
        "key": "1",
        "call": {"type": "toolCall", "id": "c1/1", "name": "echo", "arguments": {"text": "a"}},
    }
    # The receipt stays small: the result lived in the caller's documents, which retired with it.
    assert child["state"] == {"status": "terminal", "outcome": {"status": "completed", "result": {"kind": "nested"}}}
    assert await ran.harness.snapshot(NestedCallDoc, parent["id"], "1", context) is None
    assert await ran.harness.snapshot(LiveDoc, ran.root.id, context) == {}
    await ran.harness.close(context)


async def test_gives_each_nested_call_its_own_task_and_call_id() -> None:
    setup = chat_setup()
    seen: list[tuple[Any, str]] = []

    async def who(_args: Any, api: Any, _call_context: Context) -> Any:
        seen.append((api.task_id, api.call_id))
        return {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await asyncio.gather(
            api.execute_tool("who", {}, call_context, key="a"),
            api.execute_tool("who", {}, call_context, key="b"),
        )
        return {}

    add_tool(setup.registry, tool("who", who))
    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert sorted(call_id for _task, call_id in seen) == ["c1/a", "c1/b"]
    assert len({task for task, _call in seen}) == 2
    await ran.harness.close(context)


async def test_runs_the_tool_hooks_on_nested_calls_which_see_their_parent() -> None:
    setup = chat_setup()
    echo = echo_tool()
    add_tool(setup.registry, echo.registration)
    parents: list[Any] = []

    def before_tool(hook_call: Any, *_rest: Any) -> Any:
        parents.append(hook_call.parent)
        if hook_call.parent is not None and hook_call.arguments.get("text") == "secret":
            return {"block": "no secrets"}
        return None

    def after_tool(hook_call: Any, result: Any, *_rest: Any) -> Any:
        return None if hook_call.parent is None else {**result, "output": [TextContent("rewritten")]}

    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool, "after_tool": after_tool})
    nested: list[Any] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        nested.append(await api.execute_tool("echo", {"text": "secret"}, call_context))
        nested.append(await api.execute_tool("echo", {"text": "public"}, call_context))
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    [parent] = (await tool_tasks(ran.harness))[:1]
    assert parents == [None, {"taskId": parent["id"], "callId": "c1"}, {"taskId": parent["id"], "callId": "c1"}]
    assert match_object(
        nested[0],
        {
            "isError": True,
            "diagnostics": [{"severity": "error", "code": "blocked", "message": "Tool call blocked: no secrets"}],
        },
    )
    assert text(nested[1]) == "rewritten"
    assert echo.state["runs"] == 1
    await ran.harness.close(context)


async def test_returns_an_error_result_for_an_unavailable_tool_and_invalid_arguments() -> None:
    setup = chat_setup()
    add_tool(setup.registry, echo_tool().registration)
    nested: list[Any] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        nested.append(await api.execute_tool("missing", {}, call_context))
        nested.append(await api.execute_tool("echo", {"text": {"not": "a string"}}, call_context))
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert [(result["isError"], result["diagnostics"][0].get("code")) for result in nested] == [
        (True, "tool_unavailable"),
        (True, "invalid_arguments"),
    ]
    await ran.harness.close(context)


async def test_returns_the_call_a_key_already_names_and_rejects_reusing_the_key_for_another_call() -> None:
    setup = chat_setup()
    echo = echo_tool()
    add_tool(setup.registry, echo.registration)
    seen: dict[str, Any] = {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("echo", {"text": "a"}, call_context, key="k")
        seen["reused"] = await api.execute_tool("echo", {"text": "a"}, call_context, key="k")
        await api.execute_tool("echo", {"text": "b"}, call_context, key="k")
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert echo.state["runs"] == 1
    assert text(seen["reused"]) == "echo a"
    assert "Nested call c1/k was already made with another tool or other arguments" in text(results(ran.entries)[0])
    await ran.harness.close(context)


async def test_matches_a_reused_keys_arguments_regardless_of_key_order() -> None:
    setup = chat_setup()
    runs = [0]

    async def pair(args: Any, *_rest: Any) -> Any:
        runs[0] += 1
        return output(f"{args['a']}{args['b']}")

    add_tool(
        setup.registry,
        define_tool(
            name="pair",
            description="pair",
            parameters=Type.Object({"a": Type.String(), "b": Type.String()}),
            execute=pair,
        ),
    )
    nested: list[Any] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        nested.append(await api.execute_tool("pair", {"a": "x", "b": "y"}, call_context, key="k"))
        nested.append(await api.execute_tool("pair", {"b": "y", "a": "x"}, call_context, key="k"))
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert runs[0] == 1
    assert [text(result) for result in nested] == ["xy", "xy"]
    await ran.harness.close(context)


async def test_aborts_the_nested_calls_a_caller_left_running_when_it_returns_keeping_their_details() -> None:
    setup = chat_setup()
    running = deferred()

    async def hang(_args: Any, api: Any, call_context: Context) -> Any:
        api.output("partial\n")
        await api.details({"step": 1}, call_context)
        resolve(running)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        detached(api.execute_tool("hang", {}, call_context))
        await running
        return output("returned early")

    add_tool(setup.registry, tool("hang", hang))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    stream = await watch_events(harness, root.id, context)
    ends: list[Any] = []

    async def listener(batch: Any, _context: Context) -> None:
        ends.extend(ends_of(batch))

    stream.start(listener)
    settled = await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert settled["status"] == "done"
    await harness.wait_for_idle(context)
    await stream.stop()
    [result] = results(await all_entries(root))
    assert text(result) == "returned early"
    _parent, child = await tool_tasks(harness)
    assert match_object(
        child["state"], {"status": "terminal", "outcome": {"status": "aborted", "result": {"kind": "nested"}}}
    )
    # The aborted call's result keeps its details; programs get no partial output, which only the model read.
    child_end = next(event for event in ends if event.get("toolCallId") == "c1/1")
    assert match_object(child_end, {"result": {"isError": True, "details": {"step": 1}}})
    assert "structuredOutput" not in child_end["result"]
    # The nested call ends, with its result, before its caller.
    assert [(event.get("toolCallId"), "result" in event) for event in ends] == [("c1/1", True), ("c1", False)]
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    await harness.close(context)


async def test_records_a_nested_calls_usage_under_its_tool() -> None:
    setup = chat_setup()

    async def paid(*_args: Any) -> Any:
        return {"usage": USAGE}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("paid", {}, call_context)
        return {}

    add_tool(setup.registry, tool("paid", paid))
    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert ((await ran.harness.snapshot(UsageDoc, ran.root.id, context)) or {}).get("tools") == {
        "paid": usage_json(USAGE)
    }
    await ran.harness.close(context)


async def test_reports_nested_calls_in_pi_live_nested_tools_and_as_tool_events_with_their_parent() -> None:
    setup = chat_setup()
    release = deferred()
    running = deferred()

    async def slow(_args: Any, api: Any, call_context: Context) -> Any:
        api.output("working\n")
        await api.details({"step": 1}, call_context)
        resolve(running)
        await release
        return output("slow done")

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        result = await api.execute_tool("slow", {}, call_context)
        return output(text(result))

    add_tool(setup.registry, tool("slow", slow))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    stream = await watch_events(harness, root.id, context)
    assert stream.snapshot["nestedTools"] == []
    events: list[Any] = []

    async def listener(batch: Any, _context: Context) -> None:
        events.extend(batch)

    stream.start(listener)
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await running
    live: Any = await harness.snapshot(LiveDoc, root.id, context)
    [slot] = live["nestedTools"]
    assert isinstance(slot["parentTaskId"], int) and isinstance(slot["taskId"], int)
    assert {key: value for key, value in slot.items() if key not in ("parentTaskId", "taskId")} == {
        "callId": "c1/1",
        "parentCallId": "c1",
        "name": "slow",
        "arguments": {},
        "status": "running",
        "output": "working\n",
        "details": {"step": 1},
    }
    resolve(release)
    await submission.wait(context)
    await harness.wait_for_idle(context)
    await stream.stop()
    tools = [
        (event["type"], event["toolCallId"], event.get("parentToolCallId"))
        for event in events
        if event["type"] in ("tool_execution_start", "tool_execution_end")
    ]
    assert tools == [
        ("tool_execution_start", "c1", None),
        ("tool_execution_start", "c1/1", "c1"),
        ("tool_execution_end", "c1/1", "c1"),
        ("tool_execution_end", "c1", None),
    ]
    nested_end = next(
        event for event in events if event["type"] == "tool_execution_end" and event["toolCallId"] == "c1/1"
    )
    assert match_object(nested_end, {"result": {"structuredOutput": "slow done", "isError": False}})
    # Events name the call's task and, for a nested call, the calling one.
    parent, child = await tool_tasks(harness)
    parent_end = next(
        event for event in events if event["type"] == "tool_execution_end" and event["toolCallId"] == "c1"
    )
    assert match_object(nested_end, {"taskId": child["id"], "parentTaskId": parent["id"]})
    assert match_object(parent_end, {"taskId": parent["id"]})
    assert "parentTaskId" not in parent_end
    assert any(
        event["type"] == "tool_execution_update"
        and event["toolCallId"] == "c1/1"
        and event.get("parentToolCallId") == "c1"
        for event in events
    )
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    await harness.close(context)


async def _hang(_args: Any, _api: Any, call_context: Context) -> Any:
    return await aborted(call_context.abort_signal)  # type: ignore[arg-type]


async def test_aborts_nested_calls_with_their_caller() -> None:
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
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await running
    await root.abort(context)
    assert (await submission.wait(context))["status"] == "unanswered"
    parent, child = await tool_tasks(harness)

    def outcome(record: Any) -> Any:
        return record["state"]["outcome"] if record["state"]["status"] == "terminal" else None

    assert match_object(outcome(child), {"status": "aborted", "result": {"kind": "nested"}})
    assert match_object(outcome(parent), {"status": "aborted", "result": {"kind": "model"}})
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    await harness.close(context)


async def test_aborts_a_running_nested_call_when_its_caller_throws() -> None:
    setup = chat_setup()
    running = deferred()

    async def hang(_args: Any, _api: Any, call_context: Context) -> Any:
        resolve(running)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        detached(api.execute_tool("hang", {}, call_context))
        await running
        raise RuntimeError("caller failed")

    add_tool(setup.registry, tool("hang", hang))
    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert ran.status == "done"
    assert "caller failed" in text(results(ran.entries)[0])
    _parent, child = await tool_tasks(ran.harness)
    assert match_object(child["state"], {"status": "terminal", "outcome": {"status": "aborted"}})
    await ran.harness.close(context)


async def test_reattaches_a_replay_safe_caller_to_the_nested_calls_it_made_before_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    echo = echo_tool()
    add_tool(setup.registry, echo.registration)
    blocked = deferred()
    runs = [0]

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        runs[0] += 1
        first = await api.execute_tool("echo", {"text": "a"}, call_context)
        if runs[0] == 1:
            resolve(blocked)
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        # A call new to the rerun: its slot, after the reattached one's, is found and finishes.
        second = await api.execute_tool("echo", {"text": "b"}, call_context)
        return output(f"run {runs[0]}: {text(first)}, {text(second)}")

    add_tool(setup.registry, tool("batch", batch, replay="safe"))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await blocked
    await opened.harness.close(context)

    opened = await open_chat(await open_sqlite_storage(path), setup)
    opened.harness.resume()
    assert (await (await opened.harness.submission(id, context)).wait(context))["status"] == "done"
    assert runs[0] == 2
    assert echo.state["runs"] == 2
    assert text(results(await all_entries(opened.root))[0]) == "run 2: echo a, echo b"
    tasks = await tool_tasks(opened.harness)
    assert len([task for task in tasks if task["input"]["kind"] == "nested"]) == 2
    outcome = tasks[1]["state"]["outcome"] if tasks[1]["state"]["status"] == "terminal" else None
    assert (outcome or {}).get("result", {}).get("kind") == "nested"
    await opened.harness.close(context)


async def test_stops_only_the_wait_when_a_caller_cancels_it_the_call_runs_on_and_its_key_returns_its_result() -> None:
    setup = chat_setup()
    cancel = AbortController()
    release = deferred()
    attempts: list[str] = []
    seen: dict[str, Any] = {}

    async def slow(*_args: Any) -> Any:
        cancel.abort(RuntimeError("cancelled"))
        await release
        return output("slow done")

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        cancelled = with_abort_signal(cancel.signal, call_context)
        try:
            await api.execute_tool("slow", {}, cancelled, key="same")
            attempts.append("first done")
        except Exception:
            attempts.append("first cancelled")
        resolve(release)
        seen["retry"] = await api.execute_tool("slow", {}, call_context, key="same")
        return {}

    add_tool(setup.registry, tool("slow", slow))
    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert attempts == ["first cancelled"]
    assert match_object(seen["retry"], {"isError": False, "structuredOutput": "slow done"})
    assert len(await tool_tasks(ran.harness)) == 2
    await ran.harness.close(context)


async def test_admits_one_nested_call_for_a_key_used_by_concurrent_calls() -> None:
    setup = chat_setup()
    echo = echo_tool()
    add_tool(setup.registry, echo.registration)
    both: list[Any] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        both.extend(
            await asyncio.gather(
                api.execute_tool("echo", {"text": "a"}, call_context, key="k"),
                api.execute_tool("echo", {"text": "a"}, call_context, key="k"),
            )
        )
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert echo.state["runs"] == 1
    assert [text(result) for result in both] == ["echo a", "echo a"]
    assert len(await tool_tasks(ran.harness)) == 2
    await ran.harness.close(context)


async def test_aborts_left_running_nested_calls_of_a_caller_interrupted_while_it_cleaned_them_up(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    running = deferred()
    cleaning = deferred()
    release = deferred()
    hang_runs = [0]

    async def hang(_args: Any, api: Any, call_context: Context) -> Any:
        hang_runs[0] += 1
        api.output("partial\n")
        await api.details({"step": 1}, call_context)
        resolve(running)
        try:
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        except BaseException:
            pass
        # The caller's cleanup waits for this abort; hold it there until the Harness closes.
        resolve(cleaning)
        await release
        raise RuntimeError("aborted")

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        detached(api.execute_tool("hang", {}, call_context))
        await running
        return output("returned early")

    add_tool(setup.registry, tool("hang", hang))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    id = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await cleaning
    closed = opened.harness.close(context)
    resolve(release)
    await closed

    opened = await open_chat(await open_sqlite_storage(path), setup)
    opened.harness.resume()
    assert (await (await opened.harness.submission(id, context)).wait(context))["status"] == "done"
    await opened.harness.wait_for_idle(context)
    # The caller is replay-unsafe: it reports the interruption, and its cleanup aborts the nested call.
    [result] = results(await all_entries(opened.root))
    assert (result.tool_call_id, result.is_error) == ("c1", True)
    assert "Tool batch was interrupted" in text(result)
    _parent, child = await tool_tasks(opened.harness)
    assert match_object(child["state"], {"status": "terminal", "outcome": {"status": "aborted"}})
    assert hang_runs[0] == 1
    assert await opened.harness.snapshot(LiveDoc, opened.root.id, context) == {}
    await opened.harness.close(context)


async def test_aborts_a_nested_call_whose_admission_was_still_committing_when_its_caller_returned() -> None:
    setup = chat_setup()
    add_tool(setup.registry, echo_tool().registration)

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("echo", {"text": "warm"}, call_context, key="warm")
        detached(api.execute_tool("hang", {}, call_context, key="late"))
        return {}

    add_tool(setup.registry, tool("hang", _hang))
    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert ran.status == "done"
    late = next(
        task
        for task in await tool_tasks(ran.harness)
        if task["input"]["kind"] == "nested" and task["input"]["call"]["name"] == "hang"
    )
    assert match_object(late["state"], {"status": "terminal", "outcome": {"status": "aborted"}})
    assert await ran.harness.snapshot(LiveDoc, ran.root.id, context) == {}
    await ran.harness.close(context)


async def test_aborts_every_left_running_nested_call_before_waiting_for_any() -> None:
    setup = chat_setup()
    b_aborted = deferred()
    started = {"a": deferred(), "b": deferred()}

    # A only stops once B was told to stop.
    async def a(*_args: Any) -> Any:
        resolve(started["a"])
        await b_aborted
        return {}

    async def b(_args: Any, _api: Any, call_context: Context) -> Any:
        resolve(started["b"])
        try:
            await aborted(call_context.abort_signal)  # type: ignore[arg-type]
        except BaseException:
            pass
        resolve(b_aborted)
        raise RuntimeError("aborted")

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        detached(api.execute_tool("a", {}, call_context, key="a"))
        detached(api.execute_tool("b", {}, call_context, key="b"))
        await asyncio.gather(started["a"], started["b"])
        return {}

    add_tool(setup.registry, tool("a", a))
    add_tool(setup.registry, tool("b", b))
    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert ran.status == "done"
    nested = [task for task in await tool_tasks(ran.harness) if task["input"]["kind"] == "nested"]
    assert [task["state"]["status"] for task in nested] == ["terminal", "terminal"]
    await ran.harness.close(context)


async def test_rejects_explicit_keys_that_are_empty_have_a_slash_are_proto_or_are_positive_integers() -> None:
    setup = chat_setup()
    add_tool(setup.registry, echo_tool().registration)
    errors: list[str] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        for key in ["", "a/b", "__proto__", "7"]:
            try:
                await api.execute_tool("echo", {}, call_context, key=key)
            except Exception as error:
                errors.append(str(error))
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    rule = 'must be non-empty, without "/", not "__proto__", and not a positive integer'
    assert errors == [
        f'Nested call key "" {rule}',
        f'Nested call key "a/b" {rule}',
        f'Nested call key "__proto__" {rule}',
        f'Nested call key "7" {rule}',
    ]
    await ran.harness.close(context)


async def test_runs_nested_calls_of_nested_calls_and_ends_them_children_first() -> None:
    setup = chat_setup()
    add_tool(setup.registry, echo_tool().registration)

    async def inner(_args: Any, api: Any, call_context: Context) -> Any:
        result = await api.execute_tool("echo", {"text": "deep"}, call_context, key="x")
        return output(text(result))

    async def outer(_args: Any, api: Any, call_context: Context) -> Any:
        result = await api.execute_tool("inner", {}, call_context)
        return output(text(result))

    add_tool(setup.registry, tool("inner", inner))
    add_tool(setup.registry, tool("outer", outer))
    setup.faux.set_responses([call("outer"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    stream = await watch_events(harness, root.id, context)
    tools: list[tuple[Any, ...]] = []

    async def listener(batch: Any, _context: Context) -> None:
        for event in batch:
            if event["type"] in ("tool_execution_start", "tool_execution_end"):
                tools.append((event["type"], event["toolCallId"], event.get("parentToolCallId")))

    stream.start(listener)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    await harness.wait_for_idle(context)
    await stream.stop()
    assert text(results(await all_entries(root))[0]) == "echo deep"
    outer_task, inner_task, deep = await tool_tasks(harness)
    assert [inner_task["owner"], deep["owner"]] == [outer_task["id"], inner_task["id"]]
    assert tools == [
        ("tool_execution_start", "c1", None),
        ("tool_execution_start", "c1/1", "c1"),
        ("tool_execution_start", "c1/1/x", "c1/1"),
        ("tool_execution_end", "c1/1/x", "c1/1"),
        ("tool_execution_end", "c1/1", "c1"),
        ("tool_execution_end", "c1", None),
    ]
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    await harness.close(context)


async def test_reports_the_output_of_several_running_nested_calls_and_shows_them_to_a_late_subscriber() -> None:
    setup = chat_setup()
    release = deferred()
    running = {"one": deferred(), "two": deferred()}
    wrote = {"one": deferred(), "two": deferred()}

    async def slow(args: Any, api: Any, call_context: Context) -> Any:
        name = args["text"]
        resolve(running[name])
        await wrote[name]
        api.output(f"{name}\n")
        await api.details({"name": name}, call_context)
        await release
        return {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await asyncio.gather(
            api.execute_tool("slow", {"text": "one"}, call_context),
            api.execute_tool("slow", {"text": "two"}, call_context),
        )
        return {}

    add_tool(setup.registry, tool("slow", slow))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await asyncio.gather(running["one"], running["two"])
    stream = await watch_events(harness, root.id, context)
    assert [(slot["callId"], slot["status"]) for slot in stream.snapshot["nestedTools"]] == [
        ("c1/1", "running"),
        ("c1/2", "running"),
    ]
    updates: list[tuple[str, Any]] = []

    async def listener(batch: Any, _context: Context) -> None:
        for event in batch:
            if event["type"] == "tool_execution_update" and event.get("output") is not None:
                updates.append((event["toolCallId"], event["output"]))

    stream.start(listener)
    # The second slot writes first, so a wrong slot index in the update path would show.
    resolve(wrote["two"])
    await eventually(lambda: len(updates) == 1)
    resolve(wrote["one"])
    await eventually(lambda: len(updates) == 2)
    resolve(release)
    await submission.wait(context)
    await stream.stop()
    assert updates == [("c1/2", {"set": "two\n"}), ("c1/1", {"set": "one\n"})]
    await harness.close(context)


async def test_commits_only_the_status_of_a_nested_call_made_with_progress_false_and_keeps_its_output_in_the_result() -> (
    None
):
    setup = chat_setup()
    running = deferred()
    release = deferred()
    seen: dict[str, Any] = {}

    async def chatty(_args: Any, api: Any, call_context: Context) -> Any:
        api.output("line\n")
        api.diagnostic({"severity": "info", "message": "note"})
        await api.details({"step": 1}, call_context)
        resolve(running)
        await release
        return {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        seen["result"] = await api.execute_tool("chatty", {}, call_context, progress=False)
        return {}

    add_tool(setup.registry, tool("chatty", chatty))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "go"}, context)
    # `details()` resolved before this, so a progress commit would have landed already.
    await running
    live: Any = await harness.snapshot(LiveDoc, root.id, context)
    [slot] = live["nestedTools"]
    assert isinstance(slot["parentTaskId"], int) and isinstance(slot["taskId"], int)
    assert {key: value for key, value in slot.items() if key not in ("parentTaskId", "taskId")} == {
        "callId": "c1/1",
        "parentCallId": "c1",
        "name": "chatty",
        "arguments": {},
        "status": "running",
    }
    resolve(release)
    await submission.wait(context)
    assert match_object(
        seen["result"],
        {
            "structuredOutput": "line\n",
            "details": {"step": 1},
            "diagnostics": [{"severity": "info", "message": "note"}],
        },
    )
    await harness.close(context)


async def test_loses_the_running_output_of_a_nested_call_made_with_progress_false_when_it_is_aborted() -> None:
    setup = chat_setup()
    running = deferred()

    async def chatty(_args: Any, api: Any, call_context: Context) -> Any:
        api.output("line\n")
        api.diagnostic({"severity": "info", "message": "note"})
        await api.details({"step": 1}, call_context)
        resolve(running)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        detached(api.execute_tool("chatty", {}, call_context, progress=False))
        await running
        return {}

    add_tool(setup.registry, tool("chatty", chatty))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    stream = await watch_events(harness, root.id, context)
    ends: list[Any] = []

    async def listener(batch: Any, _context: Context) -> None:
        ends.extend(ends_of(batch))

    stream.start(listener)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    await harness.wait_for_idle(context)
    await stream.stop()
    # The abort's result is built from the slot, which never got the output.
    end = next(event for event in ends if event.get("toolCallId") == "c1/1")
    result = end["result"]
    assert result["isError"] is True
    assert "structuredOutput" not in result
    assert "details" not in result
    assert [diagnostic.get("code") for diagnostic in result["diagnostics"]] == ["aborted"]
    await harness.close(context)


async def test_shows_a_nested_calls_arguments_in_its_slot_as_made_then_unbounded_as_it_runs_with_them() -> None:
    setup = chat_setup()
    hook_entered = deferred()
    hook_release = deferred()
    running = deferred()
    release = deferred()
    large = "x" * (64 * 1024)

    async def slow(*_args: Any) -> Any:
        resolve(running)
        await release
        return {}

    async def before_tool(hook_call: Any, *_rest: Any) -> Any:
        if hook_call.parent is None:
            return None
        resolve(hook_entered)
        await hook_release
        return {"arguments": {"text": large}}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("slow", {"text": "asked"}, call_context)
        return {}

    add_tool(setup.registry, tool("slow", slow))
    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    stream = await watch_events(harness, root.id, context)
    starts: list[Any] = []

    async def listener(batch: Any, _context: Context) -> None:
        starts.extend(event for event in batch if event["type"] == "tool_execution_start")

    stream.start(listener)
    submission = await root.submit({"type": "input", "content": "go"}, context)

    async def slot() -> Any:
        return ((await harness.snapshot(LiveDoc, root.id, context)) or {})["nestedTools"][0]

    await hook_entered
    assert match_object(await slot(), {"status": "pending", "arguments": {"text": "asked"}})
    resolve(hook_release)
    await running
    assert match_object(await slot(), {"status": "running", "arguments": {"text": large}})
    resolve(release)
    await submission.wait(context)
    await harness.wait_for_idle(context)
    await stream.stop()
    # The start event carries the arguments the call runs with; the task input keeps the ones it was made with.
    start = next(event for event in starts if event["toolCallId"] == "c1/1")
    assert match_object(start, {"args": {"text": large}})
    nested = next(task for task in await tool_tasks(harness) if task["input"]["kind"] == "nested")
    assert match_object(nested["input"], {"call": {"arguments": {"text": "asked"}}})
    await harness.close(context)


async def test_runs_a_nested_call_whose_repaired_arguments_set_an_optional_property_to_undefined() -> None:
    setup = chat_setup()
    seen: list[Any] = []

    async def repaired(args: Any, *_rest: Any) -> Any:
        seen.append(args)
        return {}

    add_tool(setup.registry, tool("repaired", repaired, prepare_arguments=lambda _args: {"text": UNDEFINED}))
    found: dict[str, Any] = {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        found["result"] = await api.execute_tool("repaired", {"text": "x"}, call_context)
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert match_object(found["result"], {"isError": False})
    assert seen == [{}]
    await ran.harness.close(context)


async def test_finds_each_nested_calls_slot_after_slots_in_the_middle_were_removed() -> None:
    setup = chat_setup()
    add_tool(setup.registry, echo_tool().registration)

    async def inner(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("echo", {"text": "m1"}, call_context)
        await api.execute_tool("echo", {"text": "m2"}, call_context)
        return {}

    before = deferred()
    checked = deferred()
    slots: list[Any] = []

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("echo", {"text": "n1"}, call_context)
        # n2 makes m1 and m2 and settles, which removes them from between n2 and n3.
        await api.execute_tool("inner", {}, call_context)
        await api.execute_tool("echo", {"text": "n3"}, call_context)
        await api.execute_tool("echo", {"text": "n4"}, call_context)
        live = await api.snapshot(LiveDoc, api.conversation_id, call_context)
        slots.extend(
            {"callId": slot["callId"], "status": slot["status"], "summary": slot.get("summary")}
            for slot in (live or {}).get("nestedTools") or []
        )
        resolve(before)
        await checked
        return {}

    add_tool(setup.registry, tool("inner", inner))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    submission = await opened.root.submit({"type": "input", "content": "go"}, context)
    await before
    assert [slot["callId"] for slot in slots] == ["c1/1", "c1/2", "c1/3", "c1/4"]
    for slot in slots:
        assert slot["status"] == "done"
        assert set(slot["summary"]) == {"isError", "durationMs"}
        assert slot["summary"]["isError"] is False and isinstance(slot["summary"]["durationMs"], (int, float))
    resolve(checked)
    await submission.wait(context)
    await opened.harness.close(context)


async def test_gives_callers_a_result_with_its_task_id_and_only_diagnostics_for_a_nested_call_the_scheduler_faulted() -> (
    None
):
    setup = chat_setup()

    # Details that are not JSON make the result commit throw, so the task faults and stores no result.
    async def echo(*_args: Any) -> Any:
        return {"details": {"count": object()}}

    add_tool(setup.registry, tool("echo", echo))
    seen: dict[str, Any] = {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        seen["result"] = await api.execute_tool("echo", {"text": "a"}, call_context)
        return {}

    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    _parent, child = await tool_tasks(harness)
    assert match_object(child["state"], {"status": "terminal", "outcome": {"status": "faulted"}})
    result = seen["result"]
    assert set(result) == {"taskId", "isError", "diagnostics"}
    assert result["taskId"] == child["id"] and result["isError"] is True
    [diagnostic] = result["diagnostics"]
    assert {key: value for key, value in diagnostic.items() if key != "message"} == {
        "severity": "error",
        "code": "faulted",
    }
    assert "Tool echo failed" in diagnostic["message"]
    await harness.close(context)


async def test_gives_callers_what_a_schema_less_tool_streamed_before_it_threw() -> None:
    setup = chat_setup()

    async def fails(_args: Any, api: Any, _call_context: Context) -> Any:
        api.output("half way\n")
        raise RuntimeError("broke")

    add_tool(setup.registry, tool("fails", fails))
    seen: dict[str, Any] = {}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        seen["result"] = await api.execute_tool("fails", {}, call_context)
        return {}

    add_tool(setup.registry, tool("batch", batch))
    ran = await run_once(setup, [call("batch"), DONE])
    assert match_object(
        seen["result"],
        {
            "isError": True,
            "structuredOutput": "half way\n",
            "diagnostics": [{"code": "tool_error", "message": "broke"}],
        },
    )
    await ran.harness.close(context)


async def test_shows_a_finished_nested_call_as_a_done_slot_with_its_summary_until_its_caller_settles() -> None:
    setup = chat_setup()
    release = deferred()
    checked = deferred()

    # Diagnostics win over output for the error text, which is cut to 500 characters.
    async def verbose(*_args: Any) -> Any:
        return {
            "output": [TextContent("ignored")],
            "isError": True,
            "diagnostics": [{"severity": "error", "message": "e" * 600}],
        }

    async def paid(*_args: Any) -> Any:
        return {"usage": USAGE}

    # An error with mixed content: programs get the list, the summary its text.
    async def broken(*_args: Any) -> Any:
        return {"output": [TextContent("bad image"), ImageContent("AAAA", "image/png")], "isError": True}

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        await api.execute_tool("echo", {"text": "a"}, call_context)
        await api.execute_tool("missing", {}, call_context)
        await api.execute_tool("paid", {}, call_context)
        await api.execute_tool("broken", {}, call_context)
        await api.execute_tool("verbose", {}, call_context)
        resolve(checked)
        await release
        return {}

    add_tool(setup.registry, tool("verbose", verbose))
    add_tool(setup.registry, echo_tool().registration)
    add_tool(setup.registry, tool("paid", paid))
    add_tool(setup.registry, tool("broken", broken))
    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await checked
    slots = ((await harness.snapshot(LiveDoc, root.id, context)) or {})["nestedTools"]
    shown = [{"callId": slot["callId"], "status": slot["status"], "summary": dict(slot["summary"])} for slot in slots]
    for slot in shown:
        duration = slot["summary"].pop("durationMs", None)
        assert duration is None or isinstance(duration, (int, float))
    assert shown == [
        {"callId": "c1/1", "status": "done", "summary": {"isError": False}},
        {"callId": "c1/2", "status": "done", "summary": {"isError": True, "error": "Tool missing is not available"}},
        {"callId": "c1/3", "status": "done", "summary": {"isError": False, "usage": usage_json(USAGE)}},
        {"callId": "c1/4", "status": "done", "summary": {"isError": True, "error": "bad image"}},
        {"callId": "c1/5", "status": "done", "summary": {"isError": True, "error": "e" * 500}},
    ]
    assert ["durationMs" in slot.get("summary", {}) for slot in slots] == [True, False, True, True, True]
    resolve(release)
    await submission.wait(context)
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    await harness.close(context)


def test_migrates_version_1_tool_task_input_to_a_model_issued_call() -> None:
    migrate = ToolTask.definition.migrate
    assert migrate is not None
    assert migrate({"assistant": 3, "callId": "c1"}, {"phase": "call"}, 1) == {
        "input": {"kind": "model", "assistant": 3, "callId": "c1"},
        "checkpoint": {"phase": "call"},
    }


# ─── tool task version 1 records ────────────────────────────────────────────


async def store_version_1(path: Path, setup: ChatSetup, checkpoint: Any) -> Any:
    """Store a live tool task for call ``c1`` of an assistant entry, close, and rewrite its record as version 1
    stored it: ``{assistant, callId}`` input at version 1, with ``checkpoint``. Returns the task ID."""
    # No submission, so the Harness never starts scheduling and the task stays pending.
    opened = await open_chat(await open_sqlite_storage(path), setup)
    root = opened.root

    async def create(tx: Any) -> Any:
        entry = await tx.append_entry(AssistantEntry, root.id, {"model": [call("echo", {"text": "old"})]})
        return await tx.create_task(
            ToolTask,
            {"kind": "model", "assistant": entry["id"], "callId": "c1"},
            {"ownership": {"kind": "conversation"}},
        )

    id = await root.commit(create, context)
    await opened.harness.close(context)
    storage = await open_sqlite_storage(path)
    record: Any = await storage.task(id, context)
    input = {key: value for key, value in record["input"].items() if key != "kind"}
    state = {**record["state"], "checkpoint": checkpoint}
    await storage.commit([{"type": "task", "value": {**record, "version": 1, "input": input, "state": state}}], context)
    await storage.close(context)
    return id


async def test_migrates_a_pending_version_1_task_and_runs_it_as_a_model_issued_call(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    echo = echo_tool()
    add_tool(setup.registry, echo.registration)
    id = await store_version_1(path, setup, {"phase": "call"})
    opened = await open_chat(await open_sqlite_storage(path), setup)
    harness, root = opened.harness, opened.root
    assert ((await harness.get_task(id, context)) or {}).get("version") == 1
    harness.resume()
    settled = await harness.wait_for_task(id, context)
    assert settled["version"] == 2
    assert match_object(settled["input"], {"kind": "model", "callId": "c1"})
    assert match_object(settled["state"]["outcome"], {"status": "completed", "result": {"kind": "model"}})
    assert echo.state["runs"] == 1
    [result] = results(await all_entries(root))
    assert [result.tool_call_id, text(result)] == ["c1", "echo old"]
    await harness.close(context)


async def test_migrates_a_version_1_task_interrupted_after_intent_and_settles_it_interrupted(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    echo = echo_tool()
    add_tool(setup.registry, echo.registration)
    intent = {"phase": "execute", "arguments": {"text": "old"}, "replay": "unsafe"}
    id = await store_version_1(path, setup, intent)
    opened = await open_chat(await open_sqlite_storage(path), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    settled = await harness.wait_for_task(id, context)
    assert settled["version"] == 2
    assert match_object(settled["state"]["outcome"], {"status": "failed", "result": {"kind": "model"}})
    assert echo.state["runs"] == 0
    [result] = results(await all_entries(root))
    assert (result.tool_call_id, result.is_error) == ("c1", True)
    assert "Tool echo was interrupted" in text(result)
    await harness.close(context)
