"""Port of ``test/harness-live-deltas.test.ts``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from durable_harness_support import aborted, add_tool, chat_setup, open_chat, wait_for
from durable_session_support import context, document_changes

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_durable import Harness, LiveDoc, MemoryStorage, define_tool

Action = Callable[[Any], "Awaitable[None] | None"]

OUTPUT = ("tools", 0, "output")


def norm(op: Sequence[Any]) -> tuple[Any, ...]:
    """An op with its path as a tuple, for comparisons."""
    if op[0] == "r":
        return tuple(op)
    return (op[0], tuple(op[1]), *op[2:])


def norm_all(ops: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [norm(op) for op in ops]


def contains_all(ops: Sequence[Sequence[Any]], expected: Sequence[tuple[Any, ...]]) -> bool:
    """``expect.arrayContaining``."""
    normalized = norm_all(ops)
    return all(item in normalized for item in expected)


def is_output(op: Sequence[Any]) -> bool:
    return tuple(op[1]) == OUTPUT


def live_commits(harness: Harness) -> list[list[Any]]:
    """Operations of every ``pi.live`` commit with operations."""
    commits: list[list[Any]] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            if change["record"]["kind"] == "pi.live" and len(change["ops"]) > 0:
                commits.append(list(change["ops"]))

    harness.subscribe_commits(listener)
    return commits


async def live(harness: Harness) -> Any:
    return await harness.snapshot(LiveDoc, 1, context)  # type: ignore[arg-type]


class Driven:
    def __init__(self, harness: Harness, commits: list[list[Any]], push: Callable[[Any], None], submission: Any):
        self.harness = harness
        self.commits = commits
        self._push = push
        self._submission = submission

    async def step(self, action: Action) -> list[tuple[Any, ...]]:
        """Run one action inside the tool and return the operations of the commit it caused."""
        before = len(self.commits)

        async def run(api: Any) -> bool:
            result = action(api)
            if result is not None:
                await result
            return True

        self._push(run)
        await wait_for(lambda: len(self.commits) > before)
        return norm_all(self.commits[-1])

    async def finish(self) -> list[list[tuple[Any, ...]]]:
        """Let the tool return and the run finish; returns the commits made meanwhile."""
        before = len(self.commits)

        async def done(_api: Any) -> bool:
            return False

        self._push(done)
        await self._submission.wait(context)
        return [norm_all(ops) for ops in self.commits[before:]]


async def drive(output_limits: Any = None) -> Driven:
    """Drive one tool call step by step and capture the exact Chord operations of every ``pi.live`` commit."""
    setup = chat_setup()
    actions: list[Callable[[Any], Awaitable[bool]]] = []
    wake: list[asyncio.Future[None] | None] = [None]

    async def execute(_args: Any, api: Any, call_context: Any) -> Any:
        while True:
            while len(actions) == 0:
                woken: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                wake[0] = woken
                cancelled = asyncio.ensure_future(aborted(call_context.abort_signal))
                try:
                    await asyncio.wait([woken, cancelled], return_when=asyncio.FIRST_COMPLETED)
                finally:
                    if not cancelled.done():
                        cancelled.cancel()
                if cancelled.done() and not cancelled.cancelled():
                    cancelled.result()
            if not await actions.pop(0)(api):
                return {}

    add_tool(
        setup.registry,
        define_tool(
            name="drive",
            description="Driven by the test",
            parameters=Type.Object({}),
            output_limits=output_limits if output_limits is not None else {},
            execute=execute,
        ),
    )
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("drive", {}, id="c1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    commits = live_commits(harness)
    submission = await root.submit({"type": "input", "content": "go"}, context)

    async def running() -> bool:
        value = await live(harness)
        tools = None if value is None else value.get("tools")
        return bool(tools) and tools[0].get("status") == "running"

    await wait_for(running)

    def push(action: Callable[[Any], Awaitable[bool]]) -> None:
        actions.append(action)
        woken = wake[0]
        if woken is not None and not woken.done():
            woken.set_result(None)

    return Driven(harness, commits, push, submission)


def noop_tool(name: str = "noop") -> Any:
    async def execute(*_args: Any) -> Any:
        return {"output": []}

    return define_tool(name=name, description=name, parameters=Type.Object({}), execute=execute)


async def test_hands_a_generation_over_to_its_tool_round_and_starts_a_tool_with_one_field_write_each() -> None:
    setup = chat_setup()
    add_tool(setup.registry, noop_tool())
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("noop", {}, id="c1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    commits = live_commits(harness)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    tasks = await harness.commit(lambda tx: tx.scan_tasks({"conversationId": root.id}, 20), context)

    def ids(kind: str) -> list[Any]:
        return [task["id"] for task in tasks["items"] if task["kind"] == kind]

    first_generation, second_generation = sorted(ids("pi.generation"))
    [tool] = ids("pi.tool")
    entries = await root.entries({}, 10, None, context)
    result = next(entry for entry in entries["items"] if entry["kind"] == "pi.tool-result")["id"]
    assert len(commits) == 8
    normalized = [norm_all(ops) for ops in commits]
    # submission
    [[kind, path, value]] = normalized[0]
    assert (kind, path) == ("s", ("run",))
    assert value["taskId"] == first_generation and len(value["inputs"]) == 1 and isinstance(value["inputs"][0], int)
    # request
    assert normalized[1] == [("s", ("generation",), {"attempt": 1})]
    # the generation starts its tool round and keeps the run
    assert contains_all(
        commits[2],
        [
            ("d", ("generation",)),
            ("s", ("tools",), [{"callId": "c1", "name": "noop", "taskId": tool, "status": "pending"}]),
        ],
    )
    # intent
    assert normalized[3] == [("s", ("tools", 0, "status"), "running")]
    # result
    assert contains_all(commits[4], [("s", ("tools", 0, "status"), "done"), ("s", ("tools", 0, "entry"), result)])
    # the generation's tools phase hands the run to the next generation
    assert contains_all(commits[5], [("d", ("tools",)), ("s", ("run", "taskId"), second_generation)])
    assert normalized[6] == [("s", ("generation",), {"attempt": 1})]
    # the answer ends the run
    assert contains_all(commits[7], [("d", ("run",)), ("d", ("generation",))])
    assert len(commits[2]) == 2
    assert len(commits[4]) == 2
    assert len(commits[5]) == 2
    await harness.close(context)


async def test_appends_head_output_and_then_only_updates_the_dropped_counts_once_the_window_is_full() -> None:
    run = await drive({"maxLines": 2})
    assert await run.step(lambda api: api.output("one\n")) == [("s", OUTPUT, "one\n")]
    assert await run.step(lambda api: api.output("two\n")) == [("a", OUTPUT, "two\n")]
    # The window is full: the retained text stays; only the counts change.
    assert await run.step(lambda api: api.output("three\n")) == [
        ("s", ("tools", 0, "droppedBytes"), 6),
        ("s", ("tools", 0, "droppedLines"), 1),
    ]
    assert await run.step(lambda api: api.output("four\n")) == [
        ("s", ("tools", 0, "droppedBytes"), 11),
        ("s", ("tools", 0, "droppedLines"), 2),
    ]
    await run.finish()
    await run.harness.close(context)


async def test_slides_a_tail_window_as_a_front_trim_plus_an_append() -> None:
    run = await drive({"maxLines": 3, "retain": "tail"})
    assert await run.step(lambda api: api.output("line 1\nline 2\nline 3\n")) == [
        ("s", OUTPUT, "line 1\nline 2\nline 3\n")
    ]
    assert await run.step(lambda api: api.output("line 4\n")) == [
        ("t", OUTPUT, 7),
        ("a", OUTPUT, "line 4\n"),
        ("s", ("tools", 0, "droppedBytes"), 7),
        ("s", ("tools", 0, "droppedLines"), 1),
    ]
    # The buffer keeps only the window, and later slides stay minimal and exact.
    assert await run.step(lambda api: api.output("line 5\nline 6\n")) == [
        ("t", OUTPUT, 14),
        ("a", OUTPUT, "line 5\nline 6\n"),
        ("s", ("tools", 0, "droppedBytes"), 21),
        ("s", ("tools", 0, "droppedLines"), 3),
    ]
    assert (await live(run.harness))["tools"][0]["output"] == "line 4\nline 5\nline 6\n"
    await run.finish()
    await run.harness.close(context)


async def test_writes_the_whole_window_when_chords_overlap_search_cannot_find_the_shared_part() -> None:
    # A retained window beyond the 64 KiB overlap scan.
    wide = await drive({"maxBytes": 100 * 1024, "maxLines": 1_000_000, "retain": "tail"})

    def line(index: int) -> str:
        return f"{str(index).rjust(10, '0')} {'x' * 989}\n"

    text = "".join(line(index) for index in range(100))
    await wide.step(lambda api: api.output(text))
    slid = await wide.step(lambda api: api.output(line(100) + line(101) + line(102) + line(103)))
    assert [op[0] for op in slid if is_output(op)] == ["s"]
    await wide.finish()
    await wide.harness.close(context)

    # Repetitive output still finds an overlap here; Chord's bounded candidate search can give up on other inputs
    # and then writes one window.
    repetitive = await drive({"maxLines": 50, "retain": "tail"})
    await repetitive.step(lambda api: api.output("y\n" * 50))
    repeated = await repetitive.step(lambda api: api.output("z\n"))
    assert [op for op in repeated if is_output(op)] == [("t", OUTPUT, 2), ("a", OUTPUT, "z\n")]
    await repetitive.finish()
    await repetitive.harness.close(context)


async def test_diffs_details_leaf_by_leaf_and_appends_diagnostics() -> None:
    run = await drive()
    details = ("tools", 0, "details")
    assert await run.step(lambda api: api.details({"step": 1, "log": "a"}, context)) == [
        ("s", details, {"step": 1, "log": "a"})
    ]
    ops = await run.step(lambda api: api.details({"step": 2, "log": "ab"}, context))
    assert ("s", (*details, "step"), 2) in ops and ("a", (*details, "log"), "b") in ops
    assert await run.step(lambda api: api.details({"step": 2}, context)) == [("d", (*details, "log"))]
    diagnostics = ("tools", 0, "diagnostics")
    first = {"severity": "info", "message": "first"}
    second = {"severity": "warn", "message": "second"}
    assert await run.step(lambda api: api.diagnostic(first)) == [("s", diagnostics, [first])]
    assert await run.step(lambda api: api.diagnostic(second)) == [("p", diagnostics, 1, 0, [second])]
    # Settlement moves everything into the result entry and keeps the slot small.
    [settled, *_rest] = await run.finish()
    assert all(item in settled for item in [("s", ("tools", 0, "status"), "done"), ("d", details), ("d", diagnostics)])
    await run.harness.close(context)


async def test_streams_partial_text_as_appends() -> None:
    setup = chat_setup(tokens_per_second=400, token_size={"min": 4, "max": 4})
    setup.faux.set_responses([faux_assistant_message([faux_text("word " * 250)])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    commits = live_commits(harness)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    partials = [norm_all(ops) for ops in commits[2:-1]]
    assert len(partials) > 2
    [[kind, path, value]] = partials[0]
    assert (kind, path) == ("s", ("generation", "message")) and isinstance(value, dict)
    for ops in partials[1:]:
        [[kind, path, value]] = ops
        assert (kind, path) == ("a", ("generation", "message", "content", 0, "text")) and isinstance(value, str)
    await harness.close(context)


async def test_stores_a_complete_base_exactly_in_the_commits_where_nothing_runs() -> None:
    # Storage that remembers whether each commit wrote pi.live as a base or a delta.
    written: dict[int, str] = {}
    live_id: list[Any] = [None]

    class RecordingStorage(MemoryStorage):
        async def commit(self, writes: Any, context: Any) -> Any:
            kind = None
            for write in writes:
                if write["type"] == "document.change" and write["id"] == live_id[0]:
                    kind = write["content"]["kind"]
            seq = await super().commit(writes, context)
            if kind is not None:
                written[seq] = kind
            return seq

    setup = chat_setup()
    for name in ("first", "second"):

        def make(name: str) -> Any:
            async def execute(_args: Any, api: Any, _context: Any) -> Any:
                api.output(f"{name} output\n")
                await asyncio.sleep(0.15)
                api.output(f"{name} more\n")
                return {}

            return execute

        add_tool(
            setup.registry,
            define_tool(name=name, description=name, parameters=Type.Object({}), execute=make(name)),
        )
    setup.faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("first", {}, id="a"), faux_tool_call("second", {}, id="b")], stop_reason="toolUse"
            ),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    setup.settings["toolExecution"] = "sequential"
    opened = await open_chat(RecordingStorage(), setup)
    harness, root = opened.harness, opened.root
    values: dict[int, Any] = {}

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            if change["record"]["kind"] != "pi.live":
                continue
            live_id[0] = change["record"]["id"]
            if change["value"] is not None:
                values[publication["seq"]] = change["value"]

    harness.subscribe_commits(listener)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)

    def nothing_runs(value: Any) -> bool:
        return "generation" not in value and not any(slot["status"] == "running" for slot in value.get("tools", []))

    kinds: list[str] = []
    for seq, kind in written.items():
        assert (kind == "base") == nothing_runs(values[seq])
        kinds.append(kind)
    # Bases at the handover, after each sequential tool, after the round, and when the run ends.
    assert len([kind for kind in kinds if kind == "base"]) >= 5
    assert "delta" in kinds
    await harness.close(context)


async def test_starts_calls_the_request_did_not_offer_as_done_and_marks_a_faulted_tools_slot_done_without_an_entry() -> (
    None
):
    setup = chat_setup()

    async def bad(*_args: Any) -> Any:
        # Not strict JSON: the result commit throws and the scheduler faults the task.
        return {"output": [], "details": {"fn": lambda: 1}}

    add_tool(setup.registry, define_tool(name="bad", description="bad", parameters=Type.Object({}), execute=bad))
    setup.faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("ghost", {}, id="g"), faux_tool_call("bad", {}, id="b")], stop_reason="toolUse"
            ),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    commits = live_commits(harness)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    entries = await root.entries({}, 10, None, context)
    ghost_result = next(entry for entry in entries["items"] if entry["kind"] == "pi.tool-result")["id"]
    handover = next(
        norm_all(ops) for ops in commits if any(op[0] == "s" and next(iter(op[1])) == "tools" for op in ops)
    )
    slots = next(op[2] for op in handover if op[:2] == ("s", ("tools",)))
    assert slots[0] == {"callId": "g", "name": "ghost", "status": "done", "entry": ghost_result}
    assert {key: value for key, value in slots[1].items() if key != "taskId"} == {
        "callId": "b",
        "name": "bad",
        "status": "pending",
    }
    assert isinstance(slots[1]["taskId"], int)
    assert len(slots) == 2
    # The fault cleanup writes only the status.
    assert [("s", ("tools", 1, "status"), "done")] in [norm_all(ops) for ops in commits]
    await harness.close(context)


async def test_commits_the_tool_calling_answer_its_tool_tasks_the_generations_wait_and_the_tool_round_in_one_commit() -> (
    None
):
    setup = chat_setup()
    add_tool(setup.registry, noop_tool())
    setup.faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("noop", {}, id="a"), faux_tool_call("noop", {}, id="b")], stop_reason="toolUse"
            ),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    handover: list[Any] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            value = change["value"]
            if change["record"]["kind"] == "pi.live" and value is not None and len(value.get("tools", [])) == 2:
                if not handover:
                    handover.append(publication)

    harness.subscribe_commits(listener)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    changes = handover[0]["changes"]
    kinds = [change["value"]["kind"] for change in changes if change["type"] in ("entry", "task")]
    assert sorted(kinds) == ["pi.assistant", "pi.generation", "pi.tool", "pi.tool"]
    await harness.close(context)


async def test_writes_an_aborted_tools_slot_with_field_level_ops() -> None:
    run = await drive()
    await run.step(lambda api: api.output("partial\n"))
    await run.step(lambda api: api.details({"n": 1}, context))
    task_id = (await live(run.harness))["tools"][0]["taskId"]
    before = len(run.commits)
    await run.harness.abort_task(task_id, context)

    def is_abort_commit(ops: Sequence[Any]) -> bool:
        return any(op[0] == "s" and tuple(op[1]) == ("tools", 0, "status") for op in ops)

    await wait_for(lambda: any(is_abort_commit(ops) for ops in run.commits[before:]))
    abort_commit = norm_all(next(ops for ops in run.commits[before:] if is_abort_commit(ops)))
    assert ("s", ("tools", 0, "status"), "done") in abort_commit
    assert any(op[:2] == ("s", ("tools", 0, "entry")) and isinstance(op[2], int) for op in abort_commit)
    assert ("d", OUTPUT) in abort_commit
    assert ("d", ("tools", 0, "details")) in abort_commit
    assert len(abort_commit) == 4
    await run.harness.close(context)


def test_keeps_a_complete_base_exactly_while_nothing_runs() -> None:
    checkpoint_when = LiveDoc.definition.checkpoint_when
    assert checkpoint_when is not None

    def base(value: Any) -> bool:
        return checkpoint_when(value, [], {"deltasSinceBase": 1000})  # type: ignore[arg-type]

    def slot(status: str) -> Any:
        return {"callId": "c", "name": "n", "status": status}

    run = {"taskId": 1, "inputs": []}
    assert base({}) is True
    assert base({"run": run, "generation": {"attempt": 1}}) is False
    assert base({"run": run, "tools": [slot("pending"), slot("done")]}) is True
    assert base({"run": run, "tools": [slot("done"), slot("running")]}) is False
    assert base({"run": run}) is True
