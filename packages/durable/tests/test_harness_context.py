"""Port of ``test/harness-context.test.ts``.

Adaptations: Python does not freeze values (``harness/context.py``: kept ranges are immutable by contract), so
"keeps returned views and kept ranges apart" drops its ``TypeError`` assertion on a frozen kept entry, and
"freezes kept entries deeply even when storage freezes them shallowly" is skipped. Vitest's fake timers become a
recording ``setTimer`` and a fake ``now``. ``runtime.context()`` is a coroutine, so the late read is started with
``ensure_future`` and given one loop turn, as JavaScript's call runs its synchronous part at once.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from durable_harness_support import (
    add_task,
    assert_match,
    assistant,
    describe_message,
    open_harness,
    system,
    tool_result,
    user,
)
from durable_session_support import context, deferred, resolve

from pi_ai import create_models
from pi_chord.context import Context
from pi_durable import Harness, MemoryStorage, create_registry, define_task

DONE: Any = {"status": "terminal", "outcome": {"status": "completed", "result": None}}
ABORTED: Any = {"status": "terminal", "outcome": {"status": "aborted"}}


@dataclasses.dataclass(slots=True)
class Setup:
    harness: Harness
    root: Any

    async def append(self, draft: Any) -> Any:
        return await self.root.commit(lambda tx: tx.append_entry(self.root.id, draft), context)

    async def message(self, model: Any, kind: str = "message") -> Any:
        return await self.append({"kind": kind, "model": [model]})


async def setup() -> Setup:
    harness = (await open_harness(MemoryStorage())).harness
    root = await harness.root(context)
    return Setup(harness, root)


class CountingStorage(MemoryStorage):
    """Counts scanned entry rows; ``hold`` makes the next range scan wait (bounds probes, one row, do not)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows = 0
        self.hold: asyncio.Future[Any] | None = None

    async def scan_entries(self, query: Any, limit: int, cursor: Any, context: Context) -> Any:
        held = self.hold if limit > 1 else None
        if held is not None:
            self.hold = None
            await held
        page = await super().scan_entries(query, limit, cursor, context)
        self.rows += len(page["items"])
        return page


@dataclasses.dataclass(slots=True)
class Counting:
    harness: Harness
    registry: Any
    root: Any
    first: Any
    scanned: CountingStorage


async def counting_setup(**options: Any) -> Counting:
    """A Harness over MemoryStorage that counts scanned entry rows, with a 21-entry root transcript."""
    storage = CountingStorage()
    registry = create_registry()
    harness = await Harness.open(storage, {"models": create_models(), "registry": registry, **options}, context)
    root = await harness.root(context)
    first = await root.commit(
        lambda tx: tx.append_entry(root.id, {"kind": "message", "model": [user("first")]}), context
    )
    for index in range(20):
        await root.commit(
            lambda tx, index=index: tx.append_entry(root.id, {"kind": "message", "model": [assistant(f"old {index}")]}),
            context,
        )
    return Counting(harness, registry, root, first, storage)


def ids(entries: Any) -> list[Any]:
    return [entry["id"] for entry in entries]


def described(view: Any) -> list[str]:
    return [describe_message(message) for message in view["messages"]]


def one_phase(name: str, run: Callable[[Any, Any, Context], Awaitable[None]]) -> Any:
    """A task with one ``run`` phase and an abort handler that aborts."""

    async def abort(_task: Any, runtime: Any, task_context: Context) -> None:
        await runtime.commit(lambda _tx, _current: ABORTED, task_context)

    return define_task(name=name, version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=abort)


def probe_task(read: Callable[[str, Callable[[], Awaitable[Any]], Context], Awaitable[None]]) -> Any:
    """A task that reads its conversation's context once, named by its input."""

    async def run(task: Any, runtime: Any, task_context: Context) -> None:
        await read(task["input"]["name"], lambda: runtime.context(runtime.conversation_id, task_context), task_context)
        await runtime.commit(lambda _tx, _current: DONE, task_context)

    return one_phase("test.context-probe", run)


OPTIONS: Any = {"ownership": {"kind": "conversation"}}


async def run_task(harness: Harness, root: Any, task: Any, input: Any = None) -> None:
    """Create ``task`` in ``root`` and expect it to complete with ``None``."""
    id = await harness.commit(
        lambda tx: tx.create_task(task, {} if input is None else input, {**OPTIONS, "conversationId": root.id}),
        context,
    )
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == DONE["outcome"]


# ─── conversation context ───────────────────────────────────────────────────


async def test_returns_the_whole_transcript_without_a_head_and_excludes_model_less_entries_from_messages() -> None:
    s = await setup()
    first = await s.message(user("hi"))
    note = await s.append({"kind": "note", "data": {"text": "display only"}})
    answer = await s.message(assistant("hello"))
    view = await s.root.context(context)
    assert view["head"] is None
    assert ids(view["entries"]) == [first["id"], note["id"], answer["id"]]
    assert described(view) == ["user:hi", "assistant:hello"]


async def test_excludes_aborted_error_and_deferred_assistant_messages_but_keeps_their_raw_entries() -> None:
    s = await setup()
    await s.message(user("q"))
    aborted = await s.message(assistant("partial", stop_reason="aborted"))
    await s.message(assistant("failed", stop_reason="error"))
    await s.message(assistant("later", stop_reason="deferred"))
    await s.message(assistant("done", stop_reason="length"))
    view = await s.root.context(context)
    assert len(view["entries"]) == 5
    assert view["entries"][1]["id"] == aborted["id"]
    assert described(view) == ["user:q", "assistant:done"]


async def test_resolves_self_heads_and_uses_the_newest_head_marker() -> None:
    s = await setup()
    await s.message(user("old"))
    reset = await s.append({"kind": "reset", "head": "self", "model": [user("fresh start")]})
    assert reset["head"] == reset["id"]
    after = await s.message(assistant("after reset"))
    view = await s.root.context(context)
    assert view["head"]["id"] == reset["id"]
    assert ids(view["entries"]) == [reset["id"], after["id"]]
    assert described(view) == ["user:fresh start", "assistant:after reset"]

    # A compaction summary heads an earlier kept entry; older head markers in range drop out.
    summary = await s.append({"kind": "summary", "head": after["id"], "model": [user("summary")]})
    tail = await s.message(user("next"))
    view = await s.root.context(context)
    assert view["head"]["id"] == summary["id"]
    assert ids(view["entries"]) == [summary["id"], after["id"], tail["id"]]
    assert described(view) == ["user:summary", "assistant:after reset", "user:next"]


async def test_applies_the_newest_edit_per_target_within_the_active_range() -> None:
    s = await setup()
    first = await s.message(user("first"))
    second = await s.message(user("second"))
    await s.append(
        {"kind": "edit", "edits": [{"target": first["id"], "action": "replace", "messages": [user("first v2")]}]}
    )
    await s.append(
        {"kind": "edit", "edits": [{"target": first["id"], "action": "replace", "messages": [user("first v3")]}]}
    )
    await s.append({"kind": "edit", "edits": [{"target": second["id"], "action": "omit"}]})
    view = await s.root.context(context)
    assert len(view["entries"]) == 5
    assert described(view) == ["user:first v3"]

    # Edits before the active range no longer apply.
    reset = await s.append({"kind": "reset", "head": second["id"]})
    view = await s.root.context(context)
    assert view["head"]["id"] == reset["id"]
    assert described(view) == []
    await s.append(
        {"kind": "edit", "edits": [{"target": second["id"], "action": "replace", "messages": [user("second v2")]}]}
    )
    view = await s.root.context(context)
    assert described(view) == ["user:second v2"]


async def test_keeps_positional_system_messages_and_orders_tool_results_by_call_order() -> None:
    s = await setup()
    await s.message(system({"preamble": "You help."}), "pi.system")
    await s.message(user("run tools"))
    await s.message(assistant("calling", calls=["b", "a"]))
    await s.message(tool_result("a"))
    await s.append({"kind": "pi.system", "model": [system({"cwd": "/repo"})]})
    await s.message(tool_result("b"))
    await s.message(tool_result("zz"))
    await s.message(assistant("done"))
    view = await s.root.context(context)
    assert described(view) == [
        "system:preamble",
        "user:run tools",
        "assistant:calling",
        "result:b:result b",
        "result:a:result a",
        "system:cwd",
        "assistant:done",
    ]


# #10542
async def test_leads_with_a_system_message_that_only_user_messages_precede_and_keeps_later_ones_in_place() -> None:
    s = await setup()
    await s.message(user("first"))
    await s.message(user("steered"))
    await s.message(system({"preamble": "You help."}), "pi.system")
    await s.message(assistant("answer"))
    await s.message(user("next"))
    await s.message(system({"cwd": "/repo"}), "pi.system")
    view = await s.root.context(context)
    assert described(view) == [
        "system:preamble",
        "user:first",
        "user:steered",
        "assistant:answer",
        "user:next",
        "system:cwd",
    ]
    # Stored order and contributions stay as committed.
    assert [describe_message(message) for group in view["contributions"] for message in group][:3] == [
        "user:first",
        "user:steered",
        "system:preamble",
    ]

    # After a reset, the baseline written after the handoff leads too.
    await s.append({"kind": "reset", "head": "self", "model": [user("handoff")]})
    await s.message(system({"preamble": "Baseline."}), "pi.system")
    view = await s.root.context(context)
    assert described(view) == ["system:preamble", "user:handoff"]


# #10904
async def test_leads_with_the_baseline_after_a_compaction_that_keeps_assistant_messages() -> None:
    s = await setup()
    await s.message(user("old"))
    kept = await s.message(user("kept"))
    await s.message(assistant("calling", calls=["a"]))
    await s.message(tool_result("a"))
    await s.message(assistant("answer"))
    await s.append({"kind": "summary", "head": kept["id"], "model": [user("summary")]})
    await s.message(user("next"))
    await s.message(system({"preamble": "Baseline."}), "pi.system")
    view = await s.root.context(context)
    assert described(view) == [
        "system:preamble",
        "user:summary",
        "user:kept",
        "assistant:calling",
        "result:a:result a",
        "assistant:answer",
        "user:next",
    ]

    # Later system messages stay in place.
    await s.message(assistant("done"))
    await s.message(system({"cwd": "/repo"}), "pi.system")
    view = await s.root.context(context)
    assert described(view)[:2] == ["system:preamble", "user:summary"]
    assert described(view)[-2:] == ["assistant:done", "system:cwd"]


# #10904
async def test_leads_with_a_baseline_written_after_the_first_answers_when_the_agent_had_no_section_or_tool_before() -> (
    None
):
    s = await setup()
    await s.message(user("first"), "pi.user")
    await s.message(assistant("answer"), "pi.assistant")
    await s.message(user("next"), "pi.user")
    await s.message(system({"plan": "Plan."}), "pi.system")
    view = await s.root.context(context)
    assert described(view) == ["system:plan", "user:first", "assistant:answer", "user:next"]


# #10904
async def test_keeps_system_messages_other_than_the_baseline_in_place_once_an_assistant_message_precedes_them() -> None:
    s = await setup()
    await s.message(user("first"))
    await s.message(assistant("answer"))
    # Another kind's system message is positional; the baseline after it is not the first system message.
    await s.message(system({"note": "From here on."}), "host.note")
    await s.message(system({"preamble": "Baseline."}), "pi.system")
    view = await s.root.context(context)
    assert described(view) == ["user:first", "assistant:answer", "system:note", "system:preamble"]

    # After a compaction that keeps the other kind's system message, the new baseline does not move ahead of it.
    kept = await s.message(user("kept"))
    await s.message(assistant("kept answer"))
    await s.message(system({"mode": "plan"}), "host.note")
    await s.append({"kind": "summary", "head": kept["id"], "model": [user("summary")]})
    await s.message(user("next"))
    await s.message(system({"preamble": "Baseline."}), "pi.system")
    view = await s.root.context(context)
    assert described(view) == [
        "user:summary",
        "user:kept",
        "assistant:kept answer",
        "system:mode",
        "user:next",
        "system:preamble",
    ]


# #10904
async def test_does_not_lead_with_a_first_system_entry_written_after_another_system_message_even_once_that_is_omitted() -> (
    None
):
    s = await setup()
    host = await s.message(system({"legacy": "Use the legacy format."}), "host.system")
    await s.message(user("first"))
    await s.message(assistant("answer"))
    # Generation wrote this entry against the host's message: it only removes a section, so it is no baseline.
    await s.message(system({"legacy": None}), "pi.system")
    await s.append({"kind": "edit", "edits": [{"target": host["id"], "action": "omit"}]})
    view = await s.root.context(context)
    assert described(view) == ["user:first", "assistant:answer", "system:legacy"]

    # The same holds for a system message an edit put before it.
    other = await setup()
    first = await other.message(user("first"))
    await other.append(
        {
            "kind": "edit",
            "edits": [
                {"target": first["id"], "action": "replace", "messages": [user("first"), system({"note": "Note."})]}
            ],
        }
    )
    await other.message(assistant("answer"))
    await other.message(system({"note": None}), "pi.system")
    await other.append(
        {"kind": "edit", "edits": [{"target": first["id"], "action": "replace", "messages": [user("first")]}]}
    )
    view = await other.root.context(context)
    assert described(view) == ["user:first", "assistant:answer", "system:note"]


# #10904
async def test_does_not_lead_with_a_retained_edited_or_later_system_entry_in_place_of_the_baseline() -> None:
    s = await setup()
    kept = await s.message(user("kept"))
    await s.message(assistant("answer"))
    retained = await s.message(system({"cwd": "/repo"}), "pi.system")
    # Before generation writes the baseline after the head marker, a retained system entry stays in place.
    await s.append({"kind": "summary", "head": kept["id"], "model": [user("summary")]})
    await s.message(user("next"))
    view = await s.root.context(context)
    assert described(view) == ["user:summary", "user:kept", "assistant:answer", "system:cwd", "user:next"]

    # The baseline omits the retained entry, as generation writes it, and leads.
    baseline = await s.append(
        {
            "kind": "pi.system",
            "model": [system({"preamble": "Baseline."})],
            "edits": [{"target": retained["id"], "action": "omit"}],
        }
    )
    await s.message(assistant("done"))
    await s.message(system({"preamble": None}), "pi.system")
    view = await s.root.context(context)
    assert described(view) == [
        "system:preamble",
        "user:summary",
        "user:kept",
        "assistant:answer",
        "user:next",
        "assistant:done",
        "system:preamble",
    ]

    # An omitted baseline leaves no message to lead; the delta after it is not a baseline and stays in place.
    await s.append({"kind": "edit", "edits": [{"target": baseline["id"], "action": "omit"}]})
    view = await s.root.context(context)
    assert described(view) == [
        "user:summary",
        "user:kept",
        "assistant:answer",
        "user:next",
        "assistant:done",
        "system:preamble",
    ]

    # A replacement of the baseline is not one either.
    await s.append(
        {
            "kind": "edit",
            "edits": [{"target": baseline["id"], "action": "replace", "messages": [system({"other": "Other."})]}],
        }
    )
    view = await s.root.context(context)
    assert described(view) == [
        "user:summary",
        "user:kept",
        "assistant:answer",
        "user:next",
        "system:other",
        "assistant:done",
        "system:preamble",
    ]


async def test_synthesizes_missing_tool_results_after_a_fork_and_drops_results_cut_from_their_call() -> None:
    s = await setup()
    await s.message(user("go"))
    call = await s.message(assistant("calling", calls=["x", "y"]))
    await s.message(tool_result("x"))
    second = await s.message(tool_result("y"))
    child = await s.root.fork(call["id"], {"ownership": {"kind": "ownerless"}}, context)
    child_view = await child.context(context)
    assert described(child_view) == ["user:go", "assistant:calling", "result:x:error", "result:y:error"]
    missing = child_view["messages"][2]
    assert missing.role == "toolResult"
    assert_match(missing, {"tool_name": "tool-x", "details": {"reason": "missing_result"}})

    # A head between a call and its results leaves stray results that are not sent.
    await s.root.commit(lambda tx: tx.append_entry(s.root.id, {"kind": "reset", "head": second["id"]}), context)
    parent_view = await s.root.context(context)
    assert described(parent_view) == []
    assert [entry["kind"] for entry in parent_view["entries"]] == ["reset", "message"]


# #10512
async def test_reads_the_context_as_of_an_earlier_entry_as_a_fork_at_that_entry_starts() -> None:
    s = await setup()
    first = await s.message(user("first"))
    call = await s.message(assistant("calling", calls=["x", "y"]))
    result = await s.message(tool_result("x"))
    await s.message(tool_result("y"))
    edit = await s.append(
        {"kind": "edit", "edits": [{"target": first["id"], "action": "replace", "messages": [user("first v2")]}]}
    )
    reset = await s.append({"kind": "reset", "head": "self", "model": [user("fresh start")]})
    tail = await s.message(assistant("after reset"))

    for at in (first, call, result, edit, reset, tail):
        fork = await s.root.fork(at["id"], {"ownership": {"kind": "ownerless"}}, context)
        assert await s.root.context(context, at=at["id"]) == await fork.context(context)
    # Stepped back before the edit and the reset: neither applies, and a cut call gets synthesized results.
    assert described(await s.root.context(context, at=call["id"])) == [
        "user:first",
        "assistant:calling",
        "result:x:error",
        "result:y:error",
    ]
    assert await s.root.context(context, at=tail["id"]) == await s.root.context(context)
    assert await s.root.context(context, at=None) == await s.root.context(context)

    other = await s.root.fork(first["id"], {"ownership": {"kind": "ownerless"}}, context)
    with pytest.raises(Exception, match="is not visible"):
        await other.context(context, at=tail["id"])


async def test_extends_a_tasks_context_read_with_only_newer_entries() -> None:
    c = await counting_setup()
    harness, root, first, scanned = c.harness, c.root, c.first, c.scanned

    async def run(_task: Any, runtime: Any, task_context: Context) -> None:
        async def write(draft: Any) -> None:
            async def change(tx: Any, _current: Any) -> None:
                await tx.append_entry(root.id, draft)

            await runtime.commit(change, task_context)

        async def read(at: Any = None) -> tuple[Any, int]:
            before = scanned.rows
            view = await runtime.context(root.id, task_context, at=at)
            return view, scanned.rows - before

        initial, _rows = await read()
        assert initial == await root.context(task_context)

        # Three new entries: the bounds probe reads one row, the extension the three new ones.
        await write({"kind": "message", "model": [user("new")]})
        await write({"kind": "note", "data": {"text": "display only"}})
        await write({"kind": "message", "model": [assistant("answer")]})
        extended, rows = await read()
        assert rows == 4
        assert extended == await root.context(task_context)

        # A newer edit of an older entry applies to the extended range.
        await write({"kind": "edit", "edits": [{"target": first["id"], "action": "omit"}]})
        edited, rows = await read()
        assert rows == 2
        assert edited == await root.context(task_context)
        assert "user:first" not in described(edited)

        # An earlier cutoff reuses the range.
        cutoff, rows = await read(extended["entries"][-1]["id"])
        assert rows == 0
        assert cutoff == extended

        # A new head marker changes the range: read it whole.
        await write({"kind": "reset", "head": "self", "model": [user("fresh")]})
        reset, _rows = await read()
        assert reset == await root.context(task_context)
        assert described(reset) == ["user:fresh"]

        await runtime.commit(lambda _tx, _current: DONE, task_context)

    reads = one_phase("test.context-reads", run)
    add_task(c.registry, reads)
    id = await harness.commit(lambda tx: tx.create_task(reads, {}, {**OPTIONS, "conversationId": root.id}), context)
    harness.resume()
    settled = await harness.wait_for_task(id, context)
    assert settled["state"]["outcome"] == {"status": "completed", "result": None}
    await harness.close(context)


async def test_keeps_a_conversations_context_read_across_its_tasks_and_for_the_retention_period_once_idle() -> None:
    clock = [1_000]
    c = await counting_setup(now=lambda: clock[0])
    harness, root, scanned = c.harness, c.root, c.scanned
    rows: dict[str, int] = {}

    async def read(name: str, load: Callable[[], Awaitable[Any]], task_context: Context) -> None:
        before = scanned.rows
        view = await load()
        rows[name] = scanned.rows - before
        assert view == await root.context(task_context)

    async def child_run(_task: Any, runtime: Any, task_context: Context) -> None:
        async def change(tx: Any, _current: Any) -> None:
            await tx.append_entry(root.id, {"kind": "message", "model": [user("from child")]})

        await runtime.commit(change, task_context)
        await read("child", lambda: runtime.context(root.id, task_context), task_context)
        await runtime.commit(lambda _tx, _current: DONE, task_context)

    child_task = one_phase("test.context-child", child_run)

    async def parent_start(_task: Any, runtime: Any, task_context: Context) -> None:
        await read("parent", lambda: runtime.context(root.id, task_context), task_context)

        async def change(tx: Any, _current: Any) -> Any:
            child = await tx.create_task(
                child_task, {}, {"ownership": {"kind": "task", "taskId": runtime.task_id}, "conversationId": root.id}
            )
            return {"status": "waiting", "checkpoint": {"phase": "after"}, "on": [child], "policy": "allSettled"}

        await runtime.commit(change, task_context)

    async def parent_after(_task: Any, runtime: Any, task_context: Context) -> None:
        await read("after", lambda: runtime.context(root.id, task_context), task_context)
        await runtime.commit(lambda _tx, _current: DONE, task_context)

    async def abort(_task: Any, runtime: Any, task_context: Context) -> None:
        await runtime.commit(lambda _tx, _current: ABORTED, task_context)

    parent_task = define_task(
        name="test.context-parent",
        version=1,
        initial=lambda _input: {"phase": "start"},
        phases={"start": parent_start, "after": parent_after},
        abort=abort,
    )
    probe = probe_task(read)
    for task in (child_task, parent_task, probe):
        add_task(c.registry, task)
    harness.resume()
    await run_task(harness, root, parent_task)
    await root.wait_for_idle(context)

    async def run_probe(name: str) -> None:
        await run_task(harness, root, probe, {"name": name})
        await root.wait_for_idle(context)

    await run_probe("idle")
    clock[0] += 600_000
    await run_probe("expired")
    # Bounds probes read one row each. The child scans its own new entry; the waiting parent's later invocation and
    # the probe within the retention period scan nothing new. After it, the probe reads the whole transcript again.
    assert rows == {"parent": 22, "child": 2, "after": 1, "idle": 1, "expired": 23}
    await harness.close(context)


def lcg(seed: int) -> Callable[[int], int]:
    """Deterministic pseudo-random integers (32-bit LCG, high bits), as upstream's ``Math.imul`` LCG."""
    state = [seed]

    def random(n: int) -> int:
        state[0] = (state[0] * 1_664_525 + 1_013_904_223) % 2**32
        return int(state[0] / 2**32 * n)

    return random


async def test_derives_the_same_view_from_extended_reads_as_from_whole_reads() -> None:
    c = await counting_setup()
    harness, root = c.harness, c.root
    # Deterministic pseudo-random transcript: calls and results in any order, missing and stray results, excluded
    # stop reasons, system entries, notes, edits of earlier entries, and head markers.
    random = lcg(7)
    seen: dict[str, int] = {}

    def count(category: str) -> None:
        seen[category] = seen.get(category, 0) + 1

    async def run(_task: Any, runtime: Any, task_context: Context) -> None:
        written: list[Any] = []
        calls: list[str] = []
        counter = [0]

        def next_id() -> int:
            value = counter[0]
            counter[0] += 1
            return value

        def draft() -> Any:
            pick = random(14)
            if pick < 3:
                count("user")
                return {"kind": "message", "model": [user(f"u{next_id()}")]}
            if pick < 5:
                ids_ = [f"c{next_id()}" for _ in range(random(3))]
                calls.extend(ids_)
                reasons = ["aborted", "error", "deferred"]
                index = random(9)
                stop_reason = reasons[index] if index < len(reasons) else None
                count(stop_reason or "assistant")
                return {"kind": "message", "model": [assistant(f"a{next_id()}", calls=ids_, stop_reason=stop_reason)]}
            if pick < 8:
                matched = len(calls) > 0 and random(5) > 0
                count("result" if matched else "stray result")
                id = calls.pop(random(len(calls))) if matched else f"stray{next_id()}"
                return {"kind": "message", "model": [tool_result(id)]}
            if pick < 9:
                # `pi.system` entries can be the baseline that leads the context (#10904).
                kind = "pi.system" if random(2) == 0 else "message"
                count("baseline system" if kind == "pi.system" else "system")
                return {"kind": kind, "model": [system({"s": f"v{next_id()}"})]}
            if pick < 10:
                count("note")
                return {"kind": "note", "data": {"n": next_id()}}
            index = random(len(written))
            target = written[index] if index < len(written) else None
            if pick < 12 and target is not None:
                omit = random(2) == 0
                count("omit" if omit else "replace")
                action: Any = {"action": "omit"} if omit else {"action": "replace", "messages": [user(f"r{next_id()}")]}
                return {"kind": "edit", "edits": [{"target": target["id"], **action}]}
            if pick < 13 and target is not None:
                count("head")
                return {"kind": "summary", "head": target["id"], "model": [user(f"h{next_id()}")]}
            count("user")
            return {"kind": "message", "model": [user(f"u{next_id()}")]}

        async def append(conversation_id: Any) -> None:
            async def change(tx: Any, _current: Any) -> None:
                added = 1 + random(3)
                for _ in range(added):
                    written.append(await tx.append_entry(conversation_id, draft()))

            await runtime.commit(change, task_context)

        for step in range(150):
            await append(root.id)
            whole = await root.context(task_context)
            assert await runtime.context(root.id, task_context) == whole
            contributed = [message for group in whole["contributions"] for message in group]
            first = next((index for index, message in enumerate(contributed) if message.role == "system"), -1)
            if (
                first > 0
                and whole["messages"][0] is contributed[first]
                and any(message.role == "assistant" for message in contributed[:first])
            ):
                count("baseline past assistant")
            if step % 10 == 9:
                # Overlapping reads, one with an earlier cutoff, share the kept range.
                at = written[random(len(written))]["id"]
                whole, cut = await asyncio.gather(
                    runtime.context(root.id, task_context), runtime.context(root.id, task_context, at=at)
                )
                assert whole == await root.context(task_context)
                assert cut == await root.context(task_context, at=at)
        # A fork sees its parent's entries through the fork point and extends with its own.
        fork = await root.fork(written[random(len(written))]["id"], {"ownership": {"kind": "ownerless"}}, task_context)
        for _ in range(20):
            assert await runtime.context(fork.id, task_context) == await fork.context(task_context)
            await append(fork.id)
        await runtime.commit(lambda _tx, _current: DONE, task_context)

    steps = one_phase("test.context-steps", run)
    add_task(c.registry, steps)
    harness.resume()
    await run_task(harness, root, steps)
    categories = [
        "user",
        "assistant",
        "aborted",
        "error",
        "deferred",
        "result",
        "stray result",
        "system",
        "baseline system",
        "baseline past assistant",
        "note",
    ]
    for category in [*categories, "omit", "replace", "head"]:
        assert seen.get(category, 0) > 0, category
    await harness.close(context)


# #10904
async def test_finds_the_baseline_among_the_entries_an_extended_read_adds() -> None:
    c = await counting_setup()
    harness, root, first, scanned = c.harness, c.root, c.first, c.scanned

    async def run(_task: Any, runtime: Any, task_context: Context) -> None:
        async def append(conversation_id: Any, draft: Any) -> Any:
            written: list[Any] = []

            async def change(tx: Any, _current: Any) -> None:
                written.append(await tx.append_entry(conversation_id, draft))

            await runtime.commit(change, task_context)
            return written[0]

        async def read(conversation: Any) -> tuple[list[str], int]:
            before = scanned.rows
            view = await runtime.context(conversation.id, task_context)
            rows = scanned.rows - before
            assert view == await conversation.context(task_context)
            return described(view), rows

        # The cached range has answers and no system message; the baseline arrives in a later read.
        assert (await read(root))[0][0] == "user:first"
        await append(root.id, {"kind": "pi.system", "model": [system({"plan": "Plan."})]})
        messages, rows = await read(root)
        # One bounds probe and the added entry: the range was extended, not derived again.
        assert rows == 2
        assert messages[:3] == ["system:plan", "user:first", "assistant:old 0"]

        # A system message the cached range already has keeps a later first `pi.system` entry in place.
        fork = await root.fork(first["id"], {"ownership": {"kind": "ownerless"}}, task_context)
        await append(fork.id, {"kind": "message", "model": [assistant("answer")]})
        await append(fork.id, {"kind": "host.system", "model": [system({"note": "Note."})]})
        assert (await read(fork))[0] == ["user:first", "assistant:answer", "system:note"]
        await append(fork.id, {"kind": "pi.system", "model": [system({"note": None})]})
        messages, rows = await read(fork)
        assert rows == 2
        assert messages == ["user:first", "assistant:answer", "system:note", "system:note"]

        # An omitted system message still rules out a later baseline; the extended read keeps that.
        omitted = await root.fork(first["id"], {"ownership": {"kind": "ownerless"}}, task_context)
        await append(omitted.id, {"kind": "message", "model": [assistant("answer")]})
        host = await append(omitted.id, {"kind": "host.system", "model": [system({"legacy": "Legacy."})]})
        await append(omitted.id, {"kind": "edit", "edits": [{"target": host["id"], "action": "omit"}]})
        assert (await read(omitted))[0] == ["user:first", "assistant:answer"]
        await append(omitted.id, {"kind": "pi.system", "model": [system({"legacy": None})]})
        messages, rows = await read(omitted)
        assert rows == 2
        assert messages == ["user:first", "assistant:answer", "system:legacy"]

        # A first `pi.system` entry without a system message leaves no baseline for a later one to take.
        empty = await root.fork(first["id"], {"ownership": {"kind": "ownerless"}}, task_context)
        await append(empty.id, {"kind": "pi.system"})
        await append(empty.id, {"kind": "message", "model": [assistant("answer")]})
        assert (await read(empty))[0] == ["user:first", "assistant:answer"]
        await append(empty.id, {"kind": "pi.system", "model": [system({"s": "Delta."})]})
        messages, rows = await read(empty)
        assert rows == 2
        assert messages == ["user:first", "assistant:answer", "system:s"]
        await runtime.commit(lambda _tx, _current: DONE, task_context)

    steps = one_phase("test.context-baseline", run)
    add_task(c.registry, steps)
    harness.resume()
    await run_task(harness, root, steps)
    await harness.close(context)


async def test_keeps_returned_views_and_kept_ranges_apart() -> None:
    c = await counting_setup()
    harness, root = c.harness, c.root

    async def run(_task: Any, runtime: Any, task_context: Context) -> None:
        view = await runtime.context(root.id, task_context)
        view["messages"].clear()
        view["entries"].pop()
        # (Upstream also checks that kept entries are frozen; Python keeps them immutable by contract.)
        assert await runtime.context(root.id, task_context) == await root.context(task_context)
        await runtime.commit(lambda _tx, _current: DONE, task_context)

    mutate = one_phase("test.context-mutate", run)
    add_task(c.registry, mutate)
    harness.resume()
    await run_task(harness, root, mutate)
    await harness.close(context)


@pytest.mark.skip(reason="JavaScript-specific: Python does not freeze values (kept ranges are immutable by contract)")
async def test_freezes_kept_entries_deeply_even_when_storage_freezes_them_shallowly() -> None:
    pass


async def test_does_not_keep_a_read_that_finishes_after_its_task_ended() -> None:
    c = await counting_setup(settings={"contextRetentionMs": 0})
    harness, root, scanned = c.harness, c.root, c.scanned
    scans = deferred()
    probe_start = deferred()
    late: list[asyncio.Future[Any]] = []

    async def run(_task: Any, runtime: Any, task_context: Context) -> None:
        scanned.hold = scans

        async def read() -> Any:
            try:
                return await runtime.context(root.id, task_context)
            except Exception as error:
                return error

        late.append(asyncio.ensure_future(read()))
        # JavaScript runs the call's synchronous part at once.
        await asyncio.sleep(0)
        await runtime.commit(lambda _tx, _current: DONE, task_context)

    late_task = one_phase("test.context-late", run)
    rows: dict[str, int] = {}

    async def probe_read(name: str, load: Callable[[], Awaitable[Any]], _task_context: Context) -> None:
        await probe_start
        before = scanned.rows
        await load()
        rows[name] = scanned.rows - before

    probe = probe_task(probe_read)
    for task in (late_task, probe):
        add_task(c.registry, task)
    harness.resume()
    await run_task(harness, root, late_task)
    # The conversation is busy again when the late read finishes; its range must not be kept.
    probe_id = await harness.commit(
        lambda tx: tx.create_task(probe, {"name": "probe"}, {**OPTIONS, "conversationId": root.id}), context
    )
    resolve(scans)
    await late[0]
    resolve(probe_start)
    assert (await harness.wait_for_task(probe_id, context))["state"]["outcome"] == DONE["outcome"]
    assert rows == {"probe": 22}
    await harness.close(context)


class FakeTimers:
    """Vitest's fake timers for ``setTimeout`` and ``Date``: a recording ``setTimer`` and a clock."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self._timers: dict[int, tuple[float, Callable[[], None]]] = {}
        self._next = 0

    def set_timer(self, callback: Callable[[], None], delay_ms: float, _options: Any) -> Callable[[], None]:
        key = self._next
        self._next += 1
        self._timers[key] = (self.now + max(0.0, delay_ms), callback)
        return lambda: self._timers.pop(key, None) and None

    def count(self) -> int:
        return len(self._timers)

    def advance(self, ms: float) -> None:
        target = self.now + ms
        while True:
            due = sorted((at, key) for key, (at, _callback) in self._timers.items() if at <= target)
            if not due:
                break
            at, key = due[0]
            _at, callback = self._timers.pop(key)
            self.now = at
            callback()
        self.now = target


async def test_drops_an_idle_conversations_context_read_on_a_timer_when_nothing_else_runs() -> None:
    timers = FakeTimers()
    c = await counting_setup(settings={"contextRetentionMs": 1_000}, now=lambda: timers.now, setTimer=timers.set_timer)
    harness, root, scanned = c.harness, c.root, c.scanned
    rows: dict[str, int] = {}

    async def read(name: str, load: Callable[[], Awaitable[Any]], _task_context: Context) -> None:
        before = scanned.rows
        await load()
        rows[name] = scanned.rows - before

    probe = probe_task(read)
    add_task(c.registry, probe)
    harness.resume()

    async def run_probe(name: str) -> None:
        await run_task(harness, root, probe, {"name": name})
        await root.wait_for_idle(context)

    idle = timers.count()
    await run_probe("first")
    assert timers.count() == idle + 1
    timers.advance(1_000)
    assert timers.count() == idle
    await run_probe("second")
    assert rows == {"first": 22, "second": 22}
    await harness.close(context)


async def test_drops_a_conversations_context_read_once_idle_with_zero_retention() -> None:
    c = await counting_setup(settings={"contextRetentionMs": 0})
    harness, root, scanned = c.harness, c.root, c.scanned
    rows: dict[str, int] = {}

    async def read(name: str, load: Callable[[], Awaitable[Any]], _task_context: Context) -> None:
        before = scanned.rows
        await load()
        rows[name] = scanned.rows - before

    probe = probe_task(read)
    add_task(c.registry, probe)
    harness.resume()
    for name in ("first", "second"):
        await run_task(harness, root, probe, {"name": name})
        await root.wait_for_idle(context)
    assert rows == {"first": 22, "second": 22}
    await harness.close(context)
