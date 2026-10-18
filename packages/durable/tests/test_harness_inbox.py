"""Port of ``test/harness-inbox.test.ts`` (the ``inbox`` and ``usage`` groups)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from durable_harness_support import (
    ChatSetup,
    aborted,
    add_hooks,
    add_tool,
    all_entries,
    chat_setup,
    open_chat,
    text_of,
    wait_for,
)
from durable_session_support import (
    STORAGE_BACKENDS,
    ControlledStorage,
    context,
    deferred,
    document_changes,
    flush,
    resolve,
)

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage, Usage, UsageCost, UserMessage
from pi_durable import (
    GenerationTask,
    InboxDoc,
    LiveDoc,
    MemoryStorage,
    ResetEntry,
    ToolTask,
    UsageDoc,
    define_doc,
    define_tool,
)
from pi_durable.harness.usage import record_usage, usage_json
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


def answer(text: str) -> AssistantMessage:
    return faux_assistant_message([faux_text(text)])


class Gated:
    """A faux response held until ``release()`` or cancellation; ``reached`` resolves when the request is sent."""

    def __init__(self, message: AssistantMessage) -> None:
        self.reached: asyncio.Future[Any] = deferred()
        self._gate: asyncio.Future[Any] = deferred()
        self._message = message

    async def step(self, _request: Any, options: Any, *_rest: Any) -> AssistantMessage:
        resolve(self.reached)
        cancelled = asyncio.ensure_future(aborted(options["signal"]))
        try:
            await asyncio.wait([self._gate, cancelled], return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not cancelled.done():
                cancelled.cancel()
        if cancelled.done() and not cancelled.cancelled():
            cancelled.result()
        return self._message

    def release(self) -> None:
        resolve(self._gate)


def hold_tool(setup: ChatSetup, gate: asyncio.Future[Any], result: Any = None) -> None:
    """Register a ``hold`` tool whose calls wait for ``gate`` and then return ``result``."""

    async def execute(*_args: Any) -> Any:
        await gate
        return {"output": []} if result is None else result

    add_tool(
        setup.registry,
        define_tool(name="hold", description="Waits for the test", parameters=Type.Object({}), execute=execute),
    )


def queued_note(text: str) -> dict[str, Any]:
    """A write that affects model context, so it queues while busy; a data-only write would be appended at once."""
    return {"kind": "note", "model": [UserMessage(text, timestamp=0)]}


def queued_note_json(text: str) -> dict[str, Any]:
    """``queued_note`` as the inbox document stores it."""
    return {"kind": "note", "model": [{"role": "user", "content": text, "timestamp": 0}]}


HOLD = faux_assistant_message([faux_tool_call("hold", {}, id="c1")], stop_reason="toolUse")
FAILED = faux_assistant_message([], stop_reason="error", error_message="invalid request")


def matches(record: Any, expected: dict[str, Any]) -> bool:
    """``toMatchObject`` for flat records."""
    return all(key in record and record[key] == value for key, value in expected.items())


async def status(submission: Any) -> Any:
    return await submission.status(context)


def transcript(entries: Sequence[Any]) -> list[str]:
    """Kind and text of each entry, skipping system entries."""
    shown: list[str] = []
    for entry in entries:
        if entry["kind"] == "pi.system":
            continue
        message = (entry.get("model") or [None])[0]
        text = None if message is not None and message.role == "toolResult" else text_of(message)
        shown.append(entry["kind"] if text is None else f"{entry['kind']}:{text}")
    return shown


async def inbox(harness: Any, root: Any) -> list[list[Any]]:
    state = await harness.snapshot(InboxDoc, root.id, context)
    return [[item["id"], item["mode"]] for item in state["items"]]


async def tool_running(harness: Any, root: Any, index: int = 0, wanted: str = "running") -> None:
    async def check() -> bool:
        live = await harness.snapshot(LiveDoc, root.id, context)
        tools = (live or {}).get("tools") or []
        return len(tools) > index and tools[index]["status"] == wanted

    await wait_for(check)


async def live_partial(harness: Any, root: Any) -> None:
    async def check() -> bool:
        live = await harness.snapshot(LiveDoc, root.id, context)
        return ((live or {}).get("generation") or {}).get("message") is not None

    await wait_for(check)


def assistants_of(entries: Sequence[Any]) -> list[Any]:
    return [entry["model"][0] for entry in entries if entry["kind"] == "pi.assistant"]


def inbox_ops(harness: Any) -> list[list[Any]]:
    ops: list[list[Any]] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            if change["record"]["kind"] == "pi.inbox" and len(change["ops"]) > 0:
                ops.append(list(change["ops"]))

    harness.subscribe_commits(listener)
    return ops


# ─── inbox ──────────────────────────────────────────────────────────────────


async def test_queues_busy_submissions_and_places_writes_before_user_items_at_the_final_boundary() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("second"), answer("third")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    f1 = await root.submit({"type": "input", "content": "f1"}, context)
    write = await root.submit({"type": "write", "entry": queued_note("w")}, context)
    f2 = await root.submit({"type": "input", "content": "f2", "whenBusy": "followUp"}, context)
    for submission in (f1, write, f2):
        assert (await status(submission))["status"] == "queued"
    assert await inbox(harness, root) == [[f1.id, "followUp"], [write.id, "write"], [f2.id, "followUp"]]
    assert transcript(await all_entries(root)) == ["pi.user:a"]

    first.release()
    await f2.wait(context)
    assert transcript(await all_entries(root)) == [
        "pi.user:a",
        "pi.assistant:first",
        "note:w",
        "pi.user:f1",
        "pi.assistant:second",
        "pi.user:f2",
        "pi.assistant:third",
    ]
    answers = [entry for entry in await all_entries(root) if entry["kind"] == "pi.assistant"]
    assert matches(await status(input), {"status": "done", "answer": answers[0]["id"]})
    assert matches(await status(write), {"status": "done"})
    assert matches(await status(f1), {"status": "done", "answer": answers[1]["id"]})
    assert matches(await status(f2), {"status": "done", "answer": answers[2]["id"]})
    assert await inbox(harness, root) == []
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    await harness.close(context)


async def test_places_every_follow_up_in_one_successor_run_with_follow_up_mode_all() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("both")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    setup.settings["followUpMode"] = "all"
    await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    f1 = await root.submit({"type": "input", "content": "f1"}, context)
    f2 = await root.submit({"type": "input", "content": "f2"}, context)
    first.release()
    settled = await f2.wait(context)
    f1_status = await status(f1)
    assert isinstance(f1_status["entry"], int)
    assert f1_status == {**settled, "id": f1.id, "entry": f1_status["entry"]}
    assert transcript(await all_entries(root)) == [
        "pi.user:a",
        "pi.assistant:first",
        "pi.user:f1",
        "pi.user:f2",
        "pi.assistant:both",
    ]
    assert setup.faux.state.call_count == 2
    await harness.close(context)


async def test_reads_queue_modes_when_the_final_boundarys_commit_runs_on_the_session_line() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("both")])
    yielded = deferred()
    add_hooks(setup.registry, GenerationTask, {"on_yield": lambda *_: resolve(yielded)})
    storage = ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])
    opened = await open_chat(storage, setup)
    harness, root = opened.harness, opened.root
    await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    f1 = await root.submit({"type": "input", "content": "f1"}, context)
    f2 = await root.submit({"type": "input", "content": "f2"}, context)
    # Occupy the line, let the answer queue its boundary commit behind it, then change the mode.
    held = storage.hold_commits()
    marker = define_doc(kind="test.marker", version=1, scope="session", initial=lambda: {"n": 0})

    async def bump(tx: Any) -> None:
        (await tx.doc(marker))["n"] += 1

    occupying = root.commit(bump, context)
    await held.entered
    first.release()
    await yielded
    await flush()
    setup.settings["followUpMode"] = "all"
    held.release()
    await occupying
    settled = await f2.wait(context)
    f1_status = await status(f1)
    assert f1_status == {**settled, "id": f1.id, "entry": f1_status["entry"]}
    assert setup.faux.state.call_count == 2
    await harness.close(context)


async def test_adds_steers_to_the_run_at_the_post_tools_boundary_and_holds_follow_ups_for_the_final_boundary() -> None:
    setup = chat_setup()
    gate = deferred()
    hold_tool(setup, gate)
    setup.faux.set_responses([HOLD, answer("after tools"), answer("follow-up")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await tool_running(harness, root)
    steer = await root.submit({"type": "input", "content": "s", "whenBusy": "steer"}, context)
    follow_up = await root.submit({"type": "input", "content": "f"}, context)
    resolve(gate)
    await follow_up.wait(context)
    assert transcript(await all_entries(root)) == [
        "pi.user:a",
        "pi.assistant",
        "pi.tool-result",
        "pi.user:s",
        "pi.assistant:after tools",
        "pi.user:f",
        "pi.assistant:follow-up",
    ]
    answers = [entry for entry in await all_entries(root) if entry["kind"] == "pi.assistant"]
    assert matches(await status(input), {"status": "done", "answer": answers[1]["id"]})
    assert matches(await status(steer), {"status": "done", "answer": answers[1]["id"]})
    assert matches(await status(follow_up), {"status": "done", "answer": answers[2]["id"]})
    await harness.close(context)


async def test_ends_the_run_at_a_queued_reset_after_tools_and_runs_earlier_follow_ups_in_the_new_context() -> None:
    setup = chat_setup()
    gate = deferred()
    hold_tool(setup, gate)
    requests: list[list[str]] = []

    def record(request: Any, *_rest: Any) -> AssistantMessage:
        requests.append([f"{message.role}:{text_of(message) or ''}" for message in request.messages])
        return answer("fresh")

    setup.faux.set_responses([HOLD, record])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await tool_running(harness, root)
    follow_up = await root.submit({"type": "input", "content": "f"}, context)
    await root.reset(None, context)
    resolve(gate)
    await follow_up.wait(context)
    assert matches(await status(input), {"status": "unanswered", "reason": "reset"})
    assert transcript(await all_entries(root)) == [
        "pi.user:a",
        "pi.assistant",
        "pi.tool-result",
        "pi.reset",
        "pi.user:f",
        "pi.assistant:fresh",
    ]
    reset = next(entry for entry in await all_entries(root) if ResetEntry.is_(entry))
    assert reset["head"] == reset["id"]
    # The follow-up's request starts at the reset: the complete system baseline after the cut leads the follow-up.
    assert requests == [["system:", "user:f"]]
    assert setup.faux.state.call_count == 2
    await harness.close(context)


async def test_places_a_queued_reset_after_the_answer_at_the_final_boundary() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    await root.reset("handoff", context)
    first.release()
    await input.wait(context)
    await harness.wait_for_idle(context)
    assert matches(await status(input), {"status": "done"})
    assert transcript(await all_entries(root)) == ["pi.user:a", "pi.assistant:first", "pi.reset:handoff"]
    messages = (await root.context(context))["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], UserMessage) and messages[0].content == "handoff"
    assert isinstance(messages[0].timestamp, int)
    await harness.close(context)


async def test_resets_an_idle_conversation_at_once_with_or_without_handoff_text() -> None:
    setup = chat_setup()
    setup.now = lambda: 7
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)
    await root.reset(None, context)
    view = await root.context(context)
    assert view["head"]["kind"] == "pi.reset"
    assert "model" not in view["head"]
    assert view["messages"] == []
    await root.reset("carry on", context)
    view = await root.context(context)
    assert view["head"]["head"] == view["head"]["id"]
    assert view["messages"] == [UserMessage("carry on", timestamp=7)]
    await harness.close(context)


async def test_makes_a_queued_head_write_stale_when_it_targets_an_entry_before_the_active_range() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("second")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    old = await root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)
    await root.reset(None, context)
    reset = (await root.context(context))["head"]
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    stale = await root.submit({"type": "write", "entry": {"kind": "summary", "head": old["id"]}}, context)
    fresh = await root.submit({"type": "write", "entry": {"kind": "summary", "head": reset["id"]}}, context)
    first.release()
    await input.wait(context)
    assert matches(await status(stale), {"status": "unanswered", "reason": "stale"})
    assert matches(await status(fresh), {"status": "done"})
    assert await inbox(harness, root) == []

    # The fresh summary's marker starts the range at the reset: a target inside the range is not stale, even when
    # it is older than the marker itself. A reset placed earlier in the same boundary makes it stale.
    inside = next(entry for entry in await all_entries(root) if entry["kind"] == "pi.user")
    second = await root.submit({"type": "input", "content": "b"}, context)
    kept = await root.submit({"type": "write", "entry": {"kind": "summary", "head": inside["id"]}}, context)
    await second.wait(context)
    assert matches(await status(kept), {"status": "done"})
    await harness.close(context)


async def test_makes_a_head_write_stale_behind_a_reset_placed_earlier_in_the_same_boundary() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    target = (await all_entries(root))[0]
    await root.reset(None, context)
    summary = await root.submit({"type": "write", "entry": {"kind": "summary", "head": target["id"]}}, context)
    first.release()
    await input.wait(context)
    assert matches(await status(summary), {"status": "unanswered", "reason": "stale"})
    await harness.close(context)


async def test_ends_the_run_with_a_pi_reset_entry_when_a_tool_requests_a_handoff() -> None:
    setup = chat_setup()
    gate = deferred()
    resolve(gate)
    hold_tool(setup, gate, {"output": [], "control": {"handoff": "continue here"}})
    setup.faux.set_responses([HOLD])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    settled = await (await root.submit({"type": "input", "content": "a"}, context)).wait(context)
    entries = await all_entries(root)
    calling = next(entry for entry in entries if entry["kind"] == "pi.assistant")
    assert matches(settled, {"status": "done", "answer": calling["id"]})
    assert transcript(entries) == ["pi.user:a", "pi.assistant", "pi.tool-result", "pi.reset:continue here"]
    assert entries[-1]["head"] == entries[-1]["id"]
    assert setup.faux.state.call_count == 1
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    await harness.close(context)


def _continue_once(text: str) -> Any:
    yields = [0]

    def on_yield(*_args: Any) -> Any:
        yields[0] += 1
        return {"continue": text} if yields[0] == 1 else None

    return on_yield


async def test_drops_an_on_yield_continuation_when_the_final_boundary_selects_a_follow_up() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("second")])
    add_hooks(setup.registry, GenerationTask, {"on_yield": _continue_once("more")})
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    follow_up = await root.submit({"type": "input", "content": "f"}, context)
    first.release()
    await follow_up.wait(context)
    assert matches(await status(input), {"status": "done"})
    assert transcript(await all_entries(root)) == [
        "pi.user:a",
        "pi.assistant:first",
        "pi.user:f",
        "pi.assistant:second",
    ]
    await harness.close(context)


async def test_withdraws_a_queued_submission_and_removes_its_item() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    kept = await root.submit({"type": "write", "entry": queued_note("kept")}, context)
    withdrawn = await root.submit({"type": "input", "content": "f"}, context)
    assert await withdrawn.abort(context) == "aborted"
    assert matches(await withdrawn.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert await inbox(harness, root) == [[kept.id, "write"]]
    first.release()
    await input.wait(context)
    assert matches(await status(kept), {"status": "done"})
    assert setup.faux.state.call_count == 1
    await harness.close(context)


async def test_leaves_queued_items_after_a_failed_run_until_the_next_submission_places_them_in_order() -> None:
    setup = chat_setup()
    failing = Gated(FAILED)
    setup.faux.set_responses([failing.step, answer("for f"), answer("for g")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await failing.reached
    f = await root.submit({"type": "input", "content": "f"}, context)
    failing.release()
    assert matches(await input.wait(context), {"status": "unanswered", "reason": "model_error"})
    await harness.wait_for_idle(context)
    assert (await status(f))["status"] == "queued"
    assert await inbox(harness, root) == [[f.id, "followUp"]]

    # Idle with a queued item: the new input queues behind it, and a final boundary places the older one first.
    g = await root.submit({"type": "input", "content": "g", "whenBusy": "reject"}, context)
    await g.wait(context)
    assert matches(await status(f), {"status": "done"})
    assert transcript(await all_entries(root))[-4:] == [
        "pi.user:f",
        "pi.assistant:for f",
        "pi.user:g",
        "pi.assistant:for g",
    ]
    await harness.close(context)


async def test_keeps_queued_submissions_across_reopen_and_settles_them_afterwards(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("first again"), answer("f")])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    await opened.root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    f = (await opened.root.submit({"type": "input", "content": "f"}, context)).id
    await opened.harness.close(context)

    opened = await open_chat(await open_sqlite_storage(path), setup)
    submission = await opened.harness.submission(f, context)
    assert submission is not None
    assert matches(await submission.wait(context), {"status": "done"})
    assert transcript(await all_entries(opened.root))[-2:] == ["pi.user:f", "pi.assistant:f"]
    await opened.harness.close(context)


async def test_commits_inbox_changes_as_positional_chord_operations_and_a_base_when_empty() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("second")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    ops = inbox_ops(harness)
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    w1 = await root.submit({"type": "write", "entry": queued_note("w")}, context)
    f1 = await root.submit({"type": "input", "content": "f1"}, context)
    w2 = await root.submit({"type": "write", "entry": queued_note("w")}, context)
    f2 = await root.submit({"type": "input", "content": "f2"}, context)
    w3 = await root.submit({"type": "write", "entry": queued_note("w")}, context)
    assert ops == [
        [("p", ("items",), 0, 0, [{"id": w1.id, "mode": "write", "entry": queued_note_json("w")}])],
        [("p", ("items",), 1, 0, [{"id": f1.id, "mode": "followUp", "content": "f1"}])],
        [("p", ("items",), 2, 0, [{"id": w2.id, "mode": "write", "entry": queued_note_json("w")}])],
        [("p", ("items",), 3, 0, [{"id": f2.id, "mode": "followUp", "content": "f2"}])],
        [("p", ("items",), 4, 0, [{"id": w3.id, "mode": "write", "entry": queued_note_json("w")}])],
    ]
    first.release()
    await input.wait(context)
    # Every write and the first follow-up leave; only f2 at index 3 remains. No retained value is carried.
    assert all(op[0] == "p" and len(op[4]) == 0 for op in ops[5])
    assert ("p", ("items",), 4, 1, []) in ops[5]
    assert "f2" not in json.dumps([list(op) for op in ops[5]])
    await f2.wait(context)
    await harness.close(context)


async def test_settles_a_stale_write_at_once_while_idle_and_a_queued_one_behind_waiting_items() -> None:
    setup = chat_setup()
    failing = Gated(FAILED)
    setup.faux.set_responses([failing.step, answer("for f")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    old = await root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)
    await root.reset(None, context)
    before = len(await all_entries(root))
    idle = await root.submit({"type": "write", "entry": {"kind": "summary", "head": old["id"]}}, context)
    assert matches(await idle.wait(context), {"status": "unanswered", "reason": "stale"})
    assert len(await all_entries(root)) == before

    # After a failed run, a follow-up waits; a stale write queues behind it and the boundary rejects it.
    input = await root.submit({"type": "input", "content": "a"}, context)
    await failing.reached
    f = await root.submit({"type": "input", "content": "f"}, context)
    failing.release()
    await input.wait(context)
    await harness.wait_for_idle(context)
    queued = await root.submit({"type": "write", "entry": {"kind": "summary", "head": old["id"]}}, context)
    assert matches(await queued.wait(context), {"status": "unanswered", "reason": "stale"})
    assert matches(await f.wait(context), {"status": "done"})
    await harness.close(context)


async def test_keeps_an_on_yield_continuation_across_a_queued_plain_write_with_the_runs_original_input() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("second")])
    add_hooks(setup.registry, GenerationTask, {"on_yield": _continue_once("more")})
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    write = await root.submit({"type": "write", "entry": queued_note("w")}, context)
    first.release()
    await input.wait(context)
    entries = await all_entries(root)
    assert transcript(entries) == [
        "pi.user:a",
        "pi.assistant:first",
        "note:w",
        "pi.user:more",
        "pi.assistant:second",
    ]
    assert matches(await status(input), {"status": "done", "answer": entries[-1]["id"]})
    assert matches(await status(write), {"status": "done"})
    await harness.close(context)


async def test_drops_an_on_yield_continuation_for_a_queued_reset() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step])
    add_hooks(setup.registry, GenerationTask, {"on_yield": lambda *_: {"continue": "more"}})
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    await root.reset(None, context)
    first.release()
    await input.wait(context)
    await harness.wait_for_idle(context)
    assert matches(await status(input), {"status": "done"})
    assert transcript(await all_entries(root)) == ["pi.user:a", "pi.assistant:first", "pi.reset"]
    assert setup.faux.state.call_count == 1
    await harness.close(context)


async def test_adds_every_steer_to_the_run_at_the_post_tools_boundary_with_steering_mode_all() -> None:
    setup = chat_setup()
    gate = deferred()
    hold_tool(setup, gate)
    setup.faux.set_responses([HOLD, answer("after tools"), answer("follow-up")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    setup.settings["steeringMode"] = "all"
    input = await root.submit({"type": "input", "content": "a"}, context)
    await tool_running(harness, root)
    s1 = await root.submit({"type": "input", "content": "s1", "whenBusy": "steer"}, context)
    f = await root.submit({"type": "input", "content": "f"}, context)
    s2 = await root.submit({"type": "input", "content": "s2", "whenBusy": "steer"}, context)
    resolve(gate)
    await f.wait(context)
    entries = await all_entries(root)
    assert transcript(entries) == [
        "pi.user:a",
        "pi.assistant",
        "pi.tool-result",
        "pi.user:s1",
        "pi.user:s2",
        "pi.assistant:after tools",
        "pi.user:f",
        "pi.assistant:follow-up",
    ]
    answers = [entry for entry in entries if entry["kind"] == "pi.assistant"]
    for submission in (input, s1, s2):
        assert matches(await status(submission), {"status": "done", "answer": answers[1]["id"]})
    await harness.close(context)


async def test_queues_an_idle_steer_behind_waiting_items_and_places_it_with_the_first_follow_up_in_id_order() -> None:
    setup = chat_setup()
    failing = Gated(FAILED)
    setup.faux.set_responses([failing.step, answer("both")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await failing.reached
    f = await root.submit({"type": "input", "content": "f"}, context)
    failing.release()
    await input.wait(context)
    await harness.wait_for_idle(context)
    steer = await root.submit({"type": "input", "content": "s", "whenBusy": "steer"}, context)
    settled = await steer.wait(context)
    assert settled["status"] == "done"
    assert matches(await status(f), {"status": "done", "answer": settled["answer"]})
    assert transcript(await all_entries(root))[-3:] == ["pi.user:f", "pi.user:s", "pi.assistant:both"]
    await harness.close(context)


async def test_starts_a_queued_follow_up_after_a_terminating_round() -> None:
    setup = chat_setup()
    gate = deferred()
    hold_tool(setup, gate, {"output": [], "control": {"terminate": True}})
    setup.faux.set_responses([HOLD, answer("follow-up")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await tool_running(harness, root)
    f = await root.submit({"type": "input", "content": "f"}, context)
    resolve(gate)
    await f.wait(context)
    calling = next(entry for entry in await all_entries(root) if entry["kind"] == "pi.assistant")
    assert matches(await status(input), {"status": "done", "answer": calling["id"]})
    assert transcript(await all_entries(root)) == [
        "pi.user:a",
        "pi.assistant",
        "pi.tool-result",
        "pi.user:f",
        "pi.assistant:follow-up",
    ]
    await harness.close(context)


async def test_writes_the_last_handoff_in_call_order_and_then_runs_queued_follow_ups_in_the_new_context() -> None:
    setup = chat_setup()
    first_gate = deferred()
    second_gate = deferred()
    hold_tool(setup, first_gate, {"output": [], "control": {"handoff": "one"}})

    async def later(*_args: Any) -> Any:
        await second_gate
        return {"output": [], "control": {"handoff": "two"}}

    add_tool(
        setup.registry,
        define_tool(name="later", description="Finishes first", parameters=Type.Object({}), execute=later),
    )
    round = faux_assistant_message(
        [faux_tool_call("hold", {}, id="c1"), faux_tool_call("later", {}, id="c2")], stop_reason="toolUse"
    )
    setup.faux.set_responses([round, answer("follow-up")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await tool_running(harness, root)
    f = await root.submit({"type": "input", "content": "f"}, context)
    resolve(second_gate)
    await tool_running(harness, root, 1, "done")
    resolve(first_gate)
    await f.wait(context)
    assert matches(await status(input), {"status": "done"})
    assert transcript(await all_entries(root))[-4:] == [
        "pi.tool-result",
        "pi.reset:two",
        "pi.user:f",
        "pi.assistant:follow-up",
    ]
    await harness.close(context)


async def test_leaves_the_inbox_alone_when_the_runs_task_is_aborted() -> None:
    setup = chat_setup()
    first = Gated(answer("never"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    f = await root.submit({"type": "input", "content": "f"}, context)
    task_id = (await harness.snapshot(LiveDoc, root.id, context))["run"]["taskId"]
    await harness.abort_task(task_id, context)
    assert matches(await input.wait(context), {"status": "unanswered", "reason": "aborted"})
    await harness.wait_for_idle(context)
    assert (await status(f))["status"] == "queued"
    assert await inbox(harness, root) == [[f.id, "followUp"]]
    await harness.close(context)


async def test_returns_a_queued_submission_for_its_repeated_request_id_without_a_second_item() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    queued = await root.submit({"type": "input", "content": "f", "requestId": "r"}, context)
    again = await root.submit({"type": "input", "content": "f", "requestId": "r"}, context)
    assert again.id == queued.id
    assert await inbox(harness, root) == [[queued.id, "followUp"]]
    await harness.close(context)


async def test_withdraws_a_middle_item_with_one_positional_removal() -> None:
    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    items = [await root.submit({"type": "input", "content": text}, context) for text in ("x", "y", "z")]
    ops = inbox_ops(harness)
    await items[1].abort(context)
    assert ops == [[("p", ("items",), 1, 1, [])]]
    assert await inbox(harness, root) == [[items[0].id, "followUp"], [items[2].id, "followUp"]]
    await harness.close(context)


async def test_stores_the_inbox_as_a_base_exactly_when_it_becomes_empty() -> None:
    written: list[dict[str, Any]] = []
    inbox_id: list[Any] = [None]

    class RecordingStorage(MemoryStorage):
        async def commit(self, writes: Any, context: Any) -> Any:
            for write in writes:
                if write["type"] != "document.change" or write["id"] != inbox_id[0]:
                    continue
                content = write["content"]
                empty = content["kind"] == "base" and len(content["value"]["items"]) == 0
                written.append({"kind": content["kind"], "empty": empty})
            return await super().commit(writes, context)

    setup = chat_setup()
    first = Gated(answer("first"))
    setup.faux.set_responses([first.step, answer("second")])
    opened = await open_chat(RecordingStorage(), setup)
    harness, root = opened.harness, opened.root

    def learn(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            if change["record"]["kind"] == "pi.inbox":
                inbox_id[0] = change["record"]["id"]

    harness.subscribe_commits(learn)
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    await root.submit({"type": "input", "content": "f1"}, context)
    f2 = await root.submit({"type": "input", "content": "f2"}, context)
    first.release()
    await input.wait(context)
    await f2.wait(context)
    # The inbox ID is learned from the f1 push; later writes: the f2 push, removing f1, and emptying the inbox.
    assert written == [
        {"kind": "delta", "empty": False},
        {"kind": "delta", "empty": False},
        {"kind": "base", "empty": True},
    ]
    await harness.close(context)


def test_keeps_a_complete_inbox_base_exactly_while_it_is_empty_and_a_usage_base_on_every_change() -> None:
    info: Any = {"deltasSinceBase": 1000}
    item: Any = {"id": 1, "mode": "followUp", "content": "x"}
    inbox_when: Any = InboxDoc.definition.checkpoint_when
    usage_when: Any = UsageDoc.definition.checkpoint_when
    assert inbox_when({"items": []}, [], info) is True
    assert inbox_when({"items": [item]}, [], info) is False
    assert usage_when({"models": {}, "tools": {}}, [], info) is True


# ─── usage ──────────────────────────────────────────────────────────────────


async def test_totals_assistant_usage_per_model_and_tool_usage_per_tool_and_sums_the_session() -> None:
    setup = chat_setup()
    spent = Usage(1, 2, 3, 4, 10, UsageCost(0.1, 0.2, 0.3, 0.4, 1))
    gate = deferred()
    resolve(gate)
    hold_tool(setup, gate, {"output": [], "usage": spent})
    setup.faux.set_responses([HOLD, answer("done"), answer("other")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await (await root.submit({"type": "input", "content": "a"}, context)).wait(context)
    entries = await all_entries(root)
    assistants = assistants_of(entries)
    expected = {
        "input": sum(message.usage.input for message in assistants),
        "output": sum(message.usage.output for message in assistants),
        "totalTokens": sum(message.usage.total_tokens for message in assistants),
    }
    usage = await harness.snapshot(UsageDoc, root.id, context)
    assert matches(usage["models"]["faux/faux-1"], expected)
    assert usage["tools"] == {"hold": usage_json(spent)}
    result = next(entry for entry in entries if entry["kind"] == "pi.tool-result")["model"][0]
    assert result.usage == spent

    # A fork starts at zero; the Session total adds every conversation once.
    fork = await root.fork(entries[-1]["id"], {"ownership": {"kind": "ownerless"}}, context)
    assert await harness.snapshot(UsageDoc, fork.id, context) == {"models": {}, "tools": {}}
    await (await fork.submit({"type": "input", "content": "b"}, context)).wait(context)
    fork_usage = (await harness.snapshot(UsageDoc, fork.id, context))["models"]["faux/faux-1"]
    total = await harness.usage(context)
    assert total["models"]["faux/faux-1"]["output"] == expected["output"] + fork_usage["output"]
    assert total["tools"] == {"hold": usage_json(spent)}
    await harness.close(context)


async def test_counts_failed_attempts_converted_partials_and_tool_usage_replaced_by_after_tool() -> None:
    setup = chat_setup(tokens_per_second=200, token_size={"min": 1, "max": 1})
    spent = Usage(5, 0, 0, 0, 5, UsageCost())
    gate = deferred()
    resolve(gate)
    hold_tool(setup, gate)
    add_hooks(setup.registry, ToolTask, {"after_tool": lambda _call, result, *_: {**result, "usage": spent}})
    retryable = faux_assistant_message([], stop_reason="error", error_message="503 Service Unavailable")
    setup.faux.set_responses([retryable, HOLD, answer("done"), answer("x" * 400)])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    setup.settings["retry"] = {"enabled": True, "maxRetries": 1, "baseDelayMs": 1}
    await (await root.submit({"type": "input", "content": "a"}, context)).wait(context)
    assert (await harness.snapshot(UsageDoc, root.id, context))["tools"] == {"hold": usage_json(spent)}

    # A partial committed while streaming becomes an aborted entry on abort; its usage counts too.
    input = await root.submit({"type": "input", "content": "b"}, context)
    await live_partial(harness, root)
    await harness.abort_task((await harness.snapshot(LiveDoc, root.id, context))["run"]["taskId"], context)
    await input.wait(context)
    assistants = assistants_of(await all_entries(root))
    assert [message.stop_reason for message in assistants] == ["error", "toolUse", "stop", "aborted"]
    output = sum(message.usage.output for message in assistants)
    assert (await harness.snapshot(UsageDoc, root.id, context))["models"]["faux/faux-1"]["output"] == output
    await harness.close(context)


async def test_keeps_tools_named_like_object_prototype_keys_in_the_ledger_and_the_session_total() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    usage = Usage(1, 1, 0, 0, 2, UsageCost())
    names = ["constructor", "__proto__", "toString"]
    for name in names:
        await root.commit(lambda tx, name=name: record_usage(tx, root.id, "tools", name, usage), context)
        await root.commit(lambda tx, name=name: record_usage(tx, root.id, "tools", name, usage), context)
    total = await harness.usage(context)
    assert list(total["tools"]) == names
    for name in names:
        assert total["tools"][name]["output"] == 2
    await harness.close(context)


async def test_keeps_usage_totals_exact_across_reopen_counting_a_partial_converted_after_reopen_once(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "session.sqlite")
    setup = chat_setup(tokens_per_second=200, token_size={"min": 1, "max": 1})
    setup.faux.set_responses([answer("first"), answer("x" * 400), answer("again")])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    await (await opened.root.submit({"type": "input", "content": "a"}, context)).wait(context)
    await opened.root.submit({"type": "input", "content": "b"}, context)
    await live_partial(opened.harness, opened.root)
    # Closing mid-stream keeps the committed partial; the reopened request converts it into an aborted entry.
    await opened.harness.close(context)
    opened = await open_chat(await open_sqlite_storage(path), setup)
    opened.harness.resume()
    await opened.harness.wait_for_idle(context)
    assistants = assistants_of(await all_entries(opened.root))
    assert [message.stop_reason for message in assistants] == ["stop", "aborted", "stop"]
    total = (await opened.harness.usage(context))["models"]["faux/faux-1"]
    assert [total["input"], total["output"], total["totalTokens"]] == [
        sum(message.usage.input for message in assistants),
        sum(message.usage.output for message in assistants),
        sum(message.usage.total_tokens for message in assistants),
    ]
    await opened.harness.close(context)


async def test_records_usage_as_numeric_sets_on_the_ledger_in_the_entrys_commit() -> None:
    setup = chat_setup()
    setup.faux.set_responses([answer("one"), answer("two")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    commits: list[dict[str, Any]] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            if change["record"]["kind"] != "pi.usage":
                continue
            entries = len([other for other in publication["changes"] if other["type"] == "entry"])
            commits.append({"entries": entries, "ops": change["ops"]})

    harness.subscribe_commits(listener)
    await (await root.submit({"type": "input", "content": "a"}, context)).wait(context)
    await (await root.submit({"type": "input", "content": "b"}, context)).wait(context)
    assert len(commits) == 2
    first_ops = commits[0]["ops"]
    assert len(first_ops) == 1
    assert first_ops[0][0] == "s" and tuple(first_ops[0][1]) == ("models", "faux/faux-1")
    assert isinstance(first_ops[0][2], dict)
    assert all(op[0] == "s" and op[1][1] == "faux/faux-1" for op in commits[1]["ops"])
    assert all(commit["entries"] >= 1 for commit in commits)
    await harness.close(context)
