"""Port of ``test/harness-input-entries.test.ts``: inputs placed as application entries, and data-only writes
appended at once (#10407)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import (
    ChatSetup,
    add_section,
    add_tool,
    all_entries,
    assert_match,
    chat_setup,
    gated,
    open_chat,
    text_of,
    wait_for,
)
from durable_session_support import context, deferred, resolve

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage, UserMessage
from pi_durable import ConversationBusy, Harness, InboxDoc, LiveDoc, MemoryStorage, define_tool, watch_events
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


def answer(text: str) -> AssistantMessage:
    return faux_assistant_message([faux_text(text)])


def hold_tool(setup: ChatSetup, gate: asyncio.Future[Any]) -> None:
    """Register a ``hold`` tool whose calls wait for ``gate``."""

    async def execute(*_args: Any) -> Any:
        await gate
        return {"output": [{"type": "text", "text": "held"}]}

    add_tool(
        setup.registry,
        define_tool(name="hold", description="Waits for the test", parameters=Type.Object({}), execute=execute),
    )


HOLD = faux_assistant_message([faux_tool_call("hold", {}, id="c1")], stop_reason="toolUse")


def request_log(requests: list[list[str]], reply: AssistantMessage) -> Any:
    """Role and text of each request message, as the provider receives them."""

    def step(request: Any, *_rest: Any) -> AssistantMessage:
        requests.append([f"{message.role}:{text_of(message) or ''}" for message in request.messages])
        return reply

    return step


def transcript(entries: Any) -> list[list[Any]]:
    """Kind, application data, and model text of each entry, skipping system entries."""
    shown: list[list[Any]] = []
    for entry in entries:
        if entry["kind"] == "pi.system":
            continue
        model = entry.get("model")
        message = model[0] if model else None
        text = None if message is None or message.role == "toolResult" else text_of(message)
        data = [] if "data" not in entry or entry["kind"].startswith("pi.") else [entry["data"]]
        shown.append([entry["kind"], *data, *([] if text is None else [text])])
    return shown


async def inbox_items(harness: Harness, root: Any) -> list[Any]:
    snapshot = await harness.snapshot(InboxDoc, root.id, context)
    assert snapshot is not None
    return snapshot["items"]


async def tool_running(harness: Harness, root: Any) -> None:
    async def check() -> bool:
        live = await harness.snapshot(LiveDoc, root.id, context)
        tools = None if live is None else live.get("tools")
        return bool(tools) and tools[0].get("status") == "running"  # type: ignore[index]

    await wait_for(check)


def mention(text: str) -> dict[str, Any]:
    return {"kind": "chat.mention", "data": {"author": "ann", "text": text}}


# ─── input entries ──────────────────────────────────────────────────────────


async def test_places_an_idle_input_as_its_entry_the_given_kind_and_data_with_the_content_as_its_user_message() -> None:
    setup = chat_setup()
    setup.now = lambda: 42
    requests: list[list[str]] = []
    setup.faux.set_responses([request_log(requests, answer("hi ann"))])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "ann: hi", "entry": mention("hi")}, context)
    settled = await submission.wait(context)
    entries = await all_entries(root)
    assert transcript(entries) == [
        ["chat.mention", {"author": "ann", "text": "hi"}, "ann: hi"],
        ["pi.assistant", "hi ann"],
    ]
    placed = next(entry for entry in entries if entry["kind"] == "chat.mention")
    assert placed["model"] == [UserMessage("ann: hi", timestamp=42)]
    assert_match(settled, {"status": "done", "entry": placed["id"], "answer": entries[-1]["id"]})
    assert requests == [["user:ann: hi"]]
    # The context of a fork at the input is the same as for a pi.user entry.
    view = await root.context(context, at=placed["id"])
    assert [message for message in view["messages"] if message.role != "system"] == placed["model"]
    await harness.close(context)


async def test_rejects_an_input_entry_whose_kind_is_not_a_non_empty_string_or_starts_with_pi_writing_nothing() -> None:
    setup = chat_setup()
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    before = len(await all_entries(root))
    for kind in (123, "", "pi.user", "pi.compaction"):
        draft: Any = {"type": "input", "content": "x", "entry": {"kind": kind}}
        with pytest.raises(Exception, match="An input entry's kind must be"):
            await root.submit(draft, context)
    assert len(await all_entries(root)) == before
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    # Writes keep their kinds: reset() writes pi.reset.
    await root.reset(None, context)
    assert (await all_entries(root))[-1]["kind"] == "pi.reset"
    await harness.close(context)


async def test_takes_only_kind_and_data_from_an_untyped_callers_entry() -> None:
    setup = chat_setup()
    first = gated(answer("first"))
    setup.faux.set_responses([answer("idle"), first.step, answer("queued")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    extra = {
        "kind": "chat.mention",
        "model": [UserMessage("forged", timestamp=0)],
        "head": "self",
        "edits": [{"target": 1, "action": "omit"}],
    }

    def draft(content: str) -> Any:
        return {"type": "input", "content": content, "entry": extra}

    await (await root.submit(draft("idle"), context)).wait(context)
    await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    queued = await root.submit(draft("later"), context)
    assert (await inbox_items(harness, root))[-1] == {
        "id": queued.id,
        "mode": "followUp",
        "content": "later",
        "entry": {"kind": "chat.mention"},
    }
    first.release()
    await queued.wait(context)
    mentions = [entry for entry in await all_entries(root) if entry["kind"] == "chat.mention"]
    assert [
        [entry.get("head"), entry.get("edits"), text_of(entry["model"][0]), len(entry["model"])] for entry in mentions
    ] == [
        [None, None, "idle", 1],
        [None, None, "later", 1],
    ]
    await harness.close(context)


async def test_queues_busy_inputs_with_their_entries_in_the_inbox_and_places_them_as_those_entries_at_boundaries() -> (
    None
):
    setup = chat_setup()
    gate = deferred()
    hold_tool(setup, gate)
    setup.faux.set_responses([HOLD, answer("after tools"), answer("follow-up")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await tool_running(harness, root)
    steer = await root.submit(
        {"type": "input", "content": "s", "entry": mention("s"), "whenBusy": "steer", "requestId": "r-s"}, context
    )
    follow_up = await root.submit(
        {"type": "input", "content": "f", "entry": {"kind": "p2.input"}, "whenBusy": "followUp"}, context
    )
    assert await inbox_items(harness, root) == [
        {
            "id": steer.id,
            "mode": "steer",
            "content": "s",
            "entry": {"kind": "chat.mention", "data": {"author": "ann", "text": "s"}},
        },
        {"id": follow_up.id, "mode": "followUp", "content": "f", "entry": {"kind": "p2.input"}},
    ]
    # The request ID deduplicates a queued input with an entry as any other.
    again = await root.submit(
        {"type": "input", "content": "other", "entry": mention("other"), "whenBusy": "steer", "requestId": "r-s"},
        context,
    )
    assert again.id == steer.id
    with pytest.raises(ConversationBusy):
        await root.submit({"type": "input", "content": "x", "entry": mention("x"), "whenBusy": "reject"}, context)
    assert len(await inbox_items(harness, root)) == 2

    resolve(gate)
    await follow_up.wait(context)
    entries = await all_entries(root)
    assert transcript(entries) == [
        ["pi.user", "a"],
        ["pi.assistant"],
        ["pi.tool-result"],
        ["chat.mention", {"author": "ann", "text": "s"}, "s"],
        ["pi.assistant", "after tools"],
        ["p2.input", "f"],
        ["pi.assistant", "follow-up"],
    ]
    answers = [entry for entry in entries if entry["kind"] == "pi.assistant"]

    def placed(kind: str) -> Any:
        return next(entry for entry in entries if entry["kind"] == kind)["id"]

    assert_match(await input.status(context), {"status": "done", "answer": answers[1]["id"]})
    assert_match(
        await steer.status(context), {"status": "done", "entry": placed("chat.mention"), "answer": answers[1]["id"]}
    )
    assert_match(
        await follow_up.status(context), {"status": "done", "entry": placed("p2.input"), "answer": answers[2]["id"]}
    )
    await harness.close(context)


async def test_withdraws_queued_inputs_with_entries_one_by_one_and_by_conversation_abort() -> None:
    setup = chat_setup()
    first = gated(answer("first"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    withdrawn = await root.submit({"type": "input", "content": "w", "entry": mention("w")}, context)
    aborted = await root.submit({"type": "input", "content": "x", "entry": mention("x")}, context)
    assert await withdrawn.abort(context) == "aborted"
    assert [item["id"] for item in await inbox_items(harness, root)] == [aborted.id]
    await root.abort(context)
    for submission in (withdrawn, aborted):
        assert_match(await submission.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert_match(await input.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert await inbox_items(harness, root) == []
    assert not any(entry["kind"] == "chat.mention" for entry in await all_entries(root))
    await harness.close(context)


# ─── data-only writes ───────────────────────────────────────────────────────


async def test_appends_a_data_only_write_mid_run_in_time_order_leaving_the_models_messages_unchanged() -> None:
    setup = chat_setup()
    gate = deferred()
    hold_tool(setup, gate)
    requests: list[list[str]] = []
    setup.faux.set_responses([request_log(requests, HOLD), request_log(requests, answer("done"))])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    events: list[Any] = []
    stream = await watch_events(harness, root.id, context)

    async def listener(batch: Any, _context: Any) -> None:
        events.extend(batch)

    stream.start(listener)
    input = await root.submit({"type": "input", "content": "a"}, context)
    await tool_running(harness, root)
    chat = {"kind": "chat.message", "data": {"author": "bob", "text": "lunch?"}}
    write = await root.submit({"type": "write", "entry": chat, "requestId": "bob:1"}, context)
    # Settled and appended in its own commit, between the call and its still-running tool's result.
    settled = await write.status(context)
    assert settled["status"] == "done"
    assert await inbox_items(harness, root) == []
    assert transcript(await all_entries(root)) == [
        ["pi.user", "a"],
        ["pi.assistant"],
        ["chat.message", {"author": "bob", "text": "lunch?"}],
    ]
    await wait_for(lambda: any(event["type"] == "entry_appended" for event in events))
    appended = next(event for event in events if event["type"] == "entry_appended")
    assert_match(appended, {"entry": {"kind": "chat.message", "id": settled["entry"]}})
    assert not any(event["type"] == "run_end" for event in events)
    # A repeated request ID returns the same write without a second entry.
    assert (await root.submit({"type": "write", "entry": chat, "requestId": "bob:1"}, context)).id == write.id

    resolve(gate)
    await input.wait(context)
    assert transcript(await all_entries(root)) == [
        ["pi.user", "a"],
        ["pi.assistant"],
        ["chat.message", {"author": "bob", "text": "lunch?"}],
        ["pi.tool-result"],
        ["pi.assistant", "done"],
    ]
    assert requests == [
        ["system:", "user:a"],
        ["system:", "user:a", "assistant:", "toolResult:held"],
    ]
    await stream.stop()
    await harness.close(context)


async def test_queues_a_data_only_write_behind_a_queued_reset_so_it_lands_after_the_reset_and_stays_visible() -> None:
    setup = chat_setup()
    first = gated(answer("first"))
    setup.faux.set_responses([first.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    await root.reset(None, context)
    chat = await root.submit({"type": "write", "entry": {"kind": "chat.message", "data": "hi"}}, context)
    assert (await chat.status(context))["status"] == "queued"
    first.release()
    await input.wait(context)
    settled = await chat.wait(context)
    entries = await all_entries(root)
    reset = next(entry for entry in entries if entry["kind"] == "pi.reset")
    assert settled["status"] == "done" and settled["entry"] > reset["id"]
    assert [entry["kind"] for entry in (await root.context(context))["entries"]] == ["pi.reset", "chat.message"]
    # Once the reset is placed, display-only writes go in at once again.
    later = await root.submit({"type": "write", "entry": {"kind": "chat.message", "data": "again"}}, context)
    assert (await later.status(context))["status"] == "done"
    await harness.close(context)


async def test_leaves_the_request_unchanged_by_a_data_only_write_that_lands_while_the_generation_prepares() -> None:
    setup = chat_setup()
    rendering = deferred()
    render = deferred()
    renders = [0]

    async def slow(*_args: Any) -> str:
        renders[0] += 1
        if renders[0] == 2:
            resolve(rendering)
            await render
        return "SLOW"

    add_section(setup.registry, "slow", slow)
    requests: list[list[str]] = []
    setup.faux.set_responses([request_log(requests, answer("one")), request_log(requests, answer("two"))])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await (await root.submit({"type": "input", "content": "a"}, context)).wait(context)
    second = await root.submit({"type": "input", "content": "b"}, context)
    await rendering
    await root.submit({"type": "write", "entry": {"kind": "chat.message", "data": "hi"}}, context)
    resolve(render)
    await second.wait(context)
    assert requests[1] == ["system:", "user:a", "assistant:one", "user:b"]
    kinds = [entry["kind"] for entry in await all_entries(root)]
    # The unchanged prompt adds no second pi.system entry.
    assert kinds == ["pi.user", "pi.system", "pi.assistant", "pi.user", "chat.message", "pi.assistant"]
    await harness.close(context)


async def test_appends_a_data_only_write_while_idle_after_a_failed_run_without_placing_the_queued_items() -> None:
    setup = chat_setup()
    failing = gated(faux_assistant_message([], stop_reason="error", error_message="invalid request"))
    setup.faux.set_responses([failing.step, answer("for f")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await failing.reached
    f = await root.submit({"type": "input", "content": "f", "entry": mention("f")}, context)
    failing.release()
    assert_match(await input.wait(context), {"status": "unanswered", "reason": "model_error"})
    await harness.wait_for_idle(context)
    note = await root.submit({"type": "write", "entry": {"kind": "note", "data": "n"}}, context)
    assert (await note.status(context))["status"] == "done"
    await harness.wait_for_idle(context)
    assert (await f.status(context))["status"] == "queued"
    assert [item["id"] for item in await inbox_items(harness, root)] == [f.id]
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    assert setup.faux.state.call_count == 1
    # The next input places the waiting mention first.
    g = await root.submit({"type": "input", "content": "g"}, context)
    await g.wait(context)
    assert [row for row in transcript(await all_entries(root)) if row[0] != "pi.assistant"] == [
        ["pi.user", "a"],
        ["note", "n"],
        ["chat.mention", {"author": "ann", "text": "f"}, "f"],
        ["pi.user", "g"],
    ]
    await harness.close(context)


async def test_keeps_a_queued_inputs_entry_across_reopen_and_places_it_as_that_entry(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    setup = chat_setup()
    first = gated(answer("first"))
    setup.faux.set_responses([first.step, answer("first again"), answer("for f")])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    await opened.root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    f = (await opened.root.submit({"type": "input", "content": "f", "entry": mention("f")}, context)).id
    await opened.harness.close(context)

    opened = await open_chat(await open_sqlite_storage(path), setup)
    item = next(item for item in await inbox_items(opened.harness, opened.root) if item["id"] == f)
    assert_match(item, {"entry": mention("f")})
    submission = await opened.harness.submission(f, context)
    assert submission is not None
    settled = await submission.wait(context)
    placed = next(entry for entry in await all_entries(opened.root) if entry["kind"] == "chat.mention")
    assert_match(settled, {"status": "done", "entry": placed["id"]})
    assert_match(placed, {"data": mention("f")["data"], "model": [{"content": "f"}]})
    assert placed["model"][0].role == "user"
    await opened.harness.close(context)


async def test_appends_a_data_only_write_at_once_behind_a_non_empty_inbox_and_still_queues_writes_with_model_or_edits() -> (
    None
):
    setup = chat_setup()
    first = gated(answer("first"))
    setup.faux.set_responses([first.step, answer("second")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    await first.reached
    follow_up = await root.submit({"type": "input", "content": "f"}, context)
    model = await root.submit(
        {"type": "write", "entry": {"kind": "note", "model": [UserMessage("m", timestamp=0)]}}, context
    )
    target = next(entry for entry in await all_entries(root) if entry["kind"] == "pi.user")
    edit = await root.submit(
        {"type": "write", "entry": {"kind": "redact", "edits": [{"target": target["id"], "action": "omit"}]}}, context
    )
    note = await root.submit({"type": "write", "entry": {"kind": "note", "data": "n"}}, context)
    assert (await note.status(context))["status"] == "done"
    assert [[item["id"], item["mode"]] for item in await inbox_items(harness, root)] == [
        [follow_up.id, "followUp"],
        [model.id, "write"],
        [edit.id, "write"],
    ]
    first.release()
    await follow_up.wait(context)
    assert transcript(await all_entries(root)) == [
        ["pi.user", "a"],
        ["note", "n"],
        ["pi.assistant", "first"],
        ["note", "m"],
        ["redact"],
        ["pi.user", "f"],
        ["pi.assistant", "second"],
    ]
    assert_match(await input.status(context), {"status": "done"})
    await harness.close(context)
