"""Port of ``test/harness-events.test.ts``."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import pytest
from durable_harness_support import (
    ChatSetup,
    aborted,
    add_tool,
    assert_match,
    chat_setup,
    open_chat,
    text_of,
    wait_for,
)
from durable_session_support import context, deferred, document_changes, flush, resolve

from pi_ai import (
    AbortController,
    Type,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from pi_ai.serialization import from_dict, message_from_dict, to_json_value
from pi_ai.types import AssistantMessage, TextContent, Usage
from pi_chord.context import with_abort_signal
from pi_durable import AgentDoc, Harness, LiveDoc, MemoryStorage, UsageDoc, define_tool, watch_events


class Listening(NamedTuple):
    stream: Any
    batches: list[list[Any]]

    def events(self) -> list[Any]:
        return [event for batch in self.batches for event in batch]


async def listen(harness: Harness, conversation: Any) -> Listening:
    """Attach and start an event stream that records every delivered batch."""
    stream = await watch_events(harness, conversation.id, context)
    batches: list[list[Any]] = []

    async def listener(events: Any, _context: Any) -> None:
        batches.append(list(events))

    stream.start(listener)
    return Listening(stream, batches)


async def drained() -> None:
    await flush()


def slow() -> ChatSetup:
    return chat_setup(tokens_per_second=400, token_size={"min": 1, "max": 1})


def streamed_text(events: Sequence[Any]) -> str:
    """Rebuild the streamed text of block 0 from ``message_start`` and the text deltas that follow it."""
    text = ""
    for event in events:
        if event["type"] == "message_start" and event["message"].role == "assistant":
            content = event["message"].content
            block = content[0] if content else None
            text = block.text if isinstance(block, TextContent) else ""
        if event["type"] != "message_update":
            continue
        for change in event["changes"]:
            if change["type"] == "text_start" and change["contentIndex"] == 0 and change["block"].type == "text":
                text = change["block"].text
            if change["type"] == "text_delta" and change["contentIndex"] == 0:
                text += change["delta"]
    return text


def apply_changes(message: AssistantMessage, changes: Sequence[Any]) -> AssistantMessage:
    """Apply message changes to a copy of ``message``, as an events-only consumer would."""
    next_message = copy.deepcopy(message)
    for change in changes:
        kind = change["type"]
        if kind == "message":
            next_message = copy.deepcopy(change["message"])
        elif kind == "block":
            next_message.content[change["contentIndex"]] = copy.deepcopy(change["block"])
        elif kind.endswith("_start") and "block" in change:
            next_message.content.insert(change["contentIndex"], copy.deepcopy(change["block"]))
        elif kind in ("text_delta", "thinking_delta"):
            block = next_message.content[change["contentIndex"]]
            field = "text" if kind == "text_delta" else "thinking"
            setattr(block, field, f"{getattr(block, field)}{change['delta']}")
        elif kind == "toolcall_delta":
            target: Any = next_message.content[change["contentIndex"]].arguments  # type: ignore[union-attr]
            for segment in change["path"][:-1]:
                target = target[segment]
            last = change["path"][-1]
            target[last] = f"{target[last]}{change['delta']}"
    return next_message


def partials_of(harness: Harness) -> list[AssistantMessage]:
    """Committed partials of the generation, one per commit that has one."""
    partials: list[AssistantMessage] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            value = change["value"]
            generation = None if value is None else value.get("generation")
            message = None if generation is None else generation.get("message")
            if change["record"]["kind"] == "pi.live" and message is not None:
                partials.append(message_from_dict(message))  # type: ignore[arg-type]

    harness.subscribe_commits(listener)
    return partials


def rebuild_output(events: Sequence[Any], each: Callable[[str], None] | None = None) -> str:
    """The output events rebuild the slot's retained window."""
    text = ""
    for event in events:
        if event["type"] != "tool_execution_update" or "output" not in event:
            continue
        output = event["output"]
        if "set" in output:
            text = output["set"]
        else:
            text = text[output.get("trimStart", 0) :] + output.get("append", "")
        if each is not None:
            each(text)
    return text


async def gate_or_abort(gate: asyncio.Future[Any], signal: Any) -> None:
    """``Promise.race([gate, aborted(signal)])``."""
    cancelled = asyncio.ensure_future(aborted(signal))
    try:
        await asyncio.wait([gate, cancelled], return_when=asyncio.FIRST_COMPLETED)
    finally:
        if not cancelled.done():
            cancelled.cancel()
    if cancelled.done() and not cancelled.cancelled():
        cancelled.result()


def dedupe(types: list[str]) -> list[str]:
    return [kind for index, kind in enumerate(types) if index == 0 or kind != types[index - 1]]


async def test_streams_a_run_as_lifecycle_events_and_text_deltas_that_rebuild_the_answer() -> None:
    setup = slow()
    text = "streamed answer text " * 20
    setup.faux.set_responses([faux_assistant_message([faux_text(text)])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    listening = await listen(harness, root)
    assert_match(listening.stream.snapshot, {"type": "snapshot", "entries": [], "tools": [], "inbox": []})
    partials: list[str] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            value = change["value"]
            generation = None if value is None else value.get("generation")
            message = None if generation is None else generation.get("message")
            if change["record"]["kind"] == "pi.live" and message is not None:
                partials.append(text_of(message_from_dict(message)) or "")

    harness.subscribe_commits(listener)
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    await submission.wait(context)
    await drained()
    all_events = listening.events()
    assert dedupe([event["type"] for event in all_events]) == [
        "message_start",
        "message_end",
        "submission",
        "run_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "run_end",
        "submission",
        "usage_changed",
    ]
    assert next(event for event in all_events if event["type"] == "run_start") == {
        "type": "run_start",
        "inputs": [submission.id],
    }
    # After each event, the rebuilt text equals the committed partial of that commit.
    rebuilt = [
        streamed_text(all_events[: index + 1])
        for index, event in enumerate(all_events)
        if (event["type"] == "message_start" and event["message"].role == "assistant")
        or event["type"] == "message_update"
    ]
    assert rebuilt == partials
    assert len(partials) > 1
    end = [event for event in all_events if event["type"] == "message_end"][-1]
    assert text_of(end["entry"]["model"][0]) == text
    await listening.stream.stop()
    await harness.close(context)


async def test_reports_tool_start_output_appends_and_the_result_entry() -> None:
    setup = chat_setup()
    gate = deferred()

    async def execute(_args: Any, api: Any, _context: Any) -> Any:
        api.output("one\n")
        await asyncio.sleep(0.15)
        api.output("two\n")
        await gate
        return {}

    add_tool(
        setup.registry,
        define_tool(name="print", description="Prints", parameters=Type.Object({"n": Type.Number()}), execute=execute),
    )
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("print", {"n": 1}, id="c1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    listening = await listen(harness, root)
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await wait_for(lambda: rebuild_output(listening.events()) == "one\ntwo\n")
    resolve(gate)
    await submission.wait(context)
    await drained()
    tool = [event for event in listening.events() if event["type"].startswith("tool_execution")]
    start = tool[0]
    assert isinstance(start.get("taskId"), int)
    assert {key: value for key, value in start.items() if key != "taskId"} == {
        "type": "tool_execution_start",
        "toolCallId": "c1",
        "toolName": "print",
        "args": {"n": 1},
    }
    end = tool[-1]
    assert_match(end, {"type": "tool_execution_end", "toolCallId": "c1", "entry": {"kind": "pi.tool-result"}})
    # As in the coding agent, the tool ends directly before its result message.
    all_events = listening.events()
    end_index = next(index for index, event in enumerate(all_events) if event is end)
    assert [event["type"] for event in all_events[end_index : end_index + 3]] == [
        "tool_execution_end",
        "message_start",
        "message_end",
    ]
    message = all_events[end_index + 1]["message"]
    assert message.role == "toolResult" and message.tool_call_id == "c1"
    # Two turns: the tool round, and the answer.
    assert len([event for event in all_events if event["type"] == "turn_start"]) == 2
    assert len([event for event in all_events if event["type"] == "turn_end"]) == 2
    await listening.stream.stop()
    await harness.close(context)


async def test_reports_queued_submissions_inbox_changes_and_retries() -> None:
    setup = chat_setup()
    release = deferred()

    async def held(_request: Any, options: Any, *_rest: Any) -> AssistantMessage:
        await gate_or_abort(release, options["signal"])
        return faux_assistant_message([faux_text("first")])

    error = faux_assistant_message([], stop_reason="error", error_message="503 Service Unavailable")
    setup.faux.set_responses([held, error, faux_assistant_message([faux_text("second")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    setup.settings["retry"] = {"enabled": True, "maxRetries": 1, "baseDelayMs": 1}
    listening = await listen(harness, root)
    await root.submit({"type": "input", "content": "a"}, context)
    follow_up = await root.submit({"type": "input", "content": "f"}, context)
    await drained()
    assert [event for event in listening.events() if event["type"] == "inbox_update"] == [
        {"type": "inbox_update", "items": [{"id": follow_up.id, "mode": "followUp"}]}
    ]
    resolve(release)
    await follow_up.wait(context)
    await drained()
    types = [event["type"] for event in listening.events()]
    assert "auto_retry_start" in types
    assert "auto_retry_end" in types
    assert len([kind for kind in types if kind == "run_start"]) == 2
    assert [event["record"]["status"] for event in listening.events() if event["type"] == "submission"] == [
        "placed",
        "queued",
        "done",
        "placed",
        "done",
    ]
    await listening.stream.stop()
    await harness.close(context)


async def test_replaces_undelivered_batches_with_one_snapshot_after_100_pending_batches() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    stream = await watch_events(harness, root.id, context)
    for _ in range(101):
        await root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)
    batches: list[list[Any]] = []

    async def listener(events: Any, _context: Any) -> None:
        batches.append(list(events))

    stream.start(listener)
    await drained()
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert_match(batches[0][0], {"type": "snapshot"})
    assert len(batches[0][0]["entries"]) == 101
    await stream.stop()
    await harness.close(context)


async def test_rebuilds_every_committed_partial_of_thinking_text_and_tool_call_arguments_from_message_changes() -> None:
    setup = chat_setup(tokens_per_second=150, token_size={"min": 1, "max": 1})
    message = faux_assistant_message(
        [
            faux_thinking("thinking about it " * 10),
            faux_text("some text " * 10),
            faux_tool_call("missing", {"path": "a/long/path/" * 10, "note": "x" * 60}, id="c1"),
        ],
        stop_reason="toolUse",
    )
    setup.faux.set_responses([message, faux_assistant_message([faux_text("done")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    partials = partials_of(harness)
    listening = await listen(harness, root)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    await drained()
    rebuilt: list[AssistantMessage] = []
    current: AssistantMessage | None = None
    # The streamed tool-calling message, up to its end; the short final answer commits no partial.
    for event in listening.events():
        if event["type"] == "message_end" and event["entry"]["kind"] == "pi.assistant":
            break
        if event["type"] == "message_start" and event["message"].role == "assistant":
            current = event["message"]
        elif event["type"] == "message_update":
            assert current is not None
            current = apply_changes(current, event["changes"])
        else:
            continue
        assert current is not None
        rebuilt.append(current)
    assert len(partials) > 2
    assert [partial.content for partial in rebuilt] == [partial.content for partial in partials]
    types = {
        change["type"]
        for event in listening.events()
        if event["type"] == "message_update"
        for change in event["changes"]
    }
    assert "thinking_delta" in types or "text_delta" in types
    await listening.stream.stop()
    await harness.close(context)


async def test_rebuilds_a_sliding_tail_window_from_output_trims_and_appends() -> None:
    setup = chat_setup()
    gate = deferred()

    async def execute(_args: Any, api: Any, _context: Any) -> Any:
        for line in range(6):
            api.output(f"line {line}\n")
            await asyncio.sleep(0.12)
        await gate
        return {}

    add_tool(
        setup.registry,
        define_tool(
            name="tail",
            description="Prints lines",
            parameters=Type.Object({}),
            output_limits={"maxLines": 3, "retain": "tail"},
            execute=execute,
        ),
    )
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("tail", {}, id="c1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    outputs: list[str] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in document_changes(publication):
            value = change["value"]
            tools = None if value is None else value.get("tools")
            output = tools[0].get("output") if tools else None
            if change["record"]["kind"] == "pi.live" and output is not None and (not outputs or output != outputs[-1]):
                outputs.append(output)

    harness.subscribe_commits(listener)
    listening = await listen(harness, root)
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await wait_for(lambda: bool(outputs) and outputs[-1].endswith("line 5\n"))
    resolve(gate)
    await submission.wait(context)
    await drained()
    rebuilt: list[str] = []
    rebuild_output(listening.events(), rebuilt.append)
    assert rebuilt == outputs
    assert any(
        event["type"] == "tool_execution_update" and "trimStart" in event.get("output", {})
        for event in listening.events()
    )
    await listening.stream.stop()
    await harness.close(context)


async def test_emits_one_exact_batch_when_a_run_ends_and_a_queued_follow_up_starts_the_next() -> None:
    setup = chat_setup()
    release = deferred()

    async def held(_request: Any, options: Any, *_rest: Any) -> AssistantMessage:
        await gate_or_abort(release, options["signal"])
        return faux_assistant_message([faux_text("first")])

    setup.faux.set_responses([held, faux_assistant_message([faux_text("second")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "a"}, context)
    follow_up = await root.submit({"type": "input", "content": "f"}, context)
    listening = await listen(harness, root)
    resolve(release)
    await follow_up.wait(context)
    await drained()
    boundary = next(batch for batch in listening.batches if any(event["type"] == "run_end" for event in batch))
    assert [event["type"] for event in boundary] == [
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "run_end",
        "submission",
        "submission",
        "inbox_update",
        "usage_changed",
        "run_start",
        "turn_start",
    ]
    assert [event["record"]["id"] for event in boundary if event["type"] == "submission"] == [input.id, follow_up.id]
    await listening.stream.stop()
    await harness.close(context)


async def test_ends_a_call_that_never_runs_and_a_tool_aborted_with_its_generation() -> None:
    setup = chat_setup()

    async def wait(_args: Any, _api: Any, call_context: Any) -> Any:
        return await aborted(call_context.abort_signal)

    add_tool(
        setup.registry,
        define_tool(name="wait", description="Waits until aborted", parameters=Type.Object({}), execute=wait),
    )
    setup.faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("ghost", {}, id="c1"), faux_tool_call("wait", {}, id="c2")], stop_reason="toolUse"
            )
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    listening = await listen(harness, root)
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await wait_for(lambda: any(event["type"] == "tool_execution_start" for event in listening.events()))
    # Aborting the generation aborts its round: the tool ends with its aborted result first (spec §8.5).
    live = await harness.snapshot(LiveDoc, root.id, context)
    assert live is not None
    await harness.abort_task(live["run"]["taskId"], context)  # type: ignore[typeddict-item]
    await submission.wait(context)
    await drained()

    # The call not offered ends after its calling message and directly before its result message.
    def label(event: Any) -> str:
        if event["type"] == "message_end":
            return f"end:{event['entry']['kind']}"
        if event["type"] == "message_start":
            return f"start:{event['message'].role}"
        return event["type"]

    round = [label(event) for event in listening.events()]
    ghost_end = round.index("tool_execution_end")
    assert round[ghost_end - 1 : ghost_end + 3] == [
        "end:pi.assistant",
        "tool_execution_end",
        "start:toolResult",
        "end:pi.tool-result",
    ]
    tool = [event for event in listening.events() if event["type"].startswith("tool_execution")]
    assert [[event["type"], event.get("toolCallId", False), "entry" in event] for event in tool] == [
        ["tool_execution_end", "c1", True],
        ["tool_execution_start", "c2", False],
        ["tool_execution_end", "c2", True],
    ]
    await listening.stream.stop()
    await harness.close(context)


async def test_reports_a_steer_without_run_events_a_reset_as_an_appended_entry_and_nothing_for_other_conversations() -> (
    None
):
    setup = chat_setup()
    gate = deferred()

    async def hold(*_args: Any) -> Any:
        await gate
        return {}

    add_tool(setup.registry, define_tool(name="hold", description="Waits", parameters=Type.Object({}), execute=hold))
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("hold", {}, id="c1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    listening = await listen(harness, root)
    input = await root.submit({"type": "input", "content": "a"}, context)
    await wait_for(lambda: any(event["type"] == "tool_execution_start" for event in listening.events()))
    await root.submit({"type": "input", "content": "s", "whenBusy": "steer"}, context)
    before = len(listening.batches)
    await other.commit(lambda tx: tx.append_entry(other.id, {"kind": "note"}), context)
    await drained()
    assert len(listening.batches) == before
    resolve(gate)
    await input.wait(context)
    await root.reset(None, context)
    await drained()
    assert len([event for event in listening.events() if event["type"] == "run_start"]) == 1
    assert len([event for event in listening.events() if event["type"] == "run_end"]) == 1
    assert [event["type"] for event in listening.batches[-1]] == ["entry_appended", "submission"]
    await listening.stream.stop()
    await harness.close(context)


async def test_rejects_an_attachment_cancelled_while_it_waits_for_the_session_line() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    release = deferred()

    async def change(_tx: Any) -> None:
        await release

    blocking = root.commit(change, context)
    controller = AbortController()
    attaching = asyncio.ensure_future(watch_events(harness, root.id, with_abort_signal(controller.signal, context)))
    controller.abort(RuntimeError("cancelled"))
    resolve(release)
    await blocking
    with pytest.raises(Exception, match="cancelled"):
        await attaching
    await harness.close(context)


async def test_applies_deltas_after_an_overflow_snapshot_that_holds_an_in_flight_partial() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    partial = faux_assistant_message([faux_text("hel")])
    stream = await watch_events(harness, root.id, context)

    async def start(tx: Any) -> None:
        (await tx.doc(LiveDoc, root.id))["generation"] = {"attempt": 1, "message": to_json_value(partial)}

    await root.commit(start, context)
    for _ in range(101):
        await root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)
    batches: list[list[Any]] = []

    async def listener(events: Any, _context: Any) -> None:
        batches.append(list(events))

    stream.start(listener)
    await drained()
    snapshot = batches[0][0]
    assert snapshot["type"] == "snapshot"

    async def extend(tx: Any) -> None:
        block = (await tx.doc(LiveDoc, root.id))["generation"]["message"]["content"][0]
        block["text"] += "lo"

    await root.commit(extend, context)
    await drained()
    update = batches[-1][0]
    assert update["type"] == "message_update", f"Unexpected {update['type']}"
    assert update["changes"] == [{"type": "text_delta", "contentIndex": 0, "delta": "lo"}]
    rebuilt = apply_changes(snapshot["generation"]["message"], update["changes"])
    assert rebuilt.content == [TextContent("hello")]
    await stream.stop()
    await harness.close(context)


async def test_reports_usage_only_updates_cleared_tool_progress_tools_ending_without_entries_and_deferred_polls() -> (
    None
):
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    listening = await listen(harness, root)

    async def change(edit: Callable[[Any], None]) -> None:
        async def run(tx: Any) -> None:
            edit(await tx.doc(LiveDoc, root.id))

        await root.commit(run, context)

    partial = to_json_value(faux_assistant_message([faux_text("partial")]))

    def start(live: Any) -> None:
        live["generation"] = {"attempt": 1, "message": partial}

    await change(start)

    # A usage-only change sends the usage and no changes.
    def usage_only(live: Any) -> None:
        live["generation"]["message"]["usage"]["input"] = 42

    await change(usage_only)

    def deferred_poll(live: Any) -> None:
        live["generation"] = {"attempt": 1, "deferred": {"pollAt": 1}}

    await change(deferred_poll)

    def later_poll(live: Any) -> None:
        live["generation"]["deferred"]["pollAt"] = 2

    await change(later_poll)

    def running(live: Any) -> None:
        live["tools"] = [{"callId": "c1", "name": "t", "status": "running", "details": {"n": 1}, "diagnostics": []}]

    await change(running)

    # A safe replay clears the running slot's progress.
    def replay(live: Any) -> None:
        del live["tools"][0]["details"]
        del live["tools"][0]["diagnostics"]

    await change(replay)

    # A fault marks the slot done without an entry.
    def fault(live: Any) -> None:
        live["tools"][0]["status"] = "done"

    await change(fault)
    await root.commit(lambda tx: tx.retire_doc(UsageDoc, root.id), context)
    await root.commit(lambda tx: tx.retire_doc(AgentDoc, root.id), context)
    await drained()
    assert [[event for event in batch if event["type"] != "task_failed"] for batch in listening.batches] == [
        [{"type": "message_start", "message": message_from_dict(partial)}],
        [{"type": "message_update", "usage": from_dict(Usage, {**partial["usage"], "input": 42}), "changes": []}],
        [{"type": "deferred_poll", "pollAt": 1}],
        [{"type": "deferred_poll", "pollAt": 2}],
        [{"type": "tool_execution_start", "toolCallId": "c1", "toolName": "t", "args": {}}],
        [{"type": "tool_execution_update", "toolCallId": "c1", "toolName": "t", "details": None, "diagnostics": []}],
        [{"type": "tool_execution_end", "toolCallId": "c1", "toolName": "t"}],
        # Retired documents read as their initial values.
        [{"type": "usage_changed", "usage": {"models": {}, "tools": {}}}],
        [{"type": "agent_changed", "agent": {}}],
    ]
    await listening.stream.stop()
    await harness.close(context)


async def test_ends_the_stream_with_the_harness() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    stream = await watch_events(harness, root.id, context)
    await harness.close(context)
    assert await stream.closed == {"reason": "session_closed"}


async def test_starts_a_message_at_the_first_committed_partial_and_ends_it_with_the_converted_entry_on_abort() -> None:
    setup = chat_setup(tokens_per_second=100, token_size={"min": 1, "max": 1})
    setup.faux.set_responses([faux_assistant_message([faux_text("x" * 400)])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    listening = await listen(harness, root)
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    await wait_for(
        lambda: any(
            event["type"] == "message_start" and event["message"].role == "assistant" for event in listening.events()
        )
    )
    live = await harness.snapshot(LiveDoc, root.id, context)
    assert live is not None
    await harness.abort_task(live["run"]["taskId"], context)  # type: ignore[typeddict-item]
    await submission.wait(context)
    await drained()
    assistant_ends = [
        event
        for event in listening.events()
        if event["type"] == "message_end" and event["entry"]["kind"] == "pi.assistant"
    ]
    assert len(assistant_ends) == 1
    assert (
        len(
            [
                event
                for event in listening.events()
                if event["type"] == "message_start" and event["message"].role == "assistant"
            ]
        )
        == 1
    )
    await listening.stream.stop()
    await harness.close(context)
