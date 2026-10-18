"""Port of ``test/harness-compaction.test.ts``.

Adaptations: a faux step receives pi-ai's snake_case request options (``cache_retention``, ``max_tokens``,
``session_id``, ``timeout_ms``); models are patched by assigning instance attributes (upstream assigns methods);
``ControlledStorage.reopen()`` is awaited.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import (
    ChatSetup,
    aborted,
    add_hooks,
    add_section,
    add_task,
    add_tool,
    all_entries,
    assert_match,
    chat_setup,
    open_chat,
    text_of,
    tools_named,
    wait_for,
)
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, resolve

from pi_ai import Type, faux_assistant_message, faux_tool_call
from pi_ai.types import AssistantMessage, SystemMessage, TextContent, ThinkingContent, ToolResultMessage, UserMessage
from pi_chord.context import Context
from pi_durable import (
    CompactionTask,
    Harness,
    LiveDoc,
    MemoryStorage,
    ProviderDoc,
    Task,
    UsageDoc,
    define_task,
    define_tool,
    watch_events,
)
from pi_durable.harness.compaction import select_cut, serialize_conversation
from pi_durable.harness.context import order_tool_results
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


def text(label: str, tokens: int) -> str:
    """Text of about ``tokens`` estimated tokens, starting with ``label``."""
    return f"{label} {'x' * max(0, tokens * 4 - len(label) - 1)}"


@dataclasses.dataclass(slots=True)
class Request:
    messages: list[Any]
    options: Any
    model: str


Step = Any


@dataclasses.dataclass(slots=True)
class Script:
    """Scripted faux model that answers agent requests and summarization requests from separate queues."""

    agent: list[Step] = dataclasses.field(default_factory=list)
    summaries: list[Step] = dataclasses.field(default_factory=list)
    agent_requests: list[Request] = dataclasses.field(default_factory=list)
    summary_requests: list[Request] = dataclasses.field(default_factory=list)


def is_summary_request(messages: Sequence[Any]) -> bool:
    first = messages[0] if messages else None
    return (
        first is not None
        and first.role == "system"
        and isinstance(first.content, str)
        and first.content.startswith("You are a context summarization assistant")
    )


def script(setup: ChatSetup) -> Script:
    result = Script()

    async def respond(transcript: Any, options: Any, _state: Any, model: Any) -> AssistantMessage:
        request = Request(list(transcript.messages), options, model.id)
        summary = is_summary_request(request.messages)
        (result.summary_requests if summary else result.agent_requests).append(request)
        queue = result.summaries if summary else result.agent
        if not queue:
            raise RuntimeError(f"No scripted {'summary' if summary else 'agent'} response")
        step = queue.pop(0)
        if callable(step):
            outcome = step(request)
            return await outcome if asyncio.iscoroutine(outcome) else outcome
        return step

    setup.faux.set_responses([respond] * 500)
    return result


async def race(gate: asyncio.Future[Any], signal: Any) -> None:
    """``Promise.race([gate, aborted(signal)])``."""
    cancelled = asyncio.ensure_future(aborted(signal))
    try:
        await asyncio.wait([gate, cancelled], return_when=asyncio.FIRST_COMPLETED)
    finally:
        if not cancelled.done():
            cancelled.cancel()
    if not gate.done():
        cancelled.result()


def gated(gate: asyncio.Future[Any], message: AssistantMessage, reached: asyncio.Future[Any] | None = None) -> Step:
    """A step that waits for ``gate``, or rejects when the request is cancelled."""

    async def step(request: Request) -> AssistantMessage:
        if reached is not None:
            resolve(reached)
        await race(gate, request.options["signal"])
        return message

    return step


def answer(content: str) -> AssistantMessage:
    return faux_assistant_message(content)


def failure(error_message: str) -> AssistantMessage:
    return faux_assistant_message("", stop_reason="error", error_message=error_message)


OVERFLOW = "prompt is too long: 250000 tokens > 200000 maximum"

#: Small thresholds: no automatic compaction unless a test enables it.
MANUAL: dict[str, Any] = {"enabled": False, "reserveTokens": 1000, "keepRecentTokens": 150, "backgroundTokens": 0}


@dataclasses.dataclass(slots=True)
class Chat:
    harness: Harness
    root: Any
    setup: ChatSetup
    faux: Script


def model_setup(context_window: int = 100_000) -> ChatSetup:
    return chat_setup(models=[{"id": "faux-1", "context_window": context_window, "max_tokens": 900}])


async def open_compacting(
    *,
    policy: Any = None,
    context_window: int | None = None,
    storage: Any = None,
    setup: ChatSetup | None = None,
    faux: Script | None = None,
) -> Chat:
    setup = setup if setup is not None else model_setup(context_window if context_window is not None else 100_000)
    faux = faux if faux is not None else script(setup)
    add_section(setup.registry, "preamble", lambda *_: "You are helpful.", tag=False)
    opened = await open_chat(storage if storage is not None else MemoryStorage(), setup)
    # These tests exercise compaction accounting and thresholds, not the faux provider's cache simulation.
    setup.settings["stream"] = {"cacheRetention": "none"}
    setup.settings["compaction"] = policy if policy is not None else MANUAL
    setup.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 1}
    opened.harness.resume()
    return Chat(opened.harness, opened.root, setup, faux)


async def turn(chat: Chat, user_text: str, reply: str) -> None:
    """Run one turn: ``user_text`` answered by ``reply``."""
    chat.faux.agent.append(answer(reply))
    submission = await chat.root.submit({"type": "input", "content": user_text}, context)
    assert (await submission.wait(context))["status"] == "done"


async def history(chat: Chat) -> None:
    """Three turns of about 100-token messages."""
    await turn(chat, text("u1", 100), text("a1", 100))
    await turn(chat, text("u2", 100), text("a2", 100))
    await turn(chat, text("u3", 100), text("a3", 100))


def summary(content: str = "SUMMARY") -> AssistantMessage:
    return answer(content)


async def result(chat: Chat, id: Any) -> Any:
    record = await chat.harness.wait_for_task(id, context)
    return record["state"]["outcome"]


async def kinds(conversation: Any) -> list[str]:
    return [entry["kind"] for entry in await all_entries(conversation)]


async def live(chat: Chat) -> Any:
    value = await chat.harness.snapshot(LiveDoc, chat.root.id, context)
    return {} if value is None else value


def user_text(message: Any) -> str:
    return text_of(message) or ""


def first_user_text(messages: Sequence[Any]) -> str:
    """Text of the first user message; a leading baseline system message comes before it."""
    return user_text(next((message for message in messages if message.role == "user"), None))


async def submission_of(chat: Chat, outcome: Any) -> Any:
    """The submission a completed compaction's outcome names."""
    submission_id = outcome["result"].get("submissionId") if outcome["status"] == "completed" else None
    found = await chat.harness.submission(submission_id, context)  # type: ignore[arg-type]
    assert found is not None
    return found


async def models_usage(chat: Chat, conversation: Any = None) -> Any:
    usage = await chat.harness.snapshot(UsageDoc, (conversation or chat.root).id, context)
    assert usage is not None
    return usage["models"].get("faux/faux-1")


# ─── Range selection ──────────────────────────────────────────────────────

_next_id = [1]


def entry(kind: str, model: list[Any], head: int | None = None) -> Any:
    record: dict[str, Any] = {"id": _next_id[0], "conversationId": 1, "kind": kind, "model": model}
    _next_id[0] += 1
    if head is not None:
        record["head"] = head
    return record


def user(content: str) -> UserMessage:
    return UserMessage(content, timestamp=0)


def assistant(content: str, calls: Sequence[str] = (), stop_reason: str = "stop") -> AssistantMessage:
    message = faux_assistant_message([TextContent(content), *(faux_tool_call("read", {}, id=id) for id in calls)])
    message.stop_reason = stop_reason  # type: ignore[assignment]
    return message


def tool_result(call_id: str, content: str) -> ToolResultMessage:
    return ToolResultMessage(call_id, "read", [TextContent(content)], False, timestamp=0)


def view(entries: list[Any], head: Any = None) -> Any:
    """A view over ``entries`` whose contributions are their models, with excluded assistants removed."""
    records = entries if head is None else [head, *entries]
    contributions = [
        [
            message
            for message in record.get("model") or []
            if message.role != "assistant" or message.stop_reason not in ("error", "aborted", "deferred")
        ]
        for record in records
    ]
    messages = order_tool_results([message for group in contributions for message in group])
    return {"head": head, "entries": records, "contributions": contributions, "messages": messages}


def test_keeps_about_keep_recent_tokens_and_cuts_at_the_first_candidate_at_or_after_the_budget_spec_8_7_example() -> (
    None
):
    entries = [
        entry("pi.user", [user(text("1", 10))]),
        entry("pi.assistant", [assistant("2", ["c1"])]),
        entry("pi.tool-result", [tool_result("c1", text("3", 3000))]),
        entry("pi.assistant", [assistant(text("4", 10))]),
        entry("pi.user", [user(text("5", 10))]),
        entry("pi.assistant", [assistant(text("6", 10))]),
    ]
    assert select_cut(view(entries), 2000) == 3


def test_cuts_at_a_user_entry() -> None:
    entries = [
        entry("pi.user", [user(text("u1", 100))]),
        entry("pi.assistant", [assistant(text("a1", 100))]),
        entry("pi.user", [user(text("u2", 100))]),
        entry("pi.assistant", [assistant(text("a2", 100))]),
    ]
    assert select_cut(view(entries), 150) == 2


def test_cuts_at_an_assistant_in_the_middle_of_one_long_run_and_never_at_a_tool_result() -> None:
    entries = [entry("pi.user", [user("do it")])]
    for index in range(5):
        entries.append(entry("pi.assistant", [assistant(f"step {index}", [f"c{index}"])]))
        entries.append(entry("pi.tool-result", [tool_result(f"c{index}", text(f"r{index}", 100))]))
    cut = select_cut(view(entries), 150)
    assert cut is not None
    # The budget is reached at the fourth result; the cut is the last call, whose result it keeps.
    assert entries[cut]["kind"] == "pi.assistant"
    assert cut == 9


def test_never_cuts_at_a_display_only_entry_where_the_budget_is_reached_10407() -> None:
    entries = [
        entry("pi.user", [user(text("u1", 100))]),
        entry("pi.assistant", [assistant(text("a1", 100))]),
        entry("chat.message", []),
        entry("pi.assistant", [assistant(text("a2", 100))]),
    ]
    # The budget is reached at a1; the display-only entry after it is skipped for the assistant a2.
    assert select_cut(view(entries), 150) == 1
    assert select_cut(view(entries), 50) == 3


def test_keeps_a_huge_last_tool_result_together_with_its_assistant() -> None:
    entries = [
        entry("pi.user", [user("u")]),
        entry("pi.assistant", [assistant("a", ["c"])]),
        entry("pi.tool-result", [tool_result("c", text("big", 5000))]),
    ]
    assert select_cut(view(entries), 100) == 1


def test_never_cuts_at_a_system_entry_or_an_excluded_error_or_aborted_answer() -> None:
    entries = [
        entry("pi.user", [user(text("u1", 100))]),
        entry("pi.assistant", [assistant(text("a1", 100))]),
        entry("pi.system", [SystemMessage("", sections={"s": text("s", 100)}, timestamp=0)]),
        entry("pi.assistant", [assistant(text("err", 100), [], "error")]),
        entry("pi.assistant", [assistant(text("stopped", 100), [], "aborted")]),
        entry("pi.assistant", [assistant(text("a2", 100))]),
    ]
    # The walk reaches 150 at the system entry; the excluded answers after it contribute nothing.
    assert select_cut(view(entries), 150) == 5


def test_follows_edited_contributions_an_omitted_entry_adds_nothing_and_is_no_candidate() -> None:
    entries = [
        entry("pi.user", [user(text("u1", 100))]),
        entry("pi.assistant", [assistant(text("a1", 100))]),
        entry("pi.user", [user(text("u2", 100))]),
        entry("pi.assistant", [assistant(text("a2", 100))]),
    ]
    plain = view(entries)
    assert select_cut(plain, 150) == 2
    contributions = [[] if index == 2 else messages for index, messages in enumerate(plain["contributions"])]
    omitted = {
        **plain,
        "contributions": contributions,
        "messages": order_tool_results([message for group in contributions for message in group]),
    }
    assert select_cut(omitted, 150) == 1  # type: ignore[arg-type]


def test_does_not_cut_at_a_user_entry_that_a_result_of_the_preceding_call_still_follows() -> None:
    entries = [
        entry("pi.user", [user(text("u1", 100))]),
        entry("pi.assistant", [assistant("a", ["c"])]),
        entry("pi.user", [user(text("steer", 100))]),
        entry("pi.tool-result", [tool_result("c", text("r", 100))]),
        entry("pi.assistant", [assistant(text("a2", 100))]),
    ]
    # The budget is reached at the steer; its result follows it, so the cut moves to the next assistant.
    assert select_cut(view(entries), 250) == 4


def test_finds_nothing_when_the_budget_is_never_reached_or_only_the_marker_precedes_the_cut() -> None:
    small = [entry("pi.user", [user("hi")]), entry("pi.assistant", [assistant("hello")])]
    assert select_cut(view(small), 150) is None
    marker = entry("pi.compaction", [user("summary")], 0)
    # The budget is reached at the only entry after the marker, so the marker alone would be summarized.
    assert select_cut(view([entry("pi.user", [user(text("u", 200))])], marker), 150) is None


def test_summarizes_an_earlier_summary_marker_first() -> None:
    marker = entry("pi.compaction", [user("EARLIER")], 0)
    kept = [
        entry("pi.user", [user(text("u1", 100))]),
        entry("pi.assistant", [assistant(text("a1", 100))]),
        entry("pi.user", [user(text("u2", 100))]),
        entry("pi.assistant", [assistant(text("a2", 100))]),
    ]
    selected = view(kept, marker)
    assert select_cut(selected, 150) == 3
    serialized = serialize_conversation([message for group in selected["contributions"][:3] for message in group])
    assert re.match(r"^\[User\]: EARLIER", serialized)


# ─── serialization ────────────────────────────────────────────────────────


def test_writes_a_transcript_truncates_tool_results_and_omits_system_messages_and_thinking() -> None:
    call = faux_tool_call("read", {"path": "a.ts"}, id="c")
    messages: list[Any] = [
        SystemMessage("", sections={"s": "hidden"}, timestamp=0),
        user("hello"),
        faux_assistant_message([ThinkingContent("hmm"), TextContent("sure"), call]),
        tool_result("c", "y" * 2500),
    ]
    serialized = serialize_conversation(messages)
    assert "hidden" not in serialized
    assert "[User]: hello" in serialized
    assert "hmm" not in serialized
    assert "[Assistant thinking]" not in serialized
    assert "[Assistant]: sure" in serialized
    assert '[Assistant tool calls]: read(path="a.ts")' in serialized
    assert f"[Tool result]: {'y' * 2000}\n\n[... 500 more characters truncated]" in serialized


# ─── Manual compaction ────────────────────────────────────────────────────

UUID_V7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


async def test_places_the_summary_at_once_when_idle_and_keeps_raw_history() -> None:
    chat = await open_compacting()
    await history(chat)
    before = await all_entries(chat.root)
    usage_before = await models_usage(chat)
    chat.faux.summaries.append(summary())

    id = await chat.root.compact("focus on files", context)
    outcome = await result(chat, id)
    assert outcome["status"] == "completed"
    placed = await (await submission_of(chat, outcome)).wait(context)
    assert placed["status"] == "done"

    # Raw history is unchanged; one summary entry heads the first kept entry, u3.
    after = await all_entries(chat.root)
    assert after[: len(before)] == before
    marker = after[-1]
    u3 = next(record for record in before if user_text((record.get("model") or [None])[0]).startswith("u3"))
    assert_match(marker, {"kind": "pi.compaction", "head": u3["id"], "data": {"reason": "manual"}})
    assert placed["entry"] == marker["id"]
    assert user_text(marker["model"][0]) == (
        "The conversation history before this point was compacted into the following summary:\n\n"
        "<summary>\nSUMMARY\n</summary>"
    )

    # The model context is the summary followed by the kept entries.
    context_view = await chat.root.context(context)
    assert [user_text(message) for message in context_view["messages"]] == [
        user_text(marker["model"][0]),
        text("u3", 100),
        text("a3", 100),
    ]

    # The summarizer saw the serialized prefix, the prompt, and the instructions, without tools or caching.
    request = chat.faux.summary_requests[0]
    assert len(request.messages) == 2
    prompt = user_text(request.messages[1])
    assert re.match(r"^<conversation>\n\[User\]: u1 ", prompt)
    assert "[Assistant]: a2 " in prompt
    assert "u3 " not in prompt
    assert "## Goal" in prompt
    assert prompt.endswith("\n\nAdditional focus: focus on files")
    provider = await chat.harness.snapshot(ProviderDoc, chat.root.id, context)
    assert provider is not None
    assert_match(request.options, {"cache_retention": "none", "max_tokens": 800, "session_id": provider["sessionId"]})
    assert request.options.get("deferred") is None

    # The summarizer's spend is in the ledger, and nothing counts it again later.
    usage = await models_usage(chat)
    assert usage["input"] > usage_before["input"]
    await turn(chat, "next", "done")
    agent = chat.faux.agent_requests[-1].messages
    # The next request: one complete system baseline leading the summary, the kept turn, and the new input (#10904).
    assert [message.role for message in agent] == ["system", "user", "user", "assistant", "user"]
    assert_match(agent[0], {"sections": {"preamble": "You are helpful."}})
    await chat.harness.close(context)


# Regression coverage for #10424.
async def test_creates_provider_state_before_a_legacy_conversations_summarization_request() -> None:
    chat = await open_compacting()
    await history(chat)
    await chat.root.commit(lambda tx: tx.retire_doc(ProviderDoc, chat.root.id), context)
    assert await chat.harness.snapshot(ProviderDoc, chat.root.id, context) is None
    chat.faux.summaries.append(summary())
    assert (await result(chat, await chat.root.compact(None, context)))["status"] == "completed"
    stored = await chat.harness.snapshot(ProviderDoc, chat.root.id, context)
    assert stored is not None and UUID_V7.match(stored["sessionId"])
    assert chat.faux.summary_requests[0].options.get("session_id") == stored["sessionId"]
    await chat.harness.close(context)


async def test_keeps_working_while_busy_and_places_the_summary_at_the_next_final_boundary() -> None:
    chat = await open_compacting()
    await history(chat)
    gate = deferred()
    reached = deferred()
    chat.faux.agent.append(gated(gate, answer("late answer"), reached))
    input = await chat.root.submit({"type": "input", "content": "busy"}, context)
    await reached
    chat.faux.summaries.append(summary())
    outcome = await result(chat, await chat.root.compact(None, context))
    submission = await submission_of(chat, outcome)
    assert (await submission.status(context))["status"] == "queued"
    resolve(gate)
    assert (await input.wait(context))["status"] == "done"
    placed = await submission.wait(context)
    assert placed["status"] == "done"
    # The summary follows the answer; the kept range still includes the busy turn.
    assert (await kinds(chat.root))[-3:] == ["pi.user", "pi.assistant", "pi.compaction"]
    await chat.harness.close(context)


def wait_tool() -> Any:
    async def execute(*_args: Any) -> Any:
        return {"output": [TextContent("waited")]}

    return define_tool(name="wait", description="wait", parameters=Type.Object({}), execute=execute)


async def test_places_a_queued_summary_at_post_tools_and_the_run_continues_in_the_compacted_context() -> None:
    chat = await open_compacting()
    add_tool(chat.setup.registry, wait_tool())
    await chat.root.configure({"tools": tools_named(chat.setup, "wait")}, context)
    await history(chat)
    gate = deferred()
    reached = deferred()
    chat.faux.agent.append(
        gated(gate, faux_assistant_message([faux_tool_call("wait", {})], stop_reason="toolUse"), reached)
    )
    chat.faux.agent.append(answer("after tools"))
    input = await chat.root.submit({"type": "input", "content": "use a tool"}, context)
    await reached
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    resolve(gate)
    assert (await input.wait(context))["status"] == "done"
    # The continuation request starts with the new baseline, then the summary, ahead of the kept tool call (#10904).
    continuation = chat.faux.agent_requests[-1].messages
    assert continuation[0].role == "system"
    assert "<summary>\nSUMMARY\n</summary>" in user_text(continuation[1])
    assert (await kinds(chat.root))[-4:] == ["pi.tool-result", "pi.compaction", "pi.system", "pi.assistant"]
    await chat.harness.close(context)


async def test_runs_follow_ups_left_by_a_failed_run_after_placing_the_summary() -> None:
    chat = await open_compacting()
    await history(chat)
    gate = deferred()
    reached = deferred()
    chat.faux.agent.append(gated(gate, failure("bad request"), reached))
    failed = await chat.root.submit({"type": "input", "content": "fails"}, context)
    await reached
    follow_up = await chat.root.submit({"type": "input", "content": "follow-up"}, context)
    resolve(gate)
    assert (await failed.wait(context))["status"] == "unanswered"
    assert (await follow_up.status(context))["status"] == "queued"

    chat.faux.summaries.append(summary())
    chat.faux.agent.append(answer("followed"))
    await result(chat, await chat.root.compact(None, context))
    assert (await follow_up.wait(context))["status"] == "done"
    request = chat.faux.agent_requests[-1].messages
    assert "SUMMARY" in first_user_text(request)
    assert any(user_text(message) == "follow-up" for message in request)
    await chat.harness.close(context)


async def test_settles_stale_when_a_reset_lands_while_it_summarizes() -> None:
    chat = await open_compacting()
    await history(chat)
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary(), reached))
    id = await chat.root.compact(None, context)
    await reached
    await chat.root.reset(None, context)
    resolve(gate)
    outcome = await result(chat, id)
    assert_match(
        await (await submission_of(chat, outcome)).status(context), {"status": "unanswered", "reason": "stale"}
    )
    assert (await kinds(chat.root))[-1] == "pi.reset"
    await chat.harness.close(context)


async def test_does_not_make_the_conversation_busy_a_submission_during_summarization_starts_its_run_at_once() -> None:
    chat = await open_compacting()
    await history(chat)
    summary_gate = deferred()
    summary_reached = deferred()
    chat.faux.summaries.append(gated(summary_gate, summary(), summary_reached))
    id = await chat.root.compact(None, context)
    await summary_reached
    answer_gate = deferred()
    answer_reached = deferred()
    chat.faux.agent.append(gated(answer_gate, answer("a4"), answer_reached))
    input = await chat.root.submit({"type": "input", "content": "u4"}, context)
    # Placed and answered immediately with the uncompacted context, not queued behind the compaction.
    assert (await input.status(context))["status"] == "placed"
    await answer_reached
    assert first_user_text(chat.faux.agent_requests[-1].messages) == text("u1", 100)
    # The summary is ready while the run is busy, so it queues and lands after the answer.
    resolve(summary_gate)
    submission = await submission_of(chat, await result(chat, id))
    assert (await submission.status(context))["status"] == "queued"
    resolve(answer_gate)
    assert (await input.wait(context))["status"] == "done"
    assert (await submission.wait(context))["status"] == "done"
    assert (await kinds(chat.root))[-3:] == ["pi.user", "pi.assistant", "pi.compaction"]
    await chat.harness.close(context)


async def test_counts_the_spend_of_a_summary_that_ends_stale_and_writes_no_entry_for_it() -> None:
    chat = await open_compacting()
    await history(chat)
    before = (await models_usage(chat))["input"]
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary(), reached))
    id = await chat.root.compact(None, context)
    await reached
    await chat.root.reset(None, context)
    resolve(gate)
    submission = await submission_of(chat, await result(chat, id))
    assert_match(await submission.status(context), {"status": "unanswered", "reason": "stale"})
    assert (await models_usage(chat))["input"] > before
    assert "pi.compaction" not in await kinds(chat.root)
    await chat.harness.close(context)


async def test_lets_the_compaction_that_cuts_furthest_win_whatever_finishes_first() -> None:
    chat = await open_compacting()
    await history(chat)
    first = deferred()
    first_reached = deferred()
    chat.faux.summaries.append(gated(first, summary("FIRST"), first_reached))
    early = await chat.root.compact(None, context)
    await first_reached
    await turn(chat, text("u4", 100), text("a4", 100))
    chat.faux.summaries.append(summary("SECOND"))
    later = await result(chat, await chat.root.compact(None, context))
    assert later["status"] == "completed"
    resolve(first)
    outcome = await result(chat, early)
    # The early compaction cut at u3, before the later cut at u4.
    assert_match(
        await (await submission_of(chat, outcome)).status(context), {"status": "unanswered", "reason": "stale"}
    )
    assert "SECOND" in first_user_text((await chat.root.context(context))["messages"])
    await chat.harness.close(context)


async def test_places_an_older_selected_summary_that_cuts_later_than_the_newer_one() -> None:
    chat = await open_compacting()
    await history(chat)
    # A: small budget, late cut; selected first, finishes last.
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary("A"), reached))
    a = await chat.root.compact(None, context)
    await reached
    # B: larger budget, earlier cut; placed first.
    chat.setup.settings["compaction"] = {**MANUAL, "keepRecentTokens": 350}
    chat.faux.summaries.append(summary("B"))
    await result(chat, await chat.root.compact(None, context))
    assert "B" in first_user_text((await chat.root.context(context))["messages"])
    resolve(gate)
    outcome = await result(chat, a)
    assert (await (await submission_of(chat, outcome)).status(context))["status"] == "done"
    messages = (await chat.root.context(context))["messages"]
    assert "<summary>\nA\n</summary>" in first_user_text(messages)
    assert [user_text(message) for message in messages[1:]] == [text("u3", 100), text("a3", 100)]
    await chat.harness.close(context)


@pytest.mark.parametrize(
    ("order", "keeps", "stale"), [("before", [150, 350], True), ("at", [150, 150], False), ("after", [350, 150], False)]
)
async def test_places_two_queued_summaries_in_one_boundary_when_the_second_cuts_before_at_or_after_the_first(
    order: str, keeps: list[int], stale: bool
) -> None:
    chat = await open_compacting()
    await history(chat)
    gate = deferred()
    reached = deferred()
    chat.faux.agent.append(gated(gate, answer("done"), reached))
    input = await chat.root.submit({"type": "input", "content": "busy"}, context)
    await reached
    submissions = []
    for index, keep in enumerate(keeps):
        chat.setup.settings["compaction"] = {**MANUAL, "keepRecentTokens": keep}
        chat.faux.summaries.append(summary(f"S{index}"))
        submissions.append(await submission_of(chat, await result(chat, await chat.root.compact(None, context))))
    resolve(gate)
    await input.wait(context)
    statuses = [(await submission.wait(context))["status"] for submission in submissions]
    assert statuses == ["done", "unanswered" if stale else "done"], order
    await chat.harness.close(context)


async def test_is_aborted_by_conversation_abort_an_already_queued_summary_survives_it() -> None:
    chat = await open_compacting()
    await history(chat)
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    id = await chat.root.compact(None, context)
    await reached
    assert len((await live(chat))["compactions"]) == 1
    await chat.root.abort(context)
    assert (await result(chat, id))["status"] == "aborted"
    assert "compactions" not in await live(chat)
    assert "pi.compaction" not in await kinds(chat.root)

    # Queued while busy, then Esc: the queued write stays and lands with the next run's boundary.
    gate = deferred()
    busy = deferred()
    chat.faux.agent.append(gated(gate, answer("never"), busy))
    await chat.root.submit({"type": "input", "content": "busy"}, context)
    await busy
    chat.faux.summaries.append(summary())
    queued = await result(chat, await chat.root.compact(None, context))
    await chat.root.abort(context)
    submission = await submission_of(chat, queued)
    assert (await submission.status(context))["status"] == "queued"
    assert await submission.abort(context) == "aborted"
    await turn(chat, "next", "ok")
    assert_match(await submission.status(context), {"status": "unanswered", "reason": "aborted"})
    assert "pi.compaction" not in await kinds(chat.root)
    await chat.harness.close(context)


async def test_is_ordinary_work_idle_waits_include_it() -> None:
    chat = await open_compacting()
    await history(chat)
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary(), reached))
    id = await chat.root.compact(None, context)
    await reached
    wait = asyncio.ensure_future(chat.root.wait_for_idle(context))
    await asyncio.sleep(0.02)
    assert not wait.done()
    resolve(gate)
    await wait
    record = await chat.harness.get_task(id, context)
    assert record is not None and record["state"]["status"] == "terminal"
    await chat.harness.close(context)


async def test_enables_scheduling_right_after_open() -> None:
    setup = chat_setup()
    faux = script(setup)
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    setup.settings["compaction"] = {**MANUAL, "keepRecentTokens": 10}
    id = await root.compact(None, context)

    # get_task() only reads, so progress here comes from compact() itself.
    async def terminal() -> bool:
        record = await harness.get_task(id, context)
        return record is not None and record["state"]["status"] == "terminal"

    await wait_for(terminal)
    record = await harness.get_task(id, context)
    assert_match(record["state"], {"outcome": {"status": "completed", "result": {}}})  # type: ignore[index]
    assert len(faux.summary_requests) == 0
    await harness.close(context)


# ─── compaction outcomes ──────────────────────────────────────────────────


async def test_completes_without_a_summary_when_there_is_nothing_to_compact() -> None:
    chat = await open_compacting()
    await turn(chat, "hi", "hello")
    outcome = await result(chat, await chat.root.compact(None, context))
    assert outcome == {"status": "completed", "result": {}}
    assert len(chat.faux.summary_requests) == 0
    assert "compactions" not in await live(chat)
    await chat.harness.close(context)


async def test_asks_before_compact_the_first_decision_wins_a_throw_is_reported_and_skipped() -> None:
    chat = await open_compacting()
    seen: list[Any] = []

    def broken(*_args: Any) -> Any:
        raise RuntimeError("hook broke")

    def deciding(compaction: Any, *_rest: Any) -> Any:
        seen.append(compaction)
        return {"summary": "FROM HOOK"}

    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": broken})
    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": deciding})
    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": lambda *_: {"decline": True}})
    await history(chat)
    outcome = await result(chat, await chat.root.compact("why", context))
    assert outcome["status"] == "completed"
    assert len(chat.faux.summary_requests) == 0
    assert "<summary>\nFROM HOOK\n</summary>" in first_user_text((await chat.root.context(context))["messages"])
    assert [str(error) for error in chat.setup.reports] == ["hook broke"]
    compaction = seen[0]
    assert_match(compaction, {"reason": "manual", "instructions": "why"})
    assert [record["kind"] for record in compaction["entries"]] == [
        "pi.user",
        "pi.system",
        "pi.assistant",
        "pi.user",
        "pi.assistant",
    ]
    assert [user_text(message) for message in compaction["messages"] if message.role != "system"] == [
        text("u1", 100),
        text("a1", 100),
        text("u2", 100),
        text("a2", 100),
    ]
    await chat.harness.close(context)


async def test_completes_without_a_summary_when_a_hook_declines() -> None:
    chat = await open_compacting()
    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": lambda *_: {"decline": True}})
    await history(chat)
    assert await result(chat, await chat.root.compact(None, context)) == {"status": "completed", "result": {}}
    assert len(chat.faux.summary_requests) == 0
    await chat.harness.close(context)


async def test_fails_with_no_model_without_a_configured_model() -> None:
    chat = await open_compacting()
    await history(chat)
    await chat.root.configure({"model": None}, context)
    outcome = await result(chat, await chat.root.compact(None, context))
    assert_match(outcome, {"status": "failed", "error": {"detail": {"reason": "no_model"}}})
    assert "compactions" not in await live(chat)
    await chat.harness.close(context)


async def compaction_input(chat: Chat, run: Callable[[], Awaitable[Any]]) -> int:
    """Input tokens a compaction added to the ledger."""
    before = (await models_usage(chat))["input"]
    await run()
    return (await models_usage(chat))["input"] - before


async def test_retries_a_retryable_error_with_the_pinned_request_and_counts_every_attempt_once() -> None:
    single = await open_compacting()
    await history(single)
    await single.root.configure({"thinkingLevel": "high"}, context)
    single.faux.summaries.append(summary())

    async def run_single() -> Any:
        return await result(single, await single.root.compact(None, context))

    once = await compaction_input(single, run_single)
    assert once > 0
    await single.harness.close(context)

    chat = await open_compacting()
    await history(chat)
    await chat.root.configure({"thinkingLevel": "high"}, context)
    chat.setup.settings["stream"] = {"timeoutMs": 1234, "deferred": True}

    async def changing(_request: Request) -> AssistantMessage:
        # Changed during the attempt: the retry still uses the pinned request.
        await chat.root.configure({"thinkingLevel": "low"}, context)
        chat.setup.settings["stream"] = {"timeoutMs": 1}
        return failure("overloaded")

    chat.faux.summaries.append(changing)
    chat.faux.summaries.append(summary())

    async def run_twice() -> None:
        assert (await result(chat, await chat.root.compact(None, context)))["status"] == "completed"

    twice = await compaction_input(chat, run_twice)
    assert len(chat.faux.summary_requests) == 2
    assert twice == 2 * once
    provider = await chat.harness.snapshot(ProviderDoc, chat.root.id, context)
    assert provider is not None
    for request in chat.faux.summary_requests:
        assert_match(
            request.options,
            {"reasoning": "high", "timeout_ms": 1234, "cache_retention": "none", "session_id": provider["sessionId"]},
        )
        assert request.options.get("deferred") is None
    await chat.harness.close(context)


async def test_adds_no_usage_when_a_hook_declines_or_supplies_the_summary() -> None:
    for decision in ({"decline": True}, {"summary": "HOOK"}):
        chat = await open_compacting()
        add_hooks(chat.setup.registry, CompactionTask, {"before_compact": lambda *_, decision=decision: decision})
        await history(chat)

        async def run(chat: Chat = chat) -> Any:
            return await result(chat, await chat.root.compact(None, context))

        assert await compaction_input(chat, run) == 0
        await chat.harness.close(context)


async def test_caps_max_tokens_at_the_models_output_limit_and_sends_no_tools() -> None:
    chat = await open_compacting()

    async def execute(*_args: Any) -> Any:
        return {"output": []}

    add_tool(
        chat.setup.registry, define_tool(name="read", description="read", parameters=Type.Object({}), execute=execute)
    )
    await chat.root.configure({"tools": tools_named(chat.setup, "read")}, context)
    await history(chat)
    chat.setup.settings["compaction"] = {**MANUAL, "reserveTokens": 2000}
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    request = chat.faux.summary_requests[0]
    # 0.8 * 2000 = 1600, above the model's 900.
    assert request.options.get("max_tokens") == 900
    assert len(request.messages) == 2
    assert not any(message.role == "system" and message.tools_added is not None for message in request.messages)
    await chat.harness.close(context)


@pytest.mark.parametrize(
    ("name", "response", "message"),
    [
        ("retries run out", failure("overloaded"), "Summarization failed: overloaded"),
        ("a non-retryable error", failure("bad request"), "Summarization failed: bad request"),
        (
            "a length stop",
            faux_assistant_message("partial", stop_reason="length"),
            "Summarization hit the token limit; the summary is incomplete",
        ),
        (
            "a tool call",
            faux_assistant_message([faux_tool_call("read", {})], stop_reason="stop"),
            "Summarization attempted to call a tool",
        ),
        ("empty text", answer("  "), "Summarization produced no text"),
    ],
)
async def test_fails_with_model_error(name: str, response: AssistantMessage, message: str) -> None:
    chat = await open_compacting()
    await history(chat)
    for _ in range(3):
        chat.faux.summaries.append(response)
    outcome = await result(chat, await chat.root.compact(None, context))
    assert_match(outcome, {"status": "failed", "error": {"message": message, "detail": {"reason": "model_error"}}})
    # The retry policy allows two retries after the first attempt.
    assert len(chat.faux.summary_requests) == (3 if name == "retries run out" else 1)
    assert "compactions" not in await live(chat)
    assert "pi.compaction" not in await kinds(chat.root)
    await chat.harness.close(context)


# ─── Automatic compaction ─────────────────────────────────────────────────

#: Background threshold at 500 and blocking threshold at 1500 tokens of a 2000-token window.
BACKGROUND: dict[str, Any] = {"enabled": True, "reserveTokens": 500, "keepRecentTokens": 150, "backgroundTokens": 1000}
#: Blocking threshold at 700 tokens of a 1000-token window, no background compaction.
BLOCKING: dict[str, Any] = {"enabled": True, "reserveTokens": 300, "keepRecentTokens": 150, "backgroundTokens": 0}


async def compaction_tasks(chat: Chat) -> list[Any]:
    inspection = await chat.harness.inspect(context)
    return [task["record"] for task in inspection["tasks"] if task["record"]["kind"] == "pi.compaction"]


async def listen(chat: Chat) -> tuple[Any, list[Any]]:
    events: list[Any] = []
    stream = await watch_events(chat.harness, chat.root.id, context)

    async def listener(batch: Any, _context: Any) -> None:
        events.extend(batch)

    stream.start(listener)
    return stream, events


async def test_starts_above_the_background_threshold_without_blocking_the_run_idle_waits_and_esc_ignore_it() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    chat.setup.settings["compaction"] = BACKGROUND
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary(), reached))
    await turn(chat, text("u4", 100), text("a4", 100))
    await reached
    [task] = await compaction_tasks(chat)
    assert_match(task, {"background": True, "input": {"reason": "threshold"}})
    assert "owner" not in task
    assert (await live(chat))["compactions"] == [
        {"taskId": task["id"], "reason": "threshold", "blocking": False, "attempt": 1}
    ]
    # Background work: neither conversation idle nor Esc waits for or stops it.
    await chat.root.wait_for_idle(context)
    await chat.root.abort(context)
    assert (await chat.harness.get_task(task["id"], context))["state"]["status"] == "running"  # type: ignore[index]
    resolve(gate)
    outcome = await result(chat, task["id"])
    assert outcome["status"] == "completed"
    assert (await kinds(chat.root))[-1] == "pi.compaction"
    assert "SUMMARY" in first_user_text((await chat.root.context(context))["messages"])
    await chat.harness.close(context)


@pytest.mark.parametrize(
    ("name", "policy"),
    [
        ("disabled", {**BACKGROUND, "enabled": False}),
        ("backgroundTokens is 0", {**BACKGROUND, "backgroundTokens": 0}),
        ("there is no cut", {**BACKGROUND, "keepRecentTokens": 100_000}),
    ],
)
async def test_does_not_start_when(name: str, policy: Any) -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    chat.setup.settings["compaction"] = policy
    await turn(chat, text("u4", 100), text("a4", 100))
    assert await compaction_tasks(chat) == [], name
    assert len(chat.faux.summary_requests) == 0
    await chat.harness.close(context)


async def test_does_not_start_while_another_compaction_is_listed() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    manual = await chat.root.compact(None, context)
    await reached
    chat.setup.settings["compaction"] = BACKGROUND
    await turn(chat, text("u4", 100), text("a4", 100))
    assert [task["id"] for task in await compaction_tasks(chat)] == [manual]
    await chat.harness.abort_task(manual, context)
    await chat.harness.close(context)


async def test_records_a_failure_in_pi_live_the_snapshot_and_compaction_end_until_a_summary_is_produced() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    chat.setup.settings["compaction"] = BACKGROUND
    stream, events = await listen(chat)
    chat.faux.summaries.append(failure("bad request"))
    await turn(chat, text("u4", 100), text("a4", 100))
    await wait_for(lambda: any(event["type"] == "compaction_end" for event in events))
    recorded = (await live(chat)).get("compactionFailure")
    assert_match(recorded, {"reason": "threshold", "error": "Summarization failed: bad request"})
    assert next(event for event in events if event["type"] == "compaction_end") == {
        "type": "compaction_end",
        "taskId": recorded["taskId"],
        "reason": "threshold",
        "error": "Summarization failed: bad request",
    }
    reopened = await watch_events(chat.harness, chat.root.id, context)
    assert reopened.snapshot["compactionFailure"] == recorded
    await reopened.stop()
    # A compaction that produces a summary removes the failure, and ends without an error.
    chat.setup.settings["compaction"] = MANUAL
    chat.faux.summaries.append(summary())
    manual = await chat.root.compact(None, context)
    assert (await result(chat, manual))["status"] == "completed"
    assert "compactionFailure" not in await live(chat)
    await wait_for(lambda: any(event["type"] == "compaction_end" and event["taskId"] == manual for event in events))
    assert "error" not in next(
        event for event in events if event["type"] == "compaction_end" and event["taskId"] == manual
    )
    await stream.stop()
    await chat.harness.close(context)


async def test_stops_through_abort_task_and_conversation_abort_with_background() -> None:
    for stop in ("task", "conversation"):
        chat = await open_compacting(context_window=2000)
        await history(chat)
        chat.setup.settings["compaction"] = BACKGROUND
        reached = deferred()
        chat.faux.summaries.append(gated(deferred(), summary(), reached))
        await turn(chat, text("u4", 100), text("a4", 100))
        await reached
        [task] = await compaction_tasks(chat)
        if stop == "task":
            await chat.harness.abort_task(task["id"], context)
        else:
            await chat.root.abort(context, background=True)
        assert (await result(chat, task["id"]))["status"] == "aborted"
        assert "compactions" not in await live(chat)
        await chat.harness.close(context)


async def test_waits_for_its_compaction_which_appends_the_summary_before_the_request() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    # A new section: preparation has a system entry to append, but must not append it before the wait.
    add_section(chat.setup.registry, "extra", lambda *_: "EXTRA")
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary(), reached))
    chat.faux.agent.append(answer("a4"))
    input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
    await reached
    [child] = await compaction_tasks(chat)
    generation = (await live(chat))["run"]["taskId"]
    assert_match(child, {"owner": generation, "background": False, "input": {"reason": "threshold"}})
    record = await chat.harness.get_task(generation, context)
    assert_match(
        record["state"],  # type: ignore[index]
        {
            "status": "waiting",
            "on": [child["id"]],
            "checkpoint": {"phase": "prepare", "attempt": 1, "compacted": child["id"]},
        },
    )
    assert (await live(chat))["compactions"] == [
        {"taskId": child["id"], "reason": "threshold", "blocking": True, "attempt": 1}
    ]
    # Nothing was appended before the wait.
    assert (await kinds(chat.root))[-1] == "pi.user"
    resolve(gate)
    assert (await input.wait(context))["status"] == "done"
    assert (await result(chat, child["id"]))["status"] == "completed"
    assert (await kinds(chat.root))[-3:] == ["pi.compaction", "pi.system", "pi.assistant"]
    request = chat.faux.agent_requests[-1].messages
    assert "SUMMARY" in first_user_text(request)
    systems = [message for message in request if message.role == "system"]
    assert len(systems) == 1
    assert_match(systems[0], {"sections": {"preamble": "You are helpful.", "extra": "<extra>\nEXTRA\n</extra>"}})
    await chat.harness.close(context)


async def test_sends_the_request_once_without_a_second_compaction_when_the_kept_part_is_still_above_the_threshold() -> (
    None
):
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = {**BLOCKING, "keepRecentTokens": 700}
    chat.faux.summaries.append(summary())
    await turn(chat, text("u4", 400), "a4")
    assert len(chat.faux.summary_requests) == 1
    assert len([kind for kind in await kinds(chat.root) if kind == "pi.compaction"]) == 1
    await chat.harness.close(context)


async def test_sends_the_request_anyway_when_its_compaction_declines() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": lambda *_: {"decline": True}})
    await turn(chat, text("u4", 200), "a4")
    assert "pi.compaction" not in await kinds(chat.root)
    assert first_user_text(chat.faux.agent_requests[-1].messages) == text("u1", 100)
    assert "compactionFailure" not in await live(chat)
    await chat.harness.close(context)


# A failed compaction leaves a context it judged too large; sending it once filled a 1M window and got an empty
# `length` answer on every turn while the failure stayed invisible.
async def test_fails_the_run_with_the_compactions_error_without_a_request_when_its_compaction_fails() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    chat.faux.summaries.append(failure("bad request"))
    requests = len(chat.faux.agent_requests)
    input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
    assert_match(
        await input.wait(context),
        {
            "status": "unanswered",
            "reason": "model_error",
            "detail": "Compaction failed: Summarization failed: bad request",
        },
    )
    assert len(chat.faux.agent_requests) == requests
    assert (await kinds(chat.root))[-1] == "pi.user"
    state = await live(chat)
    assert "compactions" not in state
    assert_match(state.get("compactionFailure"), {"reason": "threshold", "error": "Summarization failed: bad request"})
    await chat.harness.close(context)


async def test_fails_the_run_when_its_compaction_is_aborted_directly() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    requests = len(chat.faux.agent_requests)
    input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
    await reached
    [child] = await compaction_tasks(chat)
    await chat.harness.abort_task(child["id"], context)
    assert_match(
        await input.wait(context), {"status": "unanswered", "reason": "model_error", "detail": "Compaction was aborted"}
    )
    assert len(chat.faux.agent_requests) == requests
    assert "pi.compaction" not in await kinds(chat.root)
    # Aborting is not a failure.
    assert "compactionFailure" not in await live(chat)
    await chat.harness.close(context)


async def test_is_aborted_with_its_generation_by_esc_before_the_generations_abort_handler() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    stream, events = await listen(chat)
    input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
    await reached
    [child] = await compaction_tasks(chat)
    await chat.root.abort(context)
    assert_match(await input.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert_match(await chat.harness.get_task(child["id"], context), {"state": {"outcome": {"status": "aborted"}}})
    await wait_for(lambda: any(event["type"] == "run_end" for event in events))
    types = [event["type"] for event in events]
    assert types.index("compaction_end") < types.index("run_end")
    await stream.stop()
    await chat.harness.close(context)


async def test_wins_over_a_background_compaction_still_in_flight_which_then_settles_stale() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    chat.setup.settings["compaction"] = BACKGROUND
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary("BACKGROUND"), reached))
    await turn(chat, text("u4", 100), text("a4", 100))
    await reached
    [background] = await compaction_tasks(chat)
    chat.faux.summaries.append(summary("BLOCKING"))
    await turn(chat, text("u5", 1000), "a5")
    assert "BLOCKING" in first_user_text((await chat.root.context(context))["messages"])
    resolve(gate)
    outcome = await result(chat, background["id"])
    assert_match(
        await (await submission_of(chat, outcome)).status(context), {"status": "unanswered", "reason": "stale"}
    )
    await chat.harness.close(context)


# ─── overflow compaction ──────────────────────────────────────────────────

ENABLED: dict[str, Any] = {**MANUAL, "enabled": True}


async def test_compacts_and_retries_with_the_same_attempt_leaving_the_error_out_of_the_retry() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.setup.settings["compaction"] = ENABLED
    chat.faux.summaries.append(summary())
    attempt: list[Any] = []
    chat.faux.agent.append(failure(OVERFLOW))

    async def fits(_request: Request) -> AssistantMessage:
        attempt.append((await live(chat)).get("generation", {}).get("attempt"))
        return answer("fits")

    chat.faux.agent.append(fits)
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    assert (await input.wait(context))["status"] == "done"
    assert attempt == [1]
    assert (await kinds(chat.root))[-5:] == ["pi.user", "pi.assistant", "pi.compaction", "pi.system", "pi.assistant"]
    [compaction] = [record for record in await all_entries(chat.root) if record["kind"] == "pi.compaction"]
    assert_match(compaction, {"data": {"reason": "overflow"}})
    retry = chat.faux.agent_requests[-1].messages
    assert retry[0].role == "system"
    assert "SUMMARY" in user_text(retry[1])
    assert not any(message.role == "assistant" and message.stop_reason == "error" for message in retry)
    await chat.harness.close(context)


async def test_fails_a_second_overflow_with_its_error_entry() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.setup.settings["compaction"] = ENABLED
    chat.faux.summaries.append(summary())
    chat.faux.agent.extend([failure(OVERFLOW), failure(OVERFLOW)])
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    assert_match(await input.wait(context), {"status": "unanswered", "reason": "model_error", "detail": OVERFLOW})
    assert len([kind for kind in await kinds(chat.root) if kind == "pi.compaction"]) == 1
    assert (await kinds(chat.root))[-1] == "pi.assistant"
    await chat.harness.close(context)


async def test_fails_without_compacting_when_compaction_is_disabled_even_for_a_retryable_looking_overflow() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.faux.agent.append(failure(f"overloaded: {OVERFLOW}"))
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    assert_match(await input.wait(context), {"status": "unanswered", "reason": "model_error"})
    assert len(chat.faux.agent_requests) == 4
    assert len(chat.faux.summary_requests) == 0
    await chat.harness.close(context)


async def test_fails_after_a_blocking_threshold_compaction_in_the_same_generation() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    chat.faux.summaries.append(summary())
    chat.faux.agent.append(failure(OVERFLOW))
    input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
    assert_match(await input.wait(context), {"status": "unanswered", "reason": "model_error"})
    assert len(chat.faux.summary_requests) == 1
    await chat.harness.close(context)


def _decline(chat: Chat) -> None:
    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": lambda *_: {"decline": True}})


def _fail(chat: Chat) -> None:
    chat.faux.summaries.append(failure("bad request"))


def _no_cut(chat: Chat) -> None:
    # Classification finds no cut, so no compaction starts and the ordinary failure carries the text.
    chat.setup.settings["compaction"] = {**ENABLED, "keepRecentTokens": 100_000}


@pytest.mark.parametrize(
    ("name", "prepare", "requests"), [("declines", _decline, 0), ("fails", _fail, 1), ("cannot cut", _no_cut, 0)]
)
async def test_fails_with_the_overflow_text_when_compaction(
    name: str, prepare: Callable[[Chat], None], requests: int
) -> None:
    chat = await open_compacting()
    await turn(chat, text("u1", 100), text("a1", 100))
    chat.setup.settings["compaction"] = ENABLED
    prepare(chat)
    chat.faux.agent.append(failure(OVERFLOW))
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    assert_match(await input.wait(context), {"status": "unanswered", "reason": "model_error", "detail": OVERFLOW})
    assert len(chat.faux.summary_requests) == requests, name
    await chat.harness.close(context)


# ─── compaction estimates and interactions ────────────────────────────────


@pytest.mark.parametrize("clock", ["real", "fixed"])
async def test_ignores_usage_measured_before_a_summary_placed_mid_run(clock: str) -> None:
    setup = model_setup(2000)
    if clock == "fixed":
        setup.now = lambda: 1_000
    chat = await open_compacting(context_window=2000, setup=setup)
    tool_gate = deferred()
    tool_reached = deferred()

    async def execute(*_args: Any) -> Any:
        resolve(tool_reached)
        await tool_gate
        return {"output": [TextContent(text("result", 200))]}

    add_tool(
        chat.setup.registry, define_tool(name="slow", description="slow", parameters=Type.Object({}), execute=execute)
    )
    await chat.root.configure({"tools": tools_named(chat.setup, "slow")}, context)
    await turn(chat, text("u1", 220), text("a1", 220))
    await turn(chat, text("u2", 220), text("a2", 220))
    await turn(chat, text("u3", 220), text("a3", 220))
    # Background at 900, blocking at 1500: the tool call's usage plus its result would cross 1500.
    chat.setup.settings["compaction"] = {**BACKGROUND, "backgroundTokens": 600}
    # The request starts a background compaction; its tool call's usage measures the whole context.
    chat.faux.agent.append(faux_assistant_message([faux_tool_call("slow", {})], stop_reason="toolUse"))
    chat.faux.agent.append(answer("done"))
    summary_gate = deferred()
    summary_reached = deferred()
    chat.faux.summaries.append(gated(summary_gate, summary(), summary_reached))
    input = await chat.root.submit({"type": "input", "content": text("u4", 50)}, context)
    await asyncio.gather(tool_reached, summary_reached)
    [background] = await compaction_tasks(chat)
    resolve(summary_gate)
    # Queued: the run is busy in its tool round.
    queued = await result(chat, background["id"])
    assert (await (await submission_of(chat, queued)).status(context))["status"] == "queued"
    resolve(tool_gate)
    assert (await input.wait(context))["status"] == "done"
    # The summary landed at postTools; the successor saw a small context and did not compact again.
    assert len(chat.faux.summary_requests) == 1
    assert await compaction_tasks(chat) == []
    assert "SUMMARY" in first_user_text(chat.faux.agent_requests[-1].messages)
    await chat.harness.close(context)


async def test_rebaselines_the_system_prompt_over_kept_system_deltas() -> None:
    chat = await open_compacting()
    mood = ["cheerful"]
    add_section(chat.setup.registry, "mood", lambda *_: mood[0], tag=False)
    await turn(chat, text("u1", 100), text("a1", 100))
    await turn(chat, text("u2", 100), text("a2", 100))
    mood[0] = "terse"
    await turn(chat, text("u3", 100), text("a3", 100))
    chat.setup.settings["compaction"] = {**MANUAL, "keepRecentTokens": 250}
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    # The kept range holds the delta for the terse mood; the next request has one complete baseline.
    assert any(record["kind"] == "pi.system" for record in (await chat.root.context(context))["entries"])
    await turn(chat, "next", "ok")
    systems = [message for message in chat.faux.agent_requests[-1].messages if message.role == "system"]
    assert len(systems) == 1
    assert_match(systems[0], {"sections": {"preamble": "You are helpful.", "mood": "terse"}})
    await chat.harness.close(context)


async def test_summarizes_a_replaced_entrys_replacement_and_shows_it_to_the_hook() -> None:
    chat = await open_compacting()
    await history(chat)
    [u1, *_rest] = await all_entries(chat.root)
    await chat.root.submit(
        {
            "type": "write",
            "entry": {
                "kind": "app.redact",
                "edits": [
                    {"target": u1["id"], "action": "replace", "messages": [UserMessage("REDACTED", timestamp=0)]}
                ],
            },
        },
        context,
    )
    messages: list[Any] = []

    def before_compact(compaction: Any, *_rest: Any) -> None:
        messages[:] = compaction["messages"]

    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": before_compact})
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    assert first_user_text(messages) == "REDACTED"
    assert "[User]: REDACTED" in user_text(chat.faux.summary_requests[0].messages[1])
    await chat.harness.close(context)


async def test_compacts_a_fork_whose_cut_falls_on_a_parent_entry() -> None:
    chat = await open_compacting()
    await history(chat)
    entries = await all_entries(chat.root)
    fork = await chat.root.fork(entries[-1]["id"], {"ownership": {"kind": "ownerless"}}, context)
    chat.faux.summaries.append(summary())
    outcome = await result(chat, await fork.compact(None, context))
    assert (await (await submission_of(chat, outcome)).wait(context))["status"] == "done"
    u3 = next(record for record in entries if user_text((record.get("model") or [None])[0]).startswith("u3"))
    fork_view = await fork.context(context)
    assert_match(fork_view["head"], {"kind": "pi.compaction", "head": u3["id"], "conversationId": fork.id})
    assert [user_text(message) for message in fork_view["messages"][1:]] == [text("u3", 100), text("a3", 100)]
    # The parent is untouched.
    assert "pi.compaction" not in await kinds(chat.root)
    # The fork's view keeps the parent entries the summary kept.
    state = await fork.view_state(context)
    assert list(state.value["entries"]) == list(fork_view["entries"])
    state.dispose()
    await chat.harness.close(context)


async def test_settles_stale_in_a_fork_reset_while_it_summarizes() -> None:
    chat = await open_compacting()
    await history(chat)
    entries = await all_entries(chat.root)
    fork = await chat.root.fork(entries[-1]["id"], {"ownership": {"kind": "ownerless"}}, context)
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary(), reached))
    id = await fork.compact(None, context)
    await reached
    await fork.reset(None, context)
    resolve(gate)
    submission = await submission_of(chat, await result(chat, id))
    assert_match(await submission.status(context), {"status": "unanswered", "reason": "stale"})
    assert (await fork.context(context))["head"]["kind"] == "pi.reset"
    await chat.harness.close(context)


async def test_places_an_older_queued_summary_and_the_current_one_together_when_idle_admission_drains_the_inbox() -> (
    None
):
    chat = await open_compacting()
    await history(chat)
    gate = deferred()
    reached = deferred()
    chat.faux.agent.append(gated(gate, failure("bad request"), reached))
    failed = await chat.root.submit({"type": "input", "content": "fails"}, context)
    await reached
    chat.faux.summaries.append(summary("OLDER"))
    older = await result(chat, await chat.root.compact(None, context))
    resolve(gate)
    await failed.wait(context)
    # Idle now, with the older summary still queued; the current one queues behind it and a final boundary runs.
    chat.faux.summaries.append(summary("CURRENT"))
    current = await result(chat, await chat.root.compact(None, context))
    statuses = [(await (await submission_of(chat, outcome)).status(context))["status"] for outcome in (older, current)]
    assert statuses == ["done", "done"]
    assert "CURRENT" in first_user_text((await chat.root.context(context))["messages"])
    await chat.harness.close(context)


def define_child_task(gate: asyncio.Future[Any]) -> Any:
    async def run(_task: Any, runtime: Any, task_context: Context) -> None:
        await gate
        await runtime.commit(
            lambda _tx, _current: {"status": "terminal", "outcome": {"status": "completed", "result": None}},
            task_context,
        )

    async def abort(_task: Any, runtime: Any, task_context: Context) -> None:
        await runtime.commit(
            lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, task_context
        )

    return define_task(
        name="test.child", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=abort
    )


async def test_places_a_hooks_summary_and_holds_while_work_the_hook_created_runs() -> None:
    chat = await open_compacting()
    child_gate = deferred()
    child = define_child_task(child_gate)
    add_task(chat.setup.registry, child)

    async def before_compact(_compaction: Any, api: Any, hook_context: Context) -> Any:
        await chat.harness.commit(
            lambda tx: tx.create_task(
                child, {}, {"ownership": {"kind": "task", "taskId": api.task_id}, "conversationId": chat.root.id}
            ),
            hook_context,
        )
        return {"summary": "HOOK"}

    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": before_compact})
    await history(chat)
    id = await chat.root.compact(None, context)

    async def completing() -> bool:
        record = await chat.harness.get_task(id, context)
        return record is not None and record["state"]["status"] == "completing"

    await wait_for(completing)
    # The summary and the status removal landed at the hold.
    assert (await kinds(chat.root))[-1] == "pi.compaction"
    assert "compactions" not in await live(chat)
    resolve(child_gate)
    assert (await result(chat, id))["status"] == "completed"
    await chat.harness.close(context)


async def test_removes_the_status_of_a_faulted_compaction() -> None:
    chat = await open_compacting()
    await history(chat)

    def broken(*_args: Any) -> Any:
        raise RuntimeError("models broke")

    chat.setup.models.get_model = broken
    try:
        outcome = await result(chat, await chat.root.compact(None, context))
    finally:
        del chat.setup.models.get_model
    assert_match(outcome, {"status": "faulted", "error": {"message": "models broke"}})
    assert "compactions" not in await live(chat)
    await chat.harness.close(context)


# ─── compaction events and live status ────────────────────────────────────


async def test_reports_start_and_end_and_the_retry_backoff_in_a_late_joiners_snapshot() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.setup.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 60_000}
    stream, events = await listen(chat)
    chat.faux.summaries.append(failure("overloaded"))
    id = await chat.root.compact(None, context)

    async def retrying() -> bool:
        compactions = (await live(chat)).get("compactions")
        return bool(compactions) and "retry" in compactions[0]

    await wait_for(retrying)
    late = await watch_events(chat.harness, chat.root.id, context)
    [status] = late.snapshot["compactions"]
    assert isinstance(status["retry"]["at"], (int, float))
    assert status == {
        "taskId": id,
        "reason": "manual",
        "blocking": False,
        "attempt": 1,
        "retry": {"at": status["retry"]["at"], "error": "overloaded"},
    }
    await late.stop()
    await chat.harness.abort_task(id, context)
    await wait_for(lambda: any(event["type"] == "compaction_end" for event in events))
    assert [event for event in events if event["type"].startswith("compaction_")] == [
        {"type": "compaction_start", "taskId": id, "reason": "manual", "blocking": False},
        {"type": "compaction_end", "taskId": id, "reason": "manual"},
    ]
    await stream.stop()
    await chat.harness.close(context)


async def test_lists_concurrent_compactions_in_task_id_order() -> None:
    chat = await open_compacting()
    await history(chat)
    first = deferred()
    second = deferred()
    chat.faux.summaries.extend([gated(deferred(), summary(), first), gated(deferred(), summary(), second)])
    a = await chat.root.compact(None, context)
    b = await chat.root.compact(None, context)
    await asyncio.gather(first, second)
    assert [status["taskId"] for status in (await live(chat))["compactions"]] == [a, b]
    await chat.root.abort(context)
    assert "compactions" not in await live(chat)
    await chat.harness.close(context)


# ─── compaction recovery ──────────────────────────────────────────────────


async def reopen(path: Path, setup: ChatSetup, faux: Script) -> Chat:
    opened = await open_chat(await open_sqlite_storage(str(path)), setup)
    opened.harness.resume()
    return Chat(opened.harness, opened.root, setup, faux)


async def first_open(path: Path) -> Chat:
    setup = chat_setup()
    add_section(setup.registry, "preamble", lambda *_: "You are helpful.", tag=False)
    faux = script(setup)
    opened = await open_chat(await open_sqlite_storage(str(path)), setup)
    setup.settings["compaction"] = MANUAL
    setup.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 1}
    opened.harness.resume()
    chat = Chat(opened.harness, opened.root, setup, faux)
    await history(chat)
    return chat


async def test_repeats_nothing_after_reopen_once_the_summary_is_placed(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    chat = await first_open(path)
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    entries = await all_entries(chat.root)
    usage = await chat.harness.snapshot(UsageDoc, chat.root.id, context)
    await chat.harness.close(context)
    chat = await reopen(path, chat.setup, chat.faux)
    await chat.harness.wait_for_idle(context)
    assert len(chat.faux.summary_requests) == 1
    assert await all_entries(chat.root) == entries
    assert await chat.harness.snapshot(UsageDoc, chat.root.id, context) == usage
    inspection = await chat.harness.inspect(context)
    assert [inspection["tasks"], inspection["submissions"]] == [[], []]
    await chat.harness.close(context)


async def test_fails_an_overflow_run_with_its_text_when_its_compaction_fails_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    chat = await first_open(path)
    chat.setup.settings["compaction"] = {**MANUAL, "enabled": True}
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    chat.faux.agent.append(failure(OVERFLOW))
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    await reached
    generation = (await live(chat))["run"]["taskId"]
    assert_match(
        await chat.harness.get_task(generation, context),
        {"state": {"status": "waiting", "checkpoint": {"phase": "prepare", "overflow": OVERFLOW}}},
    )
    await chat.harness.close(context)
    chat = await reopen(path, chat.setup, chat.faux)
    chat.faux.summaries.append(failure("bad request"))
    submission = await chat.harness.submission(input.id, context)
    assert submission is not None
    assert_match(await submission.wait(context), {"status": "unanswered", "reason": "model_error", "detail": OVERFLOW})
    await chat.harness.close(context)


async def test_reruns_select_and_its_hook_after_a_crash_in_select(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    chat = await first_open(path)
    calls = [0]
    reached = deferred()

    async def before_compact(_compaction: Any, _api: Any, hook_context: Context) -> None:
        calls[0] += 1
        if calls[0] == 1:
            resolve(reached)
            await aborted(hook_context.abort_signal)  # type: ignore[arg-type]

    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": before_compact})
    id = await chat.root.compact(None, context)
    await reached
    await chat.harness.close(context)
    chat = await reopen(path, chat.setup, chat.faux)
    chat.faux.summaries.append(summary())
    assert (await result(chat, id))["status"] == "completed"
    assert calls[0] == 2
    await chat.harness.close(context)


async def test_resends_an_interrupted_summarization_once_and_counts_only_the_answered_attempt(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    chat = await first_open(path)
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    id = await chat.root.compact(None, context)
    await reached
    usage_before = await models_usage(chat)
    assert_match(await chat.harness.get_task(id, context), {"state": {"checkpoint": {"phase": "summarize"}}})
    await chat.harness.close(context)
    chat = await reopen(path, chat.setup, chat.faux)
    chat.faux.summaries.append(summary())
    assert (await result(chat, id))["status"] == "completed"
    assert len(chat.faux.summary_requests) == 2
    first, resent = chat.faux.summary_requests
    assert [dataclasses.replace(message, timestamp=0) for message in resent.messages] == [
        dataclasses.replace(message, timestamp=0) for message in first.messages
    ]
    usage = await models_usage(chat)
    assert usage["input"] > usage_before["input"]
    assert usage["output"] - usage_before["output"] == 2
    await chat.harness.close(context)


async def test_resumes_a_retry_backoff_after_reopen_compaction(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    chat = await first_open(path)
    now = [int(time.time() * 1000)]
    chat.setup.now = lambda: now[0]
    chat.setup.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 60_000}
    chat.faux.summaries.append(failure("overloaded"))
    id = await chat.root.compact(None, context)

    def sleeping(chat: Chat) -> Callable[[], Awaitable[bool]]:
        async def check() -> bool:
            record = await chat.harness.get_task(id, context)
            return record is not None and record["state"]["status"] == "sleeping"

        return check

    await wait_for(sleeping(chat))
    until = now[0] + 60_000
    assert_match(
        await chat.harness.get_task(id, context),
        {
            "state": {
                "status": "sleeping",
                "checkpoint": {"phase": "retry", "attempt": 1, "until": until},
                "until": until,
            }
        },
    )
    await chat.harness.close(context)
    # Earlier versions slept inside a running `retry` phase: reopened before its time, it sleeps again.
    database = sqlite3.connect(path)
    database.execute(
        "UPDATE tasks SET status = 'running', record = json_set(json_remove(record, '$.state.until'), "
        "'$.state.status', 'running') WHERE id = ?",
        (id,),
    )
    database.commit()
    database.close()
    chat = await reopen(path, chat.setup, chat.faux)
    await wait_for(sleeping(chat))
    assert len(chat.faux.summary_requests) == 1
    await chat.harness.close(context)
    now[0] += 120_000
    chat = await reopen(path, chat.setup, chat.faux)
    chat.faux.summaries.append(summary())
    assert (await result(chat, id))["status"] == "completed"
    assert len(chat.faux.summary_requests) == 2
    await chat.harness.close(context)


async def test_keeps_a_generation_waiting_on_its_blocking_compaction_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    chat = await first_open(path)
    setup = chat.setup
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    # The faux model window is 128k; this blocking threshold needs a small context window.
    chat.setup.settings["compaction"] = {**BLOCKING, "reserveTokens": 128_000 - 700}
    input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
    await reached
    await chat.harness.close(context)
    chat = await reopen(path, setup, chat.faux)
    chat.faux.summaries.append(summary())
    chat.faux.agent.append(answer("a4"))
    submission = await chat.harness.submission(input.id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    assert (await kinds(chat.root))[-3:] == ["pi.compaction", "pi.system", "pi.assistant"]
    await chat.harness.close(context)


async def test_keeps_a_queued_summary_across_reopen_and_places_it_at_the_next_boundary(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    chat = await first_open(path)
    reached = deferred()
    chat.faux.agent.append(gated(deferred(), answer("never"), reached))
    input = await chat.root.submit({"type": "input", "content": "busy"}, context)
    await reached
    chat.faux.summaries.append(summary())
    outcome = await result(chat, await chat.root.compact(None, context))
    await chat.harness.close(context)
    chat = await reopen(path, chat.setup, chat.faux)
    chat.faux.agent.append(answer("answered"))
    submission = await chat.harness.submission(input.id, context)
    assert submission is not None
    assert (await submission.wait(context))["status"] == "done"
    assert (await (await submission_of(chat, outcome)).wait(context))["status"] == "done"
    await chat.harness.close(context)


# ─── compaction and the inbox ─────────────────────────────────────────────


async def queued_summary(chat: Chat, content: str = "SUMMARY") -> Any:
    chat.faux.summaries.append(summary(content))
    return await submission_of(chat, await result(chat, await chat.root.compact(None, context)))


@dataclasses.dataclass(slots=True)
class Busy:
    input: Any
    gate: asyncio.Future[Any]

    def release(self) -> None:
        resolve(self.gate)


async def busy(chat: Chat, reply: AssistantMessage | None = None) -> Busy:
    gate = deferred()
    reached = deferred()
    chat.faux.agent.append(gated(gate, reply if reply is not None else answer("done"), reached))
    input = await chat.root.submit({"type": "input", "content": "busy"}, context)
    await reached
    return Busy(input, gate)


async def test_places_a_reset_queued_after_the_summary_last_and_makes_a_summary_queued_after_a_reset_stale() -> None:
    for order in ("summary first", "reset first"):
        chat = await open_compacting()
        await history(chat)
        run = await busy(chat)
        if order == "summary first":
            submission = await queued_summary(chat)
            await chat.root.reset(None, context)
        else:
            await chat.root.reset(None, context)
            submission = await queued_summary(chat)
        run.release()
        await run.input.wait(context)
        settled = await submission.wait(context)
        assert settled["status"] == ("done" if order == "summary first" else "unanswered")
        assert (await kinds(chat.root))[-1] == "pi.reset"
        assert (await chat.root.context(context))["head"]["kind"] == "pi.reset"
        await chat.harness.close(context)


async def test_places_a_summary_left_queued_by_a_failed_run_at_the_next_submission_before_its_input() -> None:
    chat = await open_compacting()
    await history(chat)
    run = await busy(chat, failure("bad request"))
    submission = await queued_summary(chat)
    run.release()
    assert (await run.input.wait(context))["status"] == "unanswered"
    assert (await submission.status(context))["status"] == "queued"
    await turn(chat, "again", "ok")
    assert (await submission.status(context))["status"] == "done"
    request = chat.faux.agent_requests[-1].messages
    assert "SUMMARY" in first_user_text(request)
    assert "again" in [user_text(message) for message in request]
    await chat.harness.close(context)


async def test_keeps_the_full_retry_budget_after_an_overflow_compaction() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.setup.settings["compaction"] = {**MANUAL, "enabled": True}
    chat.faux.summaries.append(summary())
    chat.faux.agent.extend(
        [failure("prompt is too long"), failure("overloaded"), failure("overloaded"), answer("finally")]
    )
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    assert (await input.wait(context))["status"] == "done"
    await chat.harness.close(context)


async def test_loses_an_application_edit_placed_while_a_compaction_summarizes_spec_12() -> None:
    chat = await open_compacting()
    await history(chat)
    [u1, *_rest] = await all_entries(chat.root)
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary(), reached))
    id = await chat.root.compact(None, context)
    await reached
    edit = {"target": u1["id"], "action": "replace", "messages": [user("REDACTED")]}
    await chat.root.submit({"type": "write", "entry": {"kind": "app.redact", "edits": [edit]}}, context)
    resolve(gate)
    await result(chat, id)
    # The summary was made from the unredacted entry, and the edit's target left the range.
    assert "[User]: u1 " in user_text(chat.faux.summary_requests[0].messages[1])
    assert "REDACTED" not in [user_text(message) for message in (await chat.root.context(context))["messages"]]
    await chat.harness.close(context)


async def test_keeps_the_kept_entries_mounted_in_the_conversation_view() -> None:
    chat = await open_compacting()
    await history(chat)
    state = await chat.root.view_state(context)
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    await wait_for(lambda: bool(state.value["entries"]) and state.value["entries"][0]["kind"] == "pi.compaction")
    assert list(state.value["entries"]) == list((await chat.root.context(context))["entries"])
    state.dispose()
    await chat.harness.close(context)


# ─── compaction edge cases ────────────────────────────────────────────────


async def test_keeps_the_one_compaction_limit_through_a_retry_backoff() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    asked = [0]

    def before_compact(*_args: Any) -> Any:
        asked[0] += 1
        return {"decline": True}

    add_hooks(chat.setup.registry, CompactionTask, {"before_compact": before_compact})
    chat.faux.agent.append(failure("overloaded"))
    await turn(chat, text("u4", 200), "a4")
    # One blocking compaction, declined; the retried preparation, still above the threshold, started no other.
    assert asked[0] == 1
    assert len(chat.faux.agent_requests) == 5
    await chat.harness.close(context)


async def test_does_not_start_a_background_compaction_when_a_manual_one_was_admitted_during_preparation() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    chat.setup.settings["compaction"] = BACKGROUND
    rendering = deferred()
    release = deferred()
    hold = [True]

    async def slow(*_args: Any) -> str:
        if hold[0]:
            hold[0] = False
            resolve(rendering)
            await release
        return "slow"

    add_section(chat.setup.registry, "slow", slow)
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    chat.faux.agent.append(answer("a4"))
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    await rendering
    manual = await chat.root.compact(None, context)
    await reached
    resolve(release)
    assert (await input.wait(context))["status"] == "done"
    assert [task["id"] for task in await compaction_tasks(chat)] == [manual]
    # A background compaction would have sent its own summarization request.
    assert len(chat.faux.summary_requests) == 1
    await chat.harness.abort_task(manual, context)
    await chat.harness.close(context)


async def test_starts_no_background_compaction_after_a_blocking_one_in_the_same_generation() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    # Blocking at 1500, background at 200: the kept part stays above the background threshold.
    chat.setup.settings["compaction"] = {**BACKGROUND, "backgroundTokens": 1300, "keepRecentTokens": 400}
    chat.faux.summaries.append(summary())
    await turn(chat, text("u4", 1000), "a4")
    assert len(chat.faux.summary_requests) == 1
    assert await compaction_tasks(chat) == []
    await chat.harness.close(context)


async def test_fails_the_run_when_its_blocking_compaction_faults_and_records_the_fault() -> None:
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("no credentials")

    chat.setup.models.complete_simple = broken
    try:
        input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
        assert_match(
            await input.wait(context),
            {"status": "unanswered", "reason": "model_error", "detail": "Compaction failed: no credentials"},
        )
    finally:
        del chat.setup.models.complete_simple
    assert "pi.compaction" not in await kinds(chat.root)
    state = await live(chat)
    assert "compactions" not in state
    assert_match(state.get("compactionFailure"), {"reason": "threshold", "error": "no credentials"})
    await chat.harness.close(context)


async def test_fails_an_overflow_run_with_its_text_when_its_compaction_is_aborted_directly_or_itself_overflows() -> (
    None
):
    for how in ("abort", "overflow"):
        chat = await open_compacting()
        await history(chat)
        chat.setup.settings["compaction"] = {**MANUAL, "enabled": True}
        reached = deferred()
        if how == "abort":
            chat.faux.summaries.append(gated(deferred(), summary(), reached))
        else:
            chat.faux.summaries.append(failure("prompt is too long for the summary"))
        chat.faux.agent.append(failure(OVERFLOW))
        input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
        if how == "abort":
            await reached
            [child] = await compaction_tasks(chat)
            await chat.harness.abort_task(child["id"], context)
        assert_match(await input.wait(context), {"status": "unanswered", "reason": "model_error", "detail": OVERFLOW})
        assert "compactions" not in await live(chat)
        await chat.harness.close(context)


async def test_treats_a_length_stop_as_an_ordinary_answer_not_an_overflow() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.setup.settings["compaction"] = {**MANUAL, "enabled": True}
    chat.faux.agent.append(faux_assistant_message("cut short", stop_reason="length", error_message=OVERFLOW))
    input = await chat.root.submit({"type": "input", "content": "go"}, context)
    assert (await input.wait(context))["status"] == "done"
    assert len(chat.faux.summary_requests) == 0
    await chat.harness.close(context)


async def test_orders_the_events_of_a_summary_placed_at_once() -> None:
    chat = await open_compacting()
    await history(chat)
    stream, events = await listen(chat)
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    await wait_for(lambda: any(event["type"] == "compaction_end" for event in events))
    end = next(index for index, event in enumerate(events) if event["type"] == "compaction_end")
    assert [event["type"] for event in events[end - 2 : end + 3]] == [
        "message_start",
        "message_end",
        "compaction_end",
        "submission",
        "usage_changed",
    ]
    await stream.stop()
    await chat.harness.close(context)


async def test_puts_compaction_start_last_in_the_batch_of_the_preparation_commit() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    chat.setup.settings["compaction"] = BACKGROUND
    batches: list[list[Any]] = []
    stream = await watch_events(chat.harness, chat.root.id, context)

    async def listener(batch: Any, _context: Any) -> None:
        batches.append(list(batch))

    stream.start(listener)
    # A new section makes the preparation commit append a system entry next to the compaction's status.
    add_section(chat.setup.registry, "extra", lambda *_: "extra")
    chat.faux.summaries.append(gated(deferred(), summary()))
    await turn(chat, text("u4", 100), "a4")

    def started(batch: list[Any]) -> bool:
        return any(event["type"] == "compaction_start" for event in batch)

    await wait_for(lambda: any(started(batch) for batch in batches))
    batch = next(candidate for candidate in batches if started(candidate))
    assert [event["type"] for event in batch] == ["message_start", "message_end", "compaction_start"]
    await stream.stop()
    await chat.harness.abort_task((await compaction_tasks(chat))[0]["id"], context)
    await chat.harness.close(context)


async def test_keeps_a_background_summary_queued_through_a_retry_backoff_while_a_blocking_compaction_wins() -> None:
    chat = await open_compacting(context_window=2000)
    await history(chat)
    chat.setup.settings["compaction"] = BACKGROUND
    chat.setup.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 300}
    summary_gate = deferred()
    summary_reached = deferred()
    chat.faux.summaries.append(gated(summary_gate, summary("BACKGROUND"), summary_reached))
    chat.faux.summaries.append(summary("BLOCKING"))
    failed = deferred()

    async def fail_once(_request: Request) -> AssistantMessage:
        resolve(failed)
        return failure("overloaded")

    chat.faux.agent.append(fail_once)
    chat.faux.agent.append(answer("a4"))
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    await asyncio.gather(failed, summary_reached)

    # During the backoff: the background summary queues, and the thresholds drop so the retry blocks.
    async def backing_off() -> bool:
        return "retry" in (await live(chat)).get("generation", {})

    await wait_for(backing_off)
    [background] = await compaction_tasks(chat)
    resolve(summary_gate)
    submission = await submission_of(chat, await result(chat, background["id"]))
    assert (await submission.status(context))["status"] == "queued"
    chat.setup.settings["compaction"] = {**BACKGROUND, "reserveTokens": 1500, "keepRecentTokens": 50}
    assert (await input.wait(context))["status"] == "done"
    assert "BLOCKING" in first_user_text(chat.faux.agent_requests[-1].messages)
    assert_match(await submission.wait(context), {"status": "unanswered", "reason": "stale"})
    await chat.harness.close(context)


async def test_summarizes_the_previous_summary_in_a_second_compaction() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.faux.summaries.append(summary("FIRST"))
    await result(chat, await chat.root.compact(None, context))
    await turn(chat, text("u4", 100), text("a4", 100))
    await turn(chat, text("u5", 100), text("a5", 100))
    chat.faux.summaries.append(summary("SECOND"))
    await result(chat, await chat.root.compact(None, context))
    prompt = user_text(chat.faux.summary_requests[1].messages[1])
    assert re.match(r"^<conversation>\n\[User\]: The conversation history before this point was compacted", prompt)
    assert "FIRST" in prompt
    assert "SECOND" in first_user_text((await chat.root.context(context))["messages"])
    await chat.harness.close(context)


async def test_leaves_no_usage_submission_or_summary_when_the_classifying_commit_fails_which_fails_the_harness() -> (
    None
):
    storage = ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])
    chat = await open_compacting(storage=storage)
    await history(chat)
    before = await models_usage(chat)
    error = RuntimeError("disk gone")

    def failing(_request: Request) -> AssistantMessage:
        storage.fail_next_commit(error)
        return summary()

    chat.faux.summaries.append(failing)
    id = await chat.root.compact(None, context)
    assert await chat.harness.closed == {"reason": "failed", "error": error}
    await chat.harness.close(context)
    # Nothing of the failed commit landed: the task is still live, without usage, entry, or submission.
    await storage.reopen()
    assert await storage.submission_by_request(chat.root.id, f"compaction:{id}", context) is None
    task = await storage.task(id, context)
    assert task is not None and task["state"]["status"] != "terminal"
    reopened = await open_chat(storage, chat.setup)
    usage = await reopened.harness.snapshot(UsageDoc, reopened.root.id, context)
    assert usage is not None and usage["models"].get("faux/faux-1") == before
    assert "pi.compaction" not in await kinds(reopened.root)
    await reopened.harness.close(context)


# ─── blocking and manual compaction together ──────────────────────────────


@dataclasses.dataclass(slots=True)
class BlockingRun:
    input: Any
    gate: asyncio.Future[Any]
    blocking: Any


async def blocking_run(chat: Chat) -> BlockingRun:
    """A run whose generation waits on a blocking compaction whose summary is held."""
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    gate = deferred()
    reached = deferred()
    chat.faux.summaries.append(gated(gate, summary("BLOCKING"), reached))
    input = await chat.root.submit({"type": "input", "content": text("u4", 200)}, context)
    await reached
    [blocking] = await compaction_tasks(chat)
    return BlockingRun(input, gate, blocking["id"])


async def test_places_a_manual_summary_selected_before_the_blocking_one_landed_its_equal_cut_replaces_it() -> None:
    chat = await open_compacting(context_window=1000)
    run = await blocking_run(chat)
    # Selected from the same context as the blocking compaction, so it cuts at the same entry.
    chat.faux.summaries.append(summary("MANUAL"))
    submission = await submission_of(chat, await result(chat, await chat.root.compact(None, context)))
    assert (await submission.status(context))["status"] == "queued"
    chat.faux.agent.append(answer("a4"))
    resolve(run.gate)
    assert (await run.input.wait(context))["status"] == "done"
    # The request after the blocking compaction used its summary; the manual one landed at the final boundary.
    assert "BLOCKING" in first_user_text(chat.faux.agent_requests[-1].messages)
    assert (await submission.wait(context))["status"] == "done"
    markers = [record for record in await all_entries(chat.root) if record["kind"] == "pi.compaction"]
    assert len(markers) == 2
    assert markers[1]["head"] == markers[0]["head"]
    messages = (await chat.root.context(context))["messages"]
    assert "MANUAL" in first_user_text(messages)
    assert not any("BLOCKING" in user_text(message) for message in messages)
    await chat.harness.close(context)


async def test_finds_nothing_to_compact_for_a_manual_compaction_selected_after_the_blocking_summary_landed() -> None:
    chat = await open_compacting(context_window=1000)
    run = await blocking_run(chat)
    answer_gate = deferred()
    answer_reached = deferred()
    chat.faux.agent.append(gated(answer_gate, answer("a4"), answer_reached))
    resolve(run.gate)
    await answer_reached
    assert await result(chat, await chat.root.compact(None, context)) == {"status": "completed", "result": {}}
    assert len(chat.faux.summary_requests) == 1
    resolve(answer_gate)
    assert (await run.input.wait(context))["status"] == "done"
    await chat.harness.close(context)


async def test_aborts_the_run_and_both_compactions_on_esc_and_appends_nothing() -> None:
    chat = await open_compacting(context_window=1000)
    run = await blocking_run(chat)
    manual_reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary("MANUAL"), manual_reached))
    manual = await chat.root.compact(None, context)
    await manual_reached
    assert [status["blocking"] for status in (await live(chat))["compactions"]] == [True, False]
    await chat.root.abort(context)
    assert_match(await run.input.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert (await result(chat, run.blocking))["status"] == "aborted"
    assert (await result(chat, manual))["status"] == "aborted"
    assert "compactions" not in await live(chat)
    assert "run" not in await live(chat)
    assert "pi.compaction" not in await kinds(chat.root)
    assert (await chat.harness.inspect(context))["submissions"] == []
    await chat.harness.close(context)


async def test_places_a_summary_that_survived_esc_at_the_next_submission_before_its_input() -> None:
    chat = await open_compacting()
    await history(chat)
    reached = deferred()
    chat.faux.agent.append(gated(deferred(), answer("never"), reached))
    busy_input = await chat.root.submit({"type": "input", "content": "busy"}, context)
    await reached
    chat.faux.summaries.append(summary())
    submission = await submission_of(chat, await result(chat, await chat.root.compact(None, context)))
    await chat.root.abort(context)
    assert_match(await busy_input.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert (await submission.status(context))["status"] == "queued"
    await turn(chat, "u5", "a5")
    assert (await submission.status(context))["status"] == "done"
    request = chat.faux.agent_requests[-1].messages
    assert "SUMMARY" in first_user_text(request)
    assert "u5" in [user_text(message) for message in request]
    tail = [record["kind"] for record in (await all_entries(chat.root))[-4:]]
    assert tail == ["pi.compaction", "pi.user", "pi.system", "pi.assistant"]
    await chat.harness.close(context)


# ─── compaction pinning, silent overflow, and late policy changes ─────────


async def test_keeps_the_pinned_model_through_a_retry_and_advances_the_live_attempt() -> None:
    setup = chat_setup(
        models=[
            {"id": "faux-1", "context_window": 100_000, "max_tokens": 900},
            {"id": "faux-2", "context_window": 100_000, "max_tokens": 900},
        ]
    )
    chat = await open_compacting(setup=setup)
    await history(chat)
    attempt: list[Any] = []

    async def switching(_request: Request) -> AssistantMessage:
        # Switched during the attempt: the retry still uses the pinned model.
        await chat.root.configure({"model": {"provider": "faux", "modelId": "faux-2"}}, context)
        return failure("overloaded")

    async def answering(_request: Request) -> AssistantMessage:
        compactions = (await live(chat)).get("compactions") or [{}]
        attempt.append(compactions[0].get("attempt"))
        return summary()

    chat.faux.summaries.extend([switching, answering])
    assert (await result(chat, await chat.root.compact(None, context)))["status"] == "completed"
    assert [request.model for request in chat.faux.summary_requests] == ["faux-1", "faux-1"]
    assert attempt == [2]
    usage = await chat.harness.snapshot(UsageDoc, chat.root.id, context)
    assert usage is not None and list(usage["models"].keys()) == ["faux/faux-1"]
    await chat.harness.close(context)


async def test_shows_a_late_joiner_a_compaction_that_is_summarizing() -> None:
    chat = await open_compacting()
    await history(chat)
    reached = deferred()
    chat.faux.summaries.append(gated(deferred(), summary(), reached))
    id = await chat.root.compact(None, context)
    await reached
    stream = await watch_events(chat.harness, chat.root.id, context)
    assert stream.snapshot["compactions"] == [{"taskId": id, "reason": "manual", "blocking": False, "attempt": 1}]
    await stream.stop()
    await chat.harness.abort_task(id, context)
    await chat.harness.close(context)


@pytest.mark.parametrize(
    ("name", "response"),
    [
        ("a stop whose input exceeds the window", answer("fine")),
        ("a length stop that fills the window without output", faux_assistant_message("", stop_reason="length")),
    ],
)
async def test_treats_silent_overflow_as_an_ordinary_answer(name: str, response: AssistantMessage) -> None:
    chat = await open_compacting(context_window=300)
    await history(chat)
    # Thresholds out of reach, so only overflow classification could compact.
    chat.setup.settings["compaction"] = {
        "enabled": True,
        "reserveTokens": -100_000,
        "keepRecentTokens": 150,
        "backgroundTokens": 0,
    }
    chat.faux.agent.append(response)
    input = await chat.root.submit({"type": "input", "content": "go"}, context)
    assert (await input.wait(context))["status"] == "done", name
    last = (await all_entries(chat.root))[-1]
    message = last["model"][0]
    assert isinstance(message, AssistantMessage)
    assert message.usage.input >= 300
    assert len(chat.faux.summary_requests) == 0
    await chat.harness.close(context)


async def test_sends_the_request_when_a_blocking_compaction_finds_nothing_under_a_policy_changed_after_preparation() -> (
    None
):
    chat = await open_compacting(context_window=1000)
    await history(chat)
    chat.setup.settings["compaction"] = BLOCKING
    changed = [False]

    async def policy(*_args: Any) -> str:
        # Rendering runs after preparation read the policy; the compaction reads this one.
        if not changed[0]:
            changed[0] = True
            chat.setup.settings["compaction"] = {**BLOCKING, "keepRecentTokens": 100_000}
        return "p"

    add_section(chat.setup.registry, "policy", policy)
    await turn(chat, text("u4", 200), "a4")
    assert await compaction_tasks(chat) == []
    assert len(chat.faux.summary_requests) == 0
    assert "pi.compaction" not in await kinds(chat.root)
    await chat.harness.close(context)


# ─── blocked compaction ───────────────────────────────────────────────────


async def test_survives_reopen_blocked_and_is_orphaned_on_abort_with_its_status_removed(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    setup = chat_setup()
    opened = await open_chat(await open_sqlite_storage(path), setup)
    harness, root = opened.harness, opened.root
    # A compaction stored by a newer version than this process registers, for example after a downgrade.
    newer: Any = Task(dataclasses.replace(CompactionTask.definition, version=2))

    async def create(tx: Any) -> Any:
        task_id = await tx.create_task(newer, {"reason": "manual"}, {"ownership": {"kind": "conversation"}})
        state = await tx.doc(LiveDoc, root.id)
        state["compactions"] = [{"taskId": task_id, "reason": "manual", "blocking": False, "attempt": 1}]
        return task_id

    id = await root.commit(create, context)
    await harness.close(context)
    opened = await open_chat(await open_sqlite_storage(path), setup)
    opened.harness.resume()
    chat = Chat(opened.harness, opened.root, setup, script(setup))
    stream, events = await listen(chat)
    inspection = await chat.harness.inspect(context)
    found = next(task for task in inspection["tasks"] if task["record"]["id"] == id)
    assert found["state"] == {"kind": "blocked", "reason": "task_too_old"}
    await chat.harness.abort_task(id, context)
    assert (await chat.harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "orphaned",
        "reason": "task_too_old",
    }
    assert "compactions" not in await live(chat)
    await wait_for(lambda: any(event["type"] == "compaction_end" for event in events))
    await stream.stop()
    await chat.harness.close(context)


# ─── context contributions ────────────────────────────────────────────────


async def test_context_contributions_apply_edits_carried_by_an_older_head_marker_in_the_range() -> None:
    chat = await open_compacting()

    async def write(tx: Any) -> Any:
        a = await tx.append_entry(chat.root.id, {"kind": "app.note", "model": [user("a")]})
        b = await tx.append_entry(chat.root.id, {"kind": "app.note", "model": [user("b")]})
        # An older marker that omits b, then a newer one whose range still contains the older marker.
        await tx.append_entry(
            chat.root.id, {"kind": "app.head", "head": a["id"], "edits": [{"target": b["id"], "action": "omit"}]}
        )
        c = await tx.append_entry(chat.root.id, {"kind": "app.note", "model": [user("c")]})
        await tx.append_entry(chat.root.id, {"kind": "app.head", "head": a["id"], "model": [user("H")]})
        return {"a": a["id"], "b": b["id"], "c": c["id"]}

    ids = await chat.root.commit(write, context)
    context_view = await chat.root.context(context)
    assert [record["id"] for record in context_view["entries"]][1:] == [ids["a"], ids["b"], ids["c"]]
    assert [[user_text(message) for message in messages] for messages in context_view["contributions"]] == [
        ["H"],
        ["a"],
        [],
        ["c"],
    ]
    assert [user_text(message) for message in context_view["messages"]] == ["H", "a", "c"]
    # The summarizer sees the same contributions: the omitted entry stays out.
    chat.setup.settings["compaction"] = {**MANUAL, "keepRecentTokens": 1}
    chat.faux.summaries.append(summary())
    await result(chat, await chat.root.compact(None, context))
    prompt = user_text(chat.faux.summary_requests[0].messages[1])
    assert "<conversation>\n[User]: H\n\n[User]: a\n</conversation>" in prompt
    await chat.harness.close(context)


# ─── per-model compaction overrides (#10942) ──────────────────────────────


async def test_applies_the_current_models_override_to_the_blocking_threshold() -> None:
    for key in ("faux/faux-1", "faux/other"):
        chat = await open_compacting(context_window=1000)
        await history(chat)
        # Harness-wide: blocking only above the full window; the override for faux-1 blocks above 700.
        chat.setup.settings["compaction"] = {
            **BLOCKING,
            "reserveTokens": 0,
            "modelOverrides": {key: {"reserveTokens": 300}},
        }
        chat.faux.summaries.append(summary())
        await turn(chat, text("u4", 200), "a4")
        assert len(chat.faux.summary_requests) == (1 if key == "faux/faux-1" else 0)
        await chat.harness.close(context)


async def test_applies_the_request_models_override_on_a_context_overflow() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.setup.settings["compaction"] = {**MANUAL, "modelOverrides": {"faux/faux-1": {"enabled": True}}}
    chat.faux.summaries.append(summary())
    chat.faux.agent.extend([failure(OVERFLOW), answer("fits")])
    input = await chat.root.submit({"type": "input", "content": text("u4", 100)}, context)
    assert (await input.wait(context))["status"] == "done"
    assert len(chat.faux.summary_requests) == 1
    await chat.harness.close(context)


async def test_applies_the_models_override_when_a_compaction_selects_and_sizes_its_summary() -> None:
    chat = await open_compacting()
    await history(chat)
    chat.setup.settings["compaction"] = {
        **MANUAL,
        "keepRecentTokens": 100_000,
        "modelOverrides": {"faux/faux-1": {"keepRecentTokens": 150, "reserveTokens": 500}},
    }
    chat.faux.summaries.append(summary())
    assert (await result(chat, await chat.root.compact(None, context)))["status"] == "completed"
    # 0.8 * 500 from the override, below the model's 900.
    assert chat.faux.summary_requests[0].options.get("max_tokens") == 400
    messages = (await chat.root.context(context))["messages"]
    assert [user_text(message) for message in messages[1:]] == [text("u3", 100), text("a3", 100)]
    await chat.harness.close(context)
