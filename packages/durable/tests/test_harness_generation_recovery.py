"""Port of ``test/harness-generation-recovery.test.ts``."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from durable_harness_support import (
    ChatSetup,
    aborted,
    add_section,
    all_entries,
    assert_match,
    chat_setup,
    open_chat,
    text_of,
    unanswered,
    wait_for,
)
from durable_session_support import context, deferred, resolve

from pi_ai import create_models, faux_assistant_message
from pi_ai.serialization import message_from_dict
from pi_ai.types import AssistantMessage
from pi_durable import AssistantEntry, Harness, LiveDoc
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


async def open_at(path: Path, setup: ChatSetup) -> Any:
    return await open_chat(await open_sqlite_storage(str(path)), setup)


async def run_task_id(harness: Harness) -> Any:
    live = await harness.snapshot(LiveDoc, 1, context)  # type: ignore[arg-type]
    assert live is not None
    return live["run"]["taskId"]  # type: ignore[typeddict-item]


async def checkpoint(harness: Harness, id: Any) -> Any:
    record = await harness.get_task(id, context)
    if record is None or record["state"]["status"] == "terminal":
        return None
    return record["state"]["checkpoint"]  # type: ignore[typeddict-item]


def with_models(setup: ChatSetup, models: Any) -> ChatSetup:
    """``{ ...setup, models }``: the same faux, registry, reports, settings and clock over other models."""
    copy = ChatSetup(setup.faux, models, setup.registry, setup.reports, setup.settings)
    copy.now = lambda: setup.now()
    return copy


def message_text(value: Any) -> str | None:
    """Text of a live document's stored (JSON) generation message."""
    generation = None if value is None else value.get("generation")
    message = None if generation is None else generation.get("message")
    return None if message is None else text_of(message_from_dict(message))


async def test_reruns_preparation_interrupted_before_its_commit(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    reached = deferred()
    block = [True]

    async def preamble(_input: Any, ctx: Any) -> str:
        if block[0]:
            block[0] = False
            resolve(reached)
            await aborted(ctx.abort_signal)
        return "p"

    add_section(setup.registry, "preamble", preamble)
    setup.faux.set_responses([faux_assistant_message("answer")])
    opened = await open_at(path, setup)
    opened.harness.resume()
    id = (await opened.root.submit({"type": "input", "content": "hi"}, context)).id
    await reached
    task_id = await run_task_id(opened.harness)
    await opened.harness.close(context)

    opened = await open_at(path, setup)
    assert await checkpoint(opened.harness, task_id) == {"phase": "prepare", "attempt": 1}
    opened.harness.resume()
    assert (await (await opened.harness.submission(id, context)).wait(context))["status"] == "done"  # type: ignore[union-attr]
    assert [entry["kind"] for entry in await all_entries(opened.root)] == ["pi.user", "pi.system", "pi.assistant"]
    await opened.harness.close(context)


async def test_resends_a_request_interrupted_before_any_partial_without_repeating_preparation(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    add_section(setup.registry, "preamble", lambda *_: "p", tag=False)
    reached = deferred()
    sent: list[list[str]] = []
    timeouts: list[Any] = []

    async def first(_context: Any, options: Any, *_rest: Any) -> Any:
        resolve(reached)
        return await aborted(options["signal"])

    def second(request: Any, options: Any, *_rest: Any) -> AssistantMessage:
        sent.append([message.role for message in request.messages])
        timeouts.append(None if options is None else options.get("timeout_ms"))
        return faux_assistant_message("answer")

    setup.faux.set_responses([first, second])
    opened = await open_at(path, setup)
    setup.settings["stream"] = {"timeoutMs": 1234}
    opened.harness.resume()
    id = (await opened.root.submit({"type": "input", "content": "hi"}, context)).id
    await reached
    task_id = await run_task_id(opened.harness)
    await opened.harness.close(context)

    opened = await open_at(path, setup)
    assert_match(
        await checkpoint(opened.harness, task_id),
        {"phase": "request", "attempt": 1, "thinkingLevel": "off", "streamOptions": {"timeoutMs": 1234}},
    )
    # The resend uses the pinned request, not options changed after preparation.
    setup.settings["stream"] = {"timeoutMs": 999}
    assert_match(await opened.harness.snapshot(LiveDoc, opened.root.id, context), {"generation": {"attempt": 1}})
    opened.harness.resume()
    assert (await (await opened.harness.submission(id, context)).wait(context))["status"] == "done"  # type: ignore[union-attr]
    assert sent == [["system", "user"]]
    assert timeouts == [1234]
    assert [entry["kind"] for entry in await all_entries(opened.root)] == ["pi.user", "pi.system", "pi.assistant"]
    await opened.harness.close(context)


async def test_converts_a_committed_partial_into_an_aborted_entry_and_resends_the_same_messages(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    slow = chat_setup(tokens_per_second=20, token_size={"min": 1, "max": 1})
    slow.faux.set_responses([faux_assistant_message("z" * 400)])
    opened = await open_at(path, slow)
    opened.harness.resume()
    watch = await opened.harness.watch_doc(LiveDoc, opened.root.id, context)
    assert watch is not None
    watched: list[str | None] = [None]

    async def listener(value: Any, *_rest: Any) -> None:
        text = message_text(value)
        if text is not None:
            watched[0] = text

    watch.start(listener)
    id = (await opened.root.submit({"type": "input", "content": "hi"}, context)).id
    await wait_for(lambda: watched[0] is not None)
    await opened.harness.close(context)

    setup = chat_setup()
    sent: list[list[str]] = []

    def step(request: Any, *_rest: Any) -> AssistantMessage:
        sent.append([message.role for message in request.messages])
        return faux_assistant_message("answer")

    setup.faux.set_responses([step])
    opened = await open_at(path, setup)
    stored = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
    partial = message_text(stored)
    assert partial is not None and watched[0] is not None
    # Everything observers saw before the crash is durable.
    assert partial.startswith(watched[0])
    opened.harness.resume()
    assert (await (await opened.harness.submission(id, context)).wait(context))["status"] == "done"  # type: ignore[union-attr]
    assert sent == [["user"]]
    entries = await all_entries(opened.root)
    assert [entry["kind"] for entry in entries] == ["pi.user", "pi.assistant", "pi.assistant"]
    converted = entries[1]["model"][0]
    assert isinstance(converted, AssistantMessage)
    assert converted.stop_reason == "aborted"
    assert text_of(converted) == partial
    assert await opened.harness.snapshot(LiveDoc, opened.root.id, context) == {}
    await opened.harness.close(context)


async def test_resumes_a_retry_backoff_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    now = [1_000]
    setup.now = lambda: now[0]
    setup.faux.set_responses(
        [
            faux_assistant_message([], stop_reason="error", error_message="503 Service Unavailable"),
            faux_assistant_message("recovered"),
        ]
    )
    opened = await open_at(path, setup)
    opened.harness.resume()
    setup.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 60_000}
    id = (await opened.root.submit({"type": "input", "content": "hi"}, context)).id

    async def retrying() -> bool:
        live = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
        generation = None if live is None else live.get("generation")
        return generation is not None and "retry" in generation

    await wait_for(retrying)
    task_id = await run_task_id(opened.harness)
    await opened.harness.close(context)

    opened = await open_at(path, setup)
    record = await opened.harness.get_task(task_id, context)
    assert record is not None
    assert record["state"] == {
        "status": "sleeping",
        "checkpoint": {"phase": "retry", "attempt": 1, "until": 61_000},
        "until": 61_000,
    }
    await wait_for(lambda: not opened.harness.wake()["active"])
    assert opened.harness.wake() == {"active": False, "at": 61_000}
    # An early wake retries nothing.
    opened.harness.resume()
    await asyncio.sleep(0.02)
    assert setup.faux.state.call_count == 1
    assert await opened.harness.snapshot(LiveDoc, opened.root.id, context) == {
        "run": {"taskId": task_id, "inputs": [id]},
        "generation": {"attempt": 1, "retry": {"at": 61_000, "error": "503 Service Unavailable"}},
    }
    now[0] = 61_000
    opened.harness.resume()
    assert (await (await opened.harness.submission(id, context)).wait(context))["status"] == "done"  # type: ignore[union-attr]
    assert [entry["kind"] for entry in await all_entries(opened.root)] == ["pi.user", "pi.assistant", "pi.assistant"]
    await opened.harness.close(context)


async def test_sleeps_again_for_a_retry_stored_as_a_running_phase_by_an_earlier_version(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    now = [1_000]
    setup.now = lambda: now[0]
    setup.faux.set_responses(
        [
            faux_assistant_message([], stop_reason="error", error_message="503 Service Unavailable"),
            faux_assistant_message("recovered"),
        ]
    )
    opened = await open_at(path, setup)
    opened.harness.resume()
    setup.settings["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 60_000}
    id = (await opened.root.submit({"type": "input", "content": "hi"}, context)).id

    async def sleeping() -> bool:
        record = await opened.harness.get_task(await run_task_id(opened.harness), context)
        return record is not None and record["state"]["status"] == "sleeping"

    await wait_for(sleeping)
    task_id = await run_task_id(opened.harness)
    await opened.harness.close(context)
    # Earlier versions slept inside a running `retry` phase.
    database = sqlite3.connect(path)
    database.execute(
        "UPDATE tasks SET status = 'running', record = json_set(json_remove(record, '$.state.until'), "
        "'$.state.status', 'running') WHERE id = ?",
        (task_id,),
    )
    database.commit()
    database.close()

    opened = await open_at(path, setup)
    opened.harness.resume()

    async def asleep() -> bool:
        record = await opened.harness.get_task(task_id, context)
        return record is not None and record["state"]["status"] == "sleeping"

    await wait_for(asleep)
    record = await opened.harness.get_task(task_id, context)
    assert record is not None
    assert_match(record["state"], {"status": "sleeping", "until": 61_000})
    assert setup.faux.state.call_count == 1
    now[0] = 61_000
    opened.harness.resume()
    assert (await (await opened.harness.submission(id, context)).wait(context))["status"] == "done"  # type: ignore[union-attr]
    assert setup.faux.state.call_count == 2
    await opened.harness.close(context)


async def test_resumes_polling_a_deferred_response_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup(deferred={"poll_after_ms": 60_000})
    now = [1_000]
    setup.now = lambda: now[0]
    setup.faux.set_responses([faux_assistant_message("deferred answer")])
    opened = await open_at(path, setup)
    opened.harness.resume()
    setup.settings["stream"] = {"deferred": True}
    id = (await opened.root.submit({"type": "input", "content": "hi"}, context)).id

    async def deferring() -> bool:
        live = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
        generation = None if live is None else live.get("generation")
        return generation is not None and "deferred" in generation

    await wait_for(deferring)
    task_id = await run_task_id(opened.harness)
    await opened.harness.close(context)

    opened = await open_at(path, setup)
    assert_match(await checkpoint(opened.harness, task_id), {"phase": "poll", "attempt": 1, "pollAt": 61_000})
    record = await opened.harness.get_task(task_id, context)
    assert record is not None
    assert_match(record["state"], {"status": "sleeping", "until": 61_000})
    await wait_for(lambda: not opened.harness.wake()["active"])
    assert opened.harness.wake() == {"active": False, "at": 61_000}
    now[0] = 61_000
    opened.harness.resume()
    settled = await (await opened.harness.submission(id, context)).wait(context)  # type: ignore[union-attr]
    assert settled["status"] == "done" and settled["type"] == "input", f"Unexpected {settled['status']}"
    answer = await opened.root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)
    assert text_of(answer["model"][0]) == "deferred answer"
    assert setup.faux.state.deferred_fetch_count == 1
    await opened.harness.close(context)


async def test_fails_no_model_when_the_pinned_model_is_gone_after_reopen_in_request_and_in_poll(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup(deferred={"poll_after_ms": 60_000})
    now = [1_000]
    setup.now = lambda: now[0]
    busy = unanswered()
    setup.faux.set_responses([busy.step, faux_assistant_message("deferred")])
    opened = await open_at(path, setup)
    opened.harness.resume()
    requesting = (await opened.root.submit({"type": "input", "content": "one"}, context)).id
    await busy.reached
    await opened.harness.close(context)

    # Reopened without the faux provider: the request's pinned model is unknown.
    empty = with_models(setup, create_models())
    opened = await open_at(path, empty)
    opened.harness.resume()
    assert_match(
        await (await opened.harness.submission(requesting, context)).wait(context),  # type: ignore[union-attr]
        {"status": "unanswered", "reason": "no_model"},
    )
    await opened.harness.close(context)

    opened = await open_at(path, setup)
    setup.settings["stream"] = {"deferred": True}
    opened.harness.resume()
    polling = (await opened.root.submit({"type": "input", "content": "two"}, context)).id

    async def deferring() -> bool:
        live = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
        generation = None if live is None else live.get("generation")
        return generation is not None and "deferred" in generation

    await wait_for(deferring)
    await opened.harness.close(context)

    # The poll sleeps until it is due; then the model is resolved.
    now[0] = 61_000
    opened = await open_at(path, empty)
    opened.harness.resume()
    assert_match(
        await (await opened.harness.submission(polling, context)).wait(context),  # type: ignore[union-attr]
        {"status": "unanswered", "reason": "no_model"},
    )
    await opened.harness.close(context)


async def test_runs_a_print_style_turn_and_reads_the_durable_answer_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    add_section(setup.registry, "preamble", lambda *_: "You are terse.", tag=False)
    setup.faux.set_responses([faux_assistant_message("42")])
    opened = await open_at(path, setup)
    opened.harness.resume()
    submission = await opened.root.submit({"type": "input", "content": "answer?"}, context)
    settled = await submission.wait(context)
    assert settled["status"] == "done" and settled["type"] == "input", f"Unexpected {settled['status']}"
    answer = await opened.root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)
    assert text_of(answer["model"][0]) == "42"
    await opened.harness.close(context)

    opened = await open_at(path, setup)
    assert await (await opened.harness.submission(submission.id, context)).status(context) == settled  # type: ignore[union-attr]
    assert await opened.root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context) == answer
    await opened.harness.close(context)
