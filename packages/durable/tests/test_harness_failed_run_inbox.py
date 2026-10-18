"""Port of ``test/harness-failed-run-inbox.test.ts``."""

from __future__ import annotations

import asyncio
from typing import Any, NamedTuple

from durable_harness_support import chat_setup, open_chat, text_of
from durable_session_support import context, deferred, resolve

from pi_ai import faux_assistant_message
from pi_ai.types import AssistantMessage, UserMessage
from pi_durable import AssistantEntry, Harness, MemoryStorage


class FailingOnce(NamedTuple):
    step: Any
    gate: asyncio.Future[Any]
    reached: asyncio.Future[Any]


def failing_once() -> FailingOnce:
    """A response that fails for good once ``gate`` opens; ``reached`` resolves when it was requested."""
    gate = deferred()
    reached = deferred()

    async def step(*_args: Any) -> AssistantMessage:
        resolve(reached)
        await gate
        return faux_assistant_message([], stop_reason="error", error_message="Invalid request")

    return FailingOnce(step, gate, reached)


class Failed(NamedTuple):
    harness: Harness
    root: Any
    second: Any


async def fail_with_queued_input(policy: str | None) -> Failed:
    """A run that fails with an input queued behind it, under ``policy``; returns how the queued input settled."""
    setup = chat_setup()
    if policy is not None:
        setup.settings["afterFailedRun"] = policy
    failing = failing_once()
    setup.faux.set_responses(
        [failing.step, faux_assistant_message("second answer"), faux_assistant_message("third answer")]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    first = await root.submit({"type": "input", "content": "Analyze this report"}, context)
    await failing.reached
    second = await root.submit({"type": "input", "content": "Email me the summary", "whenBusy": "followUp"}, context)
    resolve(failing.gate)
    settled = await first.wait(context)
    assert settled["status"] == "unanswered" and settled["reason"] == "model_error"
    return Failed(harness, root, second)


async def test_stay_queued_by_default() -> None:
    harness, root, second = await fail_with_queued_input(None)
    await harness.wait_for_idle(context)
    assert (await second.status(context))["status"] == "queued"
    # The next input runs the queue, as before.
    third = await root.submit({"type": "input", "content": "Anything else?"}, context)
    assert (await second.wait(context))["status"] == "done"
    assert (await third.wait(context))["status"] == "done"
    await harness.close(context)


async def test_start_the_next_run_with_continue() -> None:
    harness, root, second = await fail_with_queued_input("continue")
    settled = await second.wait(context)
    assert settled["status"] == "done" and settled["type"] == "input", f"Unexpected {settled['status']}"
    answer = await root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)
    assert text_of(answer["model"][0]) == "second answer"
    await harness.close(context)


async def test_place_queued_writes_and_keep_them_with_withdraw() -> None:
    for policy in ("continue", "withdraw"):
        setup = chat_setup()
        setup.settings["afterFailedRun"] = policy
        failing = failing_once()
        setup.faux.set_responses([failing.step, faux_assistant_message("second answer")])
        opened = await open_chat(MemoryStorage(), setup)
        harness, root = opened.harness, opened.root
        first = await root.submit({"type": "input", "content": "first"}, context)
        await failing.reached
        note = UserMessage("a note the model sees", timestamp=1)
        write = await root.submit({"type": "write", "entry": {"kind": "test.note", "model": [note]}}, context)
        assert (await write.status(context))["status"] == "queued"
        resolve(failing.gate)
        await first.wait(context)
        await harness.wait_for_idle(context)
        # `continue` places the write at the failed run's boundary; `withdraw` withdraws inputs only.
        assert (await write.status(context))["status"] == ("done" if policy == "continue" else "queued")
        await harness.close(context)


async def test_settle_unanswered_with_run_failed_with_withdraw() -> None:
    harness, _root, second = await fail_with_queued_input("withdraw")
    settled = await second.wait(context)
    assert settled["status"] == "unanswered" and settled["reason"] == "run_failed"
    await harness.close(context)
