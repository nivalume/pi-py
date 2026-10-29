"""Port of ``test/harness-generation.test.ts``."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import re
from collections.abc import AsyncIterator
from typing import Any

import pytest
from durable_harness_support import (
    ChatSetup,
    add_section,
    all_entries,
    chat_setup,
    open_chat,
    text_of,
    unanswered,
    wait_for,
)
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context

from pi_ai import AbortSignal, faux_assistant_message
from pi_ai.serialization import message_from_dict
from pi_ai.types import AssistantMessage, DeferredHandle, SystemMessage, Usage, UserMessage
from pi_durable import (
    AssistantEntry,
    GenerationTask,
    Harness,
    LiveDoc,
    MemoryStorage,
    ProviderDoc,
    Task,
    UserEntry,
    create_registry,
    define_doc,
    define_extension,
    wrap_section,
)
from pi_durable.harness.agent import resolve_settings

ERROR_503 = faux_assistant_message([], stop_reason="error", error_message="503 Service Unavailable")
UUID_V7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


async def live(harness: Harness, conversation: Any) -> Any:
    return await harness.snapshot(LiveDoc, conversation.id, context)


async def run_task(harness: Harness, conversation: Any) -> Any:
    found: list[Any] = []

    async def check() -> bool:
        state = await live(harness, conversation)
        run = None if state is None else state.get("run")
        if run is None:
            return False
        found.append(run["taskId"])
        return True

    await wait_for(check)
    return found[-1]


class _WithStream:
    """``models`` with ``stream_simple`` replaced (upstream's Proxy)."""

    def __init__(self, models: Any, stream_simple: Any) -> None:
        self._models = models
        self.stream_simple = stream_simple

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _ScriptedStream:
    """A stream that yields ``events`` with ``pause`` seconds after the last one, then ends with ``final``."""

    def __init__(self, events: list[Any], final: Any, pause: float) -> None:
        self._events = events
        self._final = final
        self._pause = pause

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event
        await asyncio.sleep(self._pause)

    async def result(self) -> Any:
        return self._final


@dataclasses.dataclass
class _Start:
    partial: Any
    type: str = "start"


def _stream_of(partial: Any, final: Any) -> Any:
    def stream(*_args: Any, **_kwargs: Any) -> Any:
        return _ScriptedStream([_Start(partial)], final, 0.3)

    return stream


def invalid_final_stream() -> Any:
    """A stream that commits one partial and then ends with a final message that is not strict JSON, so the
    classification commit raises and the scheduler faults the task."""
    final = faux_assistant_message("final")
    final.usage = Usage(input=lambda: None)  # type: ignore[arg-type]
    return _stream_of(faux_assistant_message("partial", stop_reason="pending"), final)  # type: ignore[arg-type]


def with_models(base: ChatSetup, models: Any) -> ChatSetup:
    return ChatSetup(base.faux, models, base.registry, base.reports, base.settings, base.now)


def live_publications(harness: Harness) -> list[Any]:
    values: list[Any] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in publication["changes"]:
            if change["type"] == "document" and change["record"]["kind"] == "pi.live" and change["value"] is not None:
                values.append(change["value"])

    harness.subscribe_commits(listener)
    return values


def _partial_text(value: Any) -> str | None:
    generation = value.get("generation") if value is not None else None
    message = None if generation is None else generation.get("message")
    return None if message is None else text_of(message_from_dict(message))


async def _tasks(harness: Harness, conversation: Any) -> list[Any]:
    return (await harness.commit(lambda tx: tx.scan_tasks({"conversationId": conversation.id}, 10), context))["items"]


async def test_answers_an_input_and_settles_its_submission() -> None:
    setup = chat_setup()
    add_section(setup.registry, "preamble", lambda *_: "You are helpful.", tag=False)
    setup.faux.set_responses([faux_assistant_message("Hello there")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    settled = await submission.wait(context)
    assert settled["status"] == "done" and settled["type"] == "input"

    answer = await root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)  # type: ignore[typeddict-item]
    assert text_of(answer["model"][0]) == "Hello there"
    entries = await all_entries(root)
    assert [entry["kind"] for entry in entries] == ["pi.user", "pi.system", "pi.assistant"]
    assert entries[0]["id"] == settled["entry"]  # type: ignore[typeddict-item]
    [system] = entries[1]["model"]
    assert isinstance(system, SystemMessage)
    assert system.content == "" and system.sections == {"preamble": "You are helpful."}
    assert system.tools_added is None and system.tools_removed is None
    assert await live(harness, root) == {}
    task = (await _tasks(harness, root))[0]
    assert task["kind"] == "pi.generation"
    # Entries written by the generation are attributed to it; the admitted user entry is not task work.
    assert [entry.get("byTaskId") for entry in entries] == [None, task["id"], task["id"]]
    assert task["state"] == {
        "status": "terminal",
        "outcome": {"status": "completed", "result": {"entryId": settled["answer"]}},  # type: ignore[typeddict-item]
    }
    await harness.close(context)


async def test_stores_partials_as_deltas_and_a_complete_base_once_nothing_is_in_flight() -> None:
    setup = chat_setup(tokens_per_second=200, token_size={"min": 1, "max": 1})
    setup.faux.set_responses([faux_assistant_message("w" * 200)])
    storage = ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])
    opened = await open_chat(storage, setup)
    harness, root = opened.harness, opened.root
    record = await storage.find_document(
        {"kind": "pi.live", "scope": {"kind": "conversation", "conversationId": root.id}}, "current", context
    )
    assert record is not None
    harness.resume()
    await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    contents = [
        write["content"]["kind"]
        for writes in storage.commits
        for write in writes
        if write["type"] == "document.change" and write["id"] == record["id"]
    ]
    # Streaming writes deltas; the commit that settles the answer clears generation and writes a base.
    assert "delta" in contents
    assert contents[-1] == "base"
    await harness.close(context)


async def test_still_ends_a_run_whose_input_something_else_already_settled() -> None:
    setup = chat_setup()
    busy = unanswered()
    setup.faux.set_responses([busy.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    await busy.reached
    await root.commit(
        lambda tx: tx.settle_submission(submission.id, {"status": "unanswered", "reason": "withdrawn"}), context
    )
    task_id = await run_task(harness, root)
    await harness.abort_task(task_id, context)
    assert (await harness.wait_for_task(task_id, context))["state"]["outcome"] == {"status": "aborted"}
    # The earlier settlement stays; the run's own settlement leaves it unchanged.
    status = await submission.status(context)
    assert status["status"] == "unanswered" and status["reason"] == "withdrawn"  # type: ignore[typeddict-item]
    assert await live(harness, root) == {}
    await harness.close(context)


async def test_fails_with_no_model_when_no_model_is_configured_or_the_model_is_unknown() -> None:
    setup = chat_setup()
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    plain = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    harness.resume()
    unset = await (await plain.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert unset["status"] == "unanswered" and unset["reason"] == "no_model"  # type: ignore[typeddict-item]

    await root.configure({"model": {"provider": "faux", "modelId": "missing"}}, context)
    unknown = await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert unknown["status"] == "unanswered" and unknown["reason"] == "no_model"  # type: ignore[typeddict-item]
    assert isinstance(unknown["entry"], int)  # type: ignore[typeddict-item]
    assert [entry["kind"] for entry in await all_entries(root)] == ["pi.user"]
    assert (await _tasks(harness, root))[0]["state"] == {
        "status": "terminal",
        "outcome": {
            "status": "failed",
            "error": {"message": "Model faux/missing is not available", "detail": {"reason": "no_model"}},
        },
    }
    assert await live(harness, root) == {}
    assert await live(harness, plain) == {}
    await harness.close(context)


async def test_retries_a_retryable_error_after_a_durable_backoff_and_then_answers() -> None:
    setup = chat_setup()
    add_section(setup.registry, "preamble", lambda *_: "p", tag=False)
    setup.faux.set_responses([ERROR_503, faux_assistant_message("recovered")])
    setup.settings["retry"] = {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    values = live_publications(harness)
    harness.resume()
    settled = await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert settled["status"] == "done"
    entries = await all_entries(root)
    assert [entry["kind"] for entry in entries] == ["pi.user", "pi.system", "pi.assistant", "pi.assistant"]
    assert entries[2]["model"][0].stop_reason == "error"
    assert any(
        value.get("generation", {}).get("retry", {}).get("error") == "503 Service Unavailable" for value in values
    )
    assert any(value.get("generation", {}).get("attempt") == 2 for value in values)
    assert await live(harness, root) == {}
    await harness.close(context)


async def test_fails_with_model_error_once_retries_are_exhausted() -> None:
    setup = chat_setup()
    setup.faux.set_responses([ERROR_503, ERROR_503, faux_assistant_message("never")])
    setup.settings["retry"] = {"enabled": True, "maxRetries": 1, "baseDelayMs": 1}
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    settled = await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert settled["status"] == "unanswered"
    assert settled["reason"] == "model_error" and settled["detail"] == "503 Service Unavailable"  # type: ignore[typeddict-item]
    assert [entry["kind"] for entry in await all_entries(root)] == ["pi.user", "pi.assistant", "pi.assistant"]
    assert setup.faux.get_pending_response_count() == 1
    await harness.close(context)


async def test_fails_a_retryable_error_without_retrying_when_the_retry_policy_is_disabled() -> None:
    setup = chat_setup()
    setup.faux.set_responses([ERROR_503, faux_assistant_message("never")])
    setup.settings["retry"] = {"enabled": False}
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    settled = await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert settled["status"] == "unanswered" and settled["reason"] == "model_error"  # type: ignore[typeddict-item]
    assert setup.faux.state.call_count == 1
    await harness.close(context)


async def test_reports_section_wrapper_failures_while_preparing() -> None:
    setup = chat_setup()
    add_section(setup.registry, "cwd", lambda *_: "/repo")

    def fail(_inner: Any) -> Any:
        raise RuntimeError("wrapper failed")

    setup.registry.install(define_extension("broken", wraps=[wrap_section("cwd", fail)]))
    setup.faux.set_responses([faux_assistant_message("ok")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    assert (await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context))["status"] == "done"
    assert "wrapper failed" in [str(error) for error in setup.reports]
    # The failed section is absent, so nothing was rendered.
    assert [entry["kind"] for entry in await all_entries(root)] == ["pi.user", "pi.assistant"]
    await harness.close(context)


async def test_fails_a_non_retryable_error_without_retrying() -> None:
    setup = chat_setup()
    setup.faux.set_responses(
        [
            faux_assistant_message([], stop_reason="error", error_message="Invalid request"),
            faux_assistant_message("never"),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    settled = await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert settled["status"] == "unanswered"
    assert settled["reason"] == "model_error" and settled["detail"] == "Invalid request"  # type: ignore[typeddict-item]
    assert (await _tasks(harness, root))[0]["state"]["outcome"] == {
        "status": "failed",
        "error": {"message": "Invalid request", "detail": {"reason": "model_error"}},
    }
    await harness.close(context)


async def test_polls_a_deferred_response_until_it_is_ready() -> None:
    setup = chat_setup(deferred={"pending_fetches": 1, "poll_after_ms": 1})
    setup.faux.set_responses([faux_assistant_message("deferred answer")])
    setup.settings["stream"] = {"deferred": True}
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    values = live_publications(harness)
    harness.resume()
    settled = await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert settled["status"] == "done" and settled["type"] == "input"
    assert setup.faux.state.deferred_fetch_count == 2
    poll_times = [
        value["generation"]["deferred"]["pollAt"] for value in values if "deferred" in value.get("generation", {})
    ]
    assert len(poll_times) == 2
    assert poll_times[1] > poll_times[0]
    answer = await root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)  # type: ignore[typeddict-item]
    assert text_of(answer["model"][0]) == "deferred answer"
    await harness.close(context)


async def test_converts_the_committed_partial_when_aborted_during_streaming() -> None:
    setup = chat_setup(tokens_per_second=20, token_size={"min": 1, "max": 1})
    setup.faux.set_responses([faux_assistant_message("x" * 400)])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    task_id = await run_task(harness, root)

    async def has_partial() -> bool:
        return _partial_text(await live(harness, root)) is not None

    await wait_for(has_partial)
    partial = _partial_text(await live(harness, root))
    assert partial is not None
    assert await harness.abort_task(task_id, context) == "marked"
    settled = await submission.wait(context)
    assert settled["status"] == "unanswered" and settled["reason"] == "aborted"  # type: ignore[typeddict-item]
    entries = await all_entries(root)
    assert [entry["kind"] for entry in entries] == ["pi.user", "pi.assistant"]
    converted = entries[1]["model"][0]
    assert converted.stop_reason == "aborted"
    assert text_of(converted).startswith(partial)  # type: ignore[union-attr]
    assert await live(harness, root) == {}
    assert (await harness.wait_for_task(task_id, context))["state"]["outcome"] == {"status": "aborted"}
    await harness.close(context)


async def test_cancels_a_deferred_response_when_aborted_during_polling() -> None:
    setup = chat_setup(deferred={"pending_fetches": 100, "poll_after_ms": 60_000})
    setup.faux.set_responses([faux_assistant_message("never")])
    setup.settings["stream"] = {"deferred": True}
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    task_id = await run_task(harness, root)

    async def polling() -> bool:
        return "deferred" in ((await live(harness, root)) or {}).get("generation", {})

    await wait_for(polling)
    await harness.abort_task(task_id, context)
    settled = await submission.wait(context)
    assert settled["status"] == "unanswered" and settled["reason"] == "aborted"  # type: ignore[typeddict-item]
    assert len(setup.faux.state.cancelled_deferred) == 1
    assert await live(harness, root) == {}
    await harness.close(context)


async def test_reports_a_failed_deferred_cancellation_and_still_ends_the_run_aborted() -> None:
    base = chat_setup(deferred={"pending_fetches": 100, "poll_after_ms": 60_000})
    base.faux.set_responses([faux_assistant_message("never")])

    class FailingCancel:
        def __init__(self, models: Any) -> None:
            self._models = models

        def __getattr__(self, name: str) -> Any:
            return getattr(self._models, name)

        async def cancel_deferred(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("cancel failed")

    setup = with_models(base, FailingCancel(base.models))
    setup.settings["stream"] = {"deferred": True}
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    harness.resume()
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    task_id = await run_task(harness, root)

    async def polling() -> bool:
        return "deferred" in ((await live(harness, root)) or {}).get("generation", {})

    await wait_for(polling)
    await harness.abort_task(task_id, context)
    settled = await submission.wait(context)
    assert settled["status"] == "unanswered" and settled["reason"] == "aborted"  # type: ignore[typeddict-item]
    assert "cancel failed" in [str(error) for error in setup.reports]
    assert await live(harness, root) == {}
    await harness.close(context)


async def test_forwards_stream_options_and_the_thinking_level() -> None:
    setup = chat_setup()
    seen: list[Any] = []

    def capture(text: str) -> Any:
        def step(_context: Any, options: Any, *_rest: Any) -> Any:
            seen.append(options)
            return faux_assistant_message(text)

        return step

    setup.faux.set_responses([capture("a"), capture("b")])
    setup.settings["stream"] = {"timeoutMs": 1234, "headers": {"x-test": "1"}}
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.configure({"thinkingLevel": "high"}, context)
    harness.resume()
    await (await root.submit({"type": "input", "content": "one"}, context)).wait(context)
    # Both are read at the next preparation: the thinking level from pi.agent, the stream options live from settings.
    await root.configure({"thinkingLevel": None}, context)
    setup.settings["stream"] = {"timeoutMs": 99}
    await (await root.submit({"type": "input", "content": "two"}, context)).wait(context)
    session_id = (await harness.snapshot(ProviderDoc, root.id, context))["sessionId"]  # type: ignore[index]
    assert seen[0]["timeout_ms"] == 1234
    assert seen[0]["headers"] == {"x-test": "1"}
    assert seen[0]["reasoning"] == "high"
    assert seen[0]["session_id"] == session_id
    assert isinstance(seen[0]["signal"], AbortSignal)
    assert "reasoning" not in seen[1]
    assert seen[1]["timeout_ms"] == 99 and seen[1]["session_id"] == session_id
    assert "headers" not in seen[1]
    await harness.close(context)


async def test_keeps_provider_session_ids_request_local_across_concurrent_conversations() -> None:
    """Regression coverage for #10424."""
    setup = chat_setup()
    seen: dict[str, list[str]] = {}

    def capture(request: Any, options: Any, *_rest: Any) -> Any:
        message = next((candidate for candidate in reversed(request.messages) if candidate.role == "user"), None)
        text = message.content if message is not None and isinstance(message.content, str) else ""
        seen.setdefault(text, []).append((options or {}).get("session_id", ""))
        return faux_assistant_message(f"answer:{text}")

    setup.faux.set_responses([capture, capture, capture, capture])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    child = await harness.create_conversation(
        {"ownership": {"kind": "ownerless"}, "agent": {"model": {"provider": "faux", "modelId": "faux-1"}}}, context
    )
    harness.resume()

    async def ask(conversation: Any, text: str) -> None:
        await (await conversation.submit({"type": "input", "content": text}, context)).wait(context)

    await asyncio.gather(*(ask(conversation, f"first-{index}") for index, conversation in enumerate([root, child])))
    await asyncio.gather(*(ask(conversation, f"second-{index}") for index, conversation in enumerate([root, child])))
    root_id = (await harness.snapshot(ProviderDoc, root.id, context))["sessionId"]  # type: ignore[index]
    child_id = (await harness.snapshot(ProviderDoc, child.id, context))["sessionId"]  # type: ignore[index]
    assert root_id != child_id
    assert seen["first-0"] == [root_id]
    assert seen["second-0"] == [root_id]
    assert seen["first-1"] == [child_id]
    assert seen["second-1"] == [child_id]
    await harness.close(context)


async def test_creates_and_persists_provider_state_before_a_legacy_conversations_request() -> None:
    """Regression coverage for #10424."""
    setup = chat_setup()
    sent: list[Any] = []

    def step(_request: Any, options: Any, *_rest: Any) -> Any:
        sent.append((options or {}).get("session_id"))
        return faux_assistant_message("ok")

    setup.faux.set_responses([step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.commit(lambda tx: tx.retire_doc(ProviderDoc, root.id), context)
    assert await harness.snapshot(ProviderDoc, root.id, context) is None
    harness.resume()
    await (await root.submit({"type": "input", "content": "legacy"}, context)).wait(context)
    stored = await harness.snapshot(ProviderDoc, root.id, context)
    assert stored is not None and UUID_V7.match(stored["sessionId"])
    assert sent == [stored["sessionId"]]
    await harness.close(context)


async def test_reads_settings_through_getters_at_every_decision() -> None:
    setup = chat_setup()
    timeout_ms = [111]
    seen: list[Any] = []

    def first(_context: Any, options: Any, *_rest: Any) -> Any:
        seen.append(options.get("timeout_ms"))
        # The user changes the setting while the first attempt runs.
        timeout_ms[0] = 222
        return ERROR_503

    def second(_context: Any, options: Any, *_rest: Any) -> Any:
        seen.append(options.get("timeout_ms"))
        return faux_assistant_message("ok")

    setup.faux.set_responses([first, second])
    harness = await Harness.open(
        MemoryStorage(),
        {
            "models": setup.models,
            "registry": setup.registry,
            "settings": lambda: {"stream": {"timeoutMs": timeout_ms[0]}, "retry": {"baseDelayMs": 1}},
        },
        context,
    )
    root = await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})
    harness.resume()
    assert (await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context))["status"] == "done"
    # The retry prepares again, so it resolves the settings again and sends the new timeout.
    assert seen == [111, 222]
    await harness.close(context)


def test_resolves_settings_over_the_built_in_defaults() -> None:
    resolved = resolve_settings(None)
    assert isinstance(resolved["documentRetentionMs"], int | float)
    assert {key: value for key, value in resolved.items() if key != "documentRetentionMs"} == {
        "stream": {},
        "retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 2000, "maxAgentDelayMs": 60000},
        "compaction": {"enabled": True, "reserveTokens": 16384, "keepRecentTokens": 20000, "backgroundTokens": 32768},
        "progress": {"partialIntervalMs": 100, "outputIntervalMs": 100},
        "toolExecution": "parallel",
        "steeringMode": "one-at-a-time",
        "followUpMode": "one-at-a-time",
        "contextRetentionMs": 600_000,
        "afterFailedRun": "keep",
        "restartLimit": 5,
        "approvalTimeoutMs": 900_000,
        "maxContinuations": 8,
    }
    assert resolve_settings({"restartLimit": 0})["restartLimit"] == 5
    assert resolve_settings({"restartLimit": math.inf})["restartLimit"] == math.inf
    edited = resolve_settings({"retry": {"enabled": False}, "compaction": {"backgroundTokens": 0}})
    assert edited["retry"]["enabled"] is False and edited["retry"]["maxRetries"] == 3
    assert edited["retry"]["baseDelayMs"] == 2000
    assert edited["compaction"]["enabled"] is True and edited["compaction"]["backgroundTokens"] == 0
    assert resolve_settings({"progress": {"outputIntervalMs": 500}})["progress"] == {
        "partialIntervalMs": 100,
        "outputIntervalMs": 500,
    }


@pytest.mark.parametrize(("settings", "expected"), [({}, True), ({"progress": {"partialIntervalMs": 5000}}, False)])
async def test_commits_partials_no_more_often_than_progress_partial_interval_ms(settings: Any, expected: bool) -> None:
    base = chat_setup()
    stream = _stream_of(faux_assistant_message("partial", stop_reason="pending"), faux_assistant_message("final"))  # type: ignore[arg-type]
    setup = with_models(base, _WithStream(base.models, stream))
    setup.settings.update(settings)
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    values = live_publications(harness)
    harness.resume()
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    assert (await submission.wait(context))["status"] == "done"
    await harness.close(context)
    assert any(_partial_text(value) == "partial" for value in values) is expected


async def test_renders_sections_that_read_conversation_documents_through_input_read() -> None:
    AgentNote = define_doc(
        kind="test.agent",
        version=1,
        scope="conversation",
        history="latest",
        fork="current",
        initial=lambda: {"cwd": "/", "kind": "main"},
    )
    setup = chat_setup()

    async def cwd(prompt_input: Any, ctx: Any) -> Any:
        state = await prompt_input["read"].snapshot(AgentNote, prompt_input["conversationId"], ctx)
        return None if state is None else state["cwd"]

    async def agents(prompt_input: Any, ctx: Any) -> Any:
        state = await prompt_input["read"].snapshot(AgentNote, prompt_input["conversationId"], ctx)
        return None if state is not None and state["kind"] == "sub" else "Read AGENTS.md"

    add_section(setup.registry, "cwd", cwd)
    add_section(setup.registry, "agents", agents)
    setup.faux.set_responses([faux_assistant_message("a"), faux_assistant_message("b")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root

    async def set_cwd(tx: Any) -> None:
        (await tx.doc(AgentNote, root.id))["cwd"] = "/repo"

    await root.commit(set_cwd, context)

    async def init(tx: Any, id: Any) -> None:
        agent = await tx.doc(AgentNote, id)
        agent["kind"] = "sub"
        agent["cwd"] = "/sub"

    sub = await harness.create_conversation(
        {
            "ownership": {"kind": "ownerless"},
            "agent": {"model": {"provider": "faux", "modelId": "faux-1"}},
            "init": init,
        },
        context,
    )
    harness.resume()
    await (await root.submit({"type": "input", "content": "one"}, context)).wait(context)
    await (await sub.submit({"type": "input", "content": "two"}, context)).wait(context)

    async def sections(conversation: Any) -> Any:
        entry = next(entry for entry in await all_entries(conversation) if entry["kind"] == "pi.system")
        return entry["model"][0]

    assert (await sections(root)).sections == {
        "cwd": "<cwd>\n/repo\n</cwd>",
        "agents": "<agents>\nRead AGENTS.md\n</agents>",
    }
    sub_system = await sections(sub)
    assert isinstance(sub_system, SystemMessage)
    assert sub_system.content == "" and sub_system.sections == {"cwd": "<cwd>\n/sub\n</cwd>"}
    assert sub_system.tools_added is None and sub_system.tools_removed is None
    await harness.close(context)


async def test_commits_no_partial_for_a_response_that_turns_deferred_after_an_empty_start_event() -> None:
    base = chat_setup()
    handle = DeferredHandle("faux", "faux-1", "faux", "handle-1", poll_after_ms=60_000)
    stream = _stream_of(
        faux_assistant_message([], stop_reason="pending"),  # type: ignore[arg-type]
        faux_assistant_message([], stop_reason="deferred", deferred=handle),
    )
    setup = with_models(base, _WithStream(base.models, stream))
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    values = live_publications(harness)
    submission = await root.submit({"type": "input", "content": "hi"}, context)

    async def polling() -> bool:
        return "deferred" in ((await live(harness, root)) or {}).get("generation", {})

    await wait_for(polling)
    assert not any("message" in value.get("generation", {}) for value in values)
    await harness.abort_task((await live(harness, root))["run"]["taskId"], context)
    settled = await submission.wait(context)
    assert settled["status"] == "unanswered" and settled["reason"] == "aborted"  # type: ignore[typeddict-item]
    assert [entry["kind"] for entry in await all_entries(root)] == ["pi.user"]
    await harness.close(context)


async def test_faults_a_run_task_settling_its_inputs_and_converting_the_committed_partial() -> None:
    base = chat_setup()
    setup = with_models(base, _WithStream(base.models, invalid_final_stream()))
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    values = live_publications(harness)
    harness.resume()
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    settled = await submission.wait(context)
    # Python's JSON check names the value differently than ``copyJson`` ("non-JSON function").
    assert settled["status"] == "unanswered" and settled["reason"] == "faulted"  # type: ignore[typeddict-item]
    assert "not JSON serializable" in settled["detail"]  # type: ignore[typeddict-item]
    assert any(_partial_text(value) == "partial" for value in values)
    entries = await all_entries(root)
    assert [entry["kind"] for entry in entries] == ["pi.user", "pi.assistant"]
    message = entries[1]["model"][0]
    assert isinstance(message, AssistantMessage) and message.stop_reason == "aborted"
    assert text_of(message) == "partial"
    assert await live(harness, root) == {}
    state = (await _tasks(harness, root))[0]["state"]
    assert state["status"] == "terminal" and state["outcome"]["status"] == "faulted"
    await harness.close(context)


async def test_orphans_a_blocked_run_task_with_full_run_cleanup() -> None:
    setup = chat_setup()
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root

    # A run whose task was stored by a newer generation definition this process cannot run.
    async def seed(tx: Any) -> Any:
        entry = await tx.append_entry(UserEntry, root.id, {"model": [UserMessage("hi", timestamp=1)]})
        submission = await tx.create_submission(
            {"conversationId": root.id, "type": "input", "status": "placed", "entry": entry["id"]}
        )
        task_id = await tx.create_task(
            Task(dataclasses.replace(GenerationTask.definition, version=2)),
            {},
            {"ownership": {"kind": "conversation"}, "conversationId": root.id},
        )
        (await tx.doc(LiveDoc, root.id))["run"] = {"taskId": task_id, "inputs": [submission["id"]]}
        return task_id, submission["id"]

    task_id, submission_id = await harness.commit_with(seed, context)
    harness.resume()
    with pytest.raises(Exception, match="is busy"):
        await root.submit({"type": "input", "content": "busy", "whenBusy": "reject"}, context)
    assert await harness.abort_task(task_id, context) == "marked"
    assert (await harness.wait_for_task(task_id, context))["state"]["outcome"] == {
        "status": "orphaned",
        "reason": "task_too_old",
    }
    submission = await harness.submission(submission_id, context)
    assert submission is not None
    status = await submission.status(context)
    assert status["status"] == "unanswered" and status["reason"] == "task_too_old"  # type: ignore[typeddict-item]
    assert await live(harness, root) == {}
    await harness.close(context)


async def test_rejects_a_registry_without_the_built_in_tasks() -> None:
    empty = create_registry().snapshot()

    class Snapshot:
        def installed(self) -> list[Any]:
            return []

        def extension(self, _name: str) -> None:
            return None

        def tools(self) -> list[Any]:
            return []

        def sections(self) -> list[Any]:
            return []

        def tasks(self) -> list[Any]:
            return [task for task in empty.tasks() if task.definition.name != "pi.generation"]

        def task(self, name: str) -> Any:
            return None if name == "pi.generation" else empty.task(name)

    class Reader:
        def snapshot(self) -> Any:
            return Snapshot()

        def subscribe(self, _listener: Any) -> Any:
            return lambda: None

    with pytest.raises(RuntimeError, match="Registry lacks built-in tasks pi.generation"):
        await Harness.open(MemoryStorage(), {"models": chat_setup().models, "registry": Reader()}, context)
