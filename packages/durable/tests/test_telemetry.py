"""Harness telemetry: ``pi.harness.*`` and ``pi.session.commit`` spans, explicit wiring through
``HarnessOptions["telemetryContext"]``, the default data policy, and a restart that attaches to the new context."""

from __future__ import annotations

import asyncio
import json
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import ChatSetup, add_hooks, add_task, add_tool, chat_setup, wait_for
from durable_session_support import context, deferred, resolve

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.telemetry import AI_TELEMETRY_SCHEMAS
from pi_ai.types import TextContent
from pi_chord.context import Context
from pi_codemode.telemetry import CODEMODE_TELEMETRY_SCHEMAS
from pi_durable import Harness, MemoryStorage, ToolTask, create_session, define_task, define_tool
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.telemetry import DURABLE_TELEMETRY_SCHEMAS, storage_kind
from pi_telemetry import (
    InMemoryTelemetryContext,
    RecordedTelemetrySpan,
    TelemetryContext,
    with_content_capture,
)
from pi_telemetry.testing import assert_spans_match_schemas

SCHEMAS = (*DURABLE_TELEMETRY_SCHEMAS, *AI_TELEMETRY_SCHEMAS, *CODEMODE_TELEMETRY_SCHEMAS)
Parameters = Type.Object({"text": Type.Optional(Type.String())})


def tool(name: str, execute: Any, **extra: Any) -> Any:
    return define_tool(name=name, description=name, parameters=Parameters, execute=execute, **extra)


def call(name: str, args: dict[str, Any] | None = None, id: str = "c1") -> Any:
    return faux_assistant_message([faux_tool_call(name, args or {}, id=id)], stop_reason="toolUse")


DONE = faux_assistant_message([faux_text("done")])


async def open_traced(storage: Any, setup: ChatSetup, telemetry: TelemetryContext) -> tuple[Harness, Any]:
    options: Any = {
        "models": setup.models,
        "registry": setup.registry,
        "settings": setup.settings,
        "onReport": lambda error, _info: setup.reports.append(error),
        "telemetryContext": telemetry,
    }
    harness = await Harness.open(storage, options, context)
    root = await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})
    return harness, root


async def run(root: Any, content: str = "SECRET-PROMPT go") -> str:
    return (await (await root.submit({"type": "input", "content": content}, context)).wait(context))["status"]


def named(spans: Sequence[RecordedTelemetrySpan], name: str) -> list[RecordedTelemetrySpan]:
    return [span for span in spans if span.name == name]


def dump(spans: Sequence[RecordedTelemetrySpan]) -> str:
    return json.dumps(
        [[span.attributes, [[event.name, event.attributes] for event in span.events], span.status] for span in spans]
    )


async def secret_echo(args: Any, _api: Any, _context: Context) -> Any:
    return {"output": [TextContent(f"SECRET-OUTPUT {args.get('text') or ''}")]}


async def test_a_run_records_generation_tool_request_and_commit_spans() -> None:
    setup = chat_setup()
    add_tool(setup.registry, tool("echo", secret_echo))
    setup.faux.set_responses([call("echo", {"text": "SECRET-ARG"}), DONE])
    telemetry = InMemoryTelemetryContext()
    harness, root = await open_traced(MemoryStorage(), setup, telemetry)
    assert await run(root) == "done"
    await harness.close(context)
    spans = telemetry.get_spans()
    assert all(span.settled for span in spans)
    assert_spans_match_schemas(spans, SCHEMAS)
    assert "SECRET" not in dump(spans)

    generations = named(spans, "pi.harness.generation")
    phases = [span.attributes["pi.harness.phase"] for span in generations]
    assert phases[:2] == ["prepare", "request"]
    assert phases.count("request") == 2
    request_phases = [span for span in generations if span.attributes["pi.harness.phase"] == "request"]
    requests = named(spans, "pi.ai.request")
    assert [request.parent_id for request in requests] == [phase.id for phase in request_phases]
    assert request_phases[0].attributes["pi.harness.model.provider"] == "faux"
    assert request_phases[0].attributes["pi.harness.generation.stop_reason"] == "toolUse"
    assert request_phases[0].attributes["pi.harness.generation.tool_calls"] == 1
    assert request_phases[1].attributes["pi.harness.generation.stop_reason"] == "stop"
    assert all(span.attributes["pi.harness.phase_outcome"] == "returned" for span in generations)
    assert requests[0].attributes["pi.ai.session_id"]  # the provider session ID the generation sets

    [echo] = named(spans, "pi.harness.tool")
    attributes = dict(echo.attributes)
    assert isinstance(attributes.pop("pi.harness.tool.duration_ms"), (int, float))
    assert attributes == {
        "pi.harness.task_id": attributes["pi.harness.task_id"],
        "pi.harness.conversation_id": root.id,
        "pi.harness.phase": "call",
        "pi.harness.tool.name": "echo",
        "pi.harness.tool.replay": "unsafe",
        "pi.harness.tool.nested": False,
        "pi.harness.tool.extension": attributes["pi.harness.tool.extension"],
        "pi.harness.tool.outcome": "success",
        "pi.harness.tool.output_bytes": len("SECRET-OUTPUT SECRET-ARG"),
        "pi.harness.tool.truncated": False,
    }
    assert echo.parent_id is None and echo.status == {"status": "ok"}

    commits = named(spans, "pi.session.commit")
    assert commits and all(span.attributes["pi.session.storage"] == "memory" for span in commits)
    assert sum(span.attributes["pi.session.write_count"] for span in commits) > 0  # type: ignore[misc]


async def test_nested_calls_are_children_of_their_callers_span() -> None:
    setup = chat_setup()
    add_tool(setup.registry, tool("echo", secret_echo))

    async def batch(_args: Any, api: Any, call_context: Context) -> Any:
        assert api.telemetry_context is not None
        first = await api.execute_tool("echo", {"text": "a"}, call_context)
        return {"output": [TextContent(str(first["isError"]))]}

    add_tool(setup.registry, tool("batch", batch))
    setup.faux.set_responses([call("batch"), DONE])
    telemetry = InMemoryTelemetryContext()
    harness, root = await open_traced(MemoryStorage(), setup, telemetry)
    assert await run(root) == "done"
    await harness.close(context)
    spans = telemetry.get_spans()
    batch_span, echo_span = named(spans, "pi.harness.tool")
    assert batch_span.attributes["pi.harness.tool.nested"] is False
    assert echo_span.attributes["pi.harness.tool.nested"] is True
    assert echo_span.parent_id == batch_span.id
    assert echo_span.attributes["pi.harness.tool.outcome"] == "success"
    assert_spans_match_schemas(spans, SCHEMAS)


async def test_blocked_invalid_and_failing_calls_record_kinds_and_events() -> None:
    setup = chat_setup()
    add_tool(setup.registry, tool("echo", secret_echo))

    async def broken(_args: Any, _api: Any, _context: Context) -> Any:
        raise RuntimeError("SECRET-ERROR text")

    add_tool(setup.registry, tool("broken", broken))
    add_tool(
        setup.registry,
        define_tool(
            name="strict", description="strict", parameters=Type.Object({"n": Type.Number()}), execute=secret_echo
        ),
    )

    def before_tool(hook_call: Any, _runtime: Any, _context: Context) -> Any:
        return {"block": "SECRET-REASON"} if hook_call.name == "echo" else None

    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})
    setup.faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("echo", {"text": "x"}, id="c1"),
                    faux_tool_call("strict", {"n": "SECRET-NOT-A-NUMBER"}, id="c2"),
                    faux_tool_call("broken", {}, id="c3"),
                ],
                stop_reason="toolUse",
            ),
            DONE,
        ]
    )
    telemetry = InMemoryTelemetryContext()
    harness, root = await open_traced(MemoryStorage(), setup, telemetry)
    assert await run(root) == "done"
    await harness.close(context)
    spans = telemetry.get_spans()
    by_name = {span.attributes["pi.harness.tool.name"]: span for span in named(spans, "pi.harness.tool")}
    echo, strict, failing = by_name["echo"], by_name["strict"], by_name["broken"]
    assert echo.attributes["pi.harness.tool.outcome"] == "blocked"
    assert [event.name for event in echo.events] == ["pi.harness.tool.blocked"]
    assert echo.status == {"status": "error", "error": {"name": "blocked", "message": ""}}
    assert strict.attributes["pi.harness.tool.outcome"] == "error"
    assert strict.attributes["pi.harness.tool.error_kind"] == "invalid_arguments"
    assert [(event.name, event.attributes) for event in strict.events] == [
        ("pi.harness.tool.validation_failed", {"pi.harness.tool.validation_stage": "arguments"})
    ]
    assert failing.attributes["pi.harness.tool.outcome"] == "error"
    assert failing.attributes["pi.harness.tool.error_kind"] == "tool_error"
    assert "SECRET" not in dump(spans)
    assert_spans_match_schemas(spans, SCHEMAS)


async def test_custom_task_phases_record_task_spans() -> None:
    setup = chat_setup()
    finished = deferred()

    async def first(_task: Any, runtime: Any, call_context: Context) -> None:
        assert runtime.telemetry_context is not None
        await runtime.commit(
            lambda _tx, _current: {"status": "running", "checkpoint": {"phase": "second"}}, call_context
        )

    async def second(_task: Any, runtime: Any, call_context: Context) -> None:
        await runtime.commit(
            lambda _tx, _current: {"status": "terminal", "outcome": {"status": "completed"}}, call_context
        )
        resolve(finished)

    async def abort(_task: Any, runtime: Any, call_context: Context) -> None:
        await runtime.commit(
            lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, call_context
        )

    job = define_task(
        name="example.job",
        version=1,
        initial=lambda _input: {"phase": "first"},
        phases={"first": first, "second": second},
        abort=abort,
    )
    add_task(setup.registry, job, "jobs")
    telemetry = InMemoryTelemetryContext()
    harness, root = await open_traced(MemoryStorage(), setup, telemetry)
    harness.resume()
    await root.commit(lambda tx: tx.create_task(job, None, {"ownership": {"kind": "conversation"}}), context)
    await finished
    await wait_for(
        lambda: (
            len(named(telemetry.get_spans(), "pi.harness.task")) == 2
            and all(span.settled for span in named(telemetry.get_spans(), "pi.harness.task"))
        )
    )
    await harness.close(context)
    tasks = named(telemetry.get_spans(), "pi.harness.task")
    assert [span.attributes["pi.harness.phase"] for span in tasks] == ["first", "second"]
    assert all(span.attributes["pi.harness.task.kind"] == "example.job" for span in tasks)
    assert all(span.attributes["pi.harness.task.extension"] == "jobs" for span in tasks)
    assert_spans_match_schemas(telemetry.get_spans(), SCHEMAS)


async def test_content_is_recorded_only_with_an_explicit_opt_in() -> None:
    setup = chat_setup()
    add_tool(setup.registry, tool("echo", secret_echo))
    setup.faux.set_responses([call("echo", {"text": "visible"}), DONE])
    telemetry = InMemoryTelemetryContext()
    harness, root = await open_traced(MemoryStorage(), setup, with_content_capture(telemetry))
    assert await run(root) == "done"
    await harness.close(context)
    [echo] = named(telemetry.get_spans(), "pi.harness.tool")
    content = [(event.attributes["pi.content.kind"], event.attributes["pi.content.text"]) for event in echo.events]
    assert content == [("tool_arguments", '{"text": "visible"}'), ("tool_output", "SECRET-OUTPUT visible")]
    requests = named(telemetry.get_spans(), "pi.ai.request")
    assert any(event.name == "pi.content" for event in requests[0].events)
    assert_spans_match_schemas(telemetry.get_spans(), SCHEMAS)


async def test_telemetry_is_never_persisted_and_a_resumed_run_attaches_to_the_new_context(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    setup = chat_setup()
    reached = deferred()
    release = {"block": True, "runs": 0}

    async def hold(_args: Any, _api: Any, call_context: Context) -> Any:
        release["runs"] += 1
        if release["block"]:
            resolve(reached)
            await call_context.abort_signal.wait()  # type: ignore[union-attr]
        return {"output": [TextContent("held")]}

    add_tool(setup.registry, tool("hold", hold, replay="safe"))
    setup.faux.set_responses([call("hold"), DONE])
    first = InMemoryTelemetryContext()
    harness, root = await open_traced(await open_sqlite_storage(str(path)), setup, first)
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await reached
    await harness.close(context)
    assert all(span.settled for span in first.get_spans())
    [interrupted] = named(first.get_spans(), "pi.harness.tool")
    assert interrupted.attributes["pi.harness.tool.outcome"] == "interrupted"

    # Nothing of the telemetry context reached storage.
    raw = path.read_bytes()
    for needle in (b"telemetry", b"pi.harness", b"pi.session", b"InMemoryTelemetry"):
        assert needle not in raw

    release["block"] = False
    second = InMemoryTelemetryContext()
    harness, root = await open_traced(await open_sqlite_storage(str(path)), setup, second)
    harness.resume()
    reacquired = await harness.submission(submission.id, context)
    assert reacquired is not None
    assert (await reacquired.wait(context))["status"] == "done"
    await harness.close(context)
    [resumed] = named(second.get_spans(), "pi.harness.tool")
    assert resumed.attributes["pi.harness.phase"] == "execute"
    assert resumed.attributes["pi.harness.tool.replay"] == "safe"
    assert resumed.attributes["pi.harness.tool.outcome"] == "success"
    assert {span.attributes["pi.session.storage"] for span in named(second.get_spans(), "pi.session.commit")} == {
        "sqlite"
    }
    assert len(first.get_spans()) > 0 and first.get_spans()[-1].settled
    assert_spans_match_schemas(second.get_spans(), SCHEMAS)


async def test_sessions_take_a_telemetry_context_and_storage_kinds() -> None:
    telemetry = InMemoryTelemetryContext()
    session = create_session(MemoryStorage(), telemetry_context=telemetry)
    await session.commit(lambda tx: None, context)
    with pytest.raises(Exception):  # noqa: B017
        await session.commit(lambda tx: (_ for _ in ()).throw(RuntimeError("SECRET")), context)
    await session.close(context)
    empty, failed = telemetry.get_spans()
    assert empty.attributes["pi.session.storage"] == "memory"
    assert empty.attributes["pi.session.write_count"] == 0
    assert failed.status == {"status": "error", "error": {"name": "RuntimeError", "message": ""}}
    assert storage_kind(object()) == "custom"

    class Tagged:
        telemetry_kind = "postgres"

    assert storage_kind(Tagged()) == "postgres"


async def test_durable_code_mode_records_run_spans_and_replays_after_a_restart(tmp_path: Path) -> None:
    pytest.importorskip("pydantic_monty")
    from pi_codemode import MontyCodemodeSandbox
    from pi_durable.codemode.codemode import create_codemode_tool

    setup = chat_setup()
    sandbox = MontyCodemodeSandbox(memory_limit_bytes=64 * 1024 * 1024)
    add_tool(setup.registry, create_codemode_tool(sandbox=sandbox))
    add_tool(setup.registry, tool("echo", secret_echo))
    reached = deferred()
    release = {"block": True}

    async def hold(_args: Any, _api: Any, call_context: Context) -> Any:
        if release["block"]:
            resolve(reached)
            await call_context.abort_signal.wait()  # type: ignore[union-attr]
        return {"output": [TextContent("held")]}

    add_tool(setup.registry, tool("hold", hold, replay="safe"))
    code = textwrap.dedent(
        """
        a = await tools.echo(text="SECRET-A")
        held = await tools.hold()
        return [a, held]
        """
    ).strip()
    setup.faux.set_responses(
        [faux_assistant_message([faux_tool_call("codemode", {"code": code}, id="s1")], stop_reason="toolUse"), DONE]
    )
    path = tmp_path / "session.sqlite"
    first = InMemoryTelemetryContext()
    try:
        harness, root = await open_traced(await open_sqlite_storage(str(path)), setup, first)
        submission = await root.submit({"type": "input", "content": "go"}, context)
        await reached
        await harness.close(context)
        [run_one] = named(first.get_spans(), "pi.codemode.run")
        tool_one = next(
            span
            for span in named(first.get_spans(), "pi.harness.tool")
            if span.attributes["pi.harness.tool.name"] == "codemode"
        )
        assert run_one.parent_id == tool_one.id
        assert run_one.attributes["pi.codemode.mode"] == "deterministic"
        assert run_one.attributes["pi.codemode.line_count"] == 3
        assert run_one.attributes["pi.codemode.replayed"] is False
        nested_one = [
            span for span in named(first.get_spans(), "pi.harness.tool") if span.attributes["pi.harness.tool.nested"]
        ]
        assert nested_one and all(span.parent_id == tool_one.id for span in nested_one)

        release["block"] = False
        second = InMemoryTelemetryContext()
        harness, root = await open_traced(await open_sqlite_storage(str(path)), setup, second)
        harness.resume()
        reacquired = await harness.submission(submission.id, context)
        assert reacquired is not None
        assert (await reacquired.wait(context))["status"] == "done"
        await harness.close(context)
        [run_two] = named(second.get_spans(), "pi.codemode.run")
        assert run_two.attributes["pi.codemode.replayed"] is True
        assert run_two.attributes["pi.codemode.replayed_call_count"] == 1
        assert run_two.attributes["pi.codemode.outcome"] == "success"
        assert run_two.attributes["pi.codemode.nested_call_count"] == 2
        assert "SECRET" not in dump([*first.get_spans(), *second.get_spans()])
        assert_spans_match_schemas(first.get_spans(), SCHEMAS)
        assert_spans_match_schemas(second.get_spans(), SCHEMAS)
    finally:
        await sandbox.close()
    await asyncio.sleep(0)
