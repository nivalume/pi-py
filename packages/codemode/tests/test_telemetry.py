"""The ``pi.codemode.run`` span of ``CodemodeSandboxBase.execute()``."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pi_codemode import CodemodeSandbox, CodemodeTool
from pi_codemode.telemetry import CODEMODE_TELEMETRY_SCHEMAS
from pi_telemetry import InMemoryTelemetryContext, RecordedTelemetrySpan, with_content_capture
from pi_telemetry.testing import assert_spans_match_schemas

CreateSandbox = Callable[..., CodemodeSandbox]


async def _echo(args: Any, _context: Any) -> Any:
    return {"echo": args}


def _dump(spans: list[RecordedTelemetrySpan]) -> str:
    return json.dumps([[span.attributes, [[e.name, e.attributes] for e in span.events]] for span in spans])


async def test_execute_records_a_run_span_without_content(create_sandbox: CreateSandbox) -> None:
    sandbox = create_sandbox([CodemodeTool("echo", _echo)])
    telemetry = InMemoryTelemetryContext()
    code = 'a = await tools.echo(x="SECRET-ARG")\nb = await tools.echo(y=2)\ntext("SECRET-OUTPUT")\nreturn 1'
    result = await sandbox.execute(code, telemetry_context=telemetry)
    assert result.ok
    [span] = telemetry.get_spans()
    attributes = dict(span.attributes)
    assert attributes.pop("pi.codemode.duration_ms") > 0  # type: ignore[operator]
    assert attributes == {
        "pi.codemode.line_count": 4,
        "pi.codemode.mode": "execute",
        "pi.codemode.tool_count": 1,
        "pi.codemode.timeout_ms": 10_000,
        "pi.codemode.outcome": "success",
        "pi.codemode.nested_call_count": 2,
    }
    assert span.status == {"status": "ok"} and span.events == []
    assert "SECRET" not in _dump(telemetry.get_spans())
    assert_spans_match_schemas(telemetry.get_spans(), CODEMODE_TELEMETRY_SCHEMAS)


async def test_failures_record_the_kind_and_timeouts_an_event(create_sandbox: CreateSandbox) -> None:
    sandbox = create_sandbox([], timeout_ms=200)
    telemetry = InMemoryTelemetryContext()
    failed = await sandbox.execute('raise ValueError("SECRET-ERROR")', telemetry_context=telemetry)
    timed_out = await sandbox.execute("while True:\n    pass", telemetry_context=telemetry)
    assert not failed.ok and not timed_out.ok
    script, timeout = telemetry.get_spans()
    assert script.status == {"status": "error", "error": {"name": "script", "message": ""}}
    assert script.attributes["pi.codemode.error_kind"] == "script"
    assert timeout.attributes["pi.codemode.error_kind"] == "timeout"
    assert [(event.name, event.attributes) for event in timeout.events] == [
        ("pi.codemode.timeout", {"pi.codemode.timeout_ms": 200})
    ]
    assert "SECRET" not in _dump(telemetry.get_spans())
    assert_spans_match_schemas(telemetry.get_spans(), CODEMODE_TELEMETRY_SCHEMAS)


async def test_content_capture_records_the_script_and_its_output(create_sandbox: CreateSandbox) -> None:
    sandbox = create_sandbox([])
    telemetry = InMemoryTelemetryContext()
    await sandbox.execute('text("hello")\nreturn 1', telemetry_context=with_content_capture(telemetry))
    [span] = telemetry.get_spans()
    assert [(event.attributes["pi.content.kind"], event.attributes["pi.content.text"]) for event in span.events] == [
        ("code", 'text("hello")\nreturn 1'),
        ("output", "hello"),
    ]
    assert_spans_match_schemas(telemetry.get_spans(), CODEMODE_TELEMETRY_SCHEMAS)


async def test_codemode_agent_tool_passes_its_telemetry_context() -> None:
    from pi_codemode import create_codemode_tool

    telemetry = InMemoryTelemetryContext()
    tool = create_codemode_tool([], telemetry_context=telemetry)
    result = await tool.execute("call-1", {"code": "return 3"}, None, None)
    assert result.is_error is False
    [span] = telemetry.get_spans()
    assert span.name == "pi.codemode.run" and span.attributes["pi.codemode.outcome"] == "success"
