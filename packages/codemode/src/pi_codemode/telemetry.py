"""pi-codemode telemetry: the ``pi.codemode`` schema and its ``pi.codemode.run`` span (a Python addition).

``CodemodeSandboxBase.execute(..., telemetry_context=...)`` records one ``pi.codemode.run`` span per execution.
Hosts of deterministic runs (:meth:`CodemodeSandboxBase.start`), such as pi_durable's code mode tool, record it
with :func:`trace_codemode_run` and add what only they know: whether the run replays a journal, and when a replay
takes a different path.

The span records sizes, counts, durations and the outcome, never the script, its output, its arguments or error
text, unless the context opted in with :func:`pi_telemetry.with_content_capture` (then the script and its text
output are ``pi.content`` events).
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final

from pi_telemetry import (
    CONTENT_EVENT,
    CONTENT_EVENT_NAME,
    NOOP_TELEMETRY_CONTEXT,
    SchemaTelemetrySpan,
    TelemetryContext,
    TelemetrySchemaDefinition,
    TypedSpanStarter,
    bind_typed_span_starter,
    create_typed_span_starter,
    define_telemetry_schema,
    record_content,
)

from .types import CodemodeFailure, CodemodeResult

__all__ = [
    "CODEMODE_TELEMETRY_SCHEMA",
    "CODEMODE_TELEMETRY_SCHEMAS",
    "CODEMODE_RUN_SPAN",
    "CODEMODE_RUN_ERROR_KINDS",
    "trace_codemode_run",
    "record_codemode_result",
]

CODEMODE_RUN_SPAN: Final = "pi.codemode.run"

#: The sandbox's error kinds, plus the outcomes a durable host decides: a replay that took a different path, a run
#: started under another code mode version, a deadline that passed while the run was interrupted, and a host
#: failure.
CODEMODE_RUN_ERROR_KINDS: Final = (
    "script",
    "timeout",
    "aborted",
    "sandbox",
    "different_path",
    "version_mismatch",
    "deadline_passed",
    "host",
)

CODEMODE_TELEMETRY_SCHEMA: Final[TelemetrySchemaDefinition] = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            CODEMODE_RUN_SPAN: {
                "description": "One execution of a model-written script",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "pi.codemode.line_count": {
                        "type": "number",
                        "required": True,
                        "description": "Lines of the script",
                    },
                    "pi.codemode.timeout_ms": {
                        "type": "number",
                        "required": False,
                        "description": "The run's deadline; absent when it has none",
                    },
                    "pi.codemode.mode": {
                        "type": "string",
                        "required": True,
                        "values": ["execute", "deterministic"],
                        "description": "execute(): the sandbox calls tools; deterministic: start(), the host answers",
                    },
                    "pi.codemode.tool_count": {
                        "type": "number",
                        "required": True,
                        "description": "Tools the script can call",
                    },
                },
                "endAttributes": {
                    "pi.codemode.outcome": {
                        "type": "string",
                        "values": ["success", "error"],
                        "description": "Whether the script succeeded",
                    },
                    "pi.codemode.error_kind": {
                        "type": "string",
                        "values": list(CODEMODE_RUN_ERROR_KINDS),
                        "cardinality": "low",
                        "description": "Why the run failed; never the error text",
                    },
                    "pi.codemode.nested_call_count": {
                        "type": "number",
                        "description": "Tool calls the script made",
                    },
                    "pi.codemode.duration_ms": {"type": "number", "description": "Milliseconds the run took"},
                    "pi.codemode.replayed": {
                        "type": "boolean",
                        "description": "Whether the run replayed a journal after a restart",
                    },
                    "pi.codemode.replayed_call_count": {
                        "type": "number",
                        "description": "Journaled results the replay delivered",
                    },
                },
                "events": {
                    "pi.codemode.different_path": {
                        "description": "A replay took a different path than the run it resumes",
                        "attributes": {},
                    },
                    "pi.codemode.timeout": {
                        "description": "The run's deadline expired",
                        "attributes": {
                            "pi.codemode.timeout_ms": {
                                "type": "number",
                                "required": False,
                                "description": "The deadline that expired",
                            },
                        },
                    },
                    CONTENT_EVENT_NAME: CONTENT_EVENT,
                },
                "status": {"default": "ok", "errorWhen": "The script fails, times out, is aborted, or diverges"},
            },
        },
    }
)

CODEMODE_TELEMETRY_SCHEMAS: Final = (CODEMODE_TELEMETRY_SCHEMA,)

_STARTER: Final[TypedSpanStarter] = create_typed_span_starter(
    NOOP_TELEMETRY_CONTEXT, CODEMODE_TELEMETRY_SCHEMAS, on_invalid="warn"
)


def _line_count(code: str) -> int:
    return 0 if code == "" else code.count("\n") + 1


def record_codemode_result(span: SchemaTelemetrySpan, result: CodemodeResult, timeout_ms: float | None) -> None:
    """End attributes, the timeout event and the status of a sandbox result."""
    attributes: dict[str, Any] = {"pi.codemode.nested_call_count": len(result.calls)}
    if not isinstance(result, CodemodeFailure):
        attributes["pi.codemode.outcome"] = "success"
    else:
        kind = result.error.kind
        attributes["pi.codemode.outcome"] = "error"
        attributes["pi.codemode.error_kind"] = kind
        if kind == "timeout":
            timeout: dict[str, Any] = {}
            if timeout_ms is not None and math.isfinite(timeout_ms):
                timeout["pi.codemode.timeout_ms"] = timeout_ms
            span.add_event("pi.codemode.timeout", timeout)
        span.set_status({"status": "error", "error": {"name": kind, "message": ""}})
    span.set_attributes(attributes)
    if span.content_capture is not None:
        for item in result.output:
            if item.type == "text":
                record_content(span, "output", item.text)  # type: ignore[union-attr]
        if isinstance(result, CodemodeFailure):
            record_content(span, "error", result.error.message)


async def trace_codemode_run(
    telemetry: TelemetryContext,
    code: str,
    *,
    mode: str,
    timeout_ms: float | None,
    tool_count: int,
    run: Callable[[SchemaTelemetrySpan], Awaitable[Any]],
    error_kind: Callable[[BaseException], str] | None = None,
) -> Any:
    """Run ``run(span)`` inside a ``pi.codemode.run`` span and record its duration.

    ``run`` records the outcome (for a sandbox result, with :func:`record_codemode_result`). A raise records the
    error kind ``error_kind(error)`` (default ``host``) and the exception's class name, never its message.
    """
    attributes: dict[str, Any] = {
        "pi.codemode.line_count": _line_count(code),
        "pi.codemode.mode": mode,
        "pi.codemode.tool_count": tool_count,
    }
    if timeout_ms is not None and math.isfinite(timeout_ms):
        attributes["pi.codemode.timeout_ms"] = timeout_ms

    async def body(span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> Any:
        started = time.perf_counter()
        record_content(span, "code", code)
        try:
            return await run(span)
        except BaseException as error:
            kind = "host"
            if error_kind is not None:
                try:
                    kind = error_kind(error)
                except Exception:
                    pass
            span.set_attributes({"pi.codemode.outcome": "error", "pi.codemode.error_kind": kind})
            span.set_status({"status": "error", "error": {"name": type(error).__name__, "message": ""}})
            raise
        finally:
            span.set_attributes({"pi.codemode.duration_ms": round((time.perf_counter() - started) * 1000, 3)})

    return await bind_typed_span_starter(telemetry, _STARTER)(CODEMODE_RUN_SPAN, attributes, body)
