"""pi-durable telemetry: the ``pi.harness`` and ``pi.session`` schemas (a Python addition; upstream 1.3.0 emits no
spans from pi-durable).

Pass a :class:`pi_telemetry.TelemetryContext` as ``HarnessOptions["telemetryContext"]`` (or
``create_session(..., telemetry_context=...)``). The Harness then records:

- ``pi.harness.generation``: each phase handler run of a generation task, with the model request of a ``request`` or
  ``poll`` phase as a child ``pi.ai.request`` span;
- ``pi.harness.tool``: each tool call execution (``call``, recovery ``execute``, or ``abort``); a nested call's span is
  a child of its calling tool's span while that span is open in this process;
- ``pi.harness.task``: each phase handler run of any other task (compaction, extension tasks);
- ``pi.session.commit``: each Session commit.

The context is held in memory only, by the Harness or Session that was given it: it is never written to records,
documents, checkpoints or journals. A task resumed after a restart runs under the new process's context.

Nothing records content (prompts, tool arguments or output, error text) unless the context opted in with
:func:`pi_telemetry.with_content_capture`.
"""

from __future__ import annotations

from typing import Any, Final

from pi_telemetry import (
    CONTENT_EVENT,
    CONTENT_EVENT_NAME,
    NOOP_TELEMETRY_CONTEXT,
    SchemaTelemetrySpan,
    TelemetrySchemaDefinition,
    TelemetrySpanDefinition,
    TelemetryStartAttributeDefinition,
    TypedSpanStarter,
    create_typed_span_starter,
    define_telemetry_schema,
)

__all__ = [
    "HARNESS_TELEMETRY_SCHEMA",
    "SESSION_TELEMETRY_SCHEMA",
    "DURABLE_TELEMETRY_SCHEMAS",
    "GENERATION_SPAN",
    "TOOL_SPAN",
    "TASK_SPAN",
    "COMMIT_SPAN",
    "TOOL_OUTCOMES",
    "TOOL_ERROR_KINDS",
    "PHASE_OUTCOMES",
    "storage_kind",
    "phase_span",
]

GENERATION_SPAN: Final = "pi.harness.generation"
TOOL_SPAN: Final = "pi.harness.tool"
TASK_SPAN: Final = "pi.harness.task"
COMMIT_SPAN: Final = "pi.session.commit"

TOOL_OUTCOMES: Final = ("success", "error", "blocked", "interrupted")
#: ``tool_error``: the tool raised or returned an error result.
TOOL_ERROR_KINDS: Final = (
    "tool_unavailable",
    "invalid_arguments",
    "blocked",
    "tool_error",
    "invalid_structured_output",
    "interrupted",
    "aborted",
    "abandoned",
)
#: How a phase handler ended: it returned, raised, or raised after its task's abort signal fired.
PHASE_OUTCOMES: Final = ("returned", "raised", "aborted")

_TASK_ID: Final[TelemetryStartAttributeDefinition] = {
    "type": "number",
    "required": True,
    "cardinality": "high",
    "description": "Task ID",
}
_CONVERSATION_ID: Final[TelemetryStartAttributeDefinition] = {
    "type": "number",
    "required": True,
    "cardinality": "high",
    "description": "Conversation ID of the task",
}
_RESTARTS: Final[TelemetryStartAttributeDefinition] = {
    "type": "number",
    "required": False,
    "description": "Consecutive restarts that cut the task off while it ran without changing its state",
}


def _phase_span(
    description: str, phases: list[str], extra: dict[str, Any], end: dict[str, Any]
) -> TelemetrySpanDefinition:
    phase: TelemetryStartAttributeDefinition = {
        "type": "string",
        "required": True,
        "cardinality": "low",
        "description": "Phase handler (abort: the abort handler)",
    }
    if phases:
        phase["values"] = phases
    start: dict[str, TelemetryStartAttributeDefinition] = {
        "pi.harness.task_id": _TASK_ID,
        "pi.harness.conversation_id": _CONVERSATION_ID,
        "pi.harness.phase": phase,
        "pi.harness.restarts": _RESTARTS,
        **extra,
    }
    return {
        "description": description,
        "parents": {"kind": "root_or_external"},
        "startAttributes": start,
        "endAttributes": {
            "pi.harness.phase_outcome": {
                "type": "string",
                "values": list(PHASE_OUTCOMES),
                "description": "Whether the handler returned (durable progress), raised, or ended by an abort",
            },
            "pi.harness.duration_ms": {"type": "number", "description": "Milliseconds the handler ran"},
            **end,
        },
        "status": {"default": "ok", "errorWhen": "The phase handler raises"},
    }


HARNESS_TELEMETRY_SCHEMA: Final[TelemetrySchemaDefinition] = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            GENERATION_SPAN: {
                **_phase_span(
                    "One phase handler run of a generation task (pi.generation)",
                    ["prepare", "request", "retry", "poll", "tools", "abort"],
                    {
                        "pi.harness.attempt": {
                            "type": "number",
                            "required": False,
                            "description": "Generation attempt of the phase",
                        },
                        "pi.harness.model.provider": {
                            "type": "string",
                            "required": False,
                            "cardinality": "low",
                            "description": "Provider of the phase's model, once chosen",
                        },
                        "pi.harness.model.id": {
                            "type": "string",
                            "required": False,
                            "cardinality": "low",
                            "description": "Model ID of the phase's model, once chosen",
                        },
                    },
                    {
                        "pi.harness.generation.stop_reason": {
                            "type": "string",
                            "values": ["stop", "length", "toolUse", "error", "aborted", "deferred", "pending"],
                            "description": "Stop reason of the response a request or poll phase received",
                        },
                        "pi.harness.generation.tool_calls": {
                            "type": "number",
                            "description": "Tool calls in that response",
                        },
                    },
                ),
            },
            TOOL_SPAN: {
                "description": "One execution of a tool call: the call phase, a recovery, or the abort handler",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "pi.harness.task_id": _TASK_ID,
                    "pi.harness.conversation_id": _CONVERSATION_ID,
                    "pi.harness.phase": {
                        "type": "string",
                        "required": True,
                        "values": ["call", "execute", "abort"],
                        "description": "call: first run; execute: recovery after a restart; abort: the abort handler",
                    },
                    "pi.harness.tool.name": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "Called tool",
                    },
                    "pi.harness.tool.extension": {
                        "type": "string",
                        "required": False,
                        "cardinality": "low",
                        "description": "Extension that provides the tool",
                    },
                    "pi.harness.tool.replay": {
                        "type": "string",
                        "required": False,
                        "values": ["safe", "unsafe"],
                        "description": "The tool's replay policy, when it resolved",
                    },
                    "pi.harness.tool.nested": {
                        "type": "boolean",
                        "required": True,
                        "description": "Whether a running tool made the call (execute_tool()) instead of the model",
                    },
                    "pi.harness.restarts": _RESTARTS,
                },
                "endAttributes": {
                    "pi.harness.tool.outcome": {
                        "type": "string",
                        "values": list(TOOL_OUTCOMES),
                        "description": "How the call ended",
                    },
                    "pi.harness.tool.error_kind": {
                        "type": "string",
                        "values": list(TOOL_ERROR_KINDS),
                        "description": "Why the call failed; never the error text",
                    },
                    "pi.harness.tool.duration_ms": {
                        "type": "number",
                        "description": "Milliseconds the tool's execute() ran",
                    },
                    "pi.harness.tool.output_bytes": {
                        "type": "number",
                        "description": "UTF-8 bytes of the settled text output",
                    },
                    "pi.harness.tool.truncated": {
                        "type": "boolean",
                        "description": "Whether the output was cut to the tool's output limits",
                    },
                },
                "events": {
                    "pi.harness.tool.blocked": {
                        "description": "A before_tool hook blocked the call (the reason is not recorded)",
                        "attributes": {},
                    },
                    "pi.harness.tool.validation_failed": {
                        "description": "The call's arguments failed validation",
                        "attributes": {
                            "pi.harness.tool.validation_stage": {
                                "type": "string",
                                "required": True,
                                "values": ["arguments", "hook_arguments"],
                                "description": "The model's arguments, or those a before_tool hook replaced them with",
                            },
                        },
                    },
                    CONTENT_EVENT_NAME: CONTENT_EVENT,
                },
                "status": {"default": "ok", "errorWhen": "The call does not succeed"},
            },
            TASK_SPAN: _phase_span(
                "One phase handler run of a task other than generations and tool calls",
                [],
                {
                    "pi.harness.task.kind": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "Task definition name",
                    },
                    "pi.harness.task.extension": {
                        "type": "string",
                        "required": False,
                        "cardinality": "low",
                        "description": "Extension that installed the definition; absent for a built-in",
                    },
                },
                {},
            ),
        },
    }
)

SESSION_TELEMETRY_SCHEMA: Final[TelemetrySchemaDefinition] = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            COMMIT_SPAN: {
                "description": "One Session commit: the change callback, settlement, and the Storage commit",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "pi.session.storage": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "Storage backend",
                        "examples": ["memory", "sqlite", "jsonl", "custom"],
                    },
                },
                "endAttributes": {
                    "pi.session.write_count": {"type": "number", "description": "Storage writes the commit made"},
                    "pi.session.duration_ms": {"type": "number", "description": "Milliseconds on the mutation line"},
                },
                "status": {"default": "ok", "errorWhen": "The change callback, settlement, or Storage raises"},
            },
        },
    }
)

DURABLE_TELEMETRY_SCHEMAS: Final = (HARNESS_TELEMETRY_SCHEMA, SESSION_TELEMETRY_SCHEMA)

#: Indexed once; bound to each span's parent.
STARTER: Final[TypedSpanStarter] = create_typed_span_starter(
    NOOP_TELEMETRY_CONTEXT, DURABLE_TELEMETRY_SCHEMAS, on_invalid="warn"
)

_STORAGE_KINDS: Final = {"MemoryStorage": "memory", "SqliteStorage": "sqlite", "JsonlStorage": "jsonl"}


def storage_kind(storage: object) -> str:
    """``memory``, ``sqlite`` or ``jsonl`` for the built-in backends; a backend's own ``telemetry_kind``; else
    ``custom``."""
    kind = getattr(storage, "telemetry_kind", None)
    if isinstance(kind, str):
        return kind
    return _STORAGE_KINDS.get(type(storage).__name__, "custom")


def phase_span(runtime: Any) -> SchemaTelemetrySpan | None:
    """The span of the phase a task runtime serves, when telemetry is on."""
    span = getattr(runtime, "telemetry_context", None)
    return span if isinstance(span, SchemaTelemetrySpan) else None
