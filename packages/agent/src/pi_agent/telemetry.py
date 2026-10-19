"""pi-agent telemetry: the ``pi.agent`` schema and its ``pi.agent.run`` span (a Python addition; upstream 1.3.0's
agent-core emits no spans).

Pass a :class:`pi_telemetry.TelemetryContext` as ``Agent(telemetry_context=...)`` or
``AgentLoopConfig.telemetry_context``. Each run of the loop (``prompt()``/``continue_()``, ``agent_loop()``,
``run_agent_loop()``...) then records one ``pi.agent.run`` span, and every model request of the run becomes a child
``pi.ai.request`` span: the loop passes the run span to the stream function as ``telemetry_context``. Without a
context nothing is built or recorded.

The span records counts, usage and the end reason, never messages, prompts or tool arguments.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final

from pi_telemetry import (
    NOOP_TELEMETRY_CONTEXT,
    SchemaTelemetrySpan,
    TelemetryContext,
    TelemetrySchemaDefinition,
    TypedSpanStarter,
    bind_typed_span_starter,
    create_typed_span_starter,
    define_telemetry_schema,
)

__all__ = [
    "AGENT_TELEMETRY_SCHEMA",
    "AGENT_TELEMETRY_SCHEMAS",
    "AGENT_RUN_SPAN",
    "AGENT_END_REASONS",
    "trace_agent_run",
]

AGENT_RUN_SPAN: Final = "pi.agent.run"

#: ``completed``: no more tool calls, steering or follow-ups; ``ended``: ``finish_turn`` returned ``end``;
#: ``error``/``aborted``: the last response's stop reason; ``failed``: the loop raised.
AGENT_END_REASONS: Final = ("completed", "ended", "error", "aborted", "failed")

AGENT_TELEMETRY_SCHEMA: Final[TelemetrySchemaDefinition] = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            AGENT_RUN_SPAN: {
                "description": "One run of the agent loop, from its prompt (or continuation) to agent_end",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "pi.agent.provider": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "Provider of the run's initial model",
                    },
                    "pi.agent.model": {
                        "type": "string",
                        "required": True,
                        "cardinality": "low",
                        "description": "The run's initial model ID",
                    },
                    "pi.agent.tool_count": {
                        "type": "number",
                        "required": True,
                        "description": "Tools the agent can execute",
                    },
                    "pi.agent.mode": {
                        "type": "string",
                        "required": True,
                        "values": ["prompt", "continue"],
                        "description": "New prompt messages, or a continuation of the existing context",
                    },
                    "pi.agent.reasoning": {
                        "type": "string",
                        "required": False,
                        "cardinality": "low",
                        "description": "Requested thinking level",
                    },
                },
                "endAttributes": {
                    "pi.agent.turns": {"type": "number", "description": "Model responses in the run"},
                    "pi.agent.tool_calls": {"type": "number", "description": "Tool results the run produced"},
                    "pi.agent.usage.input_tokens": {
                        "type": "number",
                        "description": "Input tokens summed over the run's responses",
                    },
                    "pi.agent.usage.output_tokens": {
                        "type": "number",
                        "description": "Output tokens summed over the run's responses",
                    },
                    "pi.agent.cost": {"type": "number", "description": "Estimated cost of the run's responses in USD"},
                    "pi.agent.end_reason": {
                        "type": "string",
                        "values": list(AGENT_END_REASONS),
                        "cardinality": "low",
                        "description": "Why the run ended",
                    },
                },
                "events": {
                    "pi.agent.steering": {
                        "description": "Steering messages were injected into the run",
                        "attributes": {
                            "pi.agent.message_count": {
                                "type": "number",
                                "required": True,
                                "description": "Messages injected",
                            },
                        },
                    },
                    "pi.agent.follow_up": {
                        "description": "Follow-up messages continued the run after it would have stopped",
                        "attributes": {
                            "pi.agent.message_count": {
                                "type": "number",
                                "required": True,
                                "description": "Messages injected",
                            },
                        },
                    },
                },
                "status": {"default": "ok", "errorWhen": "The run ends with an error or abort, or the loop raises"},
            },
        },
    }
)

#: The schemas pi_agent owns. Combine with ``pi_ai.telemetry.AI_TELEMETRY_SCHEMAS`` to check a whole run.
AGENT_TELEMETRY_SCHEMAS: Final = (AGENT_TELEMETRY_SCHEMA,)

_STARTER: Final[TypedSpanStarter] = create_typed_span_starter(
    NOOP_TELEMETRY_CONTEXT, AGENT_TELEMETRY_SCHEMAS, on_invalid="warn"
)


def _end_attributes(messages: Sequence[Any], end_reason: str) -> dict[str, Any]:
    turns = tool_calls = input_tokens = output_tokens = 0
    cost = 0.0
    for message in messages:
        role = getattr(message, "role", None)
        if role == "assistant":
            turns += 1
            usage = message.usage
            input_tokens += usage.input
            output_tokens += usage.output
            cost += usage.cost.total
        elif role == "toolResult":
            tool_calls += 1
    return {
        "pi.agent.turns": turns,
        "pi.agent.tool_calls": tool_calls,
        "pi.agent.usage.input_tokens": input_tokens,
        "pi.agent.usage.output_tokens": output_tokens,
        "pi.agent.cost": cost,
        "pi.agent.end_reason": end_reason,
    }


async def trace_agent_run(
    telemetry: TelemetryContext,
    config: Any,
    tool_count: int,
    mode: str,
    new_messages: Sequence[Any],
    run: Callable[[SchemaTelemetrySpan], Awaitable[str]],
) -> None:
    """Run ``run(span)`` inside a ``pi.agent.run`` span; ``run`` returns the end reason."""
    attributes: dict[str, Any] = {
        "pi.agent.provider": config.model.provider,
        "pi.agent.model": config.model.id,
        "pi.agent.tool_count": tool_count,
        "pi.agent.mode": mode,
    }
    if config.reasoning is not None:
        attributes["pi.agent.reasoning"] = config.reasoning

    async def body(span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> None:
        try:
            end_reason = await run(span)
        except BaseException as error:
            span.set_attributes(_end_attributes(new_messages, "failed"))
            span.set_status({"status": "error", "error": {"name": type(error).__name__, "message": ""}})
            raise
        span.set_attributes(_end_attributes(new_messages, end_reason))
        if end_reason in ("error", "aborted"):
            span.set_status({"status": "error", "error": {"name": end_reason, "message": ""}})

    await bind_typed_span_starter(telemetry, _STARTER)(AGENT_RUN_SPAN, attributes, body)
