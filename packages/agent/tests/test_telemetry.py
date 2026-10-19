"""The ``pi.agent.run`` span and its ``pi.ai.request`` children."""

from __future__ import annotations

import json
from typing import Any

from pi_agent import Agent, AgentInitialState, AgentLoopConfig, AgentTool, AgentToolResult
from pi_agent.agent_loop import run_agent_loop
from pi_agent.telemetry import AGENT_TELEMETRY_SCHEMAS
from pi_agent.types import AgentContext
from pi_ai import (
    TextContent,
    Type,
    UserMessage,
    create_models,
    faux_assistant_message,
    faux_provider,
    faux_tool_call,
)
from pi_ai.telemetry import AI_TELEMETRY_SCHEMAS
from pi_telemetry import InMemoryTelemetryContext, RecordedTelemetrySpan
from pi_telemetry.testing import assert_spans_match_schemas

SCHEMAS = (*AGENT_TELEMETRY_SCHEMAS, *AI_TELEMETRY_SCHEMAS)


def _setup(responses: list[Any]) -> tuple[Any, Any]:
    faux = faux_provider()
    faux.set_responses(responses)
    models = create_models()
    models.set_provider(faux.provider)
    return faux, models


async def _read(_id: str, params: Any, _signal: Any, _update: Any) -> AgentToolResult:
    return AgentToolResult([TextContent(f"SECRET-FILE contents of {params['path']}")])


READ = AgentTool("read", "Read a file", Type.Object({"path": Type.String()}), execute=_read)


def _dump(spans: list[RecordedTelemetrySpan]) -> str:
    return json.dumps([[span.attributes, [[e.name, e.attributes] for e in span.events]] for span in spans])


async def test_agent_run_span_with_child_request_spans() -> None:
    faux, models = _setup(
        [
            faux_assistant_message([faux_tool_call("read", {"path": "SECRET-PATH"})], stop_reason="toolUse"),
            faux_assistant_message("SECRET-ANSWER done"),
        ]
    )
    telemetry = InMemoryTelemetryContext()
    agent = Agent(
        initial_state=AgentInitialState(
            system_prompt="SECRET-SYSTEM", model=faux.get_model(), tools=[READ], thinking_level="low"
        ),
        stream_fn=models,
        telemetry_context=telemetry,
    )
    await agent.prompt("SECRET-PROMPT read the file")

    run, first, second = telemetry.get_spans()
    assert run.name == "pi.agent.run" and run.parent_id is None
    assert (first.name, first.parent_id) == ("pi.ai.request", run.id)
    assert (second.name, second.parent_id) == ("pi.ai.request", run.id)
    assert first.attributes["pi.ai.stop_reason"] == "toolUse"
    assert first.attributes["pi.ai.reasoning"] == "low"
    assert second.attributes["pi.ai.message_count"] == 4
    assert run.status == {"status": "ok"}
    assert run.end_sequence is not None and second.end_sequence is not None
    assert second.end_sequence < run.end_sequence
    messages = agent.state.messages
    assistants = [message for message in messages if message.role == "assistant"]
    assert run.attributes == {
        "pi.agent.provider": "faux",
        "pi.agent.model": "faux-1",
        "pi.agent.tool_count": 1,
        "pi.agent.mode": "prompt",
        "pi.agent.reasoning": "low",
        "pi.agent.turns": 2,
        "pi.agent.tool_calls": 1,
        "pi.agent.usage.input_tokens": sum(message.usage.input for message in assistants),
        "pi.agent.usage.output_tokens": sum(message.usage.output for message in assistants),
        "pi.agent.cost": sum(message.usage.cost.total for message in assistants),
        "pi.agent.end_reason": "completed",
    }
    assert_spans_match_schemas(telemetry.get_spans(), SCHEMAS)
    assert "SECRET" not in _dump(telemetry.get_spans())

    # Without a telemetry context nothing is recorded and requests get no telemetry_context.
    seen: list[Any] = []
    faux.set_responses([faux_assistant_message("plain")])

    def stream_fn(model: Any, context: Any, options: Any) -> Any:
        seen.append(dict(options or {}))
        return models.stream_simple(model, context, options)

    quiet = Agent(initial_state=AgentInitialState(model=faux.get_model()), stream_fn=stream_fn)
    await quiet.prompt("hi")
    assert "telemetry_context" not in seen[0]


async def test_steering_follow_up_and_end_reasons() -> None:
    faux, models = _setup([faux_assistant_message("one"), faux_assistant_message("two"), faux_assistant_message("x")])
    telemetry = InMemoryTelemetryContext()
    agent = Agent(
        initial_state=AgentInitialState(model=faux.get_model()), stream_fn=models, telemetry_context=telemetry
    )
    agent.follow_up(UserMessage("later"))
    agent.steer(UserMessage("now"))
    await agent.prompt("start")
    [run, *_requests] = telemetry.get_spans()
    assert [(event.name, event.attributes) for event in run.events] == [
        ("pi.agent.steering", {"pi.agent.message_count": 1}),
        ("pi.agent.follow_up", {"pi.agent.message_count": 1}),
    ]
    assert run.attributes["pi.agent.turns"] == 2
    assert run.attributes["pi.agent.end_reason"] == "completed"

    telemetry2 = InMemoryTelemetryContext()
    faux.set_responses([faux_assistant_message([], stop_reason="error", error_message="SECRET failure")])
    agent2 = Agent(
        initial_state=AgentInitialState(model=faux.get_model()), stream_fn=models, telemetry_context=telemetry2
    )
    await agent2.prompt("start")
    run2 = telemetry2.get_spans()[0]
    assert run2.attributes["pi.agent.end_reason"] == "error"
    assert run2.status == {"status": "error", "error": {"name": "error", "message": ""}}
    assert "SECRET" not in _dump(telemetry2.get_spans())


async def test_low_level_loop_steering_events_and_finish_turn_end() -> None:
    faux, models = _setup([faux_assistant_message("one"), faux_assistant_message("two")])
    telemetry = InMemoryTelemetryContext()
    steering = [[UserMessage("steer 1"), UserMessage("steer 2")], []]

    async def get_steering() -> list[Any]:
        return steering.pop(0) if steering else []

    turns = []

    def finish_turn(turn: Any, _signal: Any) -> Any:
        turns.append(turn)
        return "end"

    config = AgentLoopConfig(
        model=faux.get_model(),
        convert_to_llm=lambda messages: messages,
        get_steering_messages=get_steering,
        finish_turn=finish_turn,
        telemetry_context=telemetry,
    )
    await run_agent_loop([UserMessage("go")], AgentContext(), config, lambda _event: None, None, models.stream_simple)
    run, request = telemetry.get_spans()
    assert [event.name for event in run.events] == ["pi.agent.steering"]
    assert run.events[0].attributes == {"pi.agent.message_count": 2}
    assert run.attributes["pi.agent.end_reason"] == "ended"
    assert request.parent_id == run.id
    assert_spans_match_schemas(telemetry.get_spans(), SCHEMAS)


async def test_a_raising_loop_records_failed_without_the_error_text() -> None:
    faux, models = _setup([faux_assistant_message("one")])
    telemetry = InMemoryTelemetryContext()

    def broken_convert(_messages: list[Any]) -> Any:
        raise RuntimeError("SECRET conversion failure")

    config = AgentLoopConfig(model=faux.get_model(), convert_to_llm=broken_convert, telemetry_context=telemetry)
    try:
        await run_agent_loop(
            [UserMessage("go")], AgentContext(), config, lambda _event: None, None, models.stream_simple
        )
    except RuntimeError:
        pass
    [run] = telemetry.get_spans()
    assert run.status == {"status": "error", "error": {"name": "RuntimeError", "message": ""}}
    assert run.attributes["pi.agent.end_reason"] == "failed"
    assert "SECRET" not in _dump(telemetry.get_spans())
