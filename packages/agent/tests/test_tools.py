"""Tests for the Python-only ``pi_agent.tools`` helpers (``agent_tool`` / ``tool_from_function``)."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import threading
from typing import Annotated, Any, Literal, Optional

import pydantic
import pytest
from agent_helpers import create_assistant_message

from pi_agent import (
    AgentContext,
    AgentTool,
    AgentToolResult,
    RunToolCallOptions,
    agent_tool,
    run_tool_call,
    schema_from_type,
    tool_from_function,
)
from pi_ai import AbortSignal, ImageContent, TextContent, ToolCall, to_tool_declaration


class Units(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class Priority(enum.Enum):
    LOW = 1
    HIGH = 2


class Location(pydantic.BaseModel):
    city: str
    country: str = "NL"


class Forecast(pydantic.BaseModel):
    location: Location
    days: int = pydantic.Field(ge=1, le=14, description="Number of days")


@dataclasses.dataclass
class Point:
    x: int
    y: int = 0


async def run(tool: AgentTool, arguments: dict[str, Any], **options: Any) -> Any:
    return await run_tool_call(
        ToolCall("call-1", tool.name, arguments),
        RunToolCallOptions(
            tools=[tool],
            assistant_message=create_assistant_message([]),
            context=AgentContext(messages=[]),
            **options,
        ),
    )


# ---------------------------------------------------------------------------
# schema_from_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (type(None), {"type": "null"}),
        (Any, {}),
        (list[str], {"type": "array", "items": {"type": "string"}}),
        (list, {"type": "array", "items": {}}),
        (set[int], {"type": "array", "items": {"type": "integer"}}),
        (tuple[int, ...], {"type": "array", "items": {"type": "integer"}}),
        (
            tuple[int, str],
            {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": False},
        ),
        (dict[str, int], {"type": "object", "additionalProperties": {"type": "integer"}}),
        (dict, {"type": "object"}),
        (Literal["a", "b"], {"type": "string", "enum": ["a", "b"]}),
        (Literal[1, 2], {"enum": [1, 2]}),
        (Optional[int], {"anyOf": [{"type": "integer"}, {"type": "null"}]}),  # noqa: UP045
        (int | None, {"anyOf": [{"type": "integer"}, {"type": "null"}]}),
        (int | str, {"anyOf": [{"type": "integer"}, {"type": "string"}]}),
        (Units, {"type": "string", "enum": ["celsius", "fahrenheit"]}),
        (Priority, {"enum": [1, 2]}),
        (Annotated[str, "A city"], {"type": "string", "description": "A city"}),
        (Annotated[int, {"minimum": 1}], {"type": "integer", "minimum": 1}),
        (
            Point,
            {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x"]},
        ),
    ],
)
def test_schema_from_type(annotation: Any, expected: dict[str, Any]) -> None:
    assert schema_from_type(annotation) == expected


def test_schema_from_type_inlines_pydantic_models() -> None:
    schema = schema_from_type(Forecast)
    assert schema["type"] == "object"
    assert schema["required"] == ["location", "days"]
    # Nested models are inlined (no $ref/$defs) and titles dropped.
    assert "$defs" not in schema
    assert schema["properties"]["location"]["properties"]["city"] == {"type": "string"}
    assert schema["properties"]["days"]["description"] == "Number of days"
    assert "title" not in schema["properties"]["days"]


# ---------------------------------------------------------------------------
# tool_from_function / agent_tool: schema and metadata
# ---------------------------------------------------------------------------


def test_builds_the_schema_from_type_hints_and_defaults() -> None:
    async def search(
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        mode: Literal["fast", "exact"] = "fast",
        units: Units = Units.CELSIUS,
        threshold: float = 0.5,
        strict: bool = False,
        tool_call_id: str = "",
        signal: AbortSignal | None = None,
        on_update: Any = None,
    ) -> str:
        return query

    tool = tool_from_function(search)

    assert isinstance(tool, AgentTool)
    assert tool.name == "search"
    assert tool.label == "search"
    assert tool.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
            "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            "mode": {"type": "string", "enum": ["fast", "exact"], "default": "fast"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "threshold": {"type": "number", "default": 0.5},
            "strict": {"type": "boolean", "default": False},
        },
        "required": ["query"],
    }
    # Injected parameters are not exposed to the model.
    for injected in ("tool_call_id", "signal", "on_update"):
        assert injected not in tool.parameters["properties"]


def test_parses_google_style_docstrings() -> None:
    @agent_tool
    def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius", days: int = 1) -> str:
        """Get the current weather.

        Uses the default provider.

        Args:
            city: City name, e.g. "Amsterdam".
            units (str): Temperature units.
            days: Number of days to forecast,
                counted from today.

        Returns:
            A short description.
        """
        return city

    assert get_weather.description == "Get the current weather.\n\nUses the default provider."
    properties = get_weather.parameters["properties"]
    assert properties["city"]["description"] == 'City name, e.g. "Amsterdam".'
    assert properties["units"]["description"] == "Temperature units."
    assert properties["days"]["description"] == "Number of days to forecast, counted from today."
    assert get_weather.parameters["required"] == ["city"]


def test_annotated_descriptions_win_over_docstrings() -> None:
    def lookup(key: Annotated[str, "The lookup key"]) -> str:
        """Look something up.

        Args:
            key: Docstring description.
        """
        return key

    tool = tool_from_function(lookup)
    assert tool.parameters["properties"]["key"] == {"type": "string", "description": "The lookup key"}


def test_decorator_options_override_metadata() -> None:
    @agent_tool(name="renamed", description="Custom description", label="Pretty", execution_mode="sequential")
    async def original(value: str) -> str:
        """Ignored summary."""
        return value

    assert original.name == "renamed"
    assert original.description == "Custom description"
    assert original.label == "Pretty"
    assert original.execution_mode == "sequential"


def test_explicit_parameters_are_used_verbatim() -> None:
    parameters = {"type": "object", "properties": {"raw": {"type": "string"}}, "required": ["raw"]}

    @agent_tool(parameters=parameters)
    def raw_tool(raw: str) -> str:
        return raw

    assert raw_tool.parameters is parameters


def test_tool_declaration_strips_the_executable() -> None:
    @agent_tool
    async def ping() -> str:
        """Ping."""
        return "pong"

    declaration = to_tool_declaration(ping)
    assert type(declaration) is not AgentTool
    assert declaration.name == "ping"
    assert declaration.description == "Ping."
    assert declaration.parameters == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def test_async_functions_receive_validated_and_converted_arguments() -> None:
    received: list[Any] = []

    @agent_tool
    async def configure(count: int, units: Units, mode: Literal["a", "b"] = "a") -> str:
        received.append((count, units, mode))
        return f"{count} {units.value} {mode}"

    # "3" is coerced to an integer by schema validation; the enum value becomes an Enum member.
    outcome = await run(configure, {"count": "3", "units": "fahrenheit"})

    assert outcome.is_error is False
    assert received == [(3, Units.FAHRENHEIT, "a")]
    assert outcome.result.content == [TextContent("3 fahrenheit a")]


async def test_sync_functions_run_in_a_worker_thread() -> None:
    main_thread = threading.get_ident()
    seen_threads: list[int] = []

    @agent_tool
    def blocking(value: str) -> str:
        seen_threads.append(threading.get_ident())
        return value.upper()

    outcome = await run(blocking, {"value": "hi"})

    assert outcome.result.content == [TextContent("HI")]
    assert seen_threads and seen_threads[0] != main_thread


async def test_sync_functions_do_not_block_the_event_loop() -> None:
    import time

    @agent_tool
    def slow() -> str:
        time.sleep(0.05)
        return "done"

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.005)
            ticks += 1

    outcome, _ = await asyncio.gather(run(slow, {}), ticker())
    assert outcome.result.content == [TextContent("done")]
    assert ticks == 5


async def test_injects_tool_call_id_signal_and_on_update() -> None:
    seen: dict[str, Any] = {}
    updates: list[Any] = []

    @agent_tool
    async def progress(steps: int, tool_call_id: str, signal: AbortSignal | None, on_update: Any) -> str:
        seen["tool_call_id"] = tool_call_id
        seen["signal"] = signal
        for step in range(steps):
            on_update(AgentToolResult([TextContent(f"step {step}")], None))
        return "finished"

    signal = AbortSignal()
    outcome = await run(progress, {"steps": 2}, signal=signal, on_update=updates.append)

    assert outcome.result.content == [TextContent("finished")]
    assert seen == {"tool_call_id": "call-1", "signal": signal}
    assert [update.content[0].text for update in updates] == ["step 0", "step 1"]


async def test_a_single_pydantic_parameter_becomes_the_schema_and_receives_an_instance() -> None:
    received: list[Any] = []

    @agent_tool
    async def forecast(request: Forecast) -> str:
        """Forecast the weather."""
        received.append(request)
        return f"{request.location.city}: {request.days} days"

    assert forecast.parameters is Forecast

    outcome = await run(forecast, {"location": {"city": "Utrecht"}, "days": 3})
    assert outcome.is_error is False
    assert outcome.result.content == [TextContent("Utrecht: 3 days")]
    assert isinstance(received[0], Forecast)
    assert received[0].location == Location(city="Utrecht", country="NL")

    invalid = await run(forecast, {"location": {"city": "Utrecht"}, "days": 30})
    assert invalid.is_error is True
    assert "days" in invalid.result.content[0].text
    assert len(received) == 1


async def test_pydantic_and_dataclass_parameters_next_to_other_parameters_receive_instances() -> None:
    received: list[Any] = []

    @agent_tool
    async def place(location: Location, point: Point, label: str = "") -> str:
        received.append((location, point, label))
        return "ok"

    properties = place.parameters["properties"]
    assert properties["location"]["properties"]["city"] == {"type": "string"}
    assert properties["point"]["required"] == ["x"]

    outcome = await run(place, {"location": {"city": "Delft"}, "point": {"x": 1}})
    assert outcome.is_error is False
    assert received == [(Location(city="Delft"), Point(1, 0), "")]


async def test_optional_enum_parameters_are_converted() -> None:
    received: list[Any] = []

    @agent_tool
    async def pick(priority: Priority | None = None) -> str:
        received.append(priority)
        return "ok"

    await run(pick, {"priority": 2})
    await run(pick, {})
    assert received == [Priority.HIGH, None]


async def test_validation_errors_do_not_call_the_function() -> None:
    calls = 0

    @agent_tool
    async def strict(count: int) -> str:
        nonlocal calls
        calls += 1
        return "ok"

    outcome = await run(strict, {"count": "many"})
    assert outcome.is_error is True
    assert 'Validation failed for tool "strict"' in outcome.result.content[0].text
    assert calls == 0


async def test_exceptions_become_error_results() -> None:
    @agent_tool
    def broken() -> str:
        raise ValueError("kaputt")

    outcome = await run(broken, {})
    assert outcome.is_error is True
    assert outcome.result.content == [TextContent("kaputt")]


@pytest.mark.parametrize(
    ("value", "content", "details"),
    [
        ("text", [TextContent("text")], None),
        (None, [], None),
        (TextContent("block"), [TextContent("block")], None),
        (
            [TextContent("a"), ImageContent("aGk=", "image/png")],
            [TextContent("a"), ImageContent("aGk=", "image/png")],
            None,
        ),
        ({"answer": 42, "unit": "ü"}, [TextContent('{"answer": 42, "unit": "ü"}')], {"answer": 42, "unit": "ü"}),
        ([1, 2, 3], [TextContent("[1, 2, 3]")], [1, 2, 3]),
        (7, [TextContent("7")], 7),
    ],
    ids=["str", "none", "block", "blocks", "dict", "list", "int"],
)
async def test_return_values_are_converted(value: Any, content: list[Any], details: Any) -> None:
    @agent_tool
    async def produce() -> Any:
        return value

    outcome = await run(produce, {})
    assert outcome.is_error is False
    assert outcome.result.content == content
    assert outcome.result.details == details


async def test_agent_tool_results_pass_through() -> None:
    result = AgentToolResult([TextContent("custom")], {"k": 1}, structured_content={"v": 2}, terminate=True)

    @agent_tool
    async def passthrough() -> AgentToolResult:
        return result

    outcome = await run(passthrough, {})
    assert outcome.result.content == [TextContent("custom")]
    assert outcome.result.details == {"k": 1}
    assert outcome.result.structured_content == {"v": 2}
    assert outcome.result.terminate is True


async def test_dataclass_return_values_are_serialized_as_json() -> None:
    @agent_tool
    def where() -> Point:
        return Point(3, 4)

    outcome = await run(where, {})
    assert outcome.result.content == [TextContent('{"x": 3, "y": 4}')]
    assert outcome.result.details == {"x": 3, "y": 4}
