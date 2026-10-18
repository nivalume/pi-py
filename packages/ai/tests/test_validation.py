from __future__ import annotations

from typing import Any

import pytest

from pi_ai import Tool, ToolCall, ToolValidationError, Type, validate_tool_arguments, validate_tool_call


def _plain(schema: dict[str, Any], value: Any) -> tuple[Tool, ToolCall]:
    tool = Tool(
        "echo",
        "Echo tool",
        {"type": "object", "properties": {"value": schema}, "required": ["value"]},
    )
    return tool, ToolCall("tool-1", "echo", {"value": value})


def test_coerces_numeric_strings_for_builder_schemas() -> None:
    tool = Tool("echo", "Echo tool", Type.Object({"count": Type.Number()}))
    assert validate_tool_arguments(tool, ToolCall("tool-1", "echo", {"count": "42"})) == {"count": 42}


@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"type": "number"}, "42", 42),
        ({"type": "number"}, True, 1),
        ({"type": "number"}, None, 0),
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"type": "boolean"}, "false", False),
        ({"type": "boolean"}, 1, True),
        ({"type": "boolean"}, 0, False),
        ({"type": "string"}, None, ""),
        ({"type": "string"}, True, "true"),
        ({"type": "null"}, "", None),
        ({"type": "null"}, 0, None),
        ({"type": "null"}, False, None),
        ({"type": ["number", "string"]}, "1", "1"),
        ({"type": ["boolean", "number"]}, "1", 1),
    ],
)
def test_coerces_plain_json_schemas_with_ajv_compatible_rules(
    schema: dict[str, Any], value: Any, expected: Any
) -> None:
    tool, call = _plain(schema, value)
    result = validate_tool_arguments(tool, call)
    assert result == {"value": expected}
    assert type(result["value"]) is type(expected)


def test_treats_null_as_omission_for_optional_non_nullable_properties() -> None:
    tool = Tool(
        "echo",
        "Echo tool",
        Type.Object(
            {
                "path": Type.String(),
                "offset": Type.Optional(Type.Number()),
                "nullable": Type.Optional(Type.Union([Type.String(), Type.Null()])),
                "metadata": Type.Object({"enabled": Type.Optional(Type.Boolean())}),
            }
        ),
    )
    call = ToolCall(
        "tool-1", "echo", {"path": "file.txt", "offset": None, "nullable": None, "metadata": {"enabled": None}}
    )
    assert validate_tool_arguments(tool, call) == {"path": "file.txt", "nullable": None, "metadata": {}}


def test_preserves_optional_nulls_whose_referenced_schema_is_nullable() -> None:
    tool = Tool(
        "echo",
        "Echo tool",
        {
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
        },
    )
    assert validate_tool_arguments(tool, ToolCall("tool-1", "echo", {"value": None})) == {"value": None}


def test_preserves_value_matching_nullable_union_arm() -> None:
    tool = Tool("echo", "Echo tool", Type.Object({"value": Type.Union([Type.Number(), Type.Null()])}))
    assert validate_tool_arguments(tool, ToolCall("tool-1", "echo", {"value": None})) == {"value": None}


def test_preserves_value_matching_one_of_nullable_arm() -> None:
    tool, call = _plain({"oneOf": [{"type": "number"}, {"type": "null"}]}, None)
    assert validate_tool_arguments(tool, call) == {"value": None}


def test_still_coerces_nullable_unions_when_no_arm_matches() -> None:
    tool, call = _plain({"anyOf": [{"type": "number"}, {"type": "null"}]}, "42")
    assert validate_tool_arguments(tool, call) == {"value": 42}


def test_accepts_null_for_nullable_array_schema() -> None:
    tool, call = _plain({"type": ["array", "null"], "items": {"type": "string"}}, None)
    assert validate_tool_arguments(tool, call) == {"value": None}


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "boolean"}, "1"),
        ({"type": "boolean"}, "0"),
        ({"type": "null"}, "null"),
        ({"type": "integer"}, "42.1"),
    ],
)
def test_rejects_invalid_coercions(schema: dict[str, Any], value: Any) -> None:
    tool, call = _plain(schema, value)
    with pytest.raises(ToolValidationError, match="Validation failed"):
        validate_tool_arguments(tool, call)


# #10897
def test_reports_malformed_arguments_json() -> None:
    tool = Tool("echo", "Echo tool", Type.Object({"value": Type.Optional(Type.String())}))
    call = ToolCall("tool-1", "echo", {}, arguments_error="Unexpected token")
    with pytest.raises(ToolValidationError, match='Invalid arguments for tool "echo": Unexpected token'):
        validate_tool_arguments(tool, call)


def test_error_message_lists_every_problem_with_paths() -> None:
    tool = Tool(
        "book",
        "",
        Type.Object(
            {
                "title": Type.String(min_length=3),
                "attendees": Type.Array(Type.String(), min_items=1),
                "when": Type.Object({"start": Type.String()}),
            }
        ),
    )
    call = ToolCall("1", "book", {"title": "ab", "attendees": [], "when": {}})
    with pytest.raises(ToolValidationError) as raised:
        validate_tool_arguments(tool, call)
    message = str(raised.value)
    assert 'Validation failed for tool "book":' in message
    assert "  - title: must NOT have fewer than 3 characters" in message
    assert "  - attendees: must NOT have fewer than 1 items" in message
    assert "  - when.start: must have required property 'start'" in message
    assert message.endswith('"when": {}\n}')


def test_validate_tool_call_unknown_tool() -> None:
    with pytest.raises(ToolValidationError, match='Tool "missing" not found'):
        validate_tool_call([], ToolCall("1", "missing", {}))


def test_schema_keywords() -> None:
    schema = Type.Object(
        {
            "mode": Type.Enum(["a", "b"]),
            "count": Type.Integer(minimum=1, maximum=5),
            "tags": Type.Array(Type.String(pattern="^[a-z]+$"), unique_items=True),
        },
        additional_properties=False,
    )
    tool = Tool("t", "", schema)
    assert validate_tool_arguments(tool, ToolCall("1", "t", {"mode": "a", "count": 2, "tags": ["x"]})) == {
        "mode": "a",
        "count": 2,
        "tags": ["x"],
    }
    for bad in (
        {"mode": "c", "count": 2, "tags": []},
        {"mode": "a", "count": 9, "tags": []},
        {"mode": "a", "count": 2, "tags": ["X"]},
        {"mode": "a", "count": 2, "tags": ["x", "x"]},
        {"mode": "a", "count": 2, "tags": [], "extra": 1},
    ):
        with pytest.raises(ToolValidationError):
            validate_tool_arguments(tool, ToolCall("1", "t", bad))


def test_pydantic_parameters_return_model_instances() -> None:
    pydantic = pytest.importorskip("pydantic")

    class Weather(pydantic.BaseModel):
        location: str
        days: int = 1

    tool = Tool("weather", "", Weather)
    result = validate_tool_arguments(tool, ToolCall("1", "weather", {"location": "Paris", "days": "3"}))
    assert isinstance(result, Weather)
    assert result.days == 3
    with pytest.raises(ToolValidationError, match="location"):
        validate_tool_arguments(tool, ToolCall("1", "weather", {"days": 2}))
