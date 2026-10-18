"""Provider-side constrained sampling: strict JSON-schema tools and OpenAI grammar tools."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..types import Tool
from ..utils.schema import parameters_schema

__all__ = [
    "UnsupportedStrictJsonSchemaError",
    "UnsupportedStrictSchemaKeywordCheck",
    "make_strict_json_schema",
    "get_json_schema_tool_parameters",
    "GrammarConstrainedSampling",
    "GrammarToolInputJsonBuffer",
    "get_grammar_tool_input",
    "append_grammar_tool_input_json_delta",
    "resolve_json_schema_strict_sampling",
    "resolve_grammar_constrained_sampling",
    "create_grammar_tool_input_properties",
]


class UnsupportedStrictJsonSchemaError(Exception):
    pass


#: Returns True when a provider's strict mode rejects this schema keyword with this value.
UnsupportedStrictSchemaKeywordCheck = Callable[[str, Any], bool]

_UNSUPPORTED_STRICT_SCHEMA_KEYS = (
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
)


def _is_schema_object(value: Any) -> bool:
    return isinstance(value, dict)


def _is_structured_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    raw = schema.get("type")
    types = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    return "object" in types or "array" in types or "properties" in schema or "items" in schema


def _schema_allows_null(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    raw = schema.get("type")
    if raw == "null" or (isinstance(raw, list) and "null" in raw):
        return True
    if "const" in schema and schema["const"] is None:
        return True
    if isinstance(schema.get("enum"), list) and None in schema["enum"]:
        return True
    any_of = schema.get("anyOf")
    return isinstance(any_of, list) and any(_schema_allows_null(variant) for variant in any_of)


def _make_node_strict(schema: Any, is_unsupported: UnsupportedStrictSchemaKeywordCheck | None) -> None:
    if not isinstance(schema, dict):
        raise UnsupportedStrictJsonSchemaError("boolean schemas are unsupported")
    for key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if key in schema:
            raise UnsupportedStrictJsonSchemaError(f"{key} schemas are unsupported")
    if is_unsupported is not None:
        for key, value in schema.items():
            if is_unsupported(key, value):
                raise UnsupportedStrictJsonSchemaError(f"{key}: {json.dumps(value)} is unsupported")

    if "anyOf" in schema:
        any_of = schema["anyOf"]
        if not isinstance(any_of, list) or not any_of:
            raise UnsupportedStrictJsonSchemaError("anyOf must contain at least one schema")
        for variant in any_of:
            if _is_structured_schema(variant):
                raise UnsupportedStrictJsonSchemaError("object and array unions are unsupported")
            _make_node_strict(variant, is_unsupported)

    if "items" in schema:
        if isinstance(schema["items"], list):
            raise UnsupportedStrictJsonSchemaError("tuple schemas are unsupported")
        _make_node_strict(schema["items"], is_unsupported)

    is_object = schema.get("type") == "object"
    if "properties" in schema and not is_object:
        raise UnsupportedStrictJsonSchemaError("properties require type object")
    if not is_object:
        return
    if "additionalProperties" in schema and schema["additionalProperties"] is not False:
        raise UnsupportedStrictJsonSchemaError("schema-valued or true additionalProperties is unsupported")
    if "properties" in schema and not isinstance(schema["properties"], dict):
        raise UnsupportedStrictJsonSchemaError("object properties must be a schema map")
    required_raw = schema.get("required")
    if required_raw is not None and (
        not isinstance(required_raw, list) or any(not isinstance(key, str) for key in required_raw)
    ):
        raise UnsupportedStrictJsonSchemaError("object required must be a string array")

    properties: dict[str, Any] = schema.get("properties") or {}
    names = list(properties)
    required = set(required_raw or [])
    if any(key not in properties for key in required):
        raise UnsupportedStrictJsonSchemaError("required contains an unknown property")
    for key, prop in list(properties.items()):
        _make_node_strict(prop, is_unsupported)
        if key not in required and not _schema_allows_null(prop):
            properties[key] = {"anyOf": [prop, {"type": "null"}]}
    if "properties" in schema:
        schema["properties"] = properties
    schema["required"] = names
    schema["additionalProperties"] = False


def make_strict_json_schema(
    schema: Any, is_unsupported: UnsupportedStrictSchemaKeywordCheck | None = None
) -> dict[str, Any]:
    """Convert a tool schema to the strict subset expected by provider constrained sampling."""
    cloned = copy.deepcopy(parameters_schema(schema)) if not isinstance(schema, dict) else copy.deepcopy(dict(schema))
    cloned = json.loads(json.dumps(cloned))
    if not isinstance(cloned, dict):
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    _make_node_strict(cloned, is_unsupported)
    if cloned.get("type") != "object":
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    return cloned


def get_json_schema_tool_parameters(tool: Tool, strict: bool | None) -> dict[str, Any]:
    """Tool parameters as sent to the provider (made strict when ``strict`` is True)."""
    return make_strict_json_schema(tool.parameters) if strict is True else parameters_schema(tool.parameters)


@dataclass(slots=True)
class GrammarConstrainedSampling:
    format: Literal["lark", "regex"]
    definition: str
    input_property: str


@dataclass(slots=True)
class GrammarToolInputJsonBuffer:
    input: str = ""
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: dict[str, Any], input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer, input_property: str, next_input: str, close: bool
) -> str | None:
    """Translate a grammar tool's raw input growth into a JSON-arguments delta."""
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not next_input.startswith(buffer.input):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')
    input_delta = next_input[len(buffer.input) :]
    if not close and not input_delta:
        return None
    delta = ""
    if not buffer.started:
        delta += "{" + json.dumps(input_property) + ':"'
        buffer.started = True
    delta += json.dumps(input_delta, ensure_ascii=False)[1:-1]
    buffer.input = next_input
    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = parameters_schema(tool.parameters)
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")
    name = required[0]
    properties = schema.get("properties") or {}
    if name not in properties:
        raise ValueError(f"grammar constrained sampling requires a properties entry for {name}")
    if (properties[name] or {}).get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {name} must have type string")
    return name


def resolve_json_schema_strict_sampling(
    tool: Tool, supports_strict_mode: bool, is_unsupported: UnsupportedStrictSchemaKeywordCheck | None = None
) -> bool | None:
    """Whether a JSON-schema tool is sent in strict mode (``None`` = not strict).

    Raises ``ValueError`` when the tool requires strict sampling the provider/schema cannot honor.
    """
    config = tool.constrained_sampling
    if not config or config.get("type") != "json_schema":
        return None
    if supports_strict_mode:
        try:
            make_strict_json_schema(tool.parameters, is_unsupported)
            return True
        except UnsupportedStrictJsonSchemaError as error:
            if config.get("strict") != "require":
                return None
            raise ValueError(f'Tool "{tool.name}" requires JSON-schema constrained sampling, but {error}.') from error
    if config.get("strict") == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool, supports_openai_grammar_tools: bool
) -> GrammarConstrainedSampling | None:
    config = tool.constrained_sampling
    if not config or config.get("type") != "grammar":
        return None
    if not supports_openai_grammar_tools:
        return None
    variants = config.get("variants") or {}
    lark = variants.get("openai_lark")
    regex = variants.get("openai_regex")
    has_lark = isinstance(lark, str) and bool(lark.strip())
    has_regex = isinstance(regex, str) and bool(regex.strip())
    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.'
        )
    try:
        return GrammarConstrainedSampling(
            format="lark" if has_lark else "regex",
            definition=lark if has_lark else regex,  # type: ignore[arg-type]
            input_property=_infer_grammar_input_property(tool),
        )
    except ValueError as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.') from error


def create_grammar_tool_input_properties(
    tools: Sequence[Tool] | None, supports_openai_grammar_tools: bool
) -> dict[str, str]:
    """Map grammar tool names to their single string input property."""
    properties: dict[str, str] = {}
    for tool in tools or ():
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            properties[tool.name] = grammar.input_property
    return properties
