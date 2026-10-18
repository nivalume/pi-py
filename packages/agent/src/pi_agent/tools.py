"""Build agent tools from plain Python functions.

::

    @agent_tool
    async def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
        \"\"\"Get the current weather.

        Args:
            city: City name.
            units: Temperature units.
        \"\"\"
        return f"Sunny in {city}"

The JSON Schema comes from the signature (``str``, ``int``, ``float``, ``bool``,
``list[...]``, ``dict``, ``Literal``, ``Enum``, ``X | None``, Pydantic models and
dataclasses), descriptions from a Google-style ``Args:`` docstring section.
Parameters named ``tool_call_id``, ``signal`` and ``on_update`` are injected
instead of exposed to the model. Return values are converted: ``str`` becomes a
text block, :class:`AgentToolResult` passes through, other values are sent as JSON.
Synchronous functions run in a worker thread.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import enum
import inspect
import json
import re
import types
import typing
from collections.abc import Callable
from typing import Any, Literal, get_args, get_origin

from pi_ai import ImageContent, TextContent
from pi_ai.utils.schema import is_pydantic_model, parameters_schema

from .types import AgentTool, AgentToolResult, ToolExecutionMode

__all__ = ["agent_tool", "tool_from_function", "schema_from_type"]

_INJECTED = frozenset({"tool_call_id", "signal", "on_update"})


def schema_from_type(annotation: Any) -> dict[str, Any]:
    """JSON Schema for a Python type annotation."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is typing.Annotated:
        schema = schema_from_type(args[0])
        for extra in args[1:]:
            if isinstance(extra, str):
                schema["description"] = extra
            elif isinstance(extra, dict):
                schema.update(extra)
        return schema
    if origin in (typing.Union, types.UnionType):
        members = [arg for arg in args if arg is not type(None)]
        nullable = len(members) != len(args)
        if len(members) == 1:
            schema = schema_from_type(members[0])
        else:
            schema = {"anyOf": [schema_from_type(member) for member in members]}
        if nullable:
            return {"anyOf": [schema, {"type": "null"}]} if schema else schema
        return schema
    if origin is Literal:
        values = list(args)
        if all(isinstance(value, str) for value in values):
            return {"type": "string", "enum": values}
        return {"enum": values}
    # ``get_origin`` reports ``collections.abc`` classes for the ``typing`` aliases (``Sequence[int]``).
    if origin in (list, tuple, set, frozenset, collections.abc.Sequence) or annotation in (list, tuple, set):
        item = args[0] if args else Any
        if origin is tuple and len(args) > 1 and args[1] is not Ellipsis:
            return {"type": "array", "prefixItems": [schema_from_type(arg) for arg in args], "items": False}
        return {"type": "array", "items": schema_from_type(item)}
    if origin in (dict, collections.abc.Mapping) or annotation is dict:
        value = args[1] if len(args) == 2 else Any
        schema: dict[str, Any] = {"type": "object"}
        value_schema = schema_from_type(value)
        if value_schema:
            schema["additionalProperties"] = value_schema
        return schema
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is type(None):
        return {"type": "null"}
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [member.value for member in annotation]
        if all(isinstance(value, str) for value in values):
            return {"type": "string", "enum": values}
        return {"enum": values}
    if is_pydantic_model(annotation):
        return dict(parameters_schema(annotation))
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        hints = typing.get_type_hints(annotation)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for item in dataclasses.fields(annotation):
            properties[item.name] = schema_from_type(hints.get(item.name, Any))
            if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
                required.append(item.name)
        result: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            result["required"] = required
        return result
    return {}


_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters):\s*$")
_SECTION_HEADER = re.compile(r"^\s*[A-Z][A-Za-z ]+:\s*$")
_ARG_LINE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}
    index = 0
    while index < len(lines) and not _ARGS_HEADER.match(lines[index]):
        if _SECTION_HEADER.match(lines[index]):
            break
        summary.append(lines[index])
        index += 1
    current: str | None = None
    if index < len(lines) and _ARGS_HEADER.match(lines[index]):
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.strip() and not line.startswith((" ", "\t")) and _SECTION_HEADER.match(line):
                break
            match = _ARG_LINE.match(line)
            if match and (line.startswith((" ", "\t")) or current is None):
                name = match.group(1).lstrip("*")
                params[name] = match.group(3).strip()
                current = name
            elif current is not None and line.strip():
                params[current] = f"{params[current]} {line.strip()}".strip()
            index += 1
    return "\n".join(summary).strip(), params


def _convert_argument(annotation: Any, value: Any) -> Any:
    """Convert a validated JSON value to its annotated type (enums, Pydantic models, dataclasses).

    The schema advertises these types, so the function receives instances, not raw JSON.
    """
    if value is None or annotation is inspect.Parameter.empty or annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is typing.Annotated:
        return _convert_argument(args[0], value)
    if origin in (typing.Union, types.UnionType):
        members = [arg for arg in args if arg is not type(None)]
        return _convert_argument(members[0], value) if len(members) == 1 else value
    if origin in (list, tuple, set, frozenset, collections.abc.Sequence) and isinstance(value, list):
        if origin is tuple and len(args) > 1 and args[1] is not Ellipsis:
            return tuple(_convert_argument(arg, item) for arg, item in zip(args, value, strict=False))
        items = [_convert_argument(args[0], item) for item in value] if args else list(value)
        return origin(items) if origin in (tuple, set, frozenset) else items
    if origin in (dict, collections.abc.Mapping) and isinstance(value, dict) and len(args) == 2:
        return {key: _convert_argument(args[1], item) for key, item in value.items()}
    if not isinstance(annotation, type):
        return value
    if issubclass(annotation, enum.Enum):
        return value if isinstance(value, annotation) else annotation(value)
    if is_pydantic_model(annotation) and isinstance(value, dict):
        return annotation.model_validate(value)
    if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
        hints = typing.get_type_hints(annotation)
        return annotation(**{key: _convert_argument(hints.get(key, Any), item) for key, item in value.items()})
    return value


def _to_result(value: Any) -> AgentToolResult:
    if isinstance(value, AgentToolResult):
        return value
    if value is None:
        return AgentToolResult([], None)
    if isinstance(value, str):
        return AgentToolResult([TextContent(value)], None)
    if isinstance(value, (TextContent, ImageContent)):
        return AgentToolResult([value], None)
    if isinstance(value, list) and value and all(isinstance(item, (TextContent, ImageContent)) for item in value):
        return AgentToolResult(list(value), None)
    from pi_ai.serialization import to_json_value

    data = to_json_value(value)
    return AgentToolResult([TextContent(json.dumps(data, ensure_ascii=False))], data)


def tool_from_function(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    label: str | None = None,
    parameters: Any = None,
    execution_mode: ToolExecutionMode | None = None,
) -> AgentTool:
    """Create an :class:`AgentTool` from a function (see the module docstring)."""
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    summary, param_docs = _parse_docstring(fn.__doc__)
    exposed = [param for param in signature.parameters.values() if param.name not in _INJECTED]
    injected = {param.name for param in signature.parameters.values() if param.name in _INJECTED}

    single_model = (
        parameters is None
        and len(exposed) == 1
        and is_pydantic_model(hints.get(exposed[0].name))
        and exposed[0].kind is not inspect.Parameter.VAR_KEYWORD
    )
    if parameters is None and not single_model:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in exposed:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            schema = schema_from_type(hints.get(param.name, param.annotation))
            if param.name in param_docs and "description" not in schema:
                schema["description"] = param_docs[param.name]
            if param.default is inspect.Parameter.empty:
                required.append(param.name)
            elif param.default is not None and isinstance(param.default, (str, int, float, bool)):
                schema.setdefault(
                    "default", param.default.value if isinstance(param.default, enum.Enum) else param.default
                )
            properties[param.name] = schema
        parameters = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
    elif single_model:
        parameters = hints[exposed[0].name]

    is_async = inspect.iscoroutinefunction(fn)
    model_param = exposed[0].name if single_model else None

    async def execute(tool_call_id: str, params: Any, signal: Any = None, on_update: Any = None) -> AgentToolResult:
        if model_param is not None:
            kwargs: dict[str, Any] = {model_param: params}
        elif isinstance(params, dict):
            kwargs = dict(params)
            for param in exposed:
                if param.name in kwargs:
                    kwargs[param.name] = _convert_argument(hints.get(param.name, Any), kwargs[param.name])
        else:
            kwargs = {"params": params}
        if "tool_call_id" in injected:
            kwargs["tool_call_id"] = tool_call_id
        if "signal" in injected:
            kwargs["signal"] = signal
        if "on_update" in injected:
            kwargs["on_update"] = on_update
        if is_async:
            value = await fn(**kwargs)
        else:
            value = await asyncio.to_thread(fn, **kwargs)
        return _to_result(value)

    tool_name = name or fn.__name__
    return AgentTool(
        tool_name,
        description if description is not None else summary,
        parameters,
        execute=execute,
        label=label or tool_name,
        execution_mode=execution_mode,
    )


@typing.overload
def agent_tool(fn: Callable[..., Any], /) -> AgentTool: ...


@typing.overload
def agent_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    label: str | None = None,
    parameters: Any = None,
    execution_mode: ToolExecutionMode | None = None,
) -> Callable[[Callable[..., Any]], AgentTool]: ...


def agent_tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    label: str | None = None,
    parameters: Any = None,
    execution_mode: ToolExecutionMode | None = None,
) -> Any:
    """Decorator form of :func:`tool_from_function` (usable with or without arguments)."""

    def wrap(function: Callable[..., Any]) -> AgentTool:
        return tool_from_function(
            function,
            name=name,
            description=description,
            label=label,
            parameters=parameters,
            execution_mode=execution_mode,
        )

    return wrap(fn) if fn is not None else wrap
