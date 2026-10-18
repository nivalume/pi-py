"""Conversion between pi-ai data classes and pi's JSON representation.

The TypeScript implementation stores contexts, messages and models as plain JSON
objects with camelCase keys. These helpers produce and read exactly that format,
so a session written by the Python port can be resumed by TypeScript pi and vice
versa::

    from pi_ai.serialization import context_to_dict, context_from_dict, dumps, loads

    text = dumps(context)            # JSON string in pi's format
    restored = loads(text, Context)  # back to data classes

Free-form JSON values (tool arguments, tool result ``details``, headers, model
``compat`` settings, ``metadata``) are passed through untouched.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from . import types as t
from .utils.diagnostics import AssistantMessageDiagnostic, DiagnosticErrorInfo

__all__ = [
    "to_dict",
    "to_json_value",
    "from_dict",
    "dumps",
    "loads",
    "message_from_dict",
    "messages_from_list",
    "content_from_dict",
    "tool_from_dict",
    "model_from_dict",
    "event_from_dict",
    "context_from_dict",
    "context_to_dict",
    "camel_case",
    "snake_case",
]

T = TypeVar("T")

_CAMEL_RE = re.compile(r"_([a-z0-9])")
_SNAKE_RE = re.compile(r"(?<=[a-z0-9])([A-Z])")


def camel_case(name: str) -> str:
    """``tool_call_id`` -> ``toolCallId`` (``cache_write_1h`` -> ``cacheWrite1h``)."""
    return _CAMEL_RE.sub(lambda match: match.group(1).upper(), name)


def snake_case(name: str) -> str:
    """``toolCallId`` -> ``tool_call_id`` (``cacheWrite1h`` -> ``cache_write_1h``)."""
    converted = _SNAKE_RE.sub(lambda match: "_" + match.group(1).lower(), name)
    return re.sub(r"(?<=[a-z])(?=\d)", "_", converted)


# Fields whose values are free-form JSON: copied verbatim, keys never converted.
_RAW_FIELDS = frozenset(
    {
        "arguments",
        "details",
        "headers",
        "compat",
        "metadata",
        "sections",
        "thinking_level_map",
        "prompt_cache",
        "sampling_params",
        "sampling_params_by_thinking_level",
        "parameters",
        "constrained_sampling",
        "data",
        "state",
        "criteria",
        "probabilities",
        "env",
    }
)


def _decode_list(decoder: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda values: [decoder(value) for value in values] if isinstance(values, list) else values


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

_field_cache: dict[type, tuple[tuple[str, str], ...]] = {}


def _fields(cls: type) -> tuple[tuple[str, str], ...]:
    cached = _field_cache.get(cls)
    if cached is None:
        cached = tuple((f.name, camel_case(f.name)) for f in dataclasses.fields(cls) if not f.name.startswith("_"))
        _field_cache[cls] = cached
    return cached


def _discriminator(obj: Any) -> tuple[str, str] | None:
    cls = type(obj)
    for key in ("type", "role"):
        value = cls.__dict__.get(key)
        if value is None:
            for base in cls.__mro__[1:]:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
        if isinstance(value, str):
            return (key, value)
    return None


def to_dict(obj: Any) -> Any:
    """Encode a pi-ai data class (or a list/dict of them) as pi JSON with camelCase keys.

    ``None`` fields are omitted, like ``undefined`` properties in TypeScript.
    """
    return to_json_value(obj)


def to_json_value(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_json_value(item) for item in obj]
    if isinstance(obj, Mapping):
        return {key: to_json_value(value) for key, value in obj.items()}
    if isinstance(obj, t.TranscriptContext):
        return {"messages": to_json_value(obj.messages)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        discriminator = _discriminator(obj)
        if discriminator is not None:
            result[discriminator[0]] = discriminator[1]
        for name, key in _fields(type(obj)):
            value = getattr(obj, name)
            if value is None:
                continue
            if name in _RAW_FIELDS:
                if name == "parameters":
                    from .utils.schema import parameters_schema

                    value = parameters_schema(value)
                result[key] = _raw_json(value)
            else:
                result[key] = to_json_value(value)
        return result
    if isinstance(obj, set | frozenset):
        return [to_json_value(item) for item in obj]
    if isinstance(obj, BaseException):
        return str(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):  # pydantic instances in tool details
        return model_dump(mode="json")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _raw_json(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        if isinstance(value, dict):
            return {key: _raw_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_raw_json(item) for item in value]
        return value
    return to_json_value(value)


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """Serialize a data class (context, message, model, ...) to a pi-compatible JSON string."""
    return json.dumps(to_json_value(obj), ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _build(cls: type[T], data: Mapping[str, Any], *, nested: Mapping[str, Callable[[Any], Any]] | None = None) -> T:
    kwargs: dict[str, Any] = {}
    for name, key in _fields(cls):
        if key in data:
            value = data[key]
        elif name in data:
            value = data[name]
        else:
            continue
        if value is None:
            continue
        decoder = nested.get(name) if nested else None
        if decoder is not None:
            value = decoder(value)
        kwargs[name] = value
    return cls(**kwargs)


def _usage_cost(data: Any) -> t.UsageCost:
    return data if isinstance(data, t.UsageCost) else _build(t.UsageCost, data)


def _usage(data: Any) -> t.Usage:
    if isinstance(data, t.Usage):
        return data
    return _build(t.Usage, data, nested={"cost": _usage_cost})


_CONTENT_TYPES: dict[str, type] = {
    "text": t.TextContent,
    "thinking": t.ThinkingContent,
    "image": t.ImageContent,
    "toolCall": t.ToolCall,
}


def content_from_dict(data: Any) -> Any:
    """Decode one content block (``{"type": "text", "text": ...}`` etc.)."""
    if not isinstance(data, Mapping):
        return data
    cls = _CONTENT_TYPES.get(data.get("type", ""))
    if cls is None:
        return dict(data)
    return _build(cls, data)


def _content_list(data: Any) -> Any:
    if isinstance(data, str) or data is None:
        return data if data is not None else []
    return [content_from_dict(item) for item in data]


def tool_from_dict(data: Any) -> t.Tool:
    if isinstance(data, t.Tool):
        return data
    return _build(t.Tool, data)


def _tool_reference(data: Any) -> t.ToolReference:
    return data if isinstance(data, t.ToolReference) else t.ToolReference(data["name"])


def _diagnostic(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    return _build(
        AssistantMessageDiagnostic,
        data,
        nested={"error": lambda value: _build(DiagnosticErrorInfo, value) if isinstance(value, Mapping) else value},
    )


def _deferred(data: Any) -> Any:
    return _build(t.DeferredHandle, data) if isinstance(data, Mapping) else data


def _nested_calls(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    return _build(
        t.NestedToolCalls,
        data,
        nested={"calls": _decode_list(lambda item: _build(t.NestedToolCallRecord, item))},
    )


_MESSAGE_NESTED: dict[str, dict[str, Callable[[Any], Any]]] = {
    "system": {
        "content": lambda value: value if isinstance(value, str) else _content_list(value),
        "tools_added": _decode_list(tool_from_dict),
        "tools_removed": _decode_list(_tool_reference),
    },
    "user": {"content": _content_list},
    "assistant": {
        "content": _content_list,
        "usage": _usage,
        "diagnostics": _decode_list(_diagnostic),
        "deferred": _deferred,
    },
    "toolResult": {
        "content": _content_list,
        "usage": _usage,
        "nested_calls": _nested_calls,
    },
}

_MESSAGE_TYPES: dict[str, type] = {
    "system": t.SystemMessage,
    "user": t.UserMessage,
    "assistant": t.AssistantMessage,
    "toolResult": t.ToolResultMessage,
}


def message_from_dict(data: Any) -> Any:
    """Decode a message by its ``role``. Unknown roles (custom agent messages) are returned unchanged."""
    if not isinstance(data, Mapping):
        return data
    role = data.get("role")
    cls = _MESSAGE_TYPES.get(role) if isinstance(role, str) else None
    if cls is None:
        return data
    return _build(cls, data, nested=_MESSAGE_NESTED[role])  # type: ignore[index]


def messages_from_list(data: list[Any]) -> list[Any]:
    return [message_from_dict(item) for item in data]


def _cost(data: Any) -> t.ModelCost:
    if isinstance(data, t.ModelCost):
        return data
    return _build(t.ModelCost, data, nested={"tiers": _decode_list(lambda tier: _build(t.ModelCostTier, tier))})


def _input_limits(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data

    def images(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return _build(
            t.ModelImageInputLimits,
            value,
            nested={"resize": lambda r: _build(t.ModelImageResizeOptions, r) if isinstance(r, Mapping) else r},
        )

    return _build(t.ModelInputLimits, data, nested={"images": images})


_MODEL_NESTED: dict[str, Callable[[Any], Any]] = {"cost": _cost, "input_limits": _input_limits}
_MODEL_TYPES: dict[str, type] = {"chat": t.Model, "image": t.ImageModel, "classifier": t.ClassifierModel}


def model_from_dict(data: Any) -> Any:
    """Decode a catalog entry. A missing ``type`` means a chat model."""
    if not isinstance(data, Mapping):
        return data
    cls = _MODEL_TYPES.get(data.get("type") or "chat")
    if cls is None:
        raise ValueError(f"Unknown model type: {data.get('type')!r}")
    return _build(cls, data, nested=_MODEL_NESTED)


_EVENT_TYPES: dict[str, type] = {
    cls.type: cls  # type: ignore[attr-defined]
    for cls in (
        t.StartEvent,
        t.TextStartEvent,
        t.TextDeltaEvent,
        t.TextEndEvent,
        t.ThinkingStartEvent,
        t.ThinkingDeltaEvent,
        t.ThinkingEndEvent,
        t.ToolCallStartEvent,
        t.ToolCallDeltaEvent,
        t.ToolCallEndEvent,
        t.DoneEvent,
        t.ErrorEvent,
    )
}


def event_from_dict(data: Mapping[str, Any]) -> Any:
    cls = _EVENT_TYPES[data["type"]]
    return _build(
        cls,
        data,
        nested={
            "partial": message_from_dict,
            "message": message_from_dict,
            "error": message_from_dict,
            "tool_call": content_from_dict,
        },
    )


def context_from_dict(data: Mapping[str, Any]) -> t.Context:
    tools = data.get("tools")
    return t.Context(
        messages=messages_from_list(list(data.get("messages") or [])),
        system_prompt=data.get("systemPrompt", data.get("system_prompt")),
        tools=[tool_from_dict(tool) for tool in tools] if tools else None,
    )


def context_to_dict(context: t.Context | t.TranscriptContext) -> dict[str, Any]:
    return to_json_value(context)


_QUESTION_TYPES = {
    "choice": t.ClassifierChoiceQuestion,
    "score": t.ClassifierScoreQuestion,
    "bool": t.ClassifierBoolQuestion,
}
_ANSWER_TYPES = {"choice": t.ClassifierChoiceAnswer, "score": t.ClassifierScoreAnswer, "bool": t.ClassifierBoolAnswer}


def classifier_question_from_dict(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    return _build(_QUESTION_TYPES[data["type"]], data)


def classifier_answer_from_dict(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    return _build(_ANSWER_TYPES[data["type"]], data)


_DECODERS: dict[type, Callable[[Any], Any]] = {
    t.Context: context_from_dict,
    t.Model: model_from_dict,
    t.ImageModel: model_from_dict,
    t.ClassifierModel: model_from_dict,
    t.Tool: tool_from_dict,
    t.Usage: _usage,
    t.ModelCost: _cost,
    t.DeferredHandle: _deferred,
    t.SystemMessage: message_from_dict,
    t.UserMessage: message_from_dict,
    t.AssistantMessage: message_from_dict,
    t.ToolResultMessage: message_from_dict,
    t.TextContent: content_from_dict,
    t.ThinkingContent: content_from_dict,
    t.ImageContent: content_from_dict,
    t.ToolCall: content_from_dict,
    t.ImagesContext: lambda data: t.ImagesContext([content_from_dict(item) for item in data.get("input", [])]),
    t.ClassifierContext: lambda data: t.ClassifierContext(
        state=data["state"],
        questions={name: classifier_question_from_dict(q) for name, q in data.get("questions", {}).items()},
        images=[content_from_dict(image) for image in data["images"]] if data.get("images") else None,
    ),
    t.ClassifierResult: lambda data: _build(
        t.ClassifierResult,
        data,
        nested={
            "answers": lambda answers: {name: classifier_answer_from_dict(a) for name, a in answers.items()},
            "usage": _usage,
        },
    ),
    t.AssistantImages: lambda data: _build(t.AssistantImages, data, nested={"output": _content_list, "usage": _usage}),
}


def from_dict(cls: type[T] | str, data: Any) -> T:
    """Decode pi JSON into data class ``cls``.

    ``cls`` may also be ``"message"``, ``"content"``, ``"model"``, ``"event"`` or
    ``"context"`` to decode by discriminator.
    """
    if isinstance(cls, str):
        decoder = {
            "message": message_from_dict,
            "content": content_from_dict,
            "model": model_from_dict,
            "event": event_from_dict,
            "context": context_from_dict,
            "messages": messages_from_list,
        }[cls]
        return decoder(data)
    decoder = _DECODERS.get(cls)
    if decoder is not None:
        return decoder(data)
    if dataclasses.is_dataclass(cls):
        return _build(cls, data)  # type: ignore[return-value]
    raise TypeError(f"Cannot decode {cls!r}")


def loads(text: str | bytes, cls: type[T] | str = "context") -> T:
    """Parse a JSON string produced by :func:`dumps` (or TypeScript pi) into ``cls``."""
    return from_dict(cls, json.loads(text))
