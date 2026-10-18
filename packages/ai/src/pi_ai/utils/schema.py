"""JSON Schema builders for tool parameters.

:class:`Type` is a small TypeBox-style builder (the TypeScript package re-exports
TypeBox's ``Type``). Every builder returns a plain ``dict`` subclass, so schemas
serialize as ordinary JSON Schema and can be mixed with hand-written dicts::

    from pi_ai import Type, StringEnum

    params = Type.Object({
        "location": Type.String(description="City name"),
        "units": StringEnum(["celsius", "fahrenheit"], default="celsius"),
        "days": Type.Optional(Type.Integer(minimum=1, maximum=14)),
    })

Tool parameters may also be Pydantic model classes; :func:`parameters_schema`
converts them to JSON Schema once and caches the result.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["TSchema", "Type", "StringEnum", "parameters_schema", "is_pydantic_model", "clone_schema"]


class TSchema(dict[str, Any]):
    """A JSON Schema node produced by :class:`Type`.

    ``optional`` marks a property as not required when used inside ``Type.Object``.
    """

    __slots__ = ("optional",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optional = False


def _node(base: Mapping[str, Any], options: Mapping[str, Any]) -> TSchema:
    schema = TSchema(base)
    for key, value in options.items():
        if value is not None:
            schema[_option_key(key)] = value
    return schema


_OPTION_KEYS = {
    "min_length": "minLength",
    "max_length": "maxLength",
    "min_items": "minItems",
    "max_items": "maxItems",
    "unique_items": "uniqueItems",
    "exclusive_minimum": "exclusiveMinimum",
    "exclusive_maximum": "exclusiveMaximum",
    "multiple_of": "multipleOf",
    "additional_properties": "additionalProperties",
    "min_properties": "minProperties",
    "max_properties": "maxProperties",
    "pattern_properties": "patternProperties",
    "content_encoding": "contentEncoding",
    "content_media_type": "contentMediaType",
}


def _option_key(key: str) -> str:
    return _OPTION_KEYS.get(key, key)


class Type:
    """TypeBox-style JSON Schema builders. Keyword options map to JSON Schema keywords
    (``min_length`` → ``minLength``; camelCase keywords are accepted as-is)."""

    @staticmethod
    def Object(properties: Mapping[str, Any] | None = None, **options: Any) -> TSchema:
        props: dict[str, Any] = {}
        required: list[str] = []
        for name, value in (properties or {}).items():
            props[name] = _strip_optional(value)
            if not getattr(value, "optional", False):
                required.append(name)
        base: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            base["required"] = required
        return _node(base, options)

    @staticmethod
    def String(**options: Any) -> TSchema:
        return _node({"type": "string"}, options)

    @staticmethod
    def Number(**options: Any) -> TSchema:
        return _node({"type": "number"}, options)

    @staticmethod
    def Integer(**options: Any) -> TSchema:
        return _node({"type": "integer"}, options)

    @staticmethod
    def Boolean(**options: Any) -> TSchema:
        return _node({"type": "boolean"}, options)

    @staticmethod
    def Null(**options: Any) -> TSchema:
        return _node({"type": "null"}, options)

    @staticmethod
    def Any(**options: Any) -> TSchema:
        return _node({}, options)

    @staticmethod
    def Unknown(**options: Any) -> TSchema:
        return _node({}, options)

    @staticmethod
    def Array(items: Mapping[str, Any], **options: Any) -> TSchema:
        return _node({"type": "array", "items": _strip_optional(items)}, options)

    @staticmethod
    def Tuple(items: Sequence[Mapping[str, Any]], **options: Any) -> TSchema:
        values = [_strip_optional(item) for item in items]
        return _node(
            {"type": "array", "prefixItems": values, "items": False, "minItems": len(values), "maxItems": len(values)},
            options,
        )

    @staticmethod
    def Literal(value: Any, **options: Any) -> TSchema:
        base: dict[str, Any] = {"const": value}
        json_type = _json_type_of(value)
        if json_type:
            base["type"] = json_type
        return _node(base, options)

    @staticmethod
    def Union(schemas: Iterable[Mapping[str, Any]], **options: Any) -> TSchema:
        return _node({"anyOf": [_strip_optional(schema) for schema in schemas]}, options)

    @staticmethod
    def Intersect(schemas: Iterable[Mapping[str, Any]], **options: Any) -> TSchema:
        return _node({"allOf": [_strip_optional(schema) for schema in schemas]}, options)

    @staticmethod
    def Enum(values: Iterable[Any] | Mapping[str, Any], **options: Any) -> TSchema:
        """An enum of literal values (``anyOf`` of ``const``). Prefer :func:`StringEnum` for Google APIs."""
        items = list(values.values()) if isinstance(values, Mapping) else list(values)
        return _node({"anyOf": [Type.Literal(item) for item in items]}, options)

    @staticmethod
    def Record(key: Mapping[str, Any], value: Mapping[str, Any], **options: Any) -> TSchema:
        base: dict[str, Any] = {"type": "object", "additionalProperties": _strip_optional(value)}
        pattern = key.get("pattern") if isinstance(key, Mapping) else None
        if pattern:
            base = {"type": "object", "patternProperties": {pattern: _strip_optional(value)}}
        return _node(base, options)

    @staticmethod
    def Optional(schema: Mapping[str, Any]) -> TSchema:
        """Mark a property of ``Type.Object`` as optional (not listed in ``required``)."""
        node = TSchema(schema)
        node.optional = True
        return node

    @staticmethod
    def Nullable(schema: Mapping[str, Any], **options: Any) -> TSchema:
        return _node({"anyOf": [_strip_optional(schema), {"type": "null"}]}, options)

    @staticmethod
    def Ref(ref: str, **options: Any) -> TSchema:
        return _node({"$ref": ref}, options)

    @staticmethod
    def Unsafe(schema: Mapping[str, Any]) -> TSchema:
        """Wrap an arbitrary JSON Schema dict."""
        return TSchema(schema)


def StringEnum(values: Sequence[str], *, description: str | None = None, default: str | None = None) -> TSchema:
    """A string enum schema compatible with Google's API and other providers that do not
    support ``anyOf``/``const`` patterns."""
    schema = TSchema({"type": "string", "enum": list(values)})
    if description:
        schema["description"] = description
    if default:
        schema["default"] = default
    return schema


def _strip_optional(schema: Any) -> Any:
    if isinstance(schema, TSchema) and schema.optional:
        return TSchema(schema)
    return schema


def _json_type_of(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


# ---------------------------------------------------------------------------
# Pydantic support
# ---------------------------------------------------------------------------

_pydantic_schema_cache: dict[type, dict[str, Any]] = {}


def is_pydantic_model(value: Any) -> bool:
    """True for Pydantic ``BaseModel`` subclasses (without importing pydantic)."""
    return isinstance(value, type) and hasattr(value, "model_json_schema") and hasattr(value, "model_validate")


def parameters_schema(parameters: Any) -> dict[str, Any]:
    """The JSON Schema for tool ``parameters`` (a schema dict or a Pydantic model class)."""
    if isinstance(parameters, dict):
        return parameters
    if is_pydantic_model(parameters):
        cached = _pydantic_schema_cache.get(parameters)
        if cached is None:
            cached = _inline_pydantic_schema(parameters.model_json_schema())
            _pydantic_schema_cache[parameters] = cached
        return cached
    if isinstance(parameters, Mapping):
        return dict(parameters)
    raise TypeError(f"Tool parameters must be a JSON Schema dict or a Pydantic model, got {type(parameters)!r}")


def _inline_pydantic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop Pydantic's ``title`` noise and inline local ``$defs`` references.

    Several providers (Google in particular) reject ``$ref``; inlining keeps the
    schema portable. Recursive models keep their references.
    """
    defs = schema.get("$defs", {})
    named_schema_maps = ("properties", "patternProperties", "dependentSchemas")
    recursive = False

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/") :]
            target = defs.get(name)
            if target is not None and name not in seen:
                resolved = resolve(target, seen | {name})
                extra = {k: resolve(v, seen) for k, v in node.items() if k not in ("$ref", "title")}
                return {**resolved, **extra}
            nonlocal recursive
            recursive = True
            return node
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("$defs", "title"):
                continue
            if key in named_schema_maps and isinstance(value, dict):
                result[key] = {name: resolve(sub, seen) for name, sub in value.items()}
            else:
                result[key] = resolve(value, seen)
        return result

    result = resolve(schema, frozenset())
    if recursive:
        result = dict(schema)
    if "type" not in result:
        result["type"] = "object"
    return result


def clone_schema(schema: Any) -> Any:
    """Deep copy a JSON-compatible schema (drops dict subclasses and non-JSON values)."""
    return json.loads(json.dumps(parameters_schema(schema) if not isinstance(schema, (list, str)) else schema))
