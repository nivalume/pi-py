"""Strict JSON values (port of chord's ``json.ts``).

Strict JSON here means: ``None``, ``bool``, ``int``, finite ``float``, ``str``, and exact
``list`` and ``dict`` (string keys) containers without cycles. Tuples, subclasses and
other objects are rejected rather than converted, like plain objects and dense arrays
in the TypeScript original.
"""

from __future__ import annotations

import math
from typing import Any, Final, TypeAlias

__all__ = [
    "JsonValue",
    "JsonObject",
    "JsonRepresentation",
    "UNDEFINED",
    "copy_json",
    "is_json_value",
]

JsonValue: TypeAlias = "bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None"
JsonObject: TypeAlias = "dict[str, JsonValue]"
#: The strict-JSON form of an application type. Python cannot compute it from a type the
#: way TypeScript does, so it is an alias of :data:`JsonValue` kept for API parity.
JsonRepresentation: TypeAlias = "JsonValue"


class _Undefined:
    """JavaScript ``undefined``: a property that is present in code but absent from JSON."""

    __slots__ = ()
    _instance: _Undefined | None = None

    def __new__(cls) -> _Undefined:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNDEFINED"

    def __bool__(self) -> bool:
        return False

    def __reduce__(self) -> str:
        return "UNDEFINED"


#: Marks a dict entry to drop with ``copy_json(..., omit_undefined_properties=True)``.
UNDEFINED: Final = _Undefined()


def copy_json(value: Any, *, omit_undefined_properties: bool = False) -> JsonValue:
    """Copy ``value`` into an alias-free strict-JSON tree owned by the caller.

    Raises :class:`TypeError` for anything that is not strict JSON. With
    ``omit_undefined_properties``, dict entries whose value is :data:`UNDEFINED` are
    dropped; :data:`UNDEFINED` anywhere else is still rejected.
    """
    return _copy(value, set(), omit_undefined_properties)


def _copy(value: Any, ancestors: set[int], omit_undefined: bool) -> JsonValue:
    kind = type(value)
    if value is None or kind is str or kind is bool or kind is int:
        return value
    if kind is float:
        if math.isfinite(value):
            return value
        raise TypeError("Value contains a non-finite number and is not strict JSON")
    if kind is list:
        marker = id(value)
        if marker in ancestors:
            raise TypeError("Value contains cycles and is not strict JSON")
        ancestors.add(marker)
        try:
            return [_copy(item, ancestors, omit_undefined) for item in value]
        finally:
            ancestors.discard(marker)
    if kind is dict:
        marker = id(value)
        if marker in ancestors:
            raise TypeError("Value contains cycles and is not strict JSON")
        ancestors.add(marker)
        try:
            result: dict[str, JsonValue] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"Value contains a non-string key {key!r} and is not strict JSON")
                if item is UNDEFINED and omit_undefined:
                    continue
                result[key] = _copy(item, ancestors, omit_undefined)
            return result
        finally:
            ancestors.discard(marker)
    if value is UNDEFINED:
        raise TypeError("Value contains UNDEFINED and is not strict JSON")
    raise TypeError(f"Value contains a non-JSON {kind.__name__}; expected strict JSON")


def is_json_value(value: Any) -> bool:
    """Whether ``value`` is strict JSON (finite numbers, exact list/dict, no cycles)."""
    return _check(value, set())


def _check(value: Any, ancestors: set[int]) -> bool:
    kind = type(value)
    if value is None or kind is str or kind is bool or kind is int:
        return True
    if kind is float:
        return math.isfinite(value)
    if kind is list or kind is dict:
        marker = id(value)
        if marker in ancestors:
            return False
        ancestors.add(marker)
        try:
            if kind is list:
                return all(_check(item, ancestors) for item in value)
            return all(type(key) is str and _check(item, ancestors) for key, item in value.items())
        finally:
            ancestors.discard(marker)
    return False
