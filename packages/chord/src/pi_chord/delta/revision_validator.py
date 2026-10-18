"""Validate immutable replica revisions incrementally (port of chord ``delta/revision-validator.ts``)."""

from __future__ import annotations

import math
from typing import Any, TypeVar

__all__ = ["JsonRevisionValidator"]

T = TypeVar("T")


class JsonRevisionValidator:
    """Validates successive immutable revisions as strict JSON, skipping containers already validated.

    Upstream remembers validated containers in a ``WeakSet``; ``dict`` and ``list`` cannot be
    weakly referenced, so this keeps the containers seen while validating the previous
    revision. Containers a new revision shares with the previous one are not walked again.
    """

    __slots__ = ("_validated",)

    def __init__(self) -> None:
        self._validated: dict[int, Any] = {}

    def validate(self, value: T) -> T:
        """Return ``value`` unchanged, or raise :class:`TypeError` if it is not strict JSON."""
        previous = self._validated
        current: dict[int, Any] = {}
        _validate(value, previous, current, set())
        self._validated = current
        return value


def _validate(value: Any, previous: dict[int, Any], current: dict[int, Any], ancestors: set[int]) -> None:
    kind = type(value)
    if kind is dict or kind is list:
        marker = id(value)
        if marker in current:
            return
        if previous.get(marker) is value:
            current[marker] = value
            return
        if marker in ancestors:
            raise TypeError("Replicated state cannot contain cycles")
        ancestors.add(marker)
        try:
            if kind is list:
                for item in value:
                    _validate(item, previous, current, ancestors)
            else:
                for key, item in value.items():
                    if type(key) is not str:
                        raise TypeError("Replicated state objects must have string keys")
                    _validate(item, previous, current, ancestors)
        finally:
            ancestors.discard(marker)
        current[marker] = value
        return
    if value is None or kind is str or kind is bool or kind is int:
        return
    if kind is float and math.isfinite(value):
        return
    if kind is float:
        raise TypeError("Replicated state values must be strict JSON")
    raise TypeError("Replicated state containers must be plain objects or arrays")
