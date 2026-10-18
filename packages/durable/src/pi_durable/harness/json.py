"""Leaf-by-leaf JSON assignment into Chord drafts (port of ``harness/json.ts``)."""

from __future__ import annotations

from typing import Any

from pi_chord.json import UNDEFINED, JsonValue

__all__ = ["assign_json"]


def assign_json(target: dict[str, Any] | list[Any], key: str | int, value: JsonValue) -> None:
    """Assign ``value`` at ``target[key]`` leaf by leaf.

    Chord records a container assignment as one full set and only emits an append when a string
    leaf is reassigned with a longer string, so writing the partial whole would store and publish
    the complete message on every flush. ``target`` is a draft (or plain) dict or list; a list
    ``key`` must address an existing element.
    """
    if isinstance(target, dict):
        current: Any = target.get(key, UNDEFINED)  # type: ignore[call-overload]
    else:
        current = target[key]  # type: ignore[index]
    if isinstance(current, dict) and isinstance(value, dict):
        for name in [name for name in current if name not in value]:
            del current[name]
        for name, child in value.items():
            assign_json(current, name, child)
        return
    if isinstance(current, list) and isinstance(value, list) and len(current) <= len(value):
        for index, item in enumerate(value):
            if index < len(current):
                assign_json(current, index, item)
            else:
                current.append(item)
        return
    if not _same(current, value):
        target[key] = value  # type: ignore[index]


def _same(current: Any, value: Any) -> bool:
    """JavaScript ``current === value`` for JSON values: identity for containers, value for primitives."""
    if current is value:
        return True
    if current is UNDEFINED or isinstance(current, (dict, list)) or isinstance(value, (dict, list)):
        return False
    if isinstance(current, bool) or isinstance(value, bool):
        return type(current) is type(value) and current == value
    return current == value
