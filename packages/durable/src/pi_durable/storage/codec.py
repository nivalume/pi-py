"""JSON encoding of records for storage backends (Python-only helpers).

Records are already their upstream JSON shape (see :mod:`pi_durable.types`) except for two things
Python holds as objects:

- pi-ai messages in ``EntryRecord["model"]`` and ``ContextEdit["messages"]`` are dataclasses;
  they are stored as :mod:`pi_ai.serialization` JSON (camelCase), the format TypeScript pi writes.
- Chord ops are tuples with tuple paths; they are stored as JSON arrays.

``encode_*`` returns strict JSON (``pi_chord.json.copy_json`` accepts it, ``json.dumps`` writes
upstream's format) and ``decode_*`` reverses it, so a backend persists ``encode_storage_write(write)``
and returns ``decode_entry(row)``. :func:`clone_value` is the detached copy the memory storage takes
at its ownership boundary: it copies JSON and round-trips messages through the same encoding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pi_ai.serialization import message_from_dict, to_json_value
from pi_ai.types import AssistantMessage, Message, SystemMessage, ToolResultMessage, UserMessage
from pi_chord.delta import Op
from pi_chord.json import JsonValue

from ..types import ContextEdit, EntryRecord, StorageWrite

__all__ = [
    "clone_value",
    "decode_entry",
    "decode_messages",
    "decode_ops",
    "decode_storage_write",
    "encode_entry",
    "encode_messages",
    "encode_ops",
    "encode_storage_write",
]

_MESSAGE_TYPES = (SystemMessage, UserMessage, AssistantMessage, ToolResultMessage)


def encode_messages(messages: Sequence[Message]) -> list[JsonValue]:
    """pi-ai messages as their camelCase JSON."""
    return [_json(to_json_value(message)) for message in messages]


def decode_messages(data: Any) -> list[Message]:
    """pi-ai messages from their camelCase JSON."""
    if not isinstance(data, list):
        raise TypeError("Stored messages must be a JSON array")
    return [_message(item) for item in data]


def _message(data: Any) -> Message:
    message = message_from_dict(data)
    if not isinstance(message, _MESSAGE_TYPES):
        raise TypeError(f"Stored message has an unknown role: {data.get('role') if isinstance(data, dict) else data!r}")
    return message


def _encode_edit(edit: Mapping[str, Any]) -> dict[str, JsonValue]:
    encoded: dict[str, JsonValue] = {}
    for key, value in edit.items():
        encoded[key] = encode_messages(value) if key == "messages" else _json(value)
    return encoded


def _decode_edit(data: Mapping[str, Any]) -> ContextEdit:
    decoded: dict[str, Any] = {}
    for key, value in data.items():
        decoded[key] = decode_messages(value) if key == "messages" else _json(value)
    return decoded  # type: ignore[return-value]


def encode_entry(entry: Mapping[str, Any]) -> dict[str, JsonValue]:
    """An entry record (or draft) as JSON: ``model`` and edit ``messages`` encoded, everything else copied."""
    encoded: dict[str, JsonValue] = {}
    for key, value in entry.items():
        if key == "model":
            encoded[key] = encode_messages(value)
        elif key == "edits":
            encoded[key] = [_encode_edit(edit) for edit in value]
        else:
            encoded[key] = _json(value)
    return encoded


def decode_entry(data: Mapping[str, Any]) -> EntryRecord:
    """An entry record from its stored JSON."""
    decoded: dict[str, Any] = {}
    for key, value in data.items():
        if key == "model":
            decoded[key] = decode_messages(value)
        elif key == "edits":
            if not isinstance(value, list):
                raise TypeError("Stored entry edits must be a JSON array")
            decoded[key] = [_decode_edit(edit) for edit in value]
        else:
            decoded[key] = _json(value)
    return decoded  # type: ignore[return-value]


def encode_ops(ops: Sequence[Sequence[Any]]) -> list[JsonValue]:
    """Chord ops as JSON arrays (paths become arrays too)."""
    return [_json(op) for op in ops]


def decode_ops(data: Any) -> list[Op]:
    """Chord ops from JSON arrays: tuples with tuple paths, as the tracker emits them."""
    if not isinstance(data, list):
        raise TypeError("Stored ops must be a JSON array")
    return [_decode_op(op) for op in data]


def _decode_op(op: Any) -> Op:
    if not isinstance(op, (list, tuple)) or len(op) == 0:
        raise TypeError(f"Stored op is malformed: {op!r}")
    verb = op[0]
    if verb == "r":
        return ("r", *op[1:])  # type: ignore[return-value]
    path = op[1] if len(op) > 1 else ()
    if not isinstance(path, (list, tuple)):
        raise TypeError(f"Stored op has a malformed path: {op!r}")
    return (verb, tuple(path), *op[2:])  # type: ignore[return-value]


def encode_storage_write(write: StorageWrite) -> dict[str, JsonValue]:
    """One storage write as JSON, in upstream's shape."""
    if write["type"] == "entry":
        return {"type": "entry", "value": encode_entry(write["value"])}
    if write["type"] == "document.change":
        content = write["content"]
        if content["kind"] == "delta":
            return {
                "type": "document.change",
                "id": write["id"],
                "content": {"version": content["version"], "kind": "delta", "ops": encode_ops(content["ops"])},
            }
    return _json(write)  # type: ignore[return-value]


def decode_storage_write(data: Mapping[str, Any]) -> StorageWrite:
    """One storage write from its JSON."""
    kind = data.get("type")
    if kind == "entry":
        return {"type": "entry", "value": decode_entry(data["value"])}
    if kind == "document.change":
        content = data["content"]
        if content.get("kind") == "delta":
            return {
                "type": "document.change",
                "id": data["id"],
                "content": {"version": content["version"], "kind": "delta", "ops": decode_ops(content["ops"])},
            }
    return _json(data)  # type: ignore[return-value]


def _json(value: Any) -> Any:
    """Copy JSON, turning tuples into lists; raises :class:`TypeError` for anything else."""
    kind = type(value)
    if value is None or kind is str or kind is bool or kind is int:
        return value
    if kind is float:
        if math.isfinite(value):
            return value
        raise TypeError("Value contains a non-finite number and is not strict JSON")
    if kind is list or kind is tuple or isinstance(value, list):
        return [_json(item) for item in value]
    if kind is dict or isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"Value contains a non-string key {key!r} and is not strict JSON")
            result[key] = _json(item)
        return result
    raise TypeError(f"Value contains a non-JSON {kind.__name__}; expected strict JSON")


def clone_value(value: Any) -> Any:
    """Detached deep copy of a record, write, or value at a storage ownership boundary.

    JSON is copied (tuples stay tuples, so ops keep their shape), pi-ai messages are round-tripped
    through their stored JSON, and anything else raises :class:`TypeError`, as a serializing backend
    would.
    """
    kind = type(value)
    if value is None or kind is str or kind is bool or kind is int or kind is float:
        return value
    if kind is dict or isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"Value contains a non-string key {key!r} and is not strict JSON")
            result[key] = clone_value(item)
        return result
    if kind is list or isinstance(value, list):
        return [clone_value(item) for item in value]
    if kind is tuple:
        return tuple(clone_value(item) for item in value)
    if isinstance(value, _MESSAGE_TYPES):
        return _message(to_json_value(value))
    raise TypeError(f"Value contains a non-JSON {kind.__name__}; expected strict JSON")
