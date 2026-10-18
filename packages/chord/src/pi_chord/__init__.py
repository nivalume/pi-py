"""pi_chord: the parts of ``@earendil-works/chord`` that pi_durable builds on.

Ported so far:

- :mod:`pi_chord.context`: explicit cancellation and scoped values (``Context``).
- :mod:`pi_chord.json`: strict JSON values (``copy_json``, ``is_json_value``).

Chord's hosts, contracts, links and hot reload are not part of this port.
"""

from .context import (
    BACKGROUND_CONTEXT,
    TODO_CONTEXT,
    Cancellable,
    Context,
    ContextKey,
    await_with_context,
    create_context_key,
    with_abort_signal,
    with_cancel,
    with_context_value,
    without_abort_signal,
)
from .json import UNDEFINED, JsonObject, JsonRepresentation, JsonValue, copy_json, is_json_value

__all__ = [
    "BACKGROUND_CONTEXT",
    "TODO_CONTEXT",
    "Cancellable",
    "Context",
    "ContextKey",
    "await_with_context",
    "create_context_key",
    "with_abort_signal",
    "with_cancel",
    "with_context_value",
    "without_abort_signal",
    "UNDEFINED",
    "JsonObject",
    "JsonRepresentation",
    "JsonValue",
    "copy_json",
    "is_json_value",
]
