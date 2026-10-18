"""pi_chord: the parts of ``@earendil-works/chord`` that pi_durable builds on.

- :mod:`pi_chord.context`: explicit cancellation and scoped values (``Context``).
- :mod:`pi_chord.json`: strict JSON values (``copy_json``, ``is_json_value``).
- :mod:`pi_chord.delta`: the transactional JSON tracker, drafts, diffs and op batches.
- :mod:`pi_chord.state`: observable replicated JSON state.
- :mod:`pi_chord.output`: bounded output buffers and paced progress.
- :mod:`pi_chord.sqlite`: the async SQLite facade and migrations;
  :mod:`pi_chord.sqlite.stdlib` adapts the standard library ``sqlite3``.

The root re-exports what chord's root entry point offers for these parts. Chord's hosts,
contracts, links and hot reload are not part of this port.
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
from .delta import Draft
from .json import UNDEFINED, JsonObject, JsonRepresentation, JsonValue, copy_json, is_json_value
from .state import (
    AttachedReplicatedState,
    MutableReplicatedState,
    ReplicatedState,
    ReplicatedStateDelivery,
    ReplicatedStateSnapshot,
    ReplicatedStateSource,
    ReplicatedStateSourceAttachment,
    ReplicatedStateSourceFrame,
    ReplicatedStateSourceOptions,
    StateListener,
    replicated_state,
)

__all__ = [
    # context
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
    # json
    "UNDEFINED",
    "JsonObject",
    "JsonRepresentation",
    "JsonValue",
    "copy_json",
    "is_json_value",
    # delta
    "Draft",
    # state
    "AttachedReplicatedState",
    "MutableReplicatedState",
    "ReplicatedState",
    "ReplicatedStateDelivery",
    "ReplicatedStateSnapshot",
    "ReplicatedStateSource",
    "ReplicatedStateSourceAttachment",
    "ReplicatedStateSourceFrame",
    "ReplicatedStateSourceOptions",
    "StateListener",
    "replicated_state",
]
