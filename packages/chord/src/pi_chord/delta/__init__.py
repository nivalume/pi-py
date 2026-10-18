"""Immutable JSON revisions and exact operation batches (port of ``@earendil-works/chord/delta``).

Immutability is an ownership contract: nothing is frozen or defensively copied, so an
illegal mutation is not detected; it silently corrupts state.

.. code-block:: python

    initial = {"output": "", "entries": []}
    tracker = track(initial)  # `initial` is transferred: never mutate it again

    change = tracker.begin_change()
    change.state["output"] += "done\\n"  # the draft is mutable only while the change is open
    change.state["entries"].append({"id": 1})  # placed values are copied
    prepared = change.prepare()  # draft handles are unusable from here on

    tracker.adopt(prepared)  # tracker.value is prepared.value
    replica = apply_immutable(prepared.base, prepared.ops)  # shares containers: never mutate it

Mutation rights:

- Roots passed to :func:`track`, :meth:`Tracker.prepare_replace`, ``replicated_state()`` or
  ``replace()``: never mutate them again; ownership moved to the tracker.
- ``change.state`` and handles read from it: only while that change is open. After
  ``prepare()``, ``abort()`` or another adoption every use raises :class:`TypeError`.
  Writes through a handle whose element was removed from the draft are ignored.
- A value after assigning or inserting it into a draft: yes, the draft stored a copy.
- ``tracker.value``, ``prepared.base``, ``prepared.value``, ``prepared.ops`` (paths,
  permutations, payloads), :func:`apply_immutable` inputs and results, values published
  by replicated state: never.
- A mutable replica passed to :func:`apply`: only through ``apply()``, with batches it
  owns exclusively (``apply()`` adopts payload containers).

Drafts follow Python semantics where JavaScript's differ: ``del draft[key]`` raises
``KeyError`` for a missing key (assign :data:`~pi_chord.json.UNDEFINED` to delete
quietly), ``del draft_list[i]`` removes the element, and index writes must address an
existing element (use ``append``). Batches stay wire-compatible with the JavaScript
tracker: ops are tuples (JSON arrays on the wire) and ``t`` counts UTF-16 code units.
"""

from __future__ import annotations

from ..json import JsonValue
from .diff import diff_revisions
from .draft import Draft
from .operations import (
    RESERVED_SEGMENTS,
    Decoder,
    Encoder,
    NonEmptyPath,
    Op,
    Path,
    PathError,
    PathRef,
    Seg,
    UnsafePathError,
    WireOp,
    apply,
    apply_immutable,
    apply_immutable_batches,
    assert_safe_path,
    assert_valid_op,
    assert_valid_wire_op,
    decoder,
    encoder,
    is_base,
    is_replace,
    overlap,
)
from .tracker import Change, DraftDict, DraftList, Prepared, Tracker, detach, track

__all__ = [
    "JsonValue",
    "Seg",
    "Path",
    "NonEmptyPath",
    "PathRef",
    "Op",
    "WireOp",
    "is_replace",
    "is_base",
    "overlap",
    "diff_revisions",
    "Draft",
    "DraftDict",
    "DraftList",
    "Change",
    "Prepared",
    "Tracker",
    "track",
    "detach",
    "RESERVED_SEGMENTS",
    "UnsafePathError",
    "PathError",
    "assert_valid_op",
    "assert_valid_wire_op",
    "assert_safe_path",
    "apply",
    "apply_immutable",
    "apply_immutable_batches",
    "Encoder",
    "Decoder",
    "encoder",
    "decoder",
]
