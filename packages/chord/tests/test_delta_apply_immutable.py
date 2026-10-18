"""Port of chord ``test/delta-apply-immutable.test.ts``.

``Object.freeze`` has no Python counterpart; inputs are compared against deep copies
taken before application instead. The ``Object.keys`` call count becomes a count of
``dict.copy`` calls observed with ``sys.setprofile``.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from pi_chord.delta import (
    PathError,
    UnsafePathError,
    apply,
    apply_immutable,
    apply_immutable_batches,
    track,
)


def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def test_copies_each_touched_container_once_while_preserving_input_and_payload_ownership() -> None:
    shared = {"nested": {"value": 1}}
    untouched = {"value": 9}
    row_payload = {"id": 4, "label": "placed"}
    base: dict[str, Any] = {
        "text": "abcdef",
        "stable": {"value": 7},
        "branch": {"value": 1},
        "copy": None,
        "placed": None,
        "untouched": None,
        "left": None,
        "right": None,
        "meta": {"count": 0, "obsolete": True},
        "rows": [{"id": 1, "label": "one"}, {"id": 2, "label": "two"}, {"id": 3, "label": "three"}],
    }
    snapshot = copy.deepcopy([base, shared, untouched, row_payload])
    operations: list[Any] = [
        ("t", ("text",), 2),
        ("a", ("text",), "!"),
        ("s", ("meta", "count"), 1),
        ("s", ("meta", "count"), 2),
        ("d", ("meta", "obsolete")),
        ("s", ("copy",), base["branch"]),
        ("s", ("copy", "value"), 2),
        ("s", ("placed",), shared),
        ("s", ("placed", "nested", "value"), 2),
        ("s", ("untouched",), untouched),
        ("s", ("left",), shared),
        ("s", ("right",), shared),
        ("s", ("left", "nested", "value"), 3),
        ("p", ("rows",), 1, 1, [row_payload]),
        ("s", ("rows", 1, "label"), "edited"),
        ("m", ("rows",), [1, 0, 2]),
        ("s", ("rows", 0, "label"), "moved"),
    ]

    result = apply_immutable(base, operations)
    mutable_result = apply(clone(base), clone(operations))

    assert result == mutable_result
    assert result["text"] == "cdef!"
    assert result["stable"] is base["stable"]
    assert result["untouched"] is untouched
    assert result["copy"] is not base["branch"]
    assert result["copy"] == {"value": 2}
    assert result["placed"] is not shared
    assert result["placed"] == {"nested": {"value": 2}}
    assert result["left"] is not shared
    assert result["right"] is shared
    assert result["rows"][0] is not row_payload
    assert result["rows"][0] == {"id": 4, "label": "moved"}
    assert [base, shared, untouched, row_payload] == snapshot


def test_protects_root_replacement_payloads_before_later_object_and_array_edits() -> None:
    replacement = {"nested": {"value": 1}, "values": [1, 2, 3]}
    result = apply_immutable_batches(
        None,
        [
            [("r", replacement)],
            [("s", ("nested", "value"), 2)],
            [("p", ("values",), 1, 1, [4, 5]), ("m", ("values",), [3, 0, 1, 2])],
        ],
    )
    assert result == {"nested": {"value": 2}, "values": [3, 1, 4, 5]}
    assert replacement == {"nested": {"value": 1}, "values": [1, 2, 3]}

    array = [1, 2, 3]
    array_result = apply_immutable(array, [("p", (), 1, 1, [4, 5]), ("m", (), [3, 0, 1, 2]), ("d", (1,))])
    assert array_result == [3, 4, 5]
    assert array == [1, 2, 3]


def test_shares_one_private_copy_on_write_scope_across_batch_partitions() -> None:
    base: dict[str, Any] = {
        "text": "abcdef",
        "meta": {"count": 0},
        "values": [{"id": 1, "value": 1}, {"id": 2, "value": 2}, {"id": 3, "value": 3}],
    }
    snapshot = copy.deepcopy(base)
    batches: list[list[Any]] = [
        [("s", ("meta", "count"), 1), ("p", ("values",), 1, 1, [{"id": 4, "value": 4}])],
        [],
        [("m", ("values",), [2, 0, 1]), ("s", ("values", 2, "value"), 40)],
        [("t", ("text",), 2), ("a", ("text",), "!")],
    ]

    intermediate = apply_immutable(base, batches[0])
    intermediate_snapshot = clone(intermediate)
    sequential = intermediate
    for batch in batches[1:]:
        sequential = apply_immutable(sequential, batch)
    streamed = apply_immutable_batches(base, batches)
    flattened = apply_immutable(base, [op for batch in batches for op in batch])

    assert streamed == sequential
    assert streamed == flattened
    assert intermediate == intermediate_snapshot
    assert base == snapshot


def test_handles_tracker_produced_batches_across_arbitrary_revision_boundaries() -> None:
    initial: dict[str, Any] = {
        "text": "start",
        "values": [{"id": index, "score": 0} for index in range(8)],
        "revision": 0,
    }
    tracker = track(initial)
    batches: list[Any] = []
    for revision in range(1, 41):
        change = tracker.begin_change()
        state = change.state
        state["revision"] = revision
        state["text"] = f"{state['text'][1:]}{revision}"
        values = state["values"]
        kind = revision % 5
        if kind == 0:
            values.reverse()
        elif kind == 1:
            values.append({"id": 100 + revision, "score": revision})
        elif kind == 2:
            values.pop(0)
        elif kind == 3:
            values[revision % len(values)]["score"] = revision
        else:
            values[1:2] = [{"id": 200 + revision, "score": revision}]
        prepared = change.prepare()
        batches.append(prepared.ops)
        tracker.adopt(prepared)

    assert apply_immutable_batches(initial, batches) == tracker.value


def test_does_not_expose_partial_application_when_validation_or_iteration_fails() -> None:
    base = {"nested": {"value": 1}}
    with pytest.raises(UnsafePathError):
        apply_immutable_batches(
            base, [[("s", ("nested", "value"), 2)], [("s", ("constructor", "prototype", "polluted"), True)]]
        )
    assert base["nested"]["value"] == 1

    def throwing_batches() -> Iterator[list[Any]]:
        yield [("s", ("nested", "value"), 3)]
        raise RuntimeError("revision stream failed")

    with pytest.raises(RuntimeError, match="revision stream failed"):
        apply_immutable_batches(base, throwing_batches())
    assert base["nested"]["value"] == 1

    advanced_past_invalid = False
    iterator_closed = False

    def invalid_operation_batches() -> Iterator[list[Any]]:
        nonlocal advanced_past_invalid, iterator_closed
        try:
            yield [("s", ("nested", "value"), 4)]
            yield [("s", ("__proto__", "polluted"), True)]
            advanced_past_invalid = True
        finally:
            iterator_closed = True

    with pytest.raises(UnsafePathError):
        apply_immutable_batches(base, invalid_operation_batches())
    assert not advanced_past_invalid
    assert iterator_closed
    assert base["nested"]["value"] == 1

    with pytest.raises(TypeError, match="path"):
        apply_immutable_batches(base, [[["s", "bad-path", 1]]])

    with pytest.raises(PathError):
        apply_immutable({"values": []}, [("s", ("values", "missing", "value"), 1)])
    with pytest.raises(UnsafePathError):
        apply_immutable({"values": [{}]}, [("s", ("values", "0", "value"), 1)])


def test_allows_one_immutable_batch_to_fan_out_without_mutating_shared_payloads() -> None:
    payload = {"nested": {"value": 1}}
    operations: list[Any] = [("s", ("placed",), payload), ("s", ("placed", "nested", "value"), 2)]
    base: dict[str, Any] = {"placed": None}
    first = apply_immutable(base, operations)
    second = apply_immutable(base, operations)
    assert first == second
    assert first is not second
    assert first["placed"] is not second["placed"]
    assert payload["nested"]["value"] == 1


def test_appends_to_an_array_not_yet_copied_in_one_copy_sharing_everything_else() -> None:
    entries = [{"index": index} for index in range(1_000)]
    base = {"log": {"entries": entries, "meta": {"count": 1_000}}, "other": {"value": 1}}
    result = apply_immutable(
        base,
        [
            ("p", ("log", "entries"), 1_000, 0, [{"index": 1_000}]),
            ("p", ("log", "entries"), 1_001, 0, [{"index": 1_001}]),
        ],
    )
    assert len(result["log"]["entries"]) == 1_002
    assert result["log"]["entries"][-2:] == [{"index": 1_000}, {"index": 1_001}]
    assert len(base["log"]["entries"]) == 1_000
    assert result["log"]["entries"][0] is entries[0]
    assert result["log"]["meta"] is base["log"]["meta"]
    assert result["other"] is base["other"]
    assert result["log"] is not base["log"]

    root_array = [1, 2]
    assert apply_immutable(root_array, [("p", (), 2, 0, [3])]) == [1, 2, 3]
    assert root_array == [1, 2]
    # An appended item that is itself an array stays one element.
    assert apply_immutable(root_array, [("p", (), 2, 0, [[3, 4]])]) == [1, 2, [3, 4]]
    # Invalid paths fail with the general path's errors.
    invalid: list[tuple[Any, type[Exception]]] = [
        (("p", ("values", "missing"), 0, 0, []), PathError),
        (("p", ("missing",), 0, 0, []), PathError),
        (("p", ("values", 0), 0, 0, []), PathError),
        (("p", ("__proto__",), 0, 0, []), UnsafePathError),
        (("p", ("list", "x"), 0, 0, []), PathError),
        (("p", ("text",), 0, 0, []), PathError),
        (("p", ("list", -1), 0, 0, []), UnsafePathError),
        (("p", ("list", 1.5), 0, 0, []), UnsafePathError),
        (("p", ("list", 0, "constructor"), 0, 0, []), UnsafePathError),
    ]
    document: dict[str, Any] = {"values": [], "list": [[1]], "text": "a"}
    for op, error in invalid:
        with pytest.raises(error):
            apply_immutable(document, [op])
    # An insert that is not at the end takes the general path.
    assert apply_immutable(root_array, [("p", (), 1, 0, [9])]) == [1, 9, 2]


def test_copies_a_wide_object_once_rather_than_once_per_repeated_write() -> None:
    width = 20_000
    base = {f"field{index}": index for index in range(width)}
    operations: list[Any] = [("s", (f"field{index}",), -index) for index in range(1_000)]
    copies = 0

    def profile(frame: Any, event: str, arg: Any) -> None:
        nonlocal copies
        if (
            event == "c_call"
            and getattr(arg, "__name__", None) == "copy"
            and getattr(arg, "__self__", None) is not None
        ):
            if len(arg.__self__) == width:
                copies += 1

    sys.setprofile(profile)
    try:
        result = apply_immutable(base, operations)
    finally:
        sys.setprofile(None)
    assert result["field999"] == -999
    assert base["field999"] == 999
    assert copies == 1
