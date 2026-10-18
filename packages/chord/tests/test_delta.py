"""Port of chord ``test/delta.test.ts``."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest

from pi_chord.delta import (
    UnsafePathError,
    apply,
    apply_immutable,
    assert_valid_op,
    assert_valid_wire_op,
    decoder,
    encoder,
    is_base,
    overlap,
    track,
)


def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


# ─── immutable tracker lifecycle ─────────────────────────────────────────────


async def test_keeps_a_draft_alive_across_await_and_adopts_only_a_prepared_change() -> None:
    initial: dict[str, Any] = {"count": 1, "nested": {"text": "a"}, "values": [1]}
    tracker = track(initial)
    assert tracker.value is initial

    change = tracker.begin_change()
    change.state["count"] = 2
    await asyncio.sleep(0)
    change.state["nested"]["text"] += "b"
    change.state["values"].append(2)
    assert tracker.value == initial

    prepared = change.prepare()
    assert prepared.base is tracker.value
    assert prepared.value == {"count": 2, "nested": {"text": "ab"}, "values": [1, 2]}
    assert prepared.ops == [
        ("s", ("count",), 2),
        ("a", ("nested", "text"), "b"),
        ("p", ("values",), 1, 0, [2]),
    ]
    with pytest.raises(TypeError):
        change.state["count"]
    assert tracker.value == initial

    tracker.adopt(prepared)
    assert tracker.value is prepared.value


def test_aborts_and_revokes_without_changing_the_committed_value() -> None:
    tracker = track({"child": {"value": 1}})
    change = tracker.begin_change()
    child = change.state["child"]
    child["value"] = 2
    change.abort()
    assert tracker.value["child"]["value"] == 1
    with pytest.raises(TypeError):
        child["value"]
    change.abort()
    with pytest.raises(RuntimeError, match="settled"):
        change.prepare()
    tracker.begin_change().abort()


def test_grows_arrays_with_explicit_nulls_and_revokes_drafts_after_preparation() -> None:
    # Upstream grows through `values.length = 4`; Python lists have no length setter.
    tracker = track({"values": [1, 2]})
    change = tracker.begin_change()
    change.state["values"].extend([None, None])
    prepared = change.prepare()
    assert prepared.value["values"] == [1, 2, None, None]
    with pytest.raises(TypeError):
        change.state["values"]
    tracker.adopt(prepared)


def test_normalizes_a_deep_noop_to_exact_previous_identity() -> None:
    tracker = track({"value": {"nested": [1, 2]}})
    change = tracker.begin_change()
    change.state["value"] = {"nested": [1, 2]}
    prepared = change.prepare()
    assert prepared.value is prepared.base
    assert prepared.ops == []
    tracker.adopt(prepared)
    assert tracker.value is prepared.base


def test_takes_immutable_ownership_of_replacement_input_and_applies_noop_normalization() -> None:
    tracker = track({"nested": {"value": 1}})
    replacement = {"nested": {"value": 2}}
    prepared = tracker.prepare_replace(replacement)
    assert prepared.value is replacement
    assert prepared.ops == [("r", replacement)]
    tracker.adopt(prepared)

    no_op = tracker.prepare_replace({"nested": {"value": 2}})
    assert no_op.value is tracker.value
    assert no_op.ops == []


def test_shares_immutable_operation_placements_with_the_prepared_candidate() -> None:
    tracker = track({"rows": []})
    change = tracker.begin_change()
    change.state["rows"].append({"id": 1})
    prepared = change.prepare()
    splice = prepared.ops[0]
    assert splice[0] == "p"
    assert splice[4][0] is prepared.value["rows"][0]


def test_rejects_foreign_stale_and_repeated_preparations_while_allowing_competing_changes() -> None:
    first = track({"value": 0})
    second = track({"value": 0})
    change = first.begin_change()
    competing = first.begin_change()
    change.state["value"] = 1
    competing.state["value"] = 2
    prepared = change.prepare()
    competing_prepared = competing.prepare()
    with pytest.raises(RuntimeError, match="different tracker"):
        second.adopt(prepared)
    first.adopt(prepared)
    with pytest.raises(RuntimeError, match="already been used"):
        first.adopt(prepared)
    with pytest.raises(RuntimeError, match="stale"):
        first.adopt(competing_prepared)

    stale = first.prepare_replace({"value": 2})
    winner = first.prepare_replace({"value": 3})
    first.adopt(winner)
    with pytest.raises(RuntimeError, match="stale"):
        first.adopt(stale)
    with pytest.raises(RuntimeError, match="stale"):
        first.adopt(stale)


def test_invalidates_a_prepared_result_when_its_change_is_aborted() -> None:
    tracker = track({"value": 0})
    change = tracker.begin_change()
    change.state["value"] = 1
    prepared = change.prepare()
    change.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        tracker.adopt(prepared)
    change.abort()


def test_makes_competing_same_base_preparations_stale_after_adopting_a_noop() -> None:
    tracker = track({"value": {"count": 1}})
    first = tracker.prepare_replace({"value": {"count": 1}})
    competing = tracker.prepare_replace({"value": {"count": 1}})
    assert first.value is first.base
    assert competing.value is competing.base
    tracker.adopt(first)
    with pytest.raises(RuntimeError, match="already been used"):
        tracker.adopt(first)
    with pytest.raises(RuntimeError, match="stale"):
        tracker.adopt(competing)


def test_emits_an_owned_root_replacement_without_traversing_large_replacement_input() -> None:
    rows = [{"value": value, "stable": {"value": value}} for value in range(10_000)]
    tracker = track({"rows": rows})
    replacement = {"rows": tracker.value["rows"][:]}
    replacement["rows"][5_000] = {"value": -1, "stable": replacement["rows"][5_000]["stable"]}
    prepared = tracker.prepare_replace(replacement)
    assert prepared.value is replacement
    assert prepared.ops == [("r", replacement)]
    assert apply_immutable(prepared.base, prepared.ops) == prepared.value


# ─── canonical strings ───────────────────────────────────────────────────────


def test_emits_append_and_rolling_window_operations() -> None:
    tracker = track({"text": "abcdefgh"})
    change = tracker.begin_change()
    change.state["text"] += "ij"
    prepared = change.prepare()
    assert prepared.ops == [("a", ("text",), "ij")]
    tracker.adopt(prepared)

    change = tracker.begin_change()
    change.state["text"] = f"{change.state['text'][3:]}xyz"
    prepared = change.prepare()
    assert prepared.ops == [("t", ("text",), 3), ("a", ("text",), "xyz")]


def test_finds_bounded_overlaps() -> None:
    assert overlap("abcdefgh", "defghxyz", 65_536) == 5
    assert overlap("abcdef", "defghi", 0) == 0


def test_counts_truncation_in_utf16_code_units() -> None:
    tracker = track({"text": "😀ab"})
    change = tracker.begin_change()
    change.state["text"] = "abc"
    prepared = change.prepare()
    # The emoji is two UTF-16 code units, as a JavaScript replica counts it.
    assert prepared.ops == [("t", ("text",), 2), ("a", ("text",), "c")]
    assert prepared.value == {"text": "abc"}
    assert apply({"text": "😀ab"}, prepared.ops) == {"text": "abc"}


# ─── operation application and validation ────────────────────────────────────


def test_applies_mutable_and_immutable_operations() -> None:
    operations: list[Any] = [
        ("a", ("text",), "b"),
        ("p", ("values",), 1, 1, [3, 4]),
        ("s", ("nested", "value"), 2),
    ]
    base = {"text": "a", "values": [1, 2], "nested": {"value": 1}, "stable": {"value": 9}}
    immutable = apply_immutable(base, operations)
    assert immutable == {"text": "ab", "values": [1, 3, 4], "nested": {"value": 2}, "stable": {"value": 9}}
    assert base == {"text": "a", "values": [1, 2], "nested": {"value": 1}, "stable": {"value": 9}}
    assert immutable["stable"] is base["stable"]
    assert apply(copy.deepcopy(base), operations) == immutable


def test_supports_root_replacement_root_splice_and_permutations() -> None:
    base: list[Any] = [("r", [1, 2, 3])]
    assert is_base(base)
    value = apply(None, base)
    value = apply(value, [("p", (), 1, 1, [4])])
    value = apply(value, [("m", (), [2, 0, 1])])
    assert value == [3, 1, 4]


def test_rejects_unsafe_and_malformed_paths() -> None:
    with pytest.raises(UnsafePathError):
        apply({}, [("s", ("constructor", "prototype", "x"), True)])
    with pytest.raises(UnsafePathError):
        apply({"values": [1]}, [("s", ("values", 3), 2)])
    with pytest.raises(Exception):  # noqa: B017
        apply({"value": 1}, [("a", ("value",), "x")])
    with pytest.raises(TypeError):
        assert_valid_op(["s", "value", 1])
    with pytest.raises(TypeError, match="bijection"):
        assert_valid_op(["m", [], [0, 0]])


def test_validates_immutable_operations_before_traversing_the_target() -> None:
    reads = 0

    class Trap(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            nonlocal reads
            reads += 1
            return {}

    with pytest.raises(TypeError, match="path"):
        apply_immutable(Trap(trap={}), [["s", "bad-path", 1]])
    assert reads == 0


def test_validates_decoded_and_wire_vocabularies_separately() -> None:
    assert_valid_op(["s", ["value"], 1])
    with pytest.raises(TypeError):
        assert_valid_op(["s", 1])
    assert_valid_wire_op(["s", 1])
    assert_valid_wire_op(["#", 0, ["value"]])


# ─── codec ───────────────────────────────────────────────────────────────────


def test_interns_paths_omits_adjacent_paths_and_round_trips() -> None:
    path = ("nested", "text")
    enc = encoder()
    dec = decoder()
    first: list[Any] = [("t", path, 1), ("a", path, "x")]
    assert dec.decode(enc.encode(first)) == first
    second: list[Any] = [("a", path, "y")]
    wire = enc.encode(second)
    assert wire == [("#", 0, path), ("a", 0, "y")]
    assert dec.decode(wire) == second


def test_resets_path_dictionaries_on_a_base() -> None:
    enc = encoder()
    path = ("value",)
    enc.encode([("s", path, 1)])
    enc.encode([("s", path, 2)])
    assert enc.encode([("r", {"value": 3})]) == [("r", {"value": 3})]
    assert enc.encode([("s", path, 4)]) == [("s", path, 4)]


def test_rejects_unresolved_short_forms_and_unsafe_interned_paths() -> None:
    with pytest.raises(Exception):  # noqa: B017
        decoder().decode([["a", "x"]])
    wire = [["#", 0, ["__proto__"]], ["s", 0, True]]
    with pytest.raises(UnsafePathError):
        decoder().decode(wire)


def test_omits_an_adjacent_repeated_path() -> None:
    enc = encoder()
    path = ("value",)
    assert enc.encode([("s", path, 1), ("s", path, 2)]) == [("s", path, 1), ("s", 2)]


def test_interns_on_second_use_rather_than_first() -> None:
    enc = encoder()
    path = ("a", "deep")
    assert enc.encode([("a", path, "1")]) == [("a", path, "1")]
    assert enc.encode([("a", path, "2")]) == [("#", 0, path), ("a", 0, "2")]


def test_does_not_collide_paths_containing_null_characters() -> None:
    operations: list[Any] = [("s", ("a\u0000b",), 1), ("s", ("a", "b"), 2)]
    assert decoder().decode(encoder().encode(operations)) == operations


def test_clears_decoder_ids_on_a_base_batch() -> None:
    dec = decoder()
    dec.decode([["#", 0, ["a"]], ["a", 0, "1"]])
    dec.decode([["r", {"a": ""}]])
    with pytest.raises(Exception):  # noqa: B017
        dec.decode([["a", 0, "2"]])


def test_makes_batches_after_a_base_self_contained() -> None:
    enc = encoder()
    path = ("a", "deep")
    enc.encode([("a", path, "1")])
    enc.encode([("a", path, "2")])
    base = enc.encode([("r", {"a": {"deep": "x"}})])
    after = enc.encode([("a", path, "3")])
    assert after == [("a", path, "3")]
    dec = decoder()
    assert dec.decode(base) == [("r", {"a": {"deep": "x"}})]
    assert dec.decode(after) == [("a", path, "3")]


def test_round_trips_deterministic_mixed_operation_streams() -> None:
    batches: list[list[Any]] = [
        [
            ("s", ("rows", index, "value"), index),
            ("a", ("output",), str(index)),
            ("p", ("tail",), index, 0, [index]),
        ]
        for index in range(100)
    ]
    enc = encoder()
    dec = decoder()
    assert [dec.decode(enc.encode(batch)) for batch in batches] == batches


def test_round_trips_through_json() -> None:
    operations: list[Any] = [("s", ("a", 0), 1), ("s", ("a", 0), 2), ("p", (), 0, 0, [1])]
    wire = json.loads(json.dumps(encoder().encode(operations)))
    assert decoder().decode(wire) == operations


# ─── immutable application ownership ─────────────────────────────────────────


def test_does_not_mutate_a_replacement_payload_targeted_by_a_later_operation() -> None:
    replacement = {"nested": {"value": 1}}
    following = apply_immutable(None, [("r", replacement), ("s", ("nested", "value"), 2)])
    assert replacement["nested"]["value"] == 1
    assert following["nested"]["value"] == 2


def test_adopts_a_mutable_replacement_payload_rather_than_copying_it() -> None:
    batch: list[Any] = [("r", {"value": 0})]
    first = apply(None, batch)
    second = apply(None, batch)
    first["value"] = 1
    assert second["value"] == 1


# ─── path and prototype safety ───────────────────────────────────────────────


def test_rejects_constructor_walks_and_forbidden_interned_paths() -> None:
    with pytest.raises(UnsafePathError):
        apply({}, json.loads('[["s",["constructor","prototype","gadget"],true]]'))
    wire = [["#", 0, ["__proto__", "w"]], ["s", 0, True]]
    with pytest.raises(UnsafePathError):
        decoder().decode(wire)


def test_allows_reserved_names_inside_values() -> None:
    value = json.loads('{"__proto__":{"z":1}}')
    out = apply({}, [("s", ("value",), value)])
    assert "__proto__" in out["value"]


# ─── array index safety ──────────────────────────────────────────────────────


def test_writes_an_existing_index_and_appends_exactly_one_past_the_end() -> None:
    assert apply({"values": [1, 2, 3]}, [("s", ("values", 1), 9)]) == {"values": [1, 9, 3]}
    assert apply({"values": [1, 2, 3]}, [("s", ("values", 3), 9)]) == {"values": [1, 2, 3, 9]}


def test_rejects_gaps_huge_indices_and_string_spelled_indices() -> None:
    with pytest.raises(UnsafePathError):
        apply({"values": [1, 2, 3]}, [("s", ("values", 5), 9)])
    with pytest.raises(UnsafePathError):
        apply({"values": []}, [("s", ("values", 4_294_967_290), 1)])
    with pytest.raises(UnsafePathError):
        apply({"values": [1]}, [("s", ("values", "0"), 9)])
    with pytest.raises(UnsafePathError):
        apply({"values": ["a"]}, [("a", ("values", "0"), "b")])
    with pytest.raises(UnsafePathError):
        apply({"values": [1]}, [("s", ("values", True), 9)])


def test_allows_explicit_growth_values_and_rejects_deletion_past_the_end() -> None:
    assert apply({"values": [1]}, [("p", ("values",), 1, 0, [None, None, 9])]) == {"values": [1, None, None, 9]}
    with pytest.raises(Exception):  # noqa: B017
        apply({"values": [1]}, [("d", ("values", 1))])


def test_applies_large_splice_payloads() -> None:
    items = [None] * 300_000
    result = apply({"values": []}, [("p", ("values",), 0, 0, items)])
    assert len(result["values"]) == len(items)


# ─── operation structure safety ──────────────────────────────────────────────


def test_rejects_unknown_verbs_malformed_tuples_paths_and_splice_payloads() -> None:
    for op in (
        ["ZZZ", ["value"], 9],
        ["p", ["values"], 0, 0, "not-an-array"],
        ["s", "value", 9],
        {"op": "s"},
        None,
        "s",
    ):
        with pytest.raises(TypeError):
            apply({"value": 1, "values": [1]}, [op])


def test_rejects_invalid_append_and_truncation_operations() -> None:
    with pytest.raises(Exception):  # noqa: B017
        apply({"value": 1}, [("a", ("missing",), "x")])
    with pytest.raises(Exception):  # noqa: B017
        apply({"value": 1}, [("a", ("value",), "x")])
    with pytest.raises(TypeError):
        apply({"value": "abc"}, [("t", ("value",), -1)])
    with pytest.raises(TypeError):
        decoder().decode([["t", ["value"], -1]])


def test_clamps_splice_removal_past_the_end() -> None:
    assert apply({"values": [1, 2]}, [("p", ("values",), 0, 10**9, [])]) == {"values": []}


# ─── operation assertions ────────────────────────────────────────────────────


def test_accepts_decoded_operations_and_rejects_wire_only_forms() -> None:
    for operation in (
        ["r", {"value": 1}],
        ["s", ["value"], 1],
        ["d", ["value"]],
        ["a", ["value"], "x"],
        ["t", ["value"], 2],
        ["p", ["value"], 0, 0, []],
        ["m", ["value"], [0]],
    ):
        assert_valid_op(operation)
    for wire_only in (
        ["s", 1],
        ["d"],
        ["a", "x"],
        ["t", 2],
        ["p", 0, 0, []],
        ["#", 0, ["value"]],
        ["s", 0, 1],
    ):
        with pytest.raises(TypeError):
            assert_valid_op(wire_only)
        assert_valid_wire_op(wire_only)


def test_does_not_recursively_inspect_operation_payloads() -> None:
    assert_valid_op(["s", ["value"], {1, 2}])
    assert_valid_wire_op(["r", object()])
