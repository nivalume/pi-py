"""Port of chord ``test/delta-tracker/tracker.test.ts``.

JavaScript-only cases are dropped: Promise ``then`` probing of drafts, integer-like key
ordering and null prototypes (Python dicts keep insertion order and have no prototypes),
property descriptors, borrowed array methods applied to other receivers, ``valueOf``
argument coercion, inherited setters, and ``length`` writes (Python lists have no length
setter; growth uses ``extend``, shrinking ``del values[n:]``). ``fill`` and ``copyWithin``
have no Python list counterpart; their cases use the equivalent slice assignments, which
record the same splices. ``sort`` takes a ``functools.cmp_to_key`` comparator where
upstream passes a comparator.
"""

from __future__ import annotations

import asyncio
import json
from functools import cmp_to_key
from typing import Any

import pytest

from pi_chord.delta import Draft, DraftDict, DraftList, Tracker, apply, decoder, encoder, track


def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def replay(base: Any, operations: Any) -> Any:
    return apply(clone(base), clone(operations))


def is_draft(value: Any) -> bool:
    return isinstance(value, (DraftDict, DraftList))


def expect_plain(value: Any) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if is_draft(current):
            raise AssertionError("revision contains a draft")
        if type(current) is dict:
            pending.extend(current.values())
        elif type(current) in (list, tuple):
            pending.extend(current)


def expect_alias_free(value: Any) -> None:
    seen: dict[int, str] = {}

    def visit(current: Any, path: str) -> None:
        if type(current) not in (dict, list):
            return
        previous = seen.get(id(current))
        if previous is not None:
            raise AssertionError(f"container at {path} aliases {previous}")
        seen[id(current)] = path
        if type(current) is list:
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")
            return
        for key, item in current.items():
            visit(item, f"{path}.{key}")

    visit(value, "$root")


def settle(tracker: Tracker[Any], mutate: Any) -> Any:
    base_root = tracker.value
    base = clone(base_root)
    change = tracker.begin_change()
    mutate(change.state)
    prepared = change.prepare()
    candidate = clone(prepared.value)
    expect_plain(prepared.value)
    assert replay(base, prepared.ops) == candidate
    assert base_root == base
    tracker.adopt(prepared)
    assert base_root == base
    assert tracker.value is prepared.value
    assert tracker.value == candidate
    return candidate


# ─── transactional overlay lifecycle ─────────────────────────────────────────


def test_materializes_an_immutable_next_revision_and_adopts_it_by_pointer_swap() -> None:
    initial: dict[str, Any] = {"count": 1, "nested": {"text": "a"}, "values": [1]}
    tracker = track(initial)
    assert tracker.value is initial
    change = tracker.begin_change()
    change.state["count"] = 2
    change.state["nested"]["text"] += "b"
    change.state["values"].append(2)
    assert initial == {"count": 1, "nested": {"text": "a"}, "values": [1]}
    prepared = change.prepare()
    following = prepared.value
    assert prepared.base_revision == 0
    assert prepared.base is initial
    assert following is not initial
    expect_plain(following)
    assert following == {"count": 2, "nested": {"text": "ab"}, "values": [1, 2]}
    assert prepared.ops == [
        ("s", ("count",), 2),
        ("a", ("nested", "text"), "b"),
        ("p", ("values",), 1, 0, [2]),
    ]
    assert tracker.value is initial
    tracker.adopt(prepared)
    assert initial == {"count": 1, "nested": {"text": "a"}, "values": [1]}
    assert tracker.value is following
    assert prepared.value is following
    assert prepared.base is initial
    with pytest.raises(TypeError):
        change.state["count"]


async def test_keeps_a_transaction_live_across_await_and_seals_it_only_at_prepare() -> None:
    tracker = track({"left": 0, "nested": {"right": 0}})
    change = tracker.begin_change()
    change.state["left"] = 1
    await asyncio.sleep(0)
    change.state["nested"]["right"] = 2
    prepared = change.prepare()
    assert prepared.value == {"left": 1, "nested": {"right": 2}}
    with pytest.raises(TypeError, match="settled"):
        change.state["left"] = 3
    tracker.adopt(prepared)


def test_revokes_held_drafts_of_a_large_change() -> None:
    tracker = track({"rows": [{"value": value} for value in range(4_100)]})
    change = tracker.begin_change()
    draft = change.state
    for row in draft["rows"]:
        assert row["value"] >= 0
    draft["rows"][0]["value"] = -1
    prepared = change.prepare()
    with pytest.raises(TypeError, match="settled"):
        draft["rows"]
    tracker.adopt(prepared)


def test_aborts_idempotently_and_revokes_held_descendants() -> None:
    tracker = track({"child": {"value": 1}})
    change = tracker.begin_change()
    child = change.state["child"]
    child["value"] = 2
    change.abort()
    change.abort()
    assert tracker.value["child"]["value"] == 1
    with pytest.raises(TypeError):
        child["value"]
    with pytest.raises(RuntimeError, match="settled"):
        change.prepare()


def test_allows_competing_contexts_and_invalidates_loser_views() -> None:
    tracker = track({"value": 0})
    first = tracker.begin_change()
    second = tracker.begin_change()
    first.state["value"] = 1
    second.state["value"] = 2
    first_prepared = first.prepare()
    second_prepared = second.prepare()
    held = second_prepared.value
    tracker.adopt(first_prepared)
    assert tracker.value["value"] == 1
    assert held["value"] == 2
    with pytest.raises(RuntimeError, match="stale"):
        tracker.adopt(second_prepared)
    with pytest.raises(RuntimeError, match="already been used"):
        tracker.adopt(first_prepared)


def test_invalidates_competing_open_overlays_without_changing_their_base_revision() -> None:
    initial = {"rows": [{"value": 0}, {"value": 1}]}
    tracker = track(initial)
    stale_change = tracker.begin_change()
    held = stale_change.state["rows"][1]
    held["value"] = 2

    winner = tracker.begin_change()
    winner.state["rows"][0]["value"] = 3
    tracker.adopt(winner.prepare())

    assert initial == {"rows": [{"value": 0}, {"value": 1}]}
    assert tracker.value == {"rows": [{"value": 3}, {"value": 1}]}
    with pytest.raises(TypeError, match="settled"):
        held["value"]
    with pytest.raises(TypeError, match="settled"):
        stale_change.prepare()
    stale_change.abort()


def test_adopts_object_edits_as_ordinary_revisions() -> None:
    initial: dict[str, Any] = {"first": 1, "second": 2}
    tracker = track(initial)
    change = tracker.begin_change()
    change.state["first"] = 3
    del change.state["second"]
    change.state["third"] = 4
    tracker.adopt(change.prepare())
    assert tracker.value is not initial
    assert initial == {"first": 1, "second": 2}
    assert tracker.value == {"first": 3, "third": 4}
    assert type(tracker.value) is dict


def test_supports_adopt_then_publish_ordering_and_retained_publication_reads() -> None:
    tracker = track({"value": 0, "nested": {"count": 0}})
    change = tracker.begin_change()
    change.state["value"] = 1
    change.state["nested"]["count"] = 1
    prepared = change.prepare()
    operations = prepared.ops
    tracker.adopt(prepared)
    assert prepared.ops is operations
    published = None if len(prepared.ops) == 0 else prepared.value
    assert len(operations) == 2
    assert published is tracker.value
    assert clone(published) == {"value": 1, "nested": {"count": 1}}

    following = tracker.begin_change()
    following.state["value"] = 2
    tracker.adopt(following.prepare())
    assert published is not tracker.value
    assert prepared.value == {"value": 1, "nested": {"count": 1}}
    assert tracker.value == {"value": 2, "nested": {"count": 1}}


def test_keeps_replacement_base_and_value_readable_after_adoption() -> None:
    original = {"value": 0}
    replacement = {"value": 1}
    tracker = track(original)
    prepared = tracker.prepare_replace(replacement)
    assert prepared.base is original
    assert prepared.value is replacement
    tracker.adopt(prepared)
    assert prepared.base is original
    assert prepared.value is replacement
    assert prepared.value is tracker.value


def test_keeps_aborted_and_stale_materialized_candidates_readable() -> None:
    tracker = track({"value": 0})
    aborted_change = tracker.begin_change()
    aborted_change.state["value"] = 1
    aborted = aborted_change.prepare()
    aborted_view = aborted.value
    aborted.abort()
    assert aborted_view["value"] == 1
    assert aborted.value["value"] == 1

    loser_change = tracker.begin_change()
    loser_change.state["value"] = 2
    loser = loser_change.prepare()
    loser_view = loser.value
    tracker.adopt(tracker.prepare_replace({"value": 3}))
    assert loser_view["value"] == 2
    assert loser.value["value"] == 2


def test_lets_a_settled_change_abort_its_prepared_result() -> None:
    tracker = track({"value": 0, "nested": {"value": 1}})
    change = tracker.begin_change()
    change.state["value"] = 1
    prepared = change.prepare()
    operations = prepared.ops
    change.abort()
    change.abort()
    assert prepared.ops is operations
    assert prepared.value == {"value": 1, "nested": {"value": 1}}
    with pytest.raises(RuntimeError, match="aborted"):
        tracker.adopt(prepared)


def test_rejects_foreign_and_aborted_prepared_values() -> None:
    first = track({"value": 0})
    second = track({"value": 0})
    prepared = first.prepare_replace({"value": 1})
    with pytest.raises(RuntimeError, match="different tracker"):
        second.adopt(prepared)
    prepared.abort()
    prepared.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        first.adopt(prepared)


def test_reads_deleted_keys_as_absent() -> None:
    tracker = track({"value": 1})
    change = tracker.begin_change()
    del change.state["value"]
    change.state["extra"] = 1
    del change.state["extra"]
    assert "value" not in change.state
    assert change.state.get("extra") is None
    with pytest.raises(KeyError):
        change.state["value"]
    prepared = change.prepare()
    assert prepared.ops == [("d", ("value",))]
    tracker.adopt(prepared)
    assert list(tracker.value) == []


def test_detaches_old_handles_for_deeply_equal_object_and_array_assignments_before_normalizing() -> None:
    initial = {"object": {"nested": {"value": 1}}, "array": [{"value": 1}], "rows": [{"value": 1}]}
    tracker = track(initial)
    change = tracker.begin_change()
    old_object = change.state["object"]
    old_array = change.state["array"]
    old_row = change.state["rows"][0]
    change.state["object"] = {"nested": {"value": 1}}
    change.state["array"] = [{"value": 1}]
    change.state["rows"][0] = {"value": 1}
    assert change.state["object"] is not old_object
    assert change.state["array"] is not old_array
    assert change.state["rows"][0] is not old_row
    old_object["nested"]["value"] = 9
    old_array[0]["value"] = 9
    old_row["value"] = 9
    assert change.state == {"object": {"nested": {"value": 1}}, "array": [{"value": 1}], "rows": [{"value": 1}]}
    prepared = change.prepare()
    assert prepared.ops == []
    tracker.adopt(prepared)
    assert tracker.value is initial
    assert tracker.value["object"] is initial["object"]
    assert tracker.value["array"] is initial["array"]
    assert tracker.value["rows"][0] is initial["rows"][0]


def test_normalizes_a_deeply_equal_replacement_without_changing_committed_identity() -> None:
    initial = {"nested": {"value": 1}, "rows": [1, 2, 3]}
    tracker = track(initial)
    prepared = tracker.prepare_replace({"nested": {"value": 1}, "rows": [1, 2, 3]})
    assert prepared.ops == []
    tracker.adopt(prepared)
    assert tracker.value is initial


def test_distinguishes_json_types_in_noop_normalization() -> None:
    tracker = track({"flag": 1, "ratio": 1, "nested": {"value": 1}})
    change = tracker.begin_change()
    change.state["flag"] = True
    change.state["ratio"] = 1.0
    change.state["nested"] = {"value": True}
    prepared = change.prepare()
    assert prepared.ops == [
        ("s", ("flag",), True),
        ("s", ("ratio",), 1.0),
        ("s", ("nested",), {"value": True}),
    ]
    assert tracker.prepare_replace({"flag": True, "ratio": 1, "nested": {"value": 1}}).ops != []


def test_takes_o1_immutable_ownership_for_replacement_and_its_operation_payload() -> None:
    tracker = track({"value": 0, "rows": []})
    replacement = {"value": 1, "rows": [{"value": 2}]}
    prepared = tracker.prepare_replace(replacement)
    assert prepared.base is tracker.value
    assert prepared.value is replacement
    operations = prepared.ops
    assert operations == [("r", replacement)]
    assert operations[0][1] is replacement
    tracker.adopt(prepared)
    assert tracker.value is replacement


# ─── policy view ─────────────────────────────────────────────────────────────


def test_supports_native_reads_keys_iteration_map_and_json() -> None:
    tracker = track({"values": [1, 2, 3], "object": {"a": 1}})
    change = tracker.begin_change()
    change.state["values"][1:2] = [4, 5]
    change.state["object"]["b"] = 2
    values = change.state["values"]
    assert isinstance(values, list)
    assert isinstance(change.state, dict)
    assert len(values) == 4
    assert [*values] == [1, 4, 5, 3]
    assert [value * 2 for value in values] == [2, 8, 10, 6]
    assert values[2] == 5
    assert values[-1] == 3
    assert values[1:3] == [4, 5]
    assert 5 in values
    assert values.index(5) == 2
    assert list(change.state["object"]) == ["a", "b"]
    assert dict(change.state["object"]) == {"a": 1, "b": 2}
    assert {**change.state["object"]} == {"a": 1, "b": 2}
    assert json.loads(json.dumps(change.state)) == {"values": [1, 4, 5, 3], "object": {"a": 1, "b": 2}}
    change.abort()


def test_keeps_document_keys_distinct_from_draft_fields() -> None:
    tracker = track({"object": {"_ctx": 1, "_base": 2, "_parent": 3, "_dirty": 4, "_writes": 5, "_key": 6}})
    change = tracker.begin_change()
    change.state["object"]["_ctx"] = 7
    change.state["object"]["_key"] = 8
    assert list(change.state["object"]) == ["_ctx", "_base", "_parent", "_dirty", "_writes", "_key"]
    prepared = change.prepare()
    assert prepared.value["object"] == {"_ctx": 7, "_base": 2, "_parent": 3, "_dirty": 4, "_writes": 5, "_key": 8}
    tracker.adopt(prepared)


def test_keeps_string_operation_forms() -> None:
    tracker = track({"text": "abcdefgh", "values": [1, 2, 3], "marker": 0})
    change = tracker.begin_change()
    change.state["text"] = f"{change.state['text'][3:]}xyz"
    change.state["values"][1:2] = [4]
    change.state["marker"] = 1
    prepared = change.prepare()
    assert prepared.value == {"text": "defghxyz", "values": [1, 4, 3], "marker": 1}
    assert ("t", ("text",), 3) in prepared.ops
    assert ("a", ("text",), "xyz") in prepared.ops
    assert replay(tracker.value, decoder().decode(encoder().encode(prepared.ops))) == prepared.value
    tracker.adopt(prepared)


def test_emits_deletion_and_supports_root_arrays() -> None:
    object_tracker = track({"keep": 1, "remove": 2})
    settle(object_tracker, lambda draft: draft.pop("remove"))
    assert object_tracker.value == {"keep": 1}

    array_tracker = track([1, 2, 3])

    def mutate(draft: Any) -> None:
        draft.reverse()
        draft.append(4)

    settle(array_tracker, mutate)
    assert array_tracker.value == [3, 2, 1, 4]


def test_preserves_key_order_after_delete_and_readd() -> None:
    tracker = track({"first": 1, "second": 2})
    change = tracker.begin_change()
    del change.state["first"]
    change.state["first"] = 1
    assert list(change.state) == ["second", "first"]
    prepared = change.prepare()
    assert prepared.ops == [("d", ("first",)), ("s", ("first",), 1)]
    tracker.adopt(prepared)
    assert list(tracker.value) == ["second", "first"]
    assert list(replay({"first": 1, "second": 2}, prepared.ops)) == ["second", "first"]


def test_normalizes_deeply_equal_container_assignments_before_materialization() -> None:
    initial = {"child": {"a": 1, "b": 2}, "count": 0}
    tracker = track(initial)
    change = tracker.begin_change()
    change.state["child"] = {"b": 2, "a": 1}
    change.state["count"] = 1
    prepared = change.prepare()
    replayed = replay(initial, prepared.ops)
    assert prepared.ops == [("s", ("count",), 1)]
    assert list(prepared.value["child"]) == ["a", "b"]
    assert list(replayed["child"]) == ["a", "b"]
    assert prepared.value == replayed
    assert prepared.value["child"] is initial["child"]
    tracker.adopt(prepared)


def test_keeps_ordering_across_deletion_and_readdition() -> None:
    tracker = track({"object": {"1": "one", "2": "two", "first": "a", "second": "b"}})
    change = tracker.begin_change()
    obj = change.state["object"]
    del obj["2"]
    obj["2"] = "two"
    del obj["first"]
    obj["first"] = "a"
    obj["3"] = "three"
    obj["0"] = "zero"
    expected = ["1", "second", "2", "first", "3", "0"]
    assert list(obj) == expected
    prepared = change.prepare()
    assert list(prepared.value["object"]) == expected
    assert list(replay({"object": {"1": "one", "2": "two", "first": "a", "second": "b"}}, prepared.ops)["object"]) == (
        expected
    )
    tracker.adopt(prepared)
    assert list(tracker.value["object"]) == expected


def test_keeps_repeated_identical_writes_instead_of_reverting_their_pending_override() -> None:
    tracker = track({"value": 0, "values": [0]})

    def mutate(draft: Any) -> None:
        draft["value"] = 1
        draft["value"] = 1
        draft["values"][0] = None
        draft["values"][0] = None
        draft["values"].append(2)
        draft["values"][1] = None
        draft["values"][1] = None

    settle(tracker, mutate)
    assert tracker.value == {"value": 1, "values": [None, None]}


def test_rejects_out_of_range_index_writes_without_mutation_while_allowing_replacement_and_append() -> None:
    tracker = track({"values": [1, 2]})
    change = tracker.begin_change()
    with pytest.raises(IndexError):
        change.state["values"][3] = 4
    with pytest.raises(IndexError):
        change.state["values"][2] = 4
    assert change.state["values"] == [1, 2]
    assert tracker.value["values"] == [1, 2]

    change.state["values"][1] = 9
    change.state["values"].append(3)
    prepared = change.prepare()
    assert prepared.value["values"] == [1, 9, 3]
    assert replay(tracker.value, prepared.ops) == prepared.value
    tracker.adopt(prepared)
    assert tracker.value["values"] == [1, 9, 3]


def test_supports_optional_deletion_null_growth_shrinking_and_default_sort() -> None:
    tracker = track({"optional": "remove", "values": [3, 1, 2]})

    def mutate(draft: Any) -> None:
        from pi_chord.json import UNDEFINED

        draft["optional"] = UNDEFINED
        draft["values"].extend([None, None])
        assert draft["values"] == [3, 1, 2, None, None]
        del draft["values"][4:]
        draft["values"][:3] = sorted(draft["values"][:3])
        del draft["values"][0]
        draft["values"].insert(0, 1)

    settle(tracker, mutate)
    assert tracker.value == {"values": [1, 2, 3, None]}


def test_sorts_with_python_defaults_and_reverse() -> None:
    tracker = track({"values": [3, 1, 2]})
    settle(tracker, lambda draft: draft["values"].sort())
    assert tracker.value["values"] == [1, 2, 3]
    settle(tracker, lambda draft: draft["values"].sort(reverse=True))
    assert tracker.value["values"] == [3, 2, 1]


def test_enumerates_wide_objects_without_duplicate_keys() -> None:
    tracker = track({"values": {}})
    change = tracker.begin_change()
    for index in range(20_000):
        change.state["values"][f"field{index}"] = index
    keys = list(change.state["values"])
    assert len(keys) == 20_000
    assert keys[0] == "field0"
    assert keys[-1] == "field19999"
    change.abort()


def test_does_not_dirty_read_only_traversals() -> None:
    tracker = track({"nested": {"rows": [{"value": 1}]}})
    change = tracker.begin_change()
    assert change.state["nested"]["rows"][0]["value"] == 1
    prepared = change.prepare()
    assert prepared.ops == []
    tracker.adopt(prepared)


def test_uses_one_draft_identity_per_accessed_container() -> None:
    tracker = track({"nested": {"value": 1}, "rows": [{"value": 2}]})
    change = tracker.begin_change()
    assert change.state["nested"] is change.state["nested"]
    assert change.state["rows"] is change.state["rows"]
    assert change.state["rows"][0] is change.state["rows"][0]
    assert next(iter(change.state["rows"])) is change.state["rows"][0]
    assert change.state.get("nested") is change.state["nested"]
    change.abort()


def test_dict_mutators_record_operations() -> None:
    tracker = track({"a": 1, "b": {"c": 2}, "d": 3, "e": 4})

    def mutate(draft: Any) -> None:
        assert draft.setdefault("a", 9) == 1
        assert draft.setdefault("f", {"g": 5}) == {"g": 5}
        draft.update({"a": 2}, h=6)
        draft |= {"i": 7}
        assert draft.pop("d") == 3
        assert draft.pop("missing", None) is None
        assert draft.popitem() == ("i", 7)
        draft["b"].clear()

    settle(tracker, mutate)
    assert tracker.value == {"a": 2, "b": {}, "e": 4, "f": {"g": 5}, "h": 6}


def test_list_mutators_record_operations() -> None:
    def mutate(document: Any) -> None:
        values = document["values"]
        values.remove(3)
        values += [6]
        values.insert(-1, 7)
        values.insert(100, 8)
        del values[::3]
        values[::2] = [10, 20]
        values *= 2
        assert values.pop(0) == 10
        values.extend(value for value in [11, 12])
        values.remove(12)

    expected = {"values": [1, 2, 3, 4, 5]}
    mutate(expected)
    tracker = track({"values": [1, 2, 3, 4, 5]})
    settle(tracker, mutate)
    assert tracker.value == expected


def test_plain_copies_of_drafts() -> None:
    tracker = track({"rows": [{"value": 1}], "meta": {"a": 1}})
    change = tracker.begin_change()
    copied = change.state["meta"].copy()
    assert type(copied) is dict
    plain = [*change.state["rows"]]
    assert plain[0] is change.state["rows"][0]
    combined = [0] + change.state["rows"]
    assert type(combined) is list
    assert combined[1] is change.state["rows"][0]
    from pi_chord.delta import detach

    detached = detach(change.state)
    assert type(detached["rows"][0]) is dict
    assert detached == {"rows": [{"value": 1}], "meta": {"a": 1}}
    change.abort()
    with pytest.raises(TypeError, match="settled"):
        detach(plain[0])


# ─── by-value placements ─────────────────────────────────────────────────────


def test_clones_property_index_append_insert_and_slice_placements() -> None:
    tracker = track({"property": None, "values": [{"value": 0}, {"value": 1}, {"value": 2}]})
    external = {"value": 5}
    change = tracker.begin_change()
    change.state["property"] = external
    change.state["values"][0] = external
    change.state["values"].append(external)
    change.state["values"].insert(0, external)
    change.state["values"][2:2] = [external]
    external["value"] = 99
    first = change.state["values"][0]
    change.state["values"][1:3] = [first, first]
    change.state["values"][3:5] = change.state["values"][0:2]
    prepared = change.prepare()
    candidate = clone(prepared.value)
    assert candidate["property"] == {"value": 5}
    assert candidate["values"][:5] == [{"value": 5}] * 5
    tracker.adopt(prepared)


def test_rejects_non_strict_json_placements_before_changing_the_draft() -> None:
    initial: dict[str, Any] = {"optional": "remove", "payload": None, "number": 0, "values": [1, 2]}
    tracker = track(initial)
    change = tracker.begin_change()
    from pi_chord.json import UNDEFINED

    with pytest.raises(TypeError, match="strict JSON"):
        change.state["values"][0] = UNDEFINED
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["values"].extend([{"staged": True}, UNDEFINED])
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["values"].insert(0, UNDEFINED)
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["values"][1:1] = [UNDEFINED]
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["values"].append(UNDEFINED)
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["payload"] = {"nested": UNDEFINED}
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["payload"] = {"nested": lambda: 1}
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["payload"] = {"nested": (1, 2)}
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["payload"] = {1: "non-string key"}
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["number"] = float("nan")
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["number"] = float("inf")
    with pytest.raises(TypeError, match="plain objects"):
        change.state["payload"] = object()

    class Mapping(dict[str, Any]):
        pass

    with pytest.raises(TypeError, match="plain objects"):
        change.state["payload"] = Mapping(value=1)
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(TypeError, match="cycles"):
        change.state["payload"] = cyclic
    with pytest.raises(TypeError, match="strings"):
        change.state[1] = 1  # type: ignore[index]

    assert change.state == initial
    change.state["optional"] = UNDEFINED
    change.state["payload"] = {"valid": True}
    change.state["values"].append({"valid": True})
    prepared = change.prepare()
    assert prepared.value == {"payload": {"valid": True}, "number": 0, "values": [1, 2, {"valid": True}]}
    assert replay(initial, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_expands_repeated_source_aliases_into_independent_placements() -> None:
    tracker = track({"left": None, "right": None, "rows": []})
    shared = {"nested": {"value": 1}}
    change = tracker.begin_change()
    change.state["left"] = shared
    change.state["right"] = shared
    change.state["rows"].extend([shared, shared])
    change.state["left"]["nested"]["value"] = 9
    change.state["rows"][0]["nested"]["value"] = 8
    prepared = change.prepare()
    assert prepared.value == {
        "left": {"nested": {"value": 9}},
        "right": {"nested": {"value": 1}},
        "rows": [{"nested": {"value": 8}}, {"nested": {"value": 1}}],
    }
    assert prepared.value["left"] is not prepared.value["right"]
    assert prepared.value["rows"][0] is not prepared.value["rows"][1]
    assert replay(tracker.value, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_deep_clones_draft_sourced_placements_within_and_across_revisions() -> None:
    tracker = track(
        {"a": {"child": {"value": 1}}, "b": None, "rows": [{"child": {"value": 1}}, {"child": {"value": 2}}]}
    )
    first = tracker.begin_change()
    first.state["b"] = first.state["a"]
    first.state["b"]["child"]["value"] = 3
    first.state["rows"][1] = first.state["rows"][0]
    first_prepared = first.prepare()
    value = first_prepared.value
    assert value["a"]["child"]["value"] == 1
    assert value["b"]["child"]["value"] == 3
    assert value["a"] is not value["b"]
    assert value["a"]["child"] is not value["b"]["child"]
    assert value["rows"][0] is not value["rows"][1]
    assert value["rows"][0]["child"] is not value["rows"][1]["child"]
    expect_alias_free(value)
    tracker.adopt(first_prepared)

    base = clone(tracker.value)
    second = tracker.begin_change()
    second.state["rows"][1]["child"]["value"] = 4
    second_prepared = second.prepare()
    assert [row["child"]["value"] for row in second_prepared.value["rows"]] == [1, 4]
    assert replay(base, second_prepared.ops) == second_prepared.value
    expect_alias_free(second_prepared.value)
    tracker.adopt(second_prepared)


def test_distinguishes_raw_committed_references_from_draft_references() -> None:
    tracker = track({"source": {"value": 1}, "rawCopy": None, "draftCopy": None})
    raw = tracker.value["source"]
    change = tracker.begin_change()
    change.state["source"]["value"] = 2
    change.state["rawCopy"] = raw
    change.state["draftCopy"] = change.state["source"]
    prepared = change.prepare()
    assert prepared.value == {"source": {"value": 2}, "rawCopy": {"value": 1}, "draftCopy": {"value": 2}}
    tracker.adopt(prepared)


def test_clones_drafts_nested_inside_plain_placements() -> None:
    tracker = track({"source": {"value": 1}, "wrapped": None})
    change = tracker.begin_change()
    change.state["source"]["value"] = 2
    change.state["wrapped"] = {"inner": change.state["source"], "list": [change.state["source"]]}
    prepared = change.prepare()
    assert prepared.value["wrapped"] == {"inner": {"value": 2}, "list": [{"value": 2}]}
    expect_alias_free(prepared.value)


def test_folds_edits_to_introduced_object_and_array_subtrees_into_placement_payloads() -> None:
    tracker = track({"nested": None, "rows": []})
    change = tracker.begin_change()
    change.state["nested"] = {"rows": [{"value": 1}]}
    change.state["nested"]["rows"][0]["value"] = 2
    change.state["nested"]["rows"].append({"value": 3})
    change.state["rows"].append({"values": [1]})
    change.state["rows"][0]["values"].append(2)
    prepared = change.prepare()
    assert prepared.ops == [
        ("s", ("nested",), {"rows": [{"value": 2}, {"value": 3}]}),
        ("p", ("rows",), 0, 0, [{"values": [1, 2]}]),
    ]
    assert replay(tracker.value, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_folds_pathological_operation_counts() -> None:
    wide = {f"field{index}": 0 for index in range(5_000)}
    wide_tracker = track(wide)
    wide_base = clone(wide_tracker.value)
    wide_change = wide_tracker.begin_change()
    for index in range(5_000):
        wide_change.state[f"field{index}"] = 1
    wide_prepared = wide_change.prepare()
    assert len(wide_prepared.ops) == 1
    assert wide_prepared.ops[0][0] == "r"
    assert replay(wide_base, wide_prepared.ops) == wide_prepared.value
    wide_tracker.adopt(wide_prepared)

    sparse_tracker = track({"rows": [{"value": value} for value in range(15_000)]})
    sparse_base = clone(sparse_tracker.value)
    sparse_change = sparse_tracker.begin_change()
    for index in range(0, 15_000, 3):
        sparse_change.state["rows"][index]["value"] = -index - 1
    sparse_prepared = sparse_change.prepare()
    assert len(sparse_prepared.ops) == 1
    assert sparse_prepared.ops[0][0] == "r"
    assert replay(sparse_base, sparse_prepared.ops) == sparse_prepared.value
    sparse_tracker.adopt(sparse_prepared)


def test_shares_immutable_operation_placements_with_the_materialized_candidate() -> None:
    tracker = track({"rows": []})
    change = tracker.begin_change()
    change.state["rows"].append({"value": 1})
    prepared = change.prepare()
    expect_plain(prepared.ops)
    splice = prepared.ops[0]
    assert splice[0] == "p"
    assert splice[4][0] is prepared.value["rows"][0]
    tracker.adopt(prepared)
    assert tracker.value is prepared.value


# ─── piece arrays ────────────────────────────────────────────────────────────


def test_emits_one_insert_for_an_append_to_a_long_array() -> None:
    tracker = track({"values": list(range(10_000))})
    change = tracker.begin_change()
    change.state["values"].append(-1)
    prepared = change.prepare()
    assert prepared.ops == [("p", ("values",), 10_000, 0, [-1])]
    tracker.adopt(prepared)
    assert tracker.value["values"][-1] == -1
    assert len(tracker.value["values"]) == 10_001


def test_emits_removals_from_the_end_backwards_then_inserts_when_kept_elements_keep_their_order() -> None:
    def mutate(draft: Any) -> None:
        values = draft["values"]
        values.pop(0)  # gap before the first kept element
        values[1:3] = [20]  # gap between kept elements, with an insert
        del values[5]  # another gap
        values.pop()  # gap after the last kept element
        values.append(30)

    tracker = track({"values": list(range(10))})
    assert settle(tracker, mutate)["values"] == [1, 20, 4, 5, 6, 8, 30]
    change = track({"values": list(range(10))}).begin_change()
    mutate(change.state)
    assert change.prepare().ops == [
        ("p", ("values",), 9, 1, []),
        ("p", ("values",), 7, 1, []),
        ("p", ("values",), 2, 2, []),
        ("p", ("values",), 0, 1, []),
        ("p", ("values",), 1, 0, [20]),
        ("p", ("values",), 6, 0, [30]),
    ]


def test_emits_a_permutation_when_kept_elements_change_order() -> None:
    def by_parity(values: Any) -> None:
        values.sort(key=lambda value: (value % 2, value))

    for mutate in (lambda values: values.reverse(), by_parity):
        tracker = track({"values": [0, 1, 2, 3, 4]})
        expected = [0, 1, 2, 3, 4]
        mutate(expected)
        change = tracker.begin_change()
        mutate(change.state["values"])
        prepared = change.prepare()
        assert any(operation[0] == "m" for operation in prepared.ops)
        assert replay(tracker.value, prepared.ops) == {"values": expected}


def test_keeps_adoption_independent_from_mutable_detached_permutation_metadata() -> None:
    tracker = track({"values": [3, 1, 2]})
    change = tracker.begin_change()
    change.state["values"].sort()
    prepared = change.prepare()
    permutation = next(operation for operation in prepared.ops if operation[0] == "m")
    permutation[2].reverse()
    tracker.adopt(prepared)
    assert tracker.value["values"] == [1, 2, 3]


def test_matches_native_slice_edits_that_change_nothing() -> None:
    tracker = track({"values": [1, 2, 3]})
    change = tracker.begin_change()
    change.state["values"][1:1] = []
    del change.state["values"][3:]
    prepared = change.prepare()
    assert prepared.value["values"] == [1, 2, 3]
    assert prepared.ops == []
    tracker.adopt(prepared)

    one = track({"values": [1, 2, 3, 4]})
    one_change = one.begin_change()
    assert one_change.state["values"][1:] == [2, 3, 4]
    del one_change.state["values"][1:]
    one_prepared = one_change.prepare()
    assert one_prepared.value["values"] == [1]
    assert replay(one.value, one_prepared.ops) == one_prepared.value
    one.adopt(one_prepared)


def test_supports_all_structural_mutators_and_native_return_values() -> None:
    tracker = track({"values": [3, 1, 2]})
    base = clone(tracker.value)
    change = tracker.begin_change()
    values = change.state["values"]
    assert values.append(4) is None
    assert values.pop() == 4
    values.insert(0, 0)
    assert len(values) == 4
    assert values.pop(0) == 0
    removed = values[1:2]
    values[1:2] = [5, 4]
    assert removed == [1]
    assert values.sort() is None
    assert values.reverse() is None
    values[1:3] = [9, 9]
    values[1:3] = values[0:2]
    prepared = change.prepare()
    assert prepared.value == {"values": [5, 5, 9, 2]}
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_tracks_held_handles_after_reindex_and_suppresses_detached_writes() -> None:
    tracker = track({"values": [{"value": "a"}, {"value": "b"}, {"value": "c"}]})
    change = tracker.begin_change()
    held = change.state["values"][1]
    change.state["values"].insert(0, {"value": "front"})
    held["value"] = "moved"
    assert change.state["values"][2]["value"] == "moved"
    del change.state["values"][2]
    held["value"] = "detached"
    prepared = change.prepare()
    assert prepared.value == {"values": [{"value": "front"}, {"value": "a"}, {"value": "c"}]}
    assert replay(tracker.value, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_replays_introduced_moved_then_edited_array_descendants() -> None:
    tracker = track({"values": []})
    base = clone(tracker.value)
    change = tracker.begin_change()
    change.state["values"].append({"id": 1, "nested": [1]})
    held = change.state["values"][0]
    change.state["values"].insert(0, {"id": 0, "nested": []})
    held["nested"].append(2)
    change.state["values"].reverse()
    held["nested"].append(3)
    prepared = change.prepare()
    assert prepared.value == {"values": [{"id": 1, "nested": [1, 2, 3]}, {"id": 0, "nested": []}]}
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_combines_movement_and_edits_with_permutation_plus_final_paths() -> None:
    tracker = track({"values": [{"rank": 3, "edited": 0}, {"rank": 1, "edited": 0}, {"rank": 2, "edited": 0}]})
    base = clone(tracker.value)
    change = tracker.begin_change()
    held = change.state["values"][0]
    change.state["values"].sort(key=lambda value: value["rank"])
    held["edited"] = 1
    prepared = change.prepare()
    assert prepared.ops[0] == ("m", ("values",), [1, 2, 0])
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_keeps_comparator_writes_and_structurally_reentrant_appends() -> None:
    tracker = track({"values": [{"rank": 2, "comparisons": 0}, {"rank": 1, "comparisons": 0}]})
    base = clone(tracker.value)
    change = tracker.begin_change()
    appended = False

    def compare(left: Any, right: Any) -> int:
        nonlocal appended
        left["comparisons"] += 1
        right["comparisons"] += 1
        if not appended:
            appended = True
            change.state["values"][0] = {"rank": 9, "comparisons": 0}
            change.state["values"].append({"rank": 3, "comparisons": 0})
        return left["rank"] - right["rank"]

    change.state["values"].sort(key=cmp_to_key(compare))
    prepared = change.prepare()
    assert [value["rank"] for value in prepared.value["values"]] == [1, 2, 3]
    assert [value["comparisons"] for value in prepared.value["values"]] == [1, 1, 0]
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_normalizes_duplicate_entries_created_by_structurally_reentrant_sort_callbacks() -> None:
    tracker = track(
        {
            "values": [
                {"rank": 2, "edited": 0, "nested": {"value": 0}},
                {"rank": 1, "edited": 0, "nested": {"value": 0}},
            ]
        }
    )
    base = clone(tracker.value)
    change = tracker.begin_change()
    inserted = False

    def compare(left: Any, right: Any) -> int:
        nonlocal inserted
        left["edited"] = left["rank"] * 10
        left["nested"]["value"] = left["rank"] * 100
        right["edited"] = right["rank"] * 10
        right["nested"]["value"] = right["rank"] * 100
        if not inserted:
            inserted = True
            change.state["values"].insert(0, {"rank": 4, "edited": 0, "nested": {"value": 0}})
        return left["rank"] - right["rank"]

    change.state["values"].sort(key=cmp_to_key(compare))
    prepared = change.prepare()
    values = prepared.value["values"]
    assert [value["rank"] for value in values] == [1, 2, 1]
    assert [value["edited"] for value in values] == [10, 20, 10]
    assert [value["nested"]["value"] for value in values] == [100, 200, 100]
    assert values[0] is not values[2]
    assert values[0]["nested"] is not values[2]["nested"]
    assert replay(base, prepared.ops) == prepared.value
    expect_alias_free(prepared.value)
    tracker.adopt(prepared)


def test_folds_dense_child_and_direct_index_edits_into_one_full_array_operation() -> None:
    rows_tracker = track({"rows": [{"value": value} for value in range(1_000)]})
    rows_base = clone(rows_tracker.value)
    rows_change = rows_tracker.begin_change()
    for row in rows_change.state["rows"]:
        row["value"] += 1
    rows_prepared = rows_change.prepare()
    assert len(rows_prepared.ops) == 1
    assert rows_prepared.ops[0][0] == "p"
    assert replay(rows_base, rows_prepared.ops) == rows_prepared.value
    rows_tracker.adopt(rows_prepared)

    values_tracker = track({"values": list(range(1_000))})
    values_base = clone(values_tracker.value)
    values_change = values_tracker.begin_change()
    for index in range(600):
        values_change.state["values"][index] = -index - 1
    values_prepared = values_change.prepare()
    assert len(values_prepared.ops) == 1
    assert values_prepared.ops[0][0] == "p"
    assert replay(values_base, values_prepared.ops) == values_prepared.value
    values_tracker.adopt(values_prepared)


def test_folds_only_a_deeply_nested_dense_array_region() -> None:
    tracker = track({"rows": [{"nested": {"value": value}} for value in range(2_000)]})
    base = clone(tracker.value)
    change = tracker.begin_change()
    change.state["rows"][0]["nested"]["value"] = -1
    for index in range(500, 1_000):
        change.state["rows"][index]["nested"]["value"] = -index
    prepared = change.prepare()
    splice = next(operation for operation in prepared.ops if operation[0] == "p")
    assert splice[:4] == ("p", ("rows",), 500, 500)
    assert len(splice[4]) == 500
    assert len(prepared.ops) == 2
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_does_not_cache_emission_paths_for_nested_reserved_key_folds_covered_by_an_outer_region() -> None:
    rows: list[Any] = [{"flag": 0} for _ in range(400)]
    rows[100]["special"] = {"__proto__": {"values": list(range(400))}}
    tracker = track({"rows": rows})
    base = clone(tracker.value)
    change = tracker.begin_change()
    for index in range(256):
        change.state["rows"][index]["flag"] = 1
    for index in range(256):
        change.state["rows"][100]["special"]["__proto__"]["values"][index] = -index - 1
    prepared = change.prepare()
    assert len(prepared.ops) == 1
    assert prepared.ops[0][:4] == ("p", ("rows",), 0, 256)
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_suppresses_nested_structural_and_leaf_operations_covered_by_an_outer_dense_region() -> None:
    tracker = track(
        {
            "rows": [
                {"flag": 0, "values": [{"value": value} for value in range(400)] if index == 100 else []}
                for index in range(400)
            ]
        }
    )
    base = clone(tracker.value)
    change = tracker.begin_change()
    for index in range(256):
        change.state["rows"][index]["flag"] = 1
    for index in range(256):
        change.state["rows"][100]["values"][index]["value"] = -index - 1
    change.state["rows"][100]["values"].append({"value": 999})
    prepared = change.prepare()
    assert len(prepared.ops) == 1
    assert prepared.ops[0][:4] == ("p", ("rows",), 0, 256)
    assert len(prepared.value["rows"][100]["values"]) == 401
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_emits_multiple_disjoint_dense_regions_and_preserves_operations_outside_their_boundaries() -> None:
    tracker = track({"values": [{"value": value} for value in range(1_400)]})
    base = clone(tracker.value)
    change = tracker.begin_change()
    for index in (97, 358, 897, 1_158):
        change.state["values"][index]["value"] = -index - 1
    for index in range(100, 356):
        change.state["values"][index]["value"] = -index - 1
    for index in range(900, 1_156):
        change.state["values"][index]["value"] = -index - 1
    prepared = change.prepare()
    splices = [operation[:4] for operation in prepared.ops if operation[0] == "p"]
    assert splices == [("p", ("values",), 100, 256), ("p", ("values",), 900, 256)]
    for index in (97, 358, 897, 1_158):
        assert ("s", ("values", index, "value"), -index - 1) in prepared.ops
    assert len(prepared.ops) == 6
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_normalizes_a_20000_operation_queue_transaction_to_two_operations() -> None:
    tracker = track({"values": [{"value": value} for value in range(20_000)]})
    base = clone(tracker.value)
    change = tracker.begin_change()
    values = change.state["values"]
    for index in range(10_000):
        values.pop(0)
        values.append({"value": 20_000 + index})
    prepared = change.prepare()
    assert len(prepared.ops) == 2
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_handles_adversarial_fragmentation_and_resolves_held_handles() -> None:
    size = 20_000
    tracker = track({"values": [{"value": value} for value in range(size)]})
    expected = clone(tracker.value)
    base = clone(tracker.value)
    change = tracker.begin_change()
    picks = [1, 1_001, 5_001, 10_001, 15_001, 19_999]
    held = [change.state["values"][index] for index in picks]
    for index in range(0, size, 2):
        change.state["values"][index : index + 1] = [{"value": -index - 1}]
        expected["values"][index : index + 1] = [{"value": -index - 1}]
    for value in held:
        value["value"] += 100_000
    for index in picks:
        expected["values"][index]["value"] += 100_000
    prepared = change.prepare()
    assert prepared.value == expected
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_normalizes_restored_overrides_and_cancelled_structural_edits_to_no_operations() -> None:
    tracker = track({"values": [1, 2, 3]})
    change = tracker.begin_change()
    change.state["values"][1] = 9
    change.state["values"][1] = 2
    change.state["values"].reverse()
    change.state["values"].reverse()
    change.state["values"].append(4)
    assert change.state["values"].pop() == 4
    prepared = change.prepare()
    assert prepared.value == {"values": [1, 2, 3]}
    assert prepared.ops == []
    tracker.adopt(prepared)


def test_handles_null_sparse_overrides_without_confusing_absence() -> None:
    tracker = track({"values": list(range(10_000))})

    def mutate(draft: Any) -> None:
        draft["values"][17] = None
        draft["values"][9_000] = None

    settle(tracker, mutate)
    assert tracker.value["values"][17] is None
    assert tracker.value["values"][9_000] is None


def test_supports_self_overlapping_slice_copies_by_value() -> None:
    tracker = track({"values": [{"n": 0}, {"n": 1}, {"n": 2}, {"n": 3}]})

    def mutate(draft: Any) -> None:
        values = draft["values"]
        values[1:4] = values[0:3]  # copyWithin(1, 0, 3)
        values[0:2] = [values[1], values[1]]  # fill(values[1], 0, 2)
        values[0]["n"] = 9

    settle(tracker, mutate)
    assert [value["n"] for value in tracker.value["values"]] == [9, 0, 1, 2]
    assert tracker.value["values"][0] is not tracker.value["values"][1]


@pytest.mark.parametrize("method", ["insert", "slice"])
def test_inserts_100000_items(method: str) -> None:
    tracker = track({"values": [-1]})
    base = clone(tracker.value)
    change = tracker.begin_change()
    items = list(range(100_000))
    if method == "insert":
        change.state["values"][0:0] = items
    else:
        change.state["values"][1:1] = items
    prepared = change.prepare()
    assert len(prepared.value["values"]) == 100_001
    assert prepared.value["values"][99_999 if method == "insert" else 100_000] == 99_999
    assert replay(base, prepared.ops) == prepared.value
    tracker.adopt(prepared)


# ─── randomized transactions ─────────────────────────────────────────────────


def mulberry32(seed: int) -> Any:
    state = seed & 0xFFFFFFFF

    def imul(left: int, right: int) -> int:
        return (left * right) & 0xFFFFFFFF

    def next_value() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        value = imul(state ^ (state >> 15), 1 | state)
        value = ((value + imul(value ^ (value >> 7), 61 | value)) & 0xFFFFFFFF) ^ value
        return ((value ^ (value >> 14)) & 0xFFFFFFFF) / 4_294_967_296

    return next_value


def random_mutate(document: Any, choice: int, value: int) -> None:
    def item() -> dict[str, int]:
        return {"id": value, "score": value % 7}

    values = document["values"]
    if choice == 0:
        document["text"] += f"-{value}"
    elif choice == 1:
        values.append(item())
    elif choice == 2:
        values.insert(0, item())
    elif choice == 3:
        if len(values) > 0:
            values.pop(0)
    elif choice == 4:
        if len(values) > 0:
            values.pop()
    elif choice == 5:
        index = 0 if len(values) == 0 else value % (len(values) + 1)
        remove = 0 if len(values) == 0 else value % 2
        values[index : index + remove] = [item()]
    elif choice == 6:
        values.reverse()
    elif choice == 7:
        values.sort(key=lambda entry: entry["id"])
    elif choice == 8:
        if len(values) > 0:
            values[value % len(values)]["score"] = value
    elif choice == 9:
        document["meta"]["revision"] += 1
        document["meta"]["label"] = f"r-{value}"
    else:
        document["meta"].pop("label", None)


def test_matches_policy_detached_replay_and_adopted_state_after_multi_operation_transactions() -> None:
    for seed in range(1, 41):
        rng = mulberry32(seed)
        initial = {
            "values": [{"id": index, "score": 0} for index in range(4)],
            "text": "start",
            "meta": {"revision": 0},
        }
        tracker = track(initial)
        expected = clone(initial)
        for transaction in range(25):
            base_root = tracker.value
            base = clone(base_root)
            change = tracker.begin_change()
            for operation in range(5):
                choice = int(rng() * 11)
                value = seed * 10_000 + transaction * 10 + operation
                random_mutate(expected, choice, value)
                random_mutate(change.state, choice, value)
            prepared = change.prepare()
            policy = clone(prepared.value)
            expect_plain(prepared.value)
            assert base_root == base, f"prepare changed base seed {seed} transaction {transaction}"
            assert policy == expected, f"policy seed {seed} transaction {transaction}"
            assert replay(base, prepared.ops) == policy, f"replay seed {seed} transaction {transaction}"
            tracker.adopt(prepared)
            assert base_root == base
            assert tracker.value is prepared.value
            assert tracker.value == policy
            expect_alias_free(tracker.value)


# ─── object emission scaling ─────────────────────────────────────────────────


def test_orders_many_reverse_depth_object_edits() -> None:
    initial: dict[str, Any] = {"value": 0}
    for _ in range(512):
        initial = {"value": 0, "next": initial}
    tracker = track(initial)
    change = tracker.begin_change()
    nodes: list[Draft[dict[str, Any]]] = []
    node: Any = change.state
    while node is not None:
        nodes.append(node)
        node = node.get("next")
    for index in range(len(nodes) - 1, -1, -1):
        nodes[index]["value"] = index + 1
    prepared = change.prepare()
    assert len(prepared.ops) == len(nodes)
    assert [len(operation[1]) for operation in prepared.ops] == sorted(len(operation[1]) for operation in prepared.ops)
    assert replay(initial, prepared.ops) == prepared.value
    tracker.adopt(prepared)
    expect_alias_free(tracker.value)


# ─── security and storage-style cloning ──────────────────────────────────────


def test_handles_reserved_own_keys() -> None:
    initial = json.loads('{"safe":{"__proto__":{"value":1}}}')
    tracker = track(initial)
    change = tracker.begin_change()
    change.state["safe"]["__proto__"]["value"] = 2
    prepared = change.prepare()
    assert prepared.ops == [("s", ("safe",), {"__proto__": {"value": 2}})]
    assert prepared.value["safe"]["__proto__"]["value"] == 2
    assert replay(tracker.value, prepared.ops) == prepared.value
    tracker.adopt(prepared)


def test_folds_reserved_key_writes_into_their_container() -> None:
    tracker = track({"safe": {"keep": 1}})
    change = tracker.begin_change()
    change.state["safe"]["constructor"] = 2
    prepared = change.prepare()
    assert prepared.ops == [("s", ("safe",), {"keep": 1, "constructor": 2})]
    tracker.adopt(prepared)
    root = track({"prototype": 1})
    root_change = root.begin_change()
    del root_change.state["prototype"]
    assert root_change.prepare().ops == [("r", {})]


def test_supports_a_storage_style_recursive_clone_of_prepared_value() -> None:
    def storage_clone(value: Any) -> Any:
        if type(value) is list:
            return [storage_clone(item) for item in value]
        if type(value) is dict:
            return {key: storage_clone(item) for key, item in value.items()}
        return value

    tracker = track({"rows": [{"value": 1}]})
    change = tracker.begin_change()
    change.state["rows"][0]["value"] = 2
    change.state["rows"].append({"value": 3})
    prepared = change.prepare()
    assert storage_clone(prepared.value) == {"rows": [{"value": 2}, {"value": 3}]}
    tracker.adopt(prepared)
