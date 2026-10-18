"""Port of chord ``test/state-draft.test.ts``.

``fill``/``copyWithin`` become slice assignments; ``delete values[0]`` (a hole upstream)
removes the element in Python, so that case checks the Python meaning instead.
"""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any

import pytest

from pi_chord.delta import apply_immutable, track
from pi_chord.json import UNDEFINED


def test_copies_only_changed_branches() -> None:
    tracker = track({"changed": {"count": 1, "sibling": {"value": "kept"}}, "untouched": {"value": 2}})
    base = tracker.value
    change = tracker.begin_change()
    first = change.state["changed"]
    assert change.state["changed"] is first
    change.state["changed"]["count"] = 3
    prepared = change.prepare()
    assert prepared.value is not base
    assert prepared.value["changed"] is not base["changed"]
    assert prepared.value["changed"]["sibling"] is base["changed"]["sibling"]
    assert prepared.value["untouched"] is base["untouched"]


def test_supports_deletion_and_native_list_mutators() -> None:
    tracker = track({"optional": "remove", "values": [3, 1, 2]})
    change = tracker.begin_change()
    del change.state["optional"]
    values = change.state["values"]
    values.append(4)
    assert values.pop() == 4
    values.insert(0, 0)
    assert values.pop(0) == 0
    assert values[1:2] == [1]
    values[1:2] = [5, 4]
    values.sort()
    values.reverse()
    values[1:3] = [9, 9]
    values[1:3] = values[0:2]
    prepared = change.prepare()
    assert prepared.value == {"values": [5, 5, 9, 2]}


def test_copies_inserted_draft_values_without_aliasing_their_original_reference() -> None:
    tracker = track({"values": [{"value": 1}, {"value": 2}]})
    change = tracker.begin_change()
    held = change.state["values"][0]
    change.state["values"].insert(0, held)
    held["value"] = 9
    prepared = change.prepare()
    assert prepared.value["values"] == [{"value": 1}, {"value": 9}, {"value": 2}]
    assert prepared.value["values"][0] is not prepared.value["values"][1]


def test_keeps_comparator_edits_when_sorting_object_drafts() -> None:
    tracker = track({"rows": [{"rank": 2, "comparisons": 0}, {"rank": 1, "comparisons": 0}]})
    change = tracker.begin_change()

    def compare(left: Any, right: Any) -> int:
        left["comparisons"] += 1
        right["comparisons"] += 1
        return left["rank"] - right["rank"]

    change.state["rows"].sort(key=cmp_to_key(compare))
    prepared = change.prepare()
    assert [row["rank"] for row in prepared.value["rows"]] == [1, 2]
    assert [row["comparisons"] for row in prepared.value["rows"]] == [1, 1]


def test_drops_writes_through_detached_child_references() -> None:
    tracker = track({"child": {"value": "removed"}, "items": [{"value": "removed"}, {"value": "kept"}]})
    change = tracker.begin_change()
    child = change.state["child"]
    shifted = change.state["items"].pop(0)
    del change.state["child"]
    child["value"] = "detached child"
    shifted["value"] = "detached item"
    assert change.prepare().value == {"items": [{"value": "kept"}]}


@pytest.mark.parametrize("method", ["insert", "slice"])
def test_inserts_100000_items(method: str) -> None:
    tracker = track({"values": [-1]})
    change = tracker.begin_change()
    items = list(range(100_000))
    if method == "insert":
        change.state["values"][0:0] = items
    else:
        change.state["values"][1:1] = items
    prepared = change.prepare()
    assert len(prepared.value["values"]) == 100_001
    assert prepared.value["values"][0 if method == "insert" else 1] == 0
    assert prepared.value["values"][99_999 if method == "insert" else 100_000] == 99_999
    assert apply_immutable(prepared.base, prepared.ops) == prepared.value


def test_rejects_non_strict_json_placements_without_mutating_the_draft_or_base() -> None:
    tracker = track({"number": 0, "payload": None, "values": [1, 2]})
    change = tracker.begin_change()
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["values"][0] = UNDEFINED
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["values"].append({"nested": UNDEFINED})
    with pytest.raises(TypeError, match="strict JSON"):
        change.state["number"] = float("nan")
    with pytest.raises(TypeError, match="plain objects"):
        change.state["payload"] = object()
    assert change.state == tracker.value
    change.abort()
    assert tracker.value == {"number": 0, "payload": None, "values": [1, 2]}


def test_deleting_a_list_element_removes_it_without_mutating_the_committed_base() -> None:
    tracker = track({"values": [1, 2]})
    change = tracker.begin_change()
    del change.state["values"][0]
    assert change.state["values"] == [2]
    change.abort()
    assert tracker.value["values"] == [1, 2]
