"""Port of chord ``test/delta-clone.test.ts`` (null prototypes and accessors are JavaScript-only)."""

from __future__ import annotations

import copy
from typing import Any

from pi_chord.delta import apply, track


def test_takes_immutable_ownership_of_the_imported_revision_in_o1() -> None:
    values: list[Any] = [0, False, None, "text", {"n": 1}]
    point = {"x": 3, "y": 7, "pressure": 0.1}
    input_value = {"point": point, "rows": [{"values": values}]}
    tracker = track(input_value)
    assert tracker.value is input_value
    assert tracker.value["point"] is point
    assert tracker.value["rows"][0]["values"][4] is values[4]


def test_keeps_an_alias_free_owned_root() -> None:
    tracker = track({"left": {"nested": [{"n": 1}]}, "right": {"nested": [{"n": 1}]}})
    assert tracker.value["left"] is not tracker.value["right"]
    assert tracker.value["left"]["nested"][0] is not tracker.value["right"]["nested"][0]


def test_does_not_traverse_a_trusted_root_while_taking_ownership() -> None:
    # A walk would reject the non-JSON value; ownership is taken without reading the root.
    marker = object()
    input_value = {"untrusted": marker}
    tracker = track(input_value)
    assert tracker.value is input_value
    assert tracker.value["untrusted"] is marker


def test_copies_assigned_and_inserted_values_immediately() -> None:
    tracker = track({"rows": []})
    assigned = {"nested": {"value": 1}}
    change = tracker.begin_change()
    change.state["rows"].append(assigned)
    assigned["nested"]["value"] = 9
    assert change.state["rows"][0]["nested"]["value"] == 1
    prepared = change.prepare()
    assert prepared.value["rows"][0]["nested"]["value"] == 1
    replica = apply(copy.deepcopy(prepared.base), prepared.ops)
    assert replica == prepared.value


def test_copy_json_accepts_live_drafts_and_rejects_settled_ones() -> None:
    from pi_chord.delta import track
    from pi_chord.json import copy_json, is_json_value

    tracker = track({"items": [{"a": 1}], "name": "x"})
    change = tracker.begin_change()
    draft = change.state
    draft["items"].append({"b": 2})
    copied = copy_json(draft)
    assert copied == {"items": [{"a": 1}, {"b": 2}], "name": "x"}
    assert type(copied) is dict and type(copied["items"]) is list
    assert is_json_value(draft)
    change.abort()
    assert not is_json_value(draft)
