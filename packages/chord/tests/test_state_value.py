"""Port of chord ``test/state-value.test.ts`` (freezing checks are JavaScript-only)."""

from __future__ import annotations

import copy

from pi_chord.delta import track


def test_takes_ownership_of_an_alias_free_mutable_json_root() -> None:
    input_value = {"left": {"value": 1}, "right": {"value": 1}}
    tracker = track(input_value)
    assert tracker.value is input_value
    assert tracker.value["left"] is not tracker.value["right"]


def test_commits_transaction_copies_while_sharing_unchanged_branches() -> None:
    tracker = track({"changed": {"value": 1}, "retained": {"value": 2}})
    base = tracker.value
    base_snapshot = copy.deepcopy(base)
    change = tracker.begin_change()
    change.state["changed"]["value"] = 3
    prepared = change.prepare()
    assert prepared.value["changed"] is not base["changed"]
    assert prepared.value["retained"] is base["retained"]
    tracker.adopt(prepared)
    assert base == base_snapshot
    assert tracker.value is prepared.value


def test_makes_repeated_placements_independent() -> None:
    tracker = track({"values": []})
    change = tracker.begin_change()
    shared = {"value": 1}
    change.state["values"].extend([shared, shared])
    prepared = change.prepare()
    assert prepared.value["values"][0] is not prepared.value["values"][1]
