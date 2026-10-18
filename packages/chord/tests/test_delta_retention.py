"""Port of chord ``test/delta-tracker/retention.test.ts`` and its worker scenarios.

Upstream runs each scenario in a child process with ``--expose-gc`` and probes liveness
with ``WeakRef``. ``dict`` and ``list`` cannot be weakly referenced, so these checks find
a target among ``gc.get_objects()`` by id, type and length after a full collection. The
payloads are lists (always tracked by the collector); heap budgets use ``tracemalloc``.
"""

from __future__ import annotations

import gc
import tracemalloc
from typing import Any

import pytest

from pi_chord.delta import Tracker, track


class Probe:
    """Identifies one container without keeping it alive."""

    __slots__ = ("identity", "kind", "length")

    def __init__(self, target: Any) -> None:
        self.identity = id(target)
        self.kind = type(target)
        self.length = len(target) if type(target) is list else None

    def alive(self) -> bool:
        gc.collect()
        for candidate in gc.get_objects():
            if id(candidate) == self.identity and type(candidate) is self.kind:
                if self.length is None or len(candidate) == self.length:
                    return True
        return False


def rows(count: int) -> list[Any]:
    return [[value] for value in range(count)]


def test_releases_an_aborted_payload() -> None:
    tracker: Tracker[Any] = track({"payload": None})

    def scenario() -> tuple[Probe, Probe]:
        change = tracker.begin_change()
        change.state["payload"] = {"rows": rows(50_000)}
        draft = change.state["payload"]["rows"]
        probes = Probe(draft), Probe(draft._base)
        change.abort()
        return probes

    for probe in scenario():
        assert not probe.alive()
    assert tracker.value == {"payload": None}


def test_releases_an_unadopted_prepared_candidate() -> None:
    tracker: Tracker[Any] = track({"payload": None})

    def scenario() -> Probe:
        change = tracker.begin_change()
        change.state["payload"] = {"rows": rows(50_000)}
        prepared = change.prepare()
        return Probe(prepared.value["payload"]["rows"])

    assert not scenario().alive()
    assert tracker.value == {"payload": None}


def test_releases_draft_handles_after_abort() -> None:
    tracker: Tracker[Any] = track({"payload": None})

    def scenario() -> Probe:
        change = tracker.begin_change()
        change.state["payload"] = {"rows": [[1]]}
        probe = Probe(change.state["payload"]["rows"][0])
        change.abort()
        return probe

    assert not scenario().alive()


def test_settled_changes_and_preparations_do_not_retain_their_tracker() -> None:
    retained: list[Any] = []

    def scenario() -> tuple[Probe, Probe]:
        tracker: Tracker[Any] = track({"payload": None})
        change = tracker.begin_change()
        change.state["payload"] = {"rows": [[1]]}
        prepared = change.prepare()
        tracker.adopt(prepared)
        retained.extend([change, prepared])
        future = tracker.prepare_replace({"payload": {"rows": rows(50_000)}})
        tracker.adopt(future)
        return Probe(tracker), Probe(future.value["payload"]["rows"])

    for probe in scenario():
        assert not probe.alive()
    assert len(retained) == 2


def test_a_retained_prepared_retains_its_revisions_until_released() -> None:
    tracker = track({"payload": {"rows": rows(50_000)}})
    retained: list[Any] = []

    def scenario() -> tuple[Probe, Probe]:
        change = tracker.begin_change()
        change.state["payload"]["rows"][0][0] = -1
        prepared = change.prepare()
        probes = Probe(prepared.base["payload"]["rows"]), Probe(prepared.value["payload"]["rows"])
        tracker.adopt(prepared)
        retained.append(prepared)
        return probes

    base, value = scenario()
    tracker.adopt(tracker.prepare_replace({"payload": {"rows": [[1]]}}))
    assert base.alive()
    assert value.alive()
    retained.clear()
    assert not base.alive()
    assert not value.alive()


def test_a_retained_settled_change_does_not_retain_revisions() -> None:
    tracker = track({"payload": {"rows": rows(50_000)}})
    retained: list[Any] = []

    def scenario() -> tuple[Probe, Probe]:
        change = tracker.begin_change()
        change.state["payload"]["rows"][0][0] = -1
        prepared = change.prepare()
        probes = Probe(prepared.base["payload"]["rows"]), Probe(prepared.value["payload"]["rows"])
        tracker.adopt(prepared)
        retained.append(change)
        return probes

    probes = scenario()
    tracker.adopt(tracker.prepare_replace({"payload": {"rows": [[1]]}}))
    for probe in probes:
        assert not probe.alive()
    assert len(retained) == 1


@pytest.mark.parametrize("visited", [0, 5_000])
def test_a_retained_settled_draft_does_not_retain_obsolete_revisions(visited: int) -> None:
    tracker = track({"child": [0], "payload": {"rows": rows(100_000)}})
    obsolete = Probe(tracker.value["payload"]["rows"])
    retained: list[Any] = []

    def commit() -> None:
        change = tracker.begin_change()
        held = change.state["child"]
        for index in range(visited):
            assert change.state["payload"]["rows"][index][0] == index
        held[0] = 1
        tracker.adopt(change.prepare())
        retained.append(held)

    commit()
    tracker.adopt(tracker.prepare_replace({"child": [2], "payload": {"rows": [[1]]}}))
    assert not obsolete.alive()
    with pytest.raises(TypeError, match="settled"):
        retained[0][0]


@pytest.mark.parametrize("kind", ["write", "writes", "insert", "override"])
def test_retained_drafts_do_not_retain_large_placements(kind: str) -> None:
    retained: list[Any] = []

    def run() -> Probe:
        tracker: Tracker[Any] = track(
            {
                "child": [0],
                "visited": rows(5_000),
                "first": None,
                "second": None,
                "rows": [None] if kind == "override" else [],
            }
        )
        change = tracker.begin_change()
        for value in change.state["visited"]:
            assert value[0] >= 0
        if kind in ("write", "writes"):
            held = change.state["child"]
            change.state["first"] = [1] * 50_000
            if kind == "writes":
                change.state["second"] = [2] * 50_000
        else:
            held = change.state["rows"]
            if kind == "insert":
                change.state["rows"].append([1] * 50_000)
            else:
                change.state["rows"][0] = [1] * 50_000
        prepared = change.prepare()
        payload = prepared.value["first"] if kind in ("write", "writes") else prepared.value["rows"][0]
        probe = Probe(payload)
        tracker.adopt(prepared)
        tracker.adopt(tracker.prepare_replace({"child": [1], "visited": [], "first": None, "second": None, "rows": []}))
        retained.append(held)
        return probe

    assert not run().alive()
    for held in retained:
        with pytest.raises(TypeError, match="settled"):
            len(held)


def test_a_stale_unprepared_change_does_not_retain_obsolete_revisions() -> None:
    tracker = track({"payload": {"rows": rows(50_000)}})
    stale = tracker.begin_change()
    stale.state["payload"]["rows"][0][0] = -1
    obsolete = Probe(tracker.value["payload"]["rows"])
    tracker.adopt(tracker.prepare_replace({"payload": {"rows": [[1]]}}))
    with pytest.raises(TypeError, match="settled"):
        stale.prepare()
    assert not obsolete.alive()


def test_same_job_cleanup_keeps_the_heap_bounded() -> None:
    tracker = track({"rows": [[value] for value in range(100_000)]})
    gc.collect()
    tracemalloc.start()
    try:
        ready = tracemalloc.get_traced_memory()[0]
        for value in range(100):
            change = tracker.begin_change()
            draft = change.state
            for index in range(4_094):
                assert draft["rows"][index][0] == index
            draft["rows"][-1][0] = -value - 1
            tracker.adopt(change.prepare())
        gc.collect()
        retained = (tracemalloc.get_traced_memory()[0] - ready) / (1024 * 1024)
    finally:
        tracemalloc.stop()
    assert retained < 16, f"same-job cleanup retained {retained:.2f} MiB"
    assert tracker.value["rows"][-1][0] == -100


def test_same_job_folded_operations_keep_the_heap_bounded() -> None:
    tracker = track({f"field{index}": 0 for index in range(5_000)})
    gc.collect()
    tracemalloc.start()
    try:
        ready = tracemalloc.get_traced_memory()[0]
        for value in range(1, 101):
            change = tracker.begin_change()
            for index in range(5_000):
                change.state[f"field{index}"] = value
            prepared = change.prepare()
            assert len(prepared.ops) == 1
            assert prepared.ops[0][0] == "r"
            tracker.adopt(prepared)
        del change, prepared
        gc.collect()
        retained = (tracemalloc.get_traced_memory()[0] - ready) / (1024 * 1024)
    finally:
        tracemalloc.stop()
    assert retained < 8, f"same-job folded operations retained {retained:.2f} MiB"
    assert tracker.value["field0"] == 100


def test_lifecycle_churn_does_not_accumulate_contexts() -> None:
    tracker = track({"value": 0})
    for index in range(100_000):
        change = tracker.begin_change()
        change.state["value"] = index
        change.abort()
    assert len(tracker._contexts) == 0
    tracker.adopt(tracker.prepare_replace({"value": 1}))
    assert tracker.value == {"value": 1}


def test_dropped_unsettled_changes_are_not_retained_by_the_tracker() -> None:
    tracker = track({"value": 0})
    for index in range(1_000):
        change = tracker.begin_change()
        change.state["value"] = index
    del change
    gc.collect()
    assert all(reference() is None for reference in tracker._contexts.values())
    tracker.adopt(tracker.prepare_replace({"value": -1}))
    assert len(tracker._contexts) == 0


def test_obsolete_revisions_are_released() -> None:
    tracker = track({"payload": {"rows": rows(50_000)}})
    old_root = Probe(tracker.value["payload"]["rows"])
    for value in range(100):
        tracker.adopt(tracker.prepare_replace({"payload": {"rows": [[value]]}}))
    assert not old_root.alive()
    assert tracker.value == {"payload": {"rows": [[99]]}}
