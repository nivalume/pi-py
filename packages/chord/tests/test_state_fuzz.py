"""Port of chord ``test/state-fuzz.test.ts``."""

from __future__ import annotations

import json
from typing import Any

from pi_chord.delta import apply_immutable, track


def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def expect_alias_free(value: Any) -> None:
    seen: dict[int, str] = {}

    def visit(current: Any, path: str) -> None:
        if type(current) not in (dict, list):
            return
        previous = seen.get(id(current))
        if previous is not None:
            raise AssertionError(f"container at {path} aliases {previous}")
        seen[id(current)] = path
        items = enumerate(current) if type(current) is list else current.items()
        for key, item in items:
            visit(item, f"{path}.{key}")

    visit(value, "$root")


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


def mutate(document: Any, choice: int, value: int) -> None:
    def item() -> dict[str, Any]:
        return {"id": value, "text": f"item-{value}", "score": value % 7}

    items = document["items"]
    if choice == 0:
        document["text"] += f"-{value}"
    elif choice == 1:
        document["text"] = f"{document['text'][min(2, len(document['text'])) :]}{value}"
    elif choice == 2:
        items.append(item())
    elif choice == 3:
        items.insert(0, item())
    elif choice == 4:
        if len(items) > 0:
            items.pop(0)
    elif choice == 5:
        if len(items) > 0:
            items.pop()
    elif choice == 6:
        index = 0 if len(items) == 0 else value % (len(items) + 1)
        remove = 0 if len(items) == 0 else value % 2
        items[index : index + remove] = [item()]
    elif choice == 7:
        items.reverse()
    elif choice == 8:
        items.sort(key=lambda entry: entry["id"])
    elif choice == 9:
        if len(items) > 0:
            items[value % len(items)]["score"] = value
    elif choice == 10:
        document["meta"]["revision"] += 1
        document["meta"]["label"] = f"revision-{value}"
    elif choice == 11:
        document["meta"].pop("label", None)
    elif choice == 12:
        if len(items) > 1:
            items[1] = clone(items[0])
    else:
        for index in range(min(2, len(items))):
            items[index] = item()


def test_converges_across_randomized_prepared_revisions() -> None:
    for seed in range(1, 101):
        rng = mulberry32(seed)
        initial = {
            "items": [{"id": index, "text": f"item-{index}", "score": 0} for index in range(4)],
            "text": "start",
            "meta": {"revision": 0},
        }
        tracker = track(initial)
        expected = clone(initial)
        replica = clone(tracker.value)
        for step in range(100):
            choice = int(rng() * 14)
            value = seed * 1_000 + step
            base_root = tracker.value
            base = clone(base_root)
            mutate(expected, choice, value)
            change = tracker.begin_change()
            mutate(change.state, choice, value)
            prepared = change.prepare()
            operations = clone(prepared.ops)
            assert base_root == base, f"prepare base seed {seed} step {step} choice {choice}"
            assert prepared.base is base_root
            replica = apply_immutable(replica, operations)
            assert replica == prepared.value, f"replay seed {seed} step {step} choice {choice}"
            tracker.adopt(prepared)
            assert base_root == base
            assert tracker.value is prepared.value
            assert tracker.value == expected, f"state seed {seed} step {step} choice {choice}"
            assert replica == expected
            expect_alias_free(tracker.value)
