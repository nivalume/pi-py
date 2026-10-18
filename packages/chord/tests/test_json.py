"""Port of chord ``test/json.test.ts``."""

from __future__ import annotations

import math

import pytest

from pi_chord import UNDEFINED, copy_json, is_json_value


def test_checks_strict_json_without_normalizing_it() -> None:
    assert is_json_value({"nested": [1, True, None, 1.5, "x"]})
    assert not is_json_value({"omitted": UNDEFINED})
    assert not is_json_value(b"\x01")
    assert not is_json_value(math.inf)
    assert not is_json_value(math.nan)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    assert not is_json_value(cyclic)


def test_copies_strict_json_without_retaining_aliases() -> None:
    shared = {"value": 1}
    data = {"left": shared, "right": shared}
    copied = copy_json(data)
    assert copied == data
    assert copied is not data
    assert isinstance(copied, dict)
    assert copied["left"] is not shared
    assert copied["right"] is not shared
    assert copied["left"] is not copied["right"]


def test_optionally_omits_undefined_properties_without_normalizing_lists() -> None:
    data = {"kept": 1, "omitted": UNDEFINED, "nested": {"omitted": UNDEFINED, "kept": True}}
    with pytest.raises(TypeError, match="strict JSON"):
        copy_json(data)
    assert copy_json(data, omit_undefined_properties=True) == {"kept": 1, "nested": {"kept": True}}
    with pytest.raises(TypeError, match="strict JSON"):
        copy_json([UNDEFINED], omit_undefined_properties=True)


def test_keeps_none_as_json_null() -> None:
    assert copy_json({"value": None}, omit_undefined_properties=True) == {"value": None}


def test_rejects_cycles_and_non_strict_containers() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    class DictSubclass(dict):  # type: ignore[type-arg]
        pass

    class ListSubclass(list):  # type: ignore[type-arg]
        pass

    for invalid in [cyclic, (1, 2), {1: "int key"}, DictSubclass(a=1), ListSubclass([1]), {1, 2}, object()]:
        assert not is_json_value(invalid)
        with pytest.raises(TypeError, match="strict JSON"):
            copy_json(invalid)


def test_shared_but_acyclic_structures_are_allowed() -> None:
    leaf = [1, 2]
    data = [leaf, leaf, {"again": leaf}]
    assert is_json_value(data)
    assert copy_json(data) == [[1, 2], [1, 2], {"again": [1, 2]}]
