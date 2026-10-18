"""Apply self-produced trusted operations immutably (port of chord ``delta/apply-immutable-trusted.ts``).

The tracker materializes a prepared revision with this: its own ops need no validation,
and each touched container is copied once.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar, cast

from .operations import _drop_utf16_prefix

__all__ = ["apply_immutable_trusted"]

T = TypeVar("T")


def apply_immutable_trusted(target: T, operations: Iterable[Any]) -> T:
    """Apply trusted ops to ``target`` without mutating it, copying each touched container once."""
    root: Any = target
    owned: dict[int, Any] = {}
    for operation in operations:
        verb = operation[0]
        if verb == "r":
            root = operation[1]
            continue
        path = operation[1]
        if verb == "p" and operation[3] == 0:
            root, appended = _append(root, path, operation[2], operation[4], owned)
            if appended:
                continue
        if verb == "p" or verb == "m":
            root, node = _copy_path(root, path, len(path), owned)
            if type(node) is not list:
                raise TypeError("Trusted array operation target is not an array")
            if verb == "p":
                start = operation[2]
                node[start : start + operation[3]] = operation[4]
            else:
                permutation = operation[2]
                if len(node) != len(permutation):
                    raise TypeError("Trusted permutation length mismatch")
                previous = node[:]
                node[:] = [previous[index] for index in permutation]
            continue
        root, parent = _copy_path(root, path, len(path) - 1, owned)
        key = path[-1]
        if verb == "s":
            if type(parent) is list and key == len(parent):
                parent.append(operation[2])
            else:
                parent[key] = operation[2]
        elif verb == "d":
            if type(parent) is list:
                if type(key) is not int:
                    raise TypeError("Trusted array deletion key is not numeric")
                del parent[key]
            else:
                parent.pop(key, None)
        else:
            current = parent[key]
            if type(current) is not str:
                kind = "append" if verb == "a" else "truncate"
                raise TypeError(f"Trusted {kind} target is not a string")
            parent[key] = current + operation[2] if verb == "a" else _drop_utf16_prefix(current, operation[2])
    return cast(T, root)


def _append(root: Any, path: Any, index: int, items: Any, owned: dict[int, Any]) -> tuple[Any, bool]:
    """Append to a list not yet copied with one ``list + items`` copy; ``(root, False)`` if not applicable."""
    if len(path) == 0:
        if type(root) is list and id(root) not in owned and index == len(root):
            appended = root + items
            owned[id(appended)] = appended
            return appended, True
        return root, False
    root, parent = _copy_path(root, path, len(path) - 1, owned)
    key = path[-1]
    node = parent[key]
    if type(node) is not list or id(node) in owned or index != len(node):
        return root, False
    appended = node + items
    parent[key] = appended
    owned[id(appended)] = appended
    return root, True


def _copy_path(root: Any, path: Any, length: int, owned: dict[int, Any]) -> tuple[Any, Any]:
    kind = type(root)
    if kind is not dict and kind is not list:
        raise TypeError("Trusted operation root is not a container")
    copied_root = root
    if id(root) not in owned:
        copied_root = root.copy()
        owned[id(copied_root)] = copied_root
    destination = copied_root
    for index in range(length):
        segment = path[index]
        child = destination[segment]
        child_kind = type(child)
        if child_kind is not dict and child_kind is not list:
            raise TypeError("Trusted operation path is not a container")
        if id(child) in owned:
            destination = child
            continue
        copy = child.copy()
        destination[segment] = copy
        owned[id(copy)] = copy
        destination = copy
    return copied_root, destination
