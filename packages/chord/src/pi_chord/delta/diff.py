"""Structural diff of two immutable JSON revisions (port of chord ``delta/diff.ts``).

Unchanged subtrees are recognized by identity first, so revisions that share structure
diff in time proportional to what changed. Array edits become splices, a permutation for
a pure reorder, or narrow leaf edits for elements matched by identity or shared subtrees.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .operations import RESERVED_SEGMENTS, Op, _utf16_length, overlap

__all__ = ["diff_revisions"]

_DEFAULT_OVERLAP_SCAN = 65_536
_MAX_DELTA_OPERATIONS = 4_096
_MAX_IDENTITY_CANDIDATES = 200_000
_MAX_SEMANTIC_CELLS = 65_536


class _Batch:
    __slots__ = ("operations", "overflowed")

    def __init__(self) -> None:
        self.operations: list[Any] = []
        self.overflowed = False

    def emit(self, operation: Any) -> None:
        if self.overflowed:
            return
        if len(self.operations) >= _MAX_DELTA_OPERATIONS:
            self.overflowed = True
            return
        self.operations.append(operation)


def _is_container(value: Any) -> bool:
    kind = type(value)
    return kind is dict or kind is list


def _emit_set(path: tuple[Any, ...], value: Any, batch: _Batch) -> None:
    if len(path) == 0:
        batch.emit(("r", value))
    else:
        batch.emit(("s", path, value))


def equal_json(left: Any, right: Any) -> bool:
    """Deep strict-JSON equality: key order is ignored, ``1``, ``1.0`` and ``True`` differ."""
    if left is right:
        return True
    kind = type(left)
    if kind is not type(right):
        return False
    if kind is dict or kind is list:
        # Native comparison rejects most differences quickly; it conflates 1, 1.0 and True,
        # so an equal result is confirmed strictly.
        return left == right and _strict_equal(left, right)
    return left == right


def _strict_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    kind = type(left)
    if kind is not type(right):
        return False
    if kind is list:
        if len(left) != len(right):
            return False
        return all(_strict_equal(a, b) for a, b in zip(left, right, strict=True))
    if kind is dict:
        if len(left) != len(right):
            return False
        return all(key in right and _strict_equal(value, right[key]) for key, value in left.items())
    return left == right


def _identity_key(value: Any) -> Any:
    """A JavaScript ``Map`` key: containers by identity, primitives by type and value."""
    kind = type(value)
    if kind is dict or kind is list:
        return id(value)
    return (kind, value)


def _identical(left: Any, right: Any) -> bool:
    """JavaScript ``===`` on JSON values."""
    if left is right:
        return True
    kind = type(left)
    return kind is type(right) and kind is not dict and kind is not list and left == right


def _same_value(left: Any, right: Any) -> bool:
    return left is right or equal_json(left, right)


def _permutation(before: Sequence[Any], after: Sequence[Any]) -> list[int] | None:
    if len(before) != len(after):
        return None
    positions: dict[Any, list[int]] = {}
    for index, value in enumerate(before):
        positions.setdefault(_identity_key(value), []).append(index)
    used: dict[Any, int] = {}
    result: list[int] = []
    for value in after:
        key = _identity_key(value)
        indices = positions.get(key)
        at = used.get(key, 0)
        if indices is None or at == len(indices):
            return None
        result.append(indices[at])
        used[key] = at + 1
    return result


def _emit_string(before: str, after: str, path: tuple[Any, ...], batch: _Batch) -> None:
    if before == after:
        return
    if len(after) > len(before) and after.startswith(before):
        batch.emit(("a", path, after[len(before) :]))
        return
    shared = overlap(before, after, _DEFAULT_OVERLAP_SCAN)
    if shared == 0:
        batch.emit(("s", path, after))
        return
    batch.emit(("t", path, _utf16_length(before[: len(before) - shared])))
    if len(after) > shared:
        batch.emit(("a", path, after[shared:]))


def _lcs_matches(
    before: Sequence[Any],
    after: Sequence[Any],
    equal: Callable[[Any, Any], bool],
    max_cells: int,
) -> list[tuple[int, int]] | None:
    if len(before) == 0 or len(after) == 0:
        return []
    if len(before) * len(after) > max_cells:
        return None
    width = len(after) + 1
    lengths = [0] * ((len(before) + 1) * width)
    for left in range(len(before) - 1, -1, -1):
        for right in range(len(after) - 1, -1, -1):
            at = left * width + right
            if equal(before[left], after[right]):
                lengths[at] = lengths[(left + 1) * width + right + 1] + 1
            else:
                lengths[at] = max(lengths[(left + 1) * width + right], lengths[left * width + right + 1])
    matches: list[tuple[int, int]] = []
    left = 0
    right = 0
    while left < len(before) and right < len(after):
        if (
            equal(before[left], after[right])
            and lengths[left * width + right] == lengths[(left + 1) * width + right + 1] + 1
        ):
            matches.append((left, right))
            left += 1
            right += 1
        elif lengths[(left + 1) * width + right] >= lengths[left * width + right + 1]:
            left += 1
        else:
            right += 1
    return matches


def _semantically_aligned(left: Any, right: Any) -> bool:
    if _same_value(left, right):
        return True
    kind = type(left)
    if kind is not type(right):
        return False
    if kind is list:
        if len(left) != len(right):
            return False
        return any(_is_container(value) and value is right[index] for index, value in enumerate(left))
    if kind is dict:
        for key, value in left.items():
            if _is_container(value) and key in right and value is right[key]:
                return True
    return False


def _lower_bound(values: Sequence[int], value: int) -> int:
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) >> 1
        if values[middle] < value:
            low = middle + 1
        else:
            high = middle
    return low


def _identity_subsequence(
    before: Sequence[Any],
    before_start: int,
    before_end: int,
    after: Sequence[Any],
    after_start: int,
    after_end: int,
) -> list[tuple[int, int]] | None:
    before_count = before_end - before_start
    after_count = after_end - after_start
    matches: list[tuple[int, int]] = []
    if after_count < before_count:
        before_index = before_start
        for after_index in range(after_start, after_end):
            while before_index < before_end and not _identical(before[before_index], after[after_index]):
                before_index += 1
            if before_index == before_end:
                return None
            matches.append((before_index, after_index))
            before_index += 1
        return matches
    if before_count < after_count:
        after_index = after_start
        for before_index in range(before_start, before_end):
            while after_index < after_end and not _identical(before[before_index], after[after_index]):
                after_index += 1
            if after_index == after_end:
                return None
            matches.append((before_index, after_index))
            after_index += 1
        return matches
    return None


def _greedy_identity_anchors(
    positions: dict[Any, list[int]],
    after: Sequence[Any],
    after_start: int,
    after_end: int,
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    previous = -1
    for after_index in range(after_start, after_end):
        candidates = positions.get(_identity_key(after[after_index]))
        if candidates is None:
            continue
        at = _lower_bound(candidates, previous + 1)
        if at >= len(candidates):
            continue
        before_index = candidates[at]
        matches.append((before_index, after_index))
        previous = before_index
    return matches


def _identity_anchors(
    before: Sequence[Any],
    before_start: int,
    before_end: int,
    after: Sequence[Any],
    after_start: int,
    after_end: int,
) -> list[tuple[int, int]]:
    positions: dict[Any, list[int]] = {}
    for index in range(before_start, before_end):
        positions.setdefault(_identity_key(before[index]), []).append(index)
    candidate_count = 0
    for index in range(after_start, after_end):
        found = positions.get(_identity_key(after[index]))
        if found is not None:
            candidate_count += len(found)
        if candidate_count > _MAX_IDENTITY_CANDIDATES:
            return _greedy_identity_anchors(positions, after, after_start, after_end)
    if candidate_count == 0:
        return []

    # Longest increasing subsequence of before-indices over after order (patience sorting).
    candidate_before: list[int] = []
    candidate_after: list[int] = []
    candidate_previous: list[int] = []
    tails: list[int] = []
    tail_values: list[int] = []
    for after_index in range(after_start, after_end):
        before_positions = positions.get(_identity_key(after[after_index]))
        if before_positions is None:
            continue
        for before_index in reversed(before_positions):
            at = _lower_bound(tail_values, before_index)
            candidate_index = len(candidate_before)
            candidate_before.append(before_index)
            candidate_after.append(after_index)
            candidate_previous.append(-1 if at == 0 else tails[at - 1])
            if at == len(tails):
                tails.append(candidate_index)
                tail_values.append(before_index)
            else:
                tails[at] = candidate_index
                tail_values[at] = before_index
    matches: list[tuple[int, int]] = []
    candidate_index = tails[-1] if tails else -1
    while candidate_index >= 0:
        matches.append((candidate_before[candidate_index], candidate_after[candidate_index]))
        candidate_index = candidate_previous[candidate_index]
    matches.reverse()
    return matches


def _process_array_matches(
    before: list[Any],
    after: list[Any],
    path: tuple[Any, ...],
    batch: _Batch,
    before_start: int,
    before_end: int,
    after_start: int,
    after_end: int,
    output_start: int,
    matches: Sequence[tuple[int, int]],
) -> None:
    before_at = before_start
    after_at = after_start
    output_at = output_start
    for before_match, after_match in matches:
        if batch.overflowed:
            return
        _diff_array_region(before, after, path, batch, before_at, before_match, after_at, after_match, output_at)
        output_at += after_match - after_at
        if not _same_value(before[before_match], after[after_match]):
            _diff_value(before[before_match], after[after_match], (*path, output_at), batch)
        output_at += 1
        before_at = before_match + 1
        after_at = after_match + 1
    if not batch.overflowed:
        _diff_array_region(before, after, path, batch, before_at, before_end, after_at, after_end, output_at)


def _diff_array_region(
    before: list[Any],
    after: list[Any],
    path: tuple[Any, ...],
    batch: _Batch,
    before_start: int,
    before_end: int,
    after_start: int,
    after_end: int,
    output_start: int,
) -> None:
    if batch.overflowed:
        return
    while (
        before_start < before_end and after_start < after_end and _same_value(before[before_start], after[after_start])
    ):
        before_start += 1
        after_start += 1
        output_start += 1
    while (
        before_start < before_end
        and after_start < after_end
        and _same_value(before[before_end - 1], after[after_end - 1])
    ):
        before_end -= 1
        after_end -= 1
    before_count = before_end - before_start
    after_count = after_end - after_start
    if before_count == 0 and after_count == 0:
        return
    if before_count == 0 or after_count == 0:
        batch.emit(("p", path, output_start, before_count, after[after_start:after_end]))
        return

    args = (before, after, path, batch, before_start, before_end, after_start, after_end, output_start)
    if before_count == after_count:
        positional = [
            (before_start + offset, after_start + offset)
            for offset in range(before_count)
            if _same_value(before[before_start + offset], after[after_start + offset])
        ]
        if positional:
            _process_array_matches(*args, positional)
            return

    subsequence = _identity_subsequence(before, before_start, before_end, after, after_start, after_end)
    if subsequence:
        _process_array_matches(*args, subsequence)
        return

    identity = _identity_anchors(before, before_start, before_end, after, after_start, after_end)
    if identity:
        _process_array_matches(*args, identity)
        return

    semantic = _lcs_matches(
        before[before_start:before_end],
        after[after_start:after_end],
        _semantically_aligned,
        _MAX_SEMANTIC_CELLS,
    )
    if semantic:
        absolute = [(before_start + left, after_start + right) for left, right in semantic]
        _process_array_matches(*args, absolute)
        return

    if before_count == 1 and after_count == 1:
        _diff_value(before[before_start], after[after_start], (*path, output_start), batch)
        return
    batch.emit(("p", path, output_start, before_count, after[after_start:after_end]))


def _diff_array(before: list[Any], after: list[Any], path: tuple[Any, ...], batch: _Batch) -> None:
    if before is after or equal_json(before, after):
        return
    if (
        len(before) == len(after)
        and len(before) > 1
        and not _same_value(before[0], after[0])
        and not _same_value(before[-1], after[-1])
    ):
        order = _permutation(before, after)
        if order is not None:
            batch.emit(("m", path, order))
            return
    _diff_array_region(before, after, path, batch, 0, len(before), 0, len(after), 0)


def _diff_object(before: dict[str, Any], after: dict[str, Any], path: tuple[Any, ...], batch: _Batch) -> None:
    if not RESERVED_SEGMENTS.isdisjoint(before) or not RESERVED_SEGMENTS.isdisjoint(after):
        if not equal_json(before, after):
            _emit_set(path, after, batch)
        return
    for key, value in after.items():
        if batch.overflowed:
            return
        if key in before:
            _diff_value(before[key], value, (*path, key), batch)
        else:
            _emit_set((*path, key), value, batch)
    for key in before:
        if batch.overflowed:
            return
        if key not in after:
            batch.emit(("d", (*path, key)))


def _diff_value(before: Any, after: Any, path: tuple[Any, ...], batch: _Batch) -> None:
    if before is after or batch.overflowed:
        return
    before_kind = type(before)
    after_kind = type(after)
    if before_kind is str and after_kind is str and len(path) > 0:
        _emit_string(before, after, path, batch)
        return
    if before_kind is list and after_kind is list:
        _diff_array(before, after, path, batch)
        return
    if before_kind is dict and after_kind is dict:
        _diff_object(before, after, path, batch)
        return
    if _identical(before, after):
        return
    _emit_set(path, after, batch)


def _number_cost(value: Any) -> int:
    return len(repr(value))


def _json_cost(value: Any) -> int:
    if value is None:
        return 4
    kind = type(value)
    if kind is str:
        return len(value) + 2
    if kind is bool:
        return 4 if value else 5
    if kind is int or kind is float:
        return _number_cost(value)
    if kind is list:
        cost = 2
        for index, item in enumerate(value):
            cost += _json_cost(item) + (0 if index == 0 else 1)
        return cost
    cost = 2
    for index, (key, item) in enumerate(value.items()):
        cost += len(key) + 3 + _json_cost(item) + (0 if index == 0 else 1)
    return cost


def _path_cost(path: Sequence[Any]) -> int:
    cost = 2
    for index, segment in enumerate(path):
        cost += (len(segment) + 2 if type(segment) is str else len(str(segment))) + (0 if index == 0 else 1)
    return cost


def _operation_cost(operation: Any) -> int:
    verb = operation[0]
    if verb == "r":
        return 6 + _json_cost(operation[1])
    if verb == "s":
        return 7 + _path_cost(operation[1]) + _json_cost(operation[2])
    if verb == "d":
        return 6 + _path_cost(operation[1])
    if verb == "a":
        return 7 + _path_cost(operation[1]) + len(operation[2]) + 2
    if verb == "t":
        return 7 + _path_cost(operation[1]) + len(str(operation[2]))
    if verb == "p":
        return (
            10 + _path_cost(operation[1]) + len(str(operation[2])) + len(str(operation[3])) + _json_cost(operation[4])
        )
    return 7 + _path_cost(operation[1]) + _json_cost(operation[2])


def diff_revisions(before: Any, after: Any) -> list[Op]:
    """Compute a compact operation batch that turns immutable revision ``before`` into ``after``.

    Payloads share containers with ``after``. Large edit sets fold into ``("r", after)``.
    """
    batch = _Batch()
    _diff_value(before, after, (), batch)
    operations = batch.operations
    if batch.overflowed:
        return [("r", after)]
    if len(operations) == 0 or operations[0][0] == "r":
        return operations
    delta_cost = 2
    for operation in operations:
        delta_cost += _operation_cost(operation) + 1
    if delta_cost < 65_536:
        return operations
    snapshot_cost = _json_cost(after) + 6
    return [("r", after)] if delta_cost >= snapshot_cost else operations
