"""Operation tuples, path safety, appliers and the wire codec (port of chord ``delta/index.ts``).

Ops are tuples. The tracker emits paths as tuples too; appliers and the decoder also
accept JSON-decoded lists in either position, since batches travel through storage and
the wire as JSON arrays.

========================  ==============================================================
Tuple                     Meaning
========================  ==============================================================
``("r", value)``          Replace the complete value.
``("s", path, value)``    Set an object property or array element.
``("d", path)``           Delete an object property or remove an array element.
``("a", path, text)``     Append to a string.
``("t", path, count)``    Remove ``count`` UTF-16 code units from a string's front.
``("p", path, i, n, xs)`` Splice an array: remove ``n`` items at ``i``, insert ``xs``.
``("m", path, perm)``     Reorder an array: ``new[i] = old[perm[i]]``.
========================  ==============================================================

``t`` counts UTF-16 code units, like the JavaScript original, so batches stay
interchangeable with JavaScript replicas when strings hold characters outside the BMP.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, TypeAlias, TypeVar, cast

from ..json import JsonValue

__all__ = [
    "JsonValue",
    "Seg",
    "Path",
    "NonEmptyPath",
    "PathRef",
    "Op",
    "WireOp",
    "is_replace",
    "is_base",
    "overlap",
    "RESERVED_SEGMENTS",
    "UnsafePathError",
    "PathError",
    "assert_valid_op",
    "assert_valid_wire_op",
    "assert_safe_path",
    "apply",
    "apply_immutable",
    "apply_immutable_batches",
    "Encoder",
    "Decoder",
    "encoder",
    "decoder",
]

T = TypeVar("T")

Seg: TypeAlias = "str | int"
#: Object keys and non-negative integer array indices, root first.
Path: TypeAlias = "tuple[Seg, ...]"
NonEmptyPath: TypeAlias = "tuple[Seg, *tuple[Seg, ...]]"
#: A path inline, or an id assigned by the encoder on second use.
PathRef: TypeAlias = "Path | int"

#: One decoded operation. ``r`` is the only op that replaces a whole value; ``s``, ``d``,
#: ``a`` and ``t`` never target the root, ``p`` and ``m`` may (a tracked value can itself
#: be an array). Operation shape is not canonical.
Op: TypeAlias = (
    tuple[Literal["r"], "JsonValue"]
    | tuple[Literal["s"], NonEmptyPath, "JsonValue"]
    | tuple[Literal["d"], NonEmptyPath]
    | tuple[Literal["a"], NonEmptyPath, str]
    | tuple[Literal["t"], NonEmptyPath, int]
    | tuple[Literal["p"], Path, int, int, "list[JsonValue]"]
    | tuple[Literal["m"], Path, list[int]]
)

#: What crosses a boundary: ``Op`` plus path interning (``("#", id, path)`` defines an id
#: on a path's second use, a numeric :data:`PathRef` references it) and short forms that
#: reuse the previous op's path (arity disambiguates).
WireOp: TypeAlias = (
    tuple[Literal["r"], "JsonValue"]
    | tuple[Literal["s"], PathRef, "JsonValue"]
    | tuple[Literal["s"], "JsonValue"]
    | tuple[Literal["d"], PathRef]
    | tuple[Literal["d"]]
    | tuple[Literal["a"], PathRef, str]
    | tuple[Literal["a"], str]
    | tuple[Literal["t"], PathRef, int]
    | tuple[Literal["t"], int]
    | tuple[Literal["p"], PathRef, int, int, "list[JsonValue]"]
    | tuple[Literal["p"], int, int, "list[JsonValue]"]
    | tuple[Literal["m"], PathRef, list[int]]
    | tuple[Literal["m"], list[int]]
    | tuple[Literal["#"], int, Path]
)

_SEQUENCES = (list, tuple)


# ─── Classification ──────────────────────────────────────────────────────────


def is_replace(op: Sequence[Any]) -> bool:
    """Whether ``op`` replaces the whole value (works on ops and wire ops)."""
    return op[0] == "r"


def is_base(ops: Sequence[Sequence[Any]]) -> bool:
    """Whether a batch begins with a replacement; ``r`` is at index 0 or absent, so this is exact."""
    return len(ops) > 0 and ops[0][0] == "r"


# ─── Overlap ─────────────────────────────────────────────────────────────────


def overlap(a: str, b: str, scan: int, probe: int = 64, max_candidates: int = 8) -> int:
    """Length of the longest suffix of ``a`` (within its last ``scan`` characters) that is a prefix of ``b``.

    Always correct: the result ``n`` satisfies ``a[len(a) - n:] == b[:n]``. Candidates are
    bounded, so repetitive text may report a shorter (or zero) overlap than the longest.
    Lengths are in code points.
    """
    if not a or not b or scan == 0:
        return 0
    tail = a[len(a) - scan :] if len(a) > scan else a
    length = len(tail)
    for size in (min(probe, len(b)), 1):
        head = b[:size]
        tried = 0
        at = tail.find(head)
        while at != -1:
            tried += 1
            if tried > max_candidates:
                break
            count = length - at
            if count <= len(b) and b.startswith(tail[at:]):
                return count
            at = tail.find(head, at + 1)
        if size == 1:
            break
    return 0


# ─── Path safety ─────────────────────────────────────────────────────────────

#: Segments that reach the prototype chain in JavaScript replicas. The tracker never emits
#: them as path segments and the appliers reject them, so batches stay safe for JS peers.
RESERVED_SEGMENTS: frozenset[str] = frozenset(("__proto__", "constructor", "prototype"))


class UnsafePathError(Exception):
    """A path segment that is reserved, not a non-negative integer index, or out of range."""

    def __init__(self, segment: object) -> None:
        super().__init__(f"unsafe path segment: {segment}")
        self.segment = segment


class PathError(Exception):
    """A path that does not resolve against the value."""

    def __init__(self, path: object) -> None:
        super().__init__(f"unresolvable path: {_path_repr(path)}")
        self.path = path


def _path_repr(path: object) -> str:
    if isinstance(path, (list, tuple)):
        return "[" + ",".join(f'"{seg}"' if isinstance(seg, str) else str(seg) for seg in path) + "]"
    return str(path)


def _is_index(value: object) -> bool:
    return type(value) is int and value >= 0  # type: ignore[operator]


def assert_safe_path(path: Iterable[Any]) -> None:
    """Raise :class:`UnsafePathError` for a reserved key or a segment that is not a non-negative int."""
    for seg in path:
        if type(seg) is str:
            if seg in RESERVED_SEGMENTS:
                raise UnsafePathError(seg)
        elif type(seg) is not int or seg < 0:
            raise UnsafePathError(seg)


def _assert_path_arg(path: object, non_empty: bool = False) -> None:
    if type(path) not in _SEQUENCES:
        raise TypeError("path is not an array")
    if non_empty and len(path) == 0:  # type: ignore[arg-type]
        raise TypeError("path is empty")
    assert_safe_path(path)  # type: ignore[arg-type]


def _assert_permutation(value: object) -> None:
    if type(value) not in _SEQUENCES:
        raise TypeError("m permutation is not an array")
    items = cast("Sequence[Any]", value)
    seen = bytearray(len(items))
    for index in items:
        if type(index) is not int or index < 0 or index >= len(items) or seen[index]:
            raise TypeError("m permutation is not a bijection")
        seen[index] = 1


def assert_valid_op(op: object) -> None:
    """Check verb, arity and shape of a **decoded** op (inline paths, no ``#``, no short forms).

    Payloads are not inspected; validate payloads of batches from untrusted sources.
    """
    if type(op) not in _SEQUENCES or len(op) == 0:  # type: ignore[arg-type]
        raise TypeError("op is not a tuple")
    items = cast("Sequence[Any]", op)
    verb = items[0]
    size = len(items)
    if verb == "r":
        if size != 2:
            raise TypeError("r arity")
    elif verb == "s":
        if size != 3:
            raise TypeError("s arity")
        _assert_path_arg(items[1], True)
    elif verb == "d":
        if size != 2:
            raise TypeError("d arity")
        _assert_path_arg(items[1], True)
    elif verb == "a":
        if size != 3 or type(items[2]) is not str:
            raise TypeError("a shape")
        _assert_path_arg(items[1], True)
    elif verb == "t":
        if size != 3 or not _is_index(items[2]):
            raise TypeError("t shape")
        _assert_path_arg(items[1], True)
    elif verb == "p":
        if size != 5:
            raise TypeError("p arity")
        _assert_path_arg(items[1])
        if not _is_index(items[2]):
            raise TypeError("p index")
        if not _is_index(items[3]):
            raise TypeError("p remove")
        if type(items[4]) not in _SEQUENCES:
            raise TypeError("p items")
    elif verb == "m":
        if size != 3:
            raise TypeError("m arity")
        _assert_path_arg(items[1])
        _assert_permutation(items[2])
    else:
        # Silently skipping an unknown verb is how a newer producer's op vanishes.
        raise TypeError(f"unknown op verb: {verb}")


def _assert_ref(ref: object) -> None:
    if type(ref) is int:
        if ref < 0:  # type: ignore[operator]
            raise TypeError("bad path id")
        return
    # A string is not a path: unchecked, it would resolve to the root.
    if type(ref) not in _SEQUENCES:
        raise TypeError("path is not an array")
    assert_safe_path(ref)  # type: ignore[arg-type]


def assert_valid_wire_op(op: object) -> None:
    """The same check for the wire grammar, where path ids, ``#`` and short forms are legal."""
    if type(op) not in _SEQUENCES or len(op) == 0:  # type: ignore[arg-type]
        raise TypeError("op is not a tuple")
    items = cast("Sequence[Any]", op)
    verb = items[0]
    size = len(items)
    if verb == "r":
        if size != 2:
            raise TypeError("r arity")
    elif verb == "s":
        if size == 3:
            _assert_ref(items[1])
        elif size != 2:
            raise TypeError("s arity")
    elif verb == "d":
        if size == 2:
            _assert_ref(items[1])
        elif size != 1:
            raise TypeError("d arity")
    elif verb == "a":
        if size == 3:
            _assert_ref(items[1])
            if type(items[2]) is not str:
                raise TypeError("a value")
        elif size == 2:
            if type(items[1]) is not str:
                raise TypeError("a value")
        else:
            raise TypeError("a arity")
    elif verb == "t":
        if size == 3:
            _assert_ref(items[1])
            if not _is_index(items[2]):
                raise TypeError("t count")
        elif size == 2:
            if not _is_index(items[1]):
                raise TypeError("t count")
        else:
            raise TypeError("t arity")
    elif verb == "p":
        if size == 5:
            index, remove, inserted = items[2], items[3], items[4]
            _assert_ref(items[1])
        elif size == 4:
            index, remove, inserted = items[1], items[2], items[3]
        else:
            raise TypeError("p arity")
        if not _is_index(index):
            raise TypeError("p index")
        if not _is_index(remove):
            raise TypeError("p remove")
        if type(inserted) not in _SEQUENCES:
            raise TypeError("p items")
    elif verb == "m":
        if size == 3:
            _assert_ref(items[1])
        elif size != 2:
            raise TypeError("m arity")
        _assert_permutation(items[-1])
    elif verb == "#":
        if size != 3 or not _is_index(items[1]) or type(items[2]) not in _SEQUENCES:
            raise TypeError("# shape")
        assert_safe_path(items[2])
    else:
        raise TypeError(f"unknown op verb: {verb}")


# ─── UTF-16 string counts ────────────────────────────────────────────────────


def _utf16_length(text: str) -> int:
    if text.isascii():
        return len(text)
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _drop_utf16_prefix(text: str, count: int) -> str:
    """``text`` without its first ``count`` UTF-16 code units (JavaScript ``text.slice(count)``)."""
    if count == 0:
        return text
    if text.isascii():
        return text[count:]
    encoded = text.encode("utf-16-le", "surrogatepass")
    if len(encoded) == 2 * len(text):
        return text[count:]
    return encoded[2 * count :].decode("utf-16-le", "surrogatepass")


# ─── Resolution ──────────────────────────────────────────────────────────────


def _array_has_own_string(array: list[Any], segment: str) -> bool:
    """JavaScript ``Object.hasOwn(array, segment)`` for a string segment."""
    if segment == "length":
        return True
    if not segment.isdigit() or (len(segment) > 1 and segment[0] == "0") or not segment.isascii():
        return False
    return int(segment) < len(array)


def _resolve_value(root: Any, path: Sequence[Any], whole: Sequence[Any]) -> Any:
    node = root
    for seg in path:
        kind = type(node)
        if kind is dict:
            key = seg if type(seg) is str else str(seg)
            # Own entries only.
            if key not in node:
                raise PathError(whole)
            node = node[key]
        elif kind is list:
            if type(seg) is not int:
                if _array_has_own_string(node, seg):
                    raise UnsafePathError(seg)
                raise PathError(whole)
            if seg >= len(node):
                raise PathError(whole)
            node = node[seg]
        else:
            raise PathError(whole)
    return node


def _resolve(root: Any, path: Sequence[Any], whole: Sequence[Any]) -> Any:
    node = _resolve_value(root, path, whole)
    kind = type(node)
    if kind is not dict and kind is not list:
        raise PathError(whole)
    return node


# ─── Applier ─────────────────────────────────────────────────────────────────


def apply(target: Any, ops: Iterable[Any]) -> Any:
    """Apply decoded ops to a plain mutable value and return the value (``r`` replaces it).

    Payloads are adopted, not copied: a batch is consumed by the replica it is applied to.
    Run :meth:`Decoder.decode` first if the ops came from a boundary.
    """
    return _apply_ops(target, ops)


def _apply_ops(target: Any, ops: Iterable[Any]) -> Any:
    root = target
    for op in ops:
        assert_valid_op(op)
        verb = op[0]
        if verb == "r":
            # Adopted, not copied: the consumer owns the batch it was handed.
            root = op[1]
            continue
        path = op[1]
        if verb == "p":
            node = root if len(path) == 0 else _resolve(root, path, path)
            if type(node) is not list:
                raise PathError(path)
            start = op[2]
            node[start : start + op[3]] = op[4]
            continue
        if verb == "m":
            node = root if len(path) == 0 else _resolve(root, path, path)
            permutation = op[2]
            if type(node) is not list or len(node) != len(permutation):
                raise PathError(path)
            previous = node[:]
            node[:] = [previous[index] for index in permutation]
            continue

        parent = _resolve(root, path[:-1], path)
        key = path[-1]
        if type(parent) is list:
            if type(key) is not int:
                raise UnsafePathError(key)
            if key > len(parent):
                raise UnsafePathError(key)
            if verb == "s":
                if key == len(parent):
                    parent.append(op[2])
                else:
                    parent[key] = op[2]
            elif verb == "d":
                if key >= len(parent):
                    raise PathError(path)
                del parent[key]
            else:
                current = parent[key] if key < len(parent) else None
                if type(current) is not str:
                    raise PathError(path)
                parent[key] = current + op[2] if verb == "a" else _drop_utf16_prefix(current, op[2])
            continue
        if type(key) is not str:
            key = str(key)
        if verb == "s":
            parent[key] = op[2]
        elif verb == "d":
            parent.pop(key, None)
        else:
            current = parent.get(key)
            if type(current) is not str:
                raise PathError(path)
            parent[key] = current + op[2] if verb == "a" else _drop_utf16_prefix(current, op[2])
    return root


def apply_immutable(target: T, ops: Iterable[Any]) -> T:
    """Apply one decoded batch without mutating ``target``; unchanged containers stay shared."""
    return apply_immutable_batches(target, (ops,))


def apply_immutable_batches(target: T, batches: Iterable[Iterable[Any]]) -> T:
    """Apply decoded batches as one final-result-only replay.

    One copy-on-write scope spans the whole call: containers copied for an earlier batch
    may be mutated by a later one, so no intermediate revision is exposed.
    """
    root: Any = target
    # id -> container copied in this replay. Holding the containers keeps their ids unique.
    owned: dict[int, Any] = {}
    iterator: Iterator[Iterable[Any]] = iter(batches)
    try:
        for ops in iterator:
            for op in ops:
                assert_valid_op(op)
                verb = op[0]
                if verb == "r":
                    root = op[1]
                    continue
                if verb == "p":
                    appended = _append_copy(root, op, owned)
                    if appended is not None:
                        root = appended
                        continue
                    root = _copy_containers(root, op[1], owned)
                elif verb == "m":
                    root = _copy_containers(root, op[1], owned)
                else:
                    root = _copy_containers(root, op[1][:-1], owned)
                root = _apply_ops(root, (op,))
    except BaseException:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        raise
    return cast(T, root)


def _append_copy(root: Any, op: Sequence[Any], owned: dict[int, Any]) -> Any:
    """An append to an array not yet copied, built as one ``list + items`` copy; ``None`` otherwise."""
    _, path, index, delete_count, items = op
    if delete_count != 0:
        return None
    node: Any = root
    for seg in path:
        if type(seg) is str:
            if seg in RESERVED_SEGMENTS or type(node) is not dict or seg not in node:
                return None
            node = node[seg]
        elif type(seg) is int and seg >= 0 and type(node) is list:
            if seg >= len(node):
                return None
            node = node[seg]
        else:
            return None
    if type(node) is not list or id(node) in owned or index != len(node):
        return None
    appended = node + list(items)
    owned[id(appended)] = appended
    if len(path) == 0:
        return appended
    parent_path = path[:-1]
    copied = _copy_containers(root, parent_path, owned)
    parent = _resolve(copied, parent_path, parent_path)
    parent[path[-1]] = appended
    return copied


def _copy_containers(root: Any, path: Sequence[Any], owned: dict[int, Any]) -> Any:
    kind = type(root)
    if kind is not dict and kind is not list:
        raise PathError(path)
    if id(root) in owned:
        copied_root = root
    else:
        copied_root = root.copy()
        owned[id(copied_root)] = copied_root
    destination = copied_root
    for seg in path:
        if type(destination) is list:
            if type(seg) is not int:
                if _array_has_own_string(destination, seg):
                    raise UnsafePathError(seg)
                raise PathError(path)
            if seg >= len(destination):
                raise PathError(path)
            key: Any = seg
        else:
            key = seg if type(seg) is str else str(seg)
            if key not in destination:
                raise PathError(path)
        child = destination[key]
        child_kind = type(child)
        if child_kind is not dict and child_kind is not list:
            raise PathError(path)
        if id(child) in owned:
            destination = child
            continue
        copied_child = child.copy()
        destination[key] = copied_child
        owned[id(copied_child)] = copied_child
        destination = copied_child
    return copied_root


# ─── Codec ───────────────────────────────────────────────────────────────────
#
# Path interning and arity omission live between the tracker and a boundary.
# Consecutive batches of one encoder may be concatenated: only path ids span batches,
# and every batch starts with an explicit path. Use ONE encoder/decoder pair per
# independent state stream.


class Encoder:
    """Interns repeated paths of one ordered state stream into :data:`WireOp` tuples.

    A path is interned on its SECOND use (a definition costs more than the path it
    replaces), and an op repeating the previous op's path in the same batch omits it.
    """

    __slots__ = ("_ids", "_next_id", "_seen")

    def __init__(self) -> None:
        self._seen: set[tuple[Any, ...]] = set()
        self._ids: dict[tuple[Any, ...], int] = {}
        self._next_id = 0

    def encode(self, ops: Iterable[Any]) -> list[Any]:
        # Arity omission is scoped to a batch; ids are the only cross-batch state.
        previous: tuple[Any, ...] | None = None
        out: list[Any] = []
        seen = self._seen
        ids = self._ids
        for op in ops:
            verb = op[0]
            if verb == "r":
                out.append(op)
                # A base batch is a recovery point: everything after it is self-contained.
                seen.clear()
                ids.clear()
                self._next_id = 0
                previous = None
                continue
            path = op[1]
            key = path if type(path) is tuple else tuple(path)
            if key == previous:
                out.append((verb, *op[2:]))
                continue
            ref: Any = path
            existing = ids.get(key)
            if existing is not None:
                ref = existing
            elif key in seen:
                ref = self._next_id
                self._next_id += 1
                ids[key] = ref
                out.append(("#", ref, path))
            else:
                seen.add(key)
            out.append((verb, ref, *op[2:]))
            previous = key
        return out


class Decoder:
    """Validates and restores :data:`Op` tuples from one stream's :data:`WireOp` batches.

    After a decode or apply error, discard the decoder and replica and recover from a later ``r``.
    """

    __slots__ = ("_paths",)

    def __init__(self) -> None:
        self._paths: dict[int, tuple[Any, ...]] = {}

    def decode(self, wire: Iterable[Any]) -> list[Any]:
        paths = self._paths
        previous: tuple[Any, ...] | None = None  # scoped to the batch, as in encode
        out: list[Any] = []
        for op in wire:
            assert_valid_wire_op(op)
            verb = op[0]
            if verb == "#":
                paths[op[1]] = tuple(op[2])
                continue
            if verb == "r":
                out.append(("r", op[1]))
                paths.clear()
                previous = None
                continue
            size = len(op)
            # Arity tells whether a ref is present: the short forms omit it.
            short = (size == 1) if verb == "d" else (size == 4) if verb == "p" else (size == 2)
            if short:
                if previous is None:
                    raise PathError([])
                path = previous
                rest = op[1:]
            else:
                ref = op[1]
                if type(ref) is int:
                    resolved = paths.get(ref)
                    if resolved is None:
                        raise PathError(ref)
                    path = resolved
                else:
                    path = tuple(ref)
                previous = path
                rest = op[2:]
            if verb != "p" and verb != "m" and len(path) == 0:
                raise PathError(path)
            out.append((verb, path, *rest))
        return out


def encoder() -> Encoder:
    """A fresh encoder for one ordered state stream."""
    return Encoder()


def decoder() -> Decoder:
    """A fresh decoder for one ordered state stream."""
    return Decoder()
