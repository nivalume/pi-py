"""Immutable revision tracking with exact operation batches (port of chord ``delta/tracker.ts``).

A :class:`Tracker` owns the latest adopted revision. :meth:`Tracker.begin_change` opens a
draft over it; the draft records what is mutated, :meth:`Change.prepare` turns that into
an operation batch plus the next immutable revision, and :meth:`Tracker.adopt` swaps the
root pointer.

Drafts are ``dict``/``list`` subclasses (:class:`DraftDict`, :class:`DraftList`). A draft
holds a shallow copy of the container it wraps, made when the container is first read
through the draft, and nested containers become drafts as they are read. Native reads
(``len``, ``in``, ``==``, ``json.dumps``) therefore see the draft's current content.
Every mutating ``dict``/``list`` method is overridden to record what changed in the same
form upstream records it (written, deleted and re-added keys; per-element base origin
and overridden base elements), so the emitted ops match the JavaScript tracker's. The
prepared revision is materialized by applying the ops to the base revision, which keeps
unchanged containers shared.

Ownership rules (see the delta README): roots passed to :func:`track` and
:meth:`Tracker.prepare_replace` are transferred and must never be mutated again; values
placed into a draft are validated and copied; ``tracker.value``, ``prepared.base``,
``prepared.value`` and ``prepared.ops`` are immutable by contract (nothing is frozen).
"""

from __future__ import annotations

import math
import operator
import weakref
from collections.abc import Callable, ItemsView, Iterable, Iterator, ValuesView
from typing import Any, Generic, SupportsIndex, TypeVar, cast

from ..json import UNDEFINED, _register_json_container
from .apply_immutable_trusted import apply_immutable_trusted
from .draft import Draft
from .operations import RESERVED_SEGMENTS, Op, _utf16_length, overlap

__all__ = [
    "Draft",
    "DraftDict",
    "DraftList",
    "Change",
    "Prepared",
    "Tracker",
    "detach",
    "track",
]

T = TypeVar("T")

_MAX_DELTA_OPERATIONS = 4_096
_DENSE_CANDIDATES = 256
_MAX_SIMPLE_OBJECT_NODES = 128

_OPEN = "open"
_PREPARED = "prepared"
_CONSUMED = "consumed"
_ABORTED = "aborted"
_STALE = "stale"

_dict_getitem = dict.__getitem__
_dict_setitem = dict.__setitem__
_dict_delitem = dict.__delitem__
_dict_get = dict.get
_dict_contains = dict.__contains__
_dict_iter = dict.__iter__
_dict_items = dict.items
_dict_keys = dict.keys
_dict_len = dict.__len__
_dict_clear = dict.clear
_dict_eq = dict.__eq__
_list_getitem = list.__getitem__
_list_setitem = list.__setitem__
_list_len = list.__len__
_list_iter = list.__iter__
_list_append = list.append
_list_clear = list.clear
_list_reverse = list.reverse
_list_contains = list.__contains__
_list_index = list.index
_list_count = list.count
_list_eq = list.__eq__


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


_MISSING: Any = _Missing()
#: ``_path`` of a node that no longer sits where it was read from.
_DETACHED: Any = ("<detached>",)


def _settled_error() -> TypeError:
    return TypeError("Cannot use a settled overlay")


# ─── Change context ──────────────────────────────────────────────────────────


class _Context:
    """One open, prepared or settled change and every draft node it created."""

    __slots__ = (
        "__weakref__",
        "base_revision",
        "base_value",
        "dirty",
        "nodes",
        "open",
        "owner",
        "root",
        "status",
        "tracker",
    )

    def __init__(self, tracker: weakref.ref[Any], owner: object, base_revision: int, base: Any) -> None:
        self.tracker = tracker
        self.owner = owner
        self.base_revision = base_revision
        self.status = _OPEN
        self.open = True
        self.base_value = base
        self.root: DraftDict | DraftList | None = None
        self.nodes: list[Any] = []
        self.dirty: list[Any] = []


def _release(context: _Context) -> None:
    """Make every draft of ``context`` unusable and drop what it references."""
    context.open = False
    tracker = context.tracker()
    if tracker is not None:
        tracker._contexts.pop(id(context), None)
    for node in context.nodes:
        if type(node) is DraftDict:
            _dict_clear(node)
        else:
            _list_clear(node)
            node._origin = None
        node._base = None
        node._parent = None
        node._path = None
    context.nodes = []
    context.dirty = []
    context.root = None
    context.base_value = None


def _abort_context(context: _Context) -> None:
    status = context.status
    if status in (_ABORTED, _CONSUMED, _STALE):
        return
    context.status = _ABORTED
    _release(context)


# ─── Placement copies ────────────────────────────────────────────────────────


def _not_strict(kind: type) -> TypeError:
    return TypeError(
        f"Value contains a non-JSON {kind.__name__} and is not strict JSON;"
        " drafts take plain objects (dict) and dense plain arrays (list)"
    )


def _detach(value: Any, ancestors: set[int]) -> Any:
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    if kind is float:
        if math.isfinite(value):
            return value
        raise TypeError("Value contains a non-finite number and is not strict JSON")
    if kind is dict or kind is DraftDict or kind is list or kind is DraftList:
        if kind is DraftDict or kind is DraftList:
            if not value._ctx.open:
                raise _settled_error()
        marker = id(value)
        if marker in ancestors:
            raise TypeError("Value contains cycles and is not strict JSON")
        ancestors.add(marker)
        try:
            if kind is dict or kind is DraftDict:
                result: dict[str, Any] = {}
                for key, item in _dict_items(value):
                    if type(key) is not str:
                        raise TypeError(f"Value contains a non-string key {key!r} and is not strict JSON")
                    result[key] = _detach(item, ancestors)
                return result
            return [_detach(item, ancestors) for item in _list_iter(value)]
        finally:
            ancestors.discard(marker)
    if value is UNDEFINED:
        raise TypeError("Value contains UNDEFINED and is not strict JSON")
    raise _not_strict(kind)


def _clone_placement(value: Any) -> Any:
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    return _detach(value, set())


def detach(value: Any) -> Any:
    """Copy a draft, or any strict-JSON value, into a plain alias-free JSON tree the caller owns.

    The draft counterpart of :func:`pi_chord.json.copy_json`, which rejects drafts because
    they are ``dict``/``list`` subclasses. Raises :class:`TypeError` for non-strict JSON and
    for drafts whose change is settled.
    """
    return _detach(value, set())


# ─── Draft nodes ─────────────────────────────────────────────────────────────


def _new_dict(context: _Context, raw: Any, parent: Any, key: Any, placement: bool) -> DraftDict:
    node = DraftDict(raw)
    node._ctx = context
    node._base = raw
    node._parent = parent
    node._key = key
    node._placement = placement
    node._dirty = False
    node._sub = False
    node._path = None
    node._writes = None
    node._deletes = None
    node._readded = None
    context.nodes.append(node)
    return node


def _new_list(context: _Context, raw: Any, parent: Any, key: Any, placement: bool) -> DraftList:
    node = DraftList(raw)
    node._ctx = context
    node._base = raw
    node._parent = parent
    node._key = key
    node._placement = placement
    node._dirty = False
    node._sub = False
    node._path = None
    node._origin = None
    node._lo = 0
    node._kept = len(raw)
    node._overrides = None
    node._structural = False
    node._gen = 0
    context.nodes.append(node)
    return node


def _mark_dirty(node: DraftDict | DraftList) -> None:
    if node._dirty:
        return
    node._dirty = True
    node._ctx.dirty.append(node)
    parent = node._parent
    while parent is not None and not parent._sub:
        parent._sub = True
        parent = parent._parent


def _dict_child(node: DraftDict, key: str, raw: Any, kind: type) -> Any:
    writes = node._writes
    placement = writes is not None and key in writes
    if kind is dict:
        child: Any = _new_dict(node._ctx, raw, node, key, placement)
    else:
        child = _new_list(node._ctx, raw, node, key, placement)
    _dict_setitem(node, key, child)
    return child


def _list_child(node: DraftList, index: int, raw: Any, kind: type) -> Any:
    origin = node._origin
    base_index = (node._lo + index if index < node._kept else -1) if origin is None else origin[index]
    overrides = node._overrides
    placement = base_index < 0 or (overrides is not None and base_index in overrides)
    if kind is dict:
        child: Any = _new_dict(node._ctx, raw, node, index, placement)
    else:
        child = _new_list(node._ctx, raw, node, index, placement)
    _list_setitem(node, index, child)
    return child


def _identical(left: Any, right: Any) -> bool:
    """JavaScript ``===`` on JSON values."""
    if left is right:
        return True
    kind = type(left)
    return kind is type(right) and kind is not dict and kind is not list and left == right


class _DraftValues(ValuesView[Any]):
    __slots__ = ()


class _DraftItems(ItemsView[str, Any]):
    __slots__ = ()


class DraftDict(dict[str, Any]):
    """A JSON object inside an open change. Mutate it like a ``dict``; reads return nested drafts.

    Assigning :data:`~pi_chord.json.UNDEFINED` deletes a key, like ``undefined`` upstream.
    Values placed into the draft are validated as strict JSON and copied. Once the change
    is prepared, aborted or stale, every use raises :class:`TypeError`.
    """

    __slots__ = (
        "_base",
        "_ctx",
        "_deletes",
        "_dirty",
        "_key",
        "_parent",
        "_path",
        "_placement",
        "_readded",
        "_sub",
        "_writes",
    )

    _ctx: _Context
    _base: Any
    _parent: DraftDict | DraftList | None
    _key: Any
    _placement: bool
    _dirty: bool
    _sub: bool
    _path: Any
    _writes: dict[str, None] | None
    _deletes: dict[str, None] | None
    _readded: dict[str, None] | None

    # ── reads ──

    def __getitem__(self, key: str) -> Any:
        if not self._ctx.open:
            raise _settled_error()
        value = _dict_getitem(self, key)
        kind = type(value)
        if kind is dict or kind is list:
            return _dict_child(self, key, value, kind)
        return value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if not self._ctx.open:
            raise _settled_error()
        value = _dict_get(self, key, _MISSING)
        if value is _MISSING:
            return default
        kind = type(value)
        if kind is dict or kind is list:
            return _dict_child(self, key, value, kind)
        return value

    def __iter__(self) -> Iterator[str]:
        if not self._ctx.open:
            raise _settled_error()
        return _dict_iter(self)

    def __len__(self) -> int:
        if not self._ctx.open:
            raise _settled_error()
        return _dict_len(self)

    def __contains__(self, key: object) -> bool:
        if not self._ctx.open:
            raise _settled_error()
        return _dict_contains(self, key)

    def keys(self) -> Any:
        if not self._ctx.open:
            raise _settled_error()
        return _dict_keys(self)

    def values(self) -> Any:
        if not self._ctx.open:
            raise _settled_error()
        return _DraftValues(self)

    def items(self) -> Any:
        if not self._ctx.open:
            raise _settled_error()
        return _DraftItems(self)

    def __reversed__(self) -> Iterator[str]:
        if not self._ctx.open:
            raise _settled_error()
        return reversed(_dict_keys(self))

    def __eq__(self, other: object) -> bool:
        if not self._ctx.open:
            raise _settled_error()
        return _dict_eq(self, other)

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    __hash__ = None  # type: ignore[assignment]

    def copy(self) -> dict[str, Any]:  # type: ignore[override]
        """A plain shallow ``dict`` whose nested containers are drafts (like ``{...draft}``)."""
        return {key: self[key] for key in self}

    def __copy__(self) -> dict[str, Any]:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        return detach(self)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        raise TypeError("Drafts cannot be pickled; detach() them first")

    def __or__(self, other: Any) -> dict[str, Any]:  # type: ignore[override]
        result = self.copy()
        result.update(other)
        return result

    def __ror__(self, other: Any) -> dict[str, Any]:  # type: ignore[override]
        result = dict(other)
        result.update(self.copy())
        return result

    # ── writes ──

    def __setitem__(self, key: str, value: Any) -> None:
        if not self._ctx.open:
            raise _settled_error()
        if type(key) is not str:
            raise TypeError(f"Draft object keys must be strings, not {type(key).__name__}")
        if value is UNDEFINED:
            if _dict_contains(self, key):
                self._delete(key)
            return
        kind = type(value)
        if kind is str or kind is int or kind is bool or kind is float or value is None:
            stored = value if kind is not float else _detach(value, set())
            current = _dict_get(self, key, _MISSING)
            if current is stored or (type(current) is kind and current == stored):
                return
        else:
            stored = _detach(value, set())
        writes = self._writes
        if writes is None:
            self._writes = {key: None}
        else:
            writes[key] = None
        deletes = self._deletes
        if deletes is not None and key in deletes:
            del deletes[key]
            if key in self._base:
                readded = self._readded
                if readded is None:
                    self._readded = {key: None}
                else:
                    readded[key] = None
        _dict_setitem(self, key, stored)
        if not self._dirty:
            _mark_dirty(self)

    def __delitem__(self, key: str) -> None:
        if not self._ctx.open:
            raise _settled_error()
        if not _dict_contains(self, key):
            raise KeyError(key)
        self._delete(key)

    def _delete(self, key: str) -> None:
        writes = self._writes
        if writes is not None:
            writes.pop(key, None)
        readded = self._readded
        if readded is not None:
            readded.pop(key, None)
        deletes = self._deletes
        if deletes is None:
            self._deletes = {key: None}
        else:
            deletes[key] = None
        _dict_delitem(self, key)
        if not self._dirty:
            _mark_dirty(self)

    def pop(self, key: str, default: Any = _MISSING) -> Any:  # type: ignore[override]
        if not self._ctx.open:
            raise _settled_error()
        if not _dict_contains(self, key):
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self[key]
        self._delete(key)
        return value

    def popitem(self) -> tuple[str, Any]:
        if not self._ctx.open:
            raise _settled_error()
        if _dict_len(self) == 0:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(_dict_keys(self)))
        value = self[key]
        self._delete(key)
        return key, value

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if not self._ctx.open:
            raise _settled_error()
        if not _dict_contains(self, key):
            self[key] = default
        return self[key]

    def update(self, other: Any = (), /, **kwargs: Any) -> None:  # type: ignore[override]
        if not self._ctx.open:
            raise _settled_error()
        if isinstance(other, dict) or hasattr(other, "keys"):
            keys = other.keys()
            pairs: Iterable[tuple[Any, Any]] = [(key, other[key]) for key in keys]
        else:
            pairs = list(other)
        for key, value in pairs:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __ior__(self, other: Any) -> DraftDict:  # type: ignore[override,misc]
        self.update(other)
        return self

    def clear(self) -> None:
        if not self._ctx.open:
            raise _settled_error()
        for key in list(_dict_keys(self)):
            self._delete(key)


class DraftList(list[Any]):
    """A JSON array inside an open change. Mutate it like a ``list``; reads return nested drafts.

    Structural edits (``append``, ``extend``, ``insert``, ``pop``, ``remove``, ``clear``,
    slice assignment and deletion, ``sort``, ``reverse``, ``+=``, ``*=``) are recorded per
    element, so held handles follow their elements and the batch carries splices and
    permutations. Index writes must address an existing element. Values placed into the
    draft are validated as strict JSON and copied.
    """

    __slots__ = (
        "_base",
        "_ctx",
        "_dirty",
        "_gen",
        "_key",
        "_kept",
        "_lo",
        "_origin",
        "_overrides",
        "_parent",
        "_path",
        "_placement",
        "_structural",
        "_sub",
    )

    _ctx: _Context
    _base: Any
    _parent: DraftDict | DraftList | None
    _key: Any
    _placement: bool
    _dirty: bool
    _sub: bool
    _path: Any
    #: Base index of each element (-1 when inserted), or None while elements
    #: ``[0, _kept)`` are base elements ``_lo.._lo+_kept-1`` and the rest are inserted
    #: (the shape that appends, pops and front removals keep).
    _origin: list[int] | None
    _lo: int
    _kept: int
    #: Base indices whose element was overwritten, in first-write order.
    _overrides: dict[int, None] | None
    _structural: bool
    #: Structural edit count, to detect edits made by a sort key function.
    _gen: int

    # ── reads ──

    def __getitem__(self, index: Any) -> Any:
        if not self._ctx.open:
            raise _settled_error()
        if type(index) is not int:
            if isinstance(index, slice):
                return [self[at] for at in range(*index.indices(_list_len(self)))]
            index = operator.index(index)
        value = _list_getitem(self, index)
        kind = type(value)
        if kind is dict or kind is list:
            if index < 0:
                index += _list_len(self)
            return _list_child(self, index, value, kind)
        return value

    def __iter__(self) -> Iterator[Any]:
        if not self._ctx.open:
            raise _settled_error()
        return self._iterate()

    def _iterate(self) -> Iterator[Any]:
        context = self._ctx
        index = 0
        while True:
            if not context.open:
                raise _settled_error()
            try:
                value = _list_getitem(self, index)
            except IndexError:
                return
            kind = type(value)
            if kind is dict or kind is list:
                value = _list_child(self, index, value, kind)
            yield value
            index += 1

    def __reversed__(self) -> Iterator[Any]:
        if not self._ctx.open:
            raise _settled_error()
        return self._iterate_reversed()

    def _iterate_reversed(self) -> Iterator[Any]:
        context = self._ctx
        index = _list_len(self) - 1
        while index >= 0:
            if not context.open:
                raise _settled_error()
            if index < _list_len(self):
                yield self[index]
            index -= 1

    def __len__(self) -> int:
        if not self._ctx.open:
            raise _settled_error()
        return _list_len(self)

    def __contains__(self, value: object) -> bool:
        if not self._ctx.open:
            raise _settled_error()
        return _list_contains(self, value)

    def index(self, value: Any, start: SupportsIndex = 0, stop: SupportsIndex = 2**63 - 1) -> int:
        if not self._ctx.open:
            raise _settled_error()
        return _list_index(self, value, start, stop)

    def count(self, value: Any) -> int:
        if not self._ctx.open:
            raise _settled_error()
        return _list_count(self, value)

    def __eq__(self, other: object) -> bool:
        if not self._ctx.open:
            raise _settled_error()
        return _list_eq(self, other)

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    __hash__ = None  # type: ignore[assignment]

    def copy(self) -> list[Any]:
        """A plain shallow ``list`` whose nested containers are drafts."""
        return [self[at] for at in range(len(self))]

    def __copy__(self) -> list[Any]:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        return detach(self)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        raise TypeError("Drafts cannot be pickled; detach() them first")

    def __add__(self, other: Any) -> list[Any]:  # type: ignore[override]
        return self.copy() + list(other)

    def __radd__(self, other: Any) -> list[Any]:
        return list(other) + self.copy()

    def __mul__(self, count: SupportsIndex) -> list[Any]:
        return self.copy() * count

    __rmul__ = __mul__

    # ── writes ──

    def _writable(self) -> None:
        if not self._ctx.open:
            raise _settled_error()

    def _base_index(self, index: int) -> int:
        origin = self._origin
        if origin is None:
            return self._lo + index if index < self._kept else -1
        return origin[index]

    def _write(self, index: int, stored: Any) -> None:
        current = _list_getitem(self, index)
        kind = type(stored)
        primitive = kind is not dict and kind is not list
        if primitive and (current is stored or (type(current) is kind and current == stored)):
            return
        base_index = self._base_index(index)
        if base_index >= 0:
            overrides = self._overrides
            if primitive and _identical(stored, self._base[base_index]):
                if overrides is not None:
                    overrides.pop(base_index, None)
            elif overrides is None:
                self._overrides = {base_index: None}
            else:
                overrides[base_index] = None
        _list_setitem(self, index, stored)
        if not self._dirty:
            _mark_dirty(self)

    def _splice(self, start: int, remove: int, items: list[Any]) -> None:
        if remove == 0 and not items:
            return
        origin = self._origin
        if origin is None:
            kept = self._kept
            if start + remove >= kept:
                # Edits that touch only the end keep the "base window + inserted tail" form.
                if start < kept:
                    self._kept = start
                    if start == 0:
                        self._lo = 0  # no base element left: one removal run
            elif start == 0 and not items:
                self._lo += remove
                self._kept = kept - remove
            else:
                origin = self._ensure_origin()
        _list_setitem(self, slice(start, start + remove), items)
        if origin is not None:
            origin[start : start + remove] = [-1] * len(items)
        self._structural = True
        self._gen += 1
        if not self._dirty:
            _mark_dirty(self)

    def _ensure_origin(self) -> list[int]:
        origin = self._origin
        if origin is None:
            kept = self._kept
            origin = list(range(self._lo, self._lo + kept))
            origin.extend([-1] * (_list_len(self) - kept))
            self._origin = origin
        return origin

    def __setitem__(self, index: Any, value: Any) -> None:
        self._writable()
        size = _list_len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(size)
            items = [_clone_placement(item) for item in list(value)]
            if step == 1:
                self._splice(start, max(stop - start, 0), items)
                return
            positions = range(start, stop, step)
            if len(positions) != len(items):
                raise ValueError(
                    f"attempt to assign sequence of size {len(items)} to extended slice of size {len(positions)}"
                )
            for at, item in zip(positions, items, strict=True):
                self._write(at, item)
            return
        index = operator.index(index)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("list assignment index out of range")
        self._write(index, _clone_placement(value))

    def __delitem__(self, index: Any) -> None:
        self._writable()
        size = _list_len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(size)
            if step == 1:
                self._splice(start, max(stop - start, 0), [])
                return
            for at in sorted(range(start, stop, step), reverse=True):
                self._splice(at, 1, [])
            return
        index = operator.index(index)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("list assignment index out of range")
        self._splice(index, 1, [])

    def append(self, value: Any) -> None:
        self._writable()
        stored = _clone_placement(value)
        if self._origin is None:
            _list_append(self, stored)
            self._structural = True
            self._gen += 1
            if not self._dirty:
                _mark_dirty(self)
        else:
            self._splice(_list_len(self), 0, [stored])

    def extend(self, values: Iterable[Any]) -> None:
        self._writable()
        items = [_clone_placement(item) for item in list(values)]
        self._splice(_list_len(self), 0, items)

    def __iadd__(self, values: Iterable[Any]) -> DraftList:  # type: ignore[override,misc]
        self.extend(values)
        return self

    def __imul__(self, count: SupportsIndex) -> DraftList:  # type: ignore[override,misc]
        self._writable()
        times = operator.index(count)
        size = _list_len(self)
        if times <= 0:
            self._splice(0, size, [])
        elif times > 1 and size > 0:
            current = [self[at] for at in range(size)]
            items = [_clone_placement(item) for _ in range(times - 1) for item in current]
            self._splice(size, 0, items)
        return self

    def insert(self, index: SupportsIndex, value: Any) -> None:
        self._writable()
        at = operator.index(index)
        size = _list_len(self)
        if at < 0:
            at = max(at + size, 0)
        elif at > size:
            at = size
        self._splice(at, 0, [_clone_placement(value)])

    def pop(self, index: SupportsIndex = -1) -> Any:
        self._writable()
        size = _list_len(self)
        if size == 0:
            raise IndexError("pop from empty list")
        at = operator.index(index)
        if at < 0:
            at += size
        if at < 0 or at >= size:
            raise IndexError("pop index out of range")
        value = self[at]
        self._splice(at, 1, [])
        return value

    def remove(self, value: Any) -> None:
        self._writable()
        self._splice(_list_index(self, value), 1, [])

    def clear(self) -> None:
        self._writable()
        self._splice(0, _list_len(self), [])

    def reverse(self) -> None:
        self._writable()
        if _list_len(self) < 2:
            return
        origin = self._ensure_origin()
        _list_reverse(self)
        origin.reverse()
        self._structural = True
        self._gen += 1
        if not self._dirty:
            _mark_dirty(self)

    def sort(self, *, key: Callable[[Any], Any] | None = None, reverse: bool = False) -> None:  # type: ignore[override]
        """Sort in place like ``list.sort``; ``key`` and comparisons see drafts for containers.

        Elements keep their identity, so held handles follow them. A key function (or a
        ``functools.cmp_to_key`` comparator) may edit elements; structural edits it makes
        are kept after the sorted elements, and an element it duplicated is copied.
        """
        self._writable()
        size = _list_len(self)
        if size < 2:
            return
        values = [self[at] for at in range(size)]
        snapshot = list(_list_iter(self))
        origins = [self._base_index(at) for at in range(size)]
        overrides_before = set(self._overrides or ())
        generation = self._gen
        if key is None:
            order = sorted(range(size), key=values.__getitem__, reverse=reverse)
        else:
            order = sorted(range(size), key=lambda at: key(values[at]), reverse=reverse)
        if not self._ctx.open:
            raise _settled_error()
        overrides = self._overrides
        if overrides_before or overrides:
            # The sorted entries keep the overrides they had when the sort began.
            for at in order:
                base_index = origins[at]
                if base_index < 0:
                    continue
                if base_index in overrides_before:
                    if overrides is None:
                        overrides = self._overrides = {}
                    if base_index not in overrides:
                        overrides[base_index] = None
                elif overrides is not None:
                    overrides.pop(base_index, None)
        reentrant = self._gen != generation
        current_size = _list_len(self)
        same = current_size >= size and all(
            _list_getitem(self, at) is snapshot[order[at]] and self._base_index(at) == origins[order[at]]
            for at in range(size)
        )
        if not same:
            replace = min(size, current_size)
            origin = self._ensure_origin()
            _list_setitem(self, slice(0, replace), [snapshot[at] for at in order])
            origin[0:replace] = [origins[at] for at in order]
            self._structural = True
            self._gen += 1
            if not self._dirty:
                _mark_dirty(self)
        if reentrant:
            self._deduplicate()

    def _deduplicate(self) -> None:
        origin = self._ensure_origin()
        seen_base: set[int] = set()
        seen_ids: set[int] = set()
        changed = False
        for at in range(_list_len(self)):
            base_index = origin[at]
            value = _list_getitem(self, at)
            kind = type(value)
            container = kind is DraftDict or kind is DraftList or kind is dict or kind is list
            if base_index >= 0:
                duplicate = base_index in seen_base
                seen_base.add(base_index)
            elif container:
                duplicate = id(value) in seen_ids
            else:
                duplicate = False
            if container:
                seen_ids.add(id(value))
            if duplicate:
                _list_setitem(self, at, _detach(value, set()))
                origin[at] = -1
                changed = True
        if changed:
            self._structural = True
            self._gen += 1
            if not self._dirty:
                _mark_dirty(self)


# ─── Emission ────────────────────────────────────────────────────────────────


def _clone_node(node: Any) -> Any:
    if type(node) is DraftDict:
        return {key: _clone_stored(value) for key, value in _dict_items(node)}
    return [_clone_stored(value) for value in _list_iter(node)]


def _clone_stored(value: Any) -> Any:
    kind = type(value)
    if kind is DraftDict or kind is DraftList:
        if value._dirty or value._sub:
            return _clone_node(value)
        return value._base
    return value


def _equal_trusted(left: Any, right: Any) -> bool:
    if left is right:
        return True
    kind = type(left)
    if kind is not type(right):
        return False
    if kind is dict or kind is list:
        return left == right and _strict_equal(left, right)
    return left == right


def _strict_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    kind = type(left)
    if kind is not type(right):
        return False
    if kind is list:
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right, strict=True))
    if kind is dict:
        if len(left) != len(right):
            return False
        return all(key in right and _strict_equal(value, right[key]) for key, value in left.items())
    return left == right


def _emit_changed(operations: list[Any], path: tuple[Any, ...], before: Any, after: Any) -> bool:
    after_kind = type(after)
    if after_kind is not dict and after_kind is not list:
        if before is not _MISSING and _identical(before, after):
            return False
    elif type(before) is after_kind and _equal_trusted(before, after):
        return False
    if after_kind is str and type(before) is str:
        if len(after) > len(before) and after.startswith(before):
            operations.append(("a", path, after[len(before) :]))
            return True
        shared = overlap(before, after, 65_536)
        if shared > 0:
            operations.append(("t", path, _utf16_length(before[: len(before) - shared])))
            if len(after) > shared:
                operations.append(("a", path, after[shared:]))
            return True
    operations.append(("s", path, after))
    return True


def _has_reserved_mutation(node: DraftDict) -> bool:
    writes = node._writes
    if writes is not None and not RESERVED_SEGMENTS.isdisjoint(writes):
        return True
    deletes = node._deletes
    return deletes is not None and not RESERVED_SEGMENTS.isdisjoint(deletes)


def _has_placement_ancestor(node: Any) -> bool:
    current = node
    while current._parent is not None:
        if current._placement:
            return True
        current = current._parent
    return False


def _has_folded_ancestor(node: Any, folded: set[int]) -> bool:
    parent = node._parent
    while parent is not None:
        if id(parent) in folded:
            return True
        parent = parent._parent
    return False


def _reserved_at(path: tuple[Any, ...]) -> int:
    for at, segment in enumerate(path):
        if type(segment) is str and segment in RESERVED_SEGMENTS:
            return at
    return -1


class _Candidates:
    __slots__ = ("bits", "indices", "length")

    def __init__(self, length: int) -> None:
        self.indices: list[int] = []
        self.bits: bytearray | None = None
        self.length = length

    def add(self, index: int) -> None:
        bits = self.bits
        if bits is not None:
            bits[index] = 1
            return
        indices = self.indices
        indices.append(index)
        if len(indices) < _DENSE_CANDIDATES:
            return
        bits = self.bits = bytearray(self.length)
        for existing in indices:
            bits[existing] = 1
        indices.clear()

    def regions(self) -> list[tuple[int, int]]:
        bits = self.bits
        if bits is None:
            return []
        regions: list[tuple[int, int]] = []
        size = len(bits)
        at = 0
        while at < size:
            at = bits.find(1, at)
            if at < 0:
                break
            start = at
            end = at
            count = 0
            gap = 0
            while at < size:
                if bits[at]:
                    count += 1
                    end = at
                    gap = 0
                else:
                    gap += 1
                    if gap > 1:
                        break
                at += 1
            length = end - start + 1
            if count >= _DENSE_CANDIDATES and count * 2 >= length:
                regions.append((start, length))
        return regions


def _depth(entry: tuple[int, Any, Any]) -> int:
    return entry[0]


def _in_regions(regions: list[tuple[int, int]], index: int) -> bool:
    return any(start <= index < start + length for start, length in regions)


class _Emitter:
    """One prepare's operation emission: resolves paths, folds dense array regions and reserved keys."""

    __slots__ = ("direct", "indexes", "operations", "regions")

    def __init__(self) -> None:
        self.operations: list[Any] = []
        self.indexes: dict[int, dict[int, int]] = {}
        self.regions: dict[int, list[tuple[int, int]]] = {}
        #: Whether the candidate may be cloned from the draft instead of applying the ops.
        self.direct = False

    def run(self, context: _Context) -> list[Any]:
        simple = self._run_simple(context)
        if simple is not None:
            return simple
        dirty = context.dirty
        operations = self.operations
        regions = self.regions
        dense: list[DraftList] = []
        count = len(dirty)
        # A dense region needs 256 candidates in one array: dirty descendants plus its own overrides.
        if count >= _DENSE_CANDIDATES or any(
            type(node) is DraftList
            and node._overrides is not None
            and count + len(node._overrides) >= _DENSE_CANDIDATES
            for node in dirty
        ):
            candidates: dict[int, tuple[DraftList, _Candidates]] = {}
            for node in dirty:
                self._record_dense_position(node, candidates)
                if type(node) is DraftList and not node._structural and node._overrides:
                    entry = candidates.get(id(node))
                    if entry is None:
                        entry = candidates[id(node)] = (node, _Candidates(len(node._base)))
                    for index in node._overrides:
                        entry[1].add(index)
            for array, found in candidates.values():
                path = self.resolve(array)
                if path is None or not RESERVED_SEGMENTS.isdisjoint(path):
                    continue
                spans = found.regions()
                if spans:
                    regions[id(array)] = spans
                    dense.append(array)

        emission: dict[int, tuple[Any, tuple[Any, ...]]] = {}
        forced: dict[int, tuple[Any, tuple[Any, ...]]] = {}
        for node in dirty:
            path = self.resolve(node)
            if path is None:
                continue
            if regions and self._covered(node, path):
                continue
            emission[id(node)] = (node, path)
            if type(node) is DraftDict and _has_reserved_mutation(node):
                forced[id(node)] = (node, path)
            if not RESERVED_SEGMENTS.isdisjoint(path):
                reserved = _reserved_at(path)
                ancestor: Any = node
                for _ in range(len(path) - reserved):
                    ancestor = ancestor._parent
                forced[id(ancestor)] = (ancestor, path[:reserved])
        emission.update(forced)
        for array in dense:
            path = self.resolve(array)
            if path is not None and not self._covered(array, path):
                emission[id(array)] = (array, path)

        ordered = sorted(emission.values(), key=lambda entry: len(entry[1]))
        folded: set[int] = set()
        for node, path in ordered:
            if _has_placement_ancestor(node) or (folded and _has_folded_ancestor(node, folded)):
                continue
            if id(node) in forced:
                value = _clone_node(node)
                operations.append(("r", value) if len(path) == 0 else ("s", path, value))
                folded.add(id(node))
                continue
            if type(node) is DraftList:
                self._emit_array(node, path, regions.get(id(node)))
            else:
                self._emit_object(node, path)
            if len(operations) > _MAX_DELTA_OPERATIONS:
                root = context.root
                assert root is not None
                return [("r", _clone_node(root))]
        return operations

    def _run_simple(self, context: _Context) -> list[Any] | None:
        """Object-only edits outside arrays and placements: the same ops, emitted without the general bookkeeping."""
        nodes: list[tuple[int, Any, tuple[Any, ...]]] = []
        for node in context.dirty:
            if type(node) is not DraftDict or _has_reserved_mutation(node):
                return None
            parent = node._parent
            current = node
            while parent is not None:
                if type(parent) is not DraftDict or current._placement:
                    return None
                current = parent
                parent = parent._parent
            path = self.resolve(node)
            if path is None:
                continue
            if not RESERVED_SEGMENTS.isdisjoint(path):
                return None
            nodes.append((len(path), node, path))
            if len(nodes) > _MAX_SIMPLE_OBJECT_NODES:
                return None
        nodes.sort(key=_depth)
        operations = self.operations
        direct = True
        for _, node, path in nodes:
            if self._emit_object(node, path):
                direct = False
            if len(operations) > _MAX_DELTA_OPERATIONS:
                root = context.root
                assert root is not None
                return [("r", _clone_node(root))]
        self.direct = direct
        return operations

    # ── paths ──

    def resolve(self, node: Any) -> tuple[Any, ...] | None:
        path = node._path
        if path is None:
            chain: list[Any] = []
            current = node
            while True:
                path = current._path
                if path is not None:
                    break
                parent = current._parent
                if parent is None:
                    path = current._path = ()
                    break
                chain.append(current)
                current = parent
            for child in reversed(chain):
                if path is not _DETACHED:
                    parent = child._parent
                    if type(parent) is DraftDict:
                        key = child._key
                        path = (*path, key) if _dict_get(parent, key, _MISSING) is child else _DETACHED
                    else:
                        index = self._index_in(parent, child)
                        path = _DETACHED if index is None else (*path, index)
                child._path = path
        return None if path is _DETACHED else path

    def _index_in(self, parent: DraftList, child: Any) -> int | None:
        if not parent._structural:
            index = child._key
            if index < _list_len(parent) and _list_getitem(parent, index) is child:
                return index
            return None
        indexes = self.indexes.get(id(parent))
        if indexes is None:
            indexes = {}
            for at, value in enumerate(_list_iter(parent)):
                kind = type(value)
                if kind is DraftDict or kind is DraftList:
                    indexes[id(value)] = at
            self.indexes[id(parent)] = indexes
        return indexes.get(id(child))

    # ── dense regions ──

    @staticmethod
    def _record_dense_position(node: Any, candidates: dict[int, tuple[DraftList, _Candidates]]) -> None:
        child = node
        parent = child._parent
        while parent is not None:
            if type(parent) is DraftList:
                if parent._structural:
                    return
                index = child._key
                if index >= _list_len(parent) or _list_getitem(parent, index) is not child:
                    return
                entry = candidates.get(id(parent))
                if entry is None:
                    entry = candidates[id(parent)] = (parent, _Candidates(len(parent._base)))
                entry[1].add(index)
                return
            child = parent
            parent = child._parent

    def _covered(self, node: Any, path: tuple[Any, ...]) -> bool:
        regions = self.regions
        parent = node._parent
        while parent is not None:
            spans = regions.get(id(parent))
            if spans is not None:
                parent_path = self.resolve(parent)
                if parent_path is not None and len(path) > len(parent_path):
                    depth = len(parent_path)
                    if path[:depth] == parent_path:
                        index = path[depth]
                        if type(index) is int and _in_regions(spans, index):
                            return True
            parent = parent._parent
        return False

    # ── objects ──

    def _emit_object(self, node: DraftDict, path: tuple[Any, ...]) -> bool:
        """Emit one object's edits; true when a container write normalized to no operation."""
        operations = self.operations
        base = node._base
        normalized = False
        readded = node._readded
        if readded:
            for key in readded:
                if key in base:
                    operations.append(("d", (*path, key)))
        writes = node._writes
        if writes:
            for key in writes:
                if len(operations) > _MAX_DELTA_OPERATIONS:
                    return normalized
                before = _MISSING if (readded and key in readded) else base.get(key, _MISSING)
                after = _clone_stored(_dict_getitem(node, key))
                if not _emit_changed(operations, (*path, key), before, after):
                    kind = type(after)
                    if (kind is dict or kind is list) and type(before) is kind:
                        normalized = True
        deletes = node._deletes
        if deletes:
            for key in deletes:
                if len(operations) > _MAX_DELTA_OPERATIONS:
                    return normalized
                if key in base:
                    operations.append(("d", (*path, key)))
        return normalized

    # ── arrays ──

    def _emit_array(self, node: DraftList, path: tuple[Any, ...], regions: list[tuple[int, int]] | None) -> None:
        operations = self.operations
        base = node._base
        if regions:
            for start, length in regions:
                if len(operations) > _MAX_DELTA_OPERATIONS:
                    return
                items = [_clone_stored(_list_getitem(node, at)) for at in range(start, start + length)]
                operations.append(("p", path, start, length, items))
        origin = node._origin
        positions: dict[int, int] | None = None
        if node._structural:
            base_length = len(base)
            if origin is None:
                lo = node._lo
                kept = node._kept
                if lo + kept < base_length:
                    operations.append(("p", path, lo + kept, base_length - lo - kept, []))
                if lo > 0:
                    operations.append(("p", path, 0, lo, []))
                if _list_len(node) > kept:
                    tail = [_clone_stored(value) for value in _list_getitem(node, slice(kept, None))]
                    operations.append(("p", path, kept, 0, tail))
            elif not self._emit_structure(node, path, origin, base_length):
                return
        overrides = node._overrides
        if overrides:
            if origin is not None:
                positions = {index: at for at, index in enumerate(origin) if index >= 0}
            lo = node._lo
            kept = node._kept
            for base_index in overrides:
                if len(operations) > _MAX_DELTA_OPERATIONS:
                    return
                if positions is None:
                    index: int | None = base_index - lo if lo <= base_index < lo + kept else None
                else:
                    index = positions.get(base_index)
                if index is None or (regions and _in_regions(regions, index)):
                    continue
                _emit_changed(
                    operations,
                    (*path, index),
                    base[base_index],
                    _clone_stored(_list_getitem(node, index)),
                )

    def _emit_structure(self, node: DraftList, path: tuple[Any, ...], origin: list[int], base_length: int) -> bool:
        """Removal runs (from the end backwards), a permutation if kept elements moved, then insert runs."""
        operations = self.operations
        target = [index for index in origin if index >= 0]
        retained = sorted(target)
        end = base_length
        for index in reversed(retained):
            if index + 1 < end:
                if len(operations) > _MAX_DELTA_OPERATIONS:
                    return False
                operations.append(("p", path, index + 1, end - index - 1, []))
            end = index
        if end > 0:
            operations.append(("p", path, 0, end, []))
        if target != retained:
            ranks = {index: rank for rank, index in enumerate(retained)}
            operations.append(("m", path, [ranks[index] for index in target]))
        size = len(origin)
        at = 0
        while True:
            try:
                at = origin.index(-1, at)
            except ValueError:
                return True
            stop = at + 1
            while stop < size and origin[stop] < 0:
                stop += 1
            if len(operations) > _MAX_DELTA_OPERATIONS:
                return False
            items = [_clone_stored(value) for value in _list_getitem(node, slice(at, stop))]
            operations.append(("p", path, at, 0, items))
            at = stop


# ─── Lifecycle ───────────────────────────────────────────────────────────────


class Prepared(Generic[T]):
    """A materialized candidate revision and the exact batch producing it from :attr:`base`.

    ``base``, ``value`` and ``ops`` (with their paths and payloads) are immutable by
    contract; payloads may be the same objects as parts of ``value``. An empty ``ops``
    means ``value is base``.
    """

    __slots__ = ("_ctx", "base", "ops", "value")

    base: T
    value: T
    ops: list[Op]

    def __init__(self, context: _Context, base: T, value: T, operations: list[Op]) -> None:
        self._ctx = context
        self.base = base
        self.value = value
        self.ops = operations

    @property
    def base_revision(self) -> int:
        """The tracker revision this candidate was prepared from."""
        return self._ctx.base_revision

    def abort(self) -> None:
        """Prevent adoption. The candidate stays readable."""
        _abort_context(self._ctx)


class Change(Generic[T]):
    """An open draft over one tracker revision. Settle it with :meth:`prepare` or :meth:`abort`."""

    __slots__ = ("_ctx", "_settled")

    def __init__(self, context: _Context) -> None:
        self._ctx = context
        self._settled = False

    @property
    def state(self) -> Draft[T]:
        """The root draft; it may stay open across ``await``. Raises :class:`TypeError` once settled."""
        context = self._ctx
        if not context.open:
            raise _settled_error()
        root = context.root
        if root is None:
            base = context.base_value
            if type(base) is dict:
                root = _new_dict(context, base, None, None, False)
            else:
                root = _new_list(context, base, None, None, False)
            context.root = root
        return cast(Any, root)

    def prepare(self) -> Prepared[T]:
        """Emit the batch and materialize the candidate; the draft becomes unusable. Authority is unchanged."""
        if self._settled:
            raise RuntimeError("Change has already been settled")
        context = self._ctx
        if not context.open:
            raise _settled_error()
        context.status = _PREPARED
        context.open = False
        try:
            base = context.base_value
            if context.dirty:
                emitter = _Emitter()
                operations = emitter.run(context)
                if not operations:
                    value = base
                elif emitter.direct:
                    value = _clone_node(context.root)
                else:
                    value = apply_immutable_trusted(base, operations)
            else:
                operations = []
                value = base
            prepared: Prepared[T] = Prepared(context, base, value, operations)
            _release(context)
            self._settled = True
            return prepared
        except BaseException:
            context.status = _ABORTED
            _release(context)
            self._settled = True
            raise

    def abort(self) -> None:
        """Discard the draft, or, after :meth:`prepare`, prevent adopting its result. Idempotent."""
        if self._settled:
            if self._ctx.status == _PREPARED:
                self._ctx.status = _ABORTED
            return
        self._settled = True
        _abort_context(self._ctx)


class Tracker(Generic[T]):
    """One revision sequence of an immutable JSON root."""

    __slots__ = ("__weakref__", "_contexts", "_owner", "_prune_at", "_revision", "_self", "_value")

    def __init__(self, initial: T) -> None:
        self._owner = object()
        # Open changes, so adoption can stale them; weak, so a dropped change is not retained.
        self._contexts: dict[int, weakref.ref[_Context]] = {}
        self._prune_at = 256
        self._value = initial
        self._revision = 0
        self._self = weakref.ref(self)

    @property
    def value(self) -> T:
        """The latest adopted revision (immutable by contract)."""
        return self._value

    @property
    def revision(self) -> int:
        """Number of adoptions so far, no-ops included."""
        return self._revision

    def begin_change(self) -> Change[T]:
        """Open a draft over :attr:`value`. Several changes may be open at once; adopting one stales the rest."""
        context = _Context(self._self, self._owner, self._revision, self._value)
        contexts = self._contexts
        contexts[id(context)] = weakref.ref(context)
        if len(contexts) >= self._prune_at:
            for key, reference in list(contexts.items()):
                if reference() is None:
                    del contexts[key]
            self._prune_at = max(256, 2 * len(contexts))
        return Change(context)

    def prepare_replace(self, value: T) -> Prepared[T]:
        """Prepare a whole-root replacement, taking ownership of ``value`` in O(1).

        If ``value`` deeply equals :attr:`value`, ``ops`` is empty and the current root is
        kept (the comparison may traverse both trees); otherwise ``ops`` is ``[("r", value)]``.
        """
        base = self._value
        context = _Context(self._self, self._owner, self._revision, base)
        context.status = _PREPARED
        context.open = False
        context.base_value = None
        if _equal_trusted(base, value):
            return Prepared(context, base, base, [])
        operations: list[Op] = [("r", cast(Any, value))]
        return Prepared(context, base, value, operations)

    def adopt(self, prepared: Prepared[T]) -> None:
        """Make ``prepared.value`` the current revision; rejects foreign, stale, aborted and used preparations."""
        context = prepared._ctx if isinstance(prepared, Prepared) else None
        if context is None or context.owner is not self._owner:
            raise RuntimeError("Prepared change belongs to a different tracker")
        status = context.status
        if status == _CONSUMED:
            raise RuntimeError("Prepared change has already been used")
        if status == _ABORTED:
            raise RuntimeError("Prepared change has been aborted")
        if status == _STALE:
            raise RuntimeError("Prepared change is stale")
        if status != _PREPARED:
            raise RuntimeError("Prepared change is not ready")
        if context.base_revision != self._revision or self._value is not prepared.base:
            context.status = _STALE
            raise RuntimeError("Prepared change is stale")
        self._value = prepared.value
        context.status = _CONSUMED
        self._revision += 1
        self._invalidate(context)

    def _invalidate(self, winner: _Context) -> None:
        if not self._contexts:
            return
        references = list(self._contexts.values())
        self._contexts.clear()
        self._prune_at = 256
        for reference in references:
            context = reference()
            if context is None or context is winner:
                continue
            if context.status == _OPEN or context.status == _PREPARED:
                context.status = _STALE
            _release(context)


def track(initial: T) -> Tracker[T]:
    """Take immutable ownership of an alias-free strict-JSON ``dict`` or ``list`` root in O(1).

    The root is trusted and never walked: it must not be mutated again, and must not reuse
    one container at two places.
    """
    if type(initial) is not dict and type(initial) is not list:
        raise TypeError("track() takes a dict or list root")
    return Tracker(initial)


_register_json_container(DraftDict, detach)
_register_json_container(DraftList, detach)
