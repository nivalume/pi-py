"""Scan order and cursor helpers shared by the built-in storages (port of ``storage/scan.ts``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from ..errors import StorageRequestError
from ..types import ScanOrder

__all__ = ["ScanStart", "next_cursor", "scan_start"]

_MAX_SAFE_INTEGER = 2**53 - 1
_ORDERS = ("ascending", "descending")


class ScanStart(NamedTuple):
    """Where a built-in storage scan starts: its order and the last ID a previous page returned."""

    order: ScanOrder
    after: int | None


def scan_start(requested: object, cursor: Mapping[str, Any] | None, fallback: ScanOrder) -> ScanStart:
    """Resolve a scan's order and position.

    A cursor continues in the order it was created with, whether the query repeats that order or
    omits it; a different ``order`` raises. A cursor without an order, written before scans had one,
    continues in the scan's default order.
    """
    if requested is not None and requested not in _ORDERS:
        raise StorageRequestError(f"Invalid scan order: {requested}")
    if cursor is None:
        return ScanStart(requested if requested is not None else fallback, None)  # type: ignore[arg-type]
    after = cursor.get("after")
    stored = cursor.get("order")
    if type(after) is not int or not -_MAX_SAFE_INTEGER <= after <= _MAX_SAFE_INTEGER:
        raise StorageRequestError("Invalid storage cursor")
    if stored is not None and stored not in _ORDERS:
        raise StorageRequestError("Invalid storage cursor")
    order: ScanOrder = stored if stored is not None else fallback
    if requested is not None and requested != order:
        raise StorageRequestError(f"The cursor continues a {order} scan; the query asks for {requested}")
    return ScanStart(order, after)


def next_cursor(id: int, order: ScanOrder) -> dict[str, Any]:
    """The continuation of a scan whose last returned item has ``id``: ``{"after": id, "order": order}``."""
    return {"after": id, "order": order}
