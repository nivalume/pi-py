"""ID and sequence brands (port of ``ids.ts``).

IDs and sequences are plain ``int`` at runtime, in memory, JSON, JSONL, and SQLite. Brands exist
only for type checkers and are applied at a trusted allocation or decoding boundary.
"""

from __future__ import annotations

from typing import Any, Final

from .types import Seq

__all__ = ["MAX_SAFE_INTEGER", "id_from_number", "seq_from_number"]

#: ``Number.MAX_SAFE_INTEGER``: the largest ID or sequence TypeScript pi-durable can represent exactly.
MAX_SAFE_INTEGER: Final = 2**53 - 1


def id_from_number(value: int) -> Any:
    """Apply an erased ID brand at a trusted numeric allocation or decoding boundary.

    Returns ``value`` unchanged, typed ``Any`` so it can be assigned to any ID type
    (``entry_id: EntryId = id_from_number(2)``), like upstream's generic ``idFromNumber<I>()``.
    """
    return value


def seq_from_number(value: int) -> Seq:
    """Apply the erased commit-sequence brand at a trusted storage boundary."""
    return Seq(value)
