"""SQLite identifier validation (port of chord ``sqlite/schema.ts``)."""

from __future__ import annotations

import json
import re

__all__ = ["validate_sqlite_schema_prefix"]

_PREFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def validate_sqlite_schema_prefix(prefix: str) -> None:
    """Validate a prefix before using it as part of a SQLite identifier.

    Raises :class:`ValueError` unless ``prefix`` is an ASCII identifier that does not start with
    SQLite's reserved ``sqlite_`` (in any case).
    """
    if not isinstance(prefix, str) or _PREFIX.fullmatch(prefix) is None:
        raise ValueError(f"Invalid SQLite schema prefix {json.dumps(prefix, ensure_ascii=False, default=repr)}")
    if prefix.lower().startswith("sqlite_"):
        raise ValueError(f"SQLite schema prefix {json.dumps(prefix, ensure_ascii=False)} is reserved")
