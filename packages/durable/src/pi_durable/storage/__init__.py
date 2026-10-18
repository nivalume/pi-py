"""Storage backends and their shared helpers (port of ``storage/``).

:class:`MemoryStorage` is the reference implementation; SQLite and JSONL backends live in
``pi_durable.storage.sqlite`` and ``pi_durable.storage.jsonl``. :mod:`pi_durable.storage.codec`
encodes records for backends that serialize them.
"""

from __future__ import annotations

from .memory import MemoryStorage, PreparedMemoryCommit
from .scan import ScanStart, next_cursor, scan_start

__all__ = ["MemoryStorage", "PreparedMemoryCommit", "ScanStart", "next_cursor", "scan_start"]
