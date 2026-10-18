"""Portable JSONL storage (port of ``storage/jsonl/index.ts``).

:class:`JsonlStorage` keeps its files through any :class:`~pi_durable.env.FileSystem`;
:mod:`pi_durable.storage.jsonl.local` opens one on the local file system (upstream ``storage/jsonl/node.ts``).
"""

from __future__ import annotations

from .storage import JsonlCorruptionError, JsonlStorage, JsonlStorageOptions, JsonlStoragePoisonedError

__all__ = ["JsonlCorruptionError", "JsonlStorage", "JsonlStorageOptions", "JsonlStoragePoisonedError"]
