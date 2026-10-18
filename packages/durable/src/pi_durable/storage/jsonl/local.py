"""JSONL storage on the local file system (replaces upstream ``storage/jsonl/node.ts``)."""

from __future__ import annotations

import os

from pi_chord.context import Context

from ...env.local import LocalExecutionEnv
from .storage import JsonlStorage, JsonlStorageOptions

__all__ = ["JsonlStorage", "JsonlStorageOptions", "open_local_jsonl_storage"]


async def open_local_jsonl_storage(directory: str, context: Context, *, fsync: bool = False) -> JsonlStorage:
    """Open or create a JSONL storage directory using the local file system (upstream ``openNodeJsonlStorage``)."""
    return await JsonlStorage.open(directory, LocalExecutionEnv(cwd=os.getcwd()), context, fsync=fsync)
