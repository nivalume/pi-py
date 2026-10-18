"""Persistent storage for dynamic model catalogs."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .abort import AbortSignal

__all__ = ["ModelsStoreEntry", "ModelsStore", "InMemoryModelsStore", "FileModelsStore"]


@dataclass(slots=True)
class ModelsStoreEntry:
    #: Persisted models of every type.
    models: list[Any] = field(default_factory=list)
    #: Unix timestamp (ms) from the remote catalog's Last-Modified header.
    last_modified: int | None = None
    #: Unix timestamp (ms) of the last completed remote check.
    checked_at: int | None = None
    #: Opaque validator from the remote catalog's ETag header, echoed back as If-None-Match.
    etag: str | None = None


class ModelsStore(Protocol):
    """Persistent model catalogs keyed by provider id."""

    async def read(self, provider_id: str, *, signal: AbortSignal | None = None) -> ModelsStoreEntry | None: ...

    async def write(self, provider_id: str, entry: ModelsStoreEntry, *, signal: AbortSignal | None = None) -> None: ...

    async def delete(self, provider_id: str, *, signal: AbortSignal | None = None) -> None: ...


class InMemoryModelsStore:
    """Default in-process store. Entries are deep-copied on read and write."""

    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(self, provider_id: str, *, signal: AbortSignal | None = None) -> ModelsStoreEntry | None:
        if signal is not None:
            signal.throw_if_aborted()
        entry = self._entries.get(provider_id)
        return copy.deepcopy(entry) if entry is not None else None

    async def write(self, provider_id: str, entry: ModelsStoreEntry, *, signal: AbortSignal | None = None) -> None:
        if signal is not None:
            signal.throw_if_aborted()
        self._entries[provider_id] = copy.deepcopy(entry)

    async def delete(self, provider_id: str, *, signal: AbortSignal | None = None) -> None:
        if signal is not None:
            signal.throw_if_aborted()
        self._entries.pop(provider_id, None)


class FileModelsStore:
    """JSON-file store (``{"<provider>": {"models": [...], "checkedAt": ...}}``) in pi's format."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser()
        self._lock = asyncio.Lock()

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".models-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    async def read(self, provider_id: str, *, signal: AbortSignal | None = None) -> ModelsStoreEntry | None:
        from .serialization import model_from_dict

        if signal is not None:
            signal.throw_if_aborted()
        raw = (await asyncio.to_thread(self._load)).get(provider_id)
        if not isinstance(raw, dict):
            return None
        models = []
        for item in raw.get("models", []):
            try:
                models.append(model_from_dict(item))
            except ValueError:
                continue  # unknown model types from newer versions
        return ModelsStoreEntry(
            models=models,
            last_modified=raw.get("lastModified"),
            checked_at=raw.get("checkedAt"),
            etag=raw.get("etag"),
        )

    async def write(self, provider_id: str, entry: ModelsStoreEntry, *, signal: AbortSignal | None = None) -> None:
        from .serialization import to_json_value

        if signal is not None:
            signal.throw_if_aborted()
        async with self._lock:
            data = await asyncio.to_thread(self._load)
            data[provider_id] = to_json_value(entry)
            await asyncio.to_thread(self._save, data)

    async def delete(self, provider_id: str, *, signal: AbortSignal | None = None) -> None:
        if signal is not None:
            signal.throw_if_aborted()
        async with self._lock:
            data = await asyncio.to_thread(self._load)
            if data.pop(provider_id, None) is not None:
                await asyncio.to_thread(self._save, data)
