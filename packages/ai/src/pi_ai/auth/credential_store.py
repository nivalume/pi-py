"""Credential stores: in-memory (default) and ``auth.json`` file-backed."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ..abort import AbortSignal, race_with_abort_signal
from .types import Credential, CredentialInfo

__all__ = ["InMemoryCredentialStore", "FileCredentialStore"]

try:  # POSIX advisory locks for cross-process serialization
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]


class _ProviderLocks:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, provider_id: str) -> asyncio.Lock:
        lock = self._locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider_id] = lock
        return lock


async def _enqueue(
    locks: _ProviderLocks,
    provider_id: str,
    task: Callable[[], Awaitable[Any]],
    signal: AbortSignal | None,
) -> Any:
    """Serialize ``task`` per provider. ``signal`` stops the caller waiting; started work completes."""
    lock = locks.get(provider_id)

    async def run() -> Any:
        async with lock:
            if signal is not None:
                signal.throw_if_aborted()
            return await task()

    return await race_with_abort_signal(run(), signal)


class InMemoryCredentialStore:
    """Default in-memory credential store; writes are serialized per provider id."""

    def __init__(self, initial: dict[str, Credential] | None = None) -> None:
        self._credentials: dict[str, Credential] = copy.deepcopy(initial) if initial else {}
        self._locks = _ProviderLocks()

    async def read(self, provider_id: str, *, signal: AbortSignal | None = None) -> Credential | None:
        if signal is not None:
            signal.throw_if_aborted()
        credential = self._credentials.get(provider_id)
        return copy.deepcopy(credential) if credential is not None else None

    async def list(self, *, signal: AbortSignal | None = None) -> Sequence[CredentialInfo]:
        if signal is not None:
            signal.throw_if_aborted()
        return [
            CredentialInfo(provider_id, credential.get("type", "api_key"))
            for provider_id, credential in self._credentials.items()
        ]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        *,
        signal: AbortSignal | None = None,
    ) -> Credential | None:
        async def task() -> Credential | None:
            current = self._credentials.get(provider_id)
            following = await fn(copy.deepcopy(current) if current is not None else None)
            if signal is not None:
                signal.throw_if_aborted()
            if following is not None:
                self._credentials[provider_id] = copy.deepcopy(following)
                return following
            return current

        return await _enqueue(self._locks, provider_id, task, signal)

    async def delete(self, provider_id: str, *, signal: AbortSignal | None = None) -> None:
        async def task() -> None:
            self._credentials.pop(provider_id, None)

        await _enqueue(self._locks, provider_id, task, signal)


class FileCredentialStore:
    """Credentials in an ``auth.json`` file shared with TypeScript pi.

    The file maps provider ids to credentials (``{"anthropic": {"type": "oauth", ...}}``)
    and is written atomically with ``0600`` permissions. On POSIX, ``modify`` also takes
    an advisory file lock so concurrent processes cannot double-refresh OAuth tokens.
    """

    def __init__(self, path: str | os.PathLike[str] = "auth.json") -> None:
        self.path = Path(path).expanduser()
        self._locks = _ProviderLocks()

    def _load(self) -> dict[str, Credential]:
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"{self.path} does not contain a JSON object")
        return data

    def _save(self, data: dict[str, Credential]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".auth-", suffix=".json")
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def _file_lock(self) -> Iterator[None]:
        if _fcntl is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with open(lock_path, "a+") as handle:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)

    async def read(self, provider_id: str, *, signal: AbortSignal | None = None) -> Credential | None:
        if signal is not None:
            signal.throw_if_aborted()
        data = await asyncio.to_thread(self._load)
        credential = data.get(provider_id)
        return credential if isinstance(credential, dict) else None

    async def list(self, *, signal: AbortSignal | None = None) -> Sequence[CredentialInfo]:
        if signal is not None:
            signal.throw_if_aborted()
        data = await asyncio.to_thread(self._load)
        return [
            CredentialInfo(provider_id, credential.get("type", "api_key"))
            for provider_id, credential in data.items()
            if isinstance(credential, dict)
        ]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        *,
        signal: AbortSignal | None = None,
    ) -> Credential | None:
        loop = asyncio.get_running_loop()

        async def task() -> Credential | None:
            lock = self._file_lock()
            await asyncio.to_thread(lock.__enter__)
            try:
                data = await asyncio.to_thread(self._load)
                current = data.get(provider_id)
                following = await fn(copy.deepcopy(current) if isinstance(current, dict) else None)
                if signal is not None:
                    signal.throw_if_aborted()
                if following is None:
                    return current if isinstance(current, dict) else None
                data[provider_id] = following
                await asyncio.to_thread(self._save, data)
                return following
            finally:
                await loop.run_in_executor(None, lock.__exit__, None, None, None)

        return await _enqueue(self._locks, provider_id, task, signal)

    async def delete(self, provider_id: str, *, signal: AbortSignal | None = None) -> None:
        async def task() -> None:
            lock = self._file_lock()
            await asyncio.to_thread(lock.__enter__)
            try:
                data = await asyncio.to_thread(self._load)
                if data.pop(provider_id, None) is not None:
                    await asyncio.to_thread(self._save, data)
            finally:
                await asyncio.to_thread(lock.__exit__, None, None, None)

        await _enqueue(self._locks, provider_id, task, signal)
