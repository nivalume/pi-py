"""Default auth context backed by ``os.environ`` and the local filesystem."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .types import AuthContext

__all__ = ["DefaultAuthContext", "default_provider_auth_context", "overlay_env_auth_context"]


class DefaultAuthContext:
    """Env vars from ``os.environ`` (blank values count as unset); file checks via ``os.path``."""

    async def env(self, name: str) -> str | None:
        value = os.environ.get(name)
        return value if value and value.strip() else None

    async def file_exists(self, path: str) -> bool:
        return os.path.exists(os.path.expanduser(path))


def default_provider_auth_context() -> AuthContext:
    return DefaultAuthContext()


class _OverlayAuthContext:
    __slots__ = ("_base", "_env")

    def __init__(self, base: AuthContext, env: Mapping[str, str]) -> None:
        self._base = base
        self._env = env

    async def env(self, name: str) -> str | None:
        value = self._env.get(name)
        if value:
            return value
        return await self._base.env(name)

    async def file_exists(self, path: str) -> bool:
        return await self._base.file_exists(path)


def overlay_env_auth_context(base: AuthContext, env: Mapping[str, str]) -> AuthContext:
    """An auth context where ``env`` values take precedence over ``base``."""
    return _OverlayAuthContext(base, env)
