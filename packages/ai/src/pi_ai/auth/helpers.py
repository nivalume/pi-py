"""Standard auth building blocks for provider definitions."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from ..abort import AbortSignal
from .types import (
    ApiKeyAuth,
    AuthContext,
    AuthPrompt,
    AuthResult,
    Credential,
    LoginOptions,
    ModelAuth,
    OAuthAuth,
    ProviderAuthInteraction,
)

__all__ = ["env_api_key_auth", "keyless_auth", "lazy_oauth", "LazyOAuthAuth"]


def env_api_key_auth(name: str, env_vars: Sequence[str]) -> ApiKeyAuth:
    """Standard API-key auth: a stored credential key wins, otherwise the first set env var.

    Includes a ``login`` that prompts for the key.
    """
    env_names = tuple(env_vars)

    async def login(interaction: ProviderAuthInteraction) -> Credential:
        interaction.signal.throw_if_aborted()
        key = await interaction.prompt(AuthPrompt(type="secret", message=f"Enter {name}"))
        interaction.signal.throw_if_aborted()
        return {"type": "api_key", "key": key}

    async def resolve(*, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> AuthResult | None:
        signal.throw_if_aborted()
        if credential and credential.get("key"):
            return AuthResult(
                ModelAuth(api_key=credential["key"]), env=credential.get("env"), source="stored credential"
            )
        for env_var in env_names:
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if value:
                return AuthResult(ModelAuth(api_key=value), source=env_var)
        return None

    return ApiKeyAuth(name, resolve, login=login)


def keyless_auth(name: str) -> ApiKeyAuth:
    """Auth for keyless local servers (Ollama, llama.cpp, vLLM): always configured, no key."""

    async def resolve(*, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> AuthResult | None:
        if credential and credential.get("key"):
            return AuthResult(
                ModelAuth(api_key=credential["key"]), env=credential.get("env"), source="stored credential"
            )
        return AuthResult(ModelAuth())

    return ApiKeyAuth(name, resolve)


class LazyOAuthAuth(OAuthAuth):
    """An :class:`OAuthAuth` whose implementation loads on first use."""

    def __init__(
        self,
        name: str,
        load: Callable[[], OAuthAuth | Awaitable[OAuthAuth]] | str,
        *,
        is_subscription: bool = False,
        login_label: str | None = None,
    ) -> None:
        self.name = name
        self.is_subscription = is_subscription
        self.login_label = login_label
        self._load = load
        self._loaded: OAuthAuth | None = None
        self._lock: asyncio.Lock | None = None

    async def _get(self) -> OAuthAuth:
        if self._loaded is not None:
            return self._loaded
        load = self._load
        if isinstance(load, str):
            module_name, _, attr = load.partition(":")
            module = importlib.import_module(module_name)
            factory: Any = getattr(module, attr or "create_oauth")
            result = factory()
        else:
            result = load()
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            result = await result
        self._loaded = result
        return result

    async def login(self, interaction: ProviderAuthInteraction, options: LoginOptions | None = None) -> Credential:
        return await (await self._get()).login(interaction, options)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return await (await self._get()).refresh(credential, signal)

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return await (await self._get()).to_auth(credential)


def lazy_oauth(
    name: str,
    load: Callable[[], OAuthAuth | Awaitable[OAuthAuth]] | str,
    *,
    is_subscription: bool = False,
    login_label: str | None = None,
) -> OAuthAuth:
    """Advertise OAuth without importing the flow implementation.

    ``load`` is a callable returning the implementation, or ``"module:factory"``.
    """
    return LazyOAuthAuth(name, load, is_subscription=is_subscription, login_label=login_label)
