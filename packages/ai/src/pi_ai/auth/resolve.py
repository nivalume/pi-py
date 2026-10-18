"""Auth resolution shared by every operation of a ``Models`` collection.

A stored credential owns its provider: ambient sources (env vars, profiles) are
consulted only when nothing is stored, and a failed OAuth refresh never silently
falls back to an env key.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..abort import AbortController, AbortError, AbortSignal, operation_signal, race_with_abort_signal
from ..utils.models_error import ModelsError
from .context import overlay_env_auth_context
from .types import ApiKeyAuth, AuthContext, AuthResult, Credential, CredentialStore, OAuthAuth, ProviderAuth

__all__ = [
    "AuthResolutionOverrides",
    "resolve_provider_auth",
    "refresh_stored_oauth_credential",
    "DEFAULT_OAUTH_MINIMUM_VALIDITY_MS",
    "DEFAULT_OAUTH_REFRESH_TIMEOUT_MS",
]

DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000
DEFAULT_OAUTH_REFRESH_TIMEOUT_MS = 15_000


@dataclass(slots=True)
class AuthResolutionOverrides:
    api_key: str | None = None
    env: dict[str, str] | None = None
    #: Require this much remaining OAuth-token validity; defaults to five minutes.
    min_oauth_validity_ms: int | None = None
    signal: AbortSignal | None = None


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


async def resolve_provider_auth(
    provider: Any,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None = None,
) -> AuthResult | None:
    signal = operation_signal(overrides.signal if overrides else None)
    return await race_with_abort_signal(
        _resolve(provider, credentials, auth_context, overrides or AuthResolutionOverrides(), signal), signal
    )


async def _resolve(
    provider: Any,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides,
    signal: AbortSignal,
) -> AuthResult | None:
    signal.throw_if_aborted()
    auth: ProviderAuth = provider.auth
    request_context = overlay_env_auth_context(auth_context, overrides.env) if overrides.env else auth_context

    if overrides.api_key is not None and auth.api_key is not None:
        credential: Credential = {"type": "api_key", "key": overrides.api_key}
        if overrides.env:
            credential["env"] = dict(overrides.env)
        return await _resolve_api_key(request_context, auth.api_key, provider.id, credential, signal)

    stored = await _read_credential(credentials, provider.id, signal)
    if stored:
        if stored.get("type") == "oauth" and auth.oauth is not None:
            return await _resolve_stored_oauth(
                credentials, provider.id, auth.oauth, stored, signal, overrides.min_oauth_validity_ms
            )
        if stored.get("type") == "api_key" and auth.api_key is not None:
            if overrides.env:
                stored = {**stored, "env": {**(stored.get("env") or {}), **overrides.env}}
            return await _resolve_api_key(request_context, auth.api_key, provider.id, stored, signal)
        return None

    if auth.api_key is not None:
        return await _resolve_api_key(request_context, auth.api_key, provider.id, None, signal)
    return None


async def refresh_stored_oauth_credential(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    needs_refresh: Callable[[Credential], bool],
    signal: AbortSignal,
) -> Credential | None:
    """Refresh a stored OAuth credential under the credential-store lock and persist it.

    ``needs_refresh`` is re-checked under the lock, so concurrent callers refresh once.
    ``signal`` cancels only the wait for the lock: once a refresh starts the provider
    may already have rotated the refresh token, so the refresh and its persistence
    are bounded only by a timeout.
    """
    lock_wait = AbortController()
    remove = signal.add_listener(lambda: lock_wait.abort(signal.reason))

    async def refresh(current: Credential | None) -> Credential | None:
        remove()
        signal.throw_if_aborted()
        if not current or current.get("type") != "oauth":
            return None  # logged out meanwhile
        if not needs_refresh(current):
            return None  # another request/process refreshed
        refresh_signal = AbortSignal.timeout(DEFAULT_OAUTH_REFRESH_TIMEOUT_MS / 1000)
        try:
            refreshed = await asyncio.wait_for(
                oauth.refresh(current, refresh_signal), DEFAULT_OAUTH_REFRESH_TIMEOUT_MS / 1000 + 1
            )
        except Exception as error:
            raise ModelsError("oauth", f"OAuth refresh failed for {provider_id}", cause=error) from error
        result = dict(refreshed)
        result.setdefault("type", "oauth")
        return result

    try:
        post = await credentials.modify(provider_id, refresh, signal=lock_wait.signal)
    except ModelsError:
        raise
    except AbortError:
        signal.throw_if_aborted()
        raise
    except Exception as error:
        signal.throw_if_aborted()
        raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error) from error
    finally:
        remove()
    return post if post and post.get("type") == "oauth" else None


async def _resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: Credential,
    signal: AbortSignal,
    min_oauth_validity_ms: int | None,
) -> AuthResult | None:
    minimum = max(DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, min_oauth_validity_ms or 0)

    def expires_soon(credential: Credential) -> bool:
        return _now_ms() + minimum >= int(credential.get("expires") or 0)

    credential = stored
    if expires_soon(credential):
        post = await refresh_stored_oauth_credential(credentials, provider_id, oauth, expires_soon, signal)
        if post is None:
            return None
        credential = post
        if min_oauth_validity_ms is not None and expires_soon(credential):
            raise ModelsError("oauth", f"OAuth refresh returned a token that expires too soon for {provider_id}")
    try:
        return AuthResult(await oauth.to_auth(credential), source="OAuth")
    except Exception as error:
        raise ModelsError("oauth", f"OAuth auth derivation failed for {provider_id}", cause=error) from error


async def _resolve_api_key(
    auth_context: AuthContext,
    api_key: ApiKeyAuth,
    provider_id: str,
    credential: Credential | None,
    signal: AbortSignal,
) -> AuthResult | None:
    try:
        return await api_key.resolve(ctx=auth_context, credential=credential, signal=signal)
    except AbortError:
        raise
    except Exception as error:
        raise ModelsError("auth", f"API key auth failed for provider {provider_id}", cause=error) from error


async def _read_credential(credentials: CredentialStore, provider_id: str, signal: AbortSignal) -> Credential | None:
    try:
        return await credentials.read(provider_id, signal=signal)
    except AbortError:
        raise
    except Exception as error:
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error) from error
