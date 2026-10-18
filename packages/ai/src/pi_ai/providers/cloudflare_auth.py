"""Cloudflare Workers AI / AI Gateway auth: API key plus account (and gateway) ids."""

from __future__ import annotations

from dataclasses import dataclass

from ..abort import AbortSignal
from ..auth.types import (
    ApiKeyAuth,
    AuthContext,
    AuthPrompt,
    AuthResult,
    Credential,
    ModelAuth,
    ProviderAuthInteraction,
)

__all__ = ["cloudflare_workers_ai_auth", "cloudflare_ai_gateway_auth"]

CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


@dataclass(slots=True)
class _ResolvedCloudflare:
    api_key: str
    env: dict[str, str]
    source: str


async def _resolve_value(name: str, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> str | None:
    # Per-field merge: prefer the credential value, fall back to ambient env.
    # A credential carrying only the API key must still pick up the account /
    # gateway id from the environment.
    from_credential: str | None = None
    if credential is not None:
        if name == CLOUDFLARE_API_KEY:
            from_credential = credential.get("key")
        else:
            from_credential = (credential.get("env") or {}).get(name)
    if from_credential is not None:
        return from_credential
    signal.throw_if_aborted()
    value = await ctx.env(name)
    signal.throw_if_aborted()
    return value


async def _resolve_cloudflare_env(
    kind: str, ctx: AuthContext, credential: Credential | None, signal: AbortSignal
) -> _ResolvedCloudflare | None:
    api_key = await _resolve_value(CLOUDFLARE_API_KEY, ctx, credential, signal)
    account_id = await _resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential, signal)
    gateway_id = await _resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential, signal) if kind == "ai-gateway" else None

    if not api_key or not account_id or (kind == "ai-gateway" and not gateway_id):
        return None

    env = {CLOUDFLARE_ACCOUNT_ID: account_id}
    if gateway_id:
        env[CLOUDFLARE_GATEWAY_ID] = gateway_id
    return _ResolvedCloudflare(api_key, env, "stored credential" if credential is not None else CLOUDFLARE_API_KEY)


def cloudflare_workers_ai_auth() -> ApiKeyAuth:
    """API key auth that also requires ``CLOUDFLARE_ACCOUNT_ID`` (returned as provider env)."""

    async def login(interaction: ProviderAuthInteraction) -> Credential:
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Cloudflare API key"))
        account_id = await interaction.prompt(AuthPrompt(type="text", message="Enter Cloudflare account ID"))
        return {"type": "api_key", "key": key, "env": {CLOUDFLARE_ACCOUNT_ID: account_id}}

    async def resolve(*, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> AuthResult | None:
        resolved = await _resolve_cloudflare_env("workers-ai", ctx, credential, signal)
        if resolved is None:
            return None
        return AuthResult(ModelAuth(api_key=resolved.api_key), env=resolved.env, source=resolved.source)

    return ApiKeyAuth("Cloudflare API key", resolve, login=login)


def cloudflare_ai_gateway_auth() -> ApiKeyAuth:
    """AI Gateway auth: the key travels as ``cf-aig-authorization``; account and gateway ids as provider env."""

    async def login(interaction: ProviderAuthInteraction) -> Credential:
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Cloudflare API key"))
        account_id = await interaction.prompt(AuthPrompt(type="text", message="Enter Cloudflare account ID"))
        gateway_id = await interaction.prompt(AuthPrompt(type="text", message="Enter Cloudflare AI Gateway ID"))
        return {
            "type": "api_key",
            "key": key,
            "env": {CLOUDFLARE_ACCOUNT_ID: account_id, CLOUDFLARE_GATEWAY_ID: gateway_id},
        }

    async def resolve(*, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> AuthResult | None:
        resolved = await _resolve_cloudflare_env("ai-gateway", ctx, credential, signal)
        if resolved is None:
            return None
        return AuthResult(
            ModelAuth(
                headers={
                    "cf-aig-authorization": f"Bearer {resolved.api_key}",
                    "Authorization": None,
                    "x-api-key": None,
                }
            ),
            env=resolved.env,
            source=resolved.source,
        )

    return ApiKeyAuth("Cloudflare API key", resolve, login=login)
