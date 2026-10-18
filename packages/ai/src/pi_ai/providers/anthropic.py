"""Anthropic provider."""

from __future__ import annotations

from ..abort import AbortSignal
from ..api.registry import anthropic_messages_api
from ..auth.helpers import lazy_oauth
from ..auth.oauth.load import load_anthropic_oauth
from ..auth.types import (
    ApiKeyAuth,
    AuthContext,
    AuthPrompt,
    AuthResult,
    Credential,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from ..env_api_keys import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_FEDERATION_RULE_ID_ENV,
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
    ANTHROPIC_OAUTH_TOKEN_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV,
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV,
    ANTHROPIC_WORKSPACE_ID_ENV,
)
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["anthropic_provider", "anthropic_api_key_auth"]


def anthropic_api_key_auth() -> ApiKeyAuth:
    """Anthropic API-key auth.

    Order: stored key, ``ANTHROPIC_AUTH_TOKEN`` (sent as ``Authorization: Bearer``),
    ``ANTHROPIC_OAUTH_TOKEN``, ``ANTHROPIC_API_KEY``, then workload identity federation.
    """

    async def login(interaction: ProviderAuthInteraction) -> Credential:
        interaction.signal.throw_if_aborted()
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Anthropic API key"))
        interaction.signal.throw_if_aborted()
        return {"type": "api_key", "key": key}

    async def resolve(*, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> AuthResult | None:
        signal.throw_if_aborted()
        if credential and credential.get("key"):
            return AuthResult(
                ModelAuth(api_key=credential["key"]), env=credential.get("env"), source="stored credential"
            )

        auth_token = await ctx.env(ANTHROPIC_AUTH_TOKEN_ENV)
        signal.throw_if_aborted()
        if auth_token:
            return AuthResult(
                ModelAuth(headers={"Authorization": f"Bearer {auth_token}"}), source=ANTHROPIC_AUTH_TOKEN_ENV
            )

        for env_var in (ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV):
            api_key = await ctx.env(env_var)
            signal.throw_if_aborted()
            if api_key:
                return AuthResult(ModelAuth(api_key=api_key), source=env_var)

        # Workload identity federation: the API implementation exchanges the identity
        # token for a short-lived access token and refreshes it itself. Last in
        # line so keys and ANTHROPIC_AUTH_TOKEN keep winning, as in the SDK. The
        # ids are provider config rather than auth, so they travel in `env`.
        federation: dict[str, str] = {}
        for env_var in (
            ANTHROPIC_FEDERATION_RULE_ID_ENV,
            ANTHROPIC_ORGANIZATION_ID_ENV,
            ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
        ):
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if not value:
                return None
            federation[env_var] = value
        for env_var in (ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV):
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if value:
                federation[env_var] = value
        return AuthResult(ModelAuth(), env=federation, source="workload identity federation")

    return ApiKeyAuth("Anthropic API key", resolve, login=login)


def anthropic_provider() -> Provider:
    """Anthropic (Messages API): API key, ``ANTHROPIC_AUTH_TOKEN``, federation, or Claude Pro/Max OAuth."""
    return create_provider(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com",
        auth=ProviderAuth(
            api_key=anthropic_api_key_auth(),
            oauth=lazy_oauth("Anthropic (Claude Pro/Max)", load_anthropic_oauth, is_subscription=True),
        ),
        models=lambda: catalog_models("anthropic"),
        api=anthropic_messages_api(),
    )
