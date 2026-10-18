"""Amazon Bedrock provider."""

from __future__ import annotations

from ..abort import AbortSignal
from ..api.registry import bedrock_converse_stream_api
from ..auth.types import (
    ApiKeyAuth,
    AuthContext,
    AuthEvent,
    AuthInfoLink,
    AuthPrompt,
    AuthResult,
    AuthSelectOption,
    Credential,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["amazon_bedrock_provider", "bedrock_auth"]


async def _login(interaction: ProviderAuthInteraction) -> Credential:
    interaction.signal.throw_if_aborted()
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select Amazon Bedrock authentication method:",
            options=[
                AuthSelectOption(id="bearer-token", label="Bearer token"),
                AuthSelectOption(id="aws-profile", label="AWS profile"),
                AuthSelectOption(id="credential-chain", label="Existing AWS credential chain"),
            ],
        )
    )
    interaction.signal.throw_if_aborted()
    if method == "bearer-token":
        return {
            "type": "api_key",
            "key": await interaction.prompt(AuthPrompt(type="secret", message="Enter Amazon Bedrock bearer token")),
        }
    interaction.notify(
        AuthEvent(
            type="info",
            message="Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials.",
            links=[
                AuthInfoLink(
                    label="AWS credential provider chain",
                    url="https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html",
                )
            ],
        )
    )
    if method == "aws-profile":
        return {
            "type": "api_key",
            "env": {"AWS_PROFILE": await interaction.prompt(AuthPrompt(type="text", message="Enter AWS profile name"))},
        }
    if method != "credential-chain":
        raise ValueError(f"Unknown Amazon Bedrock auth method: {method}")
    await interaction.prompt(AuthPrompt(type="text", message="Configure AWS credentials, then press Enter to continue"))
    return {"type": "api_key"}


async def _resolve(*, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> AuthResult | None:
    async def env(name: str) -> str | None:
        signal.throw_if_aborted()
        value = await ctx.env(name)
        signal.throw_if_aborted()
        return value

    if credential and credential.get("key"):
        return AuthResult(ModelAuth(api_key=credential["key"]), env=credential.get("env"), source="stored credential")
    if await env("AWS_BEARER_TOKEN_BEDROCK"):
        return AuthResult(ModelAuth(), source="AWS_BEARER_TOKEN_BEDROCK")
    credential_env = credential.get("env") if credential else None
    stored_profile = credential_env.get("AWS_PROFILE") if credential_env else None
    profile = stored_profile if stored_profile is not None else await env("AWS_PROFILE")
    if profile:
        return AuthResult(
            ModelAuth(),
            env=credential_env,
            source="stored credential" if stored_profile else "AWS_PROFILE",
        )
    if (await env("AWS_ACCESS_KEY_ID")) and (await env("AWS_SECRET_ACCESS_KEY")):
        return AuthResult(ModelAuth(), source="AWS access keys")
    if await env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
        return AuthResult(ModelAuth(), source="ECS task role")
    if await env("AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        return AuthResult(ModelAuth(), source="ECS task role")
    if await env("AWS_WEB_IDENTITY_TOKEN_FILE"):
        return AuthResult(ModelAuth(), source="web identity token")
    return None


def bedrock_auth() -> ApiKeyAuth:
    """Bedrock accepts a bearer token or the AWS default credential chain.

    The login flow can store a token/profile choice; ``resolve`` also detects ambient
    AWS credentials without copying them into pi's credential store.
    """
    return ApiKeyAuth("AWS credentials or bearer token", _resolve, login=_login)


def amazon_bedrock_provider() -> Provider:
    """Amazon Bedrock (Converse Stream API) with bearer-token or AWS credential-chain auth."""
    return create_provider(
        id="amazon-bedrock",
        name="Amazon Bedrock",
        auth=ProviderAuth(api_key=bedrock_auth()),
        models=lambda: catalog_models("amazon-bedrock"),
        api=bedrock_converse_stream_api(),
    )
