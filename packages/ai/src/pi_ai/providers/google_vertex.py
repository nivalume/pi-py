"""Google Vertex AI provider."""

from __future__ import annotations

from ..abort import AbortSignal
from ..api.registry import google_vertex_api
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

__all__ = ["google_vertex_provider", "vertex_auth", "VERTEX_ADC_PATH"]

VERTEX_ADC_PATH = "~/.config/gcloud/application_default_credentials.json"


async def _login(interaction: ProviderAuthInteraction) -> Credential:
    interaction.signal.throw_if_aborted()
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select Google Vertex AI authentication method:",
            options=[
                AuthSelectOption(id="api-key", label="Google Cloud API key"),
                AuthSelectOption(id="adc", label="Application Default Credentials"),
                AuthSelectOption(id="service-account", label="Service account credentials file"),
            ],
        )
    )
    interaction.signal.throw_if_aborted()
    if method == "api-key":
        return {
            "type": "api_key",
            "key": await interaction.prompt(AuthPrompt(type="secret", message="Enter Google Cloud API key")),
        }
    if method not in ("adc", "service-account"):
        raise ValueError(f"Unknown Google Vertex AI auth method: {method}")
    interaction.notify(
        AuthEvent(
            type="info",
            message=(
                "Run `gcloud auth application-default login`, then provide the project and location."
                if method == "adc"
                else "Provide a service account credentials file, project, and location."
            ),
            links=[
                AuthInfoLink(
                    label="Application Default Credentials",
                    url="https://cloud.google.com/docs/authentication/provide-credentials-adc",
                )
            ],
        )
    )
    project = await interaction.prompt(AuthPrompt(type="text", message="Enter Google Cloud project ID"))
    location = await interaction.prompt(AuthPrompt(type="text", message="Enter Google Cloud location"))
    credentials_path = (
        await interaction.prompt(AuthPrompt(type="text", message="Enter service account credentials file path"))
        if method == "service-account"
        else None
    )
    env = {"GOOGLE_CLOUD_PROJECT": project, "GOOGLE_CLOUD_LOCATION": location}
    if credentials_path:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return {"type": "api_key", "env": env}


async def _resolve(*, ctx: AuthContext, credential: Credential | None, signal: AbortSignal) -> AuthResult | None:
    async def env(name: str) -> str | None:
        signal.throw_if_aborted()
        value = await ctx.env(name)
        signal.throw_if_aborted()
        return value

    stored_key = credential.get("key") if credential else None
    key = stored_key if stored_key is not None else await env("GOOGLE_CLOUD_API_KEY")
    if key:
        return AuthResult(ModelAuth(api_key=key), source="stored credential" if stored_key else "GOOGLE_CLOUD_API_KEY")

    credential_env = (credential.get("env") if credential else None) or {}
    adc_path = credential_env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc_path is None:
        adc_path = await env("GOOGLE_APPLICATION_CREDENTIALS")
    signal.throw_if_aborted()
    has_credentials = await ctx.file_exists(adc_path if adc_path is not None else VERTEX_ADC_PATH)
    signal.throw_if_aborted()
    project = credential_env.get("GOOGLE_CLOUD_PROJECT")
    if project is None:
        project = await env("GOOGLE_CLOUD_PROJECT")
    if project is None:
        project = await env("GCLOUD_PROJECT")
    location = credential_env.get("GOOGLE_CLOUD_LOCATION")
    if location is None:
        location = await env("GOOGLE_CLOUD_LOCATION")
    if has_credentials and project and location:
        return AuthResult(
            ModelAuth(),
            env=credential.get("env") if credential else None,
            source="stored credential" if credential else "gcloud application default credentials",
        )
    return None


def vertex_auth() -> ApiKeyAuth:
    """Vertex accepts an explicit API key or Application Default Credentials.

    ADC (``gcloud auth application-default login``) additionally requires project
    and location env vars, which the API implementation reads itself.
    """
    return ApiKeyAuth("Google Cloud credentials", _resolve, login=_login)


def google_vertex_provider() -> Provider:
    """Google Vertex AI with ``GOOGLE_CLOUD_API_KEY`` or Application Default Credentials."""
    return create_provider(
        id="google-vertex",
        name="Google Vertex AI",
        auth=ProviderAuth(api_key=vertex_auth()),
        models=lambda: catalog_models("google-vertex"),
        api=google_vertex_api(),
    )
