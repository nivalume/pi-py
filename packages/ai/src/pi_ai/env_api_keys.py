"""Known API-key environment variables per built-in provider."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .utils.misc import get_provider_env_value

__all__ = [
    "ANTHROPIC_AUTH_TOKEN_ENV",
    "ANTHROPIC_OAUTH_TOKEN_ENV",
    "ANTHROPIC_API_KEY_ENV",
    "ANTHROPIC_FEDERATION_RULE_ID_ENV",
    "ANTHROPIC_ORGANIZATION_ID_ENV",
    "ANTHROPIC_SERVICE_ACCOUNT_ID_ENV",
    "ANTHROPIC_IDENTITY_TOKEN_FILE_ENV",
    "ANTHROPIC_WORKSPACE_ID_ENV",
    "PROVIDER_API_KEY_ENV_VARS",
    "find_env_keys",
    "get_env_api_key",
]

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_FEDERATION_RULE_ID_ENV = "ANTHROPIC_FEDERATION_RULE_ID"
ANTHROPIC_ORGANIZATION_ID_ENV = "ANTHROPIC_ORGANIZATION_ID"
ANTHROPIC_SERVICE_ACCOUNT_ID_ENV = "ANTHROPIC_SERVICE_ACCOUNT_ID"
ANTHROPIC_IDENTITY_TOKEN_FILE_ENV = "ANTHROPIC_IDENTITY_TOKEN_FILE"
ANTHROPIC_WORKSPACE_ID_ENV = "ANTHROPIC_WORKSPACE_ID"

PROVIDER_API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "github-copilot": ("COPILOT_GITHUB_TOKEN",),
    "anthropic": (ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV),
    "ant-ling": ("ANT_LING_API_KEY",),
    "qwen-token-plan": ("QWEN_TOKEN_PLAN_API_KEY",),
    "qwen-token-plan-cn": ("QWEN_TOKEN_PLAN_CN_API_KEY",),
    "qwen-token-plan-individual": ("QWEN_TOKEN_PLAN_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "google-vertex": ("GOOGLE_CLOUD_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "typesafe": ("TYPESAFE_API_KEY",),
    "radius": ("RADIUS_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "vercel-ai-gateway": ("AI_GATEWAY_API_KEY",),
    "zai": ("ZAI_API_KEY",),
    "zai-coding-cn": ("ZAI_CODING_CN_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "minimax-cn": ("MINIMAX_CN_API_KEY",),
    "moonshotai": ("MOONSHOT_API_KEY",),
    "moonshotai-cn": ("MOONSHOT_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "baseten": ("BASETEN_API_KEY",),
    "opencode": ("OPENCODE_API_KEY",),
    "opencode-go": ("OPENCODE_API_KEY",),
    "kimi-coding": ("KIMI_API_KEY",),
    "meta": ("META_API_KEY",),
    "cloudflare-workers-ai": ("CLOUDFLARE_API_KEY",),
    "cloudflare-ai-gateway": ("CLOUDFLARE_API_KEY",),
    "xiaomi": ("XIAOMI_API_KEY",),
    "xiaomi-token-plan-cn": ("XIAOMI_TOKEN_PLAN_CN_API_KEY",),
    "xiaomi-token-plan-ams": ("XIAOMI_TOKEN_PLAN_AMS_API_KEY",),
    "xiaomi-token-plan-sgp": ("XIAOMI_TOKEN_PLAN_SGP_API_KEY",),
}

_vertex_adc_exists: bool | None = None


def _has_vertex_adc_credentials(env: Mapping[str, str] | None) -> bool:
    global _vertex_adc_exists
    explicit = env.get("GOOGLE_APPLICATION_CREDENTIALS") if env else None
    if explicit:
        return os.path.exists(os.path.expanduser(explicit))
    if _vertex_adc_exists is None:
        path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
        if path:
            _vertex_adc_exists = os.path.exists(os.path.expanduser(path))
        else:
            _vertex_adc_exists = os.path.exists(
                os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")
            )
    return _vertex_adc_exists


def find_env_keys(provider: str, env: Mapping[str, str] | None = None) -> list[str] | None:
    """Configured env vars that provide an API key for ``provider`` (ambient sources excluded)."""
    names = PROVIDER_API_KEY_ENV_VARS.get(provider)
    if not names:
        return None
    found = [name for name in names if get_provider_env_value(name, env)]
    return found or None


def get_env_api_key(provider: str, env: Mapping[str, str] | None = None) -> str | None:
    """API key for ``provider`` from known env vars (``"<authenticated>"`` for ambient Vertex/Bedrock auth)."""
    found = find_env_keys(provider, env)
    if found:
        name = (
            next((key for key in found if key != ANTHROPIC_AUTH_TOKEN_ENV), None)
            if provider == "anthropic"
            else found[0]
        )
        if name:
            return get_provider_env_value(name, env)

    if provider == "google-vertex":
        has_project = bool(
            get_provider_env_value("GOOGLE_CLOUD_PROJECT", env) or get_provider_env_value("GCLOUD_PROJECT", env)
        )
        has_location = bool(get_provider_env_value("GOOGLE_CLOUD_LOCATION", env))
        if _has_vertex_adc_credentials(env) and has_project and has_location:
            return "<authenticated>"

    if provider == "amazon-bedrock":
        if (
            get_provider_env_value("AWS_PROFILE", env)
            or (
                get_provider_env_value("AWS_ACCESS_KEY_ID", env)
                and get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
            )
            or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env)
            or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env)
            or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env)
            or get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env)
        ):
            return "<authenticated>"
    return None
