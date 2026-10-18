"""Xiaomi Token Plan AMS provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["xiaomi_token_plan_ams_provider"]


def xiaomi_token_plan_ams_provider() -> Provider:
    """Xiaomi Token Plan AMS (OpenAI-compatible Chat Completions), authenticated with ``XIAOMI_TOKEN_PLAN_AMS_API_KEY``."""
    return create_provider(
        id="xiaomi-token-plan-ams",
        name="Xiaomi Token Plan AMS",
        base_url="https://token-plan-ams.xiaomimimo.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan AMS API key", ["XIAOMI_TOKEN_PLAN_AMS_API_KEY"])),
        models=lambda: catalog_models("xiaomi-token-plan-ams"),
        api=openai_completions_api(),
    )
