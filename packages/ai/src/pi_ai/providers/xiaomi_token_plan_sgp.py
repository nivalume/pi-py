"""Xiaomi Token Plan SGP provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["xiaomi_token_plan_sgp_provider"]


def xiaomi_token_plan_sgp_provider() -> Provider:
    """Xiaomi Token Plan SGP (OpenAI-compatible Chat Completions), authenticated with ``XIAOMI_TOKEN_PLAN_SGP_API_KEY``."""
    return create_provider(
        id="xiaomi-token-plan-sgp",
        name="Xiaomi Token Plan SGP",
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan SGP API key", ["XIAOMI_TOKEN_PLAN_SGP_API_KEY"])),
        models=lambda: catalog_models("xiaomi-token-plan-sgp"),
        api=openai_completions_api(),
    )
