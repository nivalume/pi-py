"""Xiaomi provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["xiaomi_provider"]


def xiaomi_provider() -> Provider:
    """Xiaomi (OpenAI-compatible Chat Completions), authenticated with ``XIAOMI_API_KEY``."""
    return create_provider(
        id="xiaomi",
        name="Xiaomi",
        base_url="https://api.xiaomimimo.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi API key", ["XIAOMI_API_KEY"])),
        models=lambda: catalog_models("xiaomi"),
        api=openai_completions_api(),
    )
