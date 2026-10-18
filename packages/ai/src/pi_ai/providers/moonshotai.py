"""Moonshot AI provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["moonshotai_provider"]


def moonshotai_provider() -> Provider:
    """Moonshot AI (OpenAI-compatible Chat Completions), authenticated with ``MOONSHOT_API_KEY``."""
    return create_provider(
        id="moonshotai",
        name="Moonshot AI",
        base_url="https://api.moonshot.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
        models=lambda: catalog_models("moonshotai"),
        api=openai_completions_api(),
    )
