"""Mistral provider."""

from __future__ import annotations

from ..api.registry import mistral_conversations_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["mistral_provider"]


def mistral_provider() -> Provider:
    """Mistral (Mistral Conversations API), authenticated with ``MISTRAL_API_KEY``."""
    return create_provider(
        id="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai",
        auth=ProviderAuth(api_key=env_api_key_auth("Mistral API key", ["MISTRAL_API_KEY"])),
        models=lambda: catalog_models("mistral"),
        api=mistral_conversations_api(),
    )
