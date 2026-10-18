"""Cerebras provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["cerebras_provider"]


def cerebras_provider() -> Provider:
    """Cerebras (OpenAI-compatible Chat Completions), authenticated with ``CEREBRAS_API_KEY``."""
    return create_provider(
        id="cerebras",
        name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Cerebras API key", ["CEREBRAS_API_KEY"])),
        models=lambda: catalog_models("cerebras"),
        api=openai_completions_api(),
    )
