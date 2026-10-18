"""Fireworks provider."""

from __future__ import annotations

from ..api.registry import anthropic_messages_api, openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["fireworks_provider"]


def fireworks_provider() -> Provider:
    """Fireworks (Anthropic Messages and Chat Completions), authenticated with ``FIREWORKS_API_KEY``."""
    return create_provider(
        id="fireworks",
        name="Fireworks",
        base_url="https://api.fireworks.ai/inference",
        auth=ProviderAuth(api_key=env_api_key_auth("Fireworks API key", ["FIREWORKS_API_KEY"])),
        models=lambda: catalog_models("fireworks"),
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
        },
    )
