"""Groq provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["groq_provider"]


def groq_provider() -> Provider:
    """Groq (OpenAI-compatible Chat Completions), authenticated with ``GROQ_API_KEY``."""
    return create_provider(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Groq API key", ["GROQ_API_KEY"])),
        models=lambda: catalog_models("groq"),
        api=openai_completions_api(),
    )
