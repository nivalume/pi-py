"""Hugging Face provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["huggingface_provider"]


def huggingface_provider() -> Provider:
    """Hugging Face (OpenAI-compatible Chat Completions), authenticated with ``HF_TOKEN``."""
    return create_provider(
        id="huggingface",
        name="Hugging Face",
        base_url="https://router.huggingface.co/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Hugging Face token", ["HF_TOKEN"])),
        models=lambda: catalog_models("huggingface"),
        api=openai_completions_api(),
    )
