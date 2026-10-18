"""Baseten provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["baseten_provider"]


def baseten_provider() -> Provider:
    """Baseten (OpenAI-compatible Chat Completions), authenticated with ``BASETEN_API_KEY``."""
    return create_provider(
        id="baseten",
        name="Baseten",
        base_url="https://inference.baseten.co/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Baseten API key", ["BASETEN_API_KEY"])),
        models=lambda: catalog_models("baseten"),
        api=openai_completions_api(),
    )
