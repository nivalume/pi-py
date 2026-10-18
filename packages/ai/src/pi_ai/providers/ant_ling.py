"""Ant Ling provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["ant_ling_provider"]


def ant_ling_provider() -> Provider:
    """Ant Ling (OpenAI-compatible Chat Completions), authenticated with ``ANT_LING_API_KEY``."""
    return create_provider(
        id="ant-ling",
        name="Ant Ling",
        base_url="https://api.ant-ling.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Ant Ling API key", ["ANT_LING_API_KEY"])),
        models=lambda: catalog_models("ant-ling"),
        api=openai_completions_api(),
    )
