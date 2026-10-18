"""Moonshot AI CN provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["moonshotai_cn_provider"]


def moonshotai_cn_provider() -> Provider:
    """Moonshot AI CN (OpenAI-compatible Chat Completions), authenticated with ``MOONSHOT_API_KEY``."""
    return create_provider(
        id="moonshotai-cn",
        name="Moonshot AI CN",
        base_url="https://api.moonshot.cn/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
        models=lambda: catalog_models("moonshotai-cn"),
        api=openai_completions_api(),
    )
