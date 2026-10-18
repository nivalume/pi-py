"""DeepSeek provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["deepseek_provider"]


def deepseek_provider() -> Provider:
    """DeepSeek (OpenAI-compatible Chat Completions), authenticated with ``DEEPSEEK_API_KEY``."""
    return create_provider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        auth=ProviderAuth(api_key=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])),
        models=lambda: catalog_models("deepseek"),
        api=openai_completions_api(),
    )
