"""Together provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["together_provider"]


def together_provider() -> Provider:
    """Together (OpenAI-compatible Chat Completions), authenticated with ``TOGETHER_API_KEY``."""
    return create_provider(
        id="together",
        name="Together",
        base_url="https://api.together.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Together API key", ["TOGETHER_API_KEY"])),
        models=lambda: catalog_models("together"),
        api=openai_completions_api(),
    )
