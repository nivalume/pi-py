"""Z.AI provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["zai_provider"]


def zai_provider() -> Provider:
    """Z.AI (OpenAI-compatible Chat Completions), authenticated with ``ZAI_API_KEY``."""
    return create_provider(
        id="zai",
        name="Z.AI",
        base_url="https://api.z.ai/api/coding/paas/v4",
        auth=ProviderAuth(api_key=env_api_key_auth("Z.AI API key", ["ZAI_API_KEY"])),
        models=lambda: catalog_models("zai"),
        api=openai_completions_api(),
    )
