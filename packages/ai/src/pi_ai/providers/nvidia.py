"""NVIDIA provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["nvidia_provider"]


def nvidia_provider() -> Provider:
    """NVIDIA (OpenAI-compatible Chat Completions), authenticated with ``NVIDIA_API_KEY``."""
    return create_provider(
        id="nvidia",
        name="NVIDIA",
        base_url="https://integrate.api.nvidia.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("NVIDIA API key", ["NVIDIA_API_KEY"])),
        models=lambda: catalog_models("nvidia"),
        api=openai_completions_api(),
    )
