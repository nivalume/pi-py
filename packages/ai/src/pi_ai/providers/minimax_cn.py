"""MiniMax CN provider."""

from __future__ import annotations

from ..api.registry import anthropic_messages_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["minimax_cn_provider"]


def minimax_cn_provider() -> Provider:
    """MiniMax CN (Anthropic Messages API), authenticated with ``MINIMAX_CN_API_KEY``."""
    return create_provider(
        id="minimax-cn",
        name="MiniMax CN",
        base_url="https://api.minimaxi.com/anthropic",
        auth=ProviderAuth(api_key=env_api_key_auth("MiniMax CN API key", ["MINIMAX_CN_API_KEY"])),
        models=lambda: catalog_models("minimax-cn"),
        api=anthropic_messages_api(),
    )
