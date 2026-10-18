"""MiniMax provider."""

from __future__ import annotations

from ..api.registry import anthropic_messages_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["minimax_provider"]


def minimax_provider() -> Provider:
    """MiniMax (Anthropic Messages API), authenticated with ``MINIMAX_API_KEY``."""
    return create_provider(
        id="minimax",
        name="MiniMax",
        base_url="https://api.minimax.io/anthropic",
        auth=ProviderAuth(api_key=env_api_key_auth("MiniMax API key", ["MINIMAX_API_KEY"])),
        models=lambda: catalog_models("minimax"),
        api=anthropic_messages_api(),
    )
