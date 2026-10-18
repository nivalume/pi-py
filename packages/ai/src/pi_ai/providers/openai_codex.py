"""OpenAI Codex (legacy ChatGPT backend) provider."""

from __future__ import annotations

from ..api.registry import openai_codex_responses_api
from ..auth.helpers import lazy_oauth
from ..auth.oauth.load import load_openai_codex_oauth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["openai_codex_provider"]


def openai_codex_provider() -> Provider:
    """OpenAI Codex through the ChatGPT backend; OAuth (ChatGPT Plus/Pro) only."""
    return create_provider(
        id="openai-codex",
        name="OpenAI Codex (legacy)",
        base_url="https://chatgpt.com/backend-api",
        auth=ProviderAuth(
            oauth=lazy_oauth("OpenAI (ChatGPT Plus/Pro)", load_openai_codex_oauth, is_subscription=True),
        ),
        models=lambda: catalog_models("openai-codex"),
        api=openai_codex_responses_api(),
    )
