"""Kimi For Coding provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_kimi_coding_oauth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["kimi_coding_provider"]


def kimi_coding_provider() -> Provider:
    """Kimi For Coding: ``KIMI_API_KEY`` or a Kimi Code subscription login."""
    return create_provider(
        id="kimi-coding",
        name="Kimi For Coding",
        base_url="https://api.kimi.com/coding/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("Kimi API key", ["KIMI_API_KEY"]),
            oauth=lazy_oauth(
                "Kimi Code (subscription)",
                load_kimi_coding_oauth,
                is_subscription=True,
                login_label="Sign in with Kimi Code",
            ),
        ),
        models=lambda: catalog_models("kimi-coding"),
        api=openai_completions_api(),
    )
