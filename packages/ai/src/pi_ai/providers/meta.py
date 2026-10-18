"""Meta provider."""

from __future__ import annotations

from ..api.registry import openai_responses_api
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_meta_oauth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["meta_provider"]


def meta_provider() -> Provider:
    """Meta (OpenAI Responses API): ``META_API_KEY`` or a Muse subscription login."""
    return create_provider(
        id="meta",
        name="Meta",
        base_url="https://api.meta.ai/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("Meta Model API key", ["META_API_KEY"]),
            oauth=lazy_oauth(
                "Meta (Muse subscription)",
                load_meta_oauth,
                is_subscription=True,
                login_label="Sign in with Meta",
            ),
        ),
        models=lambda: catalog_models("meta"),
        api=openai_responses_api(),
    )
