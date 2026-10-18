"""xAI provider."""

from __future__ import annotations

from ..api.registry import openai_responses_api
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_xai_oauth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["xai_provider"]


def xai_provider() -> Provider:
    """xAI (OpenAI Responses API): ``XAI_API_KEY`` or a SuperGrok / X Premium login."""
    return create_provider(
        id="xai",
        name="xAI",
        base_url="https://api.x.ai/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("xAI API key", ["XAI_API_KEY"]),
            oauth=lazy_oauth(
                "xAI (Grok/X subscription)",
                load_xai_oauth,
                is_subscription=True,
                login_label="Sign in with SuperGrok or X Premium",
            ),
        ),
        models=lambda: catalog_models("xai"),
        api=openai_responses_api(),
    )
