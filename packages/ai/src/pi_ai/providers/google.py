"""Google provider."""

from __future__ import annotations

from ..api.registry import google_generative_ai_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["google_provider"]


def google_provider() -> Provider:
    """Google (Gemini API), authenticated with ``GEMINI_API_KEY``."""
    return create_provider(
        id="google",
        name="Google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        auth=ProviderAuth(api_key=env_api_key_auth("Gemini API key", ["GEMINI_API_KEY"])),
        models=lambda: catalog_models("google"),
        api=google_generative_ai_api(),
    )
