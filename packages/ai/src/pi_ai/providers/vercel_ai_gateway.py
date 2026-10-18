"""Vercel AI Gateway provider."""

from __future__ import annotations

from ..api.registry import anthropic_messages_api, typesafe_system_one_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["vercel_ai_gateway_provider"]


def vercel_ai_gateway_provider() -> Provider:
    """Vercel AI Gateway (Anthropic Messages API), authenticated with ``AI_GATEWAY_API_KEY``."""
    return create_provider(
        id="vercel-ai-gateway",
        name="Vercel AI Gateway",
        base_url="https://ai-gateway.vercel.sh",
        auth=ProviderAuth(api_key=env_api_key_auth("Vercel AI Gateway API key", ["AI_GATEWAY_API_KEY"])),
        models=lambda: catalog_models("vercel-ai-gateway"),
        api=anthropic_messages_api(),
        # AI Gateway serves TypeSafe's System One protocol at /typesafe/v1/systemone.
        classifiers={"typesafe-system-one": typesafe_system_one_api()},
    )
