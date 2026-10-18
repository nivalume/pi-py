"""Cloudflare AI Gateway provider."""

from __future__ import annotations

from ..api.registry import anthropic_messages_api, openai_completions_api, openai_responses_api
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models
from .cloudflare_auth import cloudflare_ai_gateway_auth
from .cloudflare_stream import cloudflare_streams

__all__ = ["cloudflare_ai_gateway_provider"]


def cloudflare_ai_gateway_provider() -> Provider:
    """Cloudflare AI Gateway: ``CLOUDFLARE_API_KEY``, ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_GATEWAY_ID``."""
    # The api map is pinned to all three APIs: models.dev's gateway catalog drops and
    # restores `workers-ai/*` (openai-completions) entries over time, so the map must
    # not depend on which APIs the generated catalog happens to contain.
    return create_provider(
        id="cloudflare-ai-gateway",
        name="Cloudflare AI Gateway",
        auth=ProviderAuth(api_key=cloudflare_ai_gateway_auth()),
        models=lambda: catalog_models("cloudflare-ai-gateway"),
        api={
            "anthropic-messages": cloudflare_streams(anthropic_messages_api()),
            "openai-completions": cloudflare_streams(openai_completions_api()),
            "openai-responses": cloudflare_streams(openai_responses_api()),
        },
    )
