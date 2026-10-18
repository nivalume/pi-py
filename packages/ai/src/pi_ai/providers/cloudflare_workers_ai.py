"""Cloudflare Workers AI provider."""

from __future__ import annotations

from ..api.registry import cloudflare_workers_ai_system_one_api, openai_completions_api
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models
from .cloudflare_auth import cloudflare_workers_ai_auth
from .cloudflare_stream import cloudflare_classifier, cloudflare_streams

__all__ = ["cloudflare_workers_ai_provider"]


def cloudflare_workers_ai_provider() -> Provider:
    """Cloudflare Workers AI: ``CLOUDFLARE_API_KEY`` plus ``CLOUDFLARE_ACCOUNT_ID``."""
    return create_provider(
        id="cloudflare-workers-ai",
        name="Cloudflare Workers AI",
        auth=ProviderAuth(api_key=cloudflare_workers_ai_auth()),
        models=lambda: catalog_models("cloudflare-workers-ai"),
        api=cloudflare_streams(openai_completions_api()),
        classifiers={
            "cloudflare-workers-ai-system-one": cloudflare_classifier(cloudflare_workers_ai_system_one_api()),
        },
    )
