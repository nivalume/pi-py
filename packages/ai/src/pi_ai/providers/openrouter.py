"""OpenRouter provider."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Sequence
from typing import Any

import httpx

from ..abort import AbortSignal, abort_scope
from ..api.registry import (
    anthropic_messages_api,
    openai_completions_api,
    openrouter_images_api,
    typesafe_system_one_api,
)
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_openrouter_oauth
from ..auth.types import ProviderAuth
from ..models import Provider, RefreshModelsContext, create_provider
from ..types import AnyModel
from ..utils.http import get_http_client
from ..utils.model_operations import is_model_type
from .catalog import catalog_models
from .openrouter_api import openrouter_cost

__all__ = ["OPENROUTER_BASE_URL", "openrouter_provider", "personalize_openrouter_catalog"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_USER_MODELS_TIMEOUT_MS = 15_000

_TRAILING_SLASHES = re.compile(r"/+$")


def _truthy(value: Any) -> bool:
    """JavaScript truthiness (``{}`` and ``[]`` are truthy, ``NaN`` is not)."""
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, (dict, list)):
        return True
    return bool(value)


def _first_truthy(*values: Any) -> Any:
    for value in values[:-1]:
        if _truthy(value):
            return value
    return values[-1]


async def personalize_openrouter_catalog(
    catalog: Sequence[AnyModel],
    context: RefreshModelsContext,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> Sequence[AnyModel]:
    """Restrict an OpenRouter catalog to the chat models the key may use, with the key's limits and prices.

    Uses ``GET /models/user``. Image and classifier models are returned unchanged and a
    missing limit keeps the catalog value. The request goes to the provider's configured
    base URL, so regional endpoints such as ``https://us.openrouter.ai/api/v1`` apply their
    region's filtering. When the endpoint rejects the key (401/403) or does not exist (404),
    for example with a proxy configured in models.json, ``catalog`` itself is returned.
    """
    credential = context.credential
    if credential is not None and credential.get("type") == "oauth":
        key = credential.get("access")
    else:
        key = credential.get("key") if credential is not None else None
    if not key:
        return catalog
    base_url = context.base_url if context.base_url is not None else OPENROUTER_BASE_URL
    url = f"{_TRAILING_SLASHES.sub('', base_url)}/models/user"
    signal = AbortSignal.any([context.signal, AbortSignal.timeout(OPENROUTER_USER_MODELS_TIMEOUT_MS / 1000)])
    client = http_client if http_client is not None else get_http_client(url)
    async with abort_scope(signal):
        response = await client.get(
            url,
            headers={"accept": "application/json", "authorization": f"Bearer {key}"},
            timeout=OPENROUTER_USER_MODELS_TIMEOUT_MS / 1000,
        )
        raw = await response.aread()
    if response.status_code in (401, 403, 404):
        return catalog
    if not response.is_success:
        raise RuntimeError(f"OpenRouter model list request failed: {response.status_code}")
    body: Any = json.loads(raw)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise ValueError("Invalid OpenRouter model list response")
    allowed: dict[str, dict[str, Any]] = {}
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            allowed[entry["id"]] = entry

    personalized: list[AnyModel] = []
    for model in catalog:
        if not is_model_type(model, "chat"):
            personalized.append(model)
            continue
        listed = allowed.get(model.id)
        if listed is None:
            continue
        top_provider = listed.get("top_provider")
        top_provider = top_provider if isinstance(top_provider, dict) else {}
        pricing = listed.get("pricing")
        personalized.append(
            dataclasses.replace(
                model,  # type: ignore[arg-type]
                # `or`: OpenRouter reports a missing limit as absent, null or 0, like the generator's fallback.
                context_window=_first_truthy(
                    top_provider.get("context_length"),
                    listed.get("context_length"),
                    model.context_window,  # type: ignore[union-attr]
                ),
                max_tokens=_first_truthy(top_provider.get("max_completion_tokens"), model.max_tokens),  # type: ignore[union-attr]
                cost=openrouter_cost(pricing) if _truthy(pricing) else model.cost,
            )
        )
    return personalized


def openrouter_provider() -> Provider:
    """OpenRouter (chat, image and System One classifier models): ``OPENROUTER_API_KEY`` or OAuth."""
    return create_provider(
        id="openrouter",
        name="OpenRouter",
        base_url=OPENROUTER_BASE_URL,
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
            oauth=lazy_oauth("OpenRouter OAuth", load_openrouter_oauth, login_label="Sign in with OpenRouter"),
        ),
        models=lambda: catalog_models("openrouter"),
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
        },
        images={"openrouter-images": openrouter_images_api()},
        # OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
        classifiers={"typesafe-system-one": typesafe_system_one_api()},
    )
