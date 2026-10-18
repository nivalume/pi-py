"""GitHub Copilot provider."""

from __future__ import annotations

from collections.abc import Sequence

from ..api.registry import anthropic_messages_api, openai_completions_api, openai_responses_api
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_github_copilot_oauth
from ..auth.types import Credential, ProviderAuth
from ..models import Provider, create_provider
from ..types import Model
from .catalog import catalog_models

__all__ = ["github_copilot_provider"]


def _filter_models(models: Sequence[Model], credential: Credential | None) -> Sequence[Model]:
    # OAuth credentials record the models the Copilot plan enables.
    if credential is None or credential.get("type") != "oauth":
        return models
    available_model_ids = credential.get("availableModelIds")
    if not isinstance(available_model_ids, list) or not all(isinstance(item, str) for item in available_model_ids):
        return models
    available = set(available_model_ids)
    return [model for model in models if model.id in available]


def github_copilot_provider() -> Provider:
    """GitHub Copilot: ``COPILOT_GITHUB_TOKEN`` or a GitHub Copilot login (Anthropic/OpenAI APIs)."""
    return create_provider(
        id="github-copilot",
        name="GitHub Copilot",
        base_url="https://api.individual.githubcopilot.com",
        auth=ProviderAuth(
            api_key=env_api_key_auth("GitHub Copilot token", ["COPILOT_GITHUB_TOKEN"]),
            oauth=lazy_oauth("GitHub Copilot", load_github_copilot_oauth, is_subscription=True),
        ),
        models=lambda: catalog_models("github-copilot"),
        filter_models=_filter_models,
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
            "openai-responses": openai_responses_api(),
        },
    )
