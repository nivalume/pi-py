"""OpenAI provider."""

from __future__ import annotations

from collections.abc import Sequence

from ..api.registry import openai_decisions_api, openai_responses_api
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_openai_chatgpt_oauth
from ..auth.types import Credential, ProviderAuth
from ..models import Provider, create_provider, is_model_type
from ..types import AnyModel
from .catalog import catalog_models

__all__ = ["openai_provider"]


def _filter_all_models(models: Sequence[AnyModel], credential: Credential | None) -> Sequence[AnyModel]:
    # Sign in with ChatGPT tokens only reach the Responses API; the Decisions API rejects them.
    if credential is not None and credential.get("type") == "oauth":
        return [model for model in models if not is_model_type(model, "classifier")]
    return models


def openai_provider() -> Provider:
    """OpenAI (Responses API plus Decisions classifiers): ``OPENAI_API_KEY`` or Sign in with ChatGPT."""
    return create_provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"]),
            oauth=lazy_oauth(
                "OpenAI (ChatGPT subscription)",
                load_openai_chatgpt_oauth,
                is_subscription=True,
                login_label="Sign in with ChatGPT",
            ),
        ),
        models=lambda: catalog_models("openai"),
        filter_all_models=_filter_all_models,
        api=openai_responses_api(),
        classifiers={"openai-decisions": openai_decisions_api()},
    )
