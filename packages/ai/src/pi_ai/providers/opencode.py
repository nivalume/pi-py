"""OpenCode Zen provider."""

from __future__ import annotations

from ..api.registry import (
    anthropic_messages_api,
    google_generative_ai_api,
    openai_completions_api,
    openai_responses_api,
    typesafe_system_one_api,
)
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models
from .opencode_headers import with_opencode_session_header

__all__ = ["opencode_provider"]


def opencode_provider() -> Provider:
    """OpenCode Zen (mixed APIs), authenticated with ``OPENCODE_API_KEY``."""
    return create_provider(
        id="opencode",
        name="OpenCode Zen",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenCode API key", ["OPENCODE_API_KEY"])),
        models=lambda: catalog_models("opencode"),
        api={
            "anthropic-messages": with_opencode_session_header(anthropic_messages_api()),
            "google-generative-ai": with_opencode_session_header(google_generative_ai_api()),
            "openai-completions": with_opencode_session_header(openai_completions_api()),
            "openai-responses": with_opencode_session_header(openai_responses_api()),
        },
        # OpenCode Zen serves TypeSafe's System One protocol at /zen/v1/systemone.
        classifiers={"typesafe-system-one": typesafe_system_one_api()},
    )
