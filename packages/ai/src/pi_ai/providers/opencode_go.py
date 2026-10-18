"""OpenCode Go provider."""

from __future__ import annotations

from ..api.registry import (
    anthropic_messages_api,
    google_generative_ai_api,
    openai_completions_api,
    openai_responses_api,
)
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models
from .opencode_headers import with_opencode_session_header

__all__ = ["opencode_go_provider"]


def opencode_go_provider() -> Provider:
    """OpenCode Go (mixed APIs), authenticated with ``OPENCODE_API_KEY``."""
    return create_provider(
        id="opencode-go",
        name="OpenCode Go",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenCode API key", ["OPENCODE_API_KEY"])),
        models=lambda: catalog_models("opencode-go"),
        api={
            "anthropic-messages": with_opencode_session_header(anthropic_messages_api()),
            "google-generative-ai": with_opencode_session_header(google_generative_ai_api()),
            "openai-completions": with_opencode_session_header(openai_completions_api()),
            "openai-responses": with_opencode_session_header(openai_responses_api()),
        },
    )
