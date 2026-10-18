"""Qwen Token Plan provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["qwen_token_plan_provider"]


def qwen_token_plan_provider() -> Provider:
    """Qwen Token Plan (OpenAI-compatible Chat Completions), authenticated with ``QWEN_TOKEN_PLAN_API_KEY``."""
    return create_provider(
        id="qwen-token-plan",
        name="Qwen Token Plan",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Qwen Token Plan API key", ["QWEN_TOKEN_PLAN_API_KEY"])),
        models=lambda: catalog_models("qwen-token-plan"),
        api=openai_completions_api(),
    )
