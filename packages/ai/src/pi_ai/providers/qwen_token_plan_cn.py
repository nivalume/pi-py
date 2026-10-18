"""Qwen Token Plan CN provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["qwen_token_plan_cn_provider"]


def qwen_token_plan_cn_provider() -> Provider:
    """Qwen Token Plan CN (OpenAI-compatible Chat Completions), authenticated with ``QWEN_TOKEN_PLAN_CN_API_KEY``."""
    return create_provider(
        id="qwen-token-plan-cn",
        name="Qwen Token Plan CN",
        base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Qwen Token Plan CN API key", ["QWEN_TOKEN_PLAN_CN_API_KEY"])),
        models=lambda: catalog_models("qwen-token-plan-cn"),
        api=openai_completions_api(),
    )
