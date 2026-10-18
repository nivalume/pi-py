"""Z.AI Coding CN provider."""

from __future__ import annotations

from ..api.registry import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["zai_coding_cn_provider"]


def zai_coding_cn_provider() -> Provider:
    """Z.AI Coding CN (OpenAI-compatible Chat Completions), authenticated with ``ZAI_CODING_CN_API_KEY``."""
    return create_provider(
        id="zai-coding-cn",
        name="Z.AI Coding CN",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        auth=ProviderAuth(api_key=env_api_key_auth("Z.AI Coding CN API key", ["ZAI_CODING_CN_API_KEY"])),
        models=lambda: catalog_models("zai-coding-cn"),
        api=openai_completions_api(),
    )
