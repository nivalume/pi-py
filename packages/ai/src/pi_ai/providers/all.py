"""Every built-in provider, plus static reads of the generated catalog.

::

    from pi_ai.providers.all import builtin_models

    models = builtin_models()
    model = models.get_model("anthropic", "claude-sonnet-4-5")

Building providers is cheap: catalogs load on first access to a provider's models
and API implementations on the first request.
"""

from __future__ import annotations

from typing import TypeAlias

from ..auth.types import AuthContext, CredentialStore
from ..models import Models, Provider, create_models
from ..models_store import ModelsStore
from .amazon_bedrock import amazon_bedrock_provider
from .ant_ling import ant_ling_provider
from .anthropic import anthropic_provider
from .azure import azure_provider
from .baseten import baseten_provider
from .catalog import (
    BUILTIN_PROVIDERS,
    get_all_builtin_models,
    get_builtin_classifier_model,
    get_builtin_classifier_models,
    get_builtin_image_model,
    get_builtin_image_models,
    get_builtin_model,
    get_builtin_model_data_generated_at,
    get_builtin_models,
    get_builtin_providers,
)
from .cerebras import cerebras_provider
from .cloudflare_ai_gateway import cloudflare_ai_gateway_provider
from .cloudflare_workers_ai import cloudflare_workers_ai_provider
from .deepseek import deepseek_provider
from .fireworks import fireworks_provider
from .github_copilot import github_copilot_provider
from .google import google_provider
from .google_vertex import google_vertex_provider
from .groq import groq_provider
from .huggingface import huggingface_provider
from .kimi_coding import kimi_coding_provider
from .meta import meta_provider
from .minimax import minimax_provider
from .minimax_cn import minimax_cn_provider
from .mistral import mistral_provider
from .moonshotai import moonshotai_provider
from .moonshotai_cn import moonshotai_cn_provider
from .nvidia import nvidia_provider
from .openai import openai_provider
from .openai_codex import openai_codex_provider
from .opencode import opencode_provider
from .opencode_go import opencode_go_provider
from .openrouter import openrouter_provider
from .qwen_token_plan import qwen_token_plan_provider
from .qwen_token_plan_cn import qwen_token_plan_cn_provider
from .qwen_token_plan_individual import qwen_token_plan_individual_provider
from .radius import radius_provider
from .together import together_provider
from .typesafe import typesafe_provider
from .vercel_ai_gateway import vercel_ai_gateway_provider
from .xai import xai_provider
from .xiaomi import xiaomi_provider
from .xiaomi_token_plan_ams import xiaomi_token_plan_ams_provider
from .xiaomi_token_plan_cn import xiaomi_token_plan_cn_provider
from .xiaomi_token_plan_sgp import xiaomi_token_plan_sgp_provider
from .zai import zai_provider
from .zai_coding_cn import zai_coding_cn_provider

__all__ = [
    "BuiltinProvider",
    "BUILTIN_PROVIDERS",
    "builtin_providers",
    "builtin_models",
    "radius_provider",
    "get_builtin_model",
    "get_builtin_image_model",
    "get_builtin_classifier_model",
    "get_builtin_providers",
    "get_builtin_model_data_generated_at",
    "get_builtin_models",
    "get_builtin_image_models",
    "get_builtin_classifier_models",
    "get_all_builtin_models",
]

#: Providers present in the generated catalog (see :data:`BUILTIN_PROVIDERS`).
BuiltinProvider: TypeAlias = str


def builtin_providers() -> list[Provider]:
    """All built-in providers, freshly constructed."""
    return [
        amazon_bedrock_provider(),
        ant_ling_provider(),
        anthropic_provider(),
        azure_provider(),
        baseten_provider(),
        cerebras_provider(),
        cloudflare_ai_gateway_provider(),
        cloudflare_workers_ai_provider(),
        deepseek_provider(),
        fireworks_provider(),
        github_copilot_provider(),
        google_provider(),
        google_vertex_provider(),
        groq_provider(),
        huggingface_provider(),
        kimi_coding_provider(),
        meta_provider(),
        minimax_provider(),
        minimax_cn_provider(),
        mistral_provider(),
        moonshotai_provider(),
        moonshotai_cn_provider(),
        nvidia_provider(),
        openai_provider(),
        openai_codex_provider(),
        opencode_provider(),
        opencode_go_provider(),
        openrouter_provider(),
        qwen_token_plan_provider(),
        qwen_token_plan_cn_provider(),
        qwen_token_plan_individual_provider(),
        radius_provider(),
        together_provider(),
        typesafe_provider(),
        vercel_ai_gateway_provider(),
        xai_provider(),
        xiaomi_provider(),
        xiaomi_token_plan_ams_provider(),
        xiaomi_token_plan_cn_provider(),
        xiaomi_token_plan_sgp_provider(),
        zai_provider(),
        zai_coding_cn_provider(),
    ]


def builtin_models(
    credentials: CredentialStore | None = None,
    models_store: ModelsStore | None = None,
    auth_context: AuthContext | None = None,
) -> Models:
    """A :class:`~pi_ai.Models` collection with every built-in provider registered."""
    models = create_models(credentials=credentials, models_store=models_store, auth_context=auth_context)
    for provider in builtin_providers():
        models.set_provider(provider)
    return models
