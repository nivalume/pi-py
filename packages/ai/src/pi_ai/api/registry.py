"""Lazy wrappers for the built-in API implementations.

Provider factories reference these instead of importing implementation modules,
so an implementation loads only on the first request to a model of that API
(the analog of pi's ``api/<id>.lazy.ts`` modules).
"""

from __future__ import annotations

from .lazy import LazyApi, LazyOperation

__all__ = [
    "API_MODULES",
    "anthropic_messages_api",
    "openai_completions_api",
    "openai_responses_api",
    "azure_openai_responses_api",
    "openai_codex_responses_api",
    "google_generative_ai_api",
    "google_vertex_api",
    "mistral_conversations_api",
    "bedrock_converse_stream_api",
    "pi_messages_api",
    "openrouter_images_api",
    "typesafe_system_one_api",
    "cloudflare_workers_ai_system_one_api",
    "llama_cpp_classify_api",
    "openai_decisions_api",
]

#: Chat API id -> implementation module.
API_MODULES: dict[str, str] = {
    "anthropic-messages": "pi_ai.api.anthropic_messages",
    "openai-completions": "pi_ai.api.openai_completions",
    "openai-responses": "pi_ai.api.openai_responses",
    "azure-openai-responses": "pi_ai.api.azure_openai_responses",
    "openai-codex-responses": "pi_ai.api.openai_codex_responses",
    "google-generative-ai": "pi_ai.api.google_generative_ai",
    "google-vertex": "pi_ai.api.google_vertex",
    "mistral-conversations": "pi_ai.api.mistral_conversations",
    "bedrock-converse-stream": "pi_ai.api.bedrock_converse_stream",
    "pi-messages": "pi_ai.api.pi_messages",
}


def anthropic_messages_api() -> LazyApi:
    return LazyApi(API_MODULES["anthropic-messages"])


def openai_completions_api() -> LazyApi:
    return LazyApi(API_MODULES["openai-completions"])


def openai_responses_api() -> LazyApi:
    return LazyApi(API_MODULES["openai-responses"])


def azure_openai_responses_api() -> LazyApi:
    return LazyApi(API_MODULES["azure-openai-responses"])


def openai_codex_responses_api() -> LazyApi:
    return LazyApi(API_MODULES["openai-codex-responses"])


def google_generative_ai_api() -> LazyApi:
    return LazyApi(API_MODULES["google-generative-ai"])


def google_vertex_api() -> LazyApi:
    return LazyApi(API_MODULES["google-vertex"])


def mistral_conversations_api() -> LazyApi:
    return LazyApi(API_MODULES["mistral-conversations"])


def bedrock_converse_stream_api() -> LazyApi:
    return LazyApi(API_MODULES["bedrock-converse-stream"])


def pi_messages_api() -> LazyApi:
    return LazyApi(API_MODULES["pi-messages"])


def openrouter_images_api() -> LazyOperation:
    return LazyOperation("pi_ai.api.openrouter_images", "generate_images")


def typesafe_system_one_api() -> LazyOperation:
    return LazyOperation("pi_ai.api.typesafe_system_one", "classify")


def cloudflare_workers_ai_system_one_api() -> LazyOperation:
    return LazyOperation("pi_ai.api.cloudflare_workers_ai_system_one", "classify")


def llama_cpp_classify_api() -> LazyOperation:
    return LazyOperation("pi_ai.api.llama_cpp_classify", "classify")


def openai_decisions_api() -> LazyOperation:
    return LazyOperation("pi_ai.api.openai_decisions", "classify")
