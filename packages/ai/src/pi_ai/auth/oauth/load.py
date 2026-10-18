"""Loaders for the built-in OAuth flows.

Provider factories pass these to :func:`pi_ai.lazy_oauth` so flow modules (local
callback servers, PKCE helpers) import only when a login or refresh happens.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..types import OAuthAuth

__all__ = [
    "load_anthropic_oauth",
    "load_openai_codex_oauth",
    "load_openai_chatgpt_oauth",
    "load_github_copilot_oauth",
    "load_openrouter_oauth",
    "load_kimi_coding_oauth",
    "load_meta_oauth",
    "load_xai_oauth",
    "load_radius_oauth",
]


def _load(module: str, attribute: str) -> Any:
    return getattr(importlib.import_module(f"pi_ai.auth.oauth.{module}"), attribute)


def load_anthropic_oauth() -> OAuthAuth:
    return _load("anthropic", "anthropic_oauth")


def load_openai_codex_oauth() -> OAuthAuth:
    return _load("openai_codex", "openai_codex_oauth")


def load_openai_chatgpt_oauth() -> OAuthAuth:
    return _load("openai_chatgpt", "openai_chatgpt_oauth")


def load_github_copilot_oauth() -> OAuthAuth:
    return _load("github_copilot", "github_copilot_oauth")


def load_openrouter_oauth() -> OAuthAuth:
    return _load("openrouter", "openrouter_oauth")


def load_kimi_coding_oauth() -> OAuthAuth:
    return _load("kimi_coding", "kimi_coding_oauth")


def load_meta_oauth() -> OAuthAuth:
    return _load("meta", "meta_oauth")


def load_xai_oauth() -> OAuthAuth:
    return _load("xai", "xai_oauth")


def load_radius_oauth(*, name: str, gateway: str) -> OAuthAuth:
    return _load("radius", "create_radius_oauth")(name=name, gateway=gateway)
