"""OpenAI prompt-cache key helpers."""

from __future__ import annotations

__all__ = ["OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH", "clamp_openai_prompt_cache_key"]

OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    """Truncate a prompt cache key to OpenAI's limit, counting code points.

    JavaScript's ``Array.from`` counts code points too, so keys match the TypeScript port.
    """
    if key is None:
        return None
    if len(key) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return key[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH]
