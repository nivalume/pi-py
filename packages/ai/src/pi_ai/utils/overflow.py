"""Context overflow detection across providers.

See the TypeScript source for the catalogue of provider error messages each
pattern matches; the patterns are kept identical.
"""

from __future__ import annotations

import re

from ..types import AssistantMessage

__all__ = ["is_context_overflow", "is_recoverable_length", "get_overflow_patterns"]

_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt (?:is )?too long",  # Anthropic and z.ai token overflow
        r"prompt exceeds max length",  # z.ai CN endpoint token overflow
        r"request_too_large",  # Anthropic request byte-size overflow (HTTP 413)
        r"input is too long for requested model",  # Amazon Bedrock
        r"exceeds the context window",  # OpenAI (Completions & Responses API)
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",  # LiteLLM
        r"input token count.*exceeds the maximum",  # Google (Gemini)
        r"maximum prompt length is \d+",  # xAI (Grok)
        r"reduce the length of the messages",  # Groq
        r"maximum context length is \d+ tokens",  # OpenRouter (most backends)
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",  # OpenRouter/Poolside
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",  # Together AI
        r"exceeds the limit of \d+",  # GitHub Copilot
        r"exceeds the available context size",  # llama.cpp server
        r"greater than the context length",  # LM Studio
        r"context window exceeds limit",  # MiniMax
        r"exceeded model token limit",  # Kimi For Coding
        r"too large for model with \d+ maximum context length",  # Mistral
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",  # DS4 server
        r"model_context_window_exceeded",  # z.ai non-standard finish_reason
        r"prompt too long; exceeded (?:max )?context length",  # Ollama
        r"range of input length should be",  # DashScope / Qwen Token Plan
        r"context[_ ]length[_ ]exceeded",  # Generic fallback
        r"too many tokens",  # Generic fallback
        r"token limit exceeded",  # Generic fallback
    )
]

_CEREBRAS_BODYLESS_OVERFLOW = re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE)

_NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),  # AWS Bedrock non-overflow errors
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
]


def is_context_overflow(message: AssistantMessage, context_window: int | None = None) -> bool:
    """Whether an assistant message represents a context overflow.

    Handles error-message overflow (most providers), silent overflow where
    ``usage.input`` exceeds the context window (z.ai), and length-stop overflow with
    zero output (Xiaomi MiMo). Pass ``context_window`` for the latter two.
    """
    error_message = message.error_message
    if message.stop_reason == "error" and error_message:
        if not any(pattern.search(error_message) for pattern in _NON_OVERFLOW_PATTERNS):
            if any(pattern.search(error_message) for pattern in _OVERFLOW_PATTERNS):
                return True
            if message.provider == "cerebras" and _CEREBRAS_BODYLESS_OVERFLOW.search(error_message):
                return True

    if context_window and message.stop_reason == "stop":
        if message.usage.input + message.usage.cache_read > context_window:
            return True

    if context_window and message.stop_reason == "length" and message.usage.output == 0:
        if message.usage.input + message.usage.cache_read >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: AssistantMessage, desired_max_output: int) -> bool:
    """Whether a length stop ended below the intended output limit (context pressure)."""
    return message.stop_reason == "length" and desired_max_output > 0 and message.usage.output < desired_max_output


def get_overflow_patterns() -> list[re.Pattern[str]]:
    return list(_OVERFLOW_PATTERNS)
