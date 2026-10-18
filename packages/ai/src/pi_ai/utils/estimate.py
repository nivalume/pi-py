"""Rough token estimates for context accounting."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..types import TranscriptContext, Usage
from .text import get_system_message_text

__all__ = [
    "ContextUsageEstimate",
    "calculate_context_tokens",
    "estimate_text_tokens",
    "estimate_text_and_image_content_tokens",
    "estimate_message_tokens",
    "estimate_context_tokens",
]

_CHARS_PER_TOKEN = 3.5
_ESTIMATED_IMAGE_CHARS = 4800


@dataclass(slots=True)
class ContextUsageEstimate:
    #: Estimated total context tokens.
    tokens: int
    #: Tokens reported by the most recent applicable assistant usage block.
    usage_tokens: int
    #: Estimated tokens after that usage block.
    trailing_tokens: int
    #: Index of the message that provided usage, or ``None``.
    last_usage_index: int | None


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def _safe_json(value: Any) -> str:
    try:
        from ..serialization import to_json_value

        return json.dumps(to_json_value(value), separators=(",", ":"))
    except Exception:
        return "[unserializable]"


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content:
        chars += len(block.text) if block.type == "text" else _ESTIMATED_IMAGE_CHARS
    return chars


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def estimate_text_and_image_content_tokens(content: Any) -> int:
    return math.ceil(_content_chars(content) / _CHARS_PER_TOKEN)


def _estimate_tools_tokens(tools: Sequence[Any] | None) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json(list(tools)))


def estimate_message_tokens(message: Any) -> int:
    role = message.role
    if role == "system":
        return (
            estimate_text_tokens(get_system_message_text(message))
            + _estimate_tools_tokens(message.tools_added)
            + _estimate_tools_tokens(message.tools_removed)
        )
    if role in ("user", "toolResult"):
        return estimate_text_and_image_content_tokens(message.content)
    if role != "assistant":
        return 0
    chars = 0
    for block in message.content:
        kind = block.type
        if kind == "text":
            chars += len(block.text)
        elif kind == "thinking":
            chars += len(block.thinking)
        else:
            chars += len(block.name) + len(_safe_json(block.arguments))
    return math.ceil(chars / _CHARS_PER_TOKEN)


def _last_assistant_usage(messages: Sequence[Any]) -> tuple[Usage, int] | None:
    latest_prefix_timestamp = -math.inf
    info: tuple[Usage, int] | None = None
    for index, message in enumerate(messages):
        if message.role == "assistant":
            applies = message.timestamp >= latest_prefix_timestamp
            if (
                applies
                and message.stop_reason not in ("aborted", "error")
                and calculate_context_tokens(message.usage) > 0
            ):
                info = (message.usage, index)
        latest_prefix_timestamp = max(latest_prefix_timestamp, getattr(message, "timestamp", 0))
    return info


def estimate_context_tokens(context: TranscriptContext | Sequence[Any]) -> ContextUsageEstimate:
    """Estimate context tokens, anchored on the most recent assistant usage report."""
    messages = context.messages if isinstance(context, TranscriptContext) else context
    usage_info = _last_assistant_usage(messages)
    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing = sum(estimate_message_tokens(message) for message in messages[index + 1 :])
        return ContextUsageEstimate(usage_tokens + trailing, usage_tokens, trailing, index)
    tokens = sum(estimate_message_tokens(message) for message in messages)
    return ContextUsageEstimate(tokens, 0, tokens, None)
