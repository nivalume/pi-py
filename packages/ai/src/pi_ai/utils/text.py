"""Text helpers for message content and system prompts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import SystemMessage

__all__ = ["content_text", "get_system_message_text", "render_system_message_update", "append_text", "append_thinking"]


# ``block.text += delta`` copies the whole string on every delta because the attribute holds a
# second reference. Moving the string into a local first lets CPython resize it in place, which
# keeps accumulating a long streamed response linear instead of quadratic.


def append_text(block: Any, delta: str) -> None:
    """Append a streamed delta to ``block.text`` in amortized O(len(delta))."""
    text = block.text
    block.text = ""
    text += delta
    block.text = text


def append_thinking(block: Any, delta: str) -> None:
    """Append a streamed delta to ``block.thinking`` in amortized O(len(delta))."""
    text = block.thinking
    block.thinking = ""
    text += delta
    block.thinking = text


def content_text(content: str | Sequence[Any], separator: str = "\n") -> str:
    """Extract and join the text blocks of message content."""
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if block.type == "text")


def get_system_message_text(message: SystemMessage) -> str:
    """Render a system message as a complete prompt: its content followed by its sections."""
    parts = [content_text(message.content)]
    if message.sections:
        parts.extend(text for text in message.sections.values() if text is not None)
    return "\n\n".join(part for part in parts if part)


def render_system_message_update(message: SystemMessage) -> str:
    """Render a later system message for APIs that accept system messages mid-conversation.

    Section changes are framed by name so the model can relate them to the leading
    prompt. This framing is request-time only and may change between versions.
    """
    parts: list[str] = []
    text = content_text(message.content)
    if text:
        parts.append(text)
    for name, value in (message.sections or {}).items():
        parts.append(
            f'Removed system prompt section "{name}".'
            if value is None
            else f'Updated system prompt section "{name}":\n\n{value}'
        )
    return "\n\n".join(parts)
