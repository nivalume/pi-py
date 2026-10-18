"""Dynamic request headers required by GitHub Copilot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["infer_copilot_initiator", "has_copilot_vision_input", "build_copilot_dynamic_headers"]


def infer_copilot_initiator(messages: Sequence[Any]) -> str:
    """``"user"`` for user-initiated requests, ``"agent"`` for follow-ups after assistant/tool messages."""
    last = messages[-1] if messages else None
    return "agent" if last is not None and last.role != "user" else "user"


def has_copilot_vision_input(messages: Sequence[Any]) -> bool:
    """Copilot requires ``Copilot-Vision-Request`` when any user or tool-result message carries images."""
    for message in messages:
        if message.role in ("user", "toolResult") and not isinstance(message.content, str):
            if any(block.type == "image" for block in message.content):
                return True
    return False


def build_copilot_dynamic_headers(*, messages: Sequence[Any], has_images: bool) -> dict[str, str]:
    headers = {"X-Initiator": infer_copilot_initiator(messages), "Openai-Intent": "conversation-edits"}
    if has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers
