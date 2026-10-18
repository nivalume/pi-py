"""Cross-provider message normalization shared by API implementations."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

__all__ = ["SplitErrorToolResult", "split_error_tool_result_images", "transform_messages"]

_NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
_NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"
_ERROR_TOOL_IMAGE_PLACEHOLDER = "(image attached after the tool results)"


@dataclass(slots=True)
class SplitErrorToolResult:
    """An error tool result split into text-only content and images to send after the tool results."""

    content: list[TextContent]
    caption: str
    images: list[ImageContent]


def split_error_tool_result_images(message: ToolResultMessage) -> SplitErrorToolResult | None:
    """Split the images out of an error tool result (for APIs that reject images in error results)."""
    if not message.is_error:
        return None
    images = [block for block in message.content if block.type == "image"]
    if not images:
        return None
    return SplitErrorToolResult(
        content=_replace_images(message.content, _ERROR_TOOL_IMAGE_PLACEHOLDER),
        caption=f"Images from the error result of tool call {message.tool_call_id} ({message.tool_name}):",
        images=images,  # type: ignore[arg-type]
    )


def _replace_images(content: list[Any], placeholder: str) -> list[Any]:
    result: list[Any] = []
    previous_was_placeholder = False
    for block in content:
        if block.type == "image":
            if not previous_was_placeholder:
                result.append(TextContent(placeholder))
            previous_was_placeholder = True
            continue
        result.append(block)
        previous_was_placeholder = block.text == placeholder
    return result


def _has_images(content: Any) -> bool:
    return not isinstance(content, str) and any(block.type == "image" for block in content)


def _downgrade_unsupported_images(messages: list[Any], model: Model) -> list[Any]:
    if "image" in model.input:
        return messages
    result: list[Any] = []
    for message in messages:
        role = message.role
        if role == "user" and _has_images(message.content):
            message = dataclasses.replace(
                message, content=_replace_images(message.content, _NON_VISION_USER_IMAGE_PLACEHOLDER)
            )
        elif role == "toolResult" and _has_images(message.content):
            message = dataclasses.replace(
                message, content=_replace_images(message.content, _NON_VISION_TOOL_IMAGE_PLACEHOLDER)
            )
        result.append(message)
    return result


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    """Prepare a transcript for ``model``.

    * Images are replaced by placeholders for models without image input.
    * Thinking from another model is dropped; same-model thinking is kept for replay.
    * Tool call ids are normalized for the target API, and tool results follow.
    * Errored/aborted assistant turns are skipped.
    * Orphaned tool calls get synthetic error results so APIs accept the history.
    """
    tool_call_id_map: dict[str, str] = {}
    normalized: list[Any] = [
        dataclasses.replace(message, content=[]) if getattr(message, "content", []) is None else message
        for message in messages
    ]
    image_aware = _downgrade_unsupported_images(normalized, model)

    transformed: list[Any] = []
    for message in image_aware:
        role = message.role
        if role in ("system", "user"):
            transformed.append(message)
            continue
        if role == "toolResult":
            mapped = tool_call_id_map.get(message.tool_call_id)
            if mapped and mapped != message.tool_call_id:
                message = dataclasses.replace(message, tool_call_id=mapped)
            transformed.append(message)
            continue
        if role != "assistant":
            transformed.append(message)
            continue

        is_same_model = message.provider == model.provider and message.api == model.api and message.model == model.id
        content: list[Any] = []
        changed = False
        for block in message.content:
            kind = block.type
            if kind == "thinking":
                if not is_same_model:
                    changed = True
                    continue
                if block.redacted or block.thinking_signature:
                    content.append(block)
                    continue
                if not block.thinking or not block.thinking.strip():
                    changed = True
                    continue
                content.append(block)
            elif kind == "text":
                if is_same_model or block.text_signature is None:
                    content.append(block)
                else:
                    content.append(TextContent(block.text))
                    changed = True
            elif kind == "toolCall":
                tool_call: ToolCall = block
                if not is_same_model and tool_call.thought_signature:
                    tool_call = dataclasses.replace(tool_call, thought_signature=None)
                    changed = True
                if not is_same_model and normalize_tool_call_id is not None:
                    normalized_id = normalize_tool_call_id(tool_call.id, model, message)
                    if normalized_id != tool_call.id:
                        tool_call_id_map[tool_call.id] = normalized_id
                        tool_call = dataclasses.replace(tool_call, id=normalized_id)
                        changed = True
                content.append(tool_call)
            else:
                content.append(block)
        if changed:
            message = message.shallow_copy()
            message.content = content
        transformed.append(message)

    result: list[Any] = []
    pending_tool_calls: list[ToolCall] = []
    existing_result_ids: set[str] = set()
    held_system: list[Any] = []

    def close_pending() -> None:
        nonlocal pending_tool_calls, existing_result_ids
        if pending_tool_calls:
            for tool_call in pending_tool_calls:
                if tool_call.id not in existing_result_ids:
                    result.append(
                        ToolResultMessage(
                            tool_call.id,
                            tool_call.name,
                            [TextContent("No result provided")],
                            True,
                        )
                    )
            pending_tool_calls = []
            existing_result_ids = set()
        result.extend(held_system)
        held_system.clear()

    for message in transformed:
        role = message.role
        if role == "assistant":
            close_pending()
            if message.stop_reason in ("error", "aborted"):
                continue
            tool_calls = [block for block in message.content if block.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_result_ids = set()
            result.append(message)
        elif role == "toolResult":
            existing_result_ids.add(message.tool_call_id)
            result.append(message)
        elif role == "system":
            if pending_tool_calls:
                held_system.append(message)
            else:
                result.append(message)
        elif role == "user":
            close_pending()
            result.append(message)
        else:
            result.append(message)
    close_pending()
    return result
