"""Transcript helpers: system prompt and tool state carried by system messages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..types import Context, Message, SystemMessage, Tool, ToolReference, TranscriptContext
from .schema import parameters_schema
from .text import content_text, get_system_message_text

__all__ = [
    "TranscriptContext",
    "create_initial_system_message",
    "normalize_context",
    "get_initial_system_message",
    "without_initial_system_message",
    "get_current_tools",
    "get_current_system_message",
    "get_current_system_prompt",
    "collapse_system_messages",
    "resolve_transcript",
    "to_tool_declaration",
    "declarations_equal",
    "ToolStateChanges",
    "get_tool_state_changes",
    "get_declared_tools",
    "has_tool_redefinitions",
    "has_non_additive_tool_changes",
    "TranscriptTools",
    "resolve_transcript_tools",
]


def create_initial_system_message(system_prompt: str | None, tools: Sequence[Tool] | None) -> SystemMessage | None:
    """Build the leading system message for a prompt and tool set.

    Returns ``None`` when both are empty, so an empty transcript stays empty.
    """
    has_prompt = bool(system_prompt)
    has_tools = bool(tools)
    if not has_prompt and not has_tools:
        return None
    return SystemMessage(system_prompt or "", tools_added=list(tools) if has_tools else None, timestamp=0)


def normalize_context(context: Context | Mapping[str, Any] | TranscriptContext) -> TranscriptContext:
    """Fold ``Context.system_prompt`` and ``Context.tools`` into a leading system message.

    This is the entry point that produces a :class:`TranscriptContext`. Plain dicts
    (``{"system_prompt": ..., "messages": [...], "tools": [...]}``, snake_case or pi's
    camelCase keys) and dict messages are accepted and converted.
    """
    if isinstance(context, TranscriptContext):
        return context
    if isinstance(context, Context):
        system_prompt, tools, messages = context.system_prompt, context.tools, context.messages
    else:
        system_prompt = context.get("system_prompt", context.get("systemPrompt"))
        tools = context.get("tools")
        messages = context.get("messages") or []
    messages = _coerce_messages(messages)
    if tools:
        tools = [_coerce_tool(tool) for tool in tools]
    initial = create_initial_system_message(system_prompt, tools)
    return TranscriptContext([initial, *messages] if initial else list(messages))


def _coerce_messages(messages: Sequence[Any]) -> list[Message]:
    if all(not isinstance(message, Mapping) for message in messages):
        return messages if isinstance(messages, list) else list(messages)
    from ..serialization import message_from_dict

    return [message_from_dict(message) if isinstance(message, Mapping) else message for message in messages]


def _coerce_tool(tool: Any) -> Tool:
    if isinstance(tool, Mapping):
        from ..serialization import tool_from_dict

        return tool_from_dict(tool)
    return tool


def _is_system(message: Any) -> bool:
    return getattr(message, "role", None) == "system"


def get_initial_system_message(messages: Sequence[Any]) -> SystemMessage | None:
    """The leading system message, if the transcript starts with one."""
    if messages and _is_system(messages[0]):
        return messages[0]
    return None


def without_initial_system_message(messages: list[Message]) -> list[Message]:
    """Drop the leading system message for APIs that carry the prompt outside the message list."""
    return messages[1:] if get_initial_system_message(messages) else messages


def get_current_tools(messages: Sequence[Any]) -> list[Tool]:
    """Resolve the tools available after applying every transcript delta in order."""
    tools: dict[str, Tool] = {}
    for message in messages:
        if not _is_system(message):
            continue
        for removed in message.tools_removed or ():
            tools.pop(removed.name, None)
        for added in message.tools_added or ():
            tools[added.name] = added
    return list(tools.values())


def get_current_system_message(messages: Sequence[Any]) -> SystemMessage | None:
    """Replay every system message into one leading system message.

    Later ``content`` is appended to the base prompt, ``sections`` are patched by
    name, and tools are resolved with :func:`get_current_tools`.
    """
    content: list[str] = []
    sections: dict[str, str] = {}
    timestamp: int | None = None
    for message in messages:
        if not _is_system(message):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        text = content_text(message.content)
        if text:
            content.append(text)
        for name, value in (message.sections or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
    tools = get_current_tools(messages)
    if timestamp is None and not tools:
        return None
    return SystemMessage(
        "\n\n".join(content),
        sections=dict(sections) if sections else None,
        tools_added=tools if tools else None,
        timestamp=timestamp if timestamp is not None else 0,
    )


def get_current_system_prompt(messages: Sequence[Any]) -> str:
    """The current system prompt text after replaying every system message."""
    message = get_current_system_message(messages)
    return get_system_message_text(message) if message else ""


def collapse_system_messages(context: TranscriptContext) -> TranscriptContext:
    """Rebuild the transcript for APIs without mid-conversation system messages.

    The replayed system message leads and every later system message is dropped.
    """
    head = get_current_system_message(context.messages)
    messages = [message for message in context.messages if not _is_system(message)]
    return TranscriptContext([head, *messages] if head else messages)


def resolve_transcript(
    context: TranscriptContext, supports_mid_convo_system_messages: bool | None
) -> TranscriptContext:
    """Keep later system messages in place when the model accepts them; otherwise collapse them."""
    return context if supports_mid_convo_system_messages else collapse_system_messages(context)


def to_tool_declaration(tool: Tool) -> Tool:
    """Strip executable and display-only fields from a tool (e.g. an agent tool).

    Parameters are deep-copied as plain JSON Schema (Pydantic models are converted).
    """
    parameters = json.loads(json.dumps(parameters_schema(tool.parameters)))
    constrained = tool.constrained_sampling
    return Tool(
        tool.name,
        tool.description,
        parameters,
        constrained_sampling=json.loads(json.dumps(constrained)) if isinstance(constrained, dict) else constrained,
    )


def _declaration_key(tool: Tool) -> str:
    constrained = tool.constrained_sampling
    return json.dumps(
        [tool.name, tool.description, parameters_schema(tool.parameters), constrained if constrained else None],
        sort_keys=True,
        default=str,
    )


def declarations_equal(left: Tool, right: Tool) -> bool:
    """Whether two tools declare the same interface to the model."""
    if left is right:
        return True
    return _declaration_key(left) == _declaration_key(right)


@dataclass(slots=True)
class ToolStateChanges:
    tools_added: list[Tool]
    tools_removed: list[ToolReference]


def get_tool_state_changes(previous: Sequence[Tool], current: Sequence[Tool]) -> ToolStateChanges:
    """Compare two complete tool states. A changed definition is a removal followed by an addition."""
    previous_keys = {tool.name: _declaration_key(tool) for tool in previous}
    current_keys = {tool.name: _declaration_key(tool) for tool in current}
    added = [to_tool_declaration(tool) for tool in current if previous_keys.get(tool.name) != current_keys[tool.name]]
    removed = [ToolReference(tool.name) for tool in previous if current_keys.get(tool.name) != previous_keys[tool.name]]
    return ToolStateChanges(added, removed)


def get_declared_tools(messages: Sequence[Any]) -> list[Tool]:
    """Every definition referenced by transcript tool state, in first-declaration order."""
    definitions: dict[str, Tool] = {}
    for message in messages:
        if _is_system(message):
            for tool in message.tools_added or ():
                definitions[tool.name] = tool
    return list(definitions.values())


def has_tool_redefinitions(messages: Sequence[Any]) -> bool:
    """Whether a tool name was declared twice with different definitions (deprecated in pi)."""
    declared: dict[str, Tool] = {}
    for message in messages:
        if not _is_system(message):
            continue
        for tool in message.tools_added or ():
            previous = declared.get(tool.name)
            if previous is not None and not declarations_equal(previous, tool):
                return True
            declared[tool.name] = tool
    return False


def has_non_additive_tool_changes(messages: Sequence[Any]) -> bool:
    """Whether tool history contains a removal or same-name redeclaration."""
    declared: set[str] = set()
    for message in messages:
        if not _is_system(message):
            continue
        if message.tools_removed:
            return True
        for tool in message.tools_added or ():
            if tool.name in declared:
                return True
            declared.add(tool.name)
    return False


@dataclass(slots=True)
class TranscriptTools:
    #: Tools sent in the top-level request field.
    request_tools: list[Tool]
    #: Whether later system messages carry their own ``tools_added`` as in-place additions.
    anchors_additions: bool


def resolve_transcript_tools(messages: Sequence[Any], supports_tool_additions: bool) -> TranscriptTools:
    """Split tool declarations between the top-level request field and in-place additions."""
    anchors = supports_tool_additions and not has_non_additive_tool_changes(messages)
    if anchors:
        initial = get_initial_system_message(messages)
        return TranscriptTools(list(initial.tools_added or []) if initial else [], True)
    return TranscriptTools(get_current_tools(messages), False)
