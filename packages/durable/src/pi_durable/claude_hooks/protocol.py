"""What handlers receive and return: Claude Code's hook input and output JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["HookOutput", "claude_tool_name", "output_block", "specific"]

#: A handler's JSON output, as Claude Code defines it (every field optional): ``continue``, ``stopReason``,
#: ``systemMessage``, ``decision``, ``reason``, ``hookSpecificOutput``.
HookOutput = dict[str, Any]

_TOOL_NAMES = {"bash": "Bash", "read": "Read", "write": "Write", "edit": "Edit"}


def specific(output: Mapping[str, Any], name: str) -> Any:
    """A field of ``hookSpecificOutput``."""
    fields = output.get("hookSpecificOutput")
    return fields.get(name) if isinstance(fields, Mapping) else None


def specific_str(output: Mapping[str, Any], name: str) -> str | None:
    value = specific(output, name)
    return value if isinstance(value, str) else None


def output_block(output: Mapping[str, Any]) -> str | None:
    """Whether the output blocks (``decision: "block"`` or ``continue: false``), with the reason to show."""
    if output.get("decision") == "block":
        reason = output.get("reason")
        return reason if isinstance(reason, str) else "Blocked by hook"
    if output.get("continue") is False:
        reason = output.get("stopReason")
        return reason if isinstance(reason, str) else "Stopped by hook"
    return None


def claude_tool_name(name: str) -> str:
    """The name Claude Code's hooks know a built-in tool by (``bash`` is ``Bash``); other tools keep their names."""
    return _TOOL_NAMES.get(name, name)


def claude_tool_input(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """A call's arguments as hooks see them: ``path`` is also given as Claude Code's ``file_path``."""
    tool_input = dict(arguments)
    if isinstance(arguments.get("path"), str) and "file_path" not in tool_input:
        tool_input["file_path"] = arguments["path"]
    return tool_input


def arguments_from(updated: Any, original: Mapping[str, Any]) -> dict[str, Any] | None:
    """A hook's ``updatedInput`` as arguments: ``file_path`` back to ``path`` when the tool takes ``path``."""
    if not isinstance(updated, Mapping):
        return None
    arguments = dict(updated)
    if "path" in original and "path" not in arguments and "file_path" in arguments:
        arguments["path"] = arguments.pop("file_path")
    elif "path" in original and "file_path" not in original:
        arguments.pop("file_path", None)
    return arguments
