"""Positional system prompt and tool loadout planning (port of ``harness/prompt.ts``)."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pi_ai.types import Message, SystemMessage, Tool, ToolReference
from pi_ai.utils.transcript import declarations_equal, get_current_tools, to_tool_declaration
from pi_chord.context import Context

from ..entries import SystemEntry
from ..types import ContextEdit, TypedEntryDraft
from .types import ContextView, PromptInput, PromptSection

__all__ = ["plan_system_entries", "render_sections", "replay_sections"]


def replay_sections(messages: Sequence[Message]) -> dict[str, str]:
    """Sections in effect after replaying system messages in order: set in place, ``None`` deletes, re-adding appends."""
    shown: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, SystemMessage) or message.sections is None:
            continue
        for key, value in message.sections.items():
            if value is None:
                shown.pop(key, None)
            elif key in shown:
                shown[key] = value
            else:
                shown[key] = value
    return shown


async def render_sections(
    sections: Sequence[PromptSection],
    input: PromptInput,
    shown: Mapping[str, str],
    report: Callable[[BaseException, PromptSection], None],
    context: Context,
) -> dict[str, str]:
    """Render the agent's sections in order.

    ``None`` omits a section; tagged text is wrapped as ``<key>\\n...\\n</key>``. A section that raises keeps its
    shown text, if any, and is reported; errors after ``context`` is cancelled propagate.
    """
    desired: dict[str, str] = {}
    for section in sections:
        try:
            result: Any = section.render(input, context)
            text = await result if inspect.isawaitable(result) else result
        except Exception as error:
            signal = context.abort_signal
            if signal is not None and signal.aborted:
                raise
            report(error, section)
            kept = shown.get(section.key)
            if kept is not None:
                desired[section.key] = kept
            continue
        if text is None:
            continue
        desired[section.key] = text if section.tag is False else f"<{section.key}>\n{text}\n</{section.key}>"
    return desired


def plan_system_entries(
    view: ContextView, desired: Mapping[str, str], tools: Sequence[Tool], timestamp: float
) -> list[TypedEntryDraft[Any]]:
    """Plan the ``pi.system`` entries that make the replayed sections and tools of ``view`` equal ``desired`` and
    ``tools`` in values and order.

    - A head marker with no later ``pi.system`` entry in context: one complete baseline that omits every retained
      earlier ``pi.system`` entry, written even when it restates the replayed values.
    - Otherwise, when a minimal section patch would leave a different order: remove every shown section, then
      re-add every desired section in order.
    - Otherwise the minimal patch of changed values and ``None`` removals, or nothing.

    Tool changes ride on the last planned entry, or on one entry of their own.
    """
    head = view["head"]
    if head is not None and not any(SystemEntry.is_(entry) and entry["id"] > head["id"] for entry in view["entries"]):
        edits: list[ContextEdit] = [
            {"target": entry["id"], "action": "omit"} for entry in view["entries"] if SystemEntry.is_(entry)
        ]
        baseline = _system_entry(dict(desired), ([], [to_tool_declaration(tool) for tool in tools]), timestamp)
        return [baseline if len(edits) == 0 else {**baseline, "edits": edits}]
    sections = _plan_sections(replay_sections(view["messages"]), desired)
    changes = _plan_tools(get_current_tools(view["messages"]), tools)
    if len(changes[0]) == 0 and len(changes[1]) == 0:
        return [_system_entry(patch, None, timestamp) for patch in sections]
    if len(sections) == 0:
        return [_system_entry(None, changes, timestamp)]
    return [
        _system_entry(patch, changes if index == len(sections) - 1 else None, timestamp)
        for index, patch in enumerate(sections)
    ]


def _plan_tools(offered: Sequence[Tool], desired: Sequence[Tool]) -> tuple[list[ToolReference], list[Tool]]:
    """Tool changes from ``offered`` to ``desired``: ``(removed, added)``.

    A changed declaration is removed and re-added. Replay keeps retained tools in place and appends additions;
    when that would not yield the desired order, every offered tool is removed and every desired tool re-added.
    """
    wanted = {tool.name: tool for tool in desired}
    kept = [tool for tool in offered if (nxt := wanted.get(tool.name)) is not None and declarations_equal(tool, nxt)]
    kept_names = {tool.name for tool in kept}
    added = [tool for tool in desired if tool.name not in kept_names]
    replayed = [*kept, *added]
    if any(tool.name != desired[index].name for index, tool in enumerate(replayed)):
        return [ToolReference(tool.name) for tool in offered], [to_tool_declaration(tool) for tool in desired]
    return (
        [ToolReference(tool.name) for tool in offered if tool.name not in kept_names],
        [to_tool_declaration(tool) for tool in added],
    )


def _plan_sections(shown: Mapping[str, str], desired: Mapping[str, str]) -> list[dict[str, str | None]]:
    """Section patches: none, the minimal patch, or a remove-all/re-add-all pair when the order would differ."""
    patched_order = [*(key for key in shown if key in desired), *(key for key in desired if key not in shown)]
    desired_order = list(desired)
    if patched_order != desired_order:
        return [dict.fromkeys(shown, None), dict(desired)]
    patch: dict[str, str | None] = {}
    for key, value in shown.items():
        nxt = desired.get(key)
        if nxt != value:
            patch[key] = nxt
    for key, value in desired.items():
        if key not in shown:
            patch[key] = value
    return [] if len(patch) == 0 else [patch]


def _system_entry(
    sections: dict[str, str | None] | None,
    tools: tuple[list[ToolReference], list[Tool]] | None,
    timestamp: float,
) -> TypedEntryDraft[Any]:
    message = SystemMessage(
        "",
        sections=sections,
        tools_removed=None if tools is None or len(tools[0]) == 0 else tools[0],
        tools_added=None if tools is None or len(tools[1]) == 0 else tools[1],
        timestamp=timestamp,  # type: ignore[arg-type]
    )
    return {"model": [message]}
