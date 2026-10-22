"""Typed entry kinds and the built-in ``pi.*`` kinds (port of ``entries.ts``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from .types import Entry

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

    from .harness.types import CompactionReason, ToolDiagnostic

__all__ = [
    "AssistantEntry",
    "CompactionEntry",
    "ContextEntry",
    "CompactionEntryData",
    "ResetEntry",
    "SystemEntry",
    "ToolResultEntry",
    "ToolResultEntryData",
    "UserEntry",
    "define_entry",
]


def define_entry(kind: str) -> Entry[Any]:
    """Define a typed entry kind whose ``is_()`` guard narrows by ``EntryRecord["kind"]``.

    Annotate the token to type its ``data``: ``Notes: Entry[NoteData] = define_entry("app.note")``.
    """
    if not isinstance(kind, str) or len(kind) == 0:
        raise TypeError("Entry kind must be a non-empty string")
    return Entry(kind)


class ToolResultEntryData(TypedDict):
    """``data`` of a ``pi.tool-result`` entry."""

    diagnostics: ReadOnly[list[ToolDiagnostic]]


class CompactionEntryData(TypedDict):
    """``data`` of a ``pi.compaction`` entry."""

    reason: ReadOnly[CompactionReason]


#: User input: ``model`` is ``[UserMessage]``. Written by submissions.
UserEntry: Entry[Any] = define_entry("pi.user")
#: Provider result with any stop reason: ``model`` is ``[AssistantMessage]``. Written by generation.
AssistantEntry: Entry[Any] = define_entry("pi.assistant")
#: Positional prompt and tool change: ``model`` is ``[SystemMessage]`` with empty ``content``.
SystemEntry: Entry[Any] = define_entry("pi.system")
#: Tool result: ``model`` is ``[ToolResultMessage]``, whose content ends with the rendered diagnostics block;
#: ``data`` holds the structured diagnostics, possibly none. Written by tool tasks, and by generation for
#: calls it did not offer.
ToolResultEntry: Entry[ToolResultEntryData] = define_entry("pi.tool-result")
#: Start of a new context: always ``head: "self"``, with ``model`` absent for a plain reset or
#: ``[UserMessage]`` carrying the handoff text. Written by ``Conversation.reset()`` and the ``handoff`` control.
ResetEntry: Entry[Any] = define_entry("pi.reset")
#: Compaction summary: ``model`` is ``[UserMessage]`` with the wrapped summary, ``head`` the first kept entry.
CompactionEntry: Entry[CompactionEntryData] = define_entry("pi.compaction")
#: Context an ``on_input`` hook added for the model: ``model`` is ``[SystemMessage]`` whose ``content`` is the text.
#: Written by generation before the run's first request.
ContextEntry: Entry[Any] = define_entry("pi.context")
