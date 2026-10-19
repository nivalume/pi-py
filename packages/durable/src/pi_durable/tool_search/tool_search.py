"""The ``tool_search`` tool, a copy of the coding agent's (port of ``tool-search/tool-search.ts``).

The model searches the deferred tools it is not offered yet, and the matches are loaded with
``control.addTools``, so the next request offers them. Loading writes the conversation's ``modelTools``
(its ``add``), so loaded tools stay offered on that branch, also after a compaction.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, TypedDict

from pi_ai import TextContent, Type
from pi_ai.utils.schema import parameters_schema

from ..harness.define import define_extension, define_tool
from ..search import SearchDocument, rank_documents

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..harness.types import Extension, ToolExecutionApi, ToolExecutionResult, ToolRegistration

__all__ = [
    "DEFAULT_TOOL_SEARCH_LIMIT",
    "TOOL_SEARCH_TOOL_NAME",
    "ToolSearchToolDetails",
    "create_tool_search_extension",
    "create_tool_search_tool",
]

TOOL_SEARCH_TOOL_NAME: Final = "tool_search"
DEFAULT_TOOL_SEARCH_LIMIT: Final = 8

#: Fixed, so it does not change while deferred tools are registered, for example as MCP servers connect.
_TOOL_SEARCH_DESCRIPTION: Final = (
    "# Tool discovery\n\nSearches over deferred tool metadata with BM25 and exposes matching tools for the next "
    "model call.\n\nSome of the tools, such as tools of MCP servers, may not have been provided to you upfront, and "
    f"you should use this tool (`{TOOL_SEARCH_TOOL_NAME}`) to search for the required tools. For MCP tool "
    f"discovery, always use `{TOOL_SEARCH_TOOL_NAME}`."
)

_TOOL_SEARCH_SCHEMA = Type.Object(
    {
        "query": Type.String(description="Search query for deferred tools."),
        "limit": Type.Optional(
            Type.Number(description=f"Maximum number of tools to return. Defaults to {DEFAULT_TOOL_SEARCH_LIMIT}.")
        ),
    }
)

_LINE_BREAK = re.compile(r"\r?\n")


class ToolSearchToolDetails(TypedDict):
    """``tool_search`` result details: the tools it loaded."""

    loaded: list[str]


def _schema_text(schema: Any, parts: list[str]) -> None:
    """Schema descriptions and property names, recursively."""
    if not isinstance(schema, Mapping):
        return
    node: Mapping[str, Any] = schema  # pyright: ignore[reportUnknownVariableType]
    description = node.get("description")
    if isinstance(description, str):
        parts.append(description)
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        for name, property in properties.items():  # pyright: ignore[reportUnknownVariableType]
            parts.append(str(name))  # pyright: ignore[reportUnknownArgumentType]
            _schema_text(property, parts)
    _schema_text(node.get("items"), parts)
    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for variant in variants:  # pyright: ignore[reportUnknownVariableType]
                _schema_text(variant, parts)


def _search_text(tool: ToolRegistration) -> str:
    """Search text of a tool: its name, the name with ``_`` as spaces, its description, and its parameters' text."""
    parts = [tool.name, tool.name.replace("_", " "), tool.description]
    _schema_text(parameters_schema(tool.parameters), parts)
    return " ".join(part for part in parts if part.strip() != "")


def _is_positive_integer(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return (isinstance(value, int) or value.is_integer()) and value > 0


def create_tool_search_tool() -> ToolRegistration:
    """``tool_search``: called by the model only, safe to replay."""

    async def execute(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        query: str = args["query"]
        limit: Any = args.get("limit")
        if query.strip() == "":
            raise RuntimeError("query must not be empty")
        maximum = DEFAULT_TOOL_SEARCH_LIMIT if limit is None else limit
        if not _is_positive_integer(maximum):
            raise RuntimeError("limit must be a positive integer")
        candidates = (await api.agent(context))["deferred"]
        documents: list[SearchDocument] = [{"name": tool.name, "text": _search_text(tool)} for tool in candidates]
        loaded = rank_documents(query, documents, int(maximum))
        if len(loaded) == 0:
            text = "No matching tools found."
        else:
            lines: list[str] = []
            for name in loaded:
                description = next((tool.description for tool in candidates if tool.name == name), "")
                lines.append(f"- {name}: {_LINE_BREAK.split(description.strip())[0]}")
            plural = "" if len(loaded) == 1 else "s"
            text = f"Loaded {len(loaded)} tool{plural}. They are available from your next call:\n" + "\n".join(lines)
        details: ToolSearchToolDetails = {"loaded": loaded}
        if len(loaded) == 0:
            return {"output": [TextContent(text)], "details": details}
        return {"output": [TextContent(text)], "details": details, "control": {"addTools": loaded}}

    return define_tool(
        name=TOOL_SEARCH_TOOL_NAME,
        description=_TOOL_SEARCH_DESCRIPTION,
        parameters=_TOOL_SEARCH_SCHEMA,
        # Searching changes what the model is offered; scripts have their own `searchTools()`.
        callers=["model"],
        replay="safe",
        execute=execute,
    )


def create_tool_search_extension() -> Extension:
    """The ``tool_search`` extension: install it and select it where the model should find deferred tools."""
    return define_extension("tool-search", tools=[create_tool_search_tool()])
