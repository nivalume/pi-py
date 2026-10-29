"""What the model reads about code mode (port of ``codemode/describe.ts``).

The tool's fixed description (the script environment and its globals), and the ``codemode`` prompt section, which
lists the tools scripts can call with their declarations (``pi_codemode.render_tool_sample``), grouped by namespace
and limited to a token budget. Tools that do not fit are found with ``searchTools()``. The section renders from the
conversation's callable tools, so it changes with them; the tool's description never does.

Both follow the sandbox's language: upstream's text and TypeScript declarations for JavaScript scripts, the same
environment described for Python with Python declarations (see :mod:`.dialect`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict

from pi_ai.utils.schema import parameters_schema
from pi_codemode.declarations import render_tool_sample
from pi_codemode.identifier import to_codemode_identifier
from pi_codemode.types import CodemodeToolDeclaration

from .dialect import JAVASCRIPT, Dialect

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

    from ..harness.types import ToolRegistration

__all__ = [
    "DEFAULT_CODEMODE_INLINE_BUDGET",
    "CodemodeNamespace",
    "codemode_description",
    "codemode_section",
    "declaration_of",
    "namespace_of",
    "sample_of",
]


class CodemodeNamespace(TypedDict):
    """A group of tools named by a common prefix, such as an MCP server's ``mcp__github__``."""

    name: ReadOnly[str]
    description: ReadOnly[NotRequired[str]]
    instructions: ReadOnly[NotRequired[str]]
    #: Tool names starting with it belong to the namespace; the longest matching prefix wins.
    prefix: ReadOnly[str]


#: Default token budget of the tool declarations in the ``codemode`` section, as in the coding agent.
DEFAULT_CODEMODE_INLINE_BUDGET: Final = 3000
#: Characters per token when estimating the cost of a tool section.
_CHARS_PER_TOKEN: Final = 4

_TEXT_OUTPUT_SCHEMA: Final = {"type": "string"}


def codemode_description(models: bool, dialect: Dialect = JAVASCRIPT) -> str:
    """The code mode tool's description: the script environment and its globals, the same for every conversation."""
    return "\n\n".join(
        [
            dialect.description_intro,
            "\n".join(["Globals:", *dialect.globals_lines]),
            *([dialect.models_description] if models else []),
            "The `codemode` section of the system prompt lists the tools scripts can call, by namespace, as far as "
            "space allows.",
        ]
    )


def declaration_of(tool: ToolRegistration) -> CodemodeToolDeclaration:
    """What a script sees of a tool: its declaration. Tools without an output schema return their text output."""
    return CodemodeToolDeclaration(
        name=tool.name,
        description=tool.description,
        input_schema=parameters_schema(tool.parameters),
        output_schema=tool.structured_output_schema
        if tool.structured_output_schema is not None
        else _TEXT_OUTPUT_SCHEMA,
    )


def sample_of(tool: ToolRegistration, dialect: Dialect = JAVASCRIPT) -> str:
    """``toolSample`` of a tool: its description and declaration (TypeScript, or Python for Python scripts)."""
    return render_tool_sample(declaration_of(tool), language=dialect.language)


def namespace_of(name: str, namespaces: Sequence[CodemodeNamespace]) -> CodemodeNamespace | None:
    """The namespace of a tool: the one with the longest prefix its name starts with."""
    found: CodemodeNamespace | None = None
    for namespace in namespaces:
        if name.startswith(namespace["prefix"]) and (found is None or len(namespace["prefix"]) > len(found["prefix"])):
            found = namespace
    return found


def _render_tool_section(tool: ToolRegistration, dialect: Dialect) -> str:
    """``### `id` (`raw name`)`` followed by the tool's description and declaration."""
    id = to_codemode_identifier(tool.name, dialect.language)
    heading = f"### `{id}`" if id == tool.name else f"### `{id}` (`{tool.name}`)"
    return f"{heading}\n{sample_of(tool, dialect).strip()}"


@dataclass(slots=True)
class _CatalogEntry:
    name: str
    section: str
    cost: int


@dataclass(slots=True)
class _CatalogGroup:
    namespace: CodemodeNamespace | None
    entries: list[_CatalogEntry] = field(default_factory=lambda: [])


def _select_catalog(groups: Sequence[_CatalogGroup], budget: float) -> set[str]:
    """Pick the tool sections that fit the budget: in each round every group (tools without a namespace first, then
    namespaces by name) places its cheapest remaining tool; a group whose next tool does not fit drops out while the
    others continue. Every namespace is represented before any namespace is complete."""
    # A stable sort by cost, as upstream's.
    queues = [sorted(group.entries, key=lambda entry: entry.cost) for group in groups]
    shown: set[str] = set()
    remaining = budget
    active = [queue for queue in queues if len(queue) > 0]
    while len(active) > 0:
        kept: list[list[_CatalogEntry]] = []
        for queue in active:
            following = queue[0]
            if following.cost > remaining:
                continue
            remaining -= following.cost
            shown.add(following.name)
            queue.pop(0)
            if len(queue) > 0:
                kept.append(queue)
        active = kept
    return shown


def _group_order(group: _CatalogGroup) -> tuple[int, str]:
    # Tools without a namespace first, then namespaces by name (upstream's `localeCompare`; names are ASCII prefixes
    # in practice, compared case-insensitively first like a locale collation).
    if group.namespace is None:
        return (0, "")
    return (1, group.namespace["name"].casefold() + "\0" + group.namespace["name"])


def codemode_section(
    tools: Sequence[ToolRegistration],
    namespaces: Sequence[CodemodeNamespace],
    budget: float,
    dialect: Dialect = JAVASCRIPT,
) -> str | None:
    """The ``codemode`` section: the tools scripts can call, grouped by namespace, with their declarations as far as
    the budget allows. ``None`` without callable tools."""
    if len(tools) == 0:
        return None
    groups: dict[str, _CatalogGroup] = {"": _CatalogGroup(None)}
    for tool in tools:
        namespace = namespace_of(tool.name, namespaces)
        key = f"ns:{namespace['name']}" if namespace is not None else ""
        group = groups.get(key)
        if group is None:
            group = _CatalogGroup(namespace)
            groups[key] = group
        section = _render_tool_section(tool, dialect)
        group.entries.append(_CatalogEntry(tool.name, section, math.ceil(len(section) / _CHARS_PER_TOKEN)))
    ordered = sorted(groups.values(), key=_group_order)
    shown = _select_catalog(ordered, budget)

    tool_sections = [dialect.section_header]
    for group in ordered:
        entries = group.entries
        if len(entries) == 0:
            continue
        visible = [entry for entry in entries if entry.name in shown]
        namespace = group.namespace
        if namespace is not None:
            listing = (
                ""
                if len(visible) == len(entries)
                else " (tools not listed)"
                if len(visible) == 0
                else " (some tools not listed)"
            )
            description = (namespace.get("description") or "").strip()
            tool_sections.append(f"## {namespace['name']}{listing}" + (f"\n{description}" if description else ""))
        elif len(visible) < len(entries):
            tool_sections.append(dialect.not_listed)
        tool_sections.extend(entry.section for entry in visible)
    return "\n\n".join(tool_sections)
