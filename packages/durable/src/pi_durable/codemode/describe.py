"""What the model reads about code mode (port of ``codemode/describe.ts``).

The tool's fixed description (the script environment and its globals), and the ``codemode`` prompt section, which
lists the tools scripts can call as Python declarations (``pi_codemode.render_tool_sample``), grouped by namespace
and limited to a token budget. Tools that do not fit are found with ``search_tools()``. The section renders from the
conversation's callable tools, so it changes with them; the tool's description never does.

Upstream's text describes JavaScript scripts; this one describes the same environment for Python scripts.
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

from .models import MODELS_DESCRIPTION

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

_DESCRIPTION_INTRO: Final = """Run Python that calls other tools. The input is raw Python (not JSON, no code fence), run as the body of an async function in a Monty sandbox: top-level `await` and `return` work. No file system, network, subprocesses, or host modules; standard modules such as `json`, `re`, `math`, `datetime`, and `asyncio` can be imported.
- `await tools.<name>(key=value, ...)` returns a string, or a dict if the tool's declaration says so, and raises an Exception with the error text on failure. A tool whose declaration returns a dict returns on failure too when it returned that dict; check its fields. Calls start when made, so `await asyncio.gather(...)` (after `import asyncio`) runs several at once. Calls still running when the script ends are cancelled.
- The script survives a restart: it resumes where it was, without running finished calls again.
- Optional first line: `# @options: {"max_output_tokens": 10000, "timeout_ms": 60000}`"""


def codemode_description(models: bool) -> str:
    """The code mode tool's description: the script environment and its globals, the same for every conversation."""
    globals_ = [
        "Globals:",
        "- `text(value)`, `image(data_url_or_image_block)`, `print(...)`, and top-level `return` add output; `exit()` "
        "ends the script. With several text items, each starts with a `==> text N/M <==` line, and `print` lines "
        "follow the other output in one `<console_output>` block. `image()` also saves the image to a temp file and "
        "the result names its path.",
        "- `store(key, value)` and `load(key)` keep JSON values across codemode calls.",
        "- `ALL_TOOLS`, `await search_tools(query, limit=8, namespace=None)`, `await describe_tool(name)`, "
        "`await describe_namespace(name)`: find tools the `codemode` section does not list.",
    ]
    return "\n\n".join(
        [
            _DESCRIPTION_INTRO,
            "\n".join(globals_),
            *([MODELS_DESCRIPTION] if models else []),
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


def sample_of(tool: ToolRegistration) -> str:
    """``toolSample`` of a tool: its description and Python declaration."""
    return render_tool_sample(declaration_of(tool))


def namespace_of(name: str, namespaces: Sequence[CodemodeNamespace]) -> CodemodeNamespace | None:
    """The namespace of a tool: the one with the longest prefix its name starts with."""
    found: CodemodeNamespace | None = None
    for namespace in namespaces:
        if name.startswith(namespace["prefix"]) and (found is None or len(namespace["prefix"]) > len(found["prefix"])):
            found = namespace
    return found


def _render_tool_section(tool: ToolRegistration) -> str:
    """``### `id` (`raw name`)`` followed by the tool's description and declaration."""
    id = to_codemode_identifier(tool.name)
    heading = f"### `{id}`" if id == tool.name else f"### `{id}` (`{tool.name}`)"
    return f"{heading}\n{sample_of(tool).strip()}"


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
    tools: Sequence[ToolRegistration], namespaces: Sequence[CodemodeNamespace], budget: float
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
        section = _render_tool_section(tool)
        group.entries.append(_CatalogEntry(tool.name, section, math.ceil(len(section) / _CHARS_PER_TOKEN)))
    ordered = sorted(groups.values(), key=_group_order)
    shown = _select_catalog(ordered, budget)

    tool_sections = ["Tools scripts can call (`await tools.<name>(...)`):"]
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
            tool_sections.append("Some tools are not listed: find them with `search_tools()`.")
        tool_sections.extend(entry.section for entry in visible)
    return "\n\n".join(tool_sections)
