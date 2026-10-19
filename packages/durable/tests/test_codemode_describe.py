"""Port of ``test/codemode-describe.test.ts``: the code mode tool's description and the ``codemode`` section.

Adaptation: the declarations are Python (``async def read(...) -> str``) where upstream's are TypeScript
(``read(args: {...}): Promise<string>``), and the description shows Python calls.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from durable_harness_support import all_entries, chat_setup, open_chat
from durable_session_support import context

from pi_ai import Type, faux_assistant_message, faux_text
from pi_codemode import CODEMODE_SOURCE_GRAMMAR
from pi_durable import MemoryStorage, define_extension, define_tool
from pi_durable.codemode import CodemodeNamespace, create_codemode_extension
from pi_durable.codemode.describe import codemode_description, codemode_section


async def _no_output(_args: Any, _api: Any, _context: Any) -> Any:
    return {"output": []}


def tool(name: str, description: str, **extra: Any) -> Any:
    return define_tool(
        name=name,
        description=description,
        parameters=Type.Object({"path": Type.String(description="File path")}),
        execute=_no_output,
        **extra,
    )


GITHUB: CodemodeNamespace = {"name": "github", "description": "GitHub MCP server", "prefix": "mcp__github__"}


def test_keeps_the_tool_description_the_same_for_every_conversation() -> None:
    text = codemode_description(False)
    assert "`await tools.<name>(key=value, ...)`" in text
    assert "search_tools(query" in text
    assert "The `codemode` section of the system prompt lists the tools scripts can call" in text
    assert "`models`" not in text
    assert "`models`" in codemode_description(True)


def test_lists_callable_tools_as_declarations_grouped_by_namespace_with_namespace_descriptions() -> None:
    section = codemode_section(
        [tool("read", "Read a file."), tool("mcp__github__list_issues", "List issues.")], [GITHUB], 3000
    )
    assert section is not None
    assert "### `read`\nRead a file." in section
    assert "async def read(" in section
    assert "## github\nGitHub MCP server" in section
    # Tools before namespaces; a namespace's tools after its heading.
    assert section.index("### `read`") < section.index("## github")
    assert section.index("## github") < section.index("### `mcp__github__list_issues`")
    assert ") -> str: ..." in section


def test_lists_what_fits_the_budget_every_namespace_before_any_is_complete_and_says_what_is_left_out() -> None:
    github = [tool(f"mcp__github__tool{index}", "x" * 200) for index in range(30)]
    section = codemode_section([tool("read", "Read a file."), *github], [GITHUB], 400)
    assert section is not None
    assert "### `read`" in section
    assert "## github (some tools not listed)" in section
    assert "tool29" not in section
    assert "## github (tools not listed)" in (codemode_section([*github], [GITHUB], 10) or "")
    assert codemode_section([], [GITHUB], 3000) is None


def test_uses_the_longest_matching_namespace_prefix() -> None:
    section = codemode_section([tool("mcp__github__x", "x")], [{"name": "git", "prefix": "mcp__git"}, GITHUB], 3000)
    assert section is not None
    assert "## github" in section
    assert "## git\n" not in section


@pytest.mark.skipif(importlib.util.find_spec("pydantic_monty") is None, reason="needs Monty (the codemode extra)")
async def test_renders_the_section_from_the_conversations_callable_tools_into_the_system_prompt() -> None:
    from pi_codemode import CodemodeSandbox

    setup = chat_setup()
    sandbox = CodemodeSandbox()
    setup.registry.install(create_codemode_extension(sandbox=sandbox, namespaces=[GITHUB]))
    setup.registry.install(
        define_extension(
            "tools",
            tools=[
                tool("read", "Read a file."),
                tool("mcp__github__list_issues", "List issues.", callers=["tools"]),
                tool("mcp__github__get_issue", "Get an issue.", deferred=True),
            ],
        )
    )
    setup.faux.set_responses([faux_assistant_message([faux_text("hi")])])
    opened = await open_chat(MemoryStorage(), setup)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    system = next(entry for entry in await all_entries(opened.root) if entry["kind"] == "pi.system")
    sections = system["model"][0].sections
    assert "### `read`" in sections["codemode"]
    assert "### `mcp__github__list_issues`" in sections["codemode"]
    # Deferred tools are found with search_tools(), not listed.
    assert "mcp__github__get_issue" not in sections["codemode"]
    # The code mode tool itself is not callable from scripts.
    assert "### `codemode`" not in sections["codemode"]
    # The tool declaration carries the script grammar; pi-ai applies it only for models that support it.
    tools = system["model"][0].tools_added
    codemode = next(candidate for candidate in tools if candidate.name == "codemode")
    assert codemode.constrained_sampling == {"type": "grammar", "variants": {"openai_lark": CODEMODE_SOURCE_GRAMMAR}}
    await opened.harness.close(context)
    await sandbox.close()
