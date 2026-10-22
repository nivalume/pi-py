"""Port of ``test/codemode-describe.test.ts``: the code mode tool's description and the ``codemode`` section.

Both follow the sandbox's language. For JavaScript they are upstream's, with TypeScript declarations
(``read(args: {...}): Promise<string>``); for Python the declarations are Python (``async def read(...) -> str``) and
the description shows Python calls.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from durable_harness_support import all_entries, chat_setup, open_chat
from durable_session_support import context

from pi_ai import Type, faux_assistant_message, faux_text
from pi_codemode import CODEMODE_SOURCE_GRAMMAR, PYTHON_CODEMODE_SOURCE_GRAMMAR, CodemodeLanguage
from pi_durable import MemoryStorage, define_extension, define_tool
from pi_durable.codemode import CodemodeNamespace, create_codemode_extension
from pi_durable.codemode.describe import codemode_description, codemode_section
from pi_durable.codemode.dialect import JAVASCRIPT, PYTHON


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
    assert text.startswith("Run JavaScript that calls other tools.")
    assert "`await tools.<name>({ ...args })`" in text
    assert "`await searchTools(query, { limit?, namespace? })`" in text
    assert '`// @options: {"max_output_tokens": 10000, "timeout_ms": 60000}`' in text
    assert "The `codemode` section of the system prompt lists the tools scripts can call" in text
    assert "`models`" not in text
    assert "`models.generateImages(model, " in codemode_description(True)


def test_describes_python_scripts_for_a_python_sandbox() -> None:
    text = codemode_description(False, PYTHON)
    assert text.startswith("Run Python that calls other tools.")
    assert "`await tools.<name>(key=value, ...)`" in text
    assert "search_tools(query" in text
    assert '`# @options: {"max_output_tokens": 10000, "timeout_ms": 60000}`' in text
    assert "`models`" not in text
    assert "`models.generate_images(model, " in codemode_description(True, PYTHON)


def test_lists_callable_tools_as_declarations_grouped_by_namespace_with_namespace_descriptions() -> None:
    tools = [tool("read", "Read a file."), tool("mcp__github__list_issues", "List issues.")]
    section = codemode_section(tools, [GITHUB], 3000)
    assert section is not None
    assert section.startswith("Tools scripts can call (`await tools.<name>(args)`):")
    assert "### `read`\nRead a file." in section
    assert "read(args: {\n  // File path\n  path: string;\n}): Promise<string>;" in section
    assert "## github\nGitHub MCP server" in section
    # Tools before namespaces; a namespace's tools after its heading.
    assert section.index("### `read`") < section.index("## github")
    assert section.index("## github") < section.index("### `mcp__github__list_issues`")

    python = codemode_section(tools, [GITHUB], 3000, PYTHON)
    assert python is not None
    assert python.startswith("Tools scripts can call (`await tools.<name>(...)`):")
    assert "async def read(" in python
    assert ") -> str: ..." in python
    assert codemode_section(tools, [GITHUB], 3000, JAVASCRIPT) == section


def test_names_the_identifier_of_each_language_in_headings() -> None:
    section = codemode_section([tool("class", "A keyword."), tool("get-$x", "Dashes.")], [], 3000) or ""
    assert "### `class`\n" in section
    assert "### `get_$x` (`get-$x`)" in section
    python = codemode_section([tool("class", "A keyword."), tool("get-$x", "Dashes.")], [], 3000, PYTHON) or ""
    assert "### `class_` (`class`)" in python
    assert "### `get__x` (`get-$x`)" in python


def test_lists_what_fits_the_budget_every_namespace_before_any_is_complete_and_says_what_is_left_out() -> None:
    github = [tool(f"mcp__github__tool{index}", "x" * 200) for index in range(30)]
    section = codemode_section([tool("read", "Read a file."), *github], [GITHUB], 400)
    assert section is not None
    assert "### `read`" in section
    assert "## github (some tools not listed)" in section
    assert "tool29" not in section
    assert "## github (tools not listed)" in (codemode_section([*github], [GITHUB], 10) or "")
    assert codemode_section([], [GITHUB], 3000) is None
    unlisted = [tool(f"tool{index}", "x" * 200) for index in range(30)]
    assert "Some tools are not listed: find them with `searchTools()`." in (codemode_section(unlisted, [], 400) or "")
    assert "Some tools are not listed: find them with `search_tools()`." in (
        codemode_section(unlisted, [], 400, PYTHON) or ""
    )


def test_uses_the_longest_matching_namespace_prefix() -> None:
    section = codemode_section([tool("mcp__github__x", "x")], [{"name": "git", "prefix": "mcp__git"}, GITHUB], 3000)
    assert section is not None
    assert "## github" in section
    assert "## git\n" not in section


def _sandbox(language: CodemodeLanguage) -> Any:
    if language == "javascript":
        from pi_codemode import CodemodeSandbox

        return CodemodeSandbox()
    if importlib.util.find_spec("pydantic_monty") is None:
        pytest.skip("needs Monty (the codemode-python extra)")
    from pi_codemode import MontyCodemodeSandbox

    return MontyCodemodeSandbox()


@pytest.mark.parametrize(
    ("language", "declaration", "grammar", "search"),
    [
        ("javascript", "read(args: {", CODEMODE_SOURCE_GRAMMAR, "searchTools("),
        ("python", "async def read(", PYTHON_CODEMODE_SOURCE_GRAMMAR, "search_tools("),
    ],
)
async def test_renders_the_section_from_the_conversations_callable_tools_into_the_system_prompt(
    language: CodemodeLanguage, declaration: str, grammar: str, search: str
) -> None:
    setup = chat_setup()
    sandbox = _sandbox(language)
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
    assert declaration in sections["codemode"]
    assert "### `mcp__github__list_issues`" in sections["codemode"]
    # Deferred tools are found with searchTools() (search_tools()), not listed.
    assert "mcp__github__get_issue" not in sections["codemode"]
    # The code mode tool itself is not callable from scripts.
    assert "### `codemode`" not in sections["codemode"]
    # The tool declaration carries the script grammar; pi-ai applies it only for models that support it.
    tools = system["model"][0].tools_added
    codemode = next(candidate for candidate in tools if candidate.name == "codemode")
    assert codemode.constrained_sampling == {"type": "grammar", "variants": {"openai_lark": grammar}}
    assert search in codemode.description
    # Only one language is offered: the description never mentions the other one's globals.
    assert ("search_tools(" if language == "javascript" else "searchTools(") not in codemode.description
    await opened.harness.close(context)
    await sandbox.close()
