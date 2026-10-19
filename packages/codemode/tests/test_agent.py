"""The README's "Using with pi-agent-core", as :func:`pi_codemode.create_codemode_tool`, driven by pi_ai's faux provider."""

from __future__ import annotations

from typing import Any

import pytest

from pi_agent import Agent, AgentInitialState, AgentTool, AgentToolResult
from pi_ai import (
    TextContent,
    ToolResultMessage,
    Type,
    create_models,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_tool_call,
)
from pi_codemode import CodemodeSandbox, create_codemode_tool

PNG = "iVBORw0KGgo="


def read_tool(reads: list[Any]) -> AgentTool:
    async def execute(_id: str, params: Any, _signal: Any = None, _on_update: Any = None) -> AgentToolResult:
        reads.append(params)
        return AgentToolResult([TextContent(f"[project]\nname = {params['path']!r}")])

    return AgentTool(
        "read",
        "Read a file",
        Type.Object({"path": Type.String(description="Path to read")}),
        execute=execute,
    )


def count_tool() -> AgentTool:
    async def execute(_id: str, params: Any, _signal: Any = None, _on_update: Any = None) -> AgentToolResult:
        if params["n"] < 0:
            return AgentToolResult([TextContent("negative")], is_error=True)
        return AgentToolResult([TextContent("counted")], structured_content={"count": params["n"] * 2})

    return AgentTool(
        "count",
        "Count things",
        Type.Object({"n": Type.Integer()}),
        execute=execute,
        output_schema=Type.Object({"count": Type.Integer()}),
    )


async def run_agent(tool: AgentTool, script: str) -> ToolResultMessage:
    models = create_models()
    faux = faux_provider()
    models.set_provider(faux.provider)
    faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("codemode", {"code": script}, id="call-1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    agent = Agent(initial_state=AgentInitialState(model=faux.get_model(), tools=[tool]), stream_fn=models.stream_simple)
    await agent.prompt("Read the project file")
    results = [message for message in agent.state.messages if isinstance(message, ToolResultMessage)]
    assert len(results) == 1
    return results[0]


async def test_a_model_runs_a_script_that_calls_the_agents_tools() -> None:
    reads: list[Any] = []
    tool = create_codemode_tool([read_tool(reads), count_tool()])
    assert tool.name == "codemode"
    assert tool.description.endswith(
        "\n".join(
            [
                "class tools:",
                "    async def read(",
                "        self,",
                "        *,",
                "        # Path to read",
                "        path: str,",
                "    ) -> str:",
                '        """Read a file"""',
                '    async def count(self, *, n: int) -> TypedDict[{"count": int}]:',
                '        """Count things"""',
            ]
        )
    )

    result = await run_agent(
        tool,
        '# @options: {"timeout_ms": 30000}\n'
        'source = await tools.read(path="pyproject.toml")\n'
        'text("lines " + str(len(source.splitlines())))\n'
        f'image("data:image/png;base64,{PNG}")\n'
        "counted = await tools.count(n=21)\n"
        "return [source.splitlines()[0], counted]",
    )
    assert result.is_error is False
    assert [type(block).__name__ for block in result.content] == ["TextContent", "ImageContent", "TextContent"]
    assert getattr(result.content[0], "text", None) == "lines 2"
    assert getattr(result.content[2], "text", None) == '["[project]", {"count": 42}]'
    assert reads == [{"path": "pyproject.toml"}]


async def test_failed_scripts_and_nested_calls_are_errors_the_model_sees() -> None:
    tool = create_codemode_tool([read_tool([]), count_tool()])
    failed = await run_agent(
        tool,
        "try:\n    await tools.count(n=-1)\nexcept Exception as error:\n    text('caught ' + str(error))\n"
        "try:\n    await tools.read()\nexcept Exception as error:\n    text('invalid: ' + str(error).splitlines()[0])\n"
        "raise ValueError('boom')",
    )
    assert failed.is_error is True
    texts = [getattr(block, "text", "") for block in failed.content]
    assert texts[0] == "caught negative"
    assert texts[1] == 'invalid: Validation failed for tool "read":'
    assert texts[2].startswith("Traceback (most recent call last):")
    assert texts[2].endswith("ValueError: boom")


async def test_invalid_source_is_reported_without_running_anything() -> None:
    result = await run_agent(create_codemode_tool([]), "text(1)\n# @options: {}")
    assert result.is_error is True
    assert "must be the first line" in getattr(result.content[0], "text", "")


async def test_a_shared_pool_serves_every_call() -> None:
    pytest.importorskip("pydantic_monty")
    from pydantic_monty import AsyncMonty

    async with AsyncMonty(max_processes=1) as pool:
        made: list[CodemodeSandbox] = []

        def factory(tools: Any) -> CodemodeSandbox:
            sandbox = CodemodeSandbox(tools=tools, pool=pool)
            made.append(sandbox)
            return sandbox

        tool = create_codemode_tool([read_tool([])], sandbox=factory)
        result = await run_agent(tool, "return (await tools.read(path='a')).splitlines()[1]")
        assert getattr(result.content[0], "text", None) == "\"name = 'a'\""
        assert len(made) == 1 and made[0].closed
        async with pool.checkout() as session:
            assert await session.feed_run("1") == 1
