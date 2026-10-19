"""A ``codemode`` tool for pi_agent (the README's "Using with pi-agent-core", as code).

The agent's other tools become the sandbox's tools; the model calls one ``codemode`` tool
with a Python script that calls them as ``await tools.<name>(...)``. Nested calls never
enter the model's context, only the script's output and return value do::

    agent = Agent(
        initial_state={"model": model, "tools": [create_codemode_tool(tools)]},
        stream_fn=models.stream_simple,
    )

Nested calls run ``AgentTool.execute()`` directly, after validating their arguments against the
tool's schema, so the agent's ``before_tool_call`` and ``after_tool_call`` hooks do not see them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from pi_agent import AgentTool, AgentToolResult
from pi_ai import ImageContent, TextContent, ToolCall, Type
from pi_ai.abort import AbortSignal
from pi_ai.utils.schema import parameters_schema
from pi_ai.utils.validation import validate_tool_arguments
from pi_chord.json import JsonValue

from .declarations import render_declarations
from .runtime.execution import CodemodeSandboxBase
from .source import CodemodeSourceError, parse_codemode_source
from .types import CodemodeFailure, CodemodeTool, CodemodeToolContext

__all__ = ["codemode_tool_of", "create_codemode_tool"]

#: Schema-less tools resolve to their text output, as their declaration says.
_TEXT_OUTPUT_SCHEMA = {"type": "string"}

SandboxFactory = Callable[[list[CodemodeTool]], CodemodeSandboxBase]


def codemode_tool_of(tool: AgentTool) -> CodemodeTool:
    """An agent tool as a sandbox tool.

    A tool with an ``output_schema`` resolves to its ``structured_content``; any other tool to
    its text content joined by newlines. A result with ``is_error`` raises in the script.
    """

    async def execute(args: JsonValue, context: CodemodeToolContext) -> Any:
        arguments = {} if args is None else args
        if not isinstance(arguments, dict):
            raise TypeError(f"tools.{tool.name}() takes keyword arguments or one dict, not {type(arguments).__name__}")
        validated = validate_tool_arguments(tool, ToolCall("nested", tool.name, arguments))
        result = await tool.execute("nested", validated, context.signal)
        text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
        if result.is_error:
            raise RuntimeError(text or f"Tool {tool.name} failed")
        if tool.output_schema is not None and result.structured_content is not None:
            return result.structured_content
        return text

    return CodemodeTool(
        name=tool.name,
        execute=execute,
        description=tool.description,
        input_schema=parameters_schema(tool.parameters),
        output_schema=tool.output_schema if tool.output_schema is not None else _TEXT_OUTPUT_SCHEMA,
    )


def _default_sandbox(tools: list[CodemodeTool]) -> CodemodeSandboxBase:
    from .sandbox import CodemodeSandbox

    return CodemodeSandbox(tools=tools)


def create_codemode_tool(
    tools: Sequence[AgentTool],
    *,
    sandbox: SandboxFactory | None = None,
    name: str = "codemode",
    label: str = "Codemode",
) -> AgentTool:
    """A ``codemode`` agent tool whose scripts call ``tools``.

    Each call runs in a fresh sandbox from ``sandbox(tools)`` (default: a
    :class:`~pi_codemode.CodemodeSandbox`), closed when the script ends; pass a factory that
    shares an ``AsyncMonty`` pool (``CodemodeSandbox(tools=tools, pool=pool)``) to avoid starting
    one per call. The script may start with ``# @options: {"timeout_ms": ...}``.
    """
    sandbox_tools = [codemode_tool_of(tool) for tool in tools]
    factory = sandbox or _default_sandbox
    description = (
        "Run Python that calls tools as `await tools.<name>(...)`. Output with text() or return.\n\n"
        + render_declarations(tools=sandbox_tools)
    )

    async def execute(
        tool_call_id: str, params: Any, signal: AbortSignal | None = None, on_update: Any = None
    ) -> AgentToolResult:
        try:
            source = parse_codemode_source(params["code"])
        except CodemodeSourceError as error:
            return AgentToolResult([TextContent(str(error))], is_error=True)
        runner = factory(sandbox_tools)
        try:
            result = await runner.execute(source.code, signal=signal, timeout_ms=source.options.timeout_ms)
        finally:
            await runner.close()
        content: list[TextContent | ImageContent] = [item.to_content() for item in result.output]
        if isinstance(result, CodemodeFailure):
            content.append(TextContent(result.error.stack or result.error.message))
        elif result.value is not None:
            content.append(TextContent(json.dumps(result.value, ensure_ascii=False)))
        return AgentToolResult(content, is_error=not result.ok)

    return AgentTool(
        name=name,
        description=description,
        parameters=Type.Object({"code": Type.String(description="Python source.")}),
        label=label,
        execute=execute,
    )
