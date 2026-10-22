"""A ``codemode`` tool for pi_agent (the README's "Using with pi-agent-core", as code).

The agent's other tools become the sandbox's tools; the model calls one ``codemode`` tool
with a script that calls them as ``await tools.<name>(...)``: JavaScript by default, or Python
with ``language="python"``. Nested calls never enter the model's context, only the script's
output and return value do::

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
from typing import Any, Final

from pi_agent import AgentTool, AgentToolResult
from pi_ai import ImageContent, TextContent, ToolCall, Type
from pi_ai.abort import AbortSignal
from pi_ai.utils.schema import parameters_schema
from pi_ai.utils.validation import validate_tool_arguments
from pi_chord.json import JsonValue
from pi_telemetry import TelemetryContext

from .declarations import render_declarations
from .language import CodemodeLanguage, check_codemode_language
from .runtime.execution import CodemodeSandboxBase
from .source import CodemodeSourceError, parse_codemode_source
from .types import CodemodeFailure, CodemodeTool, CodemodeToolContext

__all__ = ["codemode_tool_of", "create_codemode_tool"]

#: Schema-less tools resolve to their text output, as their declaration says.
_TEXT_OUTPUT_SCHEMA = {"type": "string"}

SandboxFactory = Callable[[list[CodemodeTool]], CodemodeSandboxBase]


def codemode_tool_of(tool: AgentTool, language: CodemodeLanguage = "javascript") -> CodemodeTool:
    """An agent tool as a sandbox tool for scripts in ``language``.

    A tool with an ``output_schema`` resolves to its ``structured_content``; any other tool to
    its text content joined by newlines. A result with ``is_error`` raises in the script.
    """
    takes = (
        "one object of arguments"
        if check_codemode_language(language) == "javascript"
        else "keyword arguments or one dict"
    )

    async def execute(args: JsonValue, context: CodemodeToolContext) -> Any:
        arguments = {} if args is None else args
        if not isinstance(arguments, dict):
            raise TypeError(f"tools.{tool.name}() takes {takes}, not {_kind_of(arguments, language)}")
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


def _kind_of(value: Any, language: CodemodeLanguage) -> str:
    if language == "python":
        return type(value).__name__
    if isinstance(value, list):
        return "an array"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, bool):
        return "a boolean"
    return "a number"


def _default_sandbox(language: CodemodeLanguage) -> SandboxFactory:
    if language == "javascript":
        from .sandbox import CodemodeSandbox

        return lambda tools: CodemodeSandbox(tools=tools)
    from .monty import MontyCodemodeSandbox

    return lambda tools: MontyCodemodeSandbox(tools=tools)


#: The tool's first line and its ``code`` parameter, by language.
_INTRO: Final = {
    "javascript": (
        "Run JavaScript that calls tools as `await tools.<name>(args)`. Output with text() or return.",
        "JavaScript source.",
    ),
    "python": (
        "Run Python that calls tools as `await tools.<name>(...)`. Output with text() or return.",
        "Python source.",
    ),
}


def create_codemode_tool(
    tools: Sequence[AgentTool],
    *,
    language: CodemodeLanguage = "javascript",
    sandbox: SandboxFactory | None = None,
    name: str = "codemode",
    label: str = "Codemode",
    telemetry_context: TelemetryContext | None = None,
) -> AgentTool:
    """A ``codemode`` agent tool whose scripts, in ``language``, call ``tools``.

    Each call runs in a fresh sandbox from ``sandbox(tools)``, closed when the script ends. The
    default is a :class:`~pi_codemode.CodemodeSandbox` for JavaScript and a
    :class:`~pi_codemode.MontyCodemodeSandbox` for Python; for Python, pass a factory that shares
    an ``AsyncMonty`` pool (``MontyCodemodeSandbox(tools=tools, pool=pool)``) to avoid starting
    one per call. A factory must make sandboxes of ``language``. The script may start with an
    options line (``// @options: {"timeout_ms": ...}``, in Python ``# @options: ...``).
    ``telemetry_context`` is the parent of each script's ``pi.codemode.run`` span.
    """
    language = check_codemode_language(language)
    sandbox_tools = [codemode_tool_of(tool, language) for tool in tools]
    factory = sandbox or _default_sandbox(language)
    intro, code_description = _INTRO[language]
    description = intro + "\n\n" + render_declarations(tools=sandbox_tools, language=language)
    separators = (",", ":") if language == "javascript" else None

    async def execute(
        tool_call_id: str, params: Any, signal: AbortSignal | None = None, on_update: Any = None
    ) -> AgentToolResult:
        try:
            source = parse_codemode_source(params["code"], language)
        except CodemodeSourceError as error:
            return AgentToolResult([TextContent(str(error))], is_error=True)
        runner = factory(sandbox_tools)
        if runner.language != language:
            await runner.close()
            raise ValueError(f"The sandbox runs {runner.language} scripts, but the codemode tool is for {language}")
        try:
            result = await runner.execute(
                source.code,
                signal=signal,
                timeout_ms=source.options.timeout_ms,
                telemetry_context=telemetry_context,
            )
        finally:
            await runner.close()
        content: list[TextContent | ImageContent] = [item.to_content() for item in result.output]
        if isinstance(result, CodemodeFailure):
            content.append(TextContent(result.error.stack or result.error.message))
        elif result.value is not None:
            content.append(TextContent(json.dumps(result.value, ensure_ascii=False, separators=separators)))
        return AgentToolResult(content, is_error=not result.ok)

    return AgentTool(
        name=name,
        description=description,
        parameters=Type.Object({"code": Type.String(description=code_description)}),
        label=label,
        execute=execute,
    )
