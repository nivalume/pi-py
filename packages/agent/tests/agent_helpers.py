"""Shared test helpers (port of upstream ``test/utils/*.ts`` plus the factories the test files share)."""

from __future__ import annotations

import ast
import asyncio
import operator
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pi_agent import AgentTool, AgentToolResult
from pi_ai import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    Type,
    Usage,
    UserMessage,
    now_ms,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_usage() -> Usage:
    return Usage()


def create_model() -> Model:
    return Model(
        id="mock",
        name="mock",
        api="openai-responses",
        provider="openai",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        context_window=8192,
        max_tokens=2048,
    )


def create_assistant_message(
    content: str | list[TextContent | ThinkingContent | ToolCall], stop_reason: Any = "stop"
) -> AssistantMessage:
    """An assistant reply; a string becomes one text block."""
    blocks = [TextContent(content)] if isinstance(content, str) else list(content)
    return AssistantMessage(
        blocks,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason=stop_reason,
        timestamp=now_ms(),
    )


def create_assistant_tool_use_message(content: list[ToolCall]) -> AssistantMessage:
    return create_assistant_message(list(content), "toolUse")


def create_user_message(text: str) -> UserMessage:
    return UserMessage(text, timestamp=now_ms())


def identity_converter(messages: list[Any]) -> list[Message]:
    """Pass through the standard LLM messages only."""
    return [
        message
        for message in messages
        if getattr(message, "role", None) in ("system", "user", "assistant", "toolResult")
    ]


def user_texts(messages: list[Any]) -> list[str]:
    """String contents of the user messages (what upstream collects per request)."""
    return [
        message.content
        for message in messages
        if getattr(message, "role", None) == "user" and isinstance(message.content, str)
    ]


# ---------------------------------------------------------------------------
# Mock streams
# ---------------------------------------------------------------------------


def push_soon(stream: AssistantMessageEventStream, *events: AssistantMessageEvent) -> AssistantMessageEventStream:
    """Push ``events`` on the next loop iteration (the analog of ``queueMicrotask``)."""

    def push() -> None:
        for event in events:
            stream.push(event)

    asyncio.get_running_loop().call_soon(push)
    return stream


def terminal_event(message: AssistantMessage) -> AssistantMessageEvent:
    if message.stop_reason in ("error", "aborted"):
        return ErrorEvent(message.stop_reason, message)  # type: ignore[arg-type]
    return DoneEvent(message.stop_reason, message)  # type: ignore[arg-type]


def respond(message: AssistantMessage) -> AssistantMessageEventStream:
    """A stream that delivers ``message`` as its terminal event on the next loop iteration."""
    return push_soon(AssistantMessageEventStream(), terminal_event(message))


def respond_with(factory: Callable[[], AssistantMessage]) -> AssistantMessageEventStream:
    """Like :func:`respond`, building the message when the event is pushed."""
    stream = AssistantMessageEventStream()
    asyncio.get_running_loop().call_soon(lambda: stream.push(terminal_event(factory())))
    return stream


async def wait_for_tick() -> None:
    """Port of ``waitForTick`` (``setTimeout(resolve, 0)``)."""
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# calculate tool (test/utils/calculate.ts)
# ---------------------------------------------------------------------------

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def calculate(expression: str) -> AgentToolResult:
    """Evaluate an arithmetic expression (the upstream helper uses ``new Function``)."""
    try:
        value = _evaluate(ast.parse(expression, mode="eval"))
    except Exception as error:
        raise ValueError(str(error) or type(error).__name__) from error
    return AgentToolResult([TextContent(f"{expression} = {_format_number(value)}")], None)


calculate_schema = Type.Object({"expression": Type.String(description="The mathematical expression to evaluate")})


async def _execute_calculate(
    _tool_call_id: str, args: Any, _signal: Any = None, _on_update: Any = None
) -> AgentToolResult:
    return calculate(args["expression"])


calculate_tool = AgentTool(
    "calculate",
    "Evaluate mathematical expressions",
    calculate_schema,
    label="Calculator",
    execute=_execute_calculate,
)


def create_calculate_tool_with_usage(usage: Usage) -> AgentTool:
    async def execute(_tool_call_id: str, args: Any, _signal: Any = None, _on_update: Any = None) -> AgentToolResult:
        result = calculate(args["expression"])
        result.usage = usage
        return result

    return AgentTool(
        calculate_tool.name,
        calculate_tool.description,
        calculate_tool.parameters,
        label=calculate_tool.label,
        execute=execute,
    )


# ---------------------------------------------------------------------------
# get_current_time tool (test/utils/get-current-time.ts)
# ---------------------------------------------------------------------------


async def get_current_time(timezone: str | None = None) -> AgentToolResult:
    date = datetime.now(UTC)
    utc_timestamp = int(date.timestamp() * 1000)
    if timezone:
        try:
            zone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"Invalid timezone: {timezone}. Current UTC time: {date.isoformat()}") from error
        text = date.astimezone(zone).strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        return AgentToolResult([TextContent(text)], {"utc_timestamp": utc_timestamp})
    text = date.astimezone().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    return AgentToolResult([TextContent(text)], {"utc_timestamp": utc_timestamp})


get_current_time_schema = Type.Object(
    {
        "timezone": Type.Optional(
            Type.String(description="Optional timezone (e.g., 'America/New_York', 'Europe/London')")
        )
    }
)


async def _execute_get_current_time(
    _tool_call_id: str, args: Any, _signal: Any = None, _on_update: Any = None
) -> AgentToolResult:
    return await get_current_time(args.get("timezone"))


get_current_time_tool = AgentTool(
    "get_current_time",
    "Get the current date and time",
    get_current_time_schema,
    label="Current Time",
    execute=_execute_get_current_time,
)
