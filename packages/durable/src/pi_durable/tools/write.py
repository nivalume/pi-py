"""The ``write`` tool (port of ``tools/write.ts``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from pi_ai import TextContent, Type

from ..env.types import get_or_throw
from ..harness.define import define_tool
from .env import require_env
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import path_subject, resolve_tool_path

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..harness.types import ToolExecutionApi, ToolExecutionResult, ToolRegistration

__all__ = ["WriteToolInput", "create_write_tool"]

_WRITE_SCHEMA = Type.Object(
    {
        "path": Type.String(description="Path to the file to write (relative or absolute)"),
        "content": Type.String(description="Content to write to the file"),
    }
)


class WriteToolInput(TypedDict):
    path: str
    content: str


def _check_aborted(context: Context) -> None:
    signal = context.abort_signal
    if signal is not None and signal.aborted:
        raise RuntimeError("Operation aborted")


def create_write_tool() -> ToolRegistration:
    """Writes a file, creating it and its parent directories, or overwriting it."""

    async def execute(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        path: str = args["path"]
        content: str = args["content"]
        env = require_env(api)
        absolute_path = await resolve_tool_path(env, path, context)

        async def mutate() -> ToolExecutionResult:
            _check_aborted(context)
            get_or_throw(await env.write_file(absolute_path, content, context))
            _check_aborted(context)
            return {"output": [TextContent(f"Successfully wrote to {path}")]}

        return await with_file_mutation_queue(env, absolute_path, mutate, context)

    return define_tool(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically "
            "creates parent directories."
        ),
        parameters=_WRITE_SCHEMA,
        execute=execute,
        permission_subject=path_subject("write"),
    )
