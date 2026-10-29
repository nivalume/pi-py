"""The ``bash`` and ``powershell`` tools (port of ``tools/bash.ts``)."""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, NotRequired, TypedDict

from pi_ai import Type

from ..env.types import ExecutionError, Result, ShellExecOptions, ShellExecResult, ShellOutputInfo
from ..harness.define import define_tool
from ..truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from ._js import js_number
from .env import require_env

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..approval import PermissionSubject
    from ..harness.types import ToolDiagnostic, ToolExecutionApi, ToolExecutionResult, ToolRegistration

__all__ = [
    "BashExecution",
    "BashPrepare",
    "BashToolInput",
    "PowerShellToolInput",
    "ShellStructuredOutput",
    "command_subject",
    "create_bash_tool",
    "create_powershell_tool",
]

_MAX_TIMEOUT_SECONDS: Final = 2_147_483_647 / 1000

_BASH_SCHEMA = Type.Object(
    {
        "command": Type.String(description="Bash command to execute"),
        "timeout": Type.Optional(Type.Number(description="Timeout in seconds (optional, no default timeout)")),
    }
)

_POWERSHELL_SCHEMA = Type.Object(
    {
        "command": Type.String(description="PowerShell command to execute"),
        "timeout": Type.Optional(Type.Number(description="Timeout in seconds (optional, no default timeout)")),
    }
)

#: What a program that calls ``bash`` or ``powershell`` receives.
_SHELL_STRUCTURED_OUTPUT_SCHEMA = Type.Object(
    {
        "output": Type.String(description="Combined stdout and stderr: the retained tail, as the model sees it"),
        "truncated": Type.Boolean(description="Whether earlier output was dropped"),
        "fullOutputPath": Type.Optional(Type.String(description="File with the complete output, when it was spilled")),
        "exitCode": Type.Number(),
    }
)


class BashToolInput(TypedDict):
    command: str
    timeout: NotRequired[float]


class PowerShellToolInput(TypedDict):
    command: str
    timeout: NotRequired[float]


class ShellStructuredOutput(TypedDict):
    output: str
    truncated: bool
    fullOutputPath: NotRequired[str]
    exitCode: int


class BashExecution(TypedDict):
    """A command about to run, which ``prepare`` may change: the script, its working directory and environment."""

    command: str
    cwd: str
    env: dict[str, str]
    inheritEnv: bool


#: ``prepare(execution, api, context)``: change the execution in place; may be sync or async.
BashPrepare = Callable[[BashExecution, "ToolExecutionApi", "Context"], "Awaitable[None] | None"]


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("Invalid timeout: must be a finite number of seconds")
    if timeout > _MAX_TIMEOUT_SECONDS:
        raise RuntimeError(f"Invalid timeout: maximum is {js_number(_MAX_TIMEOUT_SECONDS)} seconds")


async def _prepare_execution(
    command: str, command_prefix: str | None, prepare: BashPrepare | None, api: ToolExecutionApi, context: Context
) -> BashExecution:
    """The execution of one call: the command with its prefix, in the environment's working directory, then ``prepare``."""
    execution: BashExecution = {
        "command": f"{command_prefix}\n{command}" if command_prefix else command,
        "cwd": require_env(api).cwd,
        "env": {},
        "inheritEnv": True,
    }
    if prepare is not None:
        prepared = prepare(execution, api, context)
        if inspect.isawaitable(prepared):
            await prepared
    return execution


async def _run_command(
    commands: Sequence[str | Sequence[str]],
    execution: BashExecution,
    timeout: float | None,  # noqa: ASYNC109 - seconds for the environment, as upstream
    api: ToolExecutionApi,
    context: Context,
) -> ToolExecutionResult:
    """Run each command form in turn until one starts, streaming output to ``api.output()``.

    Then turn the result into the tool's result: a spill diagnostic, an error result for a nonzero exit,
    and a raised error for a failure such as a timeout. The structured output repeats the retained output
    with the exit code.
    """
    env = require_env(api)

    def on_output(text: str, _context: Context, info: ShellOutputInfo) -> None:
        api.output(text, info.get("skipped"))

    options: ShellExecOptions = {
        "cwd": execution["cwd"],
        "env": execution["env"],
        "inheritEnv": execution["inheritEnv"],
        "onOutput": on_output,
        "spill": {"afterBytes": DEFAULT_MAX_BYTES, "afterLines": DEFAULT_MAX_LINES},
    }
    if timeout is not None:
        options["timeout"] = timeout
    # An environment may then omit output outside the retained tail and report the omission.
    window = api.output_window
    if window is not None:
        options["window"] = window

    result: Result[ShellExecResult, ExecutionError] | None = None
    for command in commands:
        result = await env.exec(command, options, context)
        # A program that could not start produced no output; the next form may.
        if result["ok"] is True or result["error"].code != "spawn_error":
            break
    if result is None:
        raise RuntimeError("No command to run")
    spill_path = result["value"].get("spillPath") if result["ok"] is True else result["error"].spill_path
    if spill_path is not None:
        api.diagnostic({"severity": "info", "code": "full_output", "message": f"Full output: {spill_path}"})
    if result["ok"] is False:
        error = result["error"]
        signal = context.abort_signal
        if error.code == "aborted" and signal is not None and signal.aborted:
            raise error
        if error.code == "timeout":
            seconds = "undefined" if timeout is None else js_number(timeout)
            raise RuntimeError(f"Command timed out after {seconds} seconds")
        if error.code == "aborted":
            raise RuntimeError("Command aborted")
        raise error
    exit_code = result["value"]["exitCode"]
    retained = api.retained_output()
    structured_output: ShellStructuredOutput = (
        {"output": retained["text"], "truncated": retained["truncated"], "exitCode": exit_code}
        if spill_path is None
        else {
            "output": retained["text"],
            "truncated": retained["truncated"],
            "fullOutputPath": spill_path,
            "exitCode": exit_code,
        }
    )
    if exit_code == 0:
        return {"structuredOutput": structured_output}
    exited: ToolDiagnostic = {
        "severity": "error",
        "code": "exit_code",
        "message": f"Command exited with code {exit_code}",
    }
    return {"structuredOutput": structured_output, "isError": True, "diagnostics": [exited]}


def create_bash_tool(*, command_prefix: str | None = None, prepare: BashPrepare | None = None) -> ToolRegistration:
    """Runs a command through the environment's shell.

    Its output streams to ``api.output()``, where the Harness keeps the tail within the default limits; the
    result content is that retained output. The retained window goes to the environment, which may omit
    output outside it and report how much it omitted. Output beyond the limits is spilled to a file whose
    path is reported as a diagnostic. A nonzero exit is an error result with the exit code; a timeout
    raises, which makes an error result that still carries the output and diagnostics.

    ``command_prefix`` is run before each command (on its own line); ``prepare(execution, api, context)``
    may change the command, its working directory, and its environment.
    """

    async def execute(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        timeout: float | None = args.get("timeout")
        _validate_timeout(timeout)
        execution = await _prepare_execution(args["command"], command_prefix, prepare, api, context)
        return await _run_command([execution["command"]], execution, timeout, api, context)

    return define_tool(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns combined stdout and stderr. Output is "
            f"truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            "If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds."
        ),
        parameters=_BASH_SCHEMA,
        structured_output_schema=_SHELL_STRUCTURED_OUTPUT_SCHEMA,
        output_limits={"retain": "tail"},
        execute=execute,
        permission_subject=command_subject,
    )


def command_subject(args: dict[str, Any], _env: Any, _context: Context) -> PermissionSubject | None:
    """``permission_subject`` of ``bash`` and ``powershell``: the command, which ``Bash(...)`` rules match."""
    command = args.get("command")
    return {"type": "command", "command": command} if isinstance(command, str) else None


#: Output in UTF-8 whatever the console's code page, as the coding agent's ``powershell`` tool does.
_UTF8_OUTPUT: Final = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}"
_POWERSHELL_ARGS: Final = ("-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command")


def create_powershell_tool(
    *,
    command_prefix: str | None = None,
    prepare: BashPrepare | None = None,
    programs: Sequence[str] | None = None,
) -> ToolRegistration:
    """Runs a PowerShell command, like ``bash`` but through PowerShell instead of the environment's shell.

    ``pwsh`` (PowerShell 7), else Windows PowerShell, started directly with the command as an argument, so
    no other shell parses it. ``command_prefix`` lines run before each command; ``programs`` are the
    PowerShell programs to try in order (default ``pwsh``, then ``powershell``); one that cannot start is
    skipped.
    """
    candidates = tuple(programs) if programs is not None else ("pwsh", "powershell")

    async def execute(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        timeout: float | None = args.get("timeout")
        _validate_timeout(timeout)
        execution = await _prepare_execution(args["command"], command_prefix, prepare, api, context)
        script = f"{_UTF8_OUTPUT}\n{execution['command']}"
        commands = [[program, *_POWERSHELL_ARGS, script] for program in candidates]
        return await _run_command(commands, execution, timeout, api, context)

    return define_tool(
        name="powershell",
        description=(
            "Execute a PowerShell command in the current working directory. Returns combined stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is "
            "hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds."
        ),
        parameters=_POWERSHELL_SCHEMA,
        structured_output_schema=_SHELL_STRUCTURED_OUTPUT_SCHEMA,
        output_limits={"retain": "tail"},
        execute=execute,
        permission_subject=command_subject,
    )
