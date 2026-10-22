"""Running an event's handlers: commands through the conversation's environment and HTTP posts, in parallel, with the
outcomes memoized in the hook's task."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, TypedDict

from pi_chord.context import Context, await_with_context

from ..errors import error_message
from .config import HookCommand, handler_timeout
from .protocol import HookOutput

if TYPE_CHECKING:
    import httpx

    from ..harness.types import HookApi

__all__ = ["Outcome", "OutcomeBlocked", "OutcomeFailed", "OutcomeOutput", "run_handlers"]


class OutcomeOutput(TypedDict):
    """Exit 0 or a 2xx response: the JSON output, or the plain text a command printed."""

    kind: Literal["output"]
    output: HookOutput
    text: NotRequired[str]


class OutcomeBlocked(TypedDict):
    """Exit 2: blocks with stderr as the reason."""

    kind: Literal["blocked"]
    reason: str


class OutcomeFailed(TypedDict):
    """Anything else: a non-blocking error."""

    kind: Literal["failed"]
    error: str


#: What one handler returned.
Outcome: TypeAlias = OutcomeOutput | OutcomeBlocked | OutcomeFailed


def memo_name(event: str) -> str:
    """The memo an event's outcomes are kept in, so a rerun of the phase does not run the handlers again."""
    return f"claude-hooks:{event}"


async def run_handlers(
    client: httpx.AsyncClient,
    api: HookApi,
    event: str,
    handlers: Sequence[HookCommand],
    input: Mapping[str, Any],
    context: Context,
) -> tuple[list[Outcome], bool]:
    """Run ``handlers`` of ``event`` with ``input``; their outcomes in handler order, and whether they ran now (a rerun
    returns the memoized ones)."""
    name = memo_name(event)
    memoized = await api.memo(name, context)
    if memoized is not None:
        return memoized, False
    if not handlers:
        return [], False
    body = json.dumps(input, ensure_ascii=False, separators=(",", ":"))
    runs = [_run_one(client, api, event, handler, body, context) for handler in handlers]
    outcomes = list(await asyncio.gather(*runs))
    return await api.memo(name, outcomes, context), True


async def _run_one(
    client: httpx.AsyncClient, api: HookApi, event: str, handler: HookCommand, body: str, context: Context
) -> Outcome:
    seconds = handler_timeout(handler, event)
    try:
        if handler["type"] == "command":
            return await _run_command(api, handler["command"], seconds, body, context)
        return await _run_http(client, handler, seconds, body, context)
    except Exception as error:
        signal = context.abort_signal
        if signal is not None and signal.aborted:
            raise
        return {"kind": "failed", "error": error_message(error)}


def parse_output(stdout: str) -> Outcome:
    """stdout as an output: a JSON object, or plain text; nothing for empty output."""
    trimmed = stdout.strip()
    if trimmed.startswith("{"):
        try:
            parsed = json.loads(trimmed)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return {"kind": "output", "output": parsed}
    outcome: OutcomeOutput = {"kind": "output", "output": {}}
    if trimmed:
        outcome["text"] = trimmed
    return outcome


async def _run_command(api: HookApi, command: str, seconds: float, body: str, context: Context) -> Outcome:
    env = await api.env(context)
    if env is None:
        return {"kind": "failed", "error": "No execution environment to run the hook command in"}
    stdout: list[str] = []
    stderr: list[str] = []

    def on_output(text: str, _context: Context, info: Any) -> None:
        (stdout if info["stream"] == "stdout" else stderr).append(text)

    options: Any = {
        "env": {"CLAUDE_PROJECT_DIR": env.cwd},
        "timeout": seconds,
        "stdin": body,
        "onOutput": on_output,
    }
    result = await env.exec(command, options, context)
    if result["ok"] is False:
        return {"kind": "failed", "error": error_message(result["error"])}
    code = result["value"]["exitCode"]
    if code == 0:
        return parse_output("".join(stdout))
    reason = "".join(stderr).strip()
    if code == 2:
        return {"kind": "blocked", "reason": reason or "Blocked by hook"}
    return {"kind": "failed", "error": f"Hook command exited with code {code}: {reason}"}


def interpolate(value: str, allowed: Sequence[str]) -> str:
    """``value`` with ``$VAR`` and ``${VAR}`` of the allowed variables replaced by their values (empty when unset)."""
    out = value
    for name in allowed:
        replacement = os.environ.get(name, "")
        out = out.replace("${" + name + "}", replacement).replace("$" + name, replacement)
    return out


async def _run_http(client: httpx.AsyncClient, handler: Any, seconds: float, body: str, context: Context) -> Outcome:
    allowed = handler.get("allowedEnvVars", [])
    headers = {name: interpolate(value, allowed) for name, value in handler.get("headers", {}).items()}
    headers.setdefault("Content-Type", "application/json")
    try:
        response = await await_with_context(
            client.post(handler["url"], content=body.encode(), headers=headers, timeout=seconds), context
        )
    except Exception as error:
        signal = context.abort_signal
        if signal is not None and signal.aborted:
            raise
        return {"kind": "failed", "error": f"Hook request failed: {error_message(error)}"}
    if not 200 <= response.status_code < 300:
        return {"kind": "failed", "error": f"Hook returned HTTP {response.status_code}"}
    trimmed = response.text.strip()
    if not trimmed:
        return {"kind": "output", "output": {}}
    try:
        parsed = json.loads(trimmed)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        return {"kind": "failed", "error": "Hook returned a body that is not a JSON object"}
    return {"kind": "output", "output": parsed}
