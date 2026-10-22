"""``ClaudeHooks``: the configured handlers as the Harness's tool, generation and compaction hooks, and a
``PermissionRequest`` approval policy decorator."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from pi_ai.types import TextContent
from pi_chord.context import Context
from pi_chord.json import copy_json

from ..approval import ApprovalCall, PermissionsDoc, PolicyDecision
from ..harness.compaction import CompactionTask
from ..harness.define import define_extension, hook
from ..harness.generation import GenerationTask
from ..harness.live import LiveDoc
from ..harness.tool import ToolTask
from ..types import ConversationId
from .config import HooksConfig
from .protocol import (
    HookOutput,
    arguments_from,
    claude_tool_input,
    claude_tool_name,
    output_block,
    specific,
    specific_str,
)
from .runner import Outcome, memo_name, run_handlers

if TYPE_CHECKING:
    import httpx

    from ..approval import ApprovalPolicy
    from ..harness.types import Extension, HookApi

__all__ = ["CLAUDE_HOOKS", "ClaudeHooks", "ClaudeHooksOptions"]

#: The name of the extension :meth:`ClaudeHooks.extension` makes.
CLAUDE_HOOKS: Final = "claude-hooks"


@dataclass(slots=True)
class ClaudeHooksOptions:
    """How :class:`ClaudeHooks` runs handlers."""

    #: For HTTP handlers; a default ``httpx.AsyncClient`` when unset.
    client: httpx.AsyncClient | None = None
    #: ``transcript_path`` of the input for a conversation; empty when unset (the Harness keeps no transcript file).
    transcript_path: Callable[[ConversationId], str] | None = None
    #: Receives each non-blocking handler failure: ``on_error(event, error)``.
    on_error: Callable[[str, str], None] | None = None


class ClaudeHooks:
    """Claude Code's command and HTTP hooks over the Harness's hook seams."""

    def __init__(self, config: HooksConfig, options: ClaudeHooksOptions | None = None) -> None:
        options = options if options is not None else ClaudeHooksOptions()
        self.config = config
        self._client = options.client
        self._transcript_path = options.transcript_path
        self._on_error = options.on_error

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient()
        return self._client

    def extension(self) -> Extension:
        """The extension holding the tool, generation and compaction hooks."""
        return define_extension(
            CLAUDE_HOOKS,
            hooks=[
                hook(ToolTask, {"before_tool": self.before_tool, "after_tool": self.after_tool}),
                hook(GenerationTask, {"on_input": self.on_input, "on_yield": self.on_yield}),
                hook(CompactionTask, {"before_compact": self.before_compact}),
            ],
        )

    def permission_policy(self, inner: ApprovalPolicy) -> ApprovalPolicy:
        """``inner`` with the ``PermissionRequest`` handlers deciding what it would ask: ``allow`` allows, ``deny`` (or
        exit 2) denies, anything else keeps the request."""
        return _PermissionRequestPolicy(self, inner)

    async def _input(self, api: HookApi, event: str, fields: Mapping[str, Any], context: Context) -> dict[str, Any]:
        """The input of ``event``: the common fields, then ``fields``."""
        try:
            cwd = await api.cwd(context) or ""
        except Exception:
            cwd = ""
        permissions = await api.snapshot(PermissionsDoc, api.conversation_id, context)
        mode = None if permissions is None else permissions.get("mode")
        transcript = "" if self._transcript_path is None else self._transcript_path(api.conversation_id)
        return {
            "session_id": str(api.conversation_id),
            "transcript_path": transcript,
            "cwd": cwd,
            "permission_mode": mode or "default",
            "hook_event_name": event,
            **fields,
        }

    async def _run(
        self, api: HookApi, event: str, values: Sequence[str], fields: Mapping[str, Any], context: Context
    ) -> list[Outcome]:
        """Run ``event``'s handlers matching ``values`` (none: every group); report the failures of a fresh run."""
        handlers = self.config.handlers(event, values)
        if not handlers and await api.memo(memo_name(event), context) is None:
            return []
        input = await self._input(api, event, fields, context)
        outcomes, fresh = await run_handlers(self._http(), api, event, handlers, input, context)
        if fresh:
            for outcome in outcomes:
                if outcome["kind"] == "failed":
                    self._report(event, outcome["error"])
        return outcomes

    def _report(self, event: str, error: str) -> None:
        if self._on_error is not None:
            self._on_error(event, error)

    # ─── tool hooks ──────────────────────────────────────────────────────────

    async def before_tool(self, call: Any, api: HookApi, context: Context) -> Any:
        """``PreToolUse``: deny (or exit 2, or a block) blocks; ``ask`` and ``allow`` reach the policy; ``updatedInput``
        replaces the arguments; ``additionalContext`` follows the result. Several handlers: deny over defer over ask
        over allow."""
        claude = claude_tool_name(call.name)
        fields = {"tool_name": claude, "tool_input": claude_tool_input(call.arguments), "tool_use_id": call.id}
        outcomes = await self._run(api, "PreToolUse", [claude, call.name], fields, context)
        if not outcomes:
            return None
        reason = _blocked(outcomes)
        if reason is not None:
            return {"block": reason}
        result: dict[str, Any] = {}
        rank = 0
        for output in _outputs(outcomes):
            reason = output_block(output)
            if reason is not None:
                return {"block": reason}
            decision = specific_str(output, "permissionDecision")
            decision_reason = specific_str(output, "permissionDecisionReason")
            if decision == "deny":
                return {"block": decision_reason or "Denied by PreToolUse hook"}
            ranked = {"defer": 3, "ask": 2, "allow": 1}.get(decision or "", 0)
            if ranked > rank:
                rank = ranked
                result.pop("permission", None)
                result.pop("reason", None)
                if decision in ("ask", "allow"):
                    result["permission"] = decision
                if decision == "ask" and decision_reason:
                    result["reason"] = decision_reason
            if "arguments" not in result:
                arguments = arguments_from(specific(output, "updatedInput"), call.arguments)
                if arguments is not None:
                    result["arguments"] = arguments
        notes = _contexts(outcomes)
        if notes is not None:
            result["context"] = notes
        return result or None

    async def after_tool(self, call: Any, result: Any, api: HookApi, context: Context) -> Any:
        """``PostToolUse`` / ``PostToolUseFailure`` (by ``isError``): ``updatedToolOutput`` replaces the output; a
        block's reason, exit 2's stderr and ``additionalContext`` are appended as one text item."""
        claude = claude_tool_name(call.name)
        fields: dict[str, Any] = {
            "tool_name": claude,
            "tool_input": claude_tool_input(call.arguments),
            "tool_use_id": call.id,
        }
        if result.get("isError") is True:
            event = "PostToolUseFailure"
            error = next(
                (item["message"] for item in result.get("diagnostics") or [] if item.get("severity") == "error"),
                None,
            )
            fields["error"] = error if error is not None else _output_text(result)
        else:
            event = "PostToolUse"
            fields["tool_response"] = {
                "output": _output_text(result),
                "structured": _json(result.get("structuredOutput")),
            }
        outcomes = await self._run(api, event, [claude, call.name], fields, context)
        changed = dict(result)
        touched = False
        updated = next(
            (text for text in (specific_str(output, "updatedToolOutput") for output in _outputs(outcomes)) if text),
            None,
        )
        if updated is not None:
            changed["output"] = [TextContent(updated)]
            touched = True
        notes: list[str] = []
        for outcome in outcomes:
            if outcome["kind"] == "blocked":
                notes.append(outcome["reason"])
            elif outcome["kind"] == "output":
                reason = output_block(outcome["output"])
                if reason is not None:
                    notes.append(reason)
        context_notes = _contexts(outcomes)
        if context_notes is not None:
            notes.append(context_notes)
        if notes:
            changed["output"] = [*(changed.get("output") or []), TextContent("\n\n".join(notes))]
            touched = True
        return changed if touched else None

    # ─── generation hooks ────────────────────────────────────────────────────

    async def on_input(self, run_input: Any, api: HookApi, context: Context) -> Any:
        """``UserPromptSubmit``: a block (or exit 2) blocks the run; ``additionalContext`` and plain stdout become
        context."""
        prompts: list[str] = []
        for entry in run_input["entries"]:
            for message in entry.get("model") or []:
                if message.role == "user":
                    prompts.append(_text_of(message.content))
        outcomes = await self._run(api, "UserPromptSubmit", [], {"prompt": "\n\n".join(prompts)}, context)
        reason = _blocked(outcomes)
        if reason is None:
            reason = next((block for block in map(output_block, _outputs(outcomes)) if block is not None), None)
        if reason is not None:
            return {"block": reason}
        texts = [outcome["text"] for outcome in outcomes if outcome["kind"] == "output" and "text" in outcome]
        notes = _contexts(outcomes)
        if notes is not None:
            texts.append(notes)
        return {"context": "\n\n".join(texts)} if texts else None

    async def on_yield(self, answer: Any, api: HookApi, context: Context) -> Any:
        """``Stop``, or ``SubagentStop`` in a conversation a task owns: ``decision: "block"`` (or exit 2) continues the
        run with the reason."""
        record = await api.conversation_record(context)
        event = "SubagentStop" if record is not None and "owner" in record else "Stop"
        live = await api.snapshot(LiveDoc, api.conversation_id, context)
        run = None if live is None else live.get("run")
        continuations = 0 if run is None else run.get("continuations", 0)
        fields = {"stop_hook_active": continuations > 0, "last_assistant_message": _text_of(answer.content)}
        outcomes = await self._run(api, event, [], fields, context)
        reason = _blocked(outcomes)
        if reason is None:
            reason = next(
                (
                    output["reason"]
                    for output in _outputs(outcomes)
                    if output.get("decision") == "block" and isinstance(output.get("reason"), str)
                ),
                None,
            )
        return None if reason is None else {"continue": reason}

    # ─── compaction hooks ────────────────────────────────────────────────────

    async def before_compact(self, compaction: Any, api: HookApi, context: Context) -> Any:
        """``PreCompact`` (matcher: ``manual`` or ``auto``): a block (or exit 2) declines the compaction."""
        trigger = "manual" if compaction["reason"] == "manual" else "auto"
        fields = {"trigger": trigger, "custom_instructions": compaction.get("instructions") or ""}
        outcomes = await self._run(api, "PreCompact", [trigger], fields, context)
        declined = _blocked(outcomes) is not None or any(output_block(output) for output in _outputs(outcomes))
        return {"decline": True} if declined else None


class _PermissionRequestPolicy:
    """:meth:`ClaudeHooks.permission_policy`."""

    def __init__(self, hooks: ClaudeHooks, inner: ApprovalPolicy) -> None:
        self._hooks = hooks
        self._inner = inner

    async def decide(self, call: ApprovalCall, api: HookApi, context: Context) -> PolicyDecision:
        decided = self._inner.decide(call, api, context)
        decision = await decided if inspect.isawaitable(decided) else decided
        if decision["type"] != "ask":
            return decision
        name = call.call.name
        claude = claude_tool_name(name)
        proposed = decision.get("proposedRule")
        fields = {
            "tool_name": claude,
            "tool_input": claude_tool_input(call.call.arguments),
            "permission_suggestions": [] if proposed is None else [proposed],
        }
        outcomes = await self._hooks._run(api, "PermissionRequest", [claude, name], fields, context)
        reason = _blocked(outcomes)
        if reason is not None:
            return {"type": "deny", "reason": reason}
        for output in _outputs(outcomes):
            answer = specific(output, "decision")
            if not isinstance(answer, Mapping):
                continue
            if "updatedInput" in answer:
                self._hooks._report("PermissionRequest", "updatedInput is not supported; the call runs as requested")
            behavior = answer.get("behavior")
            if behavior == "allow":
                return {"type": "allow"}
            if behavior == "deny":
                message = answer.get("message")
                return {
                    "type": "deny",
                    "reason": message if isinstance(message, str) else "Denied by PermissionRequest hook",
                }
        return decision


def _outputs(outcomes: Sequence[Outcome]) -> list[HookOutput]:
    """The JSON outputs of ``outcomes``."""
    return [outcome["output"] for outcome in outcomes if outcome["kind"] == "output"]


def _blocked(outcomes: Sequence[Outcome]) -> str | None:
    """The first exit-2 reason."""
    return next((outcome["reason"] for outcome in outcomes if outcome["kind"] == "blocked"), None)


def _contexts(outcomes: Sequence[Outcome]) -> str | None:
    """Every ``additionalContext``, joined."""
    texts = [text for text in (specific_str(output, "additionalContext") for output in _outputs(outcomes)) if text]
    return "\n\n".join(texts) if texts else None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(item.text for item in content if isinstance(item, TextContent))


def _output_text(result: Mapping[str, Any]) -> str:
    """The text items of a result's output."""
    return "\n".join(item.text for item in result.get("output") or [] if isinstance(item, TextContent))


def _json(value: Any) -> Any:
    """``value`` as strict JSON for a hook's input; ``None`` when it is not JSON."""
    try:
        return copy_json(value, omit_undefined_properties=True)
    except Exception:
        return None
