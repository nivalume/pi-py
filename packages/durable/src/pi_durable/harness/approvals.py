"""Approvals and permissions on the Harness surface: ``decide()``, the pending requests, and the permission documents
(see :mod:`pi_durable.approval`)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

from pi_chord.context import Context
from pi_chord.json import copy_json

from ..approval import (
    PERMISSION_MODES,
    AllowAll,
    ApprovalNotPending,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalsDoc,
    InvalidDecision,
    PendingApprovals,
    PermissionMode,
    PermissionRules,
    PermissionRulesDoc,
    PermissionsChange,
    PermissionsDoc,
    PermissionsState,
    ReviewDecision,
    Rule,
    UnknownApproval,
    apply_permissions_change,
    record_decision,
    validate_rules,
)
from ..types import ConversationId, TaskId

if TYPE_CHECKING:
    from .harness import Harness

__all__ = ["HarnessApprovals"]

_DECISIONS: Final = ("approved", "approvedForSession", "denied", "abort", "approvedWithRule")


class HarnessApprovals:
    """One Harness's approval services: the policy, the waiters of approval phases, and the abort marks of an
    ``abort`` decision. Never persisted."""

    __slots__ = ("_abort", "pending", "policy")

    def __init__(self, policy: ApprovalPolicy | None, abort: Callable[[TaskId[Any]], asyncio.Future[Any]]) -> None:
        #: ``None`` for ``AllowAll`` (or no policy): calls run without deriving a subject.
        self.policy: ApprovalPolicy | None = None if policy is None or isinstance(policy, AllowAll) else policy
        self.pending = PendingApprovals()
        self._abort = abort

    def abort_tasks(self, ids: Sequence[TaskId[Any]]) -> None:
        """Mark ``ids`` for abort, in order, without waiting for them to end."""
        for id in ids:
            marked = self._abort(id)
            marked.add_done_callback(_consume)


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def check_decision(decision: Any) -> ReviewDecision:
    """A decision a caller may make, as strict JSON; raises :class:`InvalidDecision` or ``RuleError``."""
    kind = decision.get("type") if isinstance(decision, dict) else None
    if kind == "timedOut":
        raise InvalidDecision("A caller cannot decide timedOut: only the harness times out")
    if kind not in _DECISIONS:
        raise InvalidDecision(f"Unknown decision {decision!r}")
    if kind == "denied" and decision.get("rejection") is not None and not isinstance(decision["rejection"], str):
        raise InvalidDecision("A denial's rejection must be a string")
    if kind == "approvedWithRule":
        rule = decision.get("rule")
        if not isinstance(rule, str):
            raise InvalidDecision("approvedWithRule needs a rule")
        Rule.parse(rule)
    return cast(ReviewDecision, copy_json(decision, omit_undefined_properties=True))


async def decide(
    harness: Harness, conversation_id: ConversationId, id: TaskId[Any], decision: ReviewDecision, context: Context
) -> None:
    """Decide the pending approval request ``id`` of ``conversation_id``: one commit writes the decision (and the
    ``pi.approval`` entry, a session grant for ``approvedForSession``, the rule of ``approvedWithRule``), then the
    waiting tool task is notified."""
    written = check_decision(decision)
    harness.tasks.resume()
    settings = harness.resolved_settings()
    now = harness.now()

    async def change(tx: Any) -> None:
        task = await tx.task(id)
        if task is None or task["conversationId"] != conversation_id or task["kind"] != "pi.tool":
            raise UnknownApproval(id)
        recorded, _decision = await record_decision(tx, conversation_id, id, written, now, settings)
        if recorded != "written":
            raise ApprovalNotPending(id)

    await harness.commit_with(change, context)
    harness.approvals.pending.notify(id, copy_json(written))  # type: ignore[arg-type]


async def decide_anywhere(harness: Harness, id: TaskId[Any], decision: ReviewDecision, context: Context) -> None:
    """Decide the request ``id`` in whichever conversation its tool task runs."""
    task = await harness.get_task(id, context)
    if task is None:
        raise UnknownApproval(id)
    await decide(harness, task["conversationId"], id, decision, context)


async def pending_approvals(
    harness: Harness, conversation_id: ConversationId, context: Context
) -> list[ApprovalRequest]:
    """The conversation's approval requests waiting for a decision, in id order."""
    state = await harness.snapshot(ApprovalsDoc, conversation_id, context)
    if state is None:
        return []
    return [request for request in state["requests"] if "decision" not in request]


async def all_pending_approvals(
    harness: Harness, conversation_ids: Sequence[ConversationId], context: Context
) -> list[tuple[ConversationId, ApprovalRequest]]:
    pending: list[tuple[ConversationId, ApprovalRequest]] = []
    for conversation_id in conversation_ids:
        for request in await pending_approvals(harness, conversation_id, context):
            pending.append((conversation_id, request))
    return pending


async def permissions(harness: Harness, conversation_id: ConversationId, context: Context) -> PermissionsState:
    """The conversation's permissions; ``{}`` when none were set."""
    state = await harness.snapshot(PermissionsDoc, conversation_id, context)
    return {} if state is None else state


def check_change(change: PermissionsChange) -> None:
    """Raise ``RuleError`` for an added rule that does not parse, ``ValueError`` for an unknown mode."""
    mode = change.get("mode")
    if mode is not None and mode not in PERMISSION_MODES:
        raise ValueError(f"Unknown permission mode {mode!r}")
    validate_rules(change.get("add"))


async def update_permissions(
    harness: Harness, conversation_id: ConversationId, change: PermissionsChange, context: Context
) -> PermissionsState:
    """Change the conversation's mode, rules or directories in one commit; returns the new state."""
    check_change(change)

    async def apply(tx: Any) -> PermissionsState:
        if await tx.conversation(conversation_id) is None:
            raise RuntimeError(f"Conversation {conversation_id} does not exist")
        state = await tx.doc(PermissionsDoc, conversation_id)
        apply_permissions_change(state, change)
        return cast(PermissionsState, copy_json(dict(state)))

    return await harness.commit_with(apply, context)


async def set_permission_mode(
    harness: Harness, conversation_id: ConversationId, mode: PermissionMode | None, context: Context
) -> PermissionsState:
    """Set the conversation's permission mode; ``None`` returns it to the policy's default."""
    return await update_permissions(harness, conversation_id, {"mode": mode}, context)


async def permission_rules(harness: Harness, context: Context) -> PermissionRules:
    """The allow rules approvals added (``approvedWithRule``), for every conversation."""
    rules = await harness.snapshot(PermissionRulesDoc, context)
    return {} if rules is None else rules


async def remove_permission_rule(harness: Harness, rule: str, context: Context) -> bool:
    """Remove ``rule`` from the rules approvals added; whether it was there."""
    text = rule.strip()

    async def change(tx: Any) -> bool:
        rules = await tx.doc(PermissionRulesDoc)
        allow = rules.get("allow") or []
        kept = [existing for existing in allow if existing.strip() != text]
        if len(kept) == len(allow):
            return False
        if kept:
            rules["allow"] = kept
        else:
            del rules["allow"]
        return True

    return await harness.commit_with(change, context)
