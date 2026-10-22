"""``ApprovalPolicy``: decides, per tool call, whether it runs, is denied or must be approved."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Protocol, TypeAlias, TypedDict

from .rules import PermissionSubject

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..harness.types import HookApi, ToolHookCall, ToolRegistration
    from ..types import ConversationId, TaskId

__all__ = [
    "AllowAll",
    "ApprovalCall",
    "ApprovalPolicy",
    "AskAll",
    "DenyAll",
    "HookPermission",
    "PolicyAllow",
    "PolicyAsk",
    "PolicyDecision",
    "PolicyDeny",
]

#: What the ``before_tool`` hooks said about approval (``BeforeToolResult["permission"]``), the most restrictive of the
#: chain: ``ask`` makes the call ask whatever the policy decides; ``allow`` is the policy's to honor.
HookPermission: TypeAlias = Literal["allow", "ask"]


class PolicyAllow(TypedDict):
    type: Literal["allow"]


class PolicyDeny(TypedDict):
    type: Literal["deny"]
    reason: str


class PolicyAsk(TypedDict):
    type: Literal["ask"]
    #: How long the request waits for a decision; default the settings' ``approvalTimeoutMs``.
    timeoutMs: NotRequired[float]
    prompt: NotRequired[str]
    #: The rule a UI offers to always allow such calls (``{"type": "approvedWithRule", "rule": ...}``).
    proposedRule: NotRequired[str]


#: What a policy decides (Codex's ``ExecApprovalRequirement``).
PolicyDecision: TypeAlias = PolicyAllow | PolicyDeny | PolicyAsk


@dataclass(frozen=True, slots=True)
class ApprovalCall:
    """The call a policy decides on."""

    conversation_id: ConversationId
    task_id: TaskId[Any]
    #: The call with its final arguments.
    call: ToolHookCall
    tool: ToolRegistration
    #: What permission rules match the call on (``ToolRegistration.permission_subject``).
    subject: PermissionSubject | None
    #: The call's working directory: its environment's, else the agent's ``cwd``.
    cwd: str | None
    #: What the ``before_tool`` hooks said about approval.
    hook: HookPermission | None = None


class ApprovalPolicy(Protocol):
    """Decides whether a tool call needs approval (``HarnessOptions.approval``).

    ``decide()`` runs off the Session line with the task's :class:`HookApi` (reads, models, memos); a crash before the
    call's intent commit may run it again, so a policy that must decide the same on a rerun memoizes its decision
    (``api.memo``). A raise denies the call with its text (fail closed).
    """

    def decide(
        self, call: ApprovalCall, api: HookApi, context: Context
    ) -> PolicyDecision | Awaitable[PolicyDecision]: ...


class AllowAll:
    """Every call runs, as without approvals. The default."""

    def decide(self, call: ApprovalCall, api: HookApi, context: Context) -> PolicyDecision:
        return {"type": "allow"}


class AskAll:
    """Asks for every call, with the settings' timeout."""

    def decide(self, call: ApprovalCall, api: HookApi, context: Context) -> PolicyDecision:
        return {"type": "ask"}


class DenyAll:
    """Denies every call with ``reason``."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def decide(self, call: ApprovalCall, api: HookApi, context: Context) -> PolicyDecision:
        return {"type": "deny", "reason": self.reason}
