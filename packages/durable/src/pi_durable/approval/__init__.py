"""Approvals and permissions: the policy that decides, per tool call, whether it runs, is denied or must be approved;
the durable records of a request and its decision; and Claude Code's permission rules and modes as a policy.

Codex's flow made durable. When the policy asks, the tool task commits a request into the conversation's
``pi.approvals`` with its deadline, then waits in its ``approval`` phase for the decision: a notification from
:class:`PendingApprovals`, or the committed decision on a rerun. ``Conversation.decide()`` commits the decision (and a
display-only ``pi.approval`` entry, and a session grant for ``approvedForSession``) before it notifies the waiter, so
a crash at any point leaves one durable outcome. See ``packages/durable/DESIGN.md`` ("Approvals and permissions").
"""

from __future__ import annotations

from .commands import split_command
from .document import (
    APPROVAL_ENTRY,
    ApprovalEntryData,
    ApprovalGrant,
    ApprovalRequest,
    ApprovalsDoc,
    ApprovalsState,
    Recorded,
    ReviewDecision,
    find_request,
    granted,
    record_decision,
    remove_request,
)
from .errors import ApprovalError, ApprovalNotPending, InvalidDecision, UnknownApproval
from .paths import Anchor, Anchors, PathPattern
from .pending import ApprovalWaiter, PendingApprovals
from .permissions import (
    PERMISSION_MODES,
    PermissionDefaults,
    PermissionMode,
    PermissionRules,
    PermissionRulesDoc,
    PermissionsChange,
    PermissionsDoc,
    PermissionsState,
    RulesPolicy,
    add_session_rule,
    apply_permissions_change,
    copy_owner_permissions,
    validate_rules,
)
from .policy import (
    AllowAll,
    ApprovalCall,
    ApprovalPolicy,
    AskAll,
    DenyAll,
    HookPermission,
    PolicyAllow,
    PolicyAsk,
    PolicyDecision,
    PolicyDeny,
)
from .rules import (
    CommandSubject,
    PathAccess,
    PathSubject,
    PermissionSubject,
    Rule,
    RuleCall,
    RuleError,
    RuleErrorCode,
    allowed_by,
    is_within,
    propose_rule,
)

__all__ = [
    # policy
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
    # requests and decisions
    "APPROVAL_ENTRY",
    "ApprovalEntryData",
    "ApprovalGrant",
    "ApprovalRequest",
    "ApprovalsDoc",
    "ApprovalsState",
    "Recorded",
    "ReviewDecision",
    "find_request",
    "granted",
    "record_decision",
    "remove_request",
    "ApprovalWaiter",
    "PendingApprovals",
    "ApprovalError",
    "ApprovalNotPending",
    "InvalidDecision",
    "UnknownApproval",
    # permissions
    "PERMISSION_MODES",
    "PermissionDefaults",
    "PermissionMode",
    "PermissionRules",
    "PermissionRulesDoc",
    "PermissionsChange",
    "PermissionsDoc",
    "PermissionsState",
    "RulesPolicy",
    "add_session_rule",
    "apply_permissions_change",
    "copy_owner_permissions",
    "validate_rules",
    # rules
    "Anchor",
    "Anchors",
    "CommandSubject",
    "PathAccess",
    "PathPattern",
    "PathSubject",
    "PermissionSubject",
    "Rule",
    "RuleCall",
    "RuleError",
    "RuleErrorCode",
    "allowed_by",
    "is_within",
    "propose_rule",
    "split_command",
]
