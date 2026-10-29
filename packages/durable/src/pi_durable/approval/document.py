"""``pi.approvals``: a conversation's pending approval requests and session grants, and the commit that records a
decision."""

from __future__ import annotations

from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict

from pi_chord.json import copy_json

from ..documents import define_doc
from ..types import ConversationDocToken, ConversationId, TaskId, Tx
from .permissions import add_session_rule

__all__ = [
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
]

#: The entry kind of a decision: display-only, the audit record observers see.
APPROVAL_ENTRY: Final = "pi.approval"


class _Approved(TypedDict):
    type: Literal["approved"]


class _ApprovedForSession(TypedDict):
    """Approved, and later calls with the same tool and approval key are too."""

    type: Literal["approvedForSession"]


class _Denied(TypedDict):
    type: Literal["denied"]
    rejection: NotRequired[str]


class _TimedOut(TypedDict):
    """Written by the harness when the deadline passes; refused from callers."""

    type: Literal["timedOut"]


class _Abort(TypedDict):
    """Aborts the conversation's run."""

    type: Literal["abort"]


class _ApprovedWithRule(TypedDict):
    """Approved, and ``rule`` joins the session's allow rules (``pi.permission-rules``) in the deciding commit."""

    type: Literal["approvedWithRule"]
    rule: str


#: The answer to a request (Codex's ``ReviewDecision``).
ReviewDecision: TypeAlias = _Approved | _ApprovedForSession | _Denied | _TimedOut | _Abort | _ApprovedWithRule


class ApprovalRequest(TypedDict):
    """One pending request. Its id is the id of the tool task that asks, which asks at most once."""

    id: TaskId[Any]
    callId: str
    name: str
    #: The final arguments, exactly as the call runs when approved.
    arguments: dict[str, Any]
    approvalKey: NotRequired[str]
    prompt: NotRequired[str]
    #: The rule a UI offers to always allow such calls.
    proposedRule: NotRequired[str]
    requestedAt: float
    #: Harness clock, epoch milliseconds; also in the task's checkpoint.
    deadline: float
    #: Set by the deciding commit; the task consumes it and removes the request.
    decision: NotRequired[ReviewDecision]


class ApprovalGrant(TypedDict):
    """A session grant: calls of ``tool`` whose approval key is ``key`` are approved without asking."""

    tool: str
    key: str
    grantedBy: TaskId[Any]


class ApprovalsState(TypedDict):
    #: In id order.
    requests: list[ApprovalRequest]
    #: In grant order.
    grants: list[ApprovalGrant]


class ApprovalEntryData(TypedDict):
    """The data of a ``pi.approval`` entry."""

    approvalId: TaskId[Any]
    callId: str
    name: str
    decision: ReviewDecision


#: Conversation scope, latest; a fork starts without requests or grants; a complete base whenever no request is
#: pending.
ApprovalsDoc: ConversationDocToken[ApprovalsState] = define_doc(
    kind="pi.approvals",
    version=1,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda: {"requests": [], "grants": []},
    checkpoint_when=lambda value, _ops, _info: not value.get("requests"),
)

#: What :func:`record_decision` found: the decision was written, the request already had one, or there is no such
#: request.
Recorded: TypeAlias = Literal["written", "already", "missing"]


def find_request(state: ApprovalsState | None, id: TaskId[Any]) -> ApprovalRequest | None:
    if state is None:
        return None
    return next((request for request in state["requests"] if request["id"] == id), None)


def granted(state: ApprovalsState | None, tool: str, key: str) -> bool:
    """Whether a grant covers ``tool`` with ``key``."""
    if state is None:
        return False
    return any(grant["tool"] == tool and grant["key"] == key for grant in state["grants"])


async def remove_request(tx: Tx, conversation_id: ConversationId, id: TaskId[Any]) -> None:
    requests = (await tx.doc(ApprovalsDoc, conversation_id))["requests"]
    for index, request in enumerate(requests):
        if request["id"] == id:
            del requests[index]
            return


async def record_decision(
    tx: Tx,
    conversation_id: ConversationId,
    id: TaskId[Any],
    decision: ReviewDecision,
    now: float,
    queue_modes: Any,
) -> tuple[Recorded, ReviewDecision | None]:
    """Write ``decision`` into the pending request ``id``: the request's ``decision``, a grant for
    ``approvedForSession`` when the request has an approval key, the rule of ``approvedWithRule`` in the session's
    allow rules, and the display-only ``pi.approval`` entry, admitted once under request id ``approval:<id>``.

    Returns what it found and the request's decision (``decision`` when written). Admission comes first, before any
    table write of the commit.
    """
    from ..harness.submissions import admit_submission

    state = await tx.doc(ApprovalsDoc, conversation_id)
    request = find_request(state, id)
    if request is None:
        return "missing", None
    if "decision" in request:
        return "already", copy_json(request["decision"])  # type: ignore[return-value]
    decision = copy_json(decision)  # type: ignore[assignment]
    data: ApprovalEntryData = {
        "approvalId": id,
        "callId": request["callId"],
        "name": request["name"],
        "decision": decision,
    }
    entry: Any = {"kind": APPROVAL_ENTRY, "data": copy_json(data)}
    await admit_submission(
        tx, conversation_id, {"type": "write", "entry": entry, "requestId": f"approval:{id}"}, now, queue_modes
    )
    if decision["type"] == "approvedWithRule":
        await add_session_rule(tx, decision["rule"].strip())
    request["decision"] = copy_json(decision)  # type: ignore[typeddict-item]
    key = request.get("approvalKey")
    if decision["type"] == "approvedForSession" and key is not None and not granted(state, request["name"], key):
        state["grants"].append({"tool": request["name"], "key": key, "grantedBy": id})
    return "written", decision
