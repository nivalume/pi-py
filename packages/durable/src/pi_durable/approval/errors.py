"""Why ``Conversation.decide()`` refused a decision; nothing was written."""

from __future__ import annotations

from typing import Any

__all__ = ["ApprovalError", "ApprovalNotPending", "InvalidDecision", "UnknownApproval"]


class ApprovalError(Exception):
    """Base of the refusals of ``decide()``. A rule that does not parse raises :class:`RuleError` instead."""


class UnknownApproval(ApprovalError):
    """No tool task with this id exists in the conversation."""

    def __init__(self, id: Any) -> None:
        super().__init__(f"Approval {id} does not exist")
        self.id = id


class ApprovalNotPending(ApprovalError):
    """The request is not waiting for a decision: decided already, timed out, withdrawn or never made."""

    def __init__(self, id: Any) -> None:
        super().__init__(f"Approval {id} is not pending")
        self.id = id


class InvalidDecision(ApprovalError):
    """A decision callers may not make (``timedOut`` is the harness's own), or one that is malformed."""
