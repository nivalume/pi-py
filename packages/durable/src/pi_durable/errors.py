"""Durable error classes (port of ``errors.ts``)."""

from __future__ import annotations

from .types import ConversationId

__all__ = ["ConversationBusy", "ReadAfterWrite", "SessionFailed", "StorageRequestError", "error_message"]


def error_message(error: object) -> str:
    """The message of a raised value, without raising itself.

    ``str()`` can raise for exotic values, and code that formats an error must not fail for the
    value it formats.
    """
    try:
        return str(error)
    except Exception:
        return object.__repr__(error)


class ReadAfterWrite(Exception):
    """A transaction read a table after its first table write. Read every required row before writing."""

    def __init__(self, method: str) -> None:
        super().__init__(f"Tx.{method}() cannot read tables after the first table write")


class StorageRequestError(Exception):
    """A Storage read rejected an invalid request, with no durable effect.

    Examples: an unknown conversation or a cursor from another scan. Unlike any other error a Storage
    raises, it fails only that read, not the Session; from ``commit()`` it is fatal like any other.
    """


class SessionFailed(Exception):
    """The Session failed, so nothing runs or commits any more.

    It fails when a Storage method raises, a commit cannot be adopted, or the Harness's own
    bookkeeping raises. ``cause`` (also ``__cause__``) is that first error. Reopen the Session;
    reopening recovers from what was committed.
    """

    def __init__(self, cause: object) -> None:
        super().__init__("Session failed after a storage error; close and reopen it")
        self.cause = cause
        if isinstance(cause, BaseException):
            self.__cause__ = cause


class ConversationBusy(Exception):
    """A submission reached a busy conversation and was not admitted."""

    def __init__(self, conversation_id: ConversationId) -> None:
        super().__init__(f"Conversation {conversation_id} is busy")
        self.conversation_id = conversation_id
