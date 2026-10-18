"""The error type raised by the model runtime (auth, provider lookup, catalogs)."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from .diagnostics import format_thrown_value

__all__ = ["ModelsError", "ModelsErrorCode"]

ModelsErrorCode: TypeAlias = Literal["model_source", "model_validation", "provider", "stream", "auth", "oauth"]


class ModelsError(Exception):
    """Runtime error with a machine-readable ``code``.

    The message includes the cause's detail, because callers usually surface only
    ``str(error)``.
    """

    def __init__(self, code: ModelsErrorCode, message: str, *, cause: Any = None) -> None:
        super().__init__(_with_cause_detail(message, cause))
        self.code: ModelsErrorCode = code
        if isinstance(cause, BaseException):
            self.__cause__ = cause
        self.cause = cause


def _with_cause_detail(message: str, cause: Any) -> str:
    if cause is None:
        return message
    detail = format_thrown_value(cause).strip()
    if not detail or detail in message:
        return message
    return f"{message}: {detail}"
