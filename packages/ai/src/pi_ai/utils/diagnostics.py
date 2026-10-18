"""Redacted provider/runtime diagnostics attached to assistant messages."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

from ..types import now_ms

__all__ = [
    "DiagnosticErrorInfo",
    "AssistantMessageDiagnostic",
    "format_thrown_value",
    "extract_diagnostic_error",
    "create_assistant_message_diagnostic",
    "append_assistant_message_diagnostic",
]


@dataclass(slots=True)
class DiagnosticErrorInfo:
    message: str
    name: str | None = None
    stack: str | None = None
    code: str | int | None = None


@dataclass(slots=True)
class AssistantMessageDiagnostic:
    type: str
    timestamp: int = field(default_factory=now_ms)
    error: DiagnosticErrorInfo | None = None
    details: dict[str, Any] | None = None


def format_thrown_value(value: Any) -> str:
    """Human-readable text for an exception or thrown value."""
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: Any) -> DiagnosticErrorInfo:
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(name="ThrownValue", message=format_thrown_value(error))
    code = getattr(error, "code", None)
    stack = (
        "".join(traceback.format_exception(type(error), error, error.__traceback__)) if error.__traceback__ else None
    )
    return DiagnosticErrorInfo(
        name=type(error).__name__,
        message=str(error) or type(error).__name__,
        stack=stack,
        code=code if isinstance(code, (str, int)) and not isinstance(code, bool) else None,
    )


def create_assistant_message_diagnostic(
    type: str, error: Any, details: dict[str, Any] | None = None
) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        type=type, timestamp=now_ms(), error=extract_diagnostic_error(error), details=details
    )


def append_assistant_message_diagnostic(message: Any, diagnostic: AssistantMessageDiagnostic) -> None:
    message.diagnostics = [*(message.diagnostics or []), diagnostic]
