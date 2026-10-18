"""Normalization of provider HTTP error objects into display strings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MAX_PROVIDER_ERROR_BODY_CHARS",
    "NormalizedProviderError",
    "normalize_provider_error",
    "format_provider_error",
    "truncate_error_text",
    "safe_json_stringify",
]

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


@dataclass(slots=True)
class NormalizedProviderError:
    message: str
    #: True when ``message`` already contains the body.
    message_carries_body: bool
    status: int | None = None
    body: str | None = None


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _extract_status(error: Any) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _pick_body_text(error: Any) -> str | None:
    body = getattr(error, "body", None)
    if isinstance(body, str):
        return body
    parsed = getattr(error, "error", None)
    if isinstance(parsed, dict) and parsed:
        return safe_json_stringify(parsed)
    return None


def normalize_provider_error(error: Any) -> NormalizedProviderError:
    """Probe an exception for HTTP status and body so the body reason is not lost."""
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error), message_carries_body=False)
    message = str(error) or type(error).__name__
    status = _extract_status(error)
    body_text = _pick_body_text(error)
    body: str | None = None
    if body_text is not None:
        trimmed = body_text.strip()
        if trimmed:
            body = truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)
    return NormalizedProviderError(
        message=message,
        message_carries_body=body is None or body in message,
        status=status,
        body=body,
    )


def format_provider_error(norm: NormalizedProviderError, prefix: str | None = None) -> str:
    """Compose a display string: ``"<status>: <body>"`` or ``"<prefix> (<status>): <body>"``."""
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"
