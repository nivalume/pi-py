"""Shared truncation utilities for tool outputs (port of ``truncate.ts``).

Truncation is based on two independent limits; whichever is hit first wins:

- Line limit (default: 2000 lines)
- Byte limit (default: 50KB)

Never returns partial lines. Tool output streams are bounded by ``pi_chord.output.OutputBuffer``
instead. Sizes are UTF-8 byte counts measured like JavaScript's ``TextEncoder``
(:func:`pi_chord.output.utf8_byte_length`). A :class:`TruncationResult` is plain data with upstream's
camelCase keys, since tools persist it in result ``details``.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Final, Literal, TypedDict

from pi_chord.output import utf8_byte_length

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "TruncationResult",
    "TruncationTotals",
    "format_size",
    "truncate_head",
    "truncate_head_of",
    "utf8_byte_length",
]

DEFAULT_MAX_LINES: Final = 2000
DEFAULT_MAX_BYTES: Final = 50 * 1024  # 50KB


class TruncationResult(TypedDict):
    #: The truncated content.
    content: str
    #: Whether truncation occurred.
    truncated: bool
    #: Which limit was hit: ``"lines"``, ``"bytes"``, or ``None`` if not truncated.
    truncatedBy: Literal["lines", "bytes"] | None
    #: Total number of lines in the original content.
    totalLines: int
    #: Total number of bytes in the original content.
    totalBytes: int
    #: Number of complete lines in the truncated output.
    outputLines: int
    #: Number of bytes in the truncated output.
    outputBytes: int
    #: Whether the last line was partially truncated (only for the tail truncation edge case).
    lastLinePartial: bool
    #: Whether the first line exceeded the byte limit (for head truncation).
    firstLineExceedsLimit: bool
    #: The max lines limit that was applied.
    maxLines: int
    #: The max bytes limit that was applied.
    maxBytes: int


class TruncationTotals(TypedDict):
    """Totals of a text known by a prefix (``truncate_head_of``)."""

    lines: int
    bytes: int


def _split_lines_for_counting(content: str) -> list[str]:
    if len(content) == 0:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _to_fixed_1(value: float) -> str:
    # JavaScript's toFixed(1): the exact binary value rounded half up, unlike Python's half-even format.
    return str(Decimal(value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def format_size(bytes: int) -> str:
    """Format bytes as human-readable size."""
    if bytes < 1024:
        return f"{bytes}B"
    if bytes < 1024 * 1024:
        return f"{_to_fixed_1(bytes / 1024)}KB"
    return f"{_to_fixed_1(bytes / (1024 * 1024))}MB"


def truncate_head(content: str, *, max_lines: int | None = None, max_bytes: int | None = None) -> TruncationResult:
    """Truncate content from the head (keep the first lines/bytes).

    Suitable for file reads where you want to see the beginning. Never returns partial lines. If the
    first line exceeds the byte limit, returns empty content with ``firstLineExceedsLimit`` set.
    """
    return truncate_head_of(
        content,
        {"lines": len(_split_lines_for_counting(content)), "bytes": utf8_byte_length(content)},
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_head_of(
    prefix: str, totals: TruncationTotals, *, max_lines: int | None = None, max_bytes: int | None = None
) -> TruncationResult:
    """``truncate_head`` of a text known by a prefix and its totals.

    ``totals["lines"]`` is counted like ``truncate_head``, ignoring a trailing newline. The prefix must
    be the whole text, or longer than ``max_bytes + 1`` UTF-8 bytes, or hold at least ``max_lines``
    newlines; then the result equals ``truncate_head`` of the whole text.
    """
    limit_lines = DEFAULT_MAX_LINES if max_lines is None else max_lines
    limit_bytes = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes

    total_bytes = totals["bytes"]
    lines = _split_lines_for_counting(prefix)
    total_lines = totals["lines"]

    # Check if no truncation is needed.
    if total_lines <= limit_lines and total_bytes <= limit_bytes:
        return {
            "content": prefix,
            "truncated": False,
            "truncatedBy": None,
            "totalLines": total_lines,
            "totalBytes": total_bytes,
            "outputLines": total_lines,
            "outputBytes": total_bytes,
            "lastLinePartial": False,
            "firstLineExceedsLimit": False,
            "maxLines": limit_lines,
            "maxBytes": limit_bytes,
        }

    # Check if the first line alone exceeds the byte limit.
    first_line_bytes = utf8_byte_length(lines[0]) if lines else 0
    if first_line_bytes > limit_bytes:
        return {
            "content": "",
            "truncated": True,
            "truncatedBy": "bytes",
            "totalLines": total_lines,
            "totalBytes": total_bytes,
            "outputLines": 0,
            "outputBytes": 0,
            "lastLinePartial": False,
            "firstLineExceedsLimit": True,
            "maxLines": limit_lines,
            "maxBytes": limit_bytes,
        }

    # Collect complete lines that fit.
    output_lines: list[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for index, line in enumerate(lines[:limit_lines]):
        line_bytes = utf8_byte_length(line) + (1 if index > 0 else 0)  # +1 for the newline
        if output_bytes_count + line_bytes > limit_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes_count += line_bytes

    # Without a byte break, only omitted lines prove the line limit was reached; otherwise a trailing
    # newline exceeded bytes.
    if truncated_by != "bytes":
        truncated_by = "lines" if len(output_lines) < total_lines else "bytes"

    output_content = "\n".join(output_lines)
    return {
        "content": output_content,
        "truncated": True,
        "truncatedBy": truncated_by,
        "totalLines": total_lines,
        "totalBytes": total_bytes,
        "outputLines": len(output_lines),
        "outputBytes": utf8_byte_length(output_content),
        "lastLinePartial": False,
        "firstLineExceedsLimit": False,
        "maxLines": limit_lines,
        "maxBytes": limit_bytes,
    }
