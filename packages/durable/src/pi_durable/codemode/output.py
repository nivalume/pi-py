"""A script's result as the model reads it (port of ``codemode/output.ts``).

A "Script completed/failed" header with the wall time, the script's text items (``==> text N/M <==`` when there are
several), its ``print`` lines in one ``<console_output>`` block, the returned value, and on failure the error with the
calls already made. Text past the token budget is cut to its start and end, with the complete text in a temp file;
images are saved to temp files too, with their path next to them, because a script has no other way to hand their
bytes on.

Lengths are counted in Python characters (code points) where upstream counts UTF-16 code units.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypeAlias, TypedDict

from pi_ai import ImageContent, TextContent
from pi_codemode.types import CodemodeFailure, CodemodeSuccess, CodemodeTextOutput

if TYPE_CHECKING:
    from pi_codemode.types import CodemodeOutputItem, CodemodeResult

    from ..harness.types import ToolOutputLimits

__all__ = ["DEFAULT_MAX_OUTPUT_TOKENS", "FormattedResult", "OutputFile", "OutputFileResult", "format_result"]

_Item: TypeAlias = "TextContent | ImageContent"


class OutputFileResult(TypedDict, total=False):
    """Where a result file was written (``path``), or why it could not be (``error``)."""

    path: str
    error: str


#: Writes a file of the result under a fixed name: ``write(name, data)``.
OutputFile: TypeAlias = Callable[[str, "str | bytes"], Awaitable[OutputFileResult]]

#: Lines of the result besides the script's text: the header, markers, notes.
_LINE_MARGIN: Final = 100

#: Default token budget for script output.
DEFAULT_MAX_OUTPUT_TOKENS: Final = 10_000
#: Characters per token when estimating.
_CHARS_PER_TOKEN: Final = 4

#: File extensions of the image types ``image()`` accepts.
_IMAGE_EXTENSIONS: Final = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def value_text(value: Any) -> str:
    """Like the script's ``text()``: strings as is, other values as compact JSON."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return str(value)


def _format_items(output: Sequence[CodemodeOutputItem]) -> list[_Item]:
    """Lay out the script's output so the model can tell items apart: with more than one text item, each starts with
    a ``==> text N/M <==`` line; ``print`` lines follow all other output in one ``<console_output>`` block."""
    total = sum(1 for item in output if item.type == "text" and not item.console)
    items: list[_Item] = []
    console_lines: list[str] = []
    index = 0
    for item in output:
        if item.type == "image":
            items.append(ImageContent(item.data, item.mime_type))
        elif item.console:
            console_lines.append(item.text)
        else:
            index += 1
            items.append(TextContent(f"==> text {index}/{total} <==\n{item.text}" if total > 1 else item.text))
    if len(console_lines) > 0:
        items.append(TextContent("<console_output>\n" + "\n".join(console_lines) + "\n</console_output>"))
    return items


def _join_adjacent_text(items: Sequence[_Item]) -> list[_Item]:
    """Join adjacent text items into one, each part starting on its own line."""
    joined: list[_Item] = []
    parts: list[str] = []
    last_part = ""

    def flush() -> None:
        nonlocal parts, last_part
        if len(parts) > 0:
            joined.append(TextContent("".join(parts)))
        parts = []
        last_part = ""

    for item in items:
        if not isinstance(item, TextContent):
            flush()
            joined.append(item)
            continue
        if last_part != "" and not last_part.endswith("\n"):
            parts.append("\n")
            last_part = "\n"
        parts.append(item.text)
        if item.text != "":
            last_part = item.text
    flush()
    return joined


@dataclass(slots=True)
class _Truncated:
    items: list[_Item]
    full_output_path: str | None = None


async def _truncate(items: Sequence[_Item], max_tokens: int, limits: ToolOutputLimits, write: OutputFile) -> _Truncated:
    """Apply the token budget: when the combined text exceeds it, the text items become one item that keeps the
    start and end of the text, and images follow it. The complete text is written to a temp file."""
    texts = [item.text for item in items if isinstance(item, TextContent)]
    combined = "\n".join(texts)
    # Within the limits too, as upstream's budget: a quarter of the byte limit in characters (most take at most 3
    # UTF-8 bytes), and the rest of the result is a few short lines.
    budget = min(max_tokens * _CHARS_PER_TOKEN, limits.get("maxBytes", 0) // 4)
    line_budget = (limits.get("maxLines", 0) - _LINE_MARGIN) // 2
    lines = combined.count("\n") + 1
    if len(texts) == 0 or (len(combined) <= budget and lines <= 2 * line_budget):
        return _Truncated(list(items))
    head_chars = budget // 2
    tail_chars = budget - head_chars
    head = "\n".join(combined[:head_chars].split("\n")[:line_budget])
    tail = "\n".join(combined[-tail_chars:].split("\n")[-line_budget:]) if tail_chars > 0 else ""
    removed = len(combined) - len(head) - len(tail)
    text = (
        f"Warning: truncated output (original token count: {math.ceil(len(combined) / _CHARS_PER_TOKEN)})\n"
        f"Total output lines: {lines}\n\n{head}…{math.ceil(removed / _CHARS_PER_TOKEN)} tokens truncated…{tail}"
    )
    spilled = await write("output.txt", combined)
    path = spilled.get("path")
    if path is not None:
        text += f"\n\n[Full output: {path} (read with offset/limit)]"
    else:
        text += f"\n\n[Could not save the full output: {spilled.get('error', '')}]"
    return _Truncated([TextContent(text), *(item for item in items if isinstance(item, ImageContent))], path)


def _decode_base64(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        return b""


async def _save_images(items: Sequence[_Item], write: OutputFile) -> list[_Item]:
    """Save each image to a file and put a text item with its path before it; an image shown twice is saved once."""
    labels: dict[str, str] = {}
    result: list[_Item] = []
    for item in items:
        if not isinstance(item, ImageContent):
            result.append(item)
            continue
        label = labels.get(item.data)
        if label is None:
            data = _decode_base64(item.data)
            kind = f"{item.mime_type}, {len(data)} bytes"
            # A failed write must not discard the result of a script whose calls already ran: it becomes the label.
            saved = await write(f"image-{len(labels) + 1}{_IMAGE_EXTENSIONS.get(item.mime_type, '.bin')}", data)
            path = saved.get("path")
            label = (
                f"[Image saved to {path} ({kind})]"
                if path is not None
                else f"[Image ({kind}) could not be saved: {saved.get('error', '')}]"
            )
            labels[item.data] = label
        result.extend([TextContent(label), item])
    return result


def _format_error(result: CodemodeFailure, calls: Sequence[Any]) -> str:
    error = result.error
    if error.kind == "script":
        head = error.stack or f"{error.name or 'Error'}: {error.message}"
    elif error.kind == "timeout":
        head = f"Script timed out: {error.message}"
    elif error.kind == "aborted":
        head = f"Script aborted: {error.message}"
    else:
        head = f"Script sandbox failed: {error.message}"
    summary = (
        "No tool calls were made."
        if len(calls) == 0
        else "Tool calls made before the failure (they are not undone): "
        + ", ".join(f"{call['name']} ({call['status']})" for call in calls)
    )
    return f"{head}\n\n{summary}"


@dataclass(slots=True)
class FormattedResult:
    """The tool result's content, and the path of the complete text when it was cut."""

    content: list[_Item]
    full_output_path: str | None = None


async def format_result(
    result: CodemodeResult,
    *,
    generated_images: int,
    wall_ms: float,
    max_output_tokens: int,
    calls: Sequence[Any],
    limits: ToolOutputLimits,
    write: OutputFile,
) -> FormattedResult:
    """The tool result's content for a script's result.

    Args:
        generated_images: Images ``models.generate_images()`` returned; when the script showed none, the result says so.
        calls: The code mode call's rows (``name``, ``status``), for a failure's summary.
    """
    output: list[CodemodeOutputItem] = list(result.output)
    # A Python script cannot tell `return None` from no return: neither adds a value.
    if isinstance(result, CodemodeSuccess) and result.value is not None:
        output.append(CodemodeTextOutput(value_text(result.value)))
    items = _format_items(output)
    if isinstance(result, CodemodeFailure):
        items.append(TextContent(f"Script error:\n{_format_error(result, calls)}"))
    if generated_images > 0 and not any(isinstance(item, ImageContent) for item in items):
        plural = "" if generated_images == 1 else "s"
        items.append(
            TextContent(
                f"Note: models.generate_images() returned {generated_images} image{plural} that the script did not "
                'show. Show each image block of result["output"] with image(block).'
            )
        )
    truncated = await _truncate(_join_adjacent_text(items), max_output_tokens, limits, write)
    # After truncation, which joins the text items and moves images after them, so each path stays next to its image.
    content = _join_adjacent_text(await _save_images(truncated.items, write))
    status = "Script completed" if result.ok else "Script failed"
    header = f"{status}\nWall time {wall_ms / 1000:.1f} seconds\nOutput:\n"
    return FormattedResult([TextContent(header), *content], truncated.full_output_path)
