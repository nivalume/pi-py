"""The ``read`` tool (port of ``tools/read.ts``).

Arguments keep JavaScript's number semantics: ``offset`` and ``limit`` are used as floats, so a fractional,
negative, huge, or NaN value selects the same lines, and prints the same, as upstream.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypedDict

from pi_ai import ImageContent, TextContent, Type
from pi_chord.output import character_end

from ..env.decode import range_decoder, starts_with_bom
from ..env.types import get_or_throw
from ..harness.define import define_tool
from ..truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    TruncationTotals,
    format_size,
    truncate_head_of,
    utf8_byte_length,
)
from ._js import is_safe_integer, js_min, js_number, js_trunc, js_truthy
from .env import require_env
from .image import ByteSource, detect_supported_image_mime_type_of, image_dimensions
from .image_processor import (
    DEFAULT_IMAGE_LIMITS,
    INLINE_IMAGE_TYPES,
    ImageLimits,
    ImageProcessor,
    base64_length,
    to_base64,
)
from .path_utils import resolve_read_tool_path

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..env.types import BinaryReader, FileInfo, LineScan, ScanLinesOptions
    from ..harness.types import ToolDiagnostic, ToolExecutionApi, ToolExecutionResult, ToolRegistration

__all__ = ["ReadToolDetails", "ReadToolInput", "ReadTruncation", "create_read_tool"]

_READ_SCHEMA = Type.Object(
    {
        "path": Type.String(description="Path to the file to read (relative or absolute)"),
        "offset": Type.Optional(Type.Number(description="Line number to start reading from (1-indexed)")),
        "limit": Type.Optional(Type.Number(description="Maximum number of lines to read")),
    }
)


class ReadToolInput(TypedDict):
    path: str
    offset: NotRequired[float]
    limit: NotRequired[float]


class ReadTruncation(TypedDict):
    """A :class:`~pi_durable.truncate.TruncationResult` without ``content``."""

    truncated: bool
    truncatedBy: Literal["lines", "bytes"] | None
    totalLines: int
    totalBytes: int
    outputLines: int
    outputBytes: int
    lastLinePartial: bool
    firstLineExceedsLimit: bool
    maxLines: int
    maxBytes: int


class ReadToolDetails(TypedDict, total=False):
    #: How the shown text was cut; the text itself is the result content.
    truncation: ReadTruncation


_READ_CHUNK: Final = 64 * 1024


def _slice_index(value: float) -> float:
    """``Array.prototype.slice``'s conversion of an index: NaN is 0, other values truncate toward zero."""
    return 0.0 if math.isnan(value) else js_trunc(value)


async def _read_head(reader: BinaryReader, start: int, end: int, skip_bom: bool, context: Context) -> str:
    """The decoded start of bytes ``[start, end)`` of the file, decoded as part of the whole file.

    All of it, or enough for ``truncate_head_of``: more than ``DEFAULT_MAX_BYTES + 1`` bytes, or
    ``DEFAULT_MAX_LINES`` newlines.
    """
    decoder = range_decoder()
    text = ""
    newlines = 0
    position = 3 if skip_bom and start == 0 else start
    while position < end:
        data = get_or_throw(await reader.read(position, min(_READ_CHUNK, end - position), context))
        if len(data) == 0:
            break
        position += len(data)
        decoded = decoder.decode(data, False)
        text += decoded
        newlines += decoded.count("\n")
        if newlines >= DEFAULT_MAX_LINES or utf8_byte_length(text) > DEFAULT_MAX_BYTES + 1:
            return text
    return text + decoder.decode(b"", True)


def _without_content(result: TruncationResult) -> ReadTruncation:
    rest = dict(result)
    del rest["content"]
    return rest  # type: ignore[return-value]


def create_read_tool(*, images: ImageProcessor | None = None) -> ToolRegistration:
    """Reads text files and images.

    Remarks about truncation, continuation, and image processing are diagnostics; the content is only
    file text, or one image, which programs get as its ``ImageContent``. ``images`` resizes and converts
    images; without one, images within ``DEFAULT_IMAGE_LIMITS["maxBytes"]`` pass through as they are.
    """
    formats = "jpg, png, gif, webp" if images is None else "jpg, png, gif, webp, bmp"

    async def execute(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        path: str = args["path"]
        raw_offset = args.get("offset")
        raw_limit = args.get("limit")
        offset = None if raw_offset is None else float(raw_offset)
        limit = None if raw_limit is None else float(raw_limit)
        env = require_env(api)
        absolute_path = await resolve_read_tool_path(env, path, context)
        reader = get_or_throw(await env.open_binary_reader(absolute_path, None, context))
        try:

            async def read_bytes(position: int, length: int) -> bytes:
                return get_or_throw(await reader.read(position, length, context))

            # A concurrent writer can change the file between the scan and the reads. Appending (a growing log)
            # leaves the scanned text as it was; an image must be whole. Otherwise the file is read again once.
            attempt = 0
            while True:
                before = get_or_throw(await reader.info(context))
                mime_type = await detect_supported_image_mime_type_of(ByteSource(before["size"], read_bytes))
                if mime_type is None:
                    result = await _read_text(reader, path, offset, limit, context)
                else:
                    result = await _read_image(
                        reader, before, path, mime_type, images, await _image_model(api, context), context
                    )
                after = get_or_throw(await reader.info(context))
                unchanged = after["size"] == before["size"] and after["mtimeMs"] == before["mtimeMs"]
                if unchanged or (mime_type is None and after["size"] > before["size"]):
                    return result
                if attempt == 1:
                    raise RuntimeError(f"{path} changed while it was read")
                attempt += 1
        finally:
            await reader.close(context)

    return define_tool(
        name="read",
        description=(
            f"Read the contents of a file. Supports text files and images ({formats}). Images are sent as "
            f"attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. When you "
            "need the full file, continue with offset until complete."
        ),
        parameters=_READ_SCHEMA,
        execute=execute,
    )


class _ImageModel(TypedDict):
    """What the conversation's model takes: whether images, and within which limits."""

    vision: bool
    limits: ImageLimits


async def _read_image(
    reader: BinaryReader,
    info: FileInfo,
    path: str,
    mime_type: str,
    processor: ImageProcessor | None,
    model: _ImageModel,
    context: Context,
) -> ToolExecutionResult:
    """One image block, prepared by ``processor`` or, without one, as it is when it is inline and fits.

    Without a processor its size is checked before reading it, its dimensions from its header; otherwise an
    error result says why. What happened to the image is an ``info`` diagnostic, so programs get the bare
    ``ImageContent``. A model without image input sees pi-ai's placeholder instead; a diagnostic says so.
    """
    limits = model["limits"]
    notes: list[str] = []
    if processor is not None:
        data = get_or_throw(await reader.read(0, info["size"], context))
        prepared = await processor.prepare(data, mime_type, limits)
        if prepared is None:
            return _image_error(f"{path} is an image ({mime_type}) that cannot be prepared for the model")
        image_data, image_type = prepared["data"], prepared["mimeType"]
        converted_from = prepared.get("convertedFrom")
        if converted_from is not None:
            notes.append(f"Converted from {converted_from} to {image_type}.")
        resized = prepared.get("resized")
        if resized is not None:
            source, target = resized["from"], resized["to"]
            scale = _to_fixed_2(source["width"] / target["width"])
            notes.append(
                f"Resized from {source['width']}x{source['height']} to {target['width']}x{target['height']}. "
                f"Multiply coordinates by {scale} to map them to the original."
            )
    else:
        if mime_type not in INLINE_IMAGE_TYPES:
            return _image_error(
                f"{path} is an image ({mime_type}) that needs converting, and no image processor is configured"
            )
        unshrinkable = "and no image processor is configured to shrink it"
        if base64_length(info["size"]) > limits["maxBytes"]:
            return _image_error(
                f"{path} is an image ({mime_type}) of {format_size(info['size'])}, too large to send, {unshrinkable}"
            )
        data = get_or_throw(await reader.read(0, info["size"], context))
        size = image_dimensions(data, mime_type)
        if size is None:
            return _image_error(f"{path} is an image ({mime_type}) whose size cannot be read")
        if size["width"] > limits["maxWidth"] or size["height"] > limits["maxHeight"]:
            return _image_error(
                f"{path} is an image ({mime_type}) of {size['width']}x{size['height']}, larger than "
                f"{js_number(limits['maxWidth'])}x{js_number(limits['maxHeight'])}, {unshrinkable}"
            )
        image_data, image_type = to_base64(data), mime_type
    if not model["vision"]:
        notes.append("The current model does not support images; it sees a placeholder instead.")
    diagnostics: list[ToolDiagnostic] = [
        {"severity": "info", "code": "image", "message": " ".join([f"Read image file [{image_type}].", *notes])}
    ]
    return {"output": [ImageContent(image_data, image_type)], "diagnostics": diagnostics}


def _to_fixed_2(value: float) -> str:
    """JavaScript's ``toFixed(2)``: the exact binary value rounded half up."""
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


async def _image_model(api: ToolExecutionApi, context: Context) -> _ImageModel:
    """The conversation model's image input: ``DEFAULT_IMAGE_LIMITS`` overridden by its resize limits.

    Without a resolvable model, it counts as one that takes images within the defaults.
    """
    ref = (await api.agent(context)).get("model")
    model = None if ref is None else api.models.get_model(ref["provider"], ref["modelId"])
    limits = None if model is None else model.input_limits
    images = None if limits is None else limits.images
    resize = None if images is None else images.resize
    return {
        "vision": True if model is None else "image" in model.input,
        "limits": {
            "maxWidth": DEFAULT_IMAGE_LIMITS["maxWidth"]
            if resize is None or resize.max_width is None
            else resize.max_width,
            "maxHeight": DEFAULT_IMAGE_LIMITS["maxHeight"]
            if resize is None or resize.max_height is None
            else resize.max_height,
            "maxBytes": DEFAULT_IMAGE_LIMITS["maxBytes"]
            if resize is None or resize.max_bytes is None
            else resize.max_bytes,
        },
    }


def _image_error(message: str) -> ToolExecutionResult:
    return {
        "output": [],
        "isError": True,
        "diagnostics": [{"severity": "error", "code": "unsupported_image", "message": message}],
    }


async def _read_text(
    reader: BinaryReader,
    path: str,
    offset: float | None,
    limit: float | None,
    context: Context,
) -> ToolExecutionResult:
    """The read result for the opened file.

    It equals decoding the whole file, splitting it on ``\\n``, and bounding the selected lines with
    ``truncate_head``, while reading only one scan's worth of the file plus the head.
    """
    start_line = max(0.0, offset - 1) if offset is not None and js_truthy(offset) else 0.0
    start_line_display = start_line + 1
    # Lines are selected like `allLines.slice(startLine, endLine)`, which truncates fractional indices.
    slice_start = _slice_index(start_line)
    # One pass finds the line count and the selection; a selection past the last line ends with it, as `slice`
    # does, and an empty one (a zero or negative limit) is scanned as one line and then ignored. A start beyond
    # any file is scanned from 0 only to count lines; the offset check below then fails as before.
    scan_start = int(slice_start) if is_safe_integer(slice_start) else 0
    requested_end = None if limit is None else max(scan_start + 1, _slice_index(start_line + limit))
    scan_end = int(requested_end) if requested_end is not None and is_safe_integer(requested_end) else None

    async def scan_of(end_line: int | None) -> LineScan:
        options: ScanLinesOptions = {"startLine": scan_start}
        if end_line is not None:
            options["endLine"] = end_line
        return get_or_throw(await reader.scan_lines(options, context))

    scan = await scan_of(scan_end)
    total_file_lines = scan["newlines"] + 1
    if start_line >= total_file_lines:
        shown = "undefined" if offset is None else js_number(offset)
        raise RuntimeError(f"Offset {shown} is beyond end of file ({total_file_lines} lines total)")

    user_limited_lines: float | None = None
    selected_line_count = total_file_lines - int(slice_start)
    if limit is not None:
        end_line = js_min(start_line + limit, total_file_lines)
        user_limited_lines = end_line - start_line
        # `slice` counts a negative end from the end of the lines, which only the line count tells; scan again.
        relative_end = _slice_index(end_line)
        slice_end = max(total_file_lines + relative_end, 0) if relative_end < 0 else relative_end
        selected_line_count = int(max(0, slice_end - slice_start))
        if selected_line_count > 0 and relative_end < 0:
            scan = await scan_of(int(slice_end))
    empty = selected_line_count == 0
    # Counted like `truncate_head`: a trailing newline adds no line, and empty text has none.
    ends_with_newline = not empty and scan["lastLineStart"] == scan["end"] and scan["lastLineStart"] > scan["start"]
    totals: TruncationTotals = {
        "lines": 0 if empty or scan["selectedBytes"] == 0 else selected_line_count - (1 if ends_with_newline else 0),
        "bytes": 0 if empty else scan["selectedBytes"],
    }
    first_bytes = get_or_throw(await reader.read(0, 3, context))
    head = "" if empty else await _read_head(reader, scan["start"], scan["end"], starts_with_bom(first_bytes), context)

    truncated = truncate_head_of(head, totals)
    head_text = truncated["content"]
    truncation = _without_content(truncated)
    diagnostics: list[ToolDiagnostic] = []
    output_text = head_text
    details: ReadToolDetails | None = None
    if truncation["firstLineExceedsLimit"]:
        # Show the start of the line, cut at the byte limit on a character boundary. Like `allLines[startLine]`,
        # a fractional start line names no line.
        integral = start_line.is_integer()
        line_bytes = (head.split("\n")[0] if integral else "").encode("utf-8", errors="replace")
        line_size = scan["firstLineBytes"] if integral else 0
        end = character_end(line_bytes, DEFAULT_MAX_BYTES)
        output_text = line_bytes[:end].decode("utf-8", errors="replace")
        display = js_number(start_line_display)
        diagnostics.append(
            {
                "severity": "warn",
                "code": "truncated",
                "message": (
                    f"Line {display} is {format_size(line_size)}, exceeds the {format_size(DEFAULT_MAX_BYTES)} "
                    f"limit; showing its first {format_size(end)}. Use bash: sed -n '{display}p' {path} | "
                    f"tail -c +{end + 1}"
                ),
            }
        )
        details = {"truncation": {**truncation, "outputBytes": end, "outputLines": 1}}
    elif truncation["truncated"]:
        end_line_display = start_line_display + truncation["outputLines"] - 1
        next_offset = end_line_display + 1
        limit_text = "" if truncation["truncatedBy"] == "lines" else f" ({format_size(DEFAULT_MAX_BYTES)} limit)"
        diagnostics.append(
            {
                "severity": "info",
                "code": "truncated",
                "message": (
                    f"Showing lines {js_number(start_line_display)}-{js_number(end_line_display)} of "
                    f"{total_file_lines}{limit_text}. Use offset={js_number(next_offset)} to continue."
                ),
            }
        )
        details = {"truncation": truncation}
    elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
        remaining = total_file_lines - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        diagnostics.append(
            {
                "severity": "info",
                "message": f"{js_number(remaining)} more lines in file. Use offset={js_number(next_offset)} to continue.",
            }
        )

    output: list[TextContent | ImageContent] = [] if output_text == "" else [TextContent(output_text)]
    if details is None:
        return {"output": output, "diagnostics": diagnostics}
    return {"output": output, "details": details, "diagnostics": diagnostics}
