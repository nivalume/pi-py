"""Port of ``test/tools-read-differential.test.ts``.

The ``read`` tool before bounded reads is kept as the reference: it decodes the whole file, splits it into
lines, and truncates the selection, with JavaScript's ``slice``/``Math`` semantics for fractional, negative,
huge, and NaN offsets and limits. The tool must give the same result while reading only part of the file.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pi_ai import ImageContent, TextContent
from pi_chord.context import BACKGROUND_CONTEXT, Context
from pi_chord.output import character_end
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.tools._js import js_min, js_number, js_truthy
from pi_durable.tools.image import detect_supported_image_mime_type
from pi_durable.tools.read import create_read_tool
from pi_durable.truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

context = BACKGROUND_CONTEXT
_M32 = 0xFFFFFFFF


def _js_slice_index(value: float, length: int) -> int:
    """``Array.prototype.slice``'s index conversion: NaN is 0, truncation, negative from the end, clamped."""
    if math.isnan(value):
        return 0
    if math.isinf(value):
        return 0 if value < 0 else length
    relative = math.trunc(value)
    return max(length + relative, 0) if relative < 0 else min(relative, length)


def _js_slice(items: list[str], start: float, end: float | None) -> list[str]:
    begin = _js_slice_index(start, len(items))
    final = len(items) if end is None else _js_slice_index(end, len(items))
    return items[begin:final]


def _decode(data: bytes) -> str:
    """``new TextDecoder().decode(data)``: replacement characters, a leading byte-order mark dropped."""
    text = data.decode("utf-8", errors="replace")
    return text[1:] if data.startswith(b"\xef\xbb\xbf") else text


def reference_read(data: bytes, path: str, offset: float | None, limit: float | None) -> dict[str, Any]:
    mime_type = detect_supported_image_mime_type(data)
    if mime_type:
        return {
            "output": [ImageContent(base64.b64encode(data).decode("ascii"), mime_type)],
            "diagnostics": [{"severity": "info", "code": "image", "message": f"Read image file [{mime_type}]."}],
        }
    all_lines = _decode(data).split("\n")
    total_file_lines = len(all_lines)
    start_line = max(0.0, offset - 1) if offset is not None and js_truthy(offset) else 0.0
    start_line_display = start_line + 1
    if start_line >= len(all_lines):
        raise RuntimeError(f"Offset {js_number(offset or 0)} is beyond end of file ({len(all_lines)} lines total)")
    user_limited_lines: float | None = None
    if limit is not None:
        end_line = js_min(start_line + limit, len(all_lines))
        selected_content = "\n".join(_js_slice(all_lines, start_line, end_line))
        user_limited_lines = end_line - start_line
    else:
        selected_content = "\n".join(_js_slice(all_lines, start_line, None))
    truncated = truncate_head(selected_content)
    truncation = {key: value for key, value in truncated.items() if key != "content"}
    diagnostics: list[Any] = []
    output_text = truncated["content"]
    details: Any = None
    if truncation["firstLineExceedsLimit"]:
        line = all_lines[int(start_line)] if start_line.is_integer() else ""
        line_bytes = line.encode("utf-8")
        end = character_end(line_bytes, DEFAULT_MAX_BYTES)
        output_text = line_bytes[:end].decode("utf-8", errors="replace")
        display = js_number(start_line_display)
        diagnostics.append(
            {
                "severity": "warn",
                "code": "truncated",
                "message": (
                    f"Line {display} is {format_size(len(line_bytes))}, exceeds the {format_size(DEFAULT_MAX_BYTES)} "
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
    elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
        remaining = len(all_lines) - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        diagnostics.append(
            {
                "severity": "info",
                "message": f"{js_number(remaining)} more lines in file. Use offset={js_number(next_offset)} to continue.",
            }
        )
    result: dict[str, Any] = {"output": [] if output_text == "" else [TextContent(output_text)]}
    if details is not None:
        result["details"] = details
    result["diagnostics"] = diagnostics
    return result


def random(seed: int) -> Callable[[], float]:
    """Mulberry32, as upstream: the same files for the same seeds."""
    state = seed & _M32

    def imul(a: int, b: int) -> int:
        return (a * b) & _M32

    def next_value() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & _M32
        t = state
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & _M32
        return ((t ^ (t >> 14)) & _M32) / 4294967296

    return next_value


PIECES = [
    [0x0A],
    [0x0A],
    [0x61],
    [0x62, 0x63, 0x64, 0x65],
    [0xEF, 0xBB, 0xBF],
    [0xC3, 0xA9],
    [0xE2, 0x82, 0xAC],
    [0xF0, 0x9F, 0x98, 0x80],
    [0xE2, 0x82],
    [0xFF],
    [0x0D, 0x0A],
]


def random_file(next_value: Callable[[], float]) -> bytes:
    """Mostly small files, some over the line limit, some over the byte limit, some with one huge line."""
    kind = next_value()
    if kind < 0.1:
        return b"\xef\xbb\xbf" + b"x\n" * 2100
    if kind < 0.2:
        line = bytearray(b"a" * (DEFAULT_MAX_BYTES + 200 + math.floor(next_value() * 400)))
        line[DEFAULT_MAX_BYTES - 1 + math.floor(next_value() * 3)] = 0xC3
        return bytes(line) + b"\nb"
    if kind < 0.3:
        return (b"0123456789" * 30 + b"\n") * 200
    data = bytearray()
    for _ in range(math.floor(next_value() * 80)):
        data.extend(PIECES[math.floor(next_value() * len(PIECES))])
    return bytes(data)


NAN = math.nan
OFFSETS: list[float | None] = [None, 0, 1, 2, 3, 2.5, -4, 50, 2001, 3000, 1e20, NAN]
LIMITS: list[float | None] = [None, 0, 1, 2, -3, -1e20, 1.5, 7, 2000, 2500, 1e20, NAN]


class Api:
    def __init__(self, env: LocalExecutionEnv) -> None:
        self.env = env
        self.output_window = None

    def output(self, chunk: Any, skipped: Any = None) -> None:
        return None

    def diagnostic(self, diagnostic: Any) -> None:
        return None

    async def details(self, value: Any, call_context: Context) -> None:
        return None

    async def agent(self, call_context: Context) -> Any:
        return {}


async def outcome(run: Callable[[], Any]) -> Any:
    try:
        result = run()
        if hasattr(result, "__await__"):
            result = await result
        return result
    except Exception as error:
        return {"error": str(error)}


def _same(actual: Any, expected: Any) -> bool:
    """Equality with NaN equal to itself (``toEqual``)."""
    if isinstance(actual, float) and isinstance(expected, float) and math.isnan(actual) and math.isnan(expected):
        return True
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same(actual[key], expected[key]) for key in actual)  # pyright: ignore
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected, strict=True))  # pyright: ignore
    return actual == expected


async def test_returns_exactly_what_reading_the_whole_file_returned(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    tool = create_read_tool()
    for seed in range(1, 401):
        next_value = random(seed)
        file = random_file(next_value)
        (tmp_path / "f.txt").write_bytes(file)
        for _ in range(4):
            offset = OFFSETS[math.floor(next_value() * len(OFFSETS))]
            limit = LIMITS[math.floor(next_value() * len(LIMITS))]
            args: dict[str, Any] = {"path": "f.txt"}
            if offset is not None:
                args["offset"] = offset
            if limit is not None:
                args["limit"] = limit
            actual = await outcome(lambda args=args: tool.execute(args, Api(env), context))  # type: ignore[arg-type]
            expected = await outcome(
                lambda file=file, offset=offset, limit=limit: reference_read(file, "f.txt", offset, limit)
            )
            assert _same(actual, expected), f"seed {seed} offset {offset} limit {limit}: {actual!r} != {expected!r}"


async def test_detects_animated_pngs_whose_actl_chunk_lies_far_beyond_the_header(tmp_path: Path) -> None:
    def chunk(type: str, length: int) -> bytes:
        return length.to_bytes(4, "big") + type.encode("ascii") + bytes(length + 4)

    png = b"\x89PNG\r\n\x1a\n" + chunk("IHDR", 13) + chunk("iCCP", 200_000) + chunk("acTL", 8) + chunk("IDAT", 10)
    (tmp_path / "a.png").write_bytes(png)
    env = LocalExecutionEnv(cwd=str(tmp_path))
    actual = await outcome(lambda: create_read_tool().execute({"path": "a.png"}, Api(env), context))  # type: ignore[arg-type]
    assert _same(actual, await outcome(lambda: reference_read(png, "a.png", None, None)))
