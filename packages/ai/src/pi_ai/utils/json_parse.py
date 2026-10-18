"""Tolerant JSON parsing for streamed tool-call arguments.

``parse_streaming_json`` mirrors pi's helper (best-effort parse of a JSON prefix).
:class:`StreamingJsonParser` is an incremental variant used by the provider
adapters: it keeps the lexer state between deltas, so each delta costs time
proportional to the delta (plus one C-level ``json.loads``) instead of rescanning
the whole argument text in Python.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "repair_json",
    "parse_json_with_repair",
    "parse_tool_call_arguments",
    "parse_streaming_json",
    "complete_partial_json",
    "StreamingJsonParser",
]

_VALID_ESCAPES = frozenset('"\\/bfnrt')
_HEX4 = re.compile(r"[0-9a-fA-F]{4}")
_lenient_decode = json.JSONDecoder(strict=False).decode
_strict_decode = json.JSONDecoder(strict=True).decode


def _escape_control(char: str) -> str:
    match char:
        case "\b":
            return "\\b"
        case "\f":
            return "\\f"
        case "\n":
            return "\\n"
        case "\r":
            return "\\r"
        case "\t":
            return "\\t"
        case _:
            return f"\\u{ord(char):04x}"


def repair_json(text: str) -> str:
    """Repair malformed JSON string literals.

    Escapes raw control characters inside strings and doubles backslashes before
    invalid escape characters.
    """
    out: list[str] = []
    in_string = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == '"':
            out.append(char)
            in_string = False
            index += 1
            continue
        if char == "\\":
            if index + 1 >= length:
                out.append("\\\\")
                index += 1
                continue
            next_char = text[index + 1]
            if next_char == "u" and _HEX4.fullmatch(text, index + 2, index + 6):
                out.append(text[index : index + 6])
                index += 6
                continue
            if next_char in _VALID_ESCAPES:
                out.append("\\" + next_char)
                index += 2
                continue
            out.append("\\\\")
            index += 1
            continue
        out.append(_escape_control(char) if ord(char) < 0x20 else char)
        index += 1
    return "".join(out)


def parse_json_with_repair(text: str) -> Any:
    """``json.loads`` that retries once with :func:`repair_json` applied."""
    try:
        return _strict_decode(text)
    except ValueError:
        repaired = repair_json(text)
        if repaired != text:
            return _strict_decode(repaired)
        raise


def parse_tool_call_arguments(text: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse the complete arguments text of a finished tool call.

    Returns ``(arguments, arguments_error)``. Unlike :func:`parse_streaming_json`,
    this never keeps a partially parsed prefix: text that is not a valid JSON object
    yields ``({}, error)`` where the error includes the text the model sent, so tool
    validation can report it back to the model.
    """
    if not text or not text.strip():
        return {}, None
    try:
        parsed = parse_json_with_repair(text)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return parsed, None
    except ValueError as error:
        return {}, f"{error}\n\nReceived arguments:\n{text}"


def parse_streaming_json(partial_json: str | None) -> Any:
    """Best-effort parse of possibly incomplete JSON. Always returns a value (``{}`` on failure)."""
    if not partial_json or not partial_json.strip():
        return {}
    try:
        return parse_json_with_repair(partial_json)
    except ValueError:
        pass
    parser = StreamingJsonParser()
    parser.feed(partial_json)
    value = parser.value(default=None)
    return {} if value is None else value


def complete_partial_json(partial_json: str) -> str | None:
    """Close an incomplete JSON prefix into parseable JSON text, or ``None`` if nothing is usable."""
    scanner = _Scanner()
    scanner.feed(partial_json)
    return scanner.complete(partial_json)


# ---------------------------------------------------------------------------
# Incremental scanner
# ---------------------------------------------------------------------------

# Frame states
_OBJ_KEY = 0  # after "{" or ",": expecting a key or "}"
_OBJ_IN_KEY = 1  # inside the key string
_OBJ_COLON = 2  # key done, expecting ":"
_OBJ_VALUE = 3  # after ":", expecting a value
_AFTER_VALUE = 4  # value done, expecting "," or the closer
_ARR_VALUE = 5  # after "[" or ",": expecting a value or "]"

_LITERALS = ("true", "false", "null")
_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_STRING_SPECIAL = re.compile(r'["\\]')
_SCALAR_END = re.compile(r"[\s,\]}]")


class _Frame:
    __slots__ = ("closer", "safe", "state")

    def __init__(self, closer: str, state: int, safe: int) -> None:
        self.closer = closer
        self.state = state
        #: Prefix length that is structurally valid for this container once closed.
        self.safe = safe


class _Scanner:
    """Resumable JSON lexer that tracks where a prefix can be safely cut and closed."""

    __slots__ = ("in_string", "invalid", "pos", "scalar_start", "stack", "string_is_key", "top_done")

    def __init__(self) -> None:
        self.pos = 0
        self.stack: list[_Frame] = []
        self.in_string = False
        self.string_is_key = False
        self.scalar_start = -1  # start of a number/literal token in progress
        self.top_done = False
        self.invalid = False

    def feed(self, text: str) -> None:
        if self.invalid:
            return
        pos = self.pos
        length = len(text)
        stack = self.stack
        while pos < length:
            if self.in_string:
                match = _STRING_SPECIAL.search(text, pos)
                if match is None:
                    pos = length
                    break
                index = match.start()
                if text[index] == "\\":
                    if index + 1 >= length:
                        pos = index  # wait for the escaped character
                        break
                    pos = index + 2
                    continue
                pos = index + 1
                self.in_string = False
                self._value_done(pos, is_key=self.string_is_key)
                continue

            if self.scalar_start >= 0:
                match = _SCALAR_END.search(text, pos)
                if match is None:
                    pos = length
                    break
                pos = match.start()
                self.scalar_start = -1
                self._value_done(pos, is_key=False)
                continue

            char = text[pos]
            if char in " \t\r\n":
                pos += 1
                continue
            frame = stack[-1] if stack else None
            if char == '"':
                if frame is not None and frame.state == _OBJ_KEY:
                    frame.state = _OBJ_IN_KEY
                    self.string_is_key = True
                else:
                    if not self._expect_value(frame):
                        break
                    self.string_is_key = False
                self.in_string = True
                pos += 1
                continue
            if char == "{" or char == "[":
                if not self._expect_value(frame):
                    break
                pos += 1
                stack.append(_Frame("}" if char == "{" else "]", _OBJ_KEY if char == "{" else _ARR_VALUE, pos))
                continue
            if char == "}" or char == "]":
                if frame is None or frame.closer != char or frame.state not in (_OBJ_KEY, _AFTER_VALUE, _ARR_VALUE):
                    self.invalid = True
                    break
                stack.pop()
                pos += 1
                self._value_done(pos, is_key=False)
                continue
            if char == ",":
                if frame is None or frame.state != _AFTER_VALUE:
                    self.invalid = True
                    break
                frame.state = _OBJ_KEY if frame.closer == "}" else _ARR_VALUE
                pos += 1
                continue
            if char == ":":
                if frame is None or frame.state != _OBJ_COLON:
                    self.invalid = True
                    break
                frame.state = _OBJ_VALUE
                pos += 1
                continue
            # number or literal
            if not self._expect_value(frame):
                break
            self.scalar_start = pos
            pos += 1
        self.pos = pos

    def _expect_value(self, frame: _Frame | None) -> bool:
        if frame is None:
            if self.top_done:
                self.invalid = True
                return False
            return True
        if frame.state in (_OBJ_VALUE, _ARR_VALUE):
            return True
        self.invalid = True
        return False

    def _value_done(self, end: int, *, is_key: bool) -> None:
        if not self.stack:
            self.top_done = True
            return
        frame = self.stack[-1]
        if is_key:
            frame.state = _OBJ_COLON
        else:
            frame.state = _AFTER_VALUE
            frame.safe = end

    def complete(self, text: str) -> str | None:
        """Return ``text`` cut and closed into parseable JSON (``None`` if nothing usable)."""
        if self.invalid:
            return None
        stack = self.stack
        if not stack:
            if self.top_done:
                return text
            if self.in_string:
                return _close_string(text)
            if self.scalar_start >= 0:
                return _complete_scalar(text[self.scalar_start :])
            return None

        frame = stack[-1]
        suffix = "".join(f.closer for f in reversed(stack))
        if self.in_string:
            if self.string_is_key:
                return text[: frame.safe] + suffix
            return _close_string(text) + suffix
        if self.scalar_start >= 0:
            scalar = _complete_scalar(text[self.scalar_start :])
            if scalar is None:
                return text[: frame.safe] + suffix
            return text[: self.scalar_start] + scalar + suffix
        if frame.state == _AFTER_VALUE:
            return text + suffix
        return text[: frame.safe] + suffix


def _close_string(text: str) -> str:
    # Drop a dangling escape so the closing quote is not escaped.
    end = len(text)
    backslashes = 0
    while end - backslashes - 1 >= 0 and text[end - backslashes - 1] == "\\":
        backslashes += 1
    if backslashes % 2 == 1:
        text = text[:-1]
    else:
        unicode_start = text.rfind("\\u", max(0, len(text) - 5))
        if unicode_start != -1 and len(text) - unicode_start < 6:
            preceding = 0
            while unicode_start - preceding - 1 >= 0 and text[unicode_start - preceding - 1] == "\\":
                preceding += 1
            if preceding % 2 == 0:
                text = text[:unicode_start]
    return text + '"'


def _complete_scalar(token: str) -> str | None:
    if not token:
        return None
    first = token[0]
    if first in "tfn":
        for literal in _LITERALS:
            if literal.startswith(token):
                return literal
        return None
    match = _NUMBER_RE.match(token)
    if match is None:
        return None
    return match.group(0)


class StreamingJsonParser:
    """Incrementally parse a growing JSON document.

    ::

        parser = StreamingJsonParser()
        for delta in deltas:
            parser.feed(delta)
            partial_arguments = parser.value()

    :meth:`value` returns the best parse of the text so far. When the current prefix
    cannot be parsed (for example invalid escapes the repair pass cannot fix), the
    last successful value is returned instead of an empty object, so partial
    arguments never regress during streaming.
    """

    __slots__ = ("_dirty", "_last", "_scanner", "_text")

    def __init__(self) -> None:
        self._text = ""
        self._scanner = _Scanner()
        self._last: Any = None
        self._dirty = False

    @property
    def text(self) -> str:
        return self._text

    def feed(self, delta: str) -> None:
        if not delta:
            return
        self._text += delta
        self._scanner.feed(self._text)
        self._dirty = True

    def value(self, default: Any = ...) -> Any:
        if not self._dirty:
            if self._last is not None:
                return self._last
            return {} if default is ... else default
        self._dirty = False
        text = self._text
        scanner = self._scanner
        # While the scanner knows the document is still open, a full parse can only fail.
        if not scanner.invalid and (scanner.stack or scanner.in_string or not scanner.top_done):
            parsed = _FAIL
        else:
            parsed = _try_parse(text)
        if parsed is _FAIL:
            completed = self._scanner.complete(text)
            if completed is not None:
                parsed = _try_parse(completed)
                if parsed is _FAIL:
                    parsed = _try_parse(repair_json(completed))
        if parsed is not _FAIL:
            self._last = parsed
        if self._last is None:
            return {} if default is ... else default
        return self._last


_FAIL: Any = object()


def _try_parse(text: str) -> Any:
    try:
        return _lenient_decode(text)
    except ValueError:
        return _FAIL
