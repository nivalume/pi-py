"""Line locations of a file fed in chunks (port of ``env/line-scan.ts``)."""

from __future__ import annotations

import codecs
import math

from pi_chord.output import utf8_byte_length

from .decode import range_decoder, starts_with_bom
from .types import LineScan

__all__ = ["LineScanner"]

_NEWLINE = 0x0A
_MAX_SAFE_INTEGER = 2**53 - 1


def _decoded_bytes(text: str) -> int:
    """UTF-8 byte length of decoded text."""
    return 0 if text == "" else utf8_byte_length(text)


def _is_safe_integer(value: object) -> bool:
    return type(value) is int and -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER


class LineScanner:
    """Computes a :class:`LineScan` from a file's bytes fed in order, in bounded memory.

    Decoded sizes are those of decoding the whole file at once (see :mod:`pi_durable.env.decode`).
    """

    __slots__ = (
        "_bom",
        "_end",
        "_end_line",
        "_first_line",
        "_first_line_bytes",
        "_first_line_end",
        "_head",
        "_last_line_start",
        "_line_start",
        "_newlines",
        "_position",
        "_selected_bytes",
        "_selection",
        "_start",
        "_start_line",
    )

    def __init__(self, start_line: int, end_line: int | None = None) -> None:
        """``start_line`` and ``end_line`` are non-negative integers with ``end_line > start_line``.

        ``end_line`` absent (``None``): to the end. Raises :class:`ValueError` for an invalid range.
        """
        end: float = math.inf if end_line is None else end_line
        if not _is_safe_integer(start_line) or start_line < 0 or not (end > start_line):
            raise ValueError("Invalid line range")
        if end_line is not None and not _is_safe_integer(end_line):
            raise ValueError("Invalid line range")
        self._start_line = start_line
        self._end_line = end
        self._position = 0
        self._newlines = 0
        self._line_start = 0
        self._start: int | None = None
        self._end: int | None = None
        self._first_line_end: int | None = None
        self._last_line_start: int | None = None
        self._selected_bytes = 0
        self._first_line_bytes = 0
        self._selection: codecs.IncrementalDecoder | None = None
        self._first_line: codecs.IncrementalDecoder | None = None
        # The first bytes, held until it is known whether they are a byte-order mark.
        self._head: bytearray | None = bytearray()
        self._bom = False
        if start_line == 0:
            self._begin(0)

    def push(self, chunk: bytes | bytearray | memoryview) -> None:
        data = bytes(chunk)
        if self._head is not None:
            take = min(3 - len(self._head), len(data))
            self._head += data[:take]
            if len(self._head) < 3:
                return
            self._release_head()
            data = data[take:]
        self._process(data)

    def _release_head(self) -> None:
        head = bytes(self._head or b"")
        self._head = None
        self._bom = starts_with_bom(head)
        self._process(head)

    def _process(self, chunk: bytes) -> None:
        base = self._position
        start = 0
        index = chunk.find(_NEWLINE)
        while index != -1:
            # The newline ends line `self._newlines`. It belongs to the selection between selected lines only.
            self._feed(chunk, base, start, index)
            line = self._newlines
            position = base + index
            if line == self._start_line:
                self._end_first_line(position)
            if line == self._end_line - 1:
                self._end_selection(position)
            self._feed(chunk, base, index, index + 1)
            start = index + 1
            self._newlines += 1
            self._line_start = position + 1
            if self._newlines == self._start_line:
                self._begin(self._line_start)
            if self._newlines == self._end_line - 1:
                self._last_line_start = self._line_start
            index = chunk.find(_NEWLINE, index + 1)
        self._feed(chunk, base, start, len(chunk))
        self._position += len(chunk)

    def finish(self) -> LineScan:
        if self._head is not None:
            self._release_head()
        size = self._position
        if self._start is None:
            return {
                "newlines": self._newlines,
                "start": size,
                "end": size,
                "firstLineEnd": size,
                "lastLineStart": size,
                "selectedBytes": 0,
                "firstLineBytes": 0,
            }
        if self._first_line_end is None:
            self._end_first_line(size)
        if self._end is None:
            self._end_selection(size)
        return {
            "newlines": self._newlines,
            "start": self._start,
            "end": size if self._end is None else self._end,
            "firstLineEnd": size if self._first_line_end is None else self._first_line_end,
            # A selection that reaches past the last line ends with the last line.
            "lastLineStart": self._line_start if self._last_line_start is None else self._last_line_start,
            "selectedBytes": self._selected_bytes,
            "firstLineBytes": self._first_line_bytes,
        }

    def _begin(self, start: int) -> None:
        self._start = start
        if self._start_line == self._end_line - 1:
            self._last_line_start = start
        self._selection = range_decoder()
        self._first_line = range_decoder()

    def _end_first_line(self, position: int) -> None:
        self._first_line_end = position
        if self._first_line is not None:
            self._first_line_bytes += _decoded_bytes(self._first_line.decode(b"", True))
        self._first_line = None

    def _end_selection(self, position: int) -> None:
        self._end = position
        if self._selection is not None:
            self._selected_bytes += _decoded_bytes(self._selection.decode(b"", True))
        self._selection = None

    def _feed(self, chunk: bytes, base: int, start: int, stop: int) -> None:
        """Feed ``chunk[start:stop]``, which starts at file offset ``base``, to the decoders of the open ranges."""
        # Decoding the whole file drops a leading byte-order mark.
        if self._bom and base + start < 3:
            start = min(stop, 3 - base)
        if stop <= start:
            return
        data = chunk[start:stop]
        if self._selection is not None:
            self._selected_bytes += _decoded_bytes(self._selection.decode(data, False))
        if self._first_line is not None:
            self._first_line_bytes += _decoded_bytes(self._first_line.decode(data, False))
