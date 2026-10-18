"""Bounded output streams and adaptive progress commits.

Port of ``@earendil-works/chord/output``: :class:`OutputBuffer` keeps a head or tail window of a
stream at a cost proportional to each chunk, and :class:`Progress` paces commits of a changing value
by their size.

Sizes are UTF-8 byte counts, exactly as upstream measures them on JavaScript strings. Python ``str``
indexes code points where JavaScript indexes UTF-16 code units; the module never mixes the two, since
every cut is made on the UTF-8 encoding at a code-point boundary. Text that is not well formed (lone
surrogate code points, which a Python ``str`` can hold) is measured and encoded the way JavaScript's
``TextEncoder`` treats the same UTF-16 sequence: an adjacent high/low surrogate pair is one astral
character (four bytes) and every other surrogate is U+FFFD (three bytes). Cuts fall on code points,
not grapheme clusters, so a combining mark or a ZWJ sequence can be split from its base, as upstream.
"""

from __future__ import annotations

import asyncio
import codecs
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "PROGRESS_BYTES_PER_SECOND",
    "BoundedOutput",
    "OutputBuffer",
    "OutputLimits",
    "OutputSkip",
    "OutputSlice",
    "Progress",
    "ProgressClock",
    "bound_output",
    "character_end",
    "sanitize_output",
    "utf8_byte_length",
]


@dataclass(frozen=True, slots=True)
class OutputLimits:
    """Retention limits of one output stream."""

    max_bytes: int
    max_lines: int
    retain: Literal["head", "tail"]


@dataclass(frozen=True, slots=True)
class BoundedOutput:
    """Retained output and what the limits dropped."""

    text: str
    dropped_bytes: int
    dropped_lines: int


@dataclass(frozen=True, slots=True)
class OutputSlice:
    """An exact slice of the input within the limits, and what it left out."""

    text: str
    bytes: int
    dropped_bytes: int
    dropped_lines: int


@dataclass(frozen=True, slots=True)
class OutputSkip:
    """Output a producer omitted before a chunk, measured on the decoded text.

    Every U+FFFD counts as three bytes, and no sanitizing is applied.
    """

    #: UTF-8 byte length of the omitted text.
    bytes: int
    #: Newlines (U+000A) in the omitted text.
    newlines: int
    #: Whether the omitted text ends with a newline.
    ends_with_newline: bool


_NEWLINE = b"\n"
_INVALID_OUTPUT = re.compile("[\\x00-\\x08\\x0b-\\x1f\\ufff9-\\ufffb]")
#: The C0 controls `sanitize_output` removes: all but tab (0x09) and newline (0x0A). In UTF-8 they are single bytes
#: that never occur inside a multi-byte sequence, so they can be deleted from encoded text directly.
_CONTROL_BYTES = bytes([*range(0x00, 0x09), *range(0x0B, 0x20)])
#: U+FFF9..U+FFFB (interlinear annotation controls) in UTF-8; 0xEF is always a lead byte, so matches are aligned.
_ANNOTATION_PREFIX = b"\xef\xbf"
_ANNOTATIONS = (b"\xef\xbf\xb9", b"\xef\xbf\xba", b"\xef\xbf\xbb")


def _well_formed(text: str) -> str:
    """``text`` with surrogates resolved as a UTF-16 decoder would: pairs combine, lone ones become U+FFFD."""
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _encode_checked(text: str) -> tuple[bytes, bool]:
    """UTF-8 encoding of ``text`` with JavaScript ``TextEncoder`` semantics, and whether ``text`` was well formed."""
    try:
        return text.encode("utf-8"), True
    except UnicodeEncodeError:
        return _well_formed(text).encode("utf-8"), False


def _encode(text: str) -> bytes:
    """UTF-8 encoding of ``text`` with JavaScript ``TextEncoder`` semantics for surrogates."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return _well_formed(text).encode("utf-8")


def _decode(data: bytes | memoryview) -> str:
    # Slices decode exactly: a U+FEFF at a slice's start is text, not a byte-order mark.
    return str(data, "utf-8", "replace")


def _sanitize_utf8(data: bytes, original: str | None) -> str:
    """:func:`sanitize_output` of the text ``data`` encodes; returns ``original`` itself when nothing is removed."""
    cleaned = data.translate(None, _CONTROL_BYTES)
    if _ANNOTATION_PREFIX in cleaned:
        for annotation in _ANNOTATIONS:
            cleaned = cleaned.replace(annotation, b"")
    if original is not None and len(cleaned) == len(data):
        return original
    return _decode(cleaned)


def sanitize_output(text: str) -> str:
    """Remove control characters that break display and transcripts; tabs and newlines stay."""
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError:
        # Lone surrogates stay, as in the JavaScript string upstream sanitizes.
        return _INVALID_OUTPUT.sub("", text)
    return _sanitize_utf8(data, text)


def bound_output(text: str, limits: OutputLimits) -> OutputSlice:
    """Bound ``text`` to whole lines within the limits.

    Keeps the first lines for ``head`` and the last lines for ``tail``. The result is an exact
    slice, trailing newline included. A single line longer than ``max_bytes`` is cut at the byte
    limit on a character boundary.
    """
    data = _encode(text)
    start, end = _range(data, limits)
    size = len(data)
    kept_size = end - start
    if kept_size == size:
        return OutputSlice(text, size, 0, 0)
    dropped_lines = _line_count(data, 0, size) - _line_count(data, start, end)
    return OutputSlice(_decode(memoryview(data)[start:end]), kept_size, size - kept_size, dropped_lines)


def _range(data: bytes, limits: OutputLimits) -> tuple[int, int]:
    return _head_range(data, limits) if limits.retain == "head" else _tail_range(data, limits)


def _head_range(data: bytes, limits: OutputLimits) -> tuple[int, int]:
    max_lines = limits.max_lines
    max_bytes = limits.max_bytes
    if max_lines == 0 or max_bytes == 0:
        return 0, 0
    size = len(data)
    window = min(size, max_bytes)
    # The `max_lines`-th newline matters only when it falls inside the byte window; past it, the byte cut decides.
    if data.count(_NEWLINE, 0, window) >= max_lines:
        index = data.find(_NEWLINE)
        for _ in range(max_lines - 1):
            index = data.find(_NEWLINE, index + 1)
        return 0, index + 1
    if size <= max_bytes:
        return 0, size
    newline = data.rfind(_NEWLINE, 0, max_bytes)
    return 0, character_end(data, max_bytes) if newline == -1 else newline + 1


def _tail_range(data: bytes, limits: OutputLimits) -> tuple[int, int]:
    max_lines = limits.max_lines
    max_bytes = limits.max_bytes
    size = len(data)
    if max_lines == 0 or max_bytes == 0:
        return size, size
    # A trailing newline ends the last line rather than starting another.
    last = size - 2 if size > 0 and data[size - 1] == 0x0A else size - 1
    start = 0
    # The line start matters only when it falls inside the byte window; before it, the byte cut decides.
    low = max(0, size - max_bytes - 1)
    if last >= 0 and data.count(_NEWLINE, low, last + 1) >= max_lines:
        index = data.rfind(_NEWLINE, 0, last + 1)
        for _ in range(max_lines - 1):
            index = data.rfind(_NEWLINE, 0, index)
        start = index + 1
    if size - start > max_bytes:
        cut = size - max_bytes
        newline = data.find(_NEWLINE, cut - 1)
        # The first line starting inside the byte window, or a cut of the last line when it alone is too long.
        start = newline + 1 if newline != -1 and newline + 1 < size else _character_start(data, cut)
    return start, size


def character_end(data: bytes | bytearray | memoryview, index: int) -> int:
    """The last UTF-8 character boundary at or before byte ``index`` of ``data``."""
    end = index
    size = len(data)
    while end > 0 and end < size and (data[end] & 0xC0) == 0x80:
        end -= 1
    return end


def _character_start(data: bytes, index: int) -> int:
    """The first UTF-8 character boundary at or after byte ``index``."""
    start = index
    size = len(data)
    while start < size and (data[start] & 0xC0) == 0x80:
        start += 1
    return start


def _line_count(data: bytes, start: int, end: int) -> int:
    """Lines in ``data[start:end]``; a final unterminated line counts."""
    if end <= start:
        return 0
    return data.count(_NEWLINE, start, end) + (0 if data[end - 1] == 0x0A else 1)


@dataclass(slots=True)
class _Chunk:
    text: str
    bytes: int
    newlines: int


class OutputBuffer:
    """Bounded running output of one stream, such as a tool call's or a process's.

    Accepting a chunk costs time proportional to the chunk: head retention stops storing once the
    window is full, and tail retention drops stored text the window no longer needs when it
    snapshots. Counts of the whole stream are kept so the dropped totals stay exact.

    Byte chunks are decoded as a stream: a string chunk or a skip ends an incomplete character of
    earlier bytes (it becomes U+FFFD), and the next byte chunk then starts a new character. Only a
    byte-order mark at the very start of the output is dropped, never a U+FEFF later in it.
    """

    __slots__ = (
        "_chunks",
        "_decoder",
        "_ends_with_newline",
        "_full",
        "_limits",
        "_started",
        "_stored_bytes",
        "_stored_newlines",
        "_total_bytes",
        "_total_newlines",
    )

    def __init__(self, limits: OutputLimits) -> None:
        self._limits = limits
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._started = False
        #: Stored chunks: for head the start of the stream, for tail a suffix that still contains the next window.
        self._chunks: list[_Chunk] = []
        self._stored_bytes = 0
        self._stored_newlines = 0
        self._full = False
        self._total_bytes = 0
        self._total_newlines = 0
        self._ends_with_newline = True

    @property
    def stored_bytes(self) -> int:
        """Bytes currently held; bounded by the limits plus one chunk."""
        return self._stored_bytes

    def push(self, chunk: str | bytes | bytearray | memoryview, skipped: OutputSkip | None = None) -> bool:
        """Accept a chunk; returns whether anything was accepted.

        ``skipped`` is output omitted right before the chunk, which must be more than the tail
        window by at least one byte or line; only tail retention accepts it.
        """
        is_text = isinstance(chunk, str)
        # Bytes of an incomplete character from an earlier byte chunk come first.
        pending = self._decoder.decode(b"", True) if is_text or skipped is not None else ""
        text = chunk if isinstance(chunk, str) else self._decoder.decode(chunk, False)
        first = not self._started and pending == "" and skipped is None
        if pending != "" or text != "" or skipped is not None:
            self._started = True
        if first and not is_text and text.startswith("\ufeff"):
            text = text[1:]
        if skipped is None:
            return self._accept(pending + text)
        if self._limits.retain != "tail":
            raise ValueError("Skipped output requires tail retention")
        self._accept(pending)
        self._skip(skipped)
        self._accept(text)
        return True

    def _skip(self, skipped: OutputSkip) -> None:
        """Count omitted output; nothing stored before it can be in the window once the text after it arrives."""
        if skipped.bytes == 0:
            return
        self._total_bytes += skipped.bytes
        self._total_newlines += skipped.newlines
        self._ends_with_newline = skipped.ends_with_newline
        self._chunks = []
        self._stored_bytes = 0
        self._stored_newlines = 0

    def end(self) -> None:
        """Flush an incomplete trailing character as a replacement character; call when the stream ends."""
        self._accept(self._decoder.decode(b"", True))

    def _accept(self, text: str) -> bool:
        if not text:
            return False
        size = utf8_byte_length(text)
        newlines = text.count("\n")
        self._total_bytes += size
        self._total_newlines += newlines
        self._ends_with_newline = text[-1] == "\n"
        if self._full:
            return True
        limits = self._limits
        chunks = self._chunks
        chunks.append(_Chunk(text, size, newlines))
        self._stored_bytes += size
        self._stored_newlines += newlines
        if limits.retain == "head":
            # Nothing past a full window is ever needed.
            self._full = self._stored_bytes > limits.max_bytes or self._stored_newlines >= limits.max_lines
            return True
        # Drop leading chunks while the rest still holds more than a window: more than `max_bytes` bytes or
        # `max_lines` newlines, plus one, so the window's line start can still be found. Each chunk is dropped once.
        drop = 0
        stored_bytes = self._stored_bytes
        stored_newlines = self._stored_newlines
        while len(chunks) - drop > 1:
            first = chunks[drop]
            bytes_after = stored_bytes - first.bytes
            newlines_after = stored_newlines - first.newlines
            if bytes_after <= limits.max_bytes + 1 and newlines_after <= limits.max_lines + 1:
                break
            drop += 1
            stored_bytes = bytes_after
            stored_newlines = newlines_after
        if drop:
            del chunks[:drop]
            self._stored_bytes = stored_bytes
            self._stored_newlines = stored_newlines
        return True

    def snapshot(self) -> BoundedOutput:
        """Retained, sanitized output and what the limits dropped from the whole stream."""
        limits = self._limits
        chunks = self._chunks
        stored = chunks[0].text if len(chunks) == 1 else "".join([chunk.text for chunk in chunks])
        # One encoding serves the window, the sanitized text and the compaction below.
        data, well_formed = _encode_checked(stored)
        size = len(data)
        start, end = _range(data, limits)
        if end - start == size:
            text = _sanitize_utf8(data, stored) if well_formed else _INVALID_OUTPUT.sub("", stored)
        else:
            text = _sanitize_utf8(data[start:end], None)
        kept_bytes = end - start
        kept_lines = _line_count(data, start, end)
        # Tail windows never reach back before this one, but finding a later window's first line needs what precedes
        # it: keep the shortest suffix longer than the window by a byte or a line, as `_accept` does.
        if limits.retain == "tail":
            margin = _tail_margin(data, limits)
            if margin == 0:
                compacted, compacted_bytes, compacted_newlines = stored, size, self._stored_newlines
            else:
                compacted = _decode(memoryview(data)[margin:])
                compacted_bytes = size - margin
                compacted_newlines = data.count(_NEWLINE, margin)
            self._compact(compacted, compacted_bytes, compacted_newlines)
        elif len(chunks) > 1:
            self._compact(stored, self._stored_bytes, self._stored_newlines)
        return BoundedOutput(
            text,
            self._total_bytes - kept_bytes,
            _lines(self._total_newlines, self._ends_with_newline) - kept_lines,
        )

    def _compact(self, text: str, size: int, newlines: int) -> None:
        self._chunks = [_Chunk(text, size, newlines)] if text else []
        self._stored_bytes = size
        self._stored_newlines = newlines


def _tail_margin(data: bytes, limits: OutputLimits) -> int:
    """Start of the shortest suffix of ``data`` with more than ``max_bytes`` bytes or more than ``max_lines`` newlines.

    Returns 0 (all of it) when there is none. The tail window of any text that ends with this
    suffix, followed by anything, is the same as of the whole text followed by it.
    """
    size = len(data)
    byte_start = character_end(data, size - limits.max_bytes - 1) if size > limits.max_bytes else 0
    line_start = 0
    # The (max_lines + 1)-th newline from the end matters only when it falls after the byte start.
    if data.count(_NEWLINE, byte_start) > limits.max_lines:
        index = data.rfind(_NEWLINE)
        for _ in range(limits.max_lines):
            index = data.rfind(_NEWLINE, 0, index)
        line_start = index
    return max(byte_start, line_start)


def _lines(newlines: int, terminated: bool) -> int:
    """Lines of text with ``newlines`` newlines; a final unterminated line counts."""
    return newlines if terminated else newlines + 1


def utf8_byte_length(content: str) -> int:
    """UTF-8 byte length of ``content``.

    A lone surrogate counts as U+FFFD, three bytes; an adjacent high/low surrogate pair counts as
    the astral character it encodes in UTF-16, four bytes, matching JavaScript.
    """
    if content.isascii():
        return len(content)
    return len(_encode(content))


#: Each progress commit also buys a pause proportional to what it wrote.
PROGRESS_BYTES_PER_SECOND = 100 * 1024


@dataclass(frozen=True, slots=True)
class ProgressClock:
    """The clock and timer progress commits pace on."""

    #: Current time in milliseconds.
    now: Callable[[], float]
    #: Call ``callback`` once after about ``delay_ms``; returns a function that disarms it.
    set_timer: Callable[[Callable[[], None], float], Callable[[], None]]


def _system_now() -> float:
    return time.monotonic() * 1000


def _system_set_timer(callback: Callable[[], None], delay_ms: float) -> Callable[[], None]:
    handle = asyncio.get_running_loop().call_later(delay_ms / 1000, callback)
    return handle.cancel


#: Monotonic milliseconds and the running event loop's ``call_later``.
_SYSTEM_CLOCK = ProgressClock(now=_system_now, set_timer=_system_set_timer)


class Progress:
    """Adaptive progress commits.

    The first change after an idle period commits at once; each commit then delays the next by at
    least ``min_interval_ms`` and by its written size at 100 KiB/s. At most one commit is in
    flight; changes made meanwhile coalesce into the next one. Paces on ``clock``, by default a
    monotonic clock and the running event loop's timers.

    ``write`` performs one commit and returns the number of bytes it wrote. Methods that may start
    a commit (:meth:`mark`, :meth:`mark_and_wait`, timers) must run inside an event loop.
    """

    __slots__ = (
        "_clock",
        "_dirty",
        "_in_flight",
        "_min_interval_ms",
        "_next_at",
        "_on_error",
        "_stopped",
        "_timer",
        "_waiters",
        "_write",
    )

    def __init__(
        self,
        write: Callable[[], Awaitable[int]],
        on_error: Callable[[BaseException], None],
        min_interval_ms: float,
        clock: ProgressClock | None = None,
    ) -> None:
        self._write = write
        self._on_error = on_error
        self._min_interval_ms = min_interval_ms
        self._clock = _SYSTEM_CLOCK if clock is None else clock
        self._waiters: list[asyncio.Future[None]] = []
        self._timer: Callable[[], None] | None = None
        self._in_flight: asyncio.Task[None] | None = None
        self._next_at: float = 0
        self._dirty = False
        self._stopped = False

    def mark(self) -> None:
        """Schedule a commit."""
        self._dirty = True
        self._schedule()

    def mark_and_wait(self) -> asyncio.Future[None]:
        """Schedule a commit; the future settles with the commit that includes this change."""
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        self.mark()
        return waiter

    async def stop(self) -> list[asyncio.Future[None]]:
        """Stop committing and wait for the commit in flight; returns the waiters the final commit must settle."""
        self._stopped = True
        if self._timer is not None:
            disarm, self._timer = self._timer, None
            disarm()
        if self._in_flight is not None:
            await self._in_flight
        waiters, self._waiters = self._waiters, []
        return waiters

    def _schedule(self) -> None:
        if self._stopped or self._timer is not None or self._in_flight is not None:
            return
        wait = self._next_at - self._clock.now()
        if wait <= 0:
            self._flush()
        else:
            self._timer = self._clock.set_timer(self._on_timer, wait)

    def _on_timer(self) -> None:
        self._timer = None
        # Checks the clock again: a timer that fires early waits for the rest.
        self._schedule()

    def _flush(self) -> None:
        if self._stopped or not self._dirty:
            return
        self._dirty = False
        waiters, self._waiters = self._waiters, []
        started = self._clock.now()
        self._in_flight = asyncio.ensure_future(self._commit(waiters, started))

    async def _commit(self, waiters: list[asyncio.Future[None]], started: float) -> None:
        try:
            try:
                written = await self._write()
            except Exception as error:
                self._next_at = started + self._min_interval_ms
                for waiter in waiters:
                    if not waiter.done():
                        waiter.set_exception(error)
                self._on_error(error)
            except BaseException:
                self._next_at = started + self._min_interval_ms
                for waiter in waiters:
                    waiter.cancel()
                raise
            else:
                self._next_at = started + max(self._min_interval_ms, written * 1000 / PROGRESS_BYTES_PER_SECOND)
                for waiter in waiters:
                    if not waiter.done():
                        waiter.set_result(None)
        finally:
            self._in_flight = None
            if self._dirty:
                self._schedule()
