"""Port of chord ``test/output.test.ts``."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from pi_chord.output import (
    BoundedOutput,
    OutputBuffer,
    OutputLimits,
    Progress,
    ProgressClock,
    bound_output,
    character_end,
    sanitize_output,
    utf8_byte_length,
)


def head(max_lines: int, max_bytes: int = 1000) -> OutputLimits:
    return OutputLimits(max_bytes=max_bytes, max_lines=max_lines, retain="head")


def tail(max_lines: int, max_bytes: int = 1000) -> OutputLimits:
    return OutputLimits(max_bytes=max_bytes, max_lines=max_lines, retain="tail")


def js_utf8(content: str) -> bytes:
    """``Buffer.from(content, "utf8")`` of the UTF-16 string ``content`` stands for: pairs join, lone surrogates are U+FFFD."""
    return content.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace").encode("utf-8")


def buffer_tail(content: str, max_bytes: int) -> str:
    data = js_utf8(content)
    if len(data) <= max_bytes:
        return content
    start = len(data) - max_bytes
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    return data[start:].decode("utf-8")


def assert_matches_buffer_tail(text: str, max_byte_values: list[int] | None = None) -> None:
    """A single line longer than the byte limit is cut like Buffer's tail slice on a character boundary."""
    total_bytes = len(js_utf8(text))
    values = max_byte_values if max_byte_values is not None else list(range(total_bytes + 5))
    for max_bytes in values:
        kept = bound_output(text, OutputLimits(max_bytes=max_bytes, max_lines=10, retain="tail")).text
        expected = buffer_tail(text, max_bytes)
        assert kept == expected, f"tail mismatch input={text!r} maxBytes={max_bytes}"
        assert len(js_utf8(kept)) <= max_bytes, f"tail output exceeded {max_bytes} bytes"


def sampled_byte_limits(text: str) -> list[int]:
    total = len(js_utf8(text))
    candidates = [0, 1, 2, 3, 4, 5, 8, total // 2, total - 4, total - 1, total, total + 1]
    return sorted({value for value in candidates if value >= 0})


def bound(text: str, limits: OutputLimits) -> dict[str, object]:
    result = bound_output(text, limits)
    return {"kept": result.text, "droppedBytes": result.dropped_bytes, "droppedLines": result.dropped_lines}


def out(text: str, dropped_bytes: int, dropped_lines: int) -> dict[str, object]:
    return {"kept": text, "droppedBytes": dropped_bytes, "droppedLines": dropped_lines}


# --- tool output bounds ---


def test_removes_control_characters_but_keeps_tabs_newlines_and_other_text() -> None:
    assert sanitize_output("a\0b\tc\nd\re\u0007f\ufff9g\ufffbh😀") == "ab\tc\ndefgh😀"


def test_keeps_output_within_the_limits_unchanged() -> None:
    assert bound("a\nb\n", head(2)) == out("a\nb\n", 0, 0)
    assert bound("a\nb", tail(2)) == out("a\nb", 0, 0)
    assert bound("", tail(2)) == out("", 0, 0)


def test_keeps_nothing_with_a_zero_limit() -> None:
    assert bound("ab\ncd\n", head(10, 0)) == out("", 6, 2)
    assert bound("ab\ncd\n", tail(0)) == out("", 6, 2)


def test_keeps_exact_slices_of_whole_lines_trailing_newline_included() -> None:
    assert bound("a\nb\nc\n", head(2)) == out("a\nb\n", 2, 1)
    assert bound("a\nb\nc\n", tail(2)) == out("b\nc\n", 2, 1)
    assert bound("a\nb\nc", tail(2)) == out("b\nc", 2, 1)
    # Blank lines are lines.
    assert bound("a\nb\nc\n\n", tail(3)) == out("b\nc\n\n", 2, 1)


def test_cuts_at_the_byte_limit_on_whole_lines_when_possible() -> None:
    assert bound("aa\nbb\ncc\n", head(10, 7)) == out("aa\nbb\n", 3, 1)
    assert bound("aa\nbb\ncc\n", tail(10, 7)) == out("bb\ncc\n", 3, 1)


def test_cuts_a_single_line_longer_than_the_byte_limit_on_a_character_boundary() -> None:
    # "é" is two bytes; five bytes hold two whole characters.
    assert bound("ééé\n", head(10, 5)) == out("éé", 3, 0)
    assert bound("x\néééé", tail(10, 5)) == out("éé", 6, 1)


def test_keeps_a_ufeff_at_the_start_of_a_kept_slice() -> None:
    assert bound("x\ufeffa", tail(10, 4)) == out("\ufeffa", 1, 0)


def test_cuts_tails_of_surrogate_edge_cases_exactly_like_buffer() -> None:
    # Python strings hold surrogate code points individually; they are measured as the UTF-16 sequence they spell.
    inputs = ["a\ud83d", "\ude42b", "a\ude42b", "\ud83d🙂", "🙂\ude42", "👩\u200d💻"]
    for text in inputs:
        assert_matches_buffer_tail(text)


def test_cuts_tails_exactly_like_buffer_across_deterministic_fuzz_cases() -> None:
    alphabet = [
        "a",
        "\u007f",
        "\u0080",
        "é",
        "߿",
        "ࠀ",
        "中",
        "퟿",
        "\ud800",
        "\ud83d",
        "\udc00",
        "\ude42",
        "🙂",
        "",
        "\uffff",
    ]

    def check_exhaustive(prefix: str, depth: int) -> None:
        assert_matches_buffer_tail(prefix, sampled_byte_limits(prefix))
        if depth == 0:
            return
        for character in alphabet:
            check_exhaustive(prefix + character, depth - 1)

    check_exhaustive("", 3)
    seed = 0x12345678

    def random() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / 0x100000000

    for _ in range(1_000):
        length = int(random() * 80)
        text = "".join(alphabet[int(random() * len(alphabet))] for _ in range(length))
        assert_matches_buffer_tail(text, sampled_byte_limits(text))


def test_utf8_byte_length_matches_javascript_for_surrogates() -> None:
    assert utf8_byte_length("") == 0
    assert utf8_byte_length("abc") == 3
    assert utf8_byte_length("é€😀") == 2 + 3 + 4
    assert utf8_byte_length("\ud83d") == 3
    assert utf8_byte_length("😀") == 4
    assert utf8_byte_length("\ude00\ud83d") == 6


def test_character_end_finds_the_last_boundary_at_or_before_an_index() -> None:
    data = "a€".encode()
    assert [character_end(data, index) for index in range(6)] == [0, 1, 1, 1, 4, 5]


# --- OutputBuffer ---


def test_keeps_exact_totals_across_chunks_and_decodes_utf8_split_across_byte_chunks() -> None:
    buffer = OutputBuffer(tail(2))
    data = "😀\n".encode()
    buffer.push("a\nb\n")
    buffer.push(data[:2])
    buffer.push(data[2:])
    assert buffer.snapshot() == BoundedOutput("b\n😀\n", 2, 1)


def test_sanitizes_the_retained_text_but_counts_the_raw_stream() -> None:
    buffer = OutputBuffer(tail(1))
    buffer.push("a\u0007\n")
    assert buffer.snapshot() == BoundedOutput("a\n", 0, 0)
    buffer.push("b\u001b\n")
    assert buffer.snapshot() == BoundedOutput("b\n", 3, 1)


def test_drops_a_byte_order_mark_only_at_the_start_of_the_output() -> None:
    bom = bytes([0xEF, 0xBB, 0xBF])
    buffer = OutputBuffer(tail(10))
    buffer.push(bom[:1])
    buffer.push(bytes([0xBB, 0xBF, 0x61]))
    buffer.push("b")
    # A U+FEFF after a string chunk is text.
    buffer.push(bom + b"c")
    buffer.end()
    assert buffer.snapshot().text == "ab\ufeffc"


def test_flushes_an_incomplete_character_before_a_string_chunk_and_at_the_end() -> None:
    buffer = OutputBuffer(tail(10))
    euro = "€".encode()
    buffer.push(euro[:1])
    buffer.push("x")
    buffer.push(euro[:2])
    buffer.end()
    assert buffer.snapshot().text == "\ufffdx\ufffd"


def test_accepts_bytearray_and_memoryview_chunks() -> None:
    buffer = OutputBuffer(tail(10))
    data = "é€".encode()
    buffer.push(bytearray(data[:3]))
    buffer.push(memoryview(data)[3:])
    assert buffer.snapshot() == BoundedOutput("é€", 0, 0)


def test_matches_bounding_the_whole_stream_when_several_chunks_arrive_between_snapshots() -> None:
    for limits in [head(3, 40), tail(3, 40), head(50, 25), tail(50, 25)]:
        buffer = OutputBuffer(limits)
        stream = ""
        for index in range(300):
            chunk = f"{'é' * (index % 30)}\n" if index % 7 == 0 else f"line {index}\n"
            stream += chunk
            buffer.push(chunk)
            if index % 5 != 4:
                continue
            expected = bound_output(stream, limits)
            assert buffer.snapshot() == BoundedOutput(expected.text, expected.dropped_bytes, expected.dropped_lines)


def test_stops_storing_head_output_once_the_window_is_full() -> None:
    buffer = OutputBuffer(head(2))
    for index in range(1000):
        buffer.push(f"line {index}\n")
    assert buffer.stored_bytes < 20
    assert buffer.snapshot() == BoundedOutput("line 0\nline 1\n", 8876, 998)


def test_stores_only_the_tail_window_after_each_snapshot() -> None:
    buffer = OutputBuffer(tail(3, 100))
    stream = ""
    for index in range(2000):
        chunk = f"line {index}\n\n"
        stream += chunk
        buffer.push(chunk)
        snapshot = buffer.snapshot()
        assert buffer.stored_bytes <= 100
        assert snapshot.text == bound_output(stream, tail(3, 100)).text


# --- Progress ---


async def settle() -> None:
    """Let committed tasks and their callbacks run (``advanceTimersByTimeAsync`` flushes microtasks the same way)."""
    for _ in range(10):
        await asyncio.sleep(0)


@dataclass
class _Timer:
    delay_ms: float
    callback: Callable[[], None]
    disarmed: bool = False


@dataclass
class FakeClock:
    """A controllable clock and timer queue, in place of vitest's fake timers."""

    now_ms: float = 0
    _queue: list[tuple[float, int, _Timer]] = field(default_factory=list)
    _order: itertools.count[int] = field(default_factory=itertools.count)

    def clock(self) -> ProgressClock:
        def set_timer(callback: Callable[[], None], delay_ms: float) -> Callable[[], None]:
            timer = _Timer(delay_ms, callback)
            heapq.heappush(self._queue, (self.now_ms + delay_ms, next(self._order), timer))

            def disarm() -> None:
                timer.disarmed = True

            return disarm

        return ProgressClock(now=lambda: self.now_ms, set_timer=set_timer)

    async def advance(self, ms: float) -> None:
        target = self.now_ms + ms
        await settle()
        while self._queue and self._queue[0][0] <= target:
            due, _, timer = heapq.heappop(self._queue)
            if timer.disarmed:
                continue
            self.now_ms = due
            timer.callback()
            await settle()
        self.now_ms = target
        await settle()


async def test_commits_the_first_change_at_once_then_waits_at_least_100ms_and_the_written_size() -> None:
    fake = FakeClock()
    commits: list[float] = []
    size = 50 * 1024

    async def write() -> int:
        commits.append(fake.now_ms)
        return size

    progress = Progress(write, lambda _error: None, 100, fake.clock())
    progress.mark()
    await fake.advance(0)
    assert commits == [0]
    # 50 KiB buys 500 ms; changes meanwhile coalesce into one commit.
    size = 10
    progress.mark()
    progress.mark()
    await fake.advance(499)
    assert commits == [0]
    await fake.advance(1)
    assert commits == [0, 500]
    # A small commit still waits the minimum 100 ms.
    progress.mark()
    await fake.advance(99)
    assert commits == [0, 500]
    await fake.advance(1)
    assert commits == [0, 500, 600]


async def test_waits_the_configured_minimum_interval_between_small_commits() -> None:
    fake = FakeClock()
    commits: list[float] = []

    async def write() -> int:
        commits.append(fake.now_ms)
        return 10

    progress = Progress(write, lambda _error: None, 500, fake.clock())
    progress.mark()
    await fake.advance(0)
    progress.mark()
    await fake.advance(499)
    assert commits == [0]
    await fake.advance(1)
    assert commits == [0, 500]


async def test_paces_on_the_clock_it_is_given() -> None:
    now = 1_000.0
    timers: list[_Timer] = []
    commits: list[float] = []

    async def write() -> int:
        commits.append(now)
        return 10

    def set_timer(callback: Callable[[], None], delay_ms: float) -> Callable[[], None]:
        timer = _Timer(delay_ms, callback)
        timers.append(timer)

        def disarm() -> None:
            timer.disarmed = True

        return disarm

    progress = Progress(write, lambda _error: None, 100, ProgressClock(now=lambda: now, set_timer=set_timer))
    await progress.mark_and_wait()
    assert commits == [1_000]
    # Let the first commit finish settling.
    await settle()
    now = 1_040
    second = progress.mark_and_wait()
    assert [timer.delay_ms for timer in timers] == [60]
    # A timer that fires early commits nothing and waits for the rest.
    now = 1_080
    timers[0].callback()
    assert commits == [1_000]
    assert [timer.delay_ms for timer in timers] == [60, 20]
    now = 1_100
    timers[1].callback()
    await second
    assert commits == [1_000, 1_100]
    await settle()
    progress.mark()
    assert [timer.delay_ms for timer in timers] == [60, 20, 100]
    await progress.stop()
    assert timers[2].disarmed


async def test_paces_on_the_system_clock_by_default() -> None:
    commits: list[int] = []

    async def write() -> int:
        commits.append(len(commits))
        return 0

    progress = Progress(write, lambda _error: None, 20)
    await progress.mark_and_wait()
    second = progress.mark_and_wait()
    assert commits == [0]
    await asyncio.wait_for(second, 5)
    assert commits == [0, 1]
    await progress.stop()


async def test_rejects_the_waiters_of_a_failed_commit_and_reports_its_error() -> None:
    errors: list[BaseException] = []
    failure = RuntimeError("commit failed")

    async def write() -> int:
        raise failure

    progress = Progress(write, errors.append, 100)
    with pytest.raises(RuntimeError) as raised:
        await progress.mark_and_wait()
    assert raised.value is failure
    assert errors == [failure]


async def test_stops_waits_for_the_commit_in_flight_and_hands_back_waiters_no_commit_covered_yet() -> None:
    release = asyncio.Event()

    async def write() -> int:
        await release.wait()
        return 0

    progress = Progress(write, lambda _error: None, 100)
    first = progress.mark_and_wait()
    second = progress.mark_and_wait()
    stopped = asyncio.ensure_future(progress.stop())
    release.set()
    pending = await stopped
    await first
    assert len(pending) == 1
    pending[0].set_result(None)
    await second


def test_sanitizes_like_the_upstream_pattern_across_fuzz_cases() -> None:
    reference = re.compile("[\\x00-\\x08\\x0b-\\x1f\\ufff9-\\ufffb]")
    alphabet = ["a", "\n", "\t", "\r", "\x00", "\x1b", "\x7f", "é", "\ufff8", "\ufff9", "\ufffa", "\ufffb", "\ufffc"]
    alphabet += ["", "😀", "\ud83d", "\ude00"]
    seed = 7
    for _ in range(3_000):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        length = seed % 12
        text = ""
        for _ in range(length):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            text += alphabet[seed % len(alphabet)]
        assert sanitize_output(text) == reference.sub("", text), repr(text)
    clean = "nothing to remove\n\tsee"
    assert sanitize_output(clean) is clean
