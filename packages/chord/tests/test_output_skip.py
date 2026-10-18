"""Port of chord ``test/output-skip.test.ts``."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pi_chord.output import OutputBuffer, OutputLimits, OutputSkip

_MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & _MASK


def random(seed: int) -> Callable[[], float]:
    """Deterministic PRNG (mulberry32) so failures reproduce from the printed seed."""
    state = seed & _MASK

    def next_value() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & _MASK
        t = state
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & _MASK)) & _MASK
        return ((t ^ (t >> 14)) & _MASK) / 4294967296

    return next_value


# Decoded shell output: ASCII, multi-byte and astral characters, replacement characters, CR/LF, tabs, and control
# characters that sanitizing removes after bounding.
ALPHABET = ["a", "b", "z", " ", "\n", "\n", "\n", "\r\n", "\t", "é", "€", "😀", "\ufffd", "\x01", "\x1b"]


def random_chunk(next_value: Callable[[], float]) -> str:
    length = int(next_value() * 12)
    return "".join(ALPHABET[int(next_value() * len(ALPHABET))] for _ in range(length))


def measure(text: str) -> OutputSkip:
    return OutputSkip(bytes=len(text.encode()), newlines=text.count("\n"), ends_with_newline=text.endswith("\n"))


def exceeds_window(text: str, limits: OutputLimits) -> bool:
    """Whether ``text`` proves that everything before it is outside the tail window."""
    skip = measure(text)
    return skip.bytes > limits.max_bytes or skip.newlines > limits.max_lines


def random_limits(next_value: Callable[[], float]) -> OutputLimits:
    return OutputLimits(
        max_bytes=1 + int(next_value() * 40),
        max_lines=1 + int(next_value() * 5),
        retain="tail",
    )


def check_seed(seed: int) -> None:
    """Feed random output to one buffer in full and to another through a reference skipper.

    The skipper follows the contract of ``ShellOutputInfo.skipped``: it holds undelivered text, and
    when it flushes, it may replace a prefix of it by counts if the rest exceeds the window.
    Snapshots must agree whenever both buffers have seen the same output.
    """
    next_value = random(seed)
    limits = random_limits(next_value)
    full = OutputBuffer(limits)
    skipping = OutputBuffer(limits)
    pending = ""

    def flush() -> None:
        nonlocal pending
        if pending == "":
            return
        # Python strings index code points, so every cut is a code-point boundary.
        cuts = [cut for cut in range(1, len(pending) + 1) if exceeds_window(pending[cut:], limits)]
        if cuts and next_value() < 0.7:
            cut = cuts[int(next_value() * len(cuts))]
            skipping.push(pending[cut:], measure(pending[:cut]))
        else:
            skipping.push(pending)
        pending = ""
        # Progress snapshots happen between deliveries and compact stored chunks.
        if next_value() < 0.5:
            skipping.snapshot()
        assert skipping.snapshot() == full.snapshot(), f"seed {seed}"

    chunks = int(next_value() * 40)
    for _ in range(chunks):
        chunk = random_chunk(next_value)
        full.push(chunk)
        if next_value() < 0.3:
            full.snapshot()
        pending += chunk
        if next_value() < 0.3:
            flush()
    flush()
    full.end()
    skipping.end()
    assert skipping.snapshot() == full.snapshot(), f"seed {seed}"


def test_keeps_the_same_tail_whenever_progress_snapshots_happen() -> None:
    # Progress commits snapshot at arbitrary moments; snapshots compact what is stored and must never change the
    # window. A compaction to exactly the kept window lost the character that decides where a later window's first
    # line starts.
    for seed in range(1, 3001):
        next_value = random(seed)
        limits = random_limits(next_value)
        plain = OutputBuffer(limits)
        sampled = OutputBuffer(limits)
        chunks = int(next_value() * 30)
        for _ in range(chunks):
            chunk = random_chunk(next_value)
            plain.push(chunk)
            sampled.push(chunk)
            if next_value() < 0.4:
                sampled.snapshot()
        assert sampled.snapshot() == plain.snapshot(), f"seed {seed}"


def test_matches_the_full_stream_for_every_legal_skip_pattern() -> None:
    for seed in range(1, 3001):
        check_seed(seed)


def test_counts_skipped_bytes_lines_and_the_final_newline_exactly() -> None:
    limits = OutputLimits(max_bytes=1000, max_lines=2, retain="tail")
    full = OutputBuffer(limits)
    skipping = OutputBuffer(limits)
    # The skipped text ends without a newline, so its last line continues in the delivered text.
    omitted = "one\ntwo\nthr"
    kept = "ee\nfour\nfive\nsix"
    full.push(omitted + kept)
    skipping.push(kept, measure(omitted))
    assert skipping.snapshot() == full.snapshot()
    snapshot = skipping.snapshot()
    assert (snapshot.text, snapshot.dropped_bytes, snapshot.dropped_lines) == ("five\nsix", 19, 4)


def test_refuses_skips_for_head_retention() -> None:
    buffer = OutputBuffer(OutputLimits(max_bytes=10, max_lines=2, retain="head"))
    with pytest.raises(ValueError, match="tail retention"):
        buffer.push("x\ny\nz\n", OutputSkip(bytes=3, newlines=1, ends_with_newline=True))
