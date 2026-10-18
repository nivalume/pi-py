"""Port of ``test/env-line-scan.test.ts``.

The reference decoder is WHATWG ``new TextDecoder().decode()``: Python's ``bytes.decode("utf-8",
"replace")`` with a leading byte-order mark dropped.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pi_durable.env import LineScanner, StreamDecoder

MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & MASK


def random(seed: int) -> Callable[[], float]:
    """The test's mulberry32 generator, bit for bit."""
    state = seed & MASK

    def next_value() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & MASK
        t = state
        t = _imul(t ^ (t >> 15), t | 1)
        t ^= (t + _imul(t ^ (t >> 7), t | 61)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296

    return next_value


# Newlines, ASCII, a byte-order mark, valid multi-byte sequences, and bytes that form invalid or truncated sequences.
PIECES = [
    [0x0A],
    [0x0A],
    [0x61],
    [0x62, 0x63],
    [0xEF, 0xBB, 0xBF],
    [0xC3, 0xA9],
    [0xE2, 0x82, 0xAC],
    [0xF0, 0x9F, 0x98, 0x80],
    [0xE2, 0x82],
    [0xFF],
    [0x80],
    [0xF0, 0x9F],
    [0x0D, 0x0A],
]


def random_file(next_value: Callable[[], float]) -> bytes:
    data = bytearray()
    for _ in range(int(next_value() * 60)):
        data += bytes(PIECES[int(next_value() * len(PIECES))])
    return bytes(data)


def text_decode(data: bytes, ignore_bom: bool = False) -> str:
    """``new TextDecoder("utf-8", { ignoreBOM }).decode(data)``."""
    text = data.decode("utf-8", "replace")
    return text if ignore_bom or not text.startswith("﻿") else text[1:]


def utf8(text: str) -> int:
    return len(text.encode("utf-8"))


def test_stream_decoder_keeps_a_u_feff_that_does_not_start_the_stream() -> None:
    # Node's streaming TextDecoder with BOM handling dropped this U+FEFF, which follows an invalid sequence.
    data = bytes([0xE2, 0x82, 0xEF, 0xBB, 0xBF, 0x61])
    decoder = StreamDecoder()
    text = "".join(decoder.decode(bytes([byte])) for byte in data) + decoder.decode()
    assert text == text_decode(data)
    assert text == "�﻿a"


def test_stream_decoder_decodes_any_chunking_like_decoding_the_whole_stream() -> None:
    for seed in range(1, 5001):
        next_value = random(seed)
        data = random_file(next_value)
        decoder = StreamDecoder()
        text = ""
        offset = 0
        while offset < len(data):
            size = 1 + int(next_value() * 7)
            text += decoder.decode(data[offset : offset + size])
            offset += size
        assert text + decoder.decode() == text_decode(data), f"seed {seed}"


def test_line_scanner_agrees_with_decoding_and_splitting_the_whole_file() -> None:
    for seed in range(1, 5001):
        next_value = random(seed)
        file = random_file(next_value)
        lines = text_decode(file).split("\n")
        start_line = int(next_value() * (len(lines) + 2))
        end_line = None if next_value() < 0.3 else start_line + 1 + int(next_value() * (len(lines) + 1))
        scanner = LineScanner(start_line, end_line)
        offset = 0
        while offset < len(file):
            size = 1 + int(next_value() * 7)
            scanner.push(file[offset : offset + size])
            offset += size
        scan = scanner.finish()
        message = f"seed {seed}"
        assert scan["newlines"] == len(lines) - 1, message
        selected = lines[start_line:end_line]

        def decode(start: int, stop: int, file: bytes = file) -> str:
            return text_decode(file[start:stop], ignore_bom=start > 0)

        assert decode(scan["start"], scan["end"]) == "\n".join(selected), message
        assert scan["selectedBytes"] == utf8("\n".join(selected)), message
        if start_line < len(lines):
            assert decode(scan["start"], scan["firstLineEnd"]) == lines[start_line], message
            assert scan["firstLineBytes"] == utf8(lines[start_line]), message
            last_line = min(len(lines) if end_line is None else end_line, len(lines)) - 1
            last_end = file.find(0x0A, scan["lastLineStart"]) if last_line + 1 < len(lines) else len(file)
            assert decode(scan["lastLineStart"], last_end) == lines[last_line], message
        else:
            assert (scan["start"], scan["end"], scan["selectedBytes"]) == (len(file), len(file), 0), message


def test_line_scanner_rejects_empty_or_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        LineScanner(2, 2)
    with pytest.raises(ValueError):
        LineScanner(-1)
    with pytest.raises(ValueError):
        LineScanner(1.5)  # type: ignore[arg-type]
