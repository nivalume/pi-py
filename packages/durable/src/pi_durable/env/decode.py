"""UTF-8 decoding that matches decoding the whole input when it arrives in pieces (port of ``env/decode.ts``).

The reference is WHATWG ``new TextDecoder().decode(bytes)``: invalid sequences become U+FFFD (one per
maximal subpart, as Python's ``"replace"`` handler does) and a leading byte-order mark is dropped.
Python's incremental ``utf-8`` decoder never drops a U+FEFF, so these decoders drop a leading mark
themselves.
"""

from __future__ import annotations

import codecs

__all__ = ["StreamDecoder", "range_decoder", "starts_with_bom"]


def range_decoder() -> codecs.IncrementalDecoder:
    """A streaming decoder for a byte range that keeps every U+FEFF.

    Callers that start at the beginning of a file skip a leading mark themselves. Feed it with
    ``decode(chunk)`` and finish with ``decode(b"", final=True)``.
    """
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def starts_with_bom(first_bytes: bytes | bytearray | memoryview) -> bool:
    """Whether decoding the whole input drops its first three bytes as a byte-order mark."""
    return len(first_bytes) >= 3 and first_bytes[0] == 0xEF and first_bytes[1] == 0xBB and first_bytes[2] == 0xBF


class StreamDecoder:
    """Decodes one stream chunk by chunk exactly like decoding all of it at once."""

    __slots__ = ("_decoder", "_started")

    def __init__(self) -> None:
        self._decoder = range_decoder()
        self._started = False

    def decode(self, data: bytes | bytearray | memoryview | None = None) -> str:
        """Text for ``data``, holding back an incomplete character; without ``data``, the end of the stream."""
        text = self._decoder.decode(b"", True) if data is None else self._decoder.decode(data, False)
        if self._started or text == "":
            return text
        self._started = True
        # U+FEFF encodes only as EF BB BF, so a leading U+FEFF is exactly a byte-order mark at the stream's start.
        return text[1:] if text.startswith("﻿") else text
