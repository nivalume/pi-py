"""Image format detection and header dimensions, without decoding (port of ``tools/image.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from .image_processor import ImageSize

__all__ = ["ByteSource", "detect_supported_image_mime_type", "detect_supported_image_mime_type_of", "image_dimensions"]

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
#: Bytes every check except the APNG chunk walk needs: BMP reads up to offset 29.
_HEADER_BYTES: Final = 32
_BLOCK_BYTES: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class ByteSource:
    """Positional reads of a file of ``size`` bytes: ``read(offset, length)``."""

    size: int
    read: Callable[[int, int], Awaitable[bytes]]


async def detect_supported_image_mime_type_of(source: ByteSource) -> str | None:
    """``detect_supported_image_mime_type`` of a whole file.

    Reads only its header and, for PNG, the chunk headers up to the first ``acTL`` or ``IDAT``.
    """
    header = await source.read(0, _HEADER_BYTES)
    if not header.startswith(_PNG_SIGNATURE):
        return detect_supported_image_mime_type(header)
    return "image/png" if _is_png(header) and not await _is_animated_png_of(source) else None


async def _is_animated_png_of(source: ByteSource) -> bool:
    """``_is_animated_png`` over a file read in blocks."""
    block = b""
    block_start = 0

    async def bytes_at(offset: int, length: int) -> bytes:
        nonlocal block, block_start
        if offset < block_start or offset + length > block_start + len(block):
            block_start = offset
            block = await source.read(offset, _BLOCK_BYTES)
        return block[offset - block_start : offset - block_start + length]

    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= source.size:
        chunk_header = await bytes_at(offset, 8)
        chunk_length = _uint32_be(chunk_header, 0)
        if _ascii_at(chunk_header, 4, b"acTL"):
            return True
        if _ascii_at(chunk_header, 4, b"IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > source.size:
            return False
        offset = next_offset
    return False


def detect_supported_image_mime_type(buffer: bytes | bytearray | memoryview) -> str | None:
    """The MIME type of a PNG (not animated), JPEG, GIF, WebP, or BMP by its signature; ``None`` otherwise."""
    data = bytes(buffer)
    if data.startswith(b"\xff\xd8\xff"):
        return None if _byte(data, 3) == 0xF7 else "image/jpeg"
    if data.startswith(_PNG_SIGNATURE):
        return "image/png" if _is_png(data) and not _is_animated_png(data) else None
    if _ascii_at(data, 0, b"GIF87a") or _ascii_at(data, 0, b"GIF89a"):
        return "image/gif"
    if _ascii_at(data, 0, b"RIFF") and _ascii_at(data, 8, b"WEBP"):
        return "image/webp"
    if _ascii_at(data, 0, b"BM") and _is_bmp(data):
        return "image/bmp"
    return None


def _is_png(buffer: bytes) -> bool:
    return len(buffer) >= 16 and _uint32_be(buffer, len(_PNG_SIGNATURE)) == 13 and _ascii_at(buffer, 12, b"IHDR")


def _is_animated_png(buffer: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(buffer):
        chunk_length = _uint32_be(buffer, offset)
        chunk_type_offset = offset + 4
        if _ascii_at(buffer, chunk_type_offset, b"acTL"):
            return True
        if _ascii_at(buffer, chunk_type_offset, b"IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(buffer):
            return False
        offset = next_offset
    return False


def image_dimensions(data: bytes | bytearray | memoryview, mime_type: str) -> ImageSize | None:
    """The width and height an image of ``mime_type`` declares in its header, without decoding it.

    PNG's IHDR, GIF's screen, WebP's VP8, VP8L, or VP8X chunk, JPEG's first SOF segment; ``None`` when the
    header cannot be read.
    """
    buffer = bytes(data)
    if mime_type == "image/png":
        return None if len(buffer) < 24 else {"width": _uint32_be(buffer, 16), "height": _uint32_be(buffer, 20)}
    if mime_type == "image/gif":
        return None if len(buffer) < 10 else {"width": _uint16_le(buffer, 6), "height": _uint16_le(buffer, 8)}
    if mime_type == "image/webp":
        if len(buffer) < 30:
            return None
        if _ascii_at(buffer, 12, b"VP8 "):
            return {"width": _uint16_le(buffer, 26) & 0x3FFF, "height": _uint16_le(buffer, 28) & 0x3FFF}
        if _ascii_at(buffer, 12, b"VP8L"):
            bits = _uint32_le(buffer, 21)
            return {"width": (bits & 0x3FFF) + 1, "height": ((bits >> 14) & 0x3FFF) + 1}
        if _ascii_at(buffer, 12, b"VP8X"):

            def uint24(offset: int) -> int:
                return _uint16_le(buffer, offset) + (_byte(buffer, offset + 2) << 16)

            return {"width": uint24(24) + 1, "height": uint24(27) + 1}
        return None
    if mime_type == "image/jpeg":
        offset = 2
        while offset + 9 <= len(buffer):
            if buffer[offset] != 0xFF:
                return None
            marker = buffer[offset + 1]
            if marker == 0xFF:
                offset += 1
                continue
            # SOF0 to SOF15, except DHT (C4), JPG (C8), and DAC (CC), which share the range.
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                return {"width": _uint16_be(buffer, offset + 7), "height": _uint16_be(buffer, offset + 5)}
            offset += 2 + _uint16_be(buffer, offset + 2)
        return None
    return None


def _is_bmp(buffer: bytes) -> bool:
    if len(buffer) < 26:
        return False
    declared_file_size = _uint32_le(buffer, 2)
    pixel_data_offset = _uint32_le(buffer, 10)
    dib_header_size = _uint32_le(buffer, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False
    if dib_header_size == 12:
        color_planes = _uint16_le(buffer, 22)
        bits_per_pixel = _uint16_le(buffer, 24)
    elif 40 <= dib_header_size <= 124:
        if len(buffer) < 30:
            return False
        color_planes = _uint16_le(buffer, 26)
        bits_per_pixel = _uint16_le(buffer, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in (1, 4, 8, 16, 24, 32)


def _byte(buffer: bytes, offset: int) -> int:
    """``buffer[offset] ?? 0``."""
    return buffer[offset] if 0 <= offset < len(buffer) else 0


def _uint16_le(buffer: bytes, offset: int) -> int:
    return _byte(buffer, offset) + (_byte(buffer, offset + 1) << 8)


def _uint16_be(buffer: bytes, offset: int) -> int:
    return (_byte(buffer, offset) << 8) + _byte(buffer, offset + 1)


def _uint32_be(buffer: bytes, offset: int) -> int:
    return (
        (_byte(buffer, offset) << 24)
        + (_byte(buffer, offset + 1) << 16)
        + (_byte(buffer, offset + 2) << 8)
        + _byte(buffer, offset + 3)
    )


def _uint32_le(buffer: bytes, offset: int) -> int:
    return (
        _byte(buffer, offset)
        + (_byte(buffer, offset + 1) << 8)
        + (_byte(buffer, offset + 2) << 16)
        + (_byte(buffer, offset + 3) << 24)
    )


def _ascii_at(buffer: bytes, offset: int, text: bytes) -> bool:
    return len(buffer) >= offset + len(text) and buffer[offset : offset + len(text)] == text
