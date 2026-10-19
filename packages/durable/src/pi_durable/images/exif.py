"""EXIF orientation of JPEG and WebP files, read from the bytes without decoding the image (port of ``images/exif.ts``)."""

from __future__ import annotations

__all__ = ["exif_orientation"]


def _at(data: bytes, index: int) -> int:
    """``bytes[index]``, 0 past the end (JavaScript's ``undefined`` in bit operations)."""
    return data[index] if 0 <= index < len(data) else 0


def exif_orientation(data: bytes | bytearray | memoryview) -> int:
    """The EXIF orientation of a JPEG or WebP, 1 to 8; 1 when absent or unreadable."""
    buffer = bytes(data)
    tiff = _tiff_start(buffer)
    if tiff < 0 or tiff + 8 > len(buffer):
        return 1
    little = buffer[tiff] == 0x49 and buffer[tiff + 1] == 0x49

    def u16(at: int) -> int:
        if little:
            return _at(buffer, at) | (_at(buffer, at + 1) << 8)
        return (_at(buffer, at) << 8) | _at(buffer, at + 1)

    def u32(at: int) -> int:
        if little:
            return (
                _at(buffer, at) | (_at(buffer, at + 1) << 8) | (_at(buffer, at + 2) << 16) | (_at(buffer, at + 3) << 24)
            )
        return (_at(buffer, at) << 24) | (_at(buffer, at + 1) << 16) | (_at(buffer, at + 2) << 8) | _at(buffer, at + 3)

    directory = tiff + u32(tiff + 4)
    if directory + 2 > len(buffer):
        return 1
    entries = u16(directory)
    for index in range(entries):
        entry = directory + 2 + index * 12
        if entry + 12 > len(buffer):
            return 1
        # Tag 0x0112 is Orientation, a SHORT whose value sits in the entry.
        if u16(entry) == 0x0112:
            value = u16(entry + 8)
            return value if 1 <= value <= 8 else 1
    return 1


def _tiff_start(buffer: bytes) -> int:
    """Where the TIFF header of a JPEG's APP1 or a WebP's EXIF chunk starts; -1 when there is none."""
    if len(buffer) >= 2 and buffer[0] == 0xFF and buffer[1] == 0xD8:
        offset = 2
        while offset + 4 <= len(buffer):
            if buffer[offset] != 0xFF:
                return -1
            marker = buffer[offset + 1]
            if marker == 0xFF:
                offset += 1
                continue
            if marker == 0xE1 and _is_exif_header(buffer, offset + 4):
                return offset + 10
            offset += 2 + ((buffer[offset + 2] << 8) | buffer[offset + 3])
        return -1
    if _ascii(buffer, 0, b"RIFF") and _ascii(buffer, 8, b"WEBP"):
        offset = 12
        while offset + 8 <= len(buffer):
            size = int.from_bytes(buffer[offset + 4 : offset + 8], "little")
            # Some WebP files prefix the TIFF header with "Exif\0\0".
            if _ascii(buffer, offset, b"EXIF"):
                return offset + 14 if _is_exif_header(buffer, offset + 8) else offset + 8
            # RIFF chunks are padded to an even size.
            offset += 8 + size + (size % 2)
    return -1


def _is_exif_header(buffer: bytes, offset: int) -> bool:
    return _ascii(buffer, offset, b"Exif") and buffer[offset + 4 : offset + 6] == b"\0\0"


def _ascii(buffer: bytes, offset: int, text: bytes) -> bool:
    return buffer[offset : offset + len(text)] == text
