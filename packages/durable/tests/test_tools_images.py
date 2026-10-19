"""Port of ``test/tools-images.test.ts``, with the Pillow image processor in place of Photon's.

Fixtures are built here (PNG with ``zlib``, JPEG with Pillow) instead of with Photon, and decoded with
Pillow. Cases that need Pillow skip without it; the cases without a processor run without Pillow.

Adapted: ``createNodePhotonImages`` is ``create_pillow_images``; "starts a new worker after the idle one
exited" is "starts a new worker after ``close()``" (an idle Python worker thread holds no image memory, so
there is no idle timeout). Dropped: "encodes a 3 MiB image within a 128 MB heap" (a V8 heap limit; Python's
base64 is C).
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import random
import struct
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pi_ai import ImageContent
from pi_ai.types import ModelImageInputLimits, ModelImageResizeOptions, ModelInputLimits
from pi_chord.context import BACKGROUND_CONTEXT, Context
from pi_durable.env import get_or_throw
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.tools import create_coding_tools
from pi_durable.tools.image import image_dimensions
from pi_durable.tools.image_processor import DEFAULT_IMAGE_LIMITS, ImageProcessor, to_base64
from pi_durable.tools.read import create_read_tool

context = BACKGROUND_CONTEXT
HAS_PILLOW = importlib.util.find_spec("PIL") is not None
needs_pillow = pytest.mark.skipif(not HAS_PILLOW, reason="needs Pillow (the images extra)")

Rgb = tuple[int, int, int]


def processor() -> ImageProcessor:
    from pi_durable.images import create_pillow_images

    return create_pillow_images()


def blocks(colors: list[list[Rgb]], block: int) -> tuple[bytes, int, int]:
    """A ``width`` by ``height`` image of ``block``-pixel squares coloured row by row from ``colors``."""
    width = len(colors[0]) * block
    height = len(colors) * block
    rgba = bytearray()
    for y in range(height):
        for x in range(width):
            rgba.extend((*colors[y // block][x // block], 255))
    return bytes(rgba), width, height


def pixels(width: int, height: int) -> bytes:
    """RGBA pixels of a ``width`` by ``height`` image: red, with a green top-left pixel."""
    rgba = bytearray(b"\xff\x00\x00\xff" * (width * height))
    rgba[0:4] = b"\x00\xff\x00\xff"
    return bytes(rgba)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_rgba(rgba: bytes, width: int, height: int, format: str) -> bytes:
    if format == "png":
        rows = b"".join(b"\x00" + rgba[y * width * 4 : (y + 1) * width * 4] for y in range(height))
        header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(rows, 6))
            + _png_chunk(b"IEND", b"")
        )
    from PIL import Image

    output = io.BytesIO()
    Image.frombytes("RGBA", (width, height), rgba).convert("RGB").save(output, "JPEG", quality=95)
    return output.getvalue()


def encode(width: int, height: int, format: str) -> bytes:
    return encode_rgba(pixels(width, height), width, height, format)


def noisy_png(width: int, height: int) -> bytes:
    """Noise, which neither PNG nor JPEG compresses well, so the byte limit decides."""
    rgba = bytearray(random.Random(1).randbytes(width * height * 4))
    rgba[3::4] = b"\xff" * (width * height)
    return encode_rgba(bytes(rgba), width, height, "png")


def bmp(width: int, height: int) -> bytes:
    """A 24-bit BMP, which providers do not take inline."""
    row = -(-(width * 3) // 4) * 4
    data = bytearray(54 + row * height)
    data[0:2] = b"BM"
    struct.pack_into("<I", data, 2, len(data))
    struct.pack_into("<I", data, 10, 54)
    struct.pack_into("<I", data, 14, 40)
    struct.pack_into("<i", data, 18, width)
    struct.pack_into("<i", data, 22, height)
    struct.pack_into("<H", data, 26, 1)
    struct.pack_into("<H", data, 28, 24)
    struct.pack_into("<I", data, 34, row * height)
    for y in range(height):
        for x in range(width):
            data[54 + y * row + x * 3 : 54 + y * row + x * 3 + 3] = b"\x00\x00\xff"
    return bytes(data)


def gif(width: int = 1, height: int = 1) -> bytes:
    """A 1x1 GIF with its logical screen size set to ``width`` by ``height``."""
    data = bytearray(base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"))
    struct.pack_into("<HH", data, 6, width, height)
    return bytes(data)


def tiff(orientation: int, little_endian: bool) -> bytes:
    """A TIFF header whose first directory holds only Orientation = ``orientation``."""
    order = "<" if little_endian else ">"
    data = bytearray(26)
    data[0:2] = b"II" if little_endian else b"MM"
    struct.pack_into(f"{order}H", data, 2, 42)
    struct.pack_into(f"{order}I", data, 4, 8)
    struct.pack_into(f"{order}H", data, 8, 1)
    struct.pack_into(f"{order}H", data, 10, 0x0112)
    struct.pack_into(f"{order}H", data, 12, 3)
    struct.pack_into(f"{order}I", data, 14, 1)
    struct.pack_into(f"{order}H", data, 18, orientation)
    return bytes(data)


EXIF_HEADER = b"Exif\0\0"


def with_orientation(jpeg: bytes, orientation: int, *, little: bool = False, app0: bool = False) -> bytes:
    """``jpeg`` with an APP1 segment whose EXIF says ``orientation``, after an APP0 segment when ``app0``."""
    exif = EXIF_HEADER + tiff(orientation, little)
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    app0_segment = bytes([0xFF, 0xE0, 0, 16, 0x4A, 0x46, 0x49, 0x46, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]) if app0 else b""
    return jpeg[:2] + app0_segment + app1 + jpeg[2:]


def webp_with_orientation(orientation: int, prefixed: bool) -> bytes:
    """A RIFF WebP with a dummy ``VP8 `` chunk and an ``EXIF`` chunk saying ``orientation``."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        padding = b"\0" if len(data) % 2 == 1 else b""
        return kind + struct.pack("<I", len(data)) + data + padding

    exif = (EXIF_HEADER if prefixed else b"") + tiff(orientation, True)
    body = b"WEBP" + chunk(b"VP8 ", b"\x01\x02\x03") + chunk(b"EXIF", exif)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def decoded_image(data: str) -> dict[str, Any]:
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        rgba = image.convert("RGBA")
        return {"width": rgba.width, "height": rgba.height, "rgba": rgba.tobytes()}


class RecordingReader:
    """An opened file whose positional reads are seen by ``on_read`` and recorded in ``reads``."""

    def __init__(self, inner: Any, reads: dict[str, int], on_read: Callable[[int, int], None] | None) -> None:
        self.inner = inner
        self.reads = reads
        self.on_read = on_read

    async def info(self, call_context: Context) -> Any:
        return await self.inner.info(call_context)

    async def scan_lines(self, options: Any, call_context: Context) -> Any:
        return await self.inner.scan_lines(options, call_context)

    async def close(self, call_context: Context) -> None:
        await self.inner.close(call_context)

    async def read(self, offset: int, length: int, call_context: Context) -> Any:
        self.reads["largest"] = max(self.reads["largest"], length)
        result = await self.inner.read(offset, length, call_context)
        if self.on_read is not None:
            self.on_read(offset, length)
        return result


def read_env(directory: Path, on_read: Callable[[int, int], None] | None = None) -> tuple[Any, dict[str, int]]:
    """The environment of a ``read`` call, with ``on_read`` seeing each positional read of an opened file."""
    reads = {"largest": 0}

    class Env(LocalExecutionEnv):
        async def open_binary_reader(self, path: str, options: Any, call_context: Context) -> Any:
            reader = get_or_throw(await super().open_binary_reader(path, options, call_context))
            return {"ok": True, "value": RecordingReader(reader, reads, on_read)}

    return Env(cwd=str(directory)), reads


class Api:
    def __init__(self, env: Any, model: Any) -> None:
        self.env = env
        self.models = SimpleNamespace(get_model=lambda provider, model_id: model)

    def output(self, chunk: Any, skipped: Any = None) -> None:
        return None

    def diagnostic(self, diagnostic: Any) -> None:
        return None

    async def details(self, value: Any, call_context: Context) -> None:
        return None

    async def agent(self, call_context: Context) -> Any:
        return {"model": {"provider": "test", "modelId": "test"}}


async def read(
    directory: Path,
    name: str,
    data: bytes,
    *,
    images: ImageProcessor | None = None,
    input: list[str] | None = None,
    resize: dict[str, int] | None = None,
    env: Any = None,
    tool: Any = None,
) -> Any:
    """Run ``read`` on ``data`` written to ``name``, for a conversation whose model takes ``input``."""
    (directory / name).write_bytes(data)
    limits = None
    if resize is not None:
        options = ModelImageResizeOptions(
            max_width=resize.get("maxWidth"), max_height=resize.get("maxHeight"), max_bytes=resize.get("maxBytes")
        )
        limits = ModelInputLimits(images=ModelImageInputLimits(resize=options))
    model = SimpleNamespace(input=input or ["text", "image"], input_limits=limits)
    api = Api(env or LocalExecutionEnv(cwd=str(directory)), model)
    read_tool = tool or create_read_tool(images=images)
    return await read_tool.execute({"path": name}, api, context)


def messages(result: Any) -> list[str]:
    return [diagnostic["message"] for diagnostic in result.get("diagnostics") or []]


def only_image(result: Any) -> ImageContent:
    assert len(result["output"]) == 1
    image = result["output"][0]
    assert isinstance(image, ImageContent)
    return image


# ─── without an image processor ──────────────────────────────────────────────


async def test_returns_a_supported_image_within_the_limits_as_it_is(tmp_path: Path) -> None:
    png = encode(8, 6, "png")
    result = await read(tmp_path, "a.png", png)
    assert result["output"] == [ImageContent(to_base64(png), "image/png")]
    assert messages(result) == ["Read image file [image/png]."]
    tiny = await read(tmp_path, "a.gif", gif())
    assert only_image(tiny) == ImageContent(to_base64(gif()), "image/gif")


async def test_refuses_an_image_too_large_to_send_without_reading_it(tmp_path: Path) -> None:
    env, reads = read_env(tmp_path)
    result = await read(tmp_path, "big.png", noisy_png(1100, 1100), env=env)
    assert result["output"] == []
    assert result["isError"] is True
    assert messages(result)[0].startswith("big.png is an image (image/png) of ")
    assert ", too large to send" in messages(result)[0]
    # Only the header and the chunk walk, in blocks of at most 64 KiB.
    assert reads["largest"] <= 64 * 1024


async def test_refuses_an_image_wider_or_taller_than_the_limits_by_its_header(tmp_path: Path) -> None:
    wide = await read(tmp_path, "wide.png", encode(2001, 2, "png"))
    assert wide["output"] == []
    assert wide["isError"] is True
    assert messages(wide) == [
        "wide.png is an image (image/png) of 2001x2, larger than 2000x2000, and no image processor is configured to "
        "shrink it"
    ]
    tall = await read(tmp_path, "tall.gif", gif(1, 3000))
    assert messages(tall)[0].startswith("tall.gif is an image (image/gif) of 1x3000, larger than 2000x2000")


async def test_refuses_a_format_that_needs_converting(tmp_path: Path) -> None:
    result = await read(tmp_path, "a.bmp", bmp(4, 4))
    assert result["output"] == []
    assert result["isError"] is True
    assert messages(result) == [
        "a.bmp is an image (image/bmp) that needs converting, and no image processor is configured"
    ]


async def test_says_when_the_model_sees_a_placeholder_instead_of_the_image(tmp_path: Path) -> None:
    result = await read(tmp_path, "a.png", encode(4, 4, "png"), input=["text"])
    assert len(result["output"]) == 1
    assert messages(result) == [
        "Read image file [image/png]. The current model does not support images; it sees a placeholder instead."
    ]


async def test_applies_the_models_image_limits_instead_of_the_defaults(tmp_path: Path) -> None:
    png = encode(8, 6, "png")
    refused = await read(tmp_path, "a.png", png, resize={"maxWidth": 4})
    assert "of 8x6, larger than 4x2000," in messages(refused)[0]
    if HAS_PILLOW:
        resized = await read(tmp_path, "a.png", png, resize={"maxWidth": 4}, images=processor())
        image = decoded_image(only_image(resized).data)
        assert (image["width"], image["height"]) == (4, 3)


class _NoImages:
    async def prepare(self, data: bytes, mime_type: str, limits: Any) -> Any:
        return None


def test_lists_bmp_in_the_description_only_with_a_processor() -> None:
    assert "(jpg, png, gif, webp)" in create_read_tool().description
    assert "(jpg, png, gif, webp, bmp)" in create_read_tool(images=_NoImages()).description


# ─── an image that changes while it is read ──────────────────────────────────


def _append(path: Path, data: bytes) -> None:
    with open(path, "ab") as file:
        file.write(data)


async def test_reads_it_again_rather_than_return_a_cut_off_image_even_when_it_grew(tmp_path: Path) -> None:
    png = encode(8, 6, "png")
    whole_reads: list[int] = []

    def on_read(offset: int, length: int) -> None:
        # A writer appends during the first read of the whole file; the second read sees it all.
        if offset != 0 or length < len(png):
            return
        whole_reads.append(length)
        if len(whole_reads) == 1:
            _append(tmp_path / "growing.png", bytes(16))

    env, _ = read_env(tmp_path, on_read)
    result = await read(tmp_path, "growing.png", png, env=env)
    assert whole_reads == [len(png), len(png) + 16]
    assert only_image(result).data == to_base64(png + bytes(16))


async def test_fails_when_it_changes_during_the_second_read_too(tmp_path: Path) -> None:
    png = encode(8, 6, "png")

    def on_read(offset: int, length: int) -> None:
        if offset == 0 and length >= len(png):
            _append(tmp_path / "busy.png", bytes(1))

    env, _ = read_env(tmp_path, on_read)
    with pytest.raises(RuntimeError, match="busy.png changed while it was read"):
        await read(tmp_path, "busy.png", png, env=env)


# ─── with the Pillow image processor ─────────────────────────────────────────


@needs_pillow
async def test_passes_an_image_that_needs_no_change_through_untouched(tmp_path: Path) -> None:
    png = encode(8, 6, "png")
    images = processor()
    result = await read(tmp_path, "a.png", png, images=images)
    assert result["output"] == [ImageContent(to_base64(png), "image/png")]
    assert only_image(await read(tmp_path, "a.gif", gif(), images=images)).mime_type == "image/gif"


@needs_pillow
async def test_scales_an_image_down_to_the_maximum_size_and_says_how_to_map_coordinates_back(tmp_path: Path) -> None:
    result = await read(tmp_path, "wide.png", encode(4000, 1000, "png"), images=processor())
    block = only_image(result)
    image = decoded_image(block.data)
    assert (image["width"], image["height"]) == (2000, 500)
    assert messages(result) == [
        f"Read image file [{block.mime_type}]. Resized from 4000x1000 to 2000x500. Multiply coordinates by 2.00 to "
        "map them to the original."
    ]


@needs_pillow
async def test_reencodes_a_png_too_large_for_the_byte_limit_as_jpeg_at_its_size_and_shrinks_it_when_needed(
    tmp_path: Path,
) -> None:
    images = processor()
    result = await read(tmp_path, "big.png", noisy_png(1100, 1100), images=images)
    jpeg = only_image(result)
    assert jpeg.mime_type == "image/jpeg"
    assert messages(result) == ["Read image file [image/jpeg]. Converted from image/png to image/jpeg."]
    assert len(jpeg.data) <= DEFAULT_IMAGE_LIMITS["maxBytes"]
    image = decoded_image(jpeg.data)
    assert (image["width"], image["height"]) == (1100, 1100)
    tight = {**DEFAULT_IMAGE_LIMITS, "maxBytes": 200_000}
    shrunk = await images.prepare(noisy_png(2000, 2000), "image/png", tight)  # type: ignore[arg-type]
    assert shrunk is not None
    assert len(shrunk["data"]) <= tight["maxBytes"]
    assert shrunk["resized"]["from"] == {"width": 2000, "height": 2000}  # type: ignore[typeddict-item]
    assert shrunk["resized"]["to"]["width"] < 2000  # type: ignore[typeddict-item]


@needs_pillow
async def test_keeps_at_least_one_pixel_on_each_side_of_an_extreme_aspect_ratio() -> None:
    result = await processor().prepare(encode(6000, 2, "png"), "image/png", DEFAULT_IMAGE_LIMITS)
    assert result is not None
    assert result.get("resized") == {"from": {"width": 6000, "height": 2}, "to": {"width": 2000, "height": 1}}


@needs_pillow
async def test_gives_up_when_not_even_one_pixel_fits_the_byte_limit() -> None:
    limits = {**DEFAULT_IMAGE_LIMITS, "maxBytes": 8}
    assert await processor().prepare(encode(4, 4, "png"), "image/png", limits) is None  # type: ignore[arg-type]


@needs_pillow
async def test_keeps_a_jpeg_that_must_be_reencoded_a_jpeg() -> None:
    result = await processor().prepare(with_orientation(encode(40, 30, "jpeg"), 3), "image/jpeg", DEFAULT_IMAGE_LIMITS)
    assert result is not None
    assert result["mimeType"] == "image/jpeg"


@needs_pillow
async def test_converts_a_format_providers_do_not_take_inline_to_png(tmp_path: Path) -> None:
    result = await read(tmp_path, "a.bmp", bmp(4, 4), images=processor())
    assert only_image(result).mime_type == "image/png"
    assert messages(result) == ["Read image file [image/png]. Converted from image/bmp to image/png."]


@needs_pillow
async def test_reports_bytes_it_cannot_decode_as_an_error(tmp_path: Path) -> None:
    broken = encode(4, 4, "png")[:40]
    result = await read(tmp_path, "broken.png", broken, images=processor())
    assert result["output"] == []
    assert result["isError"] is True
    assert messages(result) == ["broken.png is an image (image/png) that cannot be prepared for the model"]


@needs_pillow
async def test_is_what_create_coding_tools_images_gives_read(tmp_path: Path) -> None:
    extension = create_coding_tools(images=processor())
    tool = next(candidate for candidate in extension.tools if candidate.name == "read")
    result = await read(tmp_path, "b.bmp", bmp(4, 4), tool=tool)
    assert only_image(result).mime_type == "image/png"


# ─── EXIF orientation ────────────────────────────────────────────────────────

R: Rgb = (255, 0, 0)
G: Rgb = (0, 255, 0)
B: Rgb = (0, 0, 255)
W: Rgb = (255, 255, 255)
K: Rgb = (0, 0, 0)
Y: Rgb = (255, 255, 0)
STORED = [[R, G, B], [W, K, Y]]


def mirror(grid: list[list[Rgb]]) -> list[list[Rgb]]:
    return [list(reversed(row)) for row in grid]


def turn(grid: list[list[Rgb]]) -> list[list[Rgb]]:
    """A quarter turn clockwise: the left column, read bottom to top, becomes the top row."""
    return [list(reversed([row[x] for row in grid])) for x in range(len(grid[0]))]


# What each orientation shows, from its EXIF definition: mirror, then turn clockwise.
UPRIGHT = {
    1: [list(row) for row in STORED],
    2: mirror(STORED),
    3: turn(turn(STORED)),
    4: turn(turn(mirror(STORED))),
    5: turn(turn(turn(mirror(STORED)))),
    6: turn(STORED),
    7: turn(mirror(STORED)),
    8: turn(turn(turn(STORED))),
}


@needs_pillow
@pytest.mark.parametrize("orientation", range(1, 9))
async def test_turns_each_orientation_upright(orientation: int) -> None:
    rgba, width, height = blocks(STORED, 16)
    jpeg = with_orientation(encode_rgba(rgba, width, height, "jpeg"), orientation)
    prepared = await processor().prepare(jpeg, "image/jpeg", DEFAULT_IMAGE_LIMITS)
    assert prepared is not None
    image = decoded_image(prepared["data"])
    expected = UPRIGHT[orientation]
    assert (image["width"], image["height"]) == (len(expected[0]) * 16, len(expected) * 16)
    # The colour at the centre of each block, rounded to full channels against JPEG noise.
    seen = []
    for by, row in enumerate(expected):
        seen_row = []
        for bx in range(len(row)):
            at = ((by * 16 + 8) * image["width"] + bx * 16 + 8) * 4
            seen_row.append(tuple(255 if channel > 127 else 0 for channel in image["rgba"][at : at + 3]))
        seen.append(seen_row)
    assert seen == expected


@needs_pillow
def test_reads_big_and_little_endian_jpeg_exif_after_an_app0_segment_and_1_when_absent_or_out_of_range() -> None:
    from pi_durable.images.exif import exif_orientation

    jpeg = encode(2, 2, "jpeg")
    assert exif_orientation(with_orientation(jpeg, 6)) == 6
    assert exif_orientation(with_orientation(jpeg, 8, little=True)) == 8
    assert exif_orientation(with_orientation(jpeg, 5, app0=True)) == 5
    assert exif_orientation(with_orientation(jpeg, 9)) == 1
    assert exif_orientation(jpeg) == 1
    assert exif_orientation(encode(2, 2, "png")) == 1


@needs_pillow
def test_reads_webp_exif_chunks_with_and_without_the_exif_prefix() -> None:
    from pi_durable.images.exif import exif_orientation

    assert exif_orientation(webp_with_orientation(6, False)) == 6
    assert exif_orientation(webp_with_orientation(3, True)) == 3


# ─── image_dimensions ────────────────────────────────────────────────────────


@needs_pillow
def test_reads_the_size_each_format_declares_in_its_header() -> None:
    assert image_dimensions(encode(7, 5, "png"), "image/png") == {"width": 7, "height": 5}
    assert image_dimensions(encode(7, 5, "jpeg"), "image/jpeg") == {"width": 7, "height": 5}
    assert image_dimensions(with_orientation(encode(7, 5, "jpeg"), 6, app0=True), "image/jpeg") == {
        "width": 7,
        "height": 5,
    }
    assert image_dimensions(gif(300, 200), "image/gif") == {"width": 300, "height": 200}

    def webp(kind: bytes, data: list[int]) -> bytes:
        return b"RIFF" + bytes(4) + b"WEBP" + kind + bytes(4) + bytes(data) + bytes(16)

    # VP8: frame tag, start code, then 14-bit width and height.
    assert image_dimensions(webp(b"VP8 ", [0, 0, 0, 0x9D, 0x01, 0x2A, 0x2C, 0x01, 0xC8, 0x00]), "image/webp") == {
        "width": 300,
        "height": 200,
    }
    # VP8L: signature, then width - 1 and height - 1 in 14 bits each.
    bits = 299 | (199 << 14)
    vp8l = [0x2F, bits & 0xFF, (bits >> 8) & 0xFF, (bits >> 16) & 0xFF, bits >> 24]
    assert image_dimensions(webp(b"VP8L", vp8l), "image/webp") == {"width": 300, "height": 200}
    # VP8X: flags, then width - 1 and height - 1 in 24 bits each.
    assert image_dimensions(webp(b"VP8X", [0, 0, 0, 0, 0x2B, 0x01, 0, 0xC7, 0, 0]), "image/webp") == {
        "width": 300,
        "height": 200,
    }
    assert image_dimensions(bytes([0xFF, 0xD8, 0xFF]), "image/jpeg") is None


# ─── to_base64 ───────────────────────────────────────────────────────────────


def test_to_base64_matches_the_standard_encoding_for_every_remainder_length() -> None:
    for length in (0, 1, 2, 3, 4, 5, 8191, 8192, 8193, 10_000, 16_385):
        data = bytes((index * 37) % 256 for index in range(length))
        assert to_base64(data) == base64.b64encode(data).decode("ascii")


# ─── create_pillow_images ────────────────────────────────────────────────────


@needs_pillow
async def test_prepares_in_a_worker_thread_leaving_the_event_loop_free() -> None:
    from pi_durable.images import create_pillow_images

    photo = noisy_png(3000, 2000)

    async def prepare_on(images: ImageProcessor) -> float:
        # The longest gap between timer ticks while the image is prepared: the time the event loop was blocked.
        longest = 0.0
        last = time.perf_counter()
        stop = False

        async def ticks() -> None:
            nonlocal longest, last
            while not stop:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                longest = max(longest, now - last)
                last = now

        ticker = asyncio.ensure_future(ticks())
        await asyncio.sleep(0)
        try:
            assert await images.prepare(photo, "image/png", DEFAULT_IMAGE_LIMITS) is not None
            # Counted until the result arrived, including the gap the last tick did not see.
            longest = max(longest, time.perf_counter() - last)
        finally:
            stop = True
            await ticker
        return longest * 1000

    inline = await prepare_on(create_pillow_images(worker=False))
    threaded = await prepare_on(create_pillow_images())
    assert inline > 200
    assert threaded < 100


@needs_pillow
async def test_gives_the_same_result_as_the_calling_thread_and_keeps_the_callers_bytes() -> None:
    from pi_durable.images import create_pillow_images

    png = bytearray(noisy_png(1100, 1100))
    copy = bytes(png)
    threaded = await create_pillow_images().prepare(png, "image/png", DEFAULT_IMAGE_LIMITS)  # type: ignore[arg-type]
    inline = await create_pillow_images(worker=False).prepare(png, "image/png", DEFAULT_IMAGE_LIMITS)  # type: ignore[arg-type]
    assert threaded == inline
    assert bytes(png) == copy


@needs_pillow
async def test_prepares_concurrent_images_and_starts_a_new_worker_after_close() -> None:
    from pi_durable.images import create_pillow_images

    images = create_pillow_images()
    results = await asyncio.gather(
        images.prepare(bmp(4, 4), "image/bmp", DEFAULT_IMAGE_LIMITS),
        images.prepare(encode(4000, 1000, "png"), "image/png", DEFAULT_IMAGE_LIMITS),
        images.prepare(bytes([1, 2, 3]), "image/png", DEFAULT_IMAGE_LIMITS),
    )
    assert results[0] is not None
    assert results[0]["mimeType"] == "image/png"
    assert results[1] is not None
    assert results[1]["resized"]["to"] == {"width": 2000, "height": 500}  # type: ignore[typeddict-item]
    assert results[2] is None
    images.close()
    again = await images.prepare(bmp(2, 2), "image/bmp", DEFAULT_IMAGE_LIMITS)
    assert again is not None
    assert again["mimeType"] == "image/png"
    images.close()


@needs_pillow
async def test_runs_on_a_given_executor_also_a_process_pool() -> None:
    from concurrent.futures import ProcessPoolExecutor

    from pi_durable.images import create_pillow_images

    with ProcessPoolExecutor(max_workers=1) as executor:
        result = await create_pillow_images(executor=executor).prepare(
            encode(4000, 1000, "png"), "image/png", DEFAULT_IMAGE_LIMITS
        )
    assert result is not None
    assert result.get("resized") == {"from": {"width": 4000, "height": 1000}, "to": {"width": 2000, "height": 500}}
