"""An image processor for the ``read`` tool on Pillow (port of ``images/index.ts`` and ``images/node.ts``).

Upstream prepares images with Photon (Rust compiled to WebAssembly) in a Node worker thread; here Pillow
does the work in a worker thread. Requires the ``images`` extra (``pip install "pi-ai-python[images]"``);
importing this module without Pillow raises :class:`ImportError`. :mod:`pi_durable` and
:mod:`pi_durable.tools` never import it::

    from pi_durable.images import create_pillow_images
    from pi_durable.tools import create_coding_tools

    registry.install(create_coding_tools(images=create_pillow_images()))

What :func:`prepare_image` does, as upstream: an image already upright, in a format providers take
inline, and within the limits is returned as it is, once it decodes. Otherwise it is turned upright by
its EXIF orientation, scaled to the maximum width and height, and encoded as PNG or JPEG at falling
qualities (JPEG first for JPEG sources), then at smaller dimensions, 3/4 at a time, until the base64 fits.
Formats providers do not take inline (BMP) become PNG. The output bytes differ from Photon's; sizes,
formats, orientation and the limits are the same.

Why a thread and not a process: Pillow releases the GIL while it decodes, converts, resizes and encodes,
so a worker thread leaves the event loop (and other Python threads) running, without copying the image
to another process and back. Pass ``executor=concurrent.futures.ProcessPoolExecutor()`` to isolate the
work in processes anyway; :func:`prepare_image` and its arguments pickle.
"""

from __future__ import annotations

try:
    from PIL import Image
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        'pi_durable.images needs Pillow: install the images extra, pip install "pi-ai-python[images]"'
    ) from error

import asyncio
import io
import math
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Final

from ..tools.image_processor import (
    INLINE_IMAGE_TYPES,
    ImageLimits,
    ImageSize,
    PreparedImage,
    base64_length,
    to_base64,
)
from .exif import exif_orientation

__all__ = ["PillowImages", "create_pillow_images", "prepare_image"]

#: JPEG qualities tried in order.
_JPEG_QUALITIES: Final = (80, 70, 55, 40)

#: EXIF orientations 2 to 8 as the transposition that makes them upright.
_UPRIGHT: Final = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def prepare_image(data: bytes, mime_type: str, limits: ImageLimits) -> PreparedImage | None:
    """The image prepared to fit ``limits``, or ``None`` when it cannot be decoded or made to fit.

    Synchronous and CPU-bound: about a second for a 12-megapixel photo. :class:`PillowImages` runs it off
    the event loop.
    """
    orientation = exif_orientation(data)
    inline = mime_type in INLINE_IMAGE_TYPES
    try:
        with Image.open(io.BytesIO(data)) as opened:
            # Decoded even when it is returned as it is: a damaged file is an error, not an image the provider
            # rejects.
            opened.load()
            if (
                inline
                and orientation == 1
                and opened.width <= limits["maxWidth"]
                and opened.height <= limits["maxHeight"]
                and base64_length(len(data)) <= limits["maxBytes"]
            ):
                return {"data": to_base64(data), "mimeType": mime_type}
            image = _pixels(opened)
        transpose = _UPRIGHT.get(orientation)
        if transpose is not None:
            image = image.transpose(transpose)
        width, height = image.size
        scale = min(1, limits["maxWidth"] / width, limits["maxHeight"] / height)
        target: ImageSize = {"width": max(1, _js_round(width * scale)), "height": max(1, _js_round(height * scale))}
        while True:
            encoded = _encode_within(image, target, limits["maxBytes"], mime_type == "image/jpeg")
            if encoded is not None:
                encoded_data, encoded_type = encoded
                prepared: PreparedImage = {"data": encoded_data, "mimeType": encoded_type}
                if encoded_type != mime_type:
                    prepared = {**prepared, "convertedFrom": mime_type}
                if target["width"] != width or target["height"] != height:
                    prepared = {**prepared, "resized": {"from": {"width": width, "height": height}, "to": target}}
                return prepared
            if target["width"] == 1 and target["height"] == 1:
                return None
            target = {
                "width": max(1, math.floor(target["width"] * 0.75)),
                "height": max(1, math.floor(target["height"] * 0.75)),
            }
    except Exception:
        # Pillow raises for bytes it cannot decode (and for decompression bombs).
        return None


def _js_round(value: float) -> int:
    """``Math.round``: halves round up."""
    return math.floor(value + 0.5)


def _pixels(image: Image.Image) -> Image.Image:
    """The decoded image as RGB, or RGBA when it has transparency (Photon decodes to RGBA)."""
    has_alpha = image.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in image.info
    mode = "RGBA" if has_alpha else "RGB"
    return image.copy() if image.mode == mode else image.convert(mode)


def _encode_within(image: Image.Image, size: ImageSize, max_bytes: float, jpeg_first: bool) -> tuple[str, str] | None:
    """``image`` at ``size`` in the first encoding whose base64 fits ``max_bytes``, as ``(base64, mime type)``.

    PNG, then JPEG at falling qualities, or the JPEG qualities first when ``jpeg_first``; ``None`` if none fits.
    """
    same = size["width"] == image.width and size["height"] == image.height
    scaled = image if same else image.resize((size["width"], size["height"]), Image.Resampling.LANCZOS)

    def png() -> bytes:
        output = io.BytesIO()
        scaled.save(output, "PNG")
        return output.getvalue()

    def jpeg(quality: int) -> Callable[[], bytes]:
        def encode() -> bytes:
            output = io.BytesIO()
            # JPEG has no alpha channel: it is dropped, as Photon does.
            (scaled if scaled.mode == "RGB" else scaled.convert("RGB")).save(output, "JPEG", quality=quality)
            return output.getvalue()

        return encode

    jpegs: list[tuple[str, Callable[[], bytes]]] = [("image/jpeg", jpeg(quality)) for quality in _JPEG_QUALITIES]
    candidates = [*jpegs, ("image/png", png)] if jpeg_first else [("image/png", png), *jpegs]
    for mime_type, encode in candidates:
        encoded = encode()
        if base64_length(len(encoded)) <= max_bytes:
            return to_base64(encoded), mime_type
    return None


class PillowImages:
    """An :class:`~pi_durable.tools.ImageProcessor` that runs :func:`prepare_image` on an executor, or inline.

    Built by :func:`create_pillow_images`. The caller's bytes are never changed.
    """

    __slots__ = ("_executor", "_lock", "_own_executor", "_worker")

    def __init__(self, *, worker: bool = True, executor: Executor | None = None) -> None:
        self._worker = worker or executor is not None
        self._executor = executor
        self._own_executor = executor is None
        self._lock = threading.Lock()

    async def prepare(self, data: bytes, mime_type: str, limits: ImageLimits) -> PreparedImage | None:
        # Bytes are immutable; a bytearray or memoryview is copied, so the caller may keep changing its own.
        image = data if isinstance(data, bytes) else bytes(data)
        if not self._worker:
            return prepare_image(image, mime_type, limits)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool(), prepare_image, image, mime_type, limits)

    def _pool(self) -> Executor:
        with self._lock:
            if self._executor is None:
                # One thread, as upstream's one worker: images are prepared one at a time, which bounds memory.
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-durable-images")
            return self._executor

    def close(self) -> None:
        """Stop the worker thread this processor started (not an executor it was given); idempotent.

        Optional: an idle worker thread holds no image memory, and the interpreter stops it at exit.
        """
        if not self._own_executor:
            return
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False)


def create_pillow_images(*, worker: bool = True, executor: Executor | None = None) -> PillowImages:
    """An image processor on Pillow, for ``create_coding_tools(images=...)`` or ``create_read_tool(images=...)``.

    By default images are prepared one at a time in a worker thread of the processor's own, started at the
    first image, so a large photo does not block the event loop for the second or so Pillow takes.
    ``worker=False`` prepares them on the calling thread, blocking the loop meanwhile. ``executor`` runs
    them on that executor instead (a thread pool to prepare several at once, or a process pool).
    """
    return PillowImages(worker=worker, executor=executor)
