"""Image limits and the image processor interface of the ``read`` tool (port of ``tools/image-processor.ts``)."""

from __future__ import annotations

import base64
import math
from typing import TYPE_CHECKING, Final, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

__all__ = [
    "DEFAULT_IMAGE_LIMITS",
    "INLINE_IMAGE_TYPES",
    "ImageLimits",
    "ImageProcessor",
    "ImageSize",
    "PreparedImage",
    "PreparedImageResize",
    "base64_length",
    "to_base64",
]


class ImageLimits(TypedDict):
    """Limits an image the model sees must keep; the defaults below are the coding agent's."""

    #: Pixels. Default 2000.
    maxWidth: ReadOnly[int]
    #: Pixels. Default 2000.
    maxHeight: ReadOnly[int]
    #: Bytes of base64. Default 4.5 MiB, below Anthropic's 5 MB.
    maxBytes: ReadOnly[float]


DEFAULT_IMAGE_LIMITS: Final[ImageLimits] = {"maxWidth": 2000, "maxHeight": 2000, "maxBytes": 4.5 * 1024 * 1024}


class ImageSize(TypedDict):
    width: ReadOnly[int]
    height: ReadOnly[int]


#: ``{"from": size, "to": size}``: an image's size before and after resizing (``from`` is a keyword).
PreparedImageResize = TypedDict("PreparedImageResize", {"from": ImageSize, "to": ImageSize})


class PreparedImage(TypedDict):
    """An image the model can take: a supported format, within the limits, as base64."""

    data: ReadOnly[str]
    mimeType: ReadOnly[str]
    #: Set when the image was resized: its size before and after, for the model to map coordinates back.
    resized: ReadOnly[NotRequired[PreparedImageResize]]
    #: The original format, set when it differs from ``mimeType``: BMP becomes PNG, a large PNG may become JPEG.
    convertedFrom: ReadOnly[NotRequired[str]]


class ImageProcessor(Protocol):
    """Decodes, orients, resizes, and re-encodes images so the model can take them.

    ``read`` uses one when given; without one, it passes supported images within the byte limit through
    as they are. :mod:`pi_durable.images` provides a Pillow-based one (the ``images`` extra).
    """

    async def prepare(self, data: bytes, mime_type: str, limits: ImageLimits) -> PreparedImage | None:
        """The image prepared to fit ``limits``, or ``None`` when it cannot be decoded or made to fit.

        A coroutine, so a processor can do the work off the event loop.
        """
        ...


#: Formats every provider takes inline; others, such as BMP, need converting.
INLINE_IMAGE_TYPES: Final[frozenset[str]] = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def to_base64(data: bytes | bytearray | memoryview) -> str:
    """Standard base64 of ``data``."""
    return base64.b64encode(data).decode("ascii")


def base64_length(byte_length: int) -> int:
    """The length of the base64 of ``byte_length`` bytes."""
    return math.ceil(byte_length / 3) * 4
