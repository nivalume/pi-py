"""Helpers shared by the pi_codemode tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from pi_codemode import CodemodeTool

# Base64 of the leading bytes of each format. image() only inspects the signature.
PNG = "iVBORw0KGgo="
JPEG = "/9j/4A=="
GIF = "R0lGODlh"
WEBP = "UklGRgAAAABXRUJQ"


def echo_tool() -> CodemodeTool:
    return CodemodeTool("echo", lambda args, _context: args)


async def wait_for(check: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("Condition was not reached")
        await asyncio.sleep(0.005)


def value_of(result: Any) -> Any:
    """The value of a successful result, else its error message."""
    return result.value if result.ok else result.error.message
