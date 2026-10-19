"""Test configuration for pi_codemode.

The suite runs with ``--import-mode=importlib``, so test modules cannot import each other; the
shared helpers live in ``codemode_helpers.py`` next to this file, which is put on ``sys.path`` here.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from pi_codemode import CodemodeSandbox, CodemodeTool  # noqa: E402

SandboxFactory = Callable[..., CodemodeSandbox]


@pytest.fixture
async def new_sandbox() -> AsyncIterator[SandboxFactory]:
    """Creates sandboxes (keyword arguments as for ``CodemodeSandbox``) and closes them after the test."""
    created: list[CodemodeSandbox] = []

    def make(**options: Any) -> CodemodeSandbox:
        sandbox = CodemodeSandbox(**options)
        created.append(sandbox)
        return sandbox

    yield make
    await asyncio.gather(*(sandbox.close() for sandbox in created))


@pytest.fixture
def create_sandbox(new_sandbox: SandboxFactory) -> Callable[..., CodemodeSandbox]:
    """``create_sandbox(tools, timeout_ms=10_000)``, upstream's ``createSandbox``."""

    def make(tools: list[CodemodeTool] | None = None, timeout_ms: float = 10_000) -> CodemodeSandbox:
        return new_sandbox(tools=tools or [], timeout_ms=timeout_ms)

    return make
