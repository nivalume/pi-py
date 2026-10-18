"""Test configuration for pi_agent.

The suite runs with ``--import-mode=importlib``, so test modules cannot import each
other; the shared helpers live in ``agent_helpers.py`` next to this file.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from pi_agent import set_default_stream_fn  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_default_stream_fn() -> Iterator[None]:
    """Never leak a process-wide default stream function between tests."""
    yield
    set_default_stream_fn(None)
