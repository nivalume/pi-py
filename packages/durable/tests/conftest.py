"""Test configuration for pi_durable.

The suite runs with ``--import-mode=importlib``, so test modules cannot import each other; shared
helpers live next to this file (``durable_session_support.py``), which is put on ``sys.path`` here.

``PI_DURABLE_DOCUMENT_RETENTION_MS`` sets the document retention of every Session that keeps the
default, so the suite can run with each unattached document evicted right after use
(``PI_DURABLE_DOCUMENT_RETENTION_MS=0 pytest packages/durable``), as upstream's
``test/setup-document-retention.ts``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from durable_session_support import STORAGE_BACKENDS, StorageBackend  # noqa: E402

from pi_durable.session import set_default_document_retention  # noqa: E402

_forced_retention = os.environ.get("PI_DURABLE_DOCUMENT_RETENTION_MS")
if _forced_retention is not None:
    set_default_document_retention(float(_forced_retention))


@pytest.fixture(params=sorted(STORAGE_BACKENDS))
def storage_backend(request: pytest.FixtureRequest) -> StorageBackend:
    """The storage backend a Session test runs against (see ``durable_session_support.STORAGE_BACKENDS``)."""
    return STORAGE_BACKENDS[request.param]
