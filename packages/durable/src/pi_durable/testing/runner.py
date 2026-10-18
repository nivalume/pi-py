"""Register the conformance suites with pytest (port of ``testing/runner.ts``).

Upstream registers the cases with a Vitest/Jest-compatible ``describe``/``it``; here they become one
parametrized pytest test in the calling module, each case run on a fresh event loop so the suites need
no asyncio plugin:

.. code-block:: python

    # test_my_storage.py
    from pi_durable.testing import register_storage_conformance

    async def with_storage(use):
        storage = await MyStorage.open(...)
        try:
            await use(storage)
        finally:
            await storage.close(BACKGROUND_CONTEXT)

    register_storage_conformance(globals(), "MyStorage", with_storage)
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import MutableMapping, Sequence
from typing import Any

from .assertions import create_assertions
from .env_conformance import create_env_conformance
from .storage_conformance import create_storage_conformance
from .types import EnvConformanceOptions, EnvConformanceProvider, StorageConformanceCase, StorageConformanceProvider

__all__ = ["register_env_conformance", "register_storage_conformance"]


def _register(namespace: MutableMapping[str, Any], name: str, cases: Sequence[StorageConformanceCase]) -> None:
    import pytest

    slug = re.sub(r"\W+", "_", name).strip("_").lower()
    test_name = f"test_{slug}"
    if test_name in namespace:
        raise ValueError(f"{test_name} is already defined")

    @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases])
    def test(case: StorageConformanceCase) -> None:
        async def main() -> None:
            await case.run()

        asyncio.run(main())

    test.__name__ = test_name
    test.__qualname__ = test_name
    test.__doc__ = f"{name}: the runner-independent conformance cases."
    namespace[test_name] = test


def register_storage_conformance(
    namespace: MutableMapping[str, Any], name: str, with_storage: StorageConformanceProvider
) -> None:
    """Define ``test_<name>`` in ``namespace`` (a test module's ``globals()``), one pytest case per conformance case."""
    _register(
        namespace, name, create_storage_conformance({"assertions": create_assertions(), "withStorage": with_storage})
    )


def register_env_conformance(
    namespace: MutableMapping[str, Any],
    name: str,
    with_env: EnvConformanceProvider,
    *,
    shell: Sequence[str] | None = None,
    symlinks: bool | None = None,
    list_files: bool | None = None,
) -> None:
    """Define ``test_<name>`` in ``namespace`` running the ``ExecutionEnv`` cases (see :func:`create_env_conformance`)."""
    options: EnvConformanceOptions = {"assertions": create_assertions(), "withEnv": with_env}
    if shell is not None:
        options["shell"] = shell
    if symlinks is not None:
        options["symlinks"] = symlinks
    if list_files is not None:
        options["listFiles"] = list_files
    _register(namespace, name, create_env_conformance(options))
