"""Conformance suites and benchmark scenarios for durable backends (port of ``testing/index.ts``).

Third-party ``Storage`` and ``ExecutionEnv`` implementations run the same cases as the built-in ones:

.. code-block:: python

    # test_my_storage.py (pytest)
    from pi_durable.testing import register_storage_conformance

    async def with_storage(use):
        storage = MyStorage()
        try:
            await use(storage)
        finally:
            await storage.close(BACKGROUND_CONTEXT)

    register_storage_conformance(globals(), "MyStorage", with_storage)

:func:`create_storage_conformance` and :func:`create_env_conformance` return the runner-independent cases for
other runners; :mod:`pi_durable.testing.storage_benchmark` seeds and times representative workloads.
"""

from __future__ import annotations

from .assertions import create_assertions
from .env_conformance import create_env_conformance
from .runner import register_env_conformance, register_storage_conformance
from .storage_benchmark import (
    STORAGE_MEMORY_SCALES,
    STORAGE_READ_BENCHMARKS,
    STORAGE_WRITE_BENCHMARKS,
    TIMING_SCALE,
    StorageBenchmarkDataset,
    StorageBenchmarkScale,
    StorageReadBenchmark,
    StorageWriteBenchmark,
    seed_storage_benchmark,
    seed_storage_write_benchmark,
    storage_benchmark_primary_record_count,
)
from .storage_conformance import create_storage_conformance
from .types import (
    EnvConformanceAssertions,
    EnvConformanceCase,
    EnvConformanceOptions,
    EnvConformanceProvider,
    StorageConformanceAssertions,
    StorageConformanceCase,
    StorageConformanceOptions,
    StorageConformanceProvider,
)

__all__ = [
    "STORAGE_MEMORY_SCALES",
    "STORAGE_READ_BENCHMARKS",
    "STORAGE_WRITE_BENCHMARKS",
    "TIMING_SCALE",
    "EnvConformanceAssertions",
    "EnvConformanceCase",
    "EnvConformanceOptions",
    "EnvConformanceProvider",
    "StorageBenchmarkDataset",
    "StorageBenchmarkScale",
    "StorageConformanceAssertions",
    "StorageConformanceCase",
    "StorageConformanceOptions",
    "StorageConformanceProvider",
    "StorageReadBenchmark",
    "StorageWriteBenchmark",
    "create_assertions",
    "create_env_conformance",
    "create_storage_conformance",
    "register_env_conformance",
    "register_storage_conformance",
    "seed_storage_benchmark",
    "seed_storage_write_benchmark",
    "storage_benchmark_primary_record_count",
]
