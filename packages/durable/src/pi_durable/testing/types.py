"""Types of the runner-independent conformance suites (port of ``testing/types.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import NotRequired, Protocol, TypeAlias, TypedDict

from ..env.types import ExecutionEnv
from ..types import Storage

__all__ = [
    "EnvConformanceAssertions",
    "EnvConformanceCase",
    "EnvConformanceOptions",
    "EnvConformanceProvider",
    "StorageConformanceAssertions",
    "StorageConformanceCase",
    "StorageConformanceOptions",
    "StorageConformanceProvider",
]


class StorageConformanceAssertions(Protocol):
    """The assertions the suites make; :func:`~pi_durable.testing.create_assertions` raises ``AssertionError``."""

    def ok(self, value: object, message: str | None = None) -> None:
        """``value`` is truthy."""
        ...

    def strict_equal(self, actual: object, expected: object) -> None:
        """Equal scalars (``bool`` never equals ``int``), or the same object."""
        ...

    def deep_equal(self, actual: object, expected: object) -> None:
        """Recursively equal; tuples and lists compare as sequences."""
        ...

    def partial_deep_equal(self, actual: object, expected: object) -> None:
        """``expected``'s keys match recursively (upstream ``toMatchObject``); sequences match element-wise."""
        ...

    def greater_than(self, actual: float, expected: float) -> None: ...

    async def rejects(self, operation: Awaitable[object], message_includes: str) -> None:
        """Awaiting ``operation`` raises an exception whose message contains ``message_includes``."""
        ...


#: Calls ``use`` once with a storage, then cleans up: ``async def provider(use): await use(MemoryStorage())``.
StorageConformanceProvider: TypeAlias = Callable[[Callable[[Storage], Awaitable[None]]], Awaitable[None]]


class StorageConformanceOptions(TypedDict):
    assertions: StorageConformanceAssertions
    withStorage: StorageConformanceProvider


@dataclass(frozen=True, slots=True)
class StorageConformanceCase:
    """One runner-independent case: ``await case.run()`` passes or raises."""

    name: str
    run: Callable[[], Awaitable[None]]
    #: Cases that wait for an environment's watch latency need longer than a test runner's default timeout.
    timeout_ms: float | None = None


EnvConformanceAssertions: TypeAlias = StorageConformanceAssertions

#: Calls ``use`` once with an environment whose ``cwd`` is a fresh, empty, writable directory, then cleans up.
EnvConformanceProvider: TypeAlias = Callable[[Callable[[ExecutionEnv], Awaitable[None]]], Awaitable[None]]


class EnvConformanceOptions(TypedDict):
    assertions: EnvConformanceAssertions
    withEnv: EnvConformanceProvider
    #: Program and flag that run a POSIX shell script from the next argument; default ``["sh", "-c"]``.
    shell: NotRequired[Sequence[str]]
    #: Whether the shell's ``ln -s`` creates symbolic links; default true.
    symlinks: NotRequired[bool]
    #: Whether the environment implements ``list_files`` (rather than returning ``not_supported``), which the
    #: suite then checks; default false.
    listFiles: NotRequired[bool]


#: An environment case (the same shape as a storage case).
EnvConformanceCase: TypeAlias = StorageConformanceCase
