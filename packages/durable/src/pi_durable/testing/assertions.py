"""Plain assertions for the conformance suites (replaces upstream ``testing/assertions.ts``).

Upstream adapts a Vitest/Jest ``expect``; Python test runners all report ``AssertionError``, so
:func:`create_assertions` raises it directly, with a message that shows both values.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Mapping
from typing import Any

from .types import StorageConformanceAssertions

__all__ = ["create_assertions"]


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) and not dataclasses.is_dataclass(actual):
            return False
        for key, item in expected.items():
            if isinstance(actual, Mapping):
                if key not in actual:
                    return False
                value = actual[key]
            elif hasattr(actual, key):
                value = getattr(actual, key)
            else:
                return False
            if not _matches(value, item):
                return False
        return True
    if isinstance(expected, tuple | list):
        if not isinstance(actual, tuple | list) or len(actual) != len(expected):
            return False
        return all(_matches(a, e) for a, e in zip(actual, expected, strict=True))
    return _strict(actual, expected)


def _strict(actual: Any, expected: Any) -> bool:
    if actual is expected:
        return True
    if isinstance(actual, bool) or isinstance(expected, bool):
        return False
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        return actual == expected
    if isinstance(actual, str | bytes) or isinstance(expected, str | bytes):
        return type(actual) is type(expected) and actual == expected
    return actual == expected


class _Assertions:
    __slots__ = ()

    def ok(self, value: object, message: str | None = None) -> None:
        if not value:
            raise AssertionError(message or f"Expected a truthy value, got {value!r}")

    def strict_equal(self, actual: object, expected: object) -> None:
        if not _strict(actual, expected):
            raise AssertionError(f"Expected {expected!r}, got {actual!r}")

    def deep_equal(self, actual: object, expected: object) -> None:
        if _normalize(actual) != _normalize(expected):
            raise AssertionError(f"Expected {expected!r}, got {actual!r}")

    def partial_deep_equal(self, actual: object, expected: object) -> None:
        if not _matches(actual, expected):
            raise AssertionError(f"Expected {actual!r} to match {expected!r}")

    def greater_than(self, actual: float, expected: float) -> None:
        if not actual > expected:
            raise AssertionError(f"Expected {actual!r} to be greater than {expected!r}")

    async def rejects(self, operation: Awaitable[object], message_includes: str) -> None:
        try:
            await operation
        except Exception as error:
            if message_includes not in str(error):
                raise AssertionError(f"Expected an error including {message_includes!r}, got {error!r}") from error
            return
        raise AssertionError(f"Expected an error including {message_includes!r}, but the operation succeeded")


def create_assertions() -> StorageConformanceAssertions:
    """Assertions that raise ``AssertionError``, for pytest, unittest or a plain script."""
    return _Assertions()
