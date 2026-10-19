"""Calling user callbacks with JavaScript's argument leniency."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

__all__ = ["trim_arguments"]


def trim_arguments(callback: Callable[..., Any], count: int) -> Callable[..., Any]:
    """Return ``callback`` adapted to be called with ``count`` positional arguments.

    JavaScript ignores extra arguments, so upstream listeners such as ``(value) => ...`` receive
    only what they declare. A callback declaring fewer positional parameters (and no ``*args``)
    gets only the leading ones; any other callback is returned unchanged.
    """
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):  # builtins without a signature: call as is
        return callback
    accepted = 0
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return callback
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            accepted += 1
    if accepted >= count:
        return callback

    def trimmed(*arguments: Any) -> Any:
        return callback(*arguments[:accepted])

    return trimmed
