"""pi_chord._callbacks: JavaScript-style argument trimming for user callbacks."""

from __future__ import annotations

from functools import partial
from typing import Any

from pi_chord._callbacks import trim_arguments


def test_trims_to_declared_positional_parameters() -> None:
    def one(a: Any) -> Any:
        return (a,)

    def two(a: Any, b: Any = None) -> Any:
        return (a, b)

    assert trim_arguments(one, 3)(1, 2, 3) == (1,)
    assert trim_arguments(two, 3)(1, 2, 3) == (1, 2)
    assert trim_arguments(lambda: "none", 2)(1, 2) == "none"


def test_leaves_callbacks_that_take_everything_unchanged() -> None:
    def everything(*arguments: Any) -> Any:
        return arguments

    def three(a: Any, b: Any, c: Any) -> Any:
        return (a, b, c)

    assert trim_arguments(everything, 3) is everything
    assert trim_arguments(three, 3) is three
    assert trim_arguments(print, 3) is print  # takes *args
    bound = partial(three, 1)
    assert trim_arguments(bound, 2)(2, 3) == (1, 2, 3)
