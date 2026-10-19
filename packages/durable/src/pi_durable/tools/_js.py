"""JavaScript number semantics the tools reproduce: formatting, truncation, and NaN-propagating ``Math.min``."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Final

__all__ = ["MAX_SAFE_INTEGER", "is_safe_integer", "js_min", "js_number", "js_truthy", "js_trunc"]

MAX_SAFE_INTEGER: Final = 2**53 - 1


def js_number(value: float) -> str:
    """``String(value)``: ``5`` rather than ``5.0``, ``1e+21``, ``1.5e-7``, ``NaN``, ``Infinity``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        value = float(value) if abs(value) >= 1e21 else value
        if isinstance(value, int):
            return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    text = repr(value)
    if "e" not in text:
        return text
    if 1e-6 <= abs(value) < 1e21:
        # Python switches to exponent notation earlier than JavaScript does.
        return format(Decimal(text), "f")
    mantissa, exponent = text.split("e")
    sign = "-" if exponent.startswith("-") else "+"
    return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def js_truthy(value: float | None) -> bool:
    """Whether a number is truthy in JavaScript: not ``undefined``, ``0``, or ``NaN``."""
    return value is not None and value == value and value != 0


def js_trunc(value: float) -> float:
    """``Math.trunc``, keeping infinities and NaN."""
    if math.isnan(value) or math.isinf(value):
        return value
    return float(math.trunc(value))


def js_min(*values: float) -> float:
    """``Math.min``: NaN when any argument is NaN."""
    if any(value != value for value in values):
        return math.nan
    return min(values)


def is_safe_integer(value: float) -> bool:
    """``Number.isSafeInteger``."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_SAFE_INTEGER
    return math.isfinite(value) and value.is_integer() and abs(value) <= MAX_SAFE_INTEGER
