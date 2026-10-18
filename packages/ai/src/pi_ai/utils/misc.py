"""Small shared helpers: env lookup, headers, hashing, ids, unicode, user agent."""

from __future__ import annotations

import os
import platform
import re
import secrets
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "get_provider_env_value",
    "headers_to_record",
    "provider_headers_to_record",
    "merge_headers",
    "has_header",
    "short_hash",
    "uuidv7",
    "sanitize_surrogates",
    "get_pi_user_agent",
]


def get_provider_env_value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """Resolve a provider env value from scoped overrides, then ``os.environ``.

    Empty strings count as unset, like JavaScript's ``||``.
    """
    if env:
        value = env.get(name)
        if value:
            return value
    value = os.environ.get(name)
    return value or None


def headers_to_record(headers: Any) -> dict[str, str]:
    """Convert response headers (``httpx.Headers`` or a mapping) to a lowercase-key dict."""
    items = headers.multi_items() if hasattr(headers, "multi_items") else headers.items()
    result: dict[str, str] = {}
    for key, value in items:
        lower = key.lower()
        result[lower] = f"{result[lower]}, {value}" if lower in result else value
    return result


def provider_headers_to_record(*sources: Mapping[str, str | None] | None) -> dict[str, str] | None:
    """Merge header sources case-insensitively; later sources win and ``None`` deletes."""
    merged: dict[str, tuple[str, str]] = {}
    for source in sources:
        if not source:
            continue
        for name, value in source.items():
            lower = name.lower()
            merged.pop(lower, None)
            if value is not None:
                merged[lower] = (name, value)
    return {name: value for name, value in merged.values()} if merged else None


def merge_headers(
    base: Mapping[str, str | None] | None, override: Mapping[str, str | None] | None
) -> dict[str, str | None] | None:
    """Case-insensitive header merge that keeps ``None`` suppression markers."""
    if not base and not override:
        return None
    merged: dict[str, str | None] = dict(base or {})
    for name, value in (override or {}).items():
        lower = name.lower()
        for existing in [key for key in merged if key.lower() == lower]:
            del merged[existing]
        merged[name] = value
    return merged


def has_header(headers: Mapping[str, Any] | None, name: str) -> bool:
    if not headers:
        return False
    lower = name.lower()
    return any(key.lower() == lower and value is not None for key, value in headers.items())


_MASK = 0xFFFFFFFF
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _imul(a: int, b: int) -> int:
    return ((a & _MASK) * (b & _MASK)) & _MASK


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(_BASE36[remainder])
    return "".join(reversed(digits))


def _utf16_units(text: str) -> Iterable[int]:
    for char in text:
        code = ord(char)
        if code > 0xFFFF:
            code -= 0x10000
            yield 0xD800 + (code >> 10)
            yield 0xDC00 + (code & 0x3FF)
        else:
            yield code


def short_hash(text: str) -> str:
    """Fast deterministic hash used to shorten long ids (bit-identical to pi's ``shortHash``)."""
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    for unit in _utf16_units(text):
        h1 = _imul(h1 ^ unit, 2654435761)
        h2 = _imul(h2 ^ unit, 1597334677)
    h1 = _imul(h1 ^ (h1 >> 16), 2246822507) ^ _imul(h2 ^ (h2 >> 13), 3266489909)
    h2 = _imul(h2 ^ (h2 >> 16), 2246822507) ^ _imul(h1 ^ (h1 >> 13), 3266489909)
    return _to_base36(h2 & _MASK) + _to_base36(h1 & _MASK)


_MAX_UUID_V7_TIMESTAMP = 0xFFFFFFFFFFFF
_MAX_SEQUENCE = (1 << 41) - 1
_uuid_lock = threading.Lock()
_last_ordinary_timestamp = -1
_sequence: int | None = None


def uuidv7(timestamp_ms: int | None = None) -> str:
    """Generate a time-ordered UUIDv7. A supplied timestamp is preserved for follower ids."""
    global _last_ordinary_timestamp, _sequence
    requested = timestamp_ms if timestamp_ms is not None else time.time_ns() // 1_000_000
    if not isinstance(requested, int) or requested < 0 or requested > _MAX_UUID_V7_TIMESTAMP:
        raise ValueError(f"UUIDv7 timestamp must be an integer between 0 and {_MAX_UUID_V7_TIMESTAMP}")
    random_bytes = bytearray(secrets.token_bytes(16))
    with _uuid_lock:
        if timestamp_ms is None:
            effective = max(requested, _last_ordinary_timestamp)
            _last_ordinary_timestamp = effective
        else:
            effective = timestamp_ms
        if _sequence is None:
            _sequence = int.from_bytes(random_bytes[1:6], "big")
        else:
            if _sequence == _MAX_SEQUENCE:
                raise OverflowError("UUIDv7 generator sequence exhausted")
            _sequence += 1
        sequence = _sequence
    data = bytearray(16)
    data[0:6] = effective.to_bytes(6, "big")
    data[6] = 0x70 | ((sequence >> 37) & 0x0F)
    data[7] = (sequence >> 29) & 0xFF
    data[8] = 0x80 | ((sequence >> 23) & 0x3F)
    data[9] = (sequence >> 15) & 0xFF
    data[10] = (sequence >> 7) & 0xFF
    data[11] = ((sequence & 0x7F) << 1) | (random_bytes[11] & 0x01)
    data[12:16] = random_bytes[12:16]
    hex_text = data.hex()
    return f"{hex_text[0:8]}-{hex_text[8:12]}-{hex_text[12:16]}-{hex_text[16:20]}-{hex_text[20:32]}"


_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def sanitize_surrogates(text: str) -> str:
    """Remove unpaired UTF-16 surrogates (they break JSON/UTF-8 encoding).

    Properly paired surrogates are joined into their code point, so emoji survive.
    """
    if _SURROGATE_RE.search(text) is None:
        return text
    joined = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "ignore")
    return _SURROGATE_RE.sub("", joined)


_user_agent: str | None = None


def get_pi_user_agent() -> str:
    global _user_agent
    if _user_agent is None:
        machine = platform.machine().lower()
        arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "i386": "ia32", "i686": "ia32"}.get(
            machine, machine
        )
        _user_agent = f"pi ({sys.platform} {platform.release()}; {arch})"
    return _user_agent
