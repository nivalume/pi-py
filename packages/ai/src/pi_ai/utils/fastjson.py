"""JSON encode/decode with an optional ``orjson`` fast path.

Install ``pi-ai-python[fast]`` to use orjson for SSE chunk decoding and request bodies.
Output is compact and UTF-8 in both cases.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["loads", "dumps", "dumps_bytes", "HAS_ORJSON"]

try:  # pragma: no cover - exercised when orjson is installed
    import orjson as _orjson

    HAS_ORJSON = True
except ImportError:  # pragma: no cover
    _orjson = None
    HAS_ORJSON = False

_decode = json.JSONDecoder(strict=True).decode


def _default(value: Any) -> Any:
    from ..serialization import to_json_value

    return to_json_value(value)


if _orjson is not None:
    # Dataclasses go through ``_default`` so they serialize in pi's camelCase format.
    _ORJSON_OPTS = _orjson.OPT_NON_STR_KEYS | _orjson.OPT_PASSTHROUGH_DATACLASS

    loads = _orjson.loads  # pyright: ignore[reportAssignmentType]
    _orjson_dumps = _orjson.dumps

    def dumps_bytes(value: Any) -> bytes:
        try:
            return _orjson_dumps(value, default=_default, option=_ORJSON_OPTS)
        except TypeError:
            # orjson rejects lone surrogates and integers above 64 bits; the stdlib escapes them.
            return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=_default).encode()

    def dumps(value: Any) -> str:
        return dumps_bytes(value).decode()

else:

    def loads(data: str | bytes | bytearray | memoryview) -> Any:
        if not isinstance(data, str):
            data = bytes(data).decode()
        return _decode(data)

    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_default)

    def dumps_bytes(value: Any) -> bytes:
        text = dumps(value)
        try:
            return text.encode()
        except UnicodeEncodeError:  # lone surrogates: emit them as \u escapes
            return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=_default).encode()
