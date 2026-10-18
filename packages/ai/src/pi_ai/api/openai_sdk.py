"""Shared OpenAI SDK plumbing for the adapters built on the official ``openai`` package.

Upstream pi creates an ``OpenAI`` (or ``AzureOpenAI``) client per request, calls
``create(params, { maxRetries: 0 }).withResponse()`` and wraps that call in
``retryProviderRequest`` so the backoff sleep stays abortable. The helpers here do
the same with ``AsyncOpenAI`` / ``AsyncAzureOpenAI`` and translate SDK exceptions
into :class:`~pi_ai.utils.http.ProviderHttpError` with the JS SDK's message wording,
which pi's retry and context-overflow detection match against.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, TypeVar, cast

import httpx
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream, omit

from ..utils.http import (
    DEFAULT_TIMEOUT_MS,
    ProviderHttpError,
    get_http_client,
    make_status_error,
    retry_provider_request,
)
from ..utils.misc import get_provider_env_value

__all__ = [
    "create_openai_client",
    "merge_sdk_headers",
    "sdk_body_arguments",
    "openai_request",
    "open_sdk_stream",
    "iter_sdk_stream",
    "status_error_from_sdk",
    "stream_error_from_sdk",
]

_T = TypeVar("_T")

#: ``AsyncStream`` yielding the decoded JSON of each event. Like the JS SDK, which hands
#: back the parsed objects as-is; building pydantic models for every event would cost more
#: than pi's own event processing.
_RawStream = AsyncStream[object]


def merge_sdk_headers(*sources: Mapping[str, str | None] | None) -> dict[str, str | None]:
    """Merge header sources case-insensitively; later sources win.

    ``None`` values are kept: they remove the header, including the SDK's own defaults
    (``Authorization``, ``User-Agent``, ...), like ``null`` in the JS SDK's ``defaultHeaders``.
    """
    merged: dict[str, tuple[str, str | None]] = {}
    for source in sources:
        if not source:
            continue
        for name, value in source.items():
            lower = name.lower()
            merged.pop(lower, None)
            merged[lower] = (name, value)
    return {name: value for name, value in merged.values()}


def create_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    headers: Mapping[str, str | None],
    options: Mapping[str, Any],
    azure_api_version: str | None = None,
) -> AsyncOpenAI:
    """An ``AsyncOpenAI`` client (``AsyncAzureOpenAI`` with ``azure_api_version``) for one request.

    ``None`` header values are left out here and removed per request (see
    :func:`sdk_body_arguments`). Retries are left to :func:`openai_request`. The transport is ``options["http_client"]`` or
    pi's pooled client for the running loop, so connections are reused across requests.
    Construction is cheap (tens of microseconds), so clients are not cached.
    """
    timeout_ms = options.get("timeout_ms")
    env = options.get("env")
    default_headers = {name: value for name, value in headers.items() if value is not None}
    http_client = cast(Any, get_http_client(base_url or "", options))
    common: dict[str, Any] = {
        "api_key": api_key,
        "organization": get_provider_env_value("OPENAI_ORG_ID", env),
        "project": get_provider_env_value("OPENAI_PROJECT_ID", env),
        # Same overall budget as the JS SDK default (10 minutes); ``read`` is the idle gap
        # between chunks, so an active stream is never cut off.
        "timeout": (timeout_ms if timeout_ms is not None else DEFAULT_TIMEOUT_MS) / 1000,
        "max_retries": 0,
        "default_headers": default_headers,
        "http_client": http_client,
    }
    if azure_api_version is not None:
        return AsyncAzureOpenAI(api_version=azure_api_version, base_url=base_url or "", **common)
    return AsyncOpenAI(base_url=base_url or None, **common)


def sdk_body_arguments(
    params: Mapping[str, Any], *required: str, headers: Mapping[str, str | None] | None = None
) -> dict[str, Any]:
    """``create()`` keyword arguments that send ``params`` as the request body unchanged.

    pi builds (and ``on_payload`` may replace) the body in wire format, including fields the
    SDK does not declare, so it goes through ``extra_body``. ``required`` parameters are
    passed as ``omit`` so the body keys come only from ``params``. This also skips the SDK's
    parameter transform, which walks the whole transcript on every request.

    Headers that ``headers`` (the client's header record) maps to ``None`` are omitted per
    request: the Python SDK only accepts a removed ``Authorization`` header there.
    """
    arguments: dict[str, Any] = dict.fromkeys(required, omit)
    arguments["extra_body"] = params
    if headers:
        removed = {name: omit for name, value in headers.items() if value is None}
        if removed:
            arguments["extra_headers"] = removed
    return arguments


def _legacy_headers(headers: Any) -> httpx.Headers:
    return headers if isinstance(headers, httpx.Headers) else httpx.Headers(list(headers.multi_items()))


def status_error_from_sdk(error: openai.APIStatusError) -> ProviderHttpError:
    """``"<status> <error.message>"`` from the raw body, as the JS SDK formats ``APIError``."""
    response = error.response
    try:
        text = response.text
    except Exception:  # body could not be read
        text = ""
    return make_status_error(error.status_code, text, _legacy_headers(response.headers), "openai")


def stream_error_from_sdk(error: openai.APIError, headers: Any) -> ProviderHttpError:
    """The JS SDK's ``APIError`` for a stream event carrying an ``error`` field."""
    body = error.body
    message = body.get("message") if isinstance(body, dict) else None
    if message:
        text = message if isinstance(message, str) else _json_compact(message)
    elif body:
        text = _json_compact(body)
    else:
        text = "(no status code or body)"
    return ProviderHttpError(text, headers=_legacy_headers(headers), error=body)


def _json_compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


async def openai_request(call: Callable[[], Awaitable[_T]], options: Mapping[str, Any]) -> _T:
    """Run an SDK request with pi's abortable, SDK-compatible retry policy.

    HTTP errors become :class:`ProviderHttpError` (status, headers, parsed ``error`` body) and
    connection failures a status-less one (``"Connection error."``, ``"Request timed out."``).
    """

    async def attempt() -> _T:
        try:
            return await call()
        except openai.APIStatusError as error:
            raise status_error_from_sdk(error) from error
        except openai.APIConnectionError as error:
            raise ProviderHttpError(error.message) from error

    return await retry_provider_request(
        attempt,
        max_retries=options.get("max_retries") or 0,
        max_retry_delay_ms=options.get("max_retry_delay_ms"),
        signal=options.get("signal"),
    )


def open_sdk_stream(raw: Any) -> AsyncStream[Any]:
    """The event stream of a ``with_raw_response`` streaming call, yielding plain JSON."""
    return raw.parse(to=_RawStream)


async def iter_sdk_stream(stream: AsyncStream[Any]) -> AsyncIterator[Any]:
    """Iterate SDK stream events; ``error`` events raise like the JS SDK's ``Stream``."""
    try:
        async for event in stream:
            yield event
    except (openai.APIStatusError, openai.APIConnectionError):
        raise
    except openai.APIError as error:
        raise stream_error_from_sdk(error, stream.response.headers) from error
    finally:
        await stream.close()
