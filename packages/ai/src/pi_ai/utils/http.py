"""Shared HTTP transport for provider adapters.

Adapters built on vendor SDKs (anthropic, openai, google-genai) hand the pooled
client from :func:`get_http_client` to the SDK. Adapters for APIs that pi calls
without an SDK (Mistral, OpenAI Codex, pi-messages, classifiers, OAuth) send
requests with :func:`send_request`, which provides header suppression with
``None`` values, provider-scoped proxy settings, SDK-compatible retries and
error messages, and prompt cancellation through :class:`pi_ai.AbortSignal`.

Connection pools are shared per event loop (and per proxy configuration), so
consecutive requests reuse TLS connections.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import random
import time
import weakref
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from ..abort import AbortError, AbortSignal, sleep
from . import fastjson
from .misc import get_provider_env_value

__all__ = [
    "ProviderHttpError",
    "SSEEvent",
    "DEFAULT_TIMEOUT_MS",
    "get_http_client",
    "aclose_http_clients",
    "build_headers",
    "make_status_error",
    "send_request",
    "retry_provider_request",
    "iter_sse",
    "iter_json_lines",
]

DEFAULT_TIMEOUT_MS = 600_000
DEFAULT_MAX_RETRY_DELAY_MS = 60_000

ErrorStyle = Literal["openai", "anthropic", "raw"]


class ProviderHttpError(Exception):
    """A non-2xx provider response (or a transport failure when ``status`` is ``None``).

    Attributes follow the shape of the vendor SDK errors that pi normalizes:
    ``status``, ``headers``, the raw ``body`` text, and the parsed ``error`` object.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        headers: httpx.Headers | None = None,
        body: str | None = None,
        error: Any = None,
        include_body: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers
        #: Raw body text. Only exposed for display when ``include_body`` (Mistral-style errors).
        self.body = body if include_body else None
        self.raw_body = body
        #: Parsed error payload, like ``error.error`` on OpenAI SDK errors.
        self.error = error


def make_status_error(
    status: int, body_text: str, headers: httpx.Headers | None = None, style: ErrorStyle = "openai"
) -> ProviderHttpError:
    """Build an error whose message matches the vendor SDK wording.

    * ``openai``: ``"<status> <error.message>"`` using the body's ``error`` field.
    * ``anthropic``: ``"<status> <json body>"`` (the whole body when it has no ``message``).
    * ``raw``: ``"<status> <body text>"``.
    """
    parsed: Any = None
    if body_text:
        try:
            parsed = json.loads(body_text)
        except ValueError:
            parsed = None
    error_obj: Any
    if style == "openai":
        error_obj = parsed.get("error") if isinstance(parsed, dict) else None
        message: Any
        if isinstance(error_obj, dict) and error_obj.get("message"):
            raw = error_obj["message"]
            message = raw if isinstance(raw, str) else json.dumps(raw)
        elif error_obj:
            message = json.dumps(error_obj)
        else:
            message = None if parsed is not None else (body_text or None)
        include_body = False
    elif style == "anthropic":
        error_obj = parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
            message = parsed["message"]
        elif parsed is not None:
            message = json.dumps(parsed, separators=(",", ":"))
        else:
            message = body_text or None
        include_body = False
    else:
        error_obj = parsed
        message = body_text or None
        include_body = True
    text = f"{status} {message}" if message else f"{status} status code (no body)"
    return ProviderHttpError(
        text, status=status, headers=headers, body=body_text, error=error_obj, include_body=include_body
    )


# ---------------------------------------------------------------------------
# Client pool
# ---------------------------------------------------------------------------

_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str | None, httpx.AsyncClient]] = (
    weakref.WeakKeyDictionary()
)

_LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=40, keepalive_expiry=60)


def _proxy_for(url: str, env: Mapping[str, str] | None) -> str | None:
    """Proxy URL from provider-scoped env overrides only (``os.environ`` is honored by httpx itself)."""
    if not env:
        return None
    scheme = "https" if url.startswith("https:") else "http"
    names = ("HTTPS_PROXY", "https_proxy") if scheme == "https" else ("HTTP_PROXY", "http_proxy")
    for name in (*names, "ALL_PROXY", "all_proxy"):
        value = env.get(name)
        if value:
            no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or get_provider_env_value("NO_PROXY")
            if no_proxy and _host_matches_no_proxy(url, no_proxy):
                return None
            return value
    return None


def _host_matches_no_proxy(url: str, no_proxy: str) -> bool:
    host = httpx.URL(url).host
    for entry in (part.strip() for part in no_proxy.split(",")):
        if not entry:
            continue
        if entry == "*":
            return True
        entry = entry.lstrip(".")
        if host == entry or host.endswith("." + entry):
            return True
    return False


def get_http_client(url: str = "", options: Mapping[str, Any] | None = None) -> httpx.AsyncClient:
    """The client for a request: ``options["http_client"]`` or a pooled per-loop client."""
    if options:
        custom = options.get("http_client")
        if custom is not None:
            return custom
    proxy = _proxy_for(url, options.get("env") if options else None)
    loop = asyncio.get_running_loop()
    pool = _clients.get(loop)
    if pool is None:
        pool = {}
        _clients[loop] = pool
    client = pool.get(proxy)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            proxy=proxy,
            limits=_LIMITS,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_MS / 1000),
            follow_redirects=True,
            trust_env=proxy is None,
        )
        pool[proxy] = client
    return client


async def aclose_http_clients() -> None:
    """Close the pooled clients of the running event loop."""
    loop = asyncio.get_running_loop()
    pool = _clients.pop(loop, None)
    for client in (pool or {}).values():
        await client.aclose()


def build_headers(*sources: Mapping[str, str | None] | None) -> dict[str, str]:
    """Merge header sources case-insensitively; later sources win, ``None`` removes a header."""
    merged: dict[str, tuple[str, str]] = {}
    for source in sources:
        if not source:
            continue
        for name, value in source.items():
            lower = name.lower()
            if value is None:
                merged.pop(lower, None)
            else:
                merged[lower] = (name, value)
    return {name: value for name, value in merged.values()}


# ---------------------------------------------------------------------------
# Requests and retries
# ---------------------------------------------------------------------------


def _timeout(options: Mapping[str, Any] | None) -> httpx.Timeout:
    timeout_ms = options.get("timeout_ms") if options else None
    seconds = (timeout_ms if timeout_ms is not None else DEFAULT_TIMEOUT_MS) / 1000
    # ``read`` is the idle gap between chunks, so an active stream is never cut off.
    return httpx.Timeout(seconds)


async def send_request(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Any = None,
    content: bytes | str | None = None,
    options: Mapping[str, Any] | None = None,
    stream: bool = True,
    error_style: ErrorStyle = "openai",
    params: Mapping[str, Any] | None = None,
    no_retry_statuses: Sequence[int] = (),
) -> httpx.Response:
    """Send a provider request with SDK-compatible retries.

    ``body`` is JSON-encoded (``content`` sends raw bytes). Non-2xx responses raise
    :class:`ProviderHttpError` after reading the error body. Successful streaming
    responses are returned open; the caller iterates and closes them.
    """
    client = get_http_client(url, options)
    request_headers = dict(headers)
    if body is not None:
        content = fastjson.dumps_bytes(body)
        if not any(key.lower() == "content-type" for key in request_headers):
            request_headers["Content-Type"] = "application/json"
    timeout = _timeout(options)

    async def attempt() -> httpx.Response:
        request = client.build_request(
            method, url, headers=request_headers, content=content, params=params, timeout=timeout
        )
        try:
            response = await client.send(request, stream=stream)
        except httpx.TimeoutException as error:
            raise ProviderHttpError(f"Request timed out: {error}") from error
        except httpx.TransportError as error:
            raise ProviderHttpError(f"Connection error: {error or type(error).__name__}") from error
        if response.status_code >= 400:
            try:
                raw = await response.aread()
            finally:
                await response.aclose()
            text = raw.decode("utf-8", "replace")
            raise make_status_error(response.status_code, text, response.headers, error_style)
        return response

    signal = options.get("signal") if options else None
    return await retry_provider_request(
        attempt,
        max_retries=(options.get("max_retries") if options else None) or 0,
        max_retry_delay_ms=options.get("max_retry_delay_ms") if options else None,
        signal=signal,
        no_retry_statuses=no_retry_statuses,
    )


def _is_retryable(error: ProviderHttpError) -> bool:
    headers = error.headers
    should_retry = headers.get("x-should-retry") if headers is not None else None
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False
    status = error.status
    if status is None:
        return True
    return status in (408, 409, 429) or status >= 500


def _validate_server_delay(delay_ms: float, max_retry_delay_ms: float | None, message: str) -> float:
    max_delay = max_retry_delay_ms if max_retry_delay_ms is not None else DEFAULT_MAX_RETRY_DELAY_MS
    if max_delay > 0 and delay_ms > max_delay:
        raise ProviderHttpError(
            f"Server requested {int(-(-delay_ms // 1000))}s retry delay (max: {int(-(-max_delay // 1000))}s). {message}"
        )
    return delay_ms


def _retry_delay_ms(error: ProviderHttpError, retry_index: int, max_retry_delay_ms: float | None) -> float:
    headers = error.headers
    if headers is not None:
        retry_after_ms = headers.get("retry-after-ms")
        if retry_after_ms:
            try:
                return _validate_server_delay(float(retry_after_ms), max_retry_delay_ms, str(error))
            except ValueError:
                pass
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                delay_ms = float(retry_after) * 1000
            except ValueError:
                parsed = email.utils.parsedate_to_datetime(retry_after) if retry_after else None
                delay_ms = (parsed.timestamp() - time.time()) * 1000 if parsed else float("nan")
            if delay_ms == delay_ms:  # not NaN
                return _validate_server_delay(delay_ms, max_retry_delay_ms, str(error))
    exponential = min(0.5 * 2**retry_index, 8) * 1000
    return exponential * (1 - random.random() * 0.25)


async def retry_provider_request(
    request: Any,
    *,
    max_retries: int = 0,
    max_retry_delay_ms: float | None = None,
    signal: AbortSignal | None = None,
    no_retry_statuses: Sequence[int] = (),
) -> Any:
    """Retry ``request()`` like the OpenAI/Anthropic SDKs, with an interruptible backoff.

    Provider-requested delays above ``max_retry_delay_ms`` (60s by default, 0 disables)
    fail immediately so outer retry logic can surface them.
    """
    retries_remaining = max_retries
    while True:
        try:
            return await request()
        except ProviderHttpError as error:
            if signal is not None and signal.aborted:
                raise AbortError(signal.reason) from error
            if retries_remaining <= 0 or not _is_retryable(error):
                raise
            if error.status is not None and error.status in no_retry_statuses:
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await sleep(_retry_delay_ms(error, retry_index, max_retry_delay_ms) / 1000, signal)


# ---------------------------------------------------------------------------
# Server-sent events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SSEEvent:
    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None

    def json(self) -> Any:
        return fastjson.loads(self.data)


async def iter_sse(response: httpx.Response) -> AsyncIterator[SSEEvent]:
    """Decode a ``text/event-stream`` response into :class:`SSEEvent` objects."""
    data_lines: list[str] = []
    event_name: str | None = None
    event_id: str | None = None
    retry: int | None = None
    async for line in response.aiter_lines():
        if not line:
            if data_lines or event_name is not None:
                yield SSEEvent("\n".join(data_lines), event_name, event_id, retry)
            data_lines = []
            event_name = None
            retry = None
            continue
        if line[0] == ":":
            continue
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            data_lines.append(value)
        elif name == "event":
            event_name = value
        elif name == "id":
            event_id = value
        elif name == "retry":
            try:
                retry = int(value)
            except ValueError:
                pass
    if data_lines or event_name is not None:
        yield SSEEvent("\n".join(data_lines), event_name, event_id, retry)


async def iter_json_lines(response: httpx.Response) -> AsyncIterator[Any]:
    """Decode a newline-delimited JSON response."""
    async for line in response.aiter_lines():
        line = line.strip()
        if line:
            yield fastjson.loads(line)
