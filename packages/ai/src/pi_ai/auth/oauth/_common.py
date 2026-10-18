"""Shared plumbing for the built-in OAuth flows: HTTP, clock, URLs and JSON.

The flows run offline in tests by replacing :data:`http_client` (an
``httpx.AsyncClient`` with a mock transport), :func:`now_ms` and :func:`sleep`
(a virtual clock). Flow modules always call these through the module, never
through ``from ... import``, so a single patch applies everywhere.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import math
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard
from urllib.parse import parse_qsl, quote_plus, unquote

import httpx

from ...abort import AbortError, AbortSignal
from ...abort import abort_scope as _abort_scope
from ...abort import sleep as _abortable_sleep
from ...utils.http import get_http_client

__all__ = [
    "OAuthError",
    "http_client",
    "client_for",
    "now_ms",
    "sleep",
    "fetch",
    "login_cancelled",
    "status_text",
    "json_stringify",
    "parse_json",
    "read_json_record",
    "is_number",
    "expires_in_ms",
    "form_encode",
    "parse_search_params",
    "ParsedUrl",
    "parse_url",
    "parse_authorization_input",
    "base64url_encode",
    "random_base64url",
    "random_hex",
    "MAX_SAFE_INTEGER",
]

#: ``Number.MAX_SAFE_INTEGER``: expiry of credentials that never expire.
MAX_SAFE_INTEGER = 9007199254740991
_FETCH_TIMEOUT_MESSAGE = "The operation was aborted due to timeout"


class OAuthError(Exception):
    """A failed OAuth login, token exchange or refresh."""


#: Client for every OAuth request instead of the pooled default (tests, custom transports).
http_client: httpx.AsyncClient | None = None


def client_for(url: str) -> httpx.AsyncClient:
    return http_client if http_client is not None else get_http_client(url)


def now_ms() -> int:
    """Wall-clock time in milliseconds (``Date.now()``)."""
    return time.time_ns() // 1_000_000


async def sleep(seconds: float, signal: AbortSignal | None = None) -> None:
    """Abortable sleep; raises :class:`AbortError` with the signal's reason."""
    await _abortable_sleep(seconds, signal)


def login_cancelled() -> AbortError:
    return AbortError("Login cancelled")


async def fetch(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    body: str | bytes | None = None,
    signal: AbortSignal | None = None,
    timeout_ms: float | None = None,
) -> httpx.Response:
    """Send one request and read the whole body, like ``fetch()`` with an abort signal.

    ``signal`` aborts with :class:`AbortError`. ``timeout_ms`` bounds the whole exchange
    (``AbortSignal.timeout``) and raises :class:`TimeoutError`.
    """
    client = client_for(url)
    async with _abort_scope(signal):
        try:
            async with asyncio.timeout(timeout_ms / 1000 if timeout_ms is not None else None):
                return await client.request(method, url, headers=dict(headers) if headers else None, content=body)
        except TimeoutError:
            raise TimeoutError(_FETCH_TIMEOUT_MESSAGE) from None


def status_text(response: httpx.Response) -> str:
    """``response.statusText``."""
    return response.reason_phrase or ""


def json_stringify(value: Any) -> str:
    """``JSON.stringify`` for JSON values (compact separators, ``null`` for ``None``)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json(response: httpx.Response) -> Any:
    """``response.json()``: raises ``ValueError`` for an invalid body."""
    return json.loads(response.text)


def read_json_record(response: httpx.Response) -> dict[str, Any] | None:
    """The JSON body when it is an object, otherwise ``None`` (never raises)."""
    try:
        value = json.loads(response.text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def is_number(value: Any) -> TypeGuard[float]:
    """``typeof value === "number"`` for decoded JSON (booleans are not numbers)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def expires_in_ms(expires_in: Any, what: str = "expires_in") -> int:
    """Milliseconds for an ``expires_in`` seconds value."""
    if not is_number(expires_in) or not math.isfinite(expires_in):
        raise OAuthError(f"OAuth token response has invalid {what}")
    return int(expires_in * 1000)


# -- URLSearchParams --------------------------------------------------------------


def _form_component(value: str) -> str:
    # application/x-www-form-urlencoded byte serializer: only alphanumerics and *-._ stay as is.
    return quote_plus(value, safe="*").replace("~", "%7E")


def form_encode(fields: Mapping[str, str] | list[tuple[str, str]]) -> str:
    """``new URLSearchParams(fields).toString()``."""
    items = fields.items() if isinstance(fields, Mapping) else fields
    return "&".join(f"{_form_component(str(key))}={_form_component(str(value))}" for key, value in items)


def parse_search_params(query: str) -> list[tuple[str, str]]:
    """``new URLSearchParams(query)`` entries (a leading ``?`` is ignored)."""
    if query.startswith("?"):
        query = query[1:]
    return parse_qsl(query, keep_blank_values=True, errors="replace")


def _first_param(entries: list[tuple[str, str]], name: str) -> str | None:
    for key, value in entries:
        if key == name:
            return value
    return None


# -- WHATWG URL subset --------------------------------------------------------------

_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_SPECIAL_PORTS: dict[str, int | None] = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21, "file": None}
_C0_OR_SPACE = "".join(chr(code) for code in range(0x21))
_FORBIDDEN_DOMAIN = frozenset(" #%/:<>?@[\\]^|\x7f") | frozenset(chr(code) for code in range(0x20))
_FRAGMENT_SET = frozenset(' "<>`')
_QUERY_SET = frozenset(' "#<>')
_SPECIAL_QUERY_SET = _QUERY_SET | {"'"}
_PATH_SET = _QUERY_SET | frozenset("?`{}")
_USERINFO_SET = _PATH_SET | frozenset("/:;=@[\\]^|")


def _percent_encode(value: str, encode_set: frozenset[str]) -> str:
    out: list[str] = []
    for char in value:
        code = ord(char)
        if code < 0x20 or code > 0x7E or char in encode_set:
            out.extend(f"%{byte:02X}" for byte in char.encode("utf-8", "surrogatepass"))
        else:
            out.append(char)
    return "".join(out)


def _percent_decode(value: str) -> str:
    return unquote(value, errors="replace")


def _is_dot(segment: str) -> bool:
    return segment == "." or segment.lower() == "%2e"


def _is_double_dot(segment: str) -> bool:
    return segment.lower() in ("..", ".%2e", "%2e.", "%2e%2e")


def _normalize_path(path: str) -> str:
    segments = path.split("/")[1:]
    output: list[str] = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if _is_double_dot(segment):
            if output:
                output.pop()
            if last:
                output.append("")
        elif _is_dot(segment):
            if last:
                output.append("")
        else:
            output.append(segment)
    return "/" + "/".join(output)


def _parse_host(raw: str, special: bool) -> str | None:
    if raw.startswith("["):
        if not raw.endswith("]"):
            return None
        try:
            return f"[{ipaddress.IPv6Address(raw[1:-1]).compressed}]"
        except ValueError:
            return None
    if not special:
        return _percent_encode(raw, frozenset())
    host = _percent_decode(raw)
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return None
    host = host.lower()
    if not host or any(char in _FORBIDDEN_DOMAIN for char in host):
        return None
    return host


@dataclass(slots=True)
class ParsedUrl:
    """The parts of a WHATWG URL that the OAuth flows use (``new URL(...)``)."""

    scheme: str
    username: str
    password: str
    #: Host without port (lowercased for special schemes; ``None`` without an authority).
    hostname: str | None
    port: int | None
    pathname: str
    #: Query without ``?`` (``None`` when absent).
    query: str | None
    #: Fragment without ``#`` (``None`` when absent).
    fragment: str | None

    @property
    def protocol(self) -> str:
        return f"{self.scheme}:"

    @property
    def special(self) -> bool:
        return self.scheme in _SPECIAL_PORTS

    @property
    def host(self) -> str:
        if self.hostname is None:
            return ""
        return self.hostname if self.port is None else f"{self.hostname}:{self.port}"

    @property
    def origin(self) -> str:
        if self.special and self.scheme != "file":
            return f"{self.scheme}://{self.host}"
        return "null"

    @property
    def search(self) -> str:
        return f"?{self.query}" if self.query else ""

    @property
    def search_params(self) -> list[tuple[str, str]]:
        return parse_search_params(self.query or "")

    def param(self, name: str) -> str | None:
        """``url.searchParams.get(name)``."""
        return _first_param(self.search_params, name)

    def with_search(self, query: str) -> ParsedUrl:
        """A copy with the query replaced (``url.search = query``)."""
        encoded = _percent_encode(query, _SPECIAL_QUERY_SET if self.special else _QUERY_SET) if query else None
        return ParsedUrl(
            self.scheme,
            self.username,
            self.password,
            self.hostname,
            self.port,
            self.pathname,
            encoded,
            self.fragment,
        )

    @property
    def href(self) -> str:
        out = [self.scheme, ":"]
        if self.hostname is not None:
            out.append("//")
            if self.username or self.password:
                out.append(self.username)
                if self.password:
                    out.append(f":{self.password}")
                out.append("@")
            out.append(self.host)
        out.append(self.pathname)
        if self.query is not None:
            out.append(f"?{self.query}")
        if self.fragment is not None:
            out.append(f"#{self.fragment}")
        return "".join(out)

    def __str__(self) -> str:
        return self.href


def parse_url(value: str, base: str | None = None) -> ParsedUrl | None:
    """Parse an absolute URL like ``new URL(value)``; ``None`` where that throws.

    With ``base``, a value starting with ``/`` resolves against the base origin.
    Covers the WHATWG behavior the flows rely on (scheme detection, host validation and
    lowercasing, percent-encoding, dot segments, default ports); IPv4 shorthand forms
    are not normalized.
    """
    value = re.sub(r"[\t\n\r]", "", value.strip(_C0_OR_SPACE))
    if base is not None and not _SCHEME_RE.match(value):
        parsed_base = parse_url(base)
        if parsed_base is None:
            return None
        if value.startswith("/") and not value.startswith("//"):
            value = f"{parsed_base.origin}{value}"
        else:
            value = f"{parsed_base.origin}/{value}"
    match = _SCHEME_RE.match(value)
    if match is None:
        return None
    scheme = match.group(1).lower()
    rest = value[match.end() :]
    special = scheme in _SPECIAL_PORTS

    fragment: str | None = None
    hash_index = rest.find("#")
    if hash_index >= 0:
        rest, fragment = rest[:hash_index], _percent_encode(rest[hash_index + 1 :], _FRAGMENT_SET)
    query: str | None = None
    query_index = rest.find("?")
    if query_index >= 0:
        rest, query = rest[:query_index], rest[query_index + 1 :]
        query = _percent_encode(query, _SPECIAL_QUERY_SET if special else _QUERY_SET)

    authority: str | None
    if special:
        rest = rest.replace("\\", "/")
    if special and scheme != "file":
        authority, slash, path = rest.lstrip("/").partition("/")
        path = f"/{path}" if slash else "/"
    elif rest.startswith("//"):
        authority, slash, path = rest[2:].partition("/")
        path = f"/{path}" if slash else ("/" if special else "")
    else:
        authority, path = None, rest

    username = password = ""
    hostname: str | None = None
    port: int | None = None
    if authority is not None:
        userinfo, at, hostport = authority.rpartition("@")
        if at:
            raw_user, _, raw_password = userinfo.partition(":")
            username = _percent_encode(raw_user, _USERINFO_SET)
            password = _percent_encode(raw_password, _USERINFO_SET)
        if hostport.startswith("["):
            end = hostport.find("]")
            if end < 0:
                return None
            raw_host, port_text = hostport[: end + 1], hostport[end + 1 :]
            if port_text and not port_text.startswith(":"):
                return None
            port_text = port_text[1:]
        else:
            raw_host, _, port_text = hostport.partition(":")
        if port_text:
            if not port_text.isdigit() or int(port_text) > 65535:
                return None
            port = int(port_text)
            if port == _SPECIAL_PORTS.get(scheme):
                port = None
        if special and scheme != "file" and not raw_host:
            return None
        hostname = _parse_host(raw_host, special) if raw_host else ""
        if hostname is None:
            return None

    if authority is not None or special:
        pathname = _normalize_path(_percent_encode(path, _PATH_SET)) if path else path
    else:
        pathname = _percent_encode(path, frozenset())
    return ParsedUrl(scheme, username, password, hostname, port, pathname, query, fragment)


# -- randomness ---------------------------------------------------------------------


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def random_base64url(size: int) -> str:
    """``randomBytes(size).toString("base64url")``."""
    return base64url_encode(secrets.token_bytes(size))


def random_hex(size: int) -> str:
    """``randomBytes(size).toString("hex")``."""
    return secrets.token_hex(size)


def parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    """``(code, state)`` from a pasted redirect URL, ``code#state``, query string, or bare code."""
    value = value.strip()
    if not value:
        return None, None

    url = parse_url(value)
    if url is not None:
        return url.param("code"), url.param("state")

    if "#" in value:
        # JavaScript `split("#", 2)`: anything after a second `#` is dropped.
        parts = value.split("#")
        return parts[0], parts[1]

    if "code=" in value:
        params = parse_search_params(value)
        return _first_param(params, "code"), _first_param(params, "state")

    return value, None
