"""AWS plumbing for the Bedrock adapter.

pi talks to Bedrock through the AWS SDK. The Python port calls the Bedrock Runtime
REST API directly, so this module provides what the SDK would: Signature Version 4
request signing, the ``application/vnd.amazon.eventstream`` codec, shared
config/credentials file parsing, and a credential provider chain modeled on the
SDK's default chain (environment, shared files incl. assume-role, credential
process and web identity profiles, web identity token file, ECS container
credentials, EC2 instance metadata). When the optional ``botocore`` package is
importable its provider chain is used instead, for full fidelity (SSO, MFA
caching, ...).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import struct
import threading
import time
import uuid
import zlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlencode
from xml.etree import ElementTree

import httpx

from .misc import get_provider_env_value

__all__ = [
    "AwsCredentials",
    "AwsCredentialsError",
    "derive_signing_key",
    "sign_request",
    "EventStreamError",
    "EventStreamMessage",
    "EventStreamDecoder",
    "encode_event_stream_message",
    "iter_event_stream",
    "load_shared_profiles",
    "resolve_profile_region",
    "resolve_aws_credentials",
    "aws_dns_suffix",
    "clear_credentials_cache",
    "b64decode_forgiving",
]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AwsCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None
    #: Unix time in seconds when temporary credentials expire.
    expiration: float | None = None


class AwsCredentialsError(Exception):
    """No credential provider could supply AWS credentials (or one failed)."""


# ---------------------------------------------------------------------------
# Signature Version 4
# ---------------------------------------------------------------------------

_ALGORITHM = "AWS4-HMAC-SHA256"
#: Headers the AWS SDK never signs (proxies and clients may rewrite them).
_UNSIGNABLE_HEADERS = frozenset(
    {
        "authorization",
        "cache-control",
        "connection",
        "expect",
        "from",
        "keep-alive",
        "max-forwards",
        "pragma",
        "referer",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "user-agent",
        "x-amzn-trace-id",
    }
)
#: Headers :func:`sign_request` owns; caller values are replaced.
_SIGNER_HEADERS = frozenset({"authorization", "x-amz-date", "x-amz-security-token", "x-amz-content-sha256"})
_WHITESPACE = re.compile(r"\s+")


def derive_signing_key(secret_access_key: str, date_stamp: str, region: str, service: str) -> bytes:
    """The SigV4 signing key for ``date_stamp`` (``YYYYMMDD``), region and service."""
    key = hmac.new(f"AWS4{secret_access_key}".encode(), date_stamp.encode(), hashlib.sha256).digest()
    for part in (region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return key


def _uri_escape(text: str) -> str:
    return quote(text, safe="")


def _canonical_path(path: str) -> str:
    # Non-S3 services normalize the path and URI-encode it once more ("double encoding").
    segments: list[str] = []
    for segment in path.split("/"):
        if not segment or segment == ".":
            continue
        if segment == "..":
            if segments:
                segments.pop()
        else:
            segments.append(segment)
    normalized = ("/" if path.startswith("/") else "") + "/".join(segments)
    if segments and path.endswith("/"):
        normalized += "/"
    return quote(normalized or "/", safe="/")


def _canonical_query(query: str) -> str:
    pairs: list[tuple[str, str]] = []
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        pairs.append((_uri_escape(unquote(key)), _uri_escape(unquote(value))))
    pairs.sort()
    return "&".join(f"{key}={value}" for key, value in pairs)


def sign_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    credentials: AwsCredentials,
    *,
    region: str,
    service: str,
    now: datetime | None = None,
    apply_checksum: bool = True,
) -> dict[str, str]:
    """Sign a request with AWS Signature Version 4 (``Authorization`` header).

    Returns ``headers`` plus ``host``, ``x-amz-date``, ``x-amz-security-token`` (for
    temporary credentials), ``x-amz-content-sha256`` (when ``apply_checksum``, as the
    JS SDK sends it) and ``authorization``. Every other header present is signed except
    the ones proxies may rewrite (``user-agent``, hop-by-hop headers), so send exactly
    these headers.
    """
    parsed = httpx.URL(url)
    signing_time = (now or datetime.now(UTC)).astimezone(UTC)
    amz_date = signing_time.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = amz_date[:8]
    payload_hash = hashlib.sha256(body).hexdigest()

    result: dict[str, str] = {}
    has_host = False
    for name, value in headers.items():
        lower = name.lower()
        if lower in _SIGNER_HEADERS:
            continue
        if lower == "host":
            has_host = True
        result[name] = value
    if not has_host:
        result["host"] = parsed.netloc.decode("ascii")
    result["x-amz-date"] = amz_date
    if credentials.session_token:
        result["x-amz-security-token"] = credentials.session_token
    if apply_checksum:
        result["x-amz-content-sha256"] = payload_hash

    canonical: dict[str, str] = {}
    for name, value in result.items():
        lower = name.lower()
        if lower in _UNSIGNABLE_HEADERS or lower.startswith(("proxy-", "sec-")):
            continue
        canonical[lower] = _WHITESPACE.sub(" ", str(value).strip())
    signed_names = sorted(canonical)
    signed_headers = ";".join(signed_names)
    raw_path, _, raw_query = parsed.raw_path.decode("ascii").partition("?")
    canonical_request = "\n".join(
        (
            method.upper(),
            _canonical_path(raw_path),
            _canonical_query(raw_query),
            "".join(f"{name}:{canonical[name]}\n" for name in signed_names),
            signed_headers,
            payload_hash,
        )
    )
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join((_ALGORITHM, amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()))
    signing_key = derive_signing_key(credentials.secret_access_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    result["authorization"] = (
        f"{_ALGORITHM} Credential={credentials.access_key_id}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return result


# ---------------------------------------------------------------------------
# Event stream codec (application/vnd.amazon.eventstream)
# ---------------------------------------------------------------------------

_PRELUDE_LENGTH = 12
_MESSAGE_OVERHEAD = 16  # prelude (12) + message CRC (4)
_MAX_MESSAGE_LENGTH = 16 * 1024 * 1024
_MAX_HEADERS_LENGTH = 128 * 1024

_TYPE_TRUE = 0
_TYPE_FALSE = 1
_TYPE_BYTE = 2
_TYPE_SHORT = 3
_TYPE_INTEGER = 4
_TYPE_LONG = 5
_TYPE_BYTES = 6
_TYPE_STRING = 7
_TYPE_TIMESTAMP = 8
_TYPE_UUID = 9

_FIXED_HEADER_FORMATS = {_TYPE_BYTE: ">b", _TYPE_SHORT: ">h", _TYPE_INTEGER: ">i", _TYPE_LONG: ">q"}


class EventStreamError(Exception):
    """Malformed event stream data (bad lengths or checksum mismatch)."""


@dataclass(slots=True)
class EventStreamMessage:
    """One decoded frame. Timestamp headers decode to epoch milliseconds, UUIDs to :class:`uuid.UUID`."""

    headers: dict[str, Any] = field(default_factory=dict)
    payload: bytes = b""


def _decode_headers(data: bytes) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    position = 0
    end = len(data)
    try:
        while position < end:
            name_length = data[position]
            position += 1
            name = data[position : position + name_length].decode("utf-8")
            position += name_length
            kind = data[position]
            position += 1
            value: Any
            if kind == _TYPE_TRUE:
                value = True
            elif kind == _TYPE_FALSE:
                value = False
            elif kind in _FIXED_HEADER_FORMATS:
                fmt = _FIXED_HEADER_FORMATS[kind]
                (value,) = struct.unpack_from(fmt, data, position)
                position += struct.calcsize(fmt)
            elif kind in (_TYPE_BYTES, _TYPE_STRING):
                (length,) = struct.unpack_from(">H", data, position)
                position += 2
                raw = data[position : position + length]
                if len(raw) != length:
                    raise EventStreamError("Truncated event stream header value")
                position += length
                value = raw if kind == _TYPE_BYTES else raw.decode("utf-8")
            elif kind == _TYPE_TIMESTAMP:
                (value,) = struct.unpack_from(">q", data, position)
                position += 8
            elif kind == _TYPE_UUID:
                raw = data[position : position + 16]
                if len(raw) != 16:
                    raise EventStreamError("Truncated event stream header value")
                value = uuid.UUID(bytes=raw)
                position += 16
            else:
                raise EventStreamError(f"Unrecognized event stream header type tag: {kind}")
            if position > end:
                raise EventStreamError("Truncated event stream header")
            headers[name] = value
    except (IndexError, struct.error, UnicodeDecodeError) as error:
        raise EventStreamError(f"Malformed event stream headers: {error}") from error
    return headers


class EventStreamDecoder:
    """Incremental decoder: :meth:`feed` raw bytes, get complete messages back.

    Both the prelude and the message CRC32 checksums are verified.
    """

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[EventStreamMessage]:
        buffer = self._buffer
        buffer += data
        messages: list[EventStreamMessage] = []
        offset = 0
        available = len(buffer)
        while available - offset >= _PRELUDE_LENGTH:
            total_length, headers_length, prelude_crc = struct.unpack_from(">III", buffer, offset)
            if zlib.crc32(memoryview(buffer)[offset : offset + 8]) != prelude_crc:
                raise EventStreamError("Event stream prelude checksum mismatch")
            if (
                total_length < _MESSAGE_OVERHEAD
                or total_length > _MAX_MESSAGE_LENGTH
                or headers_length > _MAX_HEADERS_LENGTH
                or headers_length > total_length - _MESSAGE_OVERHEAD
            ):
                raise EventStreamError(f"Invalid event stream message length: {total_length}")
            if available - offset < total_length:
                break
            end = offset + total_length
            (message_crc,) = struct.unpack_from(">I", buffer, end - 4)
            if zlib.crc32(memoryview(buffer)[offset : end - 4]) != message_crc:
                raise EventStreamError("Event stream message checksum mismatch")
            headers_start = offset + _PRELUDE_LENGTH
            payload_start = headers_start + headers_length
            messages.append(
                EventStreamMessage(
                    _decode_headers(bytes(buffer[headers_start:payload_start])),
                    bytes(buffer[payload_start : end - 4]),
                )
            )
            offset = end
        if offset:
            del buffer[:offset]
        return messages

    def finish(self) -> None:
        """Raise if the stream ended in the middle of a message."""
        if self._buffer:
            raise EventStreamError(f"Event stream ended with {len(self._buffer)} bytes of an incomplete message")


def _encode_header(name: str, value: Any) -> bytes:
    raw_name = name.encode("utf-8")
    prefix = struct.pack(">B", len(raw_name)) + raw_name
    if isinstance(value, bool):
        return prefix + struct.pack(">B", _TYPE_TRUE if value else _TYPE_FALSE)
    if isinstance(value, int):
        if -(2**31) <= value < 2**31:
            return prefix + struct.pack(">Bi", _TYPE_INTEGER, value)
        return prefix + struct.pack(">Bq", _TYPE_LONG, value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return prefix + struct.pack(">BH", _TYPE_BYTES, len(data)) + data
    if isinstance(value, uuid.UUID):
        return prefix + struct.pack(">B", _TYPE_UUID) + value.bytes
    if isinstance(value, datetime):
        return prefix + struct.pack(">Bq", _TYPE_TIMESTAMP, int(value.timestamp() * 1000))
    data = str(value).encode("utf-8")
    return prefix + struct.pack(">BH", _TYPE_STRING, len(data)) + data


def encode_event_stream_message(headers: Mapping[str, Any], payload: bytes = b"") -> bytes:
    """Encode one event stream frame (strings, ints, bools, bytes, UUIDs and datetimes as header values)."""
    header_bytes = b"".join(_encode_header(name, value) for name, value in headers.items())
    prelude = struct.pack(">II", _MESSAGE_OVERHEAD + len(header_bytes) + len(payload), len(header_bytes))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + header_bytes + payload
    return message + struct.pack(">I", zlib.crc32(message))


async def iter_event_stream(response: httpx.Response) -> AsyncIterator[EventStreamMessage]:
    """Decode an ``application/vnd.amazon.eventstream`` response body."""
    decoder = EventStreamDecoder()
    async for chunk in response.aiter_bytes():
        for message in decoder.feed(chunk):
            yield message
    decoder.finish()


# ---------------------------------------------------------------------------
# Shared config and credentials files
# ---------------------------------------------------------------------------

_SECTION_PREFIX = re.compile(r"^([\w-]+)\s+([\"']?)([\w\-@+.%:/]+)\2$")
_COMMENT = re.compile(r"(?:^|\s)[;#]")


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Tolerant INI parser following the AWS SDK's rules (nested sub-properties are prefixed)."""
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    sub_section: str | None = None
    for raw_line in text.splitlines():
        line = _COMMENT.split(raw_line, maxsplit=1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            sub_section = None
            current = line[1:-1].strip()
            continue
        if current is None:
            continue
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not name:
            continue
        value = value.strip()
        if not value:
            sub_section = name
            continue
        if sub_section is not None and raw_line.lstrip() == raw_line:
            sub_section = None
        key = f"{sub_section}.{name}" if sub_section is not None else name
        sections.setdefault(current, {})[key.lower() if sub_section is None else key] = value
    return sections


def _read_ini(path: Path) -> dict[str, dict[str, str]]:
    try:
        return _parse_ini(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {}


def _home_path(*parts: str) -> Path:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or str(Path.home())
    return Path(home, *parts)


def _expand_path(value: str) -> Path:
    return _home_path(value[2:]) if value.startswith("~/") else Path(value)


def load_shared_profiles(env: Mapping[str, str] | None = None) -> dict[str, dict[str, str]]:
    """Profiles from ``~/.aws/config`` and ``~/.aws/credentials`` (credentials file values win).

    ``AWS_CONFIG_FILE`` and ``AWS_SHARED_CREDENTIALS_FILE`` override the locations.
    """
    config_file = get_provider_env_value("AWS_CONFIG_FILE", env)
    credentials_file = get_provider_env_value("AWS_SHARED_CREDENTIALS_FILE", env)
    config = _read_ini(_expand_path(config_file) if config_file else _home_path(".aws", "config"))
    credentials = _read_ini(_expand_path(credentials_file) if credentials_file else _home_path(".aws", "credentials"))
    profiles: dict[str, dict[str, str]] = {}
    for section, values in config.items():
        if section == "default":
            name = "default"
        else:
            match = _SECTION_PREFIX.match(section)
            if match is None or match.group(1) != "profile":
                continue
            name = match.group(3)
        profiles.setdefault(name, {}).update(values)
    for section, values in credentials.items():
        profiles.setdefault(section, {}).update(values)
    return profiles


def resolve_profile_region(profile: str | None, env: Mapping[str, str] | None = None) -> str | None:
    """The ``region`` configured for ``profile`` (``AWS_PROFILE`` or ``default`` when ``None``)."""
    name = profile or get_provider_env_value("AWS_PROFILE", env) or "default"
    region = load_shared_profiles(env).get(name, {}).get("region")
    return region or None


def aws_dns_suffix(region: str) -> str:
    """DNS suffix of the AWS partition that ``region`` belongs to."""
    if region.startswith("cn-"):
        return "amazonaws.com.cn"
    if region.startswith("us-isob-"):
        return "sc2s.sgov.gov"
    if region.startswith("us-isof-"):
        return "csp.hci.ic.gov"
    if region.startswith("us-iso-"):
        return "c2s.ic.gov"
    if region.startswith("eu-isoe-"):
        return "cloud.adc-e.uk"
    if region.startswith("eusc-"):
        return "amazonaws.eu"
    return "amazonaws.com"


# ---------------------------------------------------------------------------
# Credential provider chain
# ---------------------------------------------------------------------------

#: Temporary credentials are refreshed this long before they expire.
_REFRESH_MARGIN_SECONDS = 300
_METADATA_TIMEOUT = httpx.Timeout(1.0)
_CONTAINER_TIMEOUT = httpx.Timeout(2.0)
_ECS_RELATIVE_HOST = "http://169.254.170.2"
_ALLOWED_CONTAINER_HOSTS = frozenset({"169.254.170.2", "169.254.170.23", "fd00:ec2::23", "localhost"})

_cache: dict[tuple[Any, ...], AwsCredentials] = {}
_botocore_sessions: dict[str | None, Any] = {}
_botocore_lock = threading.Lock()


def clear_credentials_cache() -> None:
    """Forget cached temporary credentials (and cached botocore sessions)."""
    _cache.clear()
    with _botocore_lock:
        _botocore_sessions.clear()


def _cached(key: tuple[Any, ...]) -> AwsCredentials | None:
    credentials = _cache.get(key)
    if credentials is None:
        return None
    if credentials.expiration is not None and credentials.expiration - time.time() <= _REFRESH_MARGIN_SECONDS:
        _cache.pop(key, None)
        return None
    return credentials


def _parse_expiration(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _botocore_available() -> bool:
    try:
        import botocore.session  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def _botocore_credentials(profile: str | None) -> AwsCredentials | None:
    import botocore.session  # pyright: ignore[reportMissingImports]

    with _botocore_lock:
        session = _botocore_sessions.get(profile)
        if session is None:
            session = botocore.session.Session(profile=profile) if profile else botocore.session.Session()
            _botocore_sessions[profile] = session
    credentials = session.get_credentials()
    if credentials is None:
        return None
    frozen = credentials.get_frozen_credentials()
    return AwsCredentials(frozen.access_key, frozen.secret_key, frozen.token or None)


def _from_env(env: Mapping[str, str] | None) -> AwsCredentials | None:
    access_key_id = get_provider_env_value("AWS_ACCESS_KEY_ID", env)
    secret_access_key = get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
    if not access_key_id or not secret_access_key:
        return None
    return AwsCredentials(
        access_key_id,
        secret_access_key,
        get_provider_env_value("AWS_SESSION_TOKEN", env),
        _parse_expiration(get_provider_env_value("AWS_CREDENTIAL_EXPIRATION", env)),
    )


@dataclass(slots=True)
class _ChainContext:
    env: Mapping[str, str] | None
    region: str
    http_client: httpx.AsyncClient | None
    profiles: dict[str, dict[str, str]] | None = None

    def get_profiles(self) -> dict[str, dict[str, str]]:
        if self.profiles is None:
            self.profiles = load_shared_profiles(self.env)
        return self.profiles


async def resolve_aws_credentials(
    *,
    profile: str | None = None,
    env: Mapping[str, str] | None = None,
    region: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    use_botocore: bool | None = None,
) -> AwsCredentials:
    """Resolve credentials like the AWS SDK default provider chain.

    ``profile`` is an explicitly configured profile: it skips environment access keys,
    like the SDK. Without it ``AWS_PROFILE`` (or ``default``) selects the shared-file
    profile. ``region`` is used for STS calls. ``botocore``'s chain is used first when
    importable (``use_botocore=None``); STS, container and instance-metadata results
    are cached until shortly before they expire.
    """
    if use_botocore is None:
        use_botocore = _botocore_available()
    if use_botocore:
        # Scoped access keys are handled by the caller; botocore reads os.environ itself.
        credentials = await asyncio.to_thread(_botocore_credentials, profile)
        if credentials is None:
            raise AwsCredentialsError("Could not load credentials from any providers")
        return credentials

    context = _ChainContext(env, region or "us-east-1", http_client)
    ambient_profile = get_provider_env_value("AWS_PROFILE", env)
    if profile is None and ambient_profile is None:
        credentials = _from_env(env)
        if credentials is not None:
            return credentials
    profile_name = profile or ambient_profile or "default"
    if profile_name in context.get_profiles():
        credentials = await _from_profile(profile_name, context, set())
        if credentials is not None:
            return credentials
    for provider in (_from_web_identity_env, _from_container, _from_instance_metadata):
        credentials = await provider(context)
        if credentials is not None:
            return credentials
    raise AwsCredentialsError("Could not load credentials from any providers")


def _static_profile_credentials(data: Mapping[str, str]) -> AwsCredentials | None:
    access_key_id = data.get("aws_access_key_id")
    secret_access_key = data.get("aws_secret_access_key")
    if not access_key_id or not secret_access_key:
        return None
    return AwsCredentials(
        access_key_id,
        secret_access_key,
        data.get("aws_session_token") or None,
        _parse_expiration(data.get("aws_credential_expiration")),
    )


async def _from_profile(name: str, context: _ChainContext, visited: set[str]) -> AwsCredentials | None:
    profiles = context.get_profiles()
    data = profiles.get(name)
    if data is None:
        raise AwsCredentialsError(f"Profile {name} could not be found in the shared credentials or config files")
    role_arn = data.get("role_arn")
    source_profile = data.get("source_profile")
    credential_source = data.get("credential_source")
    if role_arn and (source_profile or credential_source):
        if name in visited:
            raise AwsCredentialsError(f"Detected a cycle attempting to resolve credentials for profile {name}")
        visited.add(name)
        if data.get("mfa_serial"):
            raise AwsCredentialsError(
                f"Profile {name} requires multi-factor authentication; install botocore to resolve it"
            )
        if source_profile:
            if source_profile == name:
                source = _static_profile_credentials(data)
            else:
                source = await _from_profile(source_profile, context, visited)
            if source is None:
                raise AwsCredentialsError(f"Source profile {source_profile} of profile {name} has no credentials")
        else:
            source = await _from_credential_source(str(credential_source), name, context)
        return await _assume_role(name, data, source, context)
    static = _static_profile_credentials(data)
    if static is not None:
        return static
    if role_arn and data.get("web_identity_token_file"):
        return await _assume_role_with_web_identity(
            role_arn,
            str(_expand_path(data["web_identity_token_file"])),
            data.get("role_session_name"),
            context,
        )
    process = data.get("credential_process")
    if process:
        return await _from_process(name, process)
    if data.get("sso_session") or data.get("sso_start_url"):
        raise AwsCredentialsError(
            f"Profile {name} uses AWS IAM Identity Center (SSO); install botocore to resolve SSO credentials"
        )
    return None


async def _from_credential_source(source: str, profile: str, context: _ChainContext) -> AwsCredentials:
    credentials: AwsCredentials | None
    if source == "Environment":
        credentials = _from_env(context.env)
    elif source == "EcsContainer":
        credentials = await _from_container(context)
    elif source == "Ec2InstanceMetadata":
        credentials = await _from_instance_metadata(context)
    else:
        raise AwsCredentialsError(f"Unsupported credential_source {source!r} in profile {profile}")
    if credentials is None:
        raise AwsCredentialsError(f"credential_source {source} of profile {profile} returned no credentials")
    return credentials


def _sts_url(region: str) -> str:
    return f"https://sts.{region}.{aws_dns_suffix(region)}/"


def _client_for(url: str, context: _ChainContext) -> httpx.AsyncClient:
    """The caller's client, or the pooled client for ``url`` (honors provider-scoped proxies)."""
    if context.http_client is not None:
        return context.http_client
    from .http import get_http_client

    return get_http_client(url, {"env": context.env} if context.env else None)


def _xml_text(root: ElementTree.Element, tag: str) -> str | None:
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == tag:
            return element.text
    return None


async def _sts_request(
    params: dict[str, str], context: _ChainContext, credentials: AwsCredentials | None
) -> AwsCredentials:
    url = _sts_url(context.region)
    body = urlencode(params).encode()
    headers = {"content-type": "application/x-www-form-urlencoded; charset=utf-8", "accept": "text/xml"}
    if credentials is not None:
        headers = sign_request("POST", url, headers, body, credentials, region=context.region, service="sts")
    client = _client_for(url, context)
    try:
        response = await client.post(url, content=body, headers=headers, timeout=httpx.Timeout(10.0))
    except httpx.HTTPError as error:
        raise AwsCredentialsError(f"STS {params['Action']} request failed: {error}") from error
    text = response.text
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise AwsCredentialsError(
            f"STS {params['Action']} returned an invalid response ({response.status_code})"
        ) from error
    if response.status_code >= 400:
        code = _xml_text(root, "Code") or str(response.status_code)
        message = _xml_text(root, "Message") or "Unknown error"
        raise AwsCredentialsError(f"STS {params['Action']} failed: {code}: {message}")
    access_key_id = _xml_text(root, "AccessKeyId")
    secret_access_key = _xml_text(root, "SecretAccessKey")
    if not access_key_id or not secret_access_key:
        raise AwsCredentialsError(f"STS {params['Action']} response has no credentials")
    return AwsCredentials(
        access_key_id,
        secret_access_key,
        _xml_text(root, "SessionToken") or None,
        _parse_expiration(_xml_text(root, "Expiration")),
    )


async def _assume_role(
    profile: str, data: Mapping[str, str], source: AwsCredentials, context: _ChainContext
) -> AwsCredentials:
    role_arn = data["role_arn"]
    key = ("assume-role", profile, role_arn, source.access_key_id, context.region)
    cached = _cached(key)
    if cached is not None:
        return cached
    params = {
        "Action": "AssumeRole",
        "Version": "2011-06-15",
        "RoleArn": role_arn,
        "RoleSessionName": data.get("role_session_name") or f"aws-sdk-js-{int(time.time() * 1000)}",
    }
    if data.get("external_id"):
        params["ExternalId"] = data["external_id"]
    if data.get("duration_seconds"):
        params["DurationSeconds"] = data["duration_seconds"]
    credentials = await _sts_request(params, context, source)
    _cache[key] = credentials
    return credentials


async def _assume_role_with_web_identity(
    role_arn: str, token_file: str, session_name: str | None, context: _ChainContext
) -> AwsCredentials:
    key = ("web-identity", role_arn, token_file, context.region)
    cached = _cached(key)
    if cached is not None:
        return cached
    try:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise AwsCredentialsError(f"Could not read web identity token file {token_file}: {error}") from error
    params = {
        "Action": "AssumeRoleWithWebIdentity",
        "Version": "2011-06-15",
        "RoleArn": role_arn,
        "RoleSessionName": session_name or f"aws-sdk-js-session-{int(time.time() * 1000)}",
        "WebIdentityToken": token,
    }
    credentials = await _sts_request(params, context, None)
    _cache[key] = credentials
    return credentials


async def _from_web_identity_env(context: _ChainContext) -> AwsCredentials | None:
    token_file = get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", context.env)
    role_arn = get_provider_env_value("AWS_ROLE_ARN", context.env)
    if not token_file or not role_arn:
        return None
    return await _assume_role_with_web_identity(
        role_arn, token_file, get_provider_env_value("AWS_ROLE_SESSION_NAME", context.env), context
    )


def _check_container_url(url: str) -> None:
    parsed = httpx.URL(url)
    if parsed.scheme == "https":
        return
    host = parsed.host
    if host in _ALLOWED_CONTAINER_HOSTS:
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise AwsCredentialsError(f"Unsupported host for AWS_CONTAINER_CREDENTIALS_FULL_URI: {host}")


def _credentials_from_json(data: Any, source: str) -> AwsCredentials:
    if not isinstance(data, dict) or not data.get("AccessKeyId") or not data.get("SecretAccessKey"):
        raise AwsCredentialsError(f"{source} returned an invalid credentials response")
    return AwsCredentials(
        str(data["AccessKeyId"]),
        str(data["SecretAccessKey"]),
        data.get("Token") or data.get("SessionToken") or None,
        _parse_expiration(data.get("Expiration")),
    )


def _metadata_client(context: _ChainContext) -> tuple[httpx.AsyncClient, bool]:
    # Link-local metadata endpoints must never go through an HTTP proxy.
    if context.http_client is not None:
        return context.http_client, False
    return httpx.AsyncClient(trust_env=False), True


async def _from_container(context: _ChainContext) -> AwsCredentials | None:
    relative = get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", context.env)
    full = get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", context.env)
    if relative:
        url = _ECS_RELATIVE_HOST + relative
    elif full:
        _check_container_url(full)
        url = full
    else:
        return None
    key = ("container", url)
    cached = _cached(key)
    if cached is not None:
        return cached
    headers: dict[str, str] = {}
    token_file = get_provider_env_value("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", context.env)
    token = get_provider_env_value("AWS_CONTAINER_AUTHORIZATION_TOKEN", context.env)
    if token_file:
        try:
            headers["Authorization"] = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise AwsCredentialsError(f"Could not read container authorization token file: {error}") from error
    elif token:
        headers["Authorization"] = token
    client, owned = _metadata_client(context)
    try:
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = await client.get(url, headers=headers, timeout=_CONTAINER_TIMEOUT)
            except httpx.HTTPError as error:
                last_error = error
                continue
            if response.status_code >= 400:
                raise AwsCredentialsError(
                    f"Container credentials endpoint returned {response.status_code}: {response.text.strip()}"
                )
            try:
                data = response.json()
            except ValueError as error:
                raise AwsCredentialsError("Container credentials endpoint returned invalid JSON") from error
            credentials = _credentials_from_json(data, "Container credentials endpoint")
            _cache[key] = credentials
            return credentials
        raise AwsCredentialsError(f"Could not reach the container credentials endpoint: {last_error}")
    finally:
        if owned:
            await client.aclose()


async def _from_instance_metadata(context: _ChainContext) -> AwsCredentials | None:
    if (get_provider_env_value("AWS_EC2_METADATA_DISABLED", context.env) or "").lower() == "true":
        return None
    endpoint = (
        get_provider_env_value("AWS_EC2_METADATA_SERVICE_ENDPOINT", context.env) or "http://169.254.169.254"
    ).rstrip("/")
    key = ("imds", endpoint)
    cached = _cached(key)
    if cached is not None:
        return cached
    client, owned = _metadata_client(context)
    try:
        headers: dict[str, str] = {}
        try:
            token_response = await client.put(
                f"{endpoint}/latest/api/token",
                headers={"x-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=_METADATA_TIMEOUT,
            )
        except httpx.HTTPError:
            return None  # not running on EC2
        if token_response.status_code == 200:
            headers["x-aws-ec2-metadata-token"] = token_response.text.strip()
        elif token_response.status_code not in (403, 404, 405):
            return None
        base = f"{endpoint}/latest/meta-data/iam/security-credentials/"
        try:
            roles = await client.get(base, headers=headers, timeout=_METADATA_TIMEOUT)
            if roles.status_code != 200:
                return None
            role = roles.text.strip().splitlines()[0].strip() if roles.text.strip() else ""
            if not role:
                return None
            response = await client.get(base + quote(role, safe=""), headers=headers, timeout=_METADATA_TIMEOUT)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        credentials = _credentials_from_json(data, "Instance metadata service")
        _cache[key] = credentials
        return credentials
    finally:
        if owned:
            await client.aclose()


async def _from_process(profile: str, command: str) -> AwsCredentials:
    key = ("process", command)
    cached = _cached(key)
    if cached is not None:
        return cached
    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise AwsCredentialsError(
            f"credential_process for profile {profile} failed: {stderr.decode('utf-8', 'replace').strip()}"
        )
    try:
        data = json.loads(stdout)
    except ValueError as error:
        raise AwsCredentialsError(f"credential_process for profile {profile} returned invalid JSON") from error
    if not isinstance(data, dict) or data.get("Version") != 1:
        raise AwsCredentialsError(f"credential_process for profile {profile} did not return version 1")
    credentials = _credentials_from_json(data, f"credential_process for profile {profile}")
    _cache[key] = credentials
    return credentials


def b64decode_forgiving(data: str) -> bytes:
    """Base64 decode like the browser ``atob``: whitespace and missing padding are tolerated.

    Raises ``ValueError`` for characters outside the base64 alphabet.
    """
    text = re.sub(r"[\t\n\f\r ]+", "", data)
    if len(text) % 4 == 0 and text.endswith("="):
        text = text[:-2] if text.endswith("==") else text[:-1]
    if len(text) % 4 == 1 or re.search(r"[^A-Za-z0-9+/]", text):
        raise ValueError("Invalid base64 data")
    return base64.b64decode(text + "=" * (-len(text) % 4))
