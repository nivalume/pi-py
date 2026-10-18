"""Google Application Default Credentials (ADC) for Vertex AI requests.

pi's TypeScript adapter gets OAuth access tokens from ``google-auth-library`` through the
``@google/genai`` SDK. This module covers the same ground without a hard dependency:

* When the ``google-auth`` package is installed, ``google.auth.default()`` (or
  ``load_credentials_from_file`` for a provider-scoped ``GOOGLE_APPLICATION_CREDENTIALS``)
  supplies and refreshes the credentials, so every ADC source it knows works.
* Otherwise the credentials file is read directly: ``GOOGLE_APPLICATION_CREDENTIALS`` or
  gcloud's ``application_default_credentials.json``. ``authorized_user`` credentials use the
  refresh-token grant and ``service_account`` credentials the JWT bearer grant (RS256 signing
  needs ``google-auth`` or ``cryptography``).

Access tokens are cached until shortly before they expire.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ..abort import AbortSignal
from ..utils.http import ProviderHttpError, send_request
from ..utils.misc import get_pi_user_agent, get_provider_env_value

__all__ = [
    "CLOUD_PLATFORM_SCOPE",
    "DEFAULT_TOKEN_URI",
    "GoogleAuthError",
    "GoogleAccessToken",
    "well_known_credentials_path",
    "find_credentials_file",
    "get_google_access_token",
    "get_google_auth_headers",
    "clear_google_auth_cache",
]

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"
_JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
#: Refresh this long before expiry (google-auth-library's eager refresh threshold).
_EAGER_REFRESH_SECONDS = 300
_SERVICE_ACCOUNT_TOKEN_LIFETIME = 3600
_NO_CREDENTIALS_MESSAGE = (
    "Could not load the default credentials. Browse to "
    "https://cloud.google.com/docs/authentication/getting-started for more information."
)


class GoogleAuthError(Exception):
    """Credentials could not be loaded or exchanged for an access token."""


@dataclass(slots=True)
class GoogleAccessToken:
    token: str
    #: Unix time in seconds; ``None`` when the token endpoint did not report a lifetime.
    expires_at: float | None = None
    #: Project billed for quota (``x-goog-user-project``), from the credentials file.
    quota_project_id: str | None = None

    def is_fresh(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return True
        return (now if now is not None else time.time()) < self.expires_at - _EAGER_REFRESH_SECONDS

    def headers(self) -> dict[str, str]:
        headers = {"authorization": f"Bearer {self.token}"}
        if self.quota_project_id:
            headers["x-goog-user-project"] = self.quota_project_id
        return headers


_token_cache: dict[tuple[Any, ...], GoogleAccessToken] = {}
_google_auth_credentials: dict[tuple[Any, ...], Any] = {}


def clear_google_auth_cache() -> None:
    """Forget cached access tokens and loaded ``google-auth`` credentials."""
    _token_cache.clear()
    _google_auth_credentials.clear()


def well_known_credentials_path(env: Mapping[str, str] | None = None) -> str:
    """gcloud's ADC file (``gcloud auth application-default login``)."""
    config_dir = get_provider_env_value("CLOUDSDK_CONFIG", env)
    if config_dir:
        return os.path.join(os.path.expanduser(config_dir), "application_default_credentials.json")
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return os.path.join(appdata, "gcloud", "application_default_credentials.json")
    return os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")


def find_credentials_file(env: Mapping[str, str] | None = None) -> str | None:
    """``GOOGLE_APPLICATION_CREDENTIALS`` (must exist) or gcloud's ADC file when present."""
    explicit = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
    if explicit:
        path = os.path.expanduser(explicit)
        if not os.path.isfile(path):
            raise GoogleAuthError(f"The file at {explicit} does not exist, or it is not a file.")
        return path
    path = well_known_credentials_path(env)
    return path if os.path.isfile(path) else None


async def get_google_auth_headers(
    *,
    env: Mapping[str, str] | None = None,
    http_client: httpx.AsyncClient | None = None,
    signal: AbortSignal | None = None,
    timeout_ms: float | None = None,
    scopes: Sequence[str] = (CLOUD_PLATFORM_SCOPE,),
) -> dict[str, str]:
    """Request headers (``authorization``, optionally ``x-goog-user-project``) from ADC."""
    google_auth = _import_google_auth()
    if google_auth is not None:
        return await _google_auth_headers(google_auth, env, scopes)
    token = await _fetch_access_token(
        env=env, http_client=http_client, signal=signal, timeout_ms=timeout_ms, scopes=scopes
    )
    return token.headers()


async def get_google_access_token(
    *,
    env: Mapping[str, str] | None = None,
    http_client: httpx.AsyncClient | None = None,
    signal: AbortSignal | None = None,
    timeout_ms: float | None = None,
    scopes: Sequence[str] = (CLOUD_PLATFORM_SCOPE,),
) -> str:
    """An OAuth access token from ADC."""
    headers = await get_google_auth_headers(
        env=env, http_client=http_client, signal=signal, timeout_ms=timeout_ms, scopes=scopes
    )
    authorization = next((value for name, value in headers.items() if name.lower() == "authorization"), "")
    return authorization.removeprefix("Bearer ").removeprefix("bearer ")


# ---------------------------------------------------------------------------
# google-auth
# ---------------------------------------------------------------------------


def _import_google_auth() -> Any | None:
    try:
        import google.auth  # type: ignore[import-not-found]
        import google.auth.exceptions  # type: ignore[import-not-found]
        import google.auth.transport  # type: ignore[import-not-found]
    except ImportError:
        return None
    return google.auth


async def _google_auth_headers(
    google_auth: Any, env: Mapping[str, str] | None, scopes: Sequence[str]
) -> dict[str, str]:
    # The SDK passes a provider-scoped GOOGLE_APPLICATION_CREDENTIALS as `keyFilename`.
    path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
    key = (path, tuple(scopes))
    credentials = _google_auth_credentials.get(key)
    if credentials is None:
        if path:
            credentials, _ = await asyncio.to_thread(
                google_auth.load_credentials_from_file, os.path.expanduser(path), scopes=list(scopes)
            )
        else:
            credentials, _ = await asyncio.to_thread(google_auth.default, scopes=list(scopes))
        _google_auth_credentials[key] = credentials
    if not credentials.valid:
        request = _httpx_transport_request(google_auth)
        try:
            await asyncio.to_thread(credentials.refresh, request)
        except google_auth.exceptions.TransportError as error:
            token_uri = getattr(credentials, "_token_uri", None) or DEFAULT_TOKEN_URI
            raise GoogleAuthError(f"request to {token_uri} failed, reason: {error}") from error
    headers: dict[str, str] = {}
    credentials.apply(headers)
    return headers


def _httpx_transport_request(google_auth: Any) -> Any:
    """A ``google.auth.transport.Request`` backed by httpx (so ``requests`` is not needed)."""
    transport = google_auth.transport
    exceptions = google_auth.exceptions

    class _Response(transport.Response):  # type: ignore[misc, name-defined]
        def __init__(self, response: httpx.Response) -> None:
            self._response = response

        @property
        def status(self) -> int:
            return self._response.status_code

        @property
        def headers(self) -> Mapping[str, str]:
            return dict(self._response.headers)

        @property
        def data(self) -> bytes:
            return self._response.content

    class _Request(transport.Request):  # type: ignore[misc, name-defined]
        def __call__(
            self,
            url: str,
            method: str = "GET",
            body: Any = None,
            headers: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: Any,
        ) -> Any:
            try:
                response = httpx.request(
                    method, url, content=body, headers=headers, timeout=timeout or 120, follow_redirects=True
                )
            except httpx.HTTPError as error:
                raise exceptions.TransportError(error) from error
            return _Response(response)

    return _Request()


# ---------------------------------------------------------------------------
# Built-in credentials file support
# ---------------------------------------------------------------------------


async def _fetch_access_token(
    *,
    env: Mapping[str, str] | None,
    http_client: httpx.AsyncClient | None,
    signal: AbortSignal | None,
    timeout_ms: float | None,
    scopes: Sequence[str],
) -> GoogleAccessToken:
    path = find_credentials_file(env)
    if path is None:
        raise GoogleAuthError(_NO_CREDENTIALS_MESSAGE)
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size, tuple(scopes))
    cached = _token_cache.get(key)
    if cached is not None and cached.is_fresh():
        return cached

    info = _read_credentials_file(path)
    kind = info.get("type")
    now = time.time()
    if kind == "authorized_user":
        _require_fields(info, ("client_id", "client_secret", "refresh_token"))
        token_uri = info.get("token_uri") or DEFAULT_TOKEN_URI
        form = {
            "grant_type": "refresh_token",
            "client_id": info["client_id"],
            "client_secret": info["client_secret"],
            "refresh_token": info["refresh_token"],
        }
    elif kind == "service_account":
        _require_fields(info, ("client_email", "private_key"))
        token_uri = info.get("token_uri") or DEFAULT_TOKEN_URI
        form = {"grant_type": _JWT_BEARER_GRANT, "assertion": _service_account_assertion(info, scopes, token_uri, now)}
    else:
        raise GoogleAuthError(
            f'Unsupported Google credentials type "{kind}" in {path}. Install google-auth '
            "(pip install google-auth) for full Application Default Credentials support."
        )

    data = await _request_token(token_uri, form, env=env, http_client=http_client, signal=signal, timeout_ms=timeout_ms)
    access_token = data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise GoogleAuthError(f"No access token in response from {token_uri}.")
    try:
        expires_at = now + float(data["expires_in"]) if data.get("expires_in") is not None else None
    except (TypeError, ValueError):
        expires_at = None
    token = GoogleAccessToken(
        access_token, expires_at=expires_at, quota_project_id=info.get("quota_project_id") or None
    )
    _token_cache[key] = token
    return token


def _read_credentials_file(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as file:
            info = json.load(file)
    except (OSError, ValueError) as error:
        raise GoogleAuthError(f"Failed to read Google credentials from {path}: {error}") from error
    if not isinstance(info, dict):
        raise GoogleAuthError(f"Google credentials file {path} must contain a JSON object.")
    return info


def _require_fields(info: Mapping[str, Any], fields: Sequence[str]) -> None:
    for name in fields:
        if not info.get(name):
            raise GoogleAuthError(f"The incoming JSON object does not contain a {name} field")


async def _request_token(
    token_uri: str,
    form: Mapping[str, str],
    *,
    env: Mapping[str, str] | None,
    http_client: httpx.AsyncClient | None,
    signal: AbortSignal | None,
    timeout_ms: float | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"env": env, "http_client": http_client, "signal": signal, "timeout_ms": timeout_ms}
    try:
        response = await send_request(
            "POST",
            token_uri,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": get_pi_user_agent()},
            content=urllib.parse.urlencode(form),
            options={name: value for name, value in options.items() if value is not None},
            stream=False,
            error_style="raw",
        )
    except ProviderHttpError as error:
        if error.status is None:
            # node-fetch wording, which pi's retry classification recognizes.
            reason = error.__cause__ if error.__cause__ is not None else error
            raise GoogleAuthError(f"request to {token_uri} failed, reason: {reason}") from error
        raise ProviderHttpError(_token_error_message(error), status=error.status) from error
    try:
        data = response.json()
    except ValueError as error:
        raise GoogleAuthError(f"Invalid token response from {token_uri}: {response.text[:200]}") from error
    if not isinstance(data, dict):
        raise GoogleAuthError(f"Invalid token response from {token_uri}.")
    return data


def _token_error_message(error: ProviderHttpError) -> str:
    body = error.raw_body or ""
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
        description = parsed.get("error_description")
        detail = f"{parsed['error']}: {description}" if description else parsed["error"]
        return f"Google OAuth token request failed ({error.status}): {detail}"
    return f"Google OAuth token request failed ({error.status}): {body.strip() or 'no body'}"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _service_account_assertion(info: Mapping[str, Any], scopes: Sequence[str], token_uri: str, now: float) -> str:
    issued_at = int(now)
    header: dict[str, Any] = {"alg": "RS256", "typ": "JWT"}
    if info.get("private_key_id"):
        header["kid"] = info["private_key_id"]
    claims = {
        "iss": info["client_email"],
        "scope": " ".join(scopes),
        "aud": token_uri,
        "iat": issued_at,
        "exp": issued_at + _SERVICE_ACCOUNT_TOKEN_LIFETIME,
    }
    signing_input = ".".join(
        _b64url(json.dumps(part, separators=(",", ":")).encode()) for part in (header, claims)
    ).encode("ascii")
    signature = _rs256_sign(info["private_key"], signing_input)
    return f"{signing_input.decode('ascii')}.{_b64url(signature)}"


def _rs256_sign(private_key_pem: str, message: bytes) -> bytes:
    """RS256 signature with ``google-auth``'s signer or ``cryptography``."""
    try:
        from google.auth import crypt  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        return crypt.RSASigner.from_string(private_key_pem).sign(message)
    try:
        from cryptography.hazmat.primitives import hashes, serialization  # type: ignore[import-not-found]
        from cryptography.hazmat.primitives.asymmetric import padding  # type: ignore[import-not-found]
    except ImportError:
        raise GoogleAuthError(
            "Service account credentials need an RSA signer to create access tokens. "
            "Install google-auth (pip install google-auth)."
        ) from None
    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    return key.sign(message, padding.PKCS1v15(), hashes.SHA256())  # type: ignore[union-attr, call-arg]
