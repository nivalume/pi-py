"""Kimi Code (subscription) OAuth flow.

RFC 8628 device authorization grant against https://auth.kimi.com with JSON
responses. The access token authenticates requests to https://api.kimi.com/coding
as an ``Authorization: Bearer`` header.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import httpx

from ...abort import AbortError, AbortSignal
from ...utils.misc import get_provider_env_value
from ..types import AuthEvent, Credential, LoginOptions, ModelAuth, OAuthAuth
from ..types import ProviderAuthInteraction as Interaction
from . import _common
from ._common import OAuthError, form_encode, json_stringify, parse_url
from .device_code import (
    DeviceCodeComplete,
    DeviceCodeFailed,
    DeviceCodePending,
    DeviceCodePollResult,
    DeviceCodeSlowDown,
    poll_oauth_device_code_flow,
)

__all__ = ["KimiCodingOAuth", "kimi_coding_oauth"]

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_MS = 30 * 1000
REFRESH_MAX_RETRIES = 3

_FORM_HEADERS = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}


@dataclass(slots=True)
class _DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    interval_seconds: float
    expires_in_seconds: float


@dataclass(slots=True)
class _TokenResponse:
    access: str
    refresh: str
    expires: int


def _oauth_host() -> str:
    override = get_provider_env_value("KIMI_CODE_OAUTH_HOST") or get_provider_env_value("KIMI_OAUTH_HOST")
    return re.sub(r"/+$", "", override or DEFAULT_OAUTH_HOST)


def _read_json(response: httpx.Response) -> Any:
    """The JSON body when it is an object or array, otherwise ``None``."""
    try:
        value = _common.parse_json(response)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


def _field(data: Any, key: str) -> Any:
    return data.get(key) if isinstance(data, dict) else None


def _trusted_http_url(value: Any) -> str | None:
    """The verification URI is opened in the user's browser; only http(s) URLs are trusted."""
    if not isinstance(value, str) or not value:
        return None
    url = parse_url(value)
    if url is None or url.protocol not in ("https:", "http:"):
        return None
    return url.href


def _positive(value: Any) -> bool:
    return _common.is_number(value) and math.isfinite(value) and value > 0


async def _start_device_authorization(oauth_host: str, signal: AbortSignal) -> _DeviceAuthorization:
    response = await _common.fetch(
        f"{oauth_host}/api/oauth/device_authorization",
        method="POST",
        headers=_FORM_HEADERS,
        body=form_encode({"client_id": CLIENT_ID}),
        signal=signal,
        timeout_ms=REQUEST_TIMEOUT_MS,
    )

    if not response.is_success:
        text = response.text
        raise OAuthError(
            f"Kimi Code device authorization failed with status {response.status_code}{f': {text}' if text else ''}"
        )

    data = _read_json(response)
    device_code = _field(data, "device_code")
    user_code = _field(data, "user_code")
    verification_uri = _field(data, "verification_uri")
    verification_uri_complete = _field(data, "verification_uri_complete")
    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or not isinstance(verification_uri_complete, str)
        or not _trusted_http_url(verification_uri_complete)
        or not _trusted_http_url(verification_uri)
    ):
        raise OAuthError(f"Invalid Kimi Code device authorization response: {json_stringify(data)}")

    interval = _field(data, "interval")
    expires_in = _field(data, "expires_in")
    return _DeviceAuthorization(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=verification_uri_complete,
        interval_seconds=interval if _positive(interval) else DEFAULT_POLL_INTERVAL_SECONDS,
        expires_in_seconds=expires_in if _positive(expires_in) else DEVICE_CODE_TIMEOUT_SECONDS,
    )


def _parse_token_response(data: Any, operation: str) -> _TokenResponse:
    access_token = _field(data, "access_token")
    refresh_token = _field(data, "refresh_token")
    expires_in = _field(data, "expires_in")
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or not _positive(expires_in)
    ):
        raise OAuthError(f"Kimi Code token {operation} response missing fields: {json_stringify(data)}")
    return _TokenResponse(access_token, refresh_token, _common.now_ms() + _common.expires_in_ms(expires_in))


async def _poll_for_token(oauth_host: str, device: _DeviceAuthorization, signal: AbortSignal) -> _TokenResponse:
    async def poll() -> DeviceCodePollResult:
        response = await _common.fetch(
            f"{oauth_host}/api/oauth/token",
            method="POST",
            headers=_FORM_HEADERS,
            body=form_encode(
                {
                    "client_id": CLIENT_ID,
                    "device_code": device.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                }
            ),
            signal=signal,
            timeout_ms=REQUEST_TIMEOUT_MS,
        )

        if response.status_code >= 500:
            text = response.text
            return DeviceCodeFailed(
                f"Kimi Code device token request failed with status {response.status_code}{f': {text}' if text else ''}"
            )

        data = _read_json(response)
        if response.is_success and isinstance(_field(data, "access_token"), str):
            try:
                return DeviceCodeComplete(_parse_token_response(data, "poll"))
            except Exception as error:
                return DeviceCodeFailed(str(error))

        error = _field(data, "error")
        error_description = _field(data, "error_description")
        description = f": {error_description}" if isinstance(error_description, str) else ""
        if error == "authorization_pending":
            return DeviceCodePending()
        if error == "slow_down":
            interval = _field(data, "interval")
            return DeviceCodeSlowDown(interval if _common.is_number(interval) and interval > 0 else None)
        if error == "expired_token":
            return DeviceCodeFailed("Kimi Code device authorization expired. Please restart login.")
        if error == "access_denied":
            return DeviceCodeFailed("Kimi Code login was denied.")
        return DeviceCodeFailed(
            f"Kimi Code device token request failed (status {response.status_code})"
            f"{f': {error}{description}' if isinstance(error, str) else ''}"
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds,
        wait_before_first_poll=True,
    )


def _is_retryable_refresh_failure(response: httpx.Response) -> bool:
    return response.status_code == 429 or response.status_code >= 500


async def _refresh_token(oauth_host: str, refresh_token: str, signal: AbortSignal) -> _TokenResponse:
    last_error: BaseException | None = None
    for attempt in range(REFRESH_MAX_RETRIES + 1):
        if attempt > 0:
            await _common.sleep(2 ** (attempt - 1), signal)
        if signal.aborted:
            raise AbortError("Kimi Code token refresh aborted")

        try:
            response = await _common.fetch(
                f"{oauth_host}/api/oauth/token",
                method="POST",
                headers=_FORM_HEADERS,
                body=form_encode(
                    {"client_id": CLIENT_ID, "grant_type": "refresh_token", "refresh_token": refresh_token}
                ),
                signal=signal,
                timeout_ms=REQUEST_TIMEOUT_MS,
            )
        except Exception as error:
            last_error = error
            continue

        data = _read_json(response)
        if response.is_success:
            return _parse_token_response(data, "refresh")

        # Unauthorized: the stored credential is dead; Models clears it and prompts re-login.
        if response.status_code in (401, 403) or _field(data, "error") == "invalid_grant":
            error_description = _field(data, "error_description")
            description = f": {error_description}" if isinstance(error_description, str) else ""
            raise OAuthError(f"Kimi Code token refresh unauthorized (status {response.status_code}){description}")

        if _is_retryable_refresh_failure(response) and attempt < REFRESH_MAX_RETRIES:
            last_error = OAuthError(f"Kimi Code token refresh failed with status {response.status_code}")
            continue

        raise OAuthError(f"Kimi Code token refresh failed with status {response.status_code}: {json_stringify(data)}")

    raise last_error if last_error is not None else OAuthError("Kimi Code token refresh failed")


def _credential(token: _TokenResponse) -> Credential:
    return {"type": "oauth", "access": token.access, "refresh": token.refresh, "expires": token.expires}


class KimiCodingOAuth(OAuthAuth):
    name = "Kimi Code (subscription)"
    is_subscription = True
    login_label = "Sign in with Kimi Code"

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        oauth_host = _oauth_host()
        device = await _start_device_authorization(oauth_host, interaction.signal)
        interaction.notify(
            AuthEvent(
                type="device_code",
                user_code=device.user_code,
                verification_uri=device.verification_uri_complete,
                interval_seconds=device.interval_seconds,
                expires_in_seconds=device.expires_in_seconds,
            )
        )
        return _credential(await _poll_for_token(oauth_host, device, interaction.signal))

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return _credential(await _refresh_token(_oauth_host(), credential["refresh"], signal))

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(headers={"Authorization": f"Bearer {credential['access']}"})


kimi_coding_oauth = KimiCodingOAuth()
