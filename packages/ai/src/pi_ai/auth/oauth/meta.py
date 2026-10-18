"""Meta Model API OAuth flow.

RFC 8628 device authorization grant against https://auth.meta.com (JSON
responses). Meta splits identity from API access: the resulting identity token
is not accepted for inference, so it is exchanged for a Model API key via the
Muse Code key-mint endpoint (minted keys live about a day). The identity token
is stored as ``refresh`` and the minted key as ``access``, so the standard OAuth
scheduler re-mints the key when it expires with no bespoke renewal machinery.
The identity token itself is not renewable (auth.meta.com answers
``grant_type=refresh_token`` with 404 and issues no refresh_token), so a 401/403
from mint means the session is dead and the user must sign in again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx

from ...abort import AbortSignal
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

__all__ = ["MetaOAuth", "meta_oauth"]

# Muse Code CLI client id.
CLIENT_ID = "1031625952748946"
AUTH_HOST = "https://auth.meta.com"
DEVICE_AUTHORIZATION_URL = f"{AUTH_HOST}/oidc/device/authorization/"
DEVICE_TOKEN_URL = f"{AUTH_HOST}/oidc/device/token/"
API_KEY_MINT_URL = "https://api.meta.ai/muse-code/key"
API_KEY_LIFETIME_MS = 24 * 60 * 60 * 1000
REQUEST_TIMEOUT_MS = 30 * 1000

_FORM_HEADERS = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}


@dataclass(slots=True)
class _DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    interval_seconds: float | None
    expires_in_seconds: float | None


def _read_json(response: httpx.Response) -> Any:
    try:
        value = _common.parse_json(response)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


def _field(data: Any, key: str) -> Any:
    return data.get(key) if isinstance(data, dict) else None


def _error_detail(data: Any) -> str:
    for key in ("error_description", "detail", "message", "error"):
        value = _field(data, key)
        if isinstance(value, str) and value.strip():
            return f": {value.strip()}"
    return ""


def _trusted_http_url(value: Any) -> str | None:
    """The verification URI is opened in the user's browser; only http(s) URLs are trusted."""
    if not isinstance(value, str) or not value:
        return None
    url = parse_url(value)
    if url is None or url.protocol not in ("https:", "http:"):
        return None
    return url.href


def _positive_number(value: Any) -> float | None:
    return value if _common.is_number(value) and math.isfinite(value) and value > 0 else None


async def _start_device_authorization(signal: AbortSignal) -> _DeviceAuthorization:
    response = await _common.fetch(
        DEVICE_AUTHORIZATION_URL,
        method="POST",
        headers=_FORM_HEADERS,
        body=form_encode({"client_id": CLIENT_ID}),
        signal=signal,
        timeout_ms=REQUEST_TIMEOUT_MS,
    )
    data = _read_json(response)
    if not response.is_success:
        raise OAuthError(f"Meta device authorization failed with status {response.status_code}{_error_detail(data)}")
    device_code = _field(data, "device_code")
    user_code = _field(data, "user_code")
    verification_uri = _trusted_http_url(_field(data, "verification_uri_complete")) or _trusted_http_url(
        _field(data, "verification_uri")
    )
    if not isinstance(device_code, str) or not device_code or not isinstance(user_code, str) or not user_code:
        raise OAuthError(f"Invalid Meta device authorization response: {json_stringify(data)}")
    if not verification_uri:
        raise OAuthError(f"Invalid Meta device authorization response: {json_stringify(data)}")
    return _DeviceAuthorization(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        interval_seconds=_positive_number(_field(data, "interval")),
        expires_in_seconds=_positive_number(_field(data, "expires_in")),
    )


async def _poll_for_identity_token(device: _DeviceAuthorization, signal: AbortSignal) -> str:
    async def poll() -> DeviceCodePollResult:
        response = await _common.fetch(
            DEVICE_TOKEN_URL,
            method="POST",
            headers=_FORM_HEADERS,
            body=form_encode(
                {
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device.device_code,
                    "client_id": CLIENT_ID,
                }
            ),
            signal=signal,
            timeout_ms=REQUEST_TIMEOUT_MS,
        )
        data = _read_json(response)
        access_token = _field(data, "access_token")
        if response.is_success and isinstance(access_token, str) and access_token:
            return DeviceCodeComplete(access_token)
        error = _field(data, "error")
        if error == "authorization_pending":
            return DeviceCodePending()
        if error == "slow_down":
            return DeviceCodeSlowDown(_positive_number(_field(data, "interval")))
        if error == "access_denied":
            return DeviceCodeFailed("Meta login was denied.")
        if error == "expired_token":
            return DeviceCodeFailed("Meta device authorization expired. Please restart login.")
        return DeviceCodeFailed(
            f"Meta device token request failed with status {response.status_code}{_error_detail(data)}"
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds,
        wait_before_first_poll=True,
    )


async def _mint_api_key(identity_token: str, signal: AbortSignal) -> Credential:
    """Exchange an identity token for a Model API key. Keys are valid for about a day."""
    response = await _common.fetch(
        API_KEY_MINT_URL,
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {identity_token}",
            "Content-Type": "application/json",
            "x-api-version": "1.0.0",
        },
        body="{}",
        signal=signal,
        timeout_ms=REQUEST_TIMEOUT_MS,
    )
    data = _read_json(response)
    if response.status_code in (401, 403):
        # Identity token is not renewable (see module docstring); only a fresh device flow helps.
        raise OAuthError(
            f"Meta session expired (status {response.status_code}). Run `/login meta` to sign in again."
            f"{_error_detail(data)}"
        )
    if not response.is_success:
        raise OAuthError(f"Meta API key mint failed with status {response.status_code}{_error_detail(data)}")
    api_key = _field(data, "api_key")
    if not isinstance(api_key, str) or not api_key:
        action_url = _trusted_http_url(_field(data, "action_url"))
        raise OAuthError(f"Meta did not issue an API key.{f' Complete setup at {action_url}' if action_url else ''}")
    return {
        "type": "oauth",
        "refresh": identity_token,
        "access": api_key,
        "expires": _common.now_ms() + API_KEY_LIFETIME_MS,
    }


async def _login_meta(interaction: Interaction) -> Credential:
    try:
        device = await _start_device_authorization(interaction.signal)
        interaction.notify(
            AuthEvent(
                type="device_code",
                user_code=device.user_code,
                verification_uri=device.verification_uri,
                interval_seconds=device.interval_seconds,
                expires_in_seconds=device.expires_in_seconds,
            )
        )
        identity_token = await _poll_for_identity_token(device, interaction.signal)
        interaction.notify(AuthEvent(type="progress", message="Enabling Meta Model API access..."))
        return await _mint_api_key(identity_token, interaction.signal)
    except Exception as error:
        # An in-flight request raises an AbortError on abort; the login UI matches on this message.
        if interaction.signal.aborted:
            raise _common.login_cancelled() from error
        raise


class MetaOAuth(OAuthAuth):
    name = "Meta (Muse subscription)"
    is_subscription = True
    login_label = "Sign in with Meta"

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        return await _login_meta(interaction)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return await _mint_api_key(credential["refresh"], signal)

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


meta_oauth = MetaOAuth()
