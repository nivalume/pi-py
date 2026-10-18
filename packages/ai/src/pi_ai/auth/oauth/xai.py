"""xAI OAuth device-code flow."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...abort import AbortSignal
from ..types import AuthEvent, Credential, LoginOptions, ModelAuth, OAuthAuth
from ..types import ProviderAuthInteraction as Interaction
from . import _common
from ._common import OAuthError, form_encode, parse_url
from .device_code import (
    DeviceCodeComplete,
    DeviceCodeFailed,
    DeviceCodePending,
    DeviceCodePollResult,
    DeviceCodeSlowDown,
    poll_oauth_device_code_flow,
)

__all__ = ["XaiOAuth", "xai_oauth"]

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
# Refresh slightly before the reported expiry to avoid using a token that dies mid-request.
REFRESH_SKEW_MS = 5 * 60 * 1000
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


@dataclass(slots=True)
class _OAuthHttpResponse:
    ok: bool
    status: int
    body: dict[str, Any]


@dataclass(slots=True)
class _XaiDeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    interval_seconds: float | None
    expires_in_seconds: float


def _required_string(body: Mapping[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise OAuthError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(body: Mapping[str, Any], field: str) -> float:
    value = body.get(field)
    if not _common.is_number(value) or not math.isfinite(value) or value <= 0:
        raise OAuthError(f"Invalid xAI OAuth response field: {field}")
    return value


def _validate_verification_uri(raw: str) -> str:
    # The verification URI is opened in the user's browser; force it to be an https URL
    # so a malicious response cannot make the launcher open something else.
    url = parse_url(raw)
    if url is None or url.protocol != "https:":
        raise OAuthError("Untrusted verification URI in xAI OAuth response")
    return url.href


async def _post_form(url: str, fields: Mapping[str, str], signal: AbortSignal) -> _OAuthHttpResponse:
    try:
        response = await _common.fetch(
            url,
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            body=form_encode(fields),
            signal=signal,
        )
    except Exception:
        if signal.aborted:
            raise _common.login_cancelled() from None
        raise

    try:
        parsed = _common.parse_json(response)
    except ValueError:
        if signal.aborted:
            raise _common.login_cancelled() from None
        raise OAuthError(f"xAI OAuth returned invalid JSON (HTTP {response.status_code})") from None
    return _OAuthHttpResponse(response.is_success, response.status_code, parsed if isinstance(parsed, dict) else {})


def _request_failure(action: str, response: _OAuthHttpResponse) -> OAuthError:
    error = response.body.get("error")
    description = response.body.get("error_description")
    detail = ": ".join(part for part in (error, description) if isinstance(part, str) and part)
    return OAuthError(f"xAI OAuth {action} failed (HTTP {response.status}){f': {detail}' if detail else ''}")


def _parse_device_code(body: Mapping[str, Any]) -> _XaiDeviceCode:
    # RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's
    # default instead of failing on non-positive or malformed values.
    interval = body.get("interval")
    interval_seconds = interval if _common.is_number(interval) and math.isfinite(interval) and interval > 0 else None
    complete = body.get("verification_uri_complete")
    verification_uri_complete = _validate_verification_uri(complete) if isinstance(complete, str) and complete else None
    return _XaiDeviceCode(
        device_code=_required_string(body, "device_code"),
        user_code=_required_string(body, "user_code"),
        verification_uri=_validate_verification_uri(_required_string(body, "verification_uri")),
        verification_uri_complete=verification_uri_complete,
        interval_seconds=interval_seconds,
        expires_in_seconds=_positive_number(body, "expires_in"),
    )


def _credentials_from_token_response(body: Mapping[str, Any], previous_refresh_token: str | None = None) -> Credential:
    access = _required_string(body, "access_token")
    # xAI may omit refresh_token on refresh when the token is not rotated.
    if "refresh_token" not in body and previous_refresh_token:
        refresh = previous_refresh_token
    else:
        refresh = _required_string(body, "refresh_token")
    expires_in_seconds = (
        DEFAULT_TOKEN_LIFETIME_SECONDS if "expires_in" not in body else _positive_number(body, "expires_in")
    )
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires": _common.now_ms() + _common.expires_in_ms(expires_in_seconds) - REFRESH_SKEW_MS,
    }


async def _request_device_code(signal: AbortSignal) -> _XaiDeviceCode:
    response = await _post_form(
        XAI_DEVICE_CODE_URL, {"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE, "referrer": "pi"}, signal
    )
    if not response.ok:
        raise _request_failure("device authorization", response)
    return _parse_device_code(response.body)


async def _poll_for_tokens(device: _XaiDeviceCode, signal: AbortSignal) -> Credential:
    async def poll() -> DeviceCodePollResult:
        response = await _post_form(
            XAI_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_CLIENT_ID,
                "device_code": device.device_code,
            },
            signal,
        )
        if response.ok:
            return DeviceCodeComplete(_credentials_from_token_response(response.body))

        error = response.body.get("error")
        if error == "authorization_pending":
            return DeviceCodePending()
        if error == "slow_down":
            interval = response.body.get("interval")
            return DeviceCodeSlowDown(interval if _common.is_number(interval) else None)
        if error in ("access_denied", "authorization_denied"):
            return DeviceCodeFailed("xAI device authorization was denied")
        if error == "expired_token":
            return DeviceCodeFailed("xAI device code expired")
        return DeviceCodeFailed(str(_request_failure("device token polling", response)))

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds,
        wait_before_first_poll=True,
    )


async def _refresh_xai_token(refresh_token: str, signal: AbortSignal) -> Credential:
    response = await _post_form(
        XAI_TOKEN_URL,
        {"grant_type": "refresh_token", "client_id": XAI_CLIENT_ID, "refresh_token": refresh_token},
        signal,
    )
    if not response.ok:
        raise _request_failure("token refresh", response)
    return _credentials_from_token_response(response.body, refresh_token)


class XaiOAuth(OAuthAuth):
    name = "xAI (Grok/X subscription)"
    is_subscription = True
    login_label = "Sign in with SuperGrok or X Premium"

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        device = await _request_device_code(interaction.signal)
        interaction.notify(
            AuthEvent(
                type="device_code",
                user_code=device.user_code,
                verification_uri=device.verification_uri_complete or device.verification_uri,
                interval_seconds=device.interval_seconds,
                expires_in_seconds=device.expires_in_seconds,
            )
        )
        return await _poll_for_tokens(device, interaction.signal)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return await _refresh_xai_token(credential["refresh"], signal)

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


xai_oauth = XaiOAuth()
