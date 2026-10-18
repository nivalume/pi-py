"""Radius gateway OAuth flow.

Radius is a pi-messages gateway. OAuth client APIs live on the configured
gateway; only the interactive browser authorization endpoint is discovered.
Model catalog loading is owned by the Radius provider.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from ...abort import AbortSignal
from ...providers.radius_config import normalize_radius_gateway_url
from ..types import AuthEvent, AuthPrompt, AuthSelectOption, Credential, LoginOptions, ModelAuth, OAuthAuth
from ..types import ProviderAuthInteraction as Interaction
from . import _common
from ._common import OAuthError, form_encode, parse_url
from .callback_server import start_oauth_callback_server
from .device_code import (
    DeviceCodeComplete,
    DeviceCodeFailed,
    DeviceCodePending,
    DeviceCodePollResult,
    DeviceCodeSlowDown,
    poll_oauth_device_code_flow,
)
from .pkce import generate_pkce

__all__ = ["RadiusOAuth", "RadiusOAuthResponseError", "create_radius_oauth", "normalize_radius_gateway_url"]

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1456
CALLBACK_PATH = "/oauth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
TOKEN_EXPIRY_SKEW_MS = 60_000
LOGIN_METHOD_BROWSER = "browser"
LOGIN_METHOD_DEVICE_CODE = "device-code"
OAUTH_CLIENT_ID = "pi-gateway"
OAUTH_SCOPE = "gateway offline_access"
OAUTH_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

_FORM_HEADERS = {"accept": "application/json", "content-type": "application/x-www-form-urlencoded"}


class RadiusOAuthResponseError(OAuthError):
    """An OAuth error response from the gateway (``status`` and the OAuth ``error`` code)."""

    def __init__(self, status: int, oauth_error: str | None, description: str | None, message: str) -> None:
        if oauth_error:
            detail = f"{oauth_error}: {description}" if description else oauth_error
        else:
            detail = description or str(status)
        super().__init__(f"{message}: {detail}")
        self.status = status
        self.oauth_error = oauth_error


@dataclass(slots=True)
class _DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: float
    interval: float | None


def _gateway_url(gateway: str, path: str) -> str:
    """``new URL(path, gateway)`` for an absolute path."""
    return urljoin(f"{gateway}/", path)


async def _load_discovery(gateway: str, signal: AbortSignal) -> str:
    response = await _common.fetch(
        _gateway_url(gateway, "/v1/oauth"), headers={"accept": "application/json"}, signal=signal
    )
    if not response.is_success:
        raise OAuthError(f"Could not load Radius OAuth config from {gateway}: {response.status_code} {response.text}")
    discovery = _common.parse_json(response)
    endpoint = discovery.get("authorizationEndpoint") if isinstance(discovery, dict) else None
    if not isinstance(endpoint, str):
        raise OAuthError(f"Invalid Radius OAuth config from {gateway}")
    return endpoint


def _read_oauth_response_error(response: httpx.Response, message: str) -> RadiusOAuthResponseError:
    text = response.text
    oauth_error: str | None = None
    description: str | None = None
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            description = text
        else:
            if isinstance(data, dict):
                oauth_error = data.get("error") if isinstance(data.get("error"), str) else None
                description = data.get("error_description") if isinstance(data.get("error_description"), str) else None
    return RadiusOAuthResponseError(response.status_code, oauth_error, description, message)


async def _request_oauth_token(gateway: str, fields: Mapping[str, str], signal: AbortSignal) -> Credential:
    try:
        response = await _common.fetch(
            _gateway_url(gateway, "/v1/oauth/token"),
            method="POST",
            headers=_FORM_HEADERS,
            body=form_encode(fields),
            signal=signal,
        )
    except Exception:
        if signal.aborted:
            raise _common.login_cancelled() from None
        raise

    if not response.is_success:
        raise _read_oauth_response_error(response, "Radius OAuth token request failed")

    data: Any = _common.parse_json(response)
    token = data if isinstance(data, dict) else {}
    credential: Credential = {"type": "oauth"}
    if token.get("access_token") is not None:
        credential["access"] = token["access_token"]
    if token.get("refresh_token") is not None:
        credential["refresh"] = token["refresh_token"]
    credential["expires"] = _common.now_ms() + _common.expires_in_ms(token.get("expires_in")) - TOKEN_EXPIRY_SKEW_MS
    if token.get("scope") is not None:
        credential["scope"] = token["scope"]
    return credential


async def _login_with_browser(gateway: str, authorization_endpoint: str, interaction: Interaction) -> Credential:
    verifier, challenge = generate_pkce()
    state = str(uuid.uuid4())
    authorize_url = parse_url(authorization_endpoint)
    if authorize_url is None:
        raise ValueError(f"Invalid URL: {authorization_endpoint}")
    authorize_url = authorize_url.with_search(
        form_encode(
            {
                "response_type": "code",
                "client_id": OAUTH_CLIENT_ID,
                "redirect_uri": REDIRECT_URI,
                "scope": OAUTH_SCOPE,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "handoff": "url",
                "state": state,
            }
        )
    )

    async def complete(code: str) -> Credential:
        return await _request_oauth_token(
            gateway,
            {
                "grant_type": "authorization_code",
                "client_id": OAUTH_CLIENT_ID,
                "redirect_uri": REDIRECT_URI,
                "code": code,
                "code_verifier": verifier,
            },
            interaction.signal,
        )

    callback = await start_oauth_callback_server(
        provider_name="Radius",
        host=CALLBACK_HOST,
        port=CALLBACK_PORT,
        path=CALLBACK_PATH,
        state=state,
        complete=complete,
        signal=interaction.signal,
    )
    try:
        interaction.notify(AuthEvent(type="progress", message=f"Listening for OAuth callback on {REDIRECT_URI}"))
        interaction.notify(AuthEvent(type="auth_url", url=authorize_url.href, instructions="Continue in your browser."))
        credential = await callback.wait()
        if not credential:
            raise OAuthError("OAuth callback did not complete.")
        return credential
    finally:
        callback.close()


async def _request_device_authorization(gateway: str, signal: AbortSignal) -> _DeviceAuthorization:
    try:
        response = await _common.fetch(
            _gateway_url(gateway, "/v1/oauth/device"),
            method="POST",
            headers=_FORM_HEADERS,
            body=form_encode({"client_id": OAUTH_CLIENT_ID, "scope": OAUTH_SCOPE}),
            signal=signal,
        )
    except Exception:
        if signal.aborted:
            raise _common.login_cancelled() from None
        raise

    if not response.is_success:
        raise _read_oauth_response_error(response, "Radius OAuth device authorization failed")

    data: Any = _common.parse_json(response)
    record = data if isinstance(data, dict) else {}
    if (
        not record.get("device_code")
        or not record.get("user_code")
        or not record.get("verification_uri")
        or not record.get("expires_in")
    ):
        raise OAuthError("Radius OAuth device authorization response is missing required fields")

    return _DeviceAuthorization(
        device_code=record["device_code"],
        user_code=record["user_code"],
        verification_uri=record["verification_uri"],
        expires_in=record["expires_in"],
        interval=record.get("interval"),
    )


async def _login_with_device_code(gateway: str, interaction: Interaction) -> Credential:
    device = await _request_device_authorization(gateway, interaction.signal)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=device.user_code,
            verification_uri=device.verification_uri,
            interval_seconds=device.interval,
            expires_in_seconds=device.expires_in,
        )
    )

    async def poll() -> DeviceCodePollResult:
        try:
            credential = await _request_oauth_token(
                gateway,
                {
                    "grant_type": OAUTH_DEVICE_CODE_GRANT_TYPE,
                    "client_id": OAUTH_CLIENT_ID,
                    "device_code": device.device_code,
                },
                interaction.signal,
            )
            return DeviceCodeComplete(credential)
        except RadiusOAuthResponseError as error:
            if error.oauth_error == "authorization_pending":
                return DeviceCodePending()
            if error.oauth_error == "slow_down":
                return DeviceCodeSlowDown()
            if error.oauth_error == "expired_token":
                return DeviceCodeFailed("Device authorization expired.")
            if error.oauth_error == "access_denied":
                return DeviceCodeFailed("Device authorization was denied.")
            raise

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=interaction.signal,
        interval_seconds=device.interval,
        expires_in_seconds=device.expires_in,
    )


class RadiusOAuth(OAuthAuth):
    """OAuth against one Radius gateway."""

    def __init__(self, *, name: str, gateway: str) -> None:
        self.name = name
        self.gateway = normalize_radius_gateway_url(gateway)

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        login_method = await interaction.prompt(
            AuthPrompt(
                type="select",
                message=f"Sign in to {self.name}:",
                options=[
                    AuthSelectOption(LOGIN_METHOD_BROWSER, "Sign in with browser (recommended)"),
                    AuthSelectOption(
                        LOGIN_METHOD_DEVICE_CODE, "Sign in with device code (when signing in from another device)"
                    ),
                ],
            )
        )
        if login_method == LOGIN_METHOD_DEVICE_CODE:
            return await _login_with_device_code(self.gateway, interaction)
        if login_method == LOGIN_METHOD_BROWSER:
            authorization_endpoint = await _load_discovery(self.gateway, interaction.signal)
            return await _login_with_browser(self.gateway, authorization_endpoint, interaction)
        raise OAuthError(f"Unknown {self.name} sign-in method: {login_method}")

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return await _request_oauth_token(
            self.gateway,
            {"grant_type": "refresh_token", "client_id": OAUTH_CLIENT_ID, "refresh_token": credential["refresh"]},
            signal,
        )

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


def create_radius_oauth(*, name: str, gateway: str) -> OAuthAuth:
    """OAuth for the Radius gateway at ``gateway`` (scheme optional), shown as ``name``."""
    return RadiusOAuth(name=name, gateway=gateway)
