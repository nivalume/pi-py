"""OpenAI Codex (ChatGPT OAuth) flow: browser login with a loopback callback, or device code."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from typing import Any

import httpx

from ...abort import AbortSignal
from ...utils.misc import get_provider_env_value
from ..types import AuthEvent, AuthPrompt, AuthSelectOption, Credential, LoginOptions, ModelAuth, OAuthAuth
from ..types import ProviderAuthInteraction as Interaction
from . import _common
from ._common import OAuthError, form_encode, json_stringify, parse_authorization_input
from .callback_server import (
    CallbackResult,
    OAuthCallbackServer,
    start_oauth_callback_server,
    wait_for_callback_or_manual_input,
)
from .device_code import (
    DeviceCodeComplete,
    DeviceCodeFailed,
    DeviceCodePending,
    DeviceCodePollResult,
    DeviceCodeSlowDown,
    poll_oauth_device_code_flow,
)
from .pkce import generate_pkce

__all__ = ["OpenAICodexOAuth", "openai_codex_oauth"]

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
OPENAI_CODEX_BROWSER_LOGIN_METHOD = "browser"
OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD = "device_code"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


@dataclass(slots=True)
class _OAuthToken:
    access: str
    refresh: str
    expires: int


@dataclass(slots=True)
class _DeviceAuthInfo:
    device_auth_id: str
    user_code: str
    interval_seconds: float


@dataclass(slots=True)
class _DeviceTokenSuccess:
    authorization_code: str
    code_verifier: str


def _callback_host() -> str:
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _decode_jwt(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1].replace("-", "+").replace("_", "/")
        decoded = base64.b64decode(payload + "=" * (-len(payload) % 4))
        value = json.loads(decoded)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


async def _fetch_with_login_cancellation(url: str, signal: AbortSignal, **kwargs: Any) -> httpx.Response:
    try:
        return await _common.fetch(url, signal=signal, **kwargs)
    except Exception:
        if signal.aborted:
            raise _common.login_cancelled() from None
        raise


async def _read_token_response(response: httpx.Response, operation: str) -> _OAuthToken:
    if not response.is_success:
        text = response.text
        raise OAuthError(
            f"OpenAI Codex token {operation} failed ({response.status_code}): {text or _common.status_text(response)}"
        )

    data = _common.parse_json(response)
    record = data if isinstance(data, dict) else {}
    if (
        not record.get("access_token")
        or not record.get("refresh_token")
        or not _common.is_number(record.get("expires_in"))
    ):
        raise OAuthError(f"OpenAI Codex token {operation} response missing fields: {json_stringify(data)}")

    return _OAuthToken(
        access=record["access_token"],
        refresh=record["refresh_token"],
        expires=_common.now_ms() + _common.expires_in_ms(record["expires_in"]),
    )


async def _exchange_authorization_code(code: str, verifier: str, redirect_uri: str, signal: AbortSignal) -> _OAuthToken:
    response = await _fetch_with_login_cancellation(
        TOKEN_URL,
        signal,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=form_encode(
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            }
        ),
    )
    return await _read_token_response(response, "exchange")


async def _refresh_access_token(refresh_token: str, signal: AbortSignal) -> _OAuthToken:
    try:
        response = await _common.fetch(
            TOKEN_URL,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=form_encode({"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CLIENT_ID}),
            signal=signal,
        )
    except Exception as error:
        raise OAuthError(f"OpenAI Codex token refresh error: {error}") from error
    return await _read_token_response(response, "refresh")


def _js_number(value: str) -> float:
    """``Number(string)``: blank is ``0``, unparsable is ``NaN``."""
    text = value.strip()
    if not text:
        return 0.0
    if "_" in text:
        return math.nan
    try:
        return float(int(text, 0)) if text.lower().startswith(("0x", "0o", "0b")) else float(text)
    except ValueError:
        return math.nan


def _plain_number(value: float) -> float:
    return int(value) if isinstance(value, float) and value.is_integer() else value


async def _start_device_auth(signal: AbortSignal) -> _DeviceAuthInfo:
    response = await _fetch_with_login_cancellation(
        DEVICE_USER_CODE_URL,
        signal,
        method="POST",
        headers={"Content-Type": "application/json"},
        body=json_stringify({"client_id": CLIENT_ID}),
    )

    if not response.is_success:
        if response.status_code == 404:
            raise OAuthError(
                "OpenAI Codex device code login is not enabled for this server. "
                "Use browser login or verify the server URL."
            )
        body = response.text
        raise OAuthError(
            f"OpenAI Codex device code request failed with status {response.status_code}{f': {body}' if body else ''}"
        )

    data = _common.parse_json(response)
    record = data if isinstance(data, dict) else {}
    interval = record.get("interval")
    interval_seconds = _js_number(interval) if isinstance(interval, str) else interval
    if (
        not record.get("device_auth_id")
        or not record.get("user_code")
        or not _common.is_number(interval_seconds)
        or not math.isfinite(interval_seconds)
        or interval_seconds < 0
    ):
        raise OAuthError(f"Invalid OpenAI Codex device code response: {json_stringify(data)}")

    return _DeviceAuthInfo(record["device_auth_id"], record["user_code"], _plain_number(interval_seconds))


async def _poll_device_auth(device: _DeviceAuthInfo, signal: AbortSignal) -> _DeviceTokenSuccess:
    async def poll() -> DeviceCodePollResult:
        response = await _fetch_with_login_cancellation(
            DEVICE_TOKEN_URL,
            signal,
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json_stringify({"device_auth_id": device.device_auth_id, "user_code": device.user_code}),
        )

        if response.is_success:
            data = _common.parse_json(response)
            record = data if isinstance(data, dict) else {}
            if not record.get("authorization_code") or not record.get("code_verifier"):
                return DeviceCodeFailed(f"Invalid OpenAI Codex device auth token response: {json_stringify(data)}")
            return DeviceCodeComplete(_DeviceTokenSuccess(record["authorization_code"], record["code_verifier"]))

        if response.status_code in (403, 404):
            return DeviceCodePending()

        body = response.text
        error_code: Any = None
        try:
            parsed = json.loads(body)
            error = parsed.get("error") if isinstance(parsed, dict) else None
            error_code = error.get("code") if isinstance(error, dict) else error
        except ValueError:
            pass

        if error_code == "deviceauth_authorization_pending":
            return DeviceCodePending()
        if error_code == "slow_down":
            return DeviceCodeSlowDown()

        return DeviceCodeFailed(
            f"OpenAI Codex device auth failed with status {response.status_code}{f': {body}' if body else ''}"
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
    )


def _create_authorization_flow(originator: str = "pi") -> tuple[str, str, str]:
    verifier, challenge = generate_pkce()
    state = _common.random_hex(16)
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    return verifier, state, f"{AUTHORIZE_URL}?{form_encode(params)}"


def get_account_id(access_token: str) -> str | None:
    """The ChatGPT account id claim of an OpenAI access token."""
    payload = _decode_jwt(access_token)
    auth = payload.get(JWT_CLAIM_PATH) if payload else None
    account_id = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    return account_id if isinstance(account_id, str) and account_id else None


def _credentials_from_token(token: _OAuthToken) -> Credential:
    account_id = get_account_id(token.access)
    if not account_id:
        raise OAuthError("Failed to extract accountId from token")
    return {
        "type": "oauth",
        "access": token.access,
        "refresh": token.refresh,
        "expires": token.expires,
        "accountId": account_id,
    }


async def _exchange_for_credentials(code: str, verifier: str, redirect_uri: str, signal: AbortSignal) -> Credential:
    return _credentials_from_token(await _exchange_authorization_code(code, verifier, redirect_uri, signal))


async def _login_device_code(interaction: Interaction) -> Credential:
    device = await _start_device_auth(interaction.signal)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=device.user_code,
            verification_uri=DEVICE_VERIFICATION_URI,
            interval_seconds=device.interval_seconds,
            expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
        )
    )
    code = await _poll_device_auth(device, interaction.signal)
    return await _exchange_for_credentials(
        code.authorization_code, code.code_verifier, DEVICE_REDIRECT_URI, interaction.signal
    )


async def _login_browser(interaction: Interaction, options: LoginOptions | None) -> Credential:
    agent_name = options.agent_name if options is not None else None
    verifier, state, url = _create_authorization_flow(agent_name if agent_name is not None else "pi")

    async def identity(code: str) -> str:
        return code

    # Port 1455 is shared with the Codex CLI; when it is taken, fall back to the pasted redirect URL.
    callback: OAuthCallbackServer[str] | None
    try:
        callback = await start_oauth_callback_server(
            provider_name="OpenAI",
            host=_callback_host(),
            port=1455,
            path="/auth/callback",
            state=state,
            complete=identity,
            signal=interaction.signal,
        )
    except Exception:
        callback = None

    interaction.notify(
        AuthEvent(type="auth_url", url=url, instructions="A browser window should open. Complete login to finish.")
    )

    try:
        result = await wait_for_callback_or_manual_input(
            interaction,
            callback,
            message="Complete login in your browser, or paste the authorization code / redirect URL here:",
            placeholder=REDIRECT_URI,
        )
        code: str | None
        if isinstance(result, CallbackResult):
            code = result.value
        else:
            code, parsed_state = parse_authorization_input(result.input)
            if parsed_state and parsed_state != state:
                raise OAuthError("State mismatch")

        if not code:
            raise OAuthError("Missing authorization code")
        return await _exchange_for_credentials(code, verifier, REDIRECT_URI, interaction.signal)
    finally:
        if callback is not None:
            callback.close()


class OpenAICodexOAuth(OAuthAuth):
    name = "OpenAI (ChatGPT Plus/Pro)"
    is_subscription = True

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        method = await interaction.prompt(
            AuthPrompt(
                type="select",
                message="Select OpenAI Codex login method:",
                options=[
                    AuthSelectOption(OPENAI_CODEX_BROWSER_LOGIN_METHOD, "Browser login (default)"),
                    AuthSelectOption(OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD, "Device code login (headless)"),
                ],
            )
        )
        if method == OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD:
            return await _login_device_code(interaction)
        if method != OPENAI_CODEX_BROWSER_LOGIN_METHOD:
            raise OAuthError(f"Unknown OpenAI Codex login method: {method}")
        return await _login_browser(interaction, options)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return _credentials_from_token(await _refresh_access_token(credential["refresh"], signal))

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


openai_codex_oauth = OpenAICodexOAuth()
