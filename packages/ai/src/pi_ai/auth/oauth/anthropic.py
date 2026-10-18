"""Anthropic OAuth flow (Claude Pro/Max).

Browser login runs a loopback callback server; copy-code login is for headless use.
"""

from __future__ import annotations

import base64
import json
import traceback
from typing import Any

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
from .pkce import generate_pkce

__all__ = ["AnthropicOAuth", "anthropic_oauth"]

CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode()
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
# Preferred so the port can be forwarded into containers or over SSH. Anthropic accepts any loopback port,
# so login falls back to a free port when this one cannot be bound (#10571).
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
COPY_CODE_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
ANTHROPIC_BROWSER_LOGIN_METHOD = "browser"
ANTHROPIC_COPY_CODE_LOGIN_METHOD = "copy_code"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
_EXPIRY_SKEW_MS = 5 * 60 * 1000
_REQUEST_TIMEOUT_MS = 30_000


def _callback_host() -> str:
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _format_error_details(error: BaseException, depth: int = 0) -> str:
    details = [f"{type(error).__name__}: {error}"]
    code = getattr(error, "code", None)
    if code:
        details.append(f"code={code}")
    errno = getattr(error, "errno", None)
    if errno is not None:
        details.append(f"errno={errno}")
    cause = error.__cause__ if error.__cause__ is not None or error.__suppress_context__ else error.__context__
    if cause is not None and depth < 8:
        details.append(f"cause={_format_error_details(cause, depth + 1)}")
    if error.__traceback__ is not None:
        details.append(f"stack={''.join(traceback.format_tb(error.__traceback__)).rstrip()}")
    return "; ".join(details)


async def _post_json(url: str, body: dict[str, Any], signal: AbortSignal) -> str:
    response = await _common.fetch(
        url,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        body=json_stringify(body),
        signal=signal,
        timeout_ms=_REQUEST_TIMEOUT_MS,
    )
    response_body = response.text
    if not response.is_success:
        raise OAuthError(f"HTTP request failed. status={response.status_code}; url={url}; body={response_body}")
    return response_body


def _credential_from_token(data: Any) -> Credential:
    token = data if isinstance(data, dict) else {}
    credential: Credential = {"type": "oauth"}
    if token.get("refresh_token") is not None:
        credential["refresh"] = token["refresh_token"]
    if token.get("access_token") is not None:
        credential["access"] = token["access_token"]
    credential["expires"] = _common.now_ms() + _common.expires_in_ms(token.get("expires_in")) - _EXPIRY_SKEW_MS
    return credential


async def _exchange_authorization_code(
    code: str, state: str, verifier: str, redirect_uri: str, signal: AbortSignal
) -> Credential:
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
            signal,
        )
    except Exception as error:
        raise OAuthError(
            f"Token exchange request failed. url={TOKEN_URL}; redirect_uri={redirect_uri}; "
            f"response_type=authorization_code; details={_format_error_details(error)}"
        ) from error

    try:
        token_data = json.loads(response_body)
    except ValueError as error:
        raise OAuthError(
            f"Token exchange returned invalid JSON. url={TOKEN_URL}; body={response_body}; "
            f"details={_format_error_details(error)}"
        ) from error

    return _credential_from_token(token_data)


def _authorize_url(redirect_uri: str, challenge: str, verifier: str) -> str:
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    return f"{AUTHORIZE_URL}?{form_encode(params)}"


async def _login_anthropic(interaction: Interaction) -> Credential:
    verifier, challenge = generate_pkce()

    async def identity(code: str) -> str:
        return code

    async def start_callback_server(port: int) -> OAuthCallbackServer[str]:
        return await start_oauth_callback_server(
            provider_name="Anthropic",
            host=_callback_host(),
            port=port,
            path=CALLBACK_PATH,
            redirect_host="localhost",
            state=verifier,
            complete=identity,
            signal=interaction.signal,
        )

    # Without a callback server, login continues with the pasted redirect URL.
    callback: OAuthCallbackServer[str] | None = None
    for port in (CALLBACK_PORT, 0):
        try:
            callback = await start_callback_server(port)
            break
        except Exception:
            continue
    redirect_uri = callback.redirect_uri if callback is not None else REDIRECT_URI

    try:
        interaction.notify(
            AuthEvent(
                type="auth_url",
                url=_authorize_url(redirect_uri, challenge, verifier),
                instructions=(
                    "Complete login in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )

        result = await wait_for_callback_or_manual_input(
            interaction,
            callback,
            message="Complete login in your browser, or paste the authorization code / redirect URL here:",
            placeholder=redirect_uri,
        )
        code: str | None
        state = verifier
        if isinstance(result, CallbackResult):
            code = result.value
        else:
            code, parsed_state = parse_authorization_input(result.input)
            if parsed_state and parsed_state != verifier:
                raise OAuthError("OAuth state mismatch")
            state = parsed_state if parsed_state is not None else verifier

        if not code:
            raise OAuthError("Missing authorization code")
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
        return await _exchange_authorization_code(code, state, verifier, redirect_uri, interaction.signal)
    finally:
        if callback is not None:
            callback.close()


async def _login_anthropic_copy_code(interaction: Interaction) -> Credential:
    verifier, challenge = generate_pkce()
    interaction.notify(
        AuthEvent(
            type="auth_url",
            url=_authorize_url(COPY_CODE_REDIRECT_URI, challenge, verifier),
            instructions="Complete login in your browser, then copy the code Anthropic shows and paste it here.",
        )
    )

    entered = await interaction.prompt(
        AuthPrompt(
            type="manual_code",
            message="Paste the code Anthropic shows after you sign in:",
            placeholder="code#state",
            signal=interaction.signal,
        )
    )
    code, state = parse_authorization_input(entered)
    if state and state != verifier:
        raise OAuthError("OAuth state mismatch")
    if not code:
        raise OAuthError("Missing authorization code")
    interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
    return await _exchange_authorization_code(
        code, state if state is not None else verifier, verifier, COPY_CODE_REDIRECT_URI, interaction.signal
    )


async def _refresh_anthropic_token(refresh_token: str, signal: AbortSignal) -> Credential:
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh_token},
            signal,
        )
    except Exception as error:
        raise OAuthError(
            f"Anthropic token refresh request failed. url={TOKEN_URL}; details={_format_error_details(error)}"
        ) from error

    try:
        data = json.loads(response_body)
    except ValueError as error:
        raise OAuthError(
            f"Anthropic token refresh returned invalid JSON. url={TOKEN_URL}; body={response_body}; "
            f"details={_format_error_details(error)}"
        ) from error

    return _credential_from_token(data)


class AnthropicOAuth(OAuthAuth):
    name = "Anthropic (Claude Pro/Max)"
    is_subscription = True

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        method = await interaction.prompt(
            AuthPrompt(
                type="select",
                message="Select Anthropic login method:",
                options=[
                    AuthSelectOption(ANTHROPIC_BROWSER_LOGIN_METHOD, "Browser login (default)"),
                    AuthSelectOption(ANTHROPIC_COPY_CODE_LOGIN_METHOD, "Copy code login (headless)"),
                ],
            )
        )
        if method == ANTHROPIC_COPY_CODE_LOGIN_METHOD:
            return await _login_anthropic_copy_code(interaction)
        if method != ANTHROPIC_BROWSER_LOGIN_METHOD:
            raise OAuthError(f"Unknown Anthropic login method: {method}")
        return await _login_anthropic(interaction)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return await _refresh_anthropic_token(credential["refresh"], signal)

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


anthropic_oauth = AnthropicOAuth()
