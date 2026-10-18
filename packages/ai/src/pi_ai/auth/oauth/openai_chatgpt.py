"""OpenAI Responses API token sharing through Sign in with ChatGPT.

This public-client flow uses no client secret and sends the resulting user
access token directly to api.openai.com.
"""

from __future__ import annotations

import asyncio
import errno
import math
import re
from dataclasses import dataclass
from typing import Any

from ...abort import AbortController, AbortSignal, abort_scope
from ...utils.misc import get_provider_env_value
from ...utils.oauth_page import oauth_error_html, oauth_success_html
from ..types import AuthEvent, AuthPrompt, Credential, LoginOptions, ModelAuth, OAuthAuth
from ..types import ProviderAuthInteraction as Interaction
from . import _common
from ._common import OAuthError, ParsedUrl, form_encode, parse_url
from .callback_server import LoopbackHttpServer
from .pkce import generate_pkce

__all__ = ["OpenAIChatGPTOAuth", "openai_chatgpt_oauth"]

# every login registers a new client with this ID; OpenAI returns the issued client ID in the callback
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = "Pi"
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}"
DIRECT_TOKEN_SCOPE = "chatgpt.tokens.use.direct"
SCOPE = f"openid profile email offline_access resource.invoke {DIRECT_TOKEN_SCOPE}"
# Refresh this long before the real expiry so a request never starts with a token about to expire.
EXPIRY_MARGIN_MS = 3 * 60 * 1000


@dataclass(frozen=True, slots=True)
class _AuthorizationResult:
    code: str
    client_id: str


def _callback_host() -> str:
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _authorization_result_from_callback(url: ParsedUrl, expected_state: str) -> _AuthorizationResult:
    code = url.param("code")
    if not code:
        raise OAuthError("Missing authorization code")
    state = url.param("state")
    if not state:
        raise OAuthError("Missing OAuth state")
    if state != expected_state:
        raise OAuthError("OAuth state mismatch")
    client_id = (url.param("client_id") or "").strip()
    if not client_id:
        raise OAuthError("OpenAI OAuth registration callback did not contain an issued client ID")
    return _AuthorizationResult(code, client_id)


def _authorization_result_from_manual_input(value: str, expected_state: str) -> _AuthorizationResult:
    url = parse_url(value.strip())
    if url is None:
        raise OAuthError("Paste the full callback URL from the browser")
    expected = parse_url(REDIRECT_URI)
    assert expected is not None
    if url.origin != expected.origin or url.pathname != expected.pathname:
        raise OAuthError(f"The pasted callback URL must start with {REDIRECT_URI}")
    error = url.param("error")
    if error:
        raise OAuthError(f"ChatGPT authorization failed: {error}")
    return _authorization_result_from_callback(url, expected_state)


class _CallbackServer:
    def __init__(self, expected_state: str) -> None:
        loop = asyncio.get_running_loop()
        self.result: asyncio.Future[_AuthorizationResult] = loop.create_future()
        self.result.add_done_callback(lambda future: future.cancelled() or future.exception())
        self._expected_state = expected_state
        self.server = LoopbackHttpServer(self._handle)

    def _reject(self, error: BaseException) -> None:
        if not self.result.done():
            self.result.set_exception(error)

    async def _handle(self, method: str, url: ParsedUrl | None) -> tuple[int, str]:
        if url is None:
            return 500, oauth_error_html("Internal error while processing the callback.")
        if url.pathname != CALLBACK_PATH:
            return 404, oauth_error_html("Callback route not found.")

        error = url.param("error")
        if error:
            self._reject(OAuthError(f"ChatGPT authorization failed: {error}"))
            return 400, oauth_error_html("ChatGPT was not connected.", f"Error: {error}")

        try:
            result = _authorization_result_from_callback(url, self._expected_state)
        except OAuthError as failure:
            return 400, oauth_error_html(str(failure) or "Invalid callback")

        if not self.result.done():
            self.result.set_result(result)
        return 200, oauth_success_html("ChatGPT authentication completed. You can close this window.")


async def _start_callback_server(expected_state: str) -> _CallbackServer:
    callback = _CallbackServer(expected_state)
    await callback.server.listen(_callback_host(), CALLBACK_PORT)
    return callback


async def _request_token(body: str, signal: AbortSignal) -> dict[str, Any]:
    response = await _common.fetch(
        TOKEN_URL,
        method="POST",
        headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
        body=body,
        signal=signal,
    )
    if not response.is_success:
        response_body = response.text
        raise OAuthError(
            f"OpenAI OAuth token request failed ({response.status_code}): "
            f"{response_body or _common.status_text(response)}"
        )
    data = _common.parse_json(response)
    if not isinstance(data, dict):
        raise OAuthError("OpenAI OAuth token response must be an object")
    return data


def _require_token_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OAuthError(f"OpenAI OAuth token response has invalid {field}")
    return value


def _credential_from_token_response(token: dict[str, Any], client_id: str) -> Credential:
    access = _require_token_string(token.get("access_token"), "access_token")
    refresh = _require_token_string(token.get("refresh_token"), "refresh_token")
    scope = _require_token_string(token.get("scope"), "scope")
    expires_in = token.get("expires_in")
    if not _common.is_number(expires_in) or not math.isfinite(expires_in) or expires_in <= 0:
        raise OAuthError("OpenAI OAuth token response has invalid expires_in")
    scopes = scope.split()
    if DIRECT_TOKEN_SCOPE not in scopes:
        raise OAuthError(f"OpenAI OAuth grant did not include {DIRECT_TOKEN_SCOPE}")
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires": _common.now_ms() + _common.expires_in_ms(expires_in) - EXPIRY_MARGIN_MS,
        "clientId": client_id,
        "scopes": scopes,
    }


async def _exchange_authorization_code(code: str, verifier: str, client_id: str, signal: AbortSignal) -> Credential:
    token = await _request_token(
        form_encode(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
                "resource": RESOURCE,
            }
        ),
        signal,
    )
    # Pi does not use the ID token to identify the user or read profile data.
    # Keep the presence check as part of the token-response contract.
    id_token = token.get("id_token")
    if not isinstance(id_token, str) or not id_token.strip():
        raise OAuthError("OpenAI OAuth token response did not contain an ID token")
    return _credential_from_token_response(token, client_id)


async def _refresh_access_token(credential: Credential, signal: AbortSignal) -> Credential:
    client_id = credential.get("clientId")
    if not isinstance(client_id, str) or not client_id.strip():
        raise OAuthError("Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT")
    token = await _request_token(
        form_encode(
            {
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": credential["refresh"],
                "resource": RESOURCE,
            }
        ),
        signal,
    )
    return _credential_from_token_response(token, client_id)


def _agent_host_id(device_id: str | None) -> str:
    """OpenAI identifies each installation ("agent host") by a stable URI such as ``urn:uuid:<uuid>``."""
    if not device_id or not UUID_PATTERN.match(device_id):
        raise OAuthError("Sign in with ChatGPT requires a device ID (UUID) for this installation")
    return f"urn:uuid:{device_id.lower()}"


async def _login_openai_chatgpt(interaction: Interaction, options: LoginOptions | None = None) -> Credential:
    get_device_id = options.get_device_id if options is not None else None
    host_id = _agent_host_id(get_device_id() if get_device_id is not None else None)
    verifier, challenge = generate_pkce()
    state = _common.random_base64url(32)
    nonce = _common.random_base64url(32)
    # Without this server, the browser's callback would reach whatever else holds the port (another
    # pending login or the Codex CLI), which rejects it as a state mismatch. Fail with a clear error instead.
    try:
        callback = await _start_callback_server(state)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        raise OAuthError(
            f"Port {CALLBACK_PORT} is in use, probably by an unfinished login in another pi session or by the "
            "Codex CLI. Cancel that login and try again."
        ) from error

    agent_name = options.agent_name if options is not None else None
    params = {
        "client_id": DYNAMIC_CLIENT_ID,
        "agent_name_hint": agent_name if agent_name is not None else AGENT_NAME_HINT,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "resource": RESOURCE,
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "nonce": nonce,
    }
    manual_abort = AbortController()
    # The manual prompt's signal also aborts with the login (`AbortSignal.any` upstream).
    remove_abort_forward = interaction.signal.add_listener(lambda: manual_abort.abort(interaction.signal.reason))
    manual_task: asyncio.Future[_AuthorizationResult] | None = None
    try:
        interaction.notify(
            AuthEvent(
                type="auth_url",
                url=f"{AUTHORIZE_URL}?{form_encode(params)}",
                instructions=(
                    "Complete sign-in in your browser. If the callback does not complete, "
                    "paste the final redirect URL here."
                ),
            )
        )

        async def manual_code() -> _AuthorizationResult:
            entered = await interaction.prompt(
                AuthPrompt(
                    type="manual_code",
                    message="Complete login in your browser, or paste the final redirect URL here:",
                    placeholder=REDIRECT_URI,
                    signal=manual_abort.signal,
                )
            )
            return _authorization_result_from_manual_input(entered, state)

        manual_task = asyncio.ensure_future(manual_code())
        manual_task.add_done_callback(_consume_task)
        async with abort_scope(interaction.signal):
            # Promise.race: the first side to settle wins, including with an error.
            done, _ = await asyncio.wait({callback.result, manual_task}, return_when=asyncio.FIRST_COMPLETED)
            winner = callback.result if callback.result in done else manual_task
            result = winner.result()
            interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
            return await _exchange_authorization_code(result.code, verifier, result.client_id, interaction.signal)
    except Exception as error:
        if interaction.signal.aborted:
            raise _common.login_cancelled() from error
        raise
    finally:
        remove_abort_forward()
        manual_abort.abort()
        if manual_task is not None and not manual_task.done():
            manual_task.cancel()
        callback.server.close()
        # close() only stops accepting new connections. Browsers open spare connections ahead of
        # time, and one that has not sent a request yet stays open and attached to this server.
        # A later login in the same process starts a new server with a new state, but the browser
        # may send that login's callback over the spare connection. This server would then handle
        # it and reject it with "OAuth state mismatch", and the new login would never see it.
        callback.server.close_all_connections()


def _consume_task(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


class OpenAIChatGPTOAuth(OAuthAuth):
    name = "OpenAI (ChatGPT subscription)"
    is_subscription = True
    login_label = "Sign in with ChatGPT"

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        return await _login_openai_chatgpt(interaction, options)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return await _refresh_access_token(credential, signal)

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


openai_chatgpt_oauth = OpenAIChatGPTOAuth()
