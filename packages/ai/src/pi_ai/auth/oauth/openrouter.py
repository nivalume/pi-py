"""OpenRouter OAuth PKCE flow.

OpenRouter exchanges an authorization code for a permanent, user-controlled API
key rather than an expiring access/refresh token pair. The callback is handled by
a one-shot loopback server on an ephemeral port, raced against a manual prompt so
remote/headless sessions can paste the redirect URL when the browser cannot reach
the loopback server.
"""

from __future__ import annotations

import uuid
from typing import Any

from ...abort import AbortSignal
from ...utils.misc import get_provider_env_value
from ..types import AuthEvent, Credential, LoginOptions, ModelAuth, OAuthAuth
from ..types import ProviderAuthInteraction as Interaction
from . import _common
from ._common import MAX_SAFE_INTEGER, OAuthError, form_encode, json_stringify, parse_search_params, parse_url
from .callback_server import CallbackResult, start_oauth_callback_server, wait_for_callback_or_manual_input
from .pkce import generate_pkce

__all__ = ["OpenRouterOAuth", "openrouter_oauth"]

AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
LOGIN_TIMEOUT_MS = 5 * 60 * 1000
TOKEN_EXCHANGE_TIMEOUT_MS = 30_000


def _callback_host() -> str:
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _parse_authorization_input(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None

    url = parse_url(value)
    if url is not None:
        return url.param("code")

    if "code=" in value:
        for key, param in parse_search_params(value):
            if key == "code":
                return param
        return None

    return value


def _error_detail(body: dict[str, Any]) -> str | None:
    if isinstance(body.get("error_description"), str):
        return body["error_description"]
    if isinstance(body.get("message"), str):
        return body["message"]
    if isinstance(body.get("error"), str):
        return body["error"]
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return None


async def _exchange_authorization_code(code: str, verifier: str, signal: AbortSignal) -> Credential:
    if signal.aborted:
        raise _common.login_cancelled()

    try:
        response = await _common.fetch(
            TOKEN_URL,
            method="POST",
            headers={"accept": "application/json", "content-type": "application/json"},
            body=json_stringify({"code": code, "code_verifier": verifier, "code_challenge_method": "S256"}),
            signal=signal,
            timeout_ms=TOKEN_EXCHANGE_TIMEOUT_MS,
        )
    except Exception as error:
        if signal.aborted:
            raise _common.login_cancelled() from error
        if isinstance(error, TimeoutError):
            raise OAuthError("OpenRouter OAuth token exchange timed out") from error
        raise

    body: dict[str, Any] = {}
    try:
        parsed = _common.parse_json(response)
        if isinstance(parsed, dict):
            body = parsed
    except ValueError:
        if response.is_success:
            raise OAuthError("OpenRouter OAuth returned invalid JSON") from None

    if not response.is_success:
        detail = _error_detail(body)
        raise OAuthError(
            f"OpenRouter OAuth key exchange failed (HTTP {response.status_code}){f': {detail}' if detail else ''}"
        )

    key = body.get("key")
    if not isinstance(key, str) or not key:
        raise OAuthError('OpenRouter OAuth response carries no "key"')

    return {"type": "oauth", "access": key, "refresh": "", "expires": MAX_SAFE_INTEGER}


async def _login_openrouter(interaction: Interaction) -> Credential:
    verifier, challenge = generate_pkce()

    async def complete(code: str) -> Credential:
        return await _exchange_authorization_code(code, verifier, interaction.signal)

    # OpenRouter sends no `state`; the random path keeps stray requests from completing the sign-in.
    callback = await start_oauth_callback_server(
        provider_name="OpenRouter",
        host=_callback_host(),
        port=0,
        path=f"/oauth/callback/{uuid.uuid4()}",
        complete=complete,
        signal=interaction.signal,
        timeout_ms=LOGIN_TIMEOUT_MS,
    )

    try:
        params = {"callback_url": callback.redirect_uri, "code_challenge": challenge, "code_challenge_method": "S256"}
        interaction.notify(
            AuthEvent(type="progress", message=f"Listening for OpenRouter OAuth callback on {callback.redirect_uri}")
        )
        interaction.notify(
            AuthEvent(
                type="auth_url",
                url=f"{AUTHORIZE_URL}?{form_encode(params)}",
                instructions=(
                    "Complete sign-in in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )

        result = await wait_for_callback_or_manual_input(
            interaction,
            callback,
            message="Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
            placeholder=callback.redirect_uri,
        )
        if isinstance(result, CallbackResult):
            return result.value
        code = _parse_authorization_input(result.input)
        if not code:
            raise OAuthError("Missing authorization code")
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for an API key..."))
        return await _exchange_authorization_code(code, verifier, interaction.signal)
    finally:
        callback.close()


class OpenRouterOAuth(OAuthAuth):
    name = "OpenRouter OAuth"
    login_label = "Sign in with OpenRouter"

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        return await _login_openrouter(interaction)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return credential

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


openrouter_oauth = OpenRouterOAuth()
