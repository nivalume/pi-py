"""GitHub Copilot OAuth flow (GitHub device code, then a Copilot API token)."""

from __future__ import annotations

import asyncio
import base64
import email.utils
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ...abort import AbortController, AbortSignal
from ..types import AuthEvent, AuthPrompt, Credential, LoginOptions, ModelAuth, OAuthAuth
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

__all__ = [
    "GitHubCopilotOAuth",
    "github_copilot_oauth",
    "get_github_copilot_base_url",
    "normalize_domain",
]

CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode()

COPILOT_HEADERS: Mapping[str, str] = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
COPILOT_API_VERSION = "2026-06-01"
INDIVIDUAL_BASE_URL = "https://api.individual.githubcopilot.com"
_REQUEST_TIMEOUT_MS = 5000


@dataclass(slots=True)
class _DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    interval: float | None
    expires_in: float


@dataclass(slots=True)
class _RetryPolicy:
    max_retries: int
    max_elapsed_ms: int


@dataclass(slots=True)
class _ModelCatalog:
    available_model_ids: list[str]
    policy_model_ids: list[str]


def normalize_domain(value: str) -> str | None:
    """The hostname of a GitHub Enterprise URL or bare domain, or ``None`` when invalid."""
    trimmed = value.strip()
    if not trimmed:
        return None
    url = parse_url(trimmed if "://" in trimmed else f"https://{trimmed}")
    return url.hostname if url is not None else None


def _get_urls(domain: str) -> dict[str, str]:
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
        "copilot_token_url": f"https://api.{domain}/copilot_internal/v2/token",
    }


def _get_base_url_from_token(token: str) -> str | None:
    """API base URL from the ``proxy-ep`` of a Copilot token.

    Token format: ``tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...``
    returns ``https://api.individual.githubcopilot.com``.
    """
    match = re.search(r"proxy-ep=([^;]+)", token)
    if match is None:
        return None
    # Convert proxy.xxx to api.xxx
    api_host = re.sub(r"^proxy\.", "api.", match.group(1))
    return f"https://{api_host}"


def get_github_copilot_base_url(token: str | None = None, enterprise_domain: str | None = None) -> str:
    """Copilot API base URL: the token's proxy endpoint, the enterprise endpoint, or Individual."""
    # If we have a token, extract the base URL from proxy-ep
    if token:
        url_from_token = _get_base_url_from_token(token)
        if url_from_token:
            return url_from_token
    # Fallback for enterprise or if token parsing fails
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return INDIVIDUAL_BASE_URL


def _record(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _known_model_ids() -> frozenset[str]:
    from ...providers.catalog import load_catalog

    return frozenset(load_catalog("github-copilot")["chat"])


def _parse_model_catalog(raw: Any, allow_policy_fallback: bool) -> _ModelCatalog:
    record = _record(raw)
    data = record.get("data") if record is not None else None
    if not isinstance(data, list):
        raise OAuthError("Invalid Copilot models response")

    account_models: list[tuple[str, bool, Any]] = []
    for raw_item in data:
        item = _record(raw_item)
        model_id = item.get("id") if item is not None else None
        if item is None or not isinstance(model_id, str):
            continue
        capabilities = _record(item.get("capabilities"))
        supports = _record(capabilities.get("supports")) if capabilities is not None else None
        if supports is not None and supports.get("tool_calls") is False:
            continue
        policy = _record(item.get("policy"))
        account_models.append(
            (model_id, item.get("model_picker_enabled") is True, policy.get("state") if policy is not None else None)
        )

    picker_model_ids = [
        model_id for model_id, picker_enabled, state in account_models if picker_enabled and state != "disabled"
    ]
    use_policy_fallback = allow_policy_fallback and not picker_model_ids
    if picker_model_ids or not allow_policy_fallback:
        available_model_ids = picker_model_ids
    else:
        available_model_ids = [model_id for model_id, _, state in account_models if state == "enabled"]
    known = _known_model_ids()
    policy_model_ids = [
        model_id
        for model_id, picker_enabled, state in account_models
        if state == "unconfigured" and model_id in known and (picker_enabled or use_policy_fallback)
    ]
    return _ModelCatalog(available_model_ids, policy_model_ids)


_FLOAT_PREFIX = re.compile(r"^\s*([+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|Infinity))")


def _parse_float(value: str) -> float:
    """``Number.parseFloat``: the leading numeric prefix, or ``NaN``."""
    match = _FLOAT_PREFIX.match(value)
    if match is None:
        return math.nan
    return float(match.group(1).replace("Infinity", "inf"))


def _parse_http_date_ms(value: str) -> float:
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return math.nan
    if parsed is None:
        return math.nan
    return parsed.timestamp() * 1000


async def _fetch_with_rate_limit_retry(
    url: str,
    signal: AbortSignal,
    retry_policy: _RetryPolicy,
    *,
    method: str = "GET",
    headers: Mapping[str, str],
    body: str | None = None,
) -> httpx.Response:
    use_budget = retry_policy.max_retries > 0 and retry_policy.max_elapsed_ms > 0
    request_controller = AbortController()
    remove = signal.add_listener(lambda: request_controller.abort(signal.reason))
    budget_timer = None
    retry_deadline: float | None = None
    if use_budget:
        budget_timer = asyncio.get_running_loop().call_later(
            retry_policy.max_elapsed_ms / 1000,
            request_controller.abort,
            TimeoutError("The operation was aborted due to timeout"),
        )
        retry_deadline = _common.now_ms() + retry_policy.max_elapsed_ms
    request_signal = request_controller.signal
    try:
        retry = 0
        while True:
            response = await _common.fetch(
                url, method=method, headers=headers, body=body, signal=request_signal, timeout_ms=_REQUEST_TIMEOUT_MS
            )
            if response.status_code != 429 or retry == retry_policy.max_retries:
                return response

            retry_after = response.headers.get("retry-after")
            delay_ms = 500 * 2**retry
            if retry_after:
                seconds = _parse_float(retry_after)
                delay_ms = (
                    _parse_http_date_ms(retry_after) - _common.now_ms() if math.isnan(seconds) else seconds * 1000
                )
                if not math.isfinite(delay_ms):
                    return response
            delay_ms = max(0, delay_ms)
            if retry_deadline is not None and delay_ms >= retry_deadline - _common.now_ms():
                return response
            await response.aclose()
            await _common.sleep(delay_ms / 1000, request_signal)
            retry += 1
    finally:
        remove()
        if budget_timer is not None:
            budget_timer.cancel()


def _status_error(response: httpx.Response) -> OAuthError:
    return OAuthError(f"{response.status_code} {_common.status_text(response)}: {response.text}")


async def _fetch_models(
    copilot_token: str, enterprise_domain: str | None, signal: AbortSignal, retry_policy: _RetryPolicy
) -> _ModelCatalog:
    base_url = get_github_copilot_base_url(copilot_token, enterprise_domain)
    # Some Individual accounts return false for every picker flag despite explicit enabled policies.
    # Limit the fallback to that endpoint so other account types keep strict picker semantics.
    allow_policy_fallback = base_url == INDIVIDUAL_BASE_URL
    response = await _fetch_with_rate_limit_retry(
        f"{base_url}/models",
        signal,
        retry_policy,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {copilot_token}",
            **COPILOT_HEADERS,
            "X-GitHub-Api-Version": COPILOT_API_VERSION,
        },
    )
    if not response.is_success:
        raise _status_error(response)
    return _parse_model_catalog(_common.parse_json(response), allow_policy_fallback)


async def _fetch_json(url: str, signal: AbortSignal, **kwargs: Any) -> Any:
    response = await _common.fetch(url, signal=signal, **kwargs)
    if not response.is_success:
        raise _status_error(response)
    return _common.parse_json(response)


async def _start_device_flow(domain: str, signal: AbortSignal) -> _DeviceCodeResponse:
    urls = _get_urls(domain)
    data = await _fetch_json(
        urls["device_code_url"],
        signal,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.35.0",
        },
        body=form_encode({"client_id": CLIENT_ID, "scope": "read:user"}),
    )
    if not isinstance(data, dict):
        raise OAuthError("Invalid device code response")

    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    interval = data.get("interval")
    expires_in = data.get("expires_in")
    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or ("interval" in data and not _common.is_number(interval))
        or not _common.is_number(expires_in)
    ):
        raise OAuthError("Invalid device code response fields")

    # The verification URI is opened in the user's browser and to prevent the launcher from
    # opening an executable or similar, we force it to be a URL.
    parsed_uri = parse_url(verification_uri)
    if parsed_uri is None or parsed_uri.protocol not in ("https:", "http:"):
        raise OAuthError("Untrusted verification_uri in device code response")

    return _DeviceCodeResponse(device_code, user_code, parsed_uri.href, interval, expires_in)


async def _poll_for_github_access_token(domain: str, device: _DeviceCodeResponse, signal: AbortSignal) -> str:
    urls = _get_urls(domain)

    async def poll() -> DeviceCodePollResult:
        raw = await _fetch_json(
            urls["access_token_url"],
            signal,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            body=form_encode(
                {
                    "client_id": CLIENT_ID,
                    "device_code": device.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                }
            ),
        )
        if isinstance(raw, dict) and isinstance(raw.get("access_token"), str):
            return DeviceCodeComplete(raw["access_token"])

        if isinstance(raw, dict) and isinstance(raw.get("error"), str):
            error = raw["error"]
            description = raw.get("error_description")
            interval = raw.get("interval")
            if error == "authorization_pending":
                return DeviceCodePending()
            if error == "slow_down":
                return DeviceCodeSlowDown(interval if _common.is_number(interval) else None)
            description_suffix = f": {description}" if description else ""
            return DeviceCodeFailed(f"Device flow failed: {error}{description_suffix}")

        return DeviceCodeFailed("Invalid device token response")

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device.interval,
        expires_in_seconds=device.expires_in,
        wait_before_first_poll=True,
    )


async def _refresh_copilot_access_token(
    refresh_token: str, enterprise_domain: str | None, signal: AbortSignal
) -> Credential:
    domain = enterprise_domain or "github.com"
    urls = _get_urls(domain)
    raw = await _fetch_json(
        urls["copilot_token_url"],
        signal,
        headers={"Accept": "application/json", "Authorization": f"Bearer {refresh_token}", **COPILOT_HEADERS},
    )
    if not isinstance(raw, dict):
        raise OAuthError("Invalid Copilot token response")

    token = raw.get("token")
    expires_at = raw.get("expires_at")
    if not isinstance(token, str) or not _common.is_number(expires_at):
        raise OAuthError("Invalid Copilot token response fields")

    credential: Credential = {
        "type": "oauth",
        "refresh": refresh_token,
        "access": token,
        "expires": int(expires_at * 1000) - 5 * 60 * 1000,
    }
    if enterprise_domain is not None:
        credential["enterpriseUrl"] = enterprise_domain
    return credential


async def _refresh_github_copilot_token(
    refresh_token: str, enterprise_domain: str | None, signal: AbortSignal
) -> Credential:
    credentials = await _refresh_copilot_access_token(refresh_token, enterprise_domain, signal)
    catalog = await _fetch_models(credentials["access"], enterprise_domain, signal, _RetryPolicy(0, 0))
    return {**credentials, "availableModelIds": catalog.available_model_ids}


async def _enable_model(token: str, model_id: str, enterprise_domain: str | None, signal: AbortSignal) -> bool:
    """Enable a model for the user's account (required for some models, like Claude and Grok)."""
    base_url = get_github_copilot_base_url(token, enterprise_domain)
    try:
        response = await _fetch_with_rate_limit_retry(
            f"{base_url}/models/{model_id}/policy",
            signal,
            _RetryPolicy(2, 5000),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                **COPILOT_HEADERS,
                "openai-intent": "chat-policy",
                "x-interaction-type": "chat-policy",
            },
            body=json_stringify({"state": "enabled"}),
        )
    except Exception:
        if signal.aborted:
            raise
        return False
    if response.status_code == 429:
        raise _status_error(response)
    return response.is_success


async def _enable_models(
    token: str, model_ids: Sequence[str], enterprise_domain: str | None, signal: AbortSignal
) -> list[str]:
    """Enable the requested models and return the successful ids.

    Policy updates are best effort; exhausted rate limiting stops the batch.
    """
    enabled: list[str] = []
    for model_id in model_ids:
        try:
            if await _enable_model(token, model_id, enterprise_domain, signal):
                enabled.append(model_id)
        except Exception:
            if signal.aborted:
                raise
            break
    return enabled


async def _login_github_copilot(interaction: Interaction) -> Credential:
    entered = await interaction.prompt(
        AuthPrompt(
            type="text",
            message="GitHub Enterprise URL/domain (blank for github.com)",
            placeholder="company.ghe.com",
        )
    )
    if interaction.signal.aborted:
        raise _common.login_cancelled()

    trimmed = entered.strip()
    enterprise_domain = normalize_domain(entered)
    if trimmed and not enterprise_domain:
        raise OAuthError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"

    device = await _start_device_flow(domain, interaction.signal)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=device.user_code,
            verification_uri=device.verification_uri,
            interval_seconds=device.interval,
            expires_in_seconds=device.expires_in,
        )
    )

    github_access_token = await _poll_for_github_access_token(domain, device, interaction.signal)
    credentials = await _refresh_copilot_access_token(github_access_token, enterprise_domain, interaction.signal)
    catalog = await _fetch_models(credentials["access"], enterprise_domain, interaction.signal, _RetryPolicy(2, 5000))
    enabled_model_ids: list[str] = []
    if catalog.policy_model_ids:
        interaction.notify(AuthEvent(type="progress", message="Enabling models..."))
        enabled_model_ids = await _enable_models(
            credentials["access"], catalog.policy_model_ids, enterprise_domain, interaction.signal
        )
    return {
        **credentials,
        "availableModelIds": list(dict.fromkeys([*catalog.available_model_ids, *enabled_model_ids])),
    }


def _copilot_enterprise_domain(credential: Credential) -> str | None:
    enterprise_url = credential.get("enterpriseUrl")
    if not isinstance(enterprise_url, str) or not enterprise_url:
        return None
    return normalize_domain(enterprise_url)


class GitHubCopilotOAuth(OAuthAuth):
    name = "GitHub Copilot"
    is_subscription = True

    async def login(self, interaction: Interaction, options: LoginOptions | None = None) -> Credential:
        return await _login_github_copilot(interaction)

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        return await _refresh_github_copilot_token(
            credential["refresh"], _copilot_enterprise_domain(credential), signal
        )

    async def to_auth(self, credential: Credential) -> ModelAuth:
        """Derive the credential-specific proxy endpoint for each request."""
        return ModelAuth(
            api_key=credential["access"],
            base_url=get_github_copilot_base_url(credential["access"], _copilot_enterprise_domain(credential)),
        )


github_copilot_oauth = GitHubCopilotOAuth()
