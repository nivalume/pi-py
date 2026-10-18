"""Offline tests for the built-in OAuth flows (ports of upstream test/*oauth*.test.ts)."""

from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import json
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

import pi_ai.oauth as extension_oauth
from pi_ai import (
    AbortController,
    AbortError,
    AbortSignal,
    AuthEvent,
    AuthPrompt,
    AuthSelectOption,
    InMemoryCredentialStore,
    LoginOptions,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
    create_models,
    create_provider,
    lazy_oauth,
)
from pi_ai.auth.oauth import _common
from pi_ai.auth.oauth.anthropic import anthropic_oauth
from pi_ai.auth.oauth.callback_server import (
    CallbackResult,
    ManualInput,
    start_oauth_callback_server,
    wait_for_callback_or_manual_input,
)
from pi_ai.auth.oauth.device_code import (
    DeviceCodeComplete,
    DeviceCodePending,
    DeviceCodeSlowDown,
    poll_oauth_device_code_flow,
)
from pi_ai.auth.oauth.github_copilot import github_copilot_oauth
from pi_ai.auth.oauth.kimi_coding import kimi_coding_oauth
from pi_ai.auth.oauth.load import load_anthropic_oauth, load_github_copilot_oauth
from pi_ai.auth.oauth.meta import meta_oauth
from pi_ai.auth.oauth.openai_chatgpt import openai_chatgpt_oauth
from pi_ai.auth.oauth.openai_codex import openai_codex_oauth
from pi_ai.auth.oauth.openrouter import openrouter_oauth
from pi_ai.auth.oauth.pkce import generate_pkce
from pi_ai.auth.oauth.radius import create_radius_oauth
from pi_ai.auth.oauth.xai import xai_oauth
from pi_ai.providers.catalog import get_builtin_models
from pi_ai.utils.oauth_page import oauth_error_html, oauth_success_html

NEVER = AbortSignal()
START = 1_773_014_400_000  # 2026-03-09T00:00:00Z

# -- helpers ------------------------------------------------------------------------


class _Inner:
    def __init__(
        self, prompt: Callable[[AuthPrompt], Awaitable[str]] | None, notify: Callable[[AuthEvent], Any] | None
    ):
        self.signal = None
        self._prompt = prompt
        self._notify = notify

    async def prompt(self, prompt: AuthPrompt) -> str:
        if self._prompt is None:
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        return await self._prompt(prompt)

    def notify(self, event: AuthEvent) -> None:
        if self._notify is not None:
            self._notify(event)


def interaction(
    prompt: Callable[[AuthPrompt], Awaitable[str]] | None = None,
    notify: Callable[[AuthEvent], Any] | None = None,
    signal: AbortSignal | None = None,
) -> ProviderAuthInteraction:
    return ProviderAuthInteraction(_Inner(prompt, notify), signal or AbortSignal())


async def pending_prompt(prompt: AuthPrompt) -> str:
    """A manual prompt that stays open until its signal aborts."""
    assert prompt.signal is not None
    await prompt.signal.wait()
    raise RuntimeError("prompt aborted")


async def forever_prompt(prompt: AuthPrompt) -> str:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


def select(choice: str, then: Callable[[AuthPrompt], Awaitable[str]] = pending_prompt):
    async def prompt(p: AuthPrompt) -> str:
        if p.type == "select":
            return choice
        return await then(p)

    return prompt


def json_response(body: Any, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers)


def form(request: httpx.Request) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(request.content.decode(), keep_blank_values=True).items()}


def query(url: str) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(urlsplit(url).query, keep_blank_values=True).items()}


def with_query(url: str, **params: str) -> str:
    parsed = _common.parse_url(url)
    assert parsed is not None
    entries = [(k, v) for k, v in parsed.search_params if k not in params] + list(params.items())
    return parsed.with_search(_common.form_encode(entries)).href


class Http:
    """Routes OAuth flow requests to ``handler`` and records them."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.handler: Callable[[httpx.Request], Any] = lambda request: (_ for _ in ()).throw(
            AssertionError(f"Unexpected request: {request.url}")
        )

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        result = self.handler(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    @property
    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> Http:
    router = Http()
    monkeypatch.setattr(_common, "http_client", httpx.AsyncClient(transport=httpx.MockTransport(router._handle)))
    return router


class Clock:
    """Virtual clock: sleeps return immediately and advance time."""

    def __init__(self, start: int = START) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def now_ms(self) -> int:
        return self.now

    async def sleep(self, seconds: float, signal: AbortSignal | None = None) -> None:
        await asyncio.sleep(0)
        if signal is not None:
            signal.throw_if_aborted()
        self.sleeps.append(seconds)
        self.now += round(seconds * 1000)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(_common, "now_ms", fake.now_ms)
    monkeypatch.setattr(_common, "sleep", fake.sleep)
    return fake


@pytest.fixture
async def local() -> Any:
    """A real HTTP client for the loopback callback servers (no proxies)."""
    async with httpx.AsyncClient(trust_env=False) as client:
        yield client


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


# -- PKCE, URL helpers, page ---------------------------------------------------------


def test_pkce_challenge_is_sha256_of_verifier():
    verifier, challenge = generate_pkce()
    assert len(verifier) == 43
    assert challenge == b64url(hashlib.sha256(verifier.encode()).digest())
    assert generate_pkce().verifier != verifier


def test_form_encode_matches_url_search_params():
    assert _common.form_encode({"scope": "read:user a b", "x": "~*-._"}) == "scope=read%3Auser+a+b&x=%7E*-._"


def test_parse_url_normalizes_like_whatwg():
    assert _common.parse_url("$(id>/tmp/pwned)") is None
    assert _common.parse_url("not a url") is None
    url = _common.parse_url("https://GitHub.com:443/login/\x1b]8;;evil")
    assert url is not None
    assert url.href == "https://github.com/login/%1B]8;;evil"
    assert _common.parse_url("https://example.com").href == "https://example.com/"  # type: ignore[union-attr]
    assert _common.parse_url("https://a.com/x/../y?q=1 2#f").href == "https://a.com/y?q=1%202#f"  # type: ignore[union-attr]
    assert _common.parse_url("file:///etc/passwd").protocol == "file:"  # type: ignore[union-attr]
    assert _common.parse_url("https://exa mple.com") is None
    assert _common.parse_url("code#state") is None


def test_parse_authorization_input_forms():
    parse = _common.parse_authorization_input
    assert parse("") == (None, None)
    assert parse("http://localhost:1/cb?code=c&state=s") == ("c", "s")
    assert parse("c#s#ignored") == ("c", "s")
    assert parse("code=c&state=s") == ("c", "s")
    assert parse("  bare  ") == ("bare", None)


def test_oauth_pages_render_logo_and_escape_details():
    assert "Authentication successful" in oauth_success_html("ok")
    page = oauth_error_html("Failed", "<b>details</b>")
    assert 'fill="#F09082"' in page
    assert "&lt;b&gt;details&lt;/b&gt;" in page
    assert '<div class="details">' not in oauth_error_html("Failed")


# -- device-code polling -------------------------------------------------------------


async def test_device_code_polls_immediately_and_returns_value(clock: Clock):
    times: list[int] = []

    async def poll():
        times.append(clock.now)
        return DeviceCodePending() if len(times) == 1 else DeviceCodeComplete("token")

    result = await poll_oauth_device_code_flow(poll=poll, signal=NEVER, interval_seconds=2, expires_in_seconds=30)
    assert result == "token"
    assert times == [START, START + 2000]


async def test_device_code_can_wait_before_first_poll(clock: Clock):
    times: list[int] = []

    async def poll():
        times.append(clock.now)
        return DeviceCodeComplete("token")

    await poll_oauth_device_code_flow(
        poll=poll, signal=NEVER, interval_seconds=2, expires_in_seconds=30, wait_before_first_poll=True
    )
    assert times == [START + 2000]


async def test_device_code_slow_down_without_interval_adds_five_seconds(clock: Clock):
    times: list[int] = []
    results = [DeviceCodeSlowDown(), DeviceCodeComplete("token")]

    async def poll():
        times.append(clock.now)
        return results.pop(0)

    await poll_oauth_device_code_flow(poll=poll, signal=NEVER, interval_seconds=2, expires_in_seconds=900)
    assert times == [START, START + 7000]


async def test_device_code_honors_server_slow_down_interval(clock: Clock):
    times: list[int] = []
    results = [DeviceCodeSlowDown(30), DeviceCodeComplete("token")]

    async def poll():
        times.append(clock.now)
        return results.pop(0)

    await poll_oauth_device_code_flow(poll=poll, signal=NEVER, interval_seconds=2, expires_in_seconds=900)
    assert times == [START, START + 30000]


async def test_device_code_cancels_an_in_flight_wait(clock: Clock):
    controller = AbortController()

    async def poll():
        controller.abort()
        return DeviceCodePending()

    with pytest.raises(AbortError, match="Login cancelled"):
        await poll_oauth_device_code_flow(
            poll=poll, signal=controller.signal, interval_seconds=5, expires_in_seconds=30
        )


# -- callback server -----------------------------------------------------------------


async def start(**overrides: Any):
    async def complete(code: str) -> str:
        return f"completed:{code}"

    options: dict[str, Any] = {
        "provider_name": "Example",
        "host": "127.0.0.1",
        "port": 0,
        "path": "/callback",
        "state": "expected-state",
        "complete": complete,
        **overrides,
    }
    return await start_oauth_callback_server(**options)


async def test_callback_server_ignores_stray_requests_and_completes(local: httpx.AsyncClient):
    server = await start()
    try:
        assert urlsplit(server.redirect_uri).hostname == "127.0.0.1"
        assert server.redirect_uri.endswith("/callback")
        base = server.redirect_uri.rsplit("/", 1)[0]
        assert (await local.get(f"{base}/other")).status_code == 404
        wrong = await local.get(with_query(server.redirect_uri, code="c", state="other"))
        assert wrong.status_code == 400
        assert wrong.headers["content-type"] == "text/html; charset=utf-8"
        assert "State mismatch." in wrong.text
        assert (await local.post(with_query(server.redirect_uri, code="c", state="expected-state"))).status_code == 404
        assert (await local.get(with_query(server.redirect_uri, state="expected-state"))).status_code == 400
        success = await local.get(with_query(server.redirect_uri, code="the-code", state="expected-state"))
        assert success.status_code == 200
        assert "Authentication successful" in success.text
        assert "Signed in to Example." in success.text
        assert await server.wait() == "completed:the-code"
    finally:
        server.close()


async def test_callback_server_redirect_host_and_no_state(local: httpx.AsyncClient):
    server = await start(redirect_host="localhost", state=None)
    try:
        assert server.redirect_uri.startswith("http://localhost:")
        url = with_query(server.redirect_uri, code="no-state").replace("localhost", "127.0.0.1")
        assert (await local.get(url)).status_code == 200
        assert await server.wait() == "completed:no-state"
    finally:
        server.close()


async def test_callback_server_shows_completion_failures(local: httpx.AsyncClient):
    async def complete(code: str) -> str:
        raise RuntimeError("token exchange failed")

    server = await start(complete=complete)
    try:
        failure = await local.get(with_query(server.redirect_uri, code="c", state="expected-state"))
        assert failure.status_code == 502
        assert "Example sign-in failed." in failure.text
        assert "token exchange failed" in failure.text
        with pytest.raises(RuntimeError, match="token exchange failed"):
            await server.wait()
    finally:
        server.close()


async def test_callback_server_rejects_provider_errors(local: httpx.AsyncClient):
    server = await start()
    try:
        failure = await local.get(
            with_query(
                server.redirect_uri,
                error="access_denied",
                error_description="User denied access",
                state="expected-state",
            )
        )
        assert failure.status_code == 400
        assert "User denied access" in failure.text
        with pytest.raises(Exception, match="Example authorization failed: User denied access"):
            await server.wait()
    finally:
        server.close()


async def test_callback_server_completes_only_the_first_callback(local: httpx.AsyncClient):
    started = asyncio.Event()
    finish: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def complete(code: str) -> str:
        started.set()
        return await finish

    server = await start(complete=complete)
    try:
        url = with_query(server.redirect_uri, code="c", state="expected-state")
        first = asyncio.ensure_future(local.get(url))
        await started.wait()
        assert (await local.get(url)).status_code == 409
        # A claimed callback keeps completing even when the caller switches to manual input.
        server.cancel()
        finish.set_result("done")
        assert (await first).status_code == 200
        assert await server.wait() == "done"
    finally:
        server.close()


async def test_callback_server_cancel_resolves_none(local: httpx.AsyncClient):
    server = await start()
    try:
        server.cancel()
        assert await server.wait() is None
        late = await local.get(with_query(server.redirect_uri, code="c", state="expected-state"))
        assert late.status_code == 409
    finally:
        server.close()


async def test_callback_server_abort_and_timeout():
    controller = AbortController()
    aborted = await start(signal=controller.signal)
    controller.abort()
    with pytest.raises(AbortError, match="Login cancelled"):
        await aborted.wait()
    aborted.close()

    timed_out = await start(timeout_ms=10)
    with pytest.raises(Exception, match="Example sign-in timed out"):
        await timed_out.wait()
    timed_out.close()

    with pytest.raises(AbortError, match="Login cancelled"):
        await start(signal=AbortSignal.abort())


async def test_callback_server_fails_when_port_is_taken():
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    try:
        with pytest.raises(OSError) as info:
            await start(port=blocker.getsockname()[1])
        assert info.value.errno == errno.EADDRINUSE
    finally:
        blocker.close()


async def test_wait_returns_browser_callback_and_aborts_manual_prompt(local: httpx.AsyncClient):
    signals: list[AbortSignal | None] = []

    async def prompt(p: AuthPrompt) -> str:
        signals.append(p.signal)
        return await pending_prompt(p)

    server = await start(state=None)
    try:
        waiting = asyncio.ensure_future(
            wait_for_callback_or_manual_input(
                interaction(prompt), server, message="paste", placeholder=server.redirect_uri
            )
        )
        await local.get(with_query(server.redirect_uri, code="from-browser"))
        assert await waiting == CallbackResult("completed:from-browser")
        assert signals[0] is not None and signals[0].aborted
    finally:
        server.close()


async def test_wait_returns_pasted_input():
    async def pasted(p: AuthPrompt) -> str:
        assert p.type == "manual_code"
        return "pasted"

    server = await start()
    try:
        result = await wait_for_callback_or_manual_input(
            interaction(pasted), server, message="paste", placeholder=server.redirect_uri
        )
        assert result == ManualInput("pasted")
    finally:
        server.close()
    result = await wait_for_callback_or_manual_input(interaction(pasted), None, message="paste", placeholder="x")
    assert result == ManualInput("pasted")


async def test_wait_propagates_manual_prompt_failures():
    async def failing(p: AuthPrompt) -> str:
        raise RuntimeError("prompt cancelled")

    server = await start()
    try:
        with pytest.raises(RuntimeError, match="prompt cancelled"):
            await wait_for_callback_or_manual_input(
                interaction(failing), server, message="paste", placeholder=server.redirect_uri
            )
    finally:
        server.close()


# -- OAuthAuth adapters (oauth-auth.test.ts) ------------------------------------------


def test_extension_oauth_module_has_no_flow_implementations():
    assert not hasattr(extension_oauth, "login_anthropic")
    assert not hasattr(extension_oauth, "anthropic_oauth")
    assert extension_oauth.OAuthPrompt("m").allow_empty is None


def test_subscription_flags():
    for oauth in (
        anthropic_oauth,
        openai_chatgpt_oauth,
        openai_codex_oauth,
        github_copilot_oauth,
        kimi_coding_oauth,
        xai_oauth,
        meta_oauth,
    ):
        assert oauth.is_subscription is True
    assert openrouter_oauth.is_subscription is not True
    assert openai_chatgpt_oauth.name == "OpenAI (ChatGPT subscription)"
    assert openai_chatgpt_oauth.login_label == "Sign in with ChatGPT"


async def test_to_auth_derivations():
    credential = {"type": "oauth", "access": "token", "refresh": "r", "expires": 0}
    for oauth in (anthropic_oauth, openai_codex_oauth, xai_oauth, openrouter_oauth, meta_oauth):
        assert await oauth.to_auth(credential) == ModelAuth(api_key="token")
    assert await kimi_coding_oauth.to_auth(credential) == ModelAuth(headers={"Authorization": "Bearer token"})
    permanent = {"type": "oauth", "access": "token", "refresh": "", "expires": _common.MAX_SAFE_INTEGER}
    assert await openrouter_oauth.refresh(permanent, NEVER) is permanent


async def test_github_copilot_to_auth_base_url():
    access = "tid=abc;exp=123;proxy-ep=proxy.enterprise.example;rest"
    auth = await github_copilot_oauth.to_auth({"type": "oauth", "access": access, "refresh": "r", "expires": 0})
    assert auth == ModelAuth(api_key=access, base_url="https://api.enterprise.example")
    enterprise = await github_copilot_oauth.to_auth(
        {
            "type": "oauth",
            "access": "no-proxy-ep",
            "refresh": "r",
            "expires": 0,
            "enterpriseUrl": "https://company.ghe.com",
        }
    )
    assert enterprise.base_url == "https://copilot-api.company.ghe.com"
    individual = await github_copilot_oauth.to_auth(
        {"type": "oauth", "access": "no-proxy-ep", "refresh": "r", "expires": 0}
    )
    assert individual.base_url == "https://api.individual.githubcopilot.com"


async def test_github_copilot_refresh_preserves_enterprise_domain(http: Http):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/models"):
            return json_response({"data": []})
        return json_response({"token": "new-token", "expires_at": 9999999999})

    http.handler = handler
    refreshed = await github_copilot_oauth.refresh(
        {"type": "oauth", "access": "old", "refresh": "gh-token", "expires": 0, "enterpriseUrl": "company.ghe.com"},
        NEVER,
    )
    assert refreshed["access"] == "new-token"
    assert refreshed["enterpriseUrl"] == "company.ghe.com"
    assert refreshed["expires"] == 9999999999 * 1000 - 300_000
    assert "api.company.ghe.com" in http.urls[0]
    assert http.requests[0].headers["authorization"] == "Bearer gh-token"


class _NoApi:
    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    stream_simple = stream


async def test_models_get_auth_through_the_lazy_oauth_chain():
    credentials = InMemoryCredentialStore(
        {
            "anthropic": {"type": "oauth", "access": "oauth-access-token", "refresh": "r", "expires": 10**15},
            "github-copilot": {
                "type": "oauth",
                "access": "tid=abc;proxy-ep=proxy.business.githubcopilot.com;rest",
                "refresh": "r",
                "expires": 10**15,
            },
        }
    )
    models = create_models(credentials=credentials)
    models.set_provider(
        create_provider(
            id="anthropic", auth=ProviderAuth(oauth=lazy_oauth("Anthropic", load_anthropic_oauth)), api=_NoApi()
        )
    )
    models.set_provider(
        create_provider(
            id="github-copilot", auth=ProviderAuth(oauth=lazy_oauth("Copilot", load_github_copilot_oauth)), api=_NoApi()
        )
    )
    result = await models.get_auth("anthropic")
    assert result is not None and result.auth.api_key == "oauth-access-token" and result.source == "OAuth"
    copilot = await models.get_auth("github-copilot")
    assert copilot is not None and copilot.auth.base_url == "https://api.business.githubcopilot.com"


# -- Anthropic ----------------------------------------------------------------------

ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"


def anthropic_tokens(request: httpx.Request) -> httpx.Response:
    assert str(request.url) == ANTHROPIC_TOKEN_URL
    return json_response({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600})


async def test_anthropic_manual_redirect_login(http: Http):
    http.handler = anthropic_tokens
    events: list[AuthEvent] = []
    signals: list[AbortSignal | None] = []

    async def manual(p: AuthPrompt) -> str:
        assert p.type == "manual_code"
        signals.append(p.signal)
        params = query(events[0].url or "")
        return f"{params['redirect_uri']}?code=manual-code&state={params['state']}"

    credential = await anthropic_oauth.login(interaction(select("browser", manual), events.append))
    assert credential["access"] == "access-token" and credential["refresh"] == "refresh-token"
    assert credential["type"] == "oauth"
    body = json.loads(http.requests[0].content)
    assert body["grant_type"] == "authorization_code" and body["code"] == "manual-code"
    assert body["redirect_uri"] == query(events[0].url or "")["redirect_uri"]
    assert body["redirect_uri"].startswith("http://localhost:") and body["redirect_uri"].endswith("/callback")
    assert len(http.requests) == 1
    # the prompt's signal is aborted once login settles, so UIs can dismiss it
    assert signals[0] is not None and signals[0].aborted


async def test_anthropic_copy_code_login(http: Http):
    prompts: list[AuthPrompt] = []
    auth_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["code"] == "copied-code"
        assert body["state"] == query(auth_urls[0])["state"]
        assert body["redirect_uri"] == "https://platform.claude.com/oauth/code/callback"
        return anthropic_tokens(request)

    http.handler = handler

    async def prompt(p: AuthPrompt) -> str:
        prompts.append(p)
        if p.type == "select":
            return "copy_code"
        return f"copied-code#{query(auth_urls[0])['state']}"

    credential = await anthropic_oauth.login(
        interaction(prompt, lambda e: e.type == "auth_url" and auth_urls.append(e.url or ""))
    )
    assert credential["access"] == "access-token"
    assert query(auth_urls[0])["redirect_uri"] == "https://platform.claude.com/oauth/code/callback"
    assert prompts[0] == AuthPrompt(
        type="select",
        message="Select Anthropic login method:",
        options=[
            AuthSelectOption("browser", "Browser login (default)"),
            AuthSelectOption("copy_code", "Copy code login (headless)"),
        ],
    )


async def test_anthropic_cancelled_method_selection():
    async def cancel(p: AuthPrompt) -> str:
        raise RuntimeError("Login cancelled")

    with pytest.raises(RuntimeError, match="Login cancelled"):
        await anthropic_oauth.login(interaction(cancel))


async def test_anthropic_refresh_omits_scope(http: Http, clock: Clock):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.method == "POST"
        assert body["grant_type"] == "refresh_token" and body["refresh_token"] == "refresh-token"
        assert body["client_id"] and "scope" not in body
        return json_response({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    http.handler = handler
    credential = await anthropic_oauth.refresh({"type": "oauth", "access": "old", "refresh": "refresh-token"}, NEVER)
    assert credential == {
        "type": "oauth",
        "refresh": "new-refresh",
        "access": "new-access",
        "expires": START + 3600_000 - 300_000,
    }


async def test_anthropic_refresh_failure_message(http: Http):
    http.handler = lambda request: json_response({"error": "invalid_grant"}, 400)
    with pytest.raises(Exception, match=r"Anthropic token refresh request failed.*status=400"):
        await anthropic_oauth.refresh({"type": "oauth", "access": "old", "refresh": "r"}, NEVER)


async def _anthropic_browser_login(http: Http, local: httpx.AsyncClient) -> dict[str, Any]:
    exchanged: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        exchanged.update(json.loads(request.content))
        return json_response({"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})

    http.handler = handler
    pages: list[asyncio.Future[httpx.Response]] = []
    redirect: list[str] = []

    def notify(event: AuthEvent) -> None:
        if event.type != "auth_url":
            return
        params = query(event.url or "")
        redirect.append(params["redirect_uri"])
        url = with_query(params["redirect_uri"], code="browser-code", state=params["state"])
        pages.append(asyncio.ensure_future(local.get(url.replace("//localhost:", "//127.0.0.1:"))))

    credential = await anthropic_oauth.login(interaction(select("browser"), notify))
    page = await pages[0]
    return {"credential": credential, "redirect": redirect[0], "exchanged": exchanged, "page": page}


async def test_anthropic_browser_callback_login(http: Http, local: httpx.AsyncClient):
    login = await _anthropic_browser_login(http, local)
    assert login["credential"]["access"] == "access"
    assert login["exchanged"]["code"] == "browser-code"
    assert login["exchanged"]["redirect_uri"] == login["redirect"]
    assert login["page"].status_code == 200
    assert "Signed in to Anthropic." in login["page"].text


async def test_anthropic_falls_back_to_a_free_port(http: Http, local: httpx.AsyncClient):
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        blocker.bind(("127.0.0.1", 53692))
        blocker.listen()
    except OSError:
        pass  # already taken by something else
    try:
        login = await _anthropic_browser_login(http, local)
        redirect = urlsplit(login["redirect"])
        assert redirect.hostname == "localhost" and redirect.path == "/callback"
        assert redirect.port != 53692
        assert login["exchanged"]["redirect_uri"] == login["redirect"]
        assert login["page"].status_code == 200
    finally:
        blocker.close()


# -- OpenAI Codex -------------------------------------------------------------------

CODEX_USERCODE = "https://auth.openai.com/api/accounts/deviceauth/usercode"
CODEX_DEVICE_TOKEN = "https://auth.openai.com/api/accounts/deviceauth/token"
CODEX_TOKEN = "https://auth.openai.com/oauth/token"
CODEX_CLIENT = "app_EMoamEEZ73f0CkXaXp7hrann"


def access_token(account_id: str) -> str:
    header = base64.b64encode(json.dumps({"alg": "none"}).encode()).decode()
    payload = base64.b64encode(json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode())
    return f"{header}.{payload.decode()}.signature"


def device_pending() -> httpx.Response:
    return json_response(
        {
            "error": {
                "message": "Device authorization is pending. Please try again.",
                "type": "invalid_request_error",
                "param": None,
                "code": "deviceauth_authorization_pending",
            }
        },
        403,
    )


def codex_device_interaction(events: list[AuthEvent], signal: AbortSignal | None = None) -> ProviderAuthInteraction:
    async def prompt(p: AuthPrompt) -> str:
        assert p.type == "select"
        return "device_code"

    return interaction(prompt, events.append, signal)


async def test_codex_device_code_login(http: Http, clock: Clock):
    token = access_token("account-123")
    poll_times: list[int] = []
    polls = [
        device_pending(),
        json_response({"authorization_code": "oauth-code", "code_verifier": "device-code-verifier"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == CODEX_USERCODE:
            assert request.headers["content-type"] == "application/json"
            assert json.loads(request.content) == {"client_id": CODEX_CLIENT}
            return json_response({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234", "interval": "5"})
        if url == CODEX_DEVICE_TOKEN:
            poll_times.append(clock.now)
            assert json.loads(request.content) == {"device_auth_id": "device-auth-id", "user_code": "ABCD-1234"}
            return polls.pop(0)
        assert url == CODEX_TOKEN
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        assert form(request) == {
            "grant_type": "authorization_code",
            "client_id": CODEX_CLIENT,
            "code": "oauth-code",
            "code_verifier": "device-code-verifier",
            "redirect_uri": "https://auth.openai.com/deviceauth/callback",
        }
        return json_response({"access_token": token, "refresh_token": "refresh-token", "expires_in": 3600})

    http.handler = handler
    events: list[AuthEvent] = []
    credential = await openai_codex_oauth.login(codex_device_interaction(events))
    assert events == [
        AuthEvent(
            type="device_code",
            user_code="ABCD-1234",
            verification_uri="https://auth.openai.com/codex/device",
            interval_seconds=5,
            expires_in_seconds=900,
        )
    ]
    assert poll_times == [START, START + 5000]
    assert credential == {
        "type": "oauth",
        "access": token,
        "refresh": "refresh-token",
        "expires": START + 5000 + 3600 * 1000,
        "accountId": "account-123",
    }


async def test_codex_offers_browser_first():
    prompts: list[AuthPrompt] = []

    async def prompt(p: AuthPrompt) -> str:
        prompts.append(p)
        raise RuntimeError("Login cancelled")

    with pytest.raises(RuntimeError, match="Login cancelled"):
        await openai_codex_oauth.login(interaction(prompt))
    assert prompts[0].message == "Select OpenAI Codex login method:"
    assert prompts[0].options == [
        AuthSelectOption("browser", "Browser login (default)"),
        AuthSelectOption("device_code", "Device code login (headless)"),
    ]


async def test_codex_device_code_cancel_while_waiting(http: Http, clock: Clock):
    controller = AbortController()

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CODEX_USERCODE:
            return json_response({"device_auth_id": "d", "user_code": "ABCD-1234", "interval": "5"})
        controller.abort()
        return device_pending()

    http.handler = handler
    with pytest.raises(AbortError, match="^Login cancelled$"):
        await openai_codex_oauth.login(codex_device_interaction([], controller.signal))


async def test_codex_device_code_times_out_after_15_minutes(http: Http, clock: Clock):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CODEX_USERCODE:
            return json_response({"device_auth_id": "d", "user_code": "ABCD-1234", "interval": "60"})
        return device_pending()

    http.handler = handler
    with pytest.raises(Exception, match="^Device flow timed out$"):
        await openai_codex_oauth.login(codex_device_interaction([]))
    assert clock.now == START + 15 * 60 * 1000


async def test_codex_device_403_and_404_are_pending(http: Http, clock: Clock):
    token = access_token("account-403-404")
    polls = [
        json_response({"error": "access_denied", "error_description": "denied"}, 403),
        httpx.Response(404, text="not ready"),
        json_response({"authorization_code": "oauth-code", "code_verifier": "v"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == CODEX_USERCODE:
            return json_response({"device_auth_id": "d", "user_code": "ABCD-1234", "interval": "1"})
        if url == CODEX_DEVICE_TOKEN:
            return polls.pop(0)
        return json_response({"access_token": token, "refresh_token": "refresh-token", "expires_in": 3600})

    http.handler = handler
    credential = await openai_codex_oauth.login(codex_device_interaction([]))
    assert credential["accountId"] == "account-403-404"
    assert sum(1 for url in http.urls if url == CODEX_DEVICE_TOKEN) == 3


async def test_codex_device_poll_failure_includes_body(http: Http, clock: Clock):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CODEX_USERCODE:
            return json_response({"device_auth_id": "d", "user_code": "ABCD-1234", "interval": "5"})
        return json_response({"error": "server_error", "error_description": "try again later"}, 500)

    http.handler = handler
    with pytest.raises(Exception) as info:
        await openai_codex_oauth.login(codex_device_interaction([]))
    assert str(info.value) == (
        'OpenAI Codex device auth failed with status 500: {"error":"server_error","error_description":"try again later"}'
    )


async def test_codex_refresh_failure_message(http: Http):
    http.handler = lambda request: json_response(
        {"error": {"message": "Could not validate your token. Please try signing in again."}}, 401
    )
    with pytest.raises(Exception, match=r"OpenAI Codex token refresh failed \(401\).*Could not validate your token"):
        await openai_codex_oauth.refresh({"type": "oauth", "access": "a", "refresh": "r", "expires": 0}, NEVER)


async def test_codex_refresh_extracts_account_id(http: Http):
    def handler(request: httpx.Request) -> httpx.Response:
        assert form(request) == {"grant_type": "refresh_token", "refresh_token": "r", "client_id": CODEX_CLIENT}
        return json_response({"access_token": access_token("acct"), "refresh_token": "r2", "expires_in": 60})

    http.handler = handler
    credential = await openai_codex_oauth.refresh({"type": "oauth", "access": "a", "refresh": "r"}, NEVER)
    assert credential["accountId"] == "acct" and credential["refresh"] == "r2"


async def test_codex_browser_login_uses_agent_name_and_pasted_redirect(http: Http):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CODEX_TOKEN
        body = form(request)
        assert body["code"] == "pasted-code"
        assert body["redirect_uri"] == "http://localhost:1455/auth/callback"
        return json_response({"access_token": access_token("acct"), "refresh_token": "refresh", "expires_in": 3600})

    http.handler = handler
    auth_urls: list[str] = []

    async def manual(p: AuthPrompt) -> str:
        assert p.type == "manual_code"
        return f"http://localhost:1455/auth/callback?code=pasted-code&state={query(auth_urls[0])['state']}"

    credential = await openai_codex_oauth.login(
        interaction(select("browser", manual), lambda e: e.type == "auth_url" and auth_urls.append(e.url or "")),
        LoginOptions(agent_name="my-app"),
    )
    assert credential["accountId"] == "acct"
    params = query(auth_urls[0])
    assert params["originator"] == "my-app"
    assert params["codex_cli_simplified_flow"] == "true"
    assert params["redirect_uri"] == "http://localhost:1455/auth/callback"


async def test_codex_browser_login_rejects_state_mismatch(http: Http):
    async def manual(p: AuthPrompt) -> str:
        return "http://localhost:1455/auth/callback?code=c&state=wrong"

    with pytest.raises(Exception, match="State mismatch"):
        await openai_codex_oauth.login(interaction(select("browser", manual)))
    assert http.requests == []


# -- OpenAI ChatGPT -----------------------------------------------------------------

CHATGPT_TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
REQUIRED_SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DEVICE_ID = "e61bbe28-07ef-466d-8e5d-a344f94ab305"


def chatgpt_tokens(scope: str = REQUIRED_SCOPE, **overrides: Any) -> dict[str, Any]:
    body = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 3600,
        "id_token": "id-token",
        "scope": scope,
    }
    body.update(overrides)
    return {key: value for key, value in body.items() if value is not None}


def chatgpt_endpoint(http: Http, response: dict[str, Any], bodies: list[dict[str, str]] | None = None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CHATGPT_TOKEN_URL
        if bodies is not None:
            bodies.append(form(request))
        return json_response(response)

    http.handler = handler


def chatgpt_interaction(client_id: str | None = None, authorize: list[str] | None = None) -> ProviderAuthInteraction:
    urls: list[str] = []

    def notify(event: AuthEvent) -> None:
        if event.type == "auth_url":
            urls.append(event.url or "")
            if authorize is not None:
                authorize.append(event.url or "")

    async def prompt(p: AuthPrompt) -> str:
        assert p.type == "manual_code"
        assert urls, "Authorization URL was not emitted before the callback prompt"
        params = query(urls[0])
        extra = {"client_id": client_id} if client_id else {}
        return with_query(params["redirect_uri"], code="authorization-code", state=params["state"], **extra)

    return interaction(prompt, notify)


async def test_chatgpt_registers_client_and_stores_scopes(http: Http):
    bodies: list[dict[str, str]] = []
    chatgpt_endpoint(http, chatgpt_tokens(), bodies)
    authorize: list[str] = []
    credential = await openai_chatgpt_oauth.login(
        chatgpt_interaction("oaiapp_issued", authorize), LoginOptions(get_device_id=lambda: DEVICE_ID)
    )
    params = query(authorize[0])
    assert params["client_id"] == "dynamic_agent_client"
    assert params["agent_name_hint"] == "Pi"
    assert params["ext_agent_host_id"] == f"urn:uuid:{DEVICE_ID}"
    assert params["scope"] == REQUIRED_SCOPE
    assert params["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"
    assert params["resource"] == "https://api.openai.com/v1"
    assert params["code_challenge_method"] == "S256"
    assert bodies[0]["client_id"] == "oaiapp_issued"
    assert bodies[0]["code"] == "authorization-code"
    assert bodies[0]["resource"] == "https://api.openai.com/v1"
    assert bodies[0]["code_verifier"]
    assert credential["access"] == "access-token"
    assert credential["clientId"] == "oaiapp_issued"
    assert credential["scopes"] == REQUIRED_SCOPE.split(" ")


async def test_chatgpt_uses_agent_name(http: Http):
    chatgpt_endpoint(http, chatgpt_tokens())
    authorize: list[str] = []
    await openai_chatgpt_oauth.login(
        chatgpt_interaction("oaiapp_issued", authorize),
        LoginOptions(get_device_id=lambda: DEVICE_ID, agent_name="my-app"),
    )
    assert query(authorize[0])["agent_name_hint"] == "my-app"


async def test_chatgpt_rejects_missing_issued_client_id(http: Http):
    chatgpt_endpoint(http, chatgpt_tokens())
    with pytest.raises(Exception, match="registration callback did not contain an issued client ID"):
        await openai_chatgpt_oauth.login(chatgpt_interaction(), LoginOptions(get_device_id=lambda: DEVICE_ID))
    assert http.requests == []


async def test_chatgpt_rejects_grant_without_direct_scope(http: Http):
    chatgpt_endpoint(http, chatgpt_tokens("openid profile email offline_access resource.invoke"))
    with pytest.raises(Exception, match="grant did not include chatgpt.tokens.use.direct"):
        await openai_chatgpt_oauth.login(
            chatgpt_interaction("oaiapp_issued"), LoginOptions(get_device_id=lambda: DEVICE_ID)
        )


async def test_chatgpt_requires_device_id():
    authorize: list[str] = []
    with pytest.raises(Exception, match="requires a device ID"):
        await openai_chatgpt_oauth.login(chatgpt_interaction(authorize=authorize))
    with pytest.raises(Exception, match="requires a device ID"):
        await openai_chatgpt_oauth.login(
            chatgpt_interaction(authorize=authorize), LoginOptions(get_device_id=lambda: "not-a-uuid")
        )
    assert authorize == []


def chatgpt_credential() -> dict[str, Any]:
    return {
        "type": "oauth",
        "access": "old-access",
        "refresh": "old-refresh",
        "expires": 0,
        "clientId": "oaiapp_existing",
        "scopes": REQUIRED_SCOPE.split(" "),
    }


async def test_chatgpt_refresh_requires_rotated_refresh_token(http: Http):
    chatgpt_endpoint(http, chatgpt_tokens(refresh_token=None))
    with pytest.raises(Exception, match="token response has invalid refresh_token"):
        await openai_chatgpt_oauth.refresh(chatgpt_credential(), NEVER)


async def test_chatgpt_refresh_uses_issued_client_id(http: Http, clock: Clock):
    bodies: list[dict[str, str]] = []
    chatgpt_endpoint(http, chatgpt_tokens(access_token="new-access", refresh_token="new-refresh"), bodies)
    credential = await openai_chatgpt_oauth.refresh(chatgpt_credential(), NEVER)
    # expires_in is 3600 seconds; the credential expires 3 minutes early so it is refreshed in time.
    assert credential["expires"] == START + (3600 - 180) * 1000
    assert bodies[0] == {
        "grant_type": "refresh_token",
        "client_id": "oaiapp_existing",
        "refresh_token": "old-refresh",
        "resource": "https://api.openai.com/v1",
    }
    assert credential["access"] == "new-access" and credential["refresh"] == "new-refresh"
    assert credential["clientId"] == "oaiapp_existing"


async def test_chatgpt_refresh_requires_client_id():
    with pytest.raises(Exception, match="does not contain an issued client ID"):
        await openai_chatgpt_oauth.refresh({"type": "oauth", "access": "a", "refresh": "r", "expires": 0}, NEVER)


# -- GitHub Copilot -----------------------------------------------------------------

COPILOT_TOKEN = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;"
COPILOT_MODELS_URL = "https://api.individual.githubcopilot.com/models"
COPILOT_IDS = [model.id for model in get_builtin_models("github-copilot")]


def copilot_login_handler(
    models: Callable[[], httpx.Response], policy: Callable[[str], httpx.Response] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/login/device/code"):
            return json_response(
                {
                    "device_code": "device-code",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 900,
                }
            )
        if url.endswith("/login/oauth/access_token"):
            return json_response({"access_token": "ghu_refresh_token"})
        if "/copilot_internal/v2/token" in url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        if url == COPILOT_MODELS_URL:
            return models()
        if url.startswith(f"{COPILOT_MODELS_URL}/") and url.endswith("/policy"):
            assert policy is not None, f"Unexpected policy request: {url}"
            return policy(url[len(COPILOT_MODELS_URL) + 1 : -len("/policy")])
        raise AssertionError(f"Unexpected fetch URL: {url}")

    return handler


def copilot_interaction(events: list[AuthEvent] | None = None, signal: AbortSignal | None = None):
    async def prompt(p: AuthPrompt) -> str:
        assert p.type == "text"
        assert p.message == "GitHub Enterprise URL/domain (blank for github.com)"
        return ""

    return interaction(prompt, events.append if events is not None else None, signal)


async def refresh_copilot(http: Http, data: list[Any], proxy_host: str = "proxy.individual.githubcopilot.com"):
    token = f"tid=test;exp=9999999999;proxy-ep={proxy_host};"
    models_url = f"https://{proxy_host.replace('proxy.', 'api.', 1)}/models"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/copilot_internal/v2/token" in url:
            return json_response({"token": token, "expires_at": 9999999999})
        assert url == models_url
        assert request.headers["authorization"] == f"Bearer {token}"
        assert request.headers["x-github-api-version"] == "2026-06-01"
        return json_response({"data": data})

    http.handler = handler
    return await github_copilot_oauth.refresh(
        {"type": "oauth", "access": "old-access-token", "refresh": "ghu_refresh_token", "expires": 0}, NEVER
    )


def tools(enabled: bool = True) -> dict[str, Any]:
    return {"supports": {"tool_calls": enabled}}


async def test_copilot_filters_to_picker_catalog(http: Http):
    picker, disabled, hidden = COPILOT_IDS[:3]
    credential = await refresh_copilot(
        http,
        [
            {"id": picker, "model_picker_enabled": True, "capabilities": tools()},
            {"id": disabled, "model_picker_enabled": True, "policy": {"state": "disabled"}, "capabilities": tools()},
            {"id": hidden, "model_picker_enabled": False, "policy": {"state": "enabled"}, "capabilities": tools()},
        ],
    )
    assert credential["availableModelIds"] == [picker]


async def test_copilot_falls_back_to_enabled_policy_models(http: Http):
    enabled = COPILOT_IDS[0]
    credential = await refresh_copilot(
        http,
        [
            {"id": enabled, "model_picker_enabled": False, "policy": {"state": "enabled"}, "capabilities": tools()},
            {"id": "policy-disabled", "model_picker_enabled": False, "policy": {"state": "disabled"}},
            {"id": "unconfigured-model", "model_picker_enabled": False, "capabilities": tools()},
            {
                "id": "no-tools",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": tools(False),
            },
        ],
    )
    assert credential["availableModelIds"] == [enabled]


async def test_copilot_no_policy_fallback_for_business(http: Http):
    credential = await refresh_copilot(
        http,
        [{"id": "gpt-4.1", "model_picker_enabled": False, "policy": {"state": "enabled"}, "capabilities": tools()}],
        "proxy.business.githubcopilot.com",
    )
    assert credential["availableModelIds"] == []


async def test_copilot_refresh_does_not_retry_catalog_throttling(http: Http):
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        if "/copilot_internal/v2/token" in str(request.url):
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        count += 1
        return json_response({"error": "too many requests"}, 429, {"Retry-After": "0"})

    http.handler = handler
    with pytest.raises(Exception, match="429"):
        await github_copilot_oauth.refresh({"type": "oauth", "access": "a", "refresh": "ghu", "expires": 0}, NEVER)
    assert count == 1


async def test_copilot_device_code_login(http: Http, clock: Clock):
    http.handler = copilot_login_handler(lambda: json_response({"data": []}))
    events: list[AuthEvent] = []
    credential = await github_copilot_oauth.login(copilot_interaction(events))
    assert events[0] == AuthEvent(
        type="device_code",
        user_code="ABCD-EFGH",
        verification_uri="https://github.com/login/device",
        interval_seconds=1,
        expires_in_seconds=900,
    )
    assert credential["access"] == COPILOT_TOKEN
    assert credential["refresh"] == "ghu_refresh_token"
    assert credential["availableModelIds"] == []
    assert "enterpriseUrl" not in credential
    device_request = http.requests[0]
    assert device_request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert "scope=read%3Auser" in device_request.content.decode()


async def test_copilot_enables_only_known_tool_capable_unconfigured_models(http: Http, clock: Clock):
    configured, unconfigured, no_tools = COPILOT_IDS[:3]
    catalog_requests = 0
    policy_ids: list[str] = []

    def models() -> httpx.Response:
        nonlocal catalog_requests
        catalog_requests += 1
        return json_response(
            {
                "data": [
                    {
                        "id": configured,
                        "model_picker_enabled": True,
                        "policy": {"state": "enabled"},
                        "capabilities": tools(),
                    },
                    {
                        "id": unconfigured,
                        "model_picker_enabled": True,
                        "policy": {"state": "unconfigured"},
                        "capabilities": tools(),
                    },
                    {"id": "remote-only-model", "model_picker_enabled": True, "policy": {"state": "unconfigured"}},
                    {
                        "id": no_tools,
                        "model_picker_enabled": True,
                        "policy": {"state": "unconfigured"},
                        "capabilities": tools(False),
                    },
                ]
            }
        )

    def policy(model_id: str) -> httpx.Response:
        policy_ids.append(model_id)
        return httpx.Response(200)

    http.handler = copilot_login_handler(models, policy)
    events: list[AuthEvent] = []
    credential = await github_copilot_oauth.login(copilot_interaction(events))
    assert catalog_requests == 1
    assert policy_ids == [unconfigured]
    assert AuthEvent(type="progress", message="Enabling models...") in events
    assert credential["availableModelIds"] == [configured, unconfigured, "remote-only-model"]


async def test_copilot_retries_throttled_policy_update(http: Http, clock: Clock):
    model_id = COPILOT_IDS[0]
    attempts = 0

    def policy(_: str) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return json_response({"error": "too many requests"}, 429, {"Retry-After": "1"})
        return httpx.Response(200)

    http.handler = copilot_login_handler(
        lambda: json_response(
            {"data": [{"id": model_id, "model_picker_enabled": True, "policy": {"state": "unconfigured"}}]}
        ),
        policy,
    )
    await github_copilot_oauth.login(copilot_interaction())
    assert attempts == 2
    assert clock.sleeps == [1, 1]  # device poll wait, then Retry-After


async def test_copilot_continues_after_policy_transport_failure(http: Http, clock: Clock):
    ids = COPILOT_IDS[:2]
    policy_ids: list[str] = []

    def policy(model_id: str) -> httpx.Response:
        policy_ids.append(model_id)
        if len(policy_ids) == 1:
            raise httpx.ConnectError("fetch failed")
        return httpx.Response(200)

    http.handler = copilot_login_handler(
        lambda: json_response(
            {"data": [{"id": i, "model_picker_enabled": True, "policy": {"state": "unconfigured"}} for i in ids]}
        ),
        policy,
    )
    credential = await github_copilot_oauth.login(copilot_interaction())
    assert policy_ids == ids
    assert credential["availableModelIds"] == ids


async def test_copilot_stops_policy_updates_when_retry_exceeds_budget(http: Http, clock: Clock):
    first, second = COPILOT_IDS[:2]
    policy_ids: list[str] = []

    def policy(model_id: str) -> httpx.Response:
        policy_ids.append(model_id)
        return json_response({"error": "too many requests"}, 429, {"Retry-After": "5"})

    http.handler = copilot_login_handler(
        lambda: json_response(
            {
                "data": [
                    {"id": first, "model_picker_enabled": True, "policy": {"state": "unconfigured"}},
                    {"id": second, "model_picker_enabled": True, "policy": {"state": "unconfigured"}},
                ]
            }
        ),
        policy,
    )
    store = InMemoryCredentialStore()
    models = create_models(credentials=store)
    models.set_provider(
        create_provider(id="github-copilot", auth=ProviderAuth(oauth=github_copilot_oauth), api=_NoApi())
    )
    credential = await models.login("github-copilot", "oauth", _Inner(lambda p: asyncio.sleep(0, ""), None))
    assert credential["type"] == "oauth" and credential["access"] == COPILOT_TOKEN
    assert policy_ids == [first]
    assert await store.read("github-copilot") == credential


async def test_copilot_rejects_untrusted_verification_uri(http: Http):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "device_code": "d",
                "user_code": "U",
                "verification_uri": "$(id>/tmp/pwned)",
                "interval": 1,
                "expires_in": 900,
            }
        )

    http.handler = handler
    events: list[AuthEvent] = []
    with pytest.raises(Exception, match="Untrusted verification_uri"):
        await github_copilot_oauth.login(copilot_interaction(events))
    assert events == []


async def test_copilot_normalizes_verification_uri(http: Http, clock: Clock):
    base = copilot_login_handler(lambda: json_response({"data": []}))

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/login/device/code"):
            return json_response(
                {
                    "device_code": "d",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/\x1b]8;;evil",
                    "interval": 1,
                    "expires_in": 900,
                }
            )
        return base(request)

    http.handler = handler
    events: list[AuthEvent] = []
    await github_copilot_oauth.login(copilot_interaction(events))
    assert events[0].verification_uri == "https://github.com/login/%1B]8;;evil"


async def test_copilot_waits_before_polling_and_honors_slow_down(http: Http, clock: Clock):
    poll_times: list[int] = []
    replies = [
        json_response({"error": "authorization_pending", "error_description": "pending"}),
        json_response({"error": "slow_down", "error_description": "slow down", "interval": 7}),
        json_response({"access_token": "ghu_refresh_token"}),
    ]
    base = copilot_login_handler(lambda: json_response({"data": []}))

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/login/device/code"):
            return json_response(
                {
                    "device_code": "device-code",
                    "user_code": "U",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 5,
                    "expires_in": 900,
                }
            )
        if url.endswith("/login/oauth/access_token"):
            poll_times.append(clock.now)
            body = request.content.decode()
            assert "device_code=device-code" in body
            assert "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code" in body
            return replies.pop(0)
        return base(request)

    http.handler = handler
    await github_copilot_oauth.login(copilot_interaction())
    assert poll_times == [START + 5000, START + 10000, START + 17000]


async def test_copilot_times_out_after_slow_downs(http: Http, clock: Clock):
    poll_times: list[int] = []
    replies = [
        json_response({"error": "slow_down"}),
        json_response({"error": "slow_down"}),
        json_response({"error": "authorization_pending"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/login/device/code"):
            return json_response(
                {
                    "device_code": "d",
                    "user_code": "U",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 5,
                    "expires_in": 25,
                }
            )
        poll_times.append(clock.now)
        return replies.pop(0)

    http.handler = handler
    with pytest.raises(Exception, match="Device flow timed out after one or more slow_down responses"):
        await github_copilot_oauth.login(copilot_interaction())
    assert poll_times == [START + 5000, START + 15000]


async def test_copilot_rejects_invalid_enterprise_domain():
    async def prompt(p: AuthPrompt) -> str:
        return "exa mple"

    with pytest.raises(Exception, match="Invalid GitHub Enterprise URL/domain"):
        await github_copilot_oauth.login(interaction(prompt))


# -- OpenRouter ---------------------------------------------------------------------

OPENROUTER_TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"


def callback_url_of(event: AuthEvent) -> str:
    return query(event.url or "")["callback_url"]


async def test_openrouter_loopback_pkce_login(http: Http, local: httpx.AsyncClient):
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == OPENROUTER_TOKEN_URL
        bodies.append(json.loads(request.content))
        return json_response({"key": "sk-or-test"})

    http.handler = handler
    pages: list[asyncio.Future[httpx.Response]] = []
    auth_urls: list[str] = []
    signals: list[AbortSignal | None] = []

    def notify(event: AuthEvent) -> None:
        if event.type != "auth_url":
            return
        auth_urls.append(event.url or "")
        pages.append(asyncio.ensure_future(local.get(with_query(callback_url_of(event), code="authorization-code"))))

    async def prompt(p: AuthPrompt) -> str:
        signals.append(p.signal)
        return await forever_prompt(p)

    credential = await openrouter_oauth.login(interaction(prompt, notify))
    assert credential == {"type": "oauth", "access": "sk-or-test", "refresh": "", "expires": _common.MAX_SAFE_INTEGER}
    assert (await pages[0]).status_code == 200
    assert signals[0] is not None and signals[0].aborted
    authorize = urlsplit(auth_urls[0])
    assert (authorize.scheme, authorize.netloc, authorize.path) == ("https", "openrouter.ai", "/auth")
    params = query(auth_urls[0])
    assert params["code_challenge_method"] == "S256"
    callback = urlsplit(params["callback_url"])
    assert callback.hostname == "127.0.0.1"
    assert callback.path.startswith("/oauth/callback/")
    assert bodies[0]["code"] == "authorization-code" and bodies[0]["code_challenge_method"] == "S256"
    assert params["code_challenge"] == b64url(hashlib.sha256(bodies[0]["code_verifier"].encode()).digest())
    assert len(http.requests) == 1


async def _openrouter_failed_login(http: Http, local: httpx.AsyncClient, response: httpx.Response, match: str):
    http.handler = lambda request: response
    pages: list[asyncio.Future[httpx.Response]] = []

    def notify(event: AuthEvent) -> None:
        if event.type == "auth_url":
            pages.append(asyncio.ensure_future(local.get(with_query(callback_url_of(event), code="bad-code"))))

    with pytest.raises(Exception, match=match):
        await openrouter_oauth.login(interaction(forever_prompt, notify))
    assert (await pages[0]).status_code == 502


async def test_openrouter_exchange_failure_reaches_page_and_login(http: Http, local: httpx.AsyncClient):
    await _openrouter_failed_login(
        http,
        local,
        json_response({"error": {"message": "invalid code"}}, 403),
        r"OpenRouter OAuth key exchange failed \(HTTP 403\): invalid code",
    )


async def test_openrouter_rejects_response_without_key(http: Http, local: httpx.AsyncClient):
    await _openrouter_failed_login(http, local, json_response({"user_id": "user-1"}), 'carries no "key"')


async def test_openrouter_allows_only_one_exchange(http: Http, local: httpx.AsyncClient):
    exchange: asyncio.Future[httpx.Response] = asyncio.get_running_loop().create_future()
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        return await exchange

    http.handler = handler
    urls: list[str] = []
    first: list[asyncio.Future[httpx.Response]] = []

    def notify(event: AuthEvent) -> None:
        if event.type == "auth_url":
            urls.append(with_query(callback_url_of(event), code="authorization-code"))
            first.append(asyncio.ensure_future(local.get(urls[0])))

    login = asyncio.ensure_future(openrouter_oauth.login(interaction(forever_prompt, notify)))
    await started.wait()
    assert (await local.get(urls[0])).status_code == 409
    assert len(http.requests) == 1
    exchange.set_result(json_response({"key": "sk-or-test"}))
    assert (await login)["access"] == "sk-or-test"
    assert (await first[0]).status_code == 200


async def test_openrouter_pasted_redirect_and_bare_code(http: Http):
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return json_response({"key": "sk-or-manual"})

    http.handler = handler
    callback: list[str] = []

    async def pasted(p: AuthPrompt) -> str:
        assert p.type == "manual_code"
        return f"{callback[0]}?code=manual-code"

    credential = await openrouter_oauth.login(
        interaction(pasted, lambda e: e.type == "auth_url" and callback.append(callback_url_of(e)))
    )
    assert credential["access"] == "sk-or-manual"
    assert bodies[0]["code"] == "manual-code"

    async def bare(p: AuthPrompt) -> str:
        return "  manual-code  "

    await openrouter_oauth.login(interaction(bare))
    assert bodies[1]["code"] == "manual-code"
    assert len(http.requests) == 2


async def test_openrouter_manual_prompt_cancel_and_empty_input(http: Http):
    async def cancel(p: AuthPrompt) -> str:
        raise RuntimeError("Login cancelled")

    with pytest.raises(RuntimeError, match="Login cancelled"):
        await openrouter_oauth.login(interaction(cancel))

    async def blank(p: AuthPrompt) -> str:
        return "   "

    with pytest.raises(Exception, match="Missing authorization code"):
        await openrouter_oauth.login(interaction(blank))
    assert http.requests == []


async def test_openrouter_closes_pending_callback_on_cancel(local: httpx.AsyncClient):
    controller = AbortController()
    callback: list[str] = []

    def notify(event: AuthEvent) -> None:
        if event.type == "auth_url":
            callback.append(callback_url_of(event))
            controller.abort()

    with pytest.raises(AbortError, match="Login cancelled"):
        await openrouter_oauth.login(interaction(forever_prompt, notify, controller.signal))
    with pytest.raises(httpx.HTTPError):
        await local.get(callback[0])


async def test_openrouter_rejects_already_cancelled_login():
    def notify(event: AuthEvent) -> None:
        raise AssertionError("Cancelled login must not emit events")

    with pytest.raises(AbortError, match="Login cancelled"):
        await openrouter_oauth.login(interaction(forever_prompt, notify, AbortSignal.abort()))


async def test_openrouter_uses_configured_callback_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PI_OAUTH_CALLBACK_HOST", "localhost")
    controller = AbortController()
    callback: list[str] = []

    def notify(event: AuthEvent) -> None:
        if event.type == "auth_url":
            callback.append(callback_url_of(event))
            controller.abort()

    with pytest.raises(AbortError, match="Login cancelled"):
        await openrouter_oauth.login(interaction(forever_prompt, notify, controller.signal))
    assert urlsplit(callback[0]).hostname == "localhost"


# -- Kimi Code ----------------------------------------------------------------------

KIMI_CLIENT = "17e5f671-d194-4dfb-9706-5516cb48c098"
KIMI_HOST = "https://auth.kimi.com"


def kimi_device(**overrides: Any) -> httpx.Response:
    return json_response(
        {
            "user_code": "ABCD-1234",
            "device_code": "device-code-123",
            "verification_uri": "https://www.kimi.com/code",
            "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
            "interval": 5,
            "expires_in": 600,
            **overrides,
        }
    )


async def test_kimi_device_login(http: Http, clock: Clock):
    poll_times: list[int] = []
    polls = [
        json_response({"error": "authorization_pending"}, 400),
        json_response({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{KIMI_HOST}/api/oauth/device_authorization":
            assert request.headers["accept"] == "application/json"
            assert form(request) == {"client_id": KIMI_CLIENT}
            return kimi_device()
        assert url == f"{KIMI_HOST}/api/oauth/token"
        poll_times.append(clock.now)
        assert form(request) == {
            "client_id": KIMI_CLIENT,
            "device_code": "device-code-123",
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        return polls.pop(0)

    http.handler = handler
    events: list[AuthEvent] = []
    credential = await kimi_coding_oauth.login(interaction(None, events.append))
    assert events == [
        AuthEvent(
            type="device_code",
            user_code="ABCD-1234",
            verification_uri="https://www.kimi.com/code?user_code=ABCD-1234",
            interval_seconds=5,
            expires_in_seconds=600,
        )
    ]
    assert poll_times == [START + 5000, START + 10000]
    assert credential == {
        "type": "oauth",
        "access": "access-token",
        "refresh": "refresh-token",
        "expires": START + 10000 + 3600 * 1000,
    }


@pytest.mark.parametrize(("error", "match"), [("expired_token", "expired"), ("access_denied", "denied")])
async def test_kimi_device_failures(http: Http, clock: Clock, error: str, match: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/device_authorization"):
            return kimi_device()
        return json_response({"error": error}, 400)

    http.handler = handler
    with pytest.raises(Exception, match=match):
        await kimi_coding_oauth.login(interaction())


async def test_kimi_honors_host_override(http: Http, clock: Clock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KIMI_CODE_OAUTH_HOST", "https://auth.example.com/")

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/device_authorization"):
            return kimi_device(interval=1)
        return json_response({"access_token": "a", "refresh_token": "r", "expires_in": 60})

    http.handler = handler
    credential = await kimi_coding_oauth.login(interaction())
    assert credential["access"] == "a" and credential["refresh"] == "r"
    assert http.urls == [
        "https://auth.example.com/api/oauth/device_authorization",
        "https://auth.example.com/api/oauth/token",
    ]


async def test_kimi_refresh(http: Http, clock: Clock):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{KIMI_HOST}/api/oauth/token"
        assert form(request) == {
            "client_id": KIMI_CLIENT,
            "grant_type": "refresh_token",
            "refresh_token": "old-refresh",
        }
        return json_response({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    http.handler = handler
    credential = await kimi_coding_oauth.refresh(
        {"type": "oauth", "access": "old-access", "refresh": "old-refresh", "expires": 0}, NEVER
    )
    assert credential == {
        "type": "oauth",
        "access": "new-access",
        "refresh": "new-refresh",
        "expires": START + 3600_000,
    }
    assert await kimi_coding_oauth.to_auth(credential) == ModelAuth(headers={"Authorization": "Bearer new-access"})


async def test_kimi_refresh_retries_429_and_fails_on_invalid_grant(http: Http, clock: Clock):
    replies = [
        json_response({"error": "temporarily_unavailable"}, 429),
        json_response({"access_token": "a", "refresh_token": "r", "expires_in": 60}),
    ]
    http.handler = lambda request: replies.pop(0)
    credential = await kimi_coding_oauth.refresh({"type": "oauth", "access": "o", "refresh": "o", "expires": 0}, NEVER)
    assert credential["access"] == "a"
    assert clock.sleeps == [1]

    http.handler = lambda request: json_response({"error": "invalid_grant"}, 400)
    with pytest.raises(Exception, match="unauthorized"):
        await kimi_coding_oauth.refresh({"type": "oauth", "access": "o", "refresh": "o", "expires": 0}, NEVER)


# -- Meta ---------------------------------------------------------------------------

META_CLIENT = "1031625952748946"
META_DEVICE_URL = "https://auth.meta.com/oidc/device/authorization/"
META_TOKEN_URL = "https://auth.meta.com/oidc/device/token/"
META_MINT_URL = "https://api.meta.ai/muse-code/key"
DAY_MS = 24 * 60 * 60 * 1000


async def test_meta_device_login_mints_api_key(http: Http, clock: Clock):
    polls = [
        json_response({"error": "authorization_pending"}, 400),
        json_response({"access_token": "identity-token", "token_type": "Bearer"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == META_DEVICE_URL:
            assert form(request) == {"client_id": META_CLIENT}
            return json_response(
                {
                    "device_code": "device-code-123",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://auth.meta.com/oauth/device/",
                    "verification_uri_complete": "https://auth.meta.com/oauth/device/?code=ABCD-1234",
                    "interval": 5,
                    "expires_in": 600,
                }
            )
        if url == META_TOKEN_URL:
            assert form(request) == {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": "device-code-123",
                "client_id": META_CLIENT,
            }
            return polls.pop(0)
        assert url == META_MINT_URL
        assert request.headers["authorization"] == "Bearer identity-token"
        assert request.headers["x-api-version"] == "1.0.0"
        return json_response({"api_key": "LLM|minted-key"})

    http.handler = handler
    events: list[AuthEvent] = []
    credential = await meta_oauth.login(interaction(None, events.append))
    assert events[0] == AuthEvent(
        type="device_code",
        user_code="ABCD-1234",
        verification_uri="https://auth.meta.com/oauth/device/?code=ABCD-1234",
        interval_seconds=5,
        expires_in_seconds=600,
    )
    assert credential == {
        "type": "oauth",
        "refresh": "identity-token",
        "access": "LLM|minted-key",
        "expires": START + 10000 + DAY_MS,
    }


async def test_meta_refresh_re_mints(http: Http, clock: Clock):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == META_MINT_URL
        assert request.headers["authorization"] == "Bearer identity-token"
        return json_response({"api_key": "LLM|fresh-key"})

    http.handler = handler
    credential = await meta_oauth.refresh(
        {"type": "oauth", "refresh": "identity-token", "access": "LLM|old", "expires": 1}, NEVER
    )
    assert credential == {
        "type": "oauth",
        "refresh": "identity-token",
        "access": "LLM|fresh-key",
        "expires": START + DAY_MS,
    }


async def test_meta_reports_setup_url_and_expired_session(http: Http):
    http.handler = lambda request: json_response({"require_payment": True, "action_url": "https://dev.meta.ai/billing"})
    with pytest.raises(Exception, match="Complete setup at https://dev.meta.ai/billing"):
        await meta_oauth.refresh({"type": "oauth", "refresh": "t", "access": "", "expires": 1}, NEVER)
    http.handler = lambda request: json_response({"detail": "expired"}, 401)
    with pytest.raises(Exception, match=r"Meta session expired \(status 401\).*: expired"):
        await meta_oauth.refresh({"type": "oauth", "refresh": "t", "access": "", "expires": 1}, NEVER)


# -- xAI ----------------------------------------------------------------------------

XAI_CLIENT = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_DEVICE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"


def xai_device(**overrides: Any) -> dict[str, Any]:
    return {
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://accounts.x.ai/oauth2/device",
        "expires_in": 900,
        "interval": 5,
        **overrides,
    }


def xai_tokens(**overrides: Any) -> dict[str, Any]:
    body = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 21_600,
        "token_type": "Bearer",
    }
    body.update(overrides)
    return {key: value for key, value in body.items() if value is not None}


async def test_xai_device_grant_with_pending_and_slow_down(http: Http, clock: Clock):
    poll_times: list[int] = []
    replies = [
        json_response({"error": "authorization_pending"}, 400),
        json_response({"error": "slow_down", "interval": 10}, 400),
        json_response(xai_tokens()),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == XAI_DEVICE_URL:
            assert form(request) == {
                "client_id": XAI_CLIENT,
                "scope": "openid profile email offline_access grok-cli:access api:access",
                "referrer": "pi",
            }
            return json_response(xai_device())
        assert url == XAI_TOKEN_URL
        poll_times.append(clock.now)
        assert form(request) == {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": XAI_CLIENT,
            "device_code": "device-code",
        }
        return replies.pop(0)

    http.handler = handler
    events: list[AuthEvent] = []
    credential = await xai_oauth.login(interaction(None, events.append))
    assert events == [
        AuthEvent(
            type="device_code",
            user_code="ABCD-1234",
            verification_uri="https://accounts.x.ai/oauth2/device",
            interval_seconds=5,
            expires_in_seconds=900,
        )
    ]
    assert poll_times == [START + 5000, START + 10_000, START + 20_000]
    assert credential == {
        "type": "oauth",
        "access": "access-token",
        "refresh": "refresh-token",
        "expires": START + 20_000 + 21_600_000 - 300_000,
    }


async def test_xai_interval_zero_uses_default(http: Http, clock: Clock):
    poll_times: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == XAI_DEVICE_URL:
            return json_response(xai_device(interval=0))
        poll_times.append(clock.now)
        return json_response(xai_tokens())

    http.handler = handler
    await xai_oauth.login(interaction())
    assert poll_times == [START + 5000]


async def test_xai_prefers_verification_uri_complete(http: Http, clock: Clock):
    complete = "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == XAI_DEVICE_URL:
            return json_response(xai_device(verification_uri_complete=complete))
        return json_response(xai_tokens())

    http.handler = handler
    events: list[AuthEvent] = []
    await xai_oauth.login(interaction(None, events.append))
    assert events[0].verification_uri == complete


@pytest.mark.parametrize(
    "overrides",
    [
        {"verification_uri_complete": "http://accounts.x.ai/oauth2/device?user_code=ABCD-1234"},
        {"verification_uri": "http://accounts.x.ai/oauth2/device"},
        {"verification_uri": "file:///etc/passwd"},
        {"verification_uri": "not a url"},
    ],
)
async def test_xai_rejects_untrusted_verification_uri(http: Http, overrides: dict[str, str]):
    http.handler = lambda request: json_response(xai_device(**overrides))
    with pytest.raises(Exception, match="Untrusted verification URI"):
        await xai_oauth.login(interaction())


@pytest.mark.parametrize("error", ["access_denied", "authorization_denied"])
async def test_xai_denied(http: Http, clock: Clock, error: str):
    replies = [json_response(xai_device(interval=1)), json_response({"error": error}, 400)]
    http.handler = lambda request: replies.pop(0)
    with pytest.raises(Exception, match="xAI device authorization was denied"):
        await xai_oauth.login(interaction())


async def test_xai_cancel_while_waiting_for_first_poll(http: Http, clock: Clock):
    controller = AbortController()
    http.handler = lambda request: json_response(xai_device())
    with pytest.raises(AbortError, match="Login cancelled"):
        await xai_oauth.login(interaction(None, lambda e: controller.abort(), controller.signal))
    assert len(http.requests) == 1


async def test_xai_refresh_rotation_and_preservation(http: Http):
    replies = [
        json_response(xai_tokens(access_token="new-access", refresh_token="new-refresh")),
        json_response(xai_tokens(access_token="newer-access", refresh_token=None)),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert form(request)["grant_type"] == "refresh_token"
        return replies.pop(0)

    http.handler = handler
    rotated = await xai_oauth.refresh({"type": "oauth", "access": "a", "refresh": "old-refresh", "expires": 0}, NEVER)
    preserved = await xai_oauth.refresh(
        {"type": "oauth", "access": "a", "refresh": "keep-refresh", "expires": 0}, NEVER
    )
    assert (rotated["access"], rotated["refresh"]) == ("new-access", "new-refresh")
    assert (preserved["access"], preserved["refresh"]) == ("newer-access", "keep-refresh")
    assert form(http.requests[1])["refresh_token"] == "keep-refresh"
    assert xai_oauth.name == "xAI (Grok/X subscription)"


async def test_xai_refresh_defaults_and_errors(http: Http, clock: Clock):
    http.handler = lambda request: json_response(xai_tokens(expires_in=None))
    credential = await xai_oauth.refresh({"type": "oauth", "access": "a", "refresh": "r", "expires": 0}, NEVER)
    assert credential["expires"] == START + 3_600_000 - 300_000

    http.handler = lambda request: json_response(xai_tokens(access_token=None))
    with pytest.raises(Exception, match="Invalid xAI OAuth response field: access_token"):
        await xai_oauth.refresh({"type": "oauth", "access": "a", "refresh": "r", "expires": 0}, NEVER)

    http.handler = lambda request: json_response(
        {"error": "invalid_grant", "error_description": "refresh token revoked"}, 400
    )
    with pytest.raises(
        Exception, match=r"xAI OAuth token refresh failed \(HTTP 400\): invalid_grant: refresh token revoked"
    ):
        await xai_oauth.refresh({"type": "oauth", "access": "a", "refresh": "r", "expires": 0}, NEVER)


# -- Radius -------------------------------------------------------------------------

GATEWAY = "https://radius.example"


def radius_interaction(method: str, events: list[AuthEvent] | None = None, notify: Any = None):
    async def prompt(p: AuthPrompt) -> str:
        assert p.type == "select" and p.message == "Sign in to Radius:"
        return method

    return interaction(prompt, notify or (events.append if events is not None else None))


async def test_radius_device_login_uses_gateway_endpoints(http: Http, clock: Clock):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = form(request)
        if url == f"{GATEWAY}/v1/oauth/device":
            assert body == {"client_id": "pi-gateway", "scope": "gateway offline_access"}
            return json_response(
                {
                    "device_code": "device-code",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://radius-ui.example/pair",
                    "expires_in": 600,
                    "interval": 5,
                }
            )
        assert url == f"{GATEWAY}/v1/oauth/token"
        assert body == {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": "pi-gateway",
            "device_code": "device-code",
        }
        return json_response(
            {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "scope": "gateway offline_access",
            }
        )

    http.handler = handler
    events: list[AuthEvent] = []
    oauth = create_radius_oauth(name="Radius", gateway=GATEWAY)
    credential = await oauth.login(radius_interaction("device-code", events))
    assert credential == {
        "type": "oauth",
        "access": "access-token",
        "refresh": "refresh-token",
        "expires": START + 3600 * 1000 - 60_000,
        "scope": "gateway offline_access",
    }
    assert events == [
        AuthEvent(
            type="device_code",
            user_code="ABCD-1234",
            verification_uri="https://radius-ui.example/pair",
            interval_seconds=5,
            expires_in_seconds=600,
        )
    ]
    assert http.urls == [f"{GATEWAY}/v1/oauth/device", f"{GATEWAY}/v1/oauth/token"]


async def test_radius_device_login_pending_then_complete(http: Http, clock: Clock):
    replies = [
        json_response({"device_code": "d", "user_code": "U", "verification_uri": "https://x/pair", "expires_in": 600}),
        json_response({"error": "authorization_pending"}, 400),
        json_response({"access_token": "a", "refresh_token": "r", "expires_in": 60}),
    ]
    http.handler = lambda request: replies.pop(0)
    oauth = create_radius_oauth(name="Radius", gateway="radius.example/")
    credential = await oauth.login(radius_interaction("device-code"))
    assert credential["access"] == "a"
    assert clock.sleeps == [5]


async def test_radius_refresh_without_discovery(http: Http):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{GATEWAY}/v1/oauth/token"
        assert form(request) == {
            "grant_type": "refresh_token",
            "client_id": "pi-gateway",
            "refresh_token": "old-refresh",
        }
        return json_response({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    http.handler = handler
    oauth = create_radius_oauth(name="Radius", gateway=GATEWAY)
    credential = await oauth.refresh({"type": "oauth", "access": "old", "refresh": "old-refresh", "expires": 0}, NEVER)
    assert credential["access"] == "new-access" and credential["refresh"] == "new-refresh"
    assert "scope" not in credential
    assert len(http.requests) == 1


async def test_radius_discovers_only_browser_endpoint(http: Http):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{GATEWAY}/v1/oauth"
        return json_response({"issuer": "https://radius-ui.example"})

    http.handler = handler
    oauth = create_radius_oauth(name="Radius", gateway=GATEWAY)
    with pytest.raises(Exception, match=f"Invalid Radius OAuth config from {GATEWAY}"):
        await oauth.login(radius_interaction("browser"))
    assert len(http.requests) == 1


async def test_radius_browser_callback_exchanges_before_page(http: Http, local: httpx.AsyncClient):
    token_status = 400

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{GATEWAY}/v1/oauth":
            return json_response({"authorizationEndpoint": f"{GATEWAY}/authorize"})
        assert url == f"{GATEWAY}/v1/oauth/token"
        assert form(request)["code"] == "browser-code"
        if token_status == 200:
            return json_response({"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})
        return json_response({"error": "invalid_grant", "error_description": "code expired"}, token_status)

    http.handler = handler
    oauth = create_radius_oauth(name="Radius", gateway=GATEWAY)

    async def login() -> dict[str, Any]:
        pages: list[asyncio.Future[httpx.Response]] = []

        def notify(event: AuthEvent) -> None:
            if event.type != "auth_url":
                return
            params = query(event.url or "")
            assert event.url and event.url.startswith(f"{GATEWAY}/authorize?")
            assert params["handoff"] == "url"
            url = with_query(params["redirect_uri"], code="browser-code", state=params["state"])
            pages.append(asyncio.ensure_future(local.get(url)))

        outcome: dict[str, Any] = {}
        try:
            outcome["credential"] = await oauth.login(radius_interaction("browser", notify=notify))
        except Exception as error:
            outcome["error"] = error
        page = await pages[0]
        return {**outcome, "status": page.status_code, "page": page.text}

    failed = await login()
    assert "invalid_grant: code expired" in str(failed["error"])
    assert failed["status"] == 502
    assert "code expired" in failed["page"]

    token_status = 200
    succeeded = await login()
    assert succeeded["credential"]["access"] == "access"
    assert succeeded["status"] == 200
    assert "Signed in to Radius." in succeeded["page"]
