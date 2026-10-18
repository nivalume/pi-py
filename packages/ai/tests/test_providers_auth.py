"""Provider-owned auth: Anthropic auth token / federation, Bedrock, Vertex, Cloudflare, OAuth metadata, filters.

Ports the provider-level parts of upstream ``anthropic-auth-token.test.ts``,
``anthropic-federation.test.ts``, ``bedrock-credentials.test.ts``,
``google-vertex-api-key-resolution.test.ts`` and ``classifier-models.test.ts``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from pi_ai import (
    AbortController,
    AbortError,
    AbortSignal,
    AssistantMessageEventStream,
    AuthEvent,
    AuthResult,
    Context,
    DoneEvent,
    InMemoryCredentialStore,
    Model,
    ModelAuth,
    ModelCost,
    StartEvent,
    UserMessage,
    create_models,
    create_provider,
    faux_assistant_message,
    normalize_context,
    now_ms,
)
from pi_ai.auth.helpers import LazyOAuthAuth
from pi_ai.auth.types import AuthPrompt
from pi_ai.providers.all import builtin_models, builtin_providers
from pi_ai.providers.amazon_bedrock import amazon_bedrock_provider
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.cloudflare_auth import cloudflare_ai_gateway_auth, cloudflare_workers_ai_auth
from pi_ai.providers.github_copilot import github_copilot_provider
from pi_ai.providers.google_vertex import google_vertex_provider
from pi_ai.providers.openai import openai_provider


class FakeAuthContext:
    def __init__(self, env: Mapping[str, str] | None = None, files: list[str] | None = None) -> None:
        self._env = dict(env or {})
        self._files = list(files or [])
        self.reads: list[str] = []

    async def env(self, name: str) -> str | None:
        self.reads.append(name)
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return path in self._files


class ScriptedInteraction:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.signal = AbortSignal()
        self.events: list[AuthEvent] = []
        self.prompts: list[AuthPrompt] = []

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)


NEVER_ABORTED = AbortSignal()
ADC_PATH = "~/.config/gcloud/application_default_credentials.json"
CONTEXT = normalize_context(Context(messages=[UserMessage("Hello")], system_prompt="System prompt."))


async def _resolve(
    provider: Any, env: Mapping[str, str], credential: Any = None, files: list[str] | None = None
) -> Any:
    return await provider.auth.api_key.resolve(
        ctx=FakeAuthContext(env, files), credential=credential, signal=NEVER_ABORTED
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

FEDERATION_ENV = {
    "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
    "ANTHROPIC_ORGANIZATION_ID": "org-test",
    "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_test",
    "ANTHROPIC_IDENTITY_TOKEN_FILE": "/tmp/identity.jwt",
}


async def test_anthropic_resolves_auth_token_as_a_bearer_authorization_header() -> None:
    result = await _resolve(
        anthropic_provider(),
        {"ANTHROPIC_AUTH_TOKEN": "auth-token", "ANTHROPIC_OAUTH_TOKEN": "oauth-token", "ANTHROPIC_API_KEY": "api-key"},
    )
    assert result == AuthResult(
        ModelAuth(headers={"Authorization": "Bearer auth-token"}), source="ANTHROPIC_AUTH_TOKEN"
    )


async def test_anthropic_preserves_oauth_token_as_oauth_shaped_api_auth() -> None:
    result = await _resolve(
        anthropic_provider(), {"ANTHROPIC_OAUTH_TOKEN": "oauth-token", "ANTHROPIC_API_KEY": "api-key"}
    )
    assert result == AuthResult(ModelAuth(api_key="oauth-token"), source="ANTHROPIC_OAUTH_TOKEN")


async def test_anthropic_falls_back_to_the_api_key() -> None:
    result = await _resolve(anthropic_provider(), {"ANTHROPIC_API_KEY": "api-key"})
    assert result == AuthResult(ModelAuth(api_key="api-key"), source="ANTHROPIC_API_KEY")


async def test_anthropic_stored_key_wins_over_env() -> None:
    result = await _resolve(
        anthropic_provider(), {"ANTHROPIC_AUTH_TOKEN": "auth-token"}, {"type": "api_key", "key": "stored"}
    )
    assert result == AuthResult(ModelAuth(api_key="stored"), source="stored credential")


async def test_anthropic_resolves_federation_variables_as_provider_env_with_no_request_auth() -> None:
    assert await _resolve(anthropic_provider(), FEDERATION_ENV) == AuthResult(
        ModelAuth(), env=FEDERATION_ENV, source="workload identity federation"
    )


async def test_anthropic_passes_workspace_id_through_when_set() -> None:
    result = await _resolve(anthropic_provider(), {**FEDERATION_ENV, "ANTHROPIC_WORKSPACE_ID": "wrkspc_test"})
    assert result is not None and result.env is not None
    assert result.env["ANTHROPIC_WORKSPACE_ID"] == "wrkspc_test"


async def test_anthropic_is_not_configured_when_a_federation_variable_is_missing() -> None:
    partial = {key: value for key, value in FEDERATION_ENV.items() if key != "ANTHROPIC_IDENTITY_TOKEN_FILE"}
    assert await _resolve(anthropic_provider(), partial) is None
    assert await _resolve(anthropic_provider(), {}) is None


async def test_anthropic_treats_the_service_account_id_as_optional() -> None:
    partial = {key: value for key, value in FEDERATION_ENV.items() if key != "ANTHROPIC_SERVICE_ACCOUNT_ID"}
    assert await _resolve(anthropic_provider(), partial) == AuthResult(
        ModelAuth(), env=partial, source="workload identity federation"
    )


async def test_anthropic_keeps_api_key_and_auth_token_precedence_over_federation() -> None:
    assert await _resolve(anthropic_provider(), {**FEDERATION_ENV, "ANTHROPIC_API_KEY": "api-key"}) == AuthResult(
        ModelAuth(api_key="api-key"), source="ANTHROPIC_API_KEY"
    )
    assert await _resolve(anthropic_provider(), {**FEDERATION_ENV, "ANTHROPIC_AUTH_TOKEN": "auth-token"}) == (
        AuthResult(ModelAuth(headers={"Authorization": "Bearer auth-token"}), source="ANTHROPIC_AUTH_TOKEN")
    )


async def test_anthropic_resolve_honors_an_aborted_signal() -> None:
    controller = AbortController()
    controller.abort()
    auth = anthropic_provider().auth.api_key
    assert auth is not None
    with pytest.raises(AbortError):
        await auth.resolve(ctx=FakeAuthContext({"ANTHROPIC_API_KEY": "k"}), credential=None, signal=controller.signal)


async def test_anthropic_login_prompts_for_a_secret() -> None:
    auth = anthropic_provider().auth.api_key
    assert auth is not None and auth.login is not None
    interaction = ScriptedInteraction(["sk-ant"])
    assert await auth.login(interaction) == {"type": "api_key", "key": "sk-ant"}  # type: ignore[arg-type]
    assert interaction.prompts[0].type == "secret"
    assert interaction.prompts[0].message == "Enter Anthropic API key"


class _CapturingStreams:
    def __init__(self) -> None:
        self.options: list[dict[str, Any]] = []
        self.models: list[Model] = []

    def _respond(self, model: Model, options: Mapping[str, Any] | None) -> AssistantMessageEventStream:
        self.models.append(model)
        self.options.append(dict(options or {}))
        stream = AssistantMessageEventStream()
        message = faux_assistant_message("ok")
        stream.push(StartEvent(message))
        stream.push(DoneEvent("stop", message))
        stream.end(message)
        return stream

    def stream(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        return self._respond(model, options)

    def stream_simple(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        return self._respond(model, options)


def _anthropic_with(streams: _CapturingStreams) -> Any:
    real = anthropic_provider()
    return create_provider(id="anthropic", auth=real.auth, models=list(real.get_models()), api=streams)


ANTHROPIC_MODEL = Model(
    id="claude-test",
    api="anthropic-messages",
    provider="anthropic",
    base_url="https://api.anthropic.com",
    cost=ModelCost(),
    context_window=100000,
    max_tokens=4096,
)


async def test_anthropic_threads_auth_context_auth_token_through_request_headers() -> None:
    streams = _CapturingStreams()
    models = create_models(auth_context=FakeAuthContext({"ANTHROPIC_AUTH_TOKEN": "ctx-token"}))
    models.set_provider(_anthropic_with(streams))

    await models.stream_simple(ANTHROPIC_MODEL, CONTEXT).result()

    assert "api_key" not in streams.options[0]
    assert streams.options[0]["headers"] == {"Authorization": "Bearer ctx-token"}


async def test_anthropic_explicit_request_headers_override_the_auth_token() -> None:
    streams = _CapturingStreams()
    models = create_models(auth_context=FakeAuthContext({"ANTHROPIC_AUTH_TOKEN": "ctx-token"}))
    models.set_provider(_anthropic_with(streams))

    await models.stream_simple(ANTHROPIC_MODEL, CONTEXT, headers={"Authorization": "Bearer explicit-token"}).result()

    assert streams.options[0]["headers"] == {"Authorization": "Bearer explicit-token"}


async def test_anthropic_threads_federation_variables_through_models() -> None:
    streams = _CapturingStreams()
    models = create_models(auth_context=FakeAuthContext(FEDERATION_ENV))
    models.set_provider(_anthropic_with(streams))

    await models.stream_simple(ANTHROPIC_MODEL, CONTEXT).result()

    assert "api_key" not in streams.options[0]
    assert streams.options[0]["env"] == FEDERATION_ENV


# ---------------------------------------------------------------------------
# Amazon Bedrock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "source"),
    [
        ({"AWS_BEARER_TOKEN_BEDROCK": "token"}, "AWS_BEARER_TOKEN_BEDROCK"),
        ({"AWS_PROFILE": "dev"}, "AWS_PROFILE"),
        ({"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "secret"}, "AWS access keys"),
        ({"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/creds"}, "ECS task role"),
        ({"AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://169.254.170.2/creds"}, "ECS task role"),
        ({"AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token"}, "web identity token"),
    ],
)
async def test_bedrock_detects_ambient_aws_credential_sources(env: dict[str, str], source: str) -> None:
    result = await _resolve(amazon_bedrock_provider(), env)
    assert result == AuthResult(ModelAuth(), source=source)


async def test_bedrock_requires_both_access_key_parts() -> None:
    assert await _resolve(amazon_bedrock_provider(), {"AWS_ACCESS_KEY_ID": "AKIA"}) is None


async def test_bedrock_bearer_token_precedes_profiles_and_access_keys() -> None:
    result = await _resolve(
        amazon_bedrock_provider(),
        {"AWS_BEARER_TOKEN_BEDROCK": "t", "AWS_PROFILE": "p", "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"},
    )
    assert result is not None and result.source == "AWS_BEARER_TOKEN_BEDROCK"


async def test_bedrock_profile_precedes_ambient_access_keys() -> None:
    result = await _resolve(
        amazon_bedrock_provider(), {"AWS_PROFILE": "ambient", "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"}
    )
    assert result is not None and result.source == "AWS_PROFILE"


async def test_bedrock_stored_key_and_stored_profile() -> None:
    stored_key = await _resolve(
        amazon_bedrock_provider(), {"AWS_PROFILE": "dev"}, {"type": "api_key", "key": "bedrock-token"}
    )
    assert stored_key == AuthResult(ModelAuth(api_key="bedrock-token"), source="stored credential")

    stored_profile = await _resolve(
        amazon_bedrock_provider(), {"AWS_PROFILE": "ambient"}, {"type": "api_key", "env": {"AWS_PROFILE": "work"}}
    )
    assert stored_profile == AuthResult(ModelAuth(), env={"AWS_PROFILE": "work"}, source="stored credential")

    # A credential-chain login stores an empty api_key credential: ambient sources still apply.
    chain = await _resolve(amazon_bedrock_provider(), {"AWS_PROFILE": "dev"}, {"type": "api_key"})
    assert chain == AuthResult(ModelAuth(), source="AWS_PROFILE")


async def test_bedrock_credential_chain_login_and_unknown_methods() -> None:
    auth = amazon_bedrock_provider().auth.api_key
    assert auth is not None and auth.login is not None
    interaction = ScriptedInteraction(["credential-chain", ""])
    assert await auth.login(interaction) == {"type": "api_key"}  # type: ignore[arg-type]
    assert interaction.prompts[0].type == "select"
    assert [option.id for option in interaction.prompts[0].options or []] == [
        "bearer-token",
        "aws-profile",
        "credential-chain",
    ]
    assert interaction.prompts[1].message == "Configure AWS credentials, then press Enter to continue"

    with pytest.raises(ValueError, match="Unknown Amazon Bedrock auth method: nope"):
        await auth.login(ScriptedInteraction(["nope"]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Google Vertex
# ---------------------------------------------------------------------------


async def test_vertex_prefers_a_stored_key_then_the_env_key() -> None:
    assert await _resolve(
        google_vertex_provider(), {"GOOGLE_CLOUD_API_KEY": "env-key"}, {"type": "api_key", "key": "stored-key"}
    ) == AuthResult(ModelAuth(api_key="stored-key"), source="stored credential")
    assert await _resolve(google_vertex_provider(), {"GOOGLE_CLOUD_API_KEY": "env-key"}) == AuthResult(
        ModelAuth(api_key="env-key"), source="GOOGLE_CLOUD_API_KEY"
    )


async def test_vertex_uses_gcloud_project_and_an_explicit_credentials_file() -> None:
    result = await _resolve(
        google_vertex_provider(),
        {
            "GCLOUD_PROJECT": "proj",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_APPLICATION_CREDENTIALS": "/keys/sa.json",
        },
        files=["/keys/sa.json"],
    )
    assert result == AuthResult(ModelAuth(), source="gcloud application default credentials")

    # The explicit credentials file must exist; the default ADC path is not consulted.
    missing = await _resolve(
        google_vertex_provider(),
        {
            "GOOGLE_CLOUD_PROJECT": "proj",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_APPLICATION_CREDENTIALS": "/keys/missing.json",
        },
        files=[ADC_PATH],
    )
    assert missing is None


async def test_vertex_stored_service_account_credential() -> None:
    credential = {
        "type": "api_key",
        "env": {
            "GOOGLE_CLOUD_PROJECT": "project-id",
            "GOOGLE_CLOUD_LOCATION": "europe-west4",
            "GOOGLE_APPLICATION_CREDENTIALS": "/keys/sa.json",
        },
    }
    result = await _resolve(google_vertex_provider(), {}, credential, files=["/keys/sa.json"])
    assert result == AuthResult(ModelAuth(), env=credential["env"], source="stored credential")


async def test_vertex_service_account_login_flow() -> None:
    auth = google_vertex_provider().auth.api_key
    assert auth is not None and auth.login is not None
    interaction = ScriptedInteraction(["service-account", "project-id", "us-east5", "/keys/sa.json"])
    assert await auth.login(interaction) == {  # type: ignore[arg-type]
        "type": "api_key",
        "env": {
            "GOOGLE_CLOUD_PROJECT": "project-id",
            "GOOGLE_CLOUD_LOCATION": "us-east5",
            "GOOGLE_APPLICATION_CREDENTIALS": "/keys/sa.json",
        },
    }
    assert interaction.events[0].message == "Provide a service account credentials file, project, and location."

    with pytest.raises(ValueError, match="Unknown Google Vertex AI auth method: other"):
        await auth.login(ScriptedInteraction(["other"]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cloudflare
# ---------------------------------------------------------------------------


async def test_cloudflare_merges_credential_fields_with_ambient_env_per_field() -> None:
    auth = cloudflare_workers_ai_auth()
    # A credential carrying only the key still picks up the account id from the environment.
    result = await auth.resolve(
        ctx=FakeAuthContext({"CLOUDFLARE_API_KEY": "env-key", "CLOUDFLARE_ACCOUNT_ID": "env-account"}),
        credential={"type": "api_key", "key": "stored-key"},
        signal=NEVER_ABORTED,
    )
    assert result == AuthResult(
        ModelAuth(api_key="stored-key"), env={"CLOUDFLARE_ACCOUNT_ID": "env-account"}, source="stored credential"
    )

    gateway = await cloudflare_ai_gateway_auth().resolve(
        ctx=FakeAuthContext({"CLOUDFLARE_GATEWAY_ID": "env-gateway"}),
        credential={"type": "api_key", "key": "stored-key", "env": {"CLOUDFLARE_ACCOUNT_ID": "stored-account"}},
        signal=NEVER_ABORTED,
    )
    assert gateway is not None
    assert gateway.env == {"CLOUDFLARE_ACCOUNT_ID": "stored-account", "CLOUDFLARE_GATEWAY_ID": "env-gateway"}
    assert gateway.source == "stored credential"


async def test_cloudflare_workers_ai_does_not_read_the_gateway_id() -> None:
    ctx = FakeAuthContext({"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_GATEWAY_ID": "g"})
    result = await cloudflare_workers_ai_auth().resolve(ctx=ctx, credential=None, signal=NEVER_ABORTED)
    assert result is not None and result.env == {"CLOUDFLARE_ACCOUNT_ID": "a"}
    assert result.source == "CLOUDFLARE_API_KEY"
    assert "CLOUDFLARE_GATEWAY_ID" not in ctx.reads


async def test_cloudflare_login_flows_store_account_and_gateway_ids() -> None:
    workers = cloudflare_workers_ai_auth()
    assert workers.login is not None
    assert await workers.login(ScriptedInteraction(["key", "account"])) == {  # type: ignore[arg-type]
        "type": "api_key",
        "key": "key",
        "env": {"CLOUDFLARE_ACCOUNT_ID": "account"},
    }
    gateway = cloudflare_ai_gateway_auth()
    assert gateway.login is not None
    assert await gateway.login(ScriptedInteraction(["key", "account", "gateway"])) == {  # type: ignore[arg-type]
        "type": "api_key",
        "key": "key",
        "env": {"CLOUDFLARE_ACCOUNT_ID": "account", "CLOUDFLARE_GATEWAY_ID": "gateway"},
    }


# ---------------------------------------------------------------------------
# OAuth metadata and credential-dependent model filters
# ---------------------------------------------------------------------------


def test_builtin_oauth_methods_advertise_upstream_metadata() -> None:
    expected = {
        "anthropic": ("Anthropic (Claude Pro/Max)", True, None),
        "github-copilot": ("GitHub Copilot", True, None),
        "kimi-coding": ("Kimi Code (subscription)", True, "Sign in with Kimi Code"),
        "meta": ("Meta (Muse subscription)", True, "Sign in with Meta"),
        "openai": ("OpenAI (ChatGPT subscription)", True, "Sign in with ChatGPT"),
        "openai-codex": ("OpenAI (ChatGPT Plus/Pro)", True, None),
        "openrouter": ("OpenRouter OAuth", False, "Sign in with OpenRouter"),
        "radius": ("Radius", False, None),
        "xai": ("xAI (Grok/X subscription)", True, "Sign in with SuperGrok or X Premium"),
    }
    actual = {
        provider.id: (provider.auth.oauth.name, provider.auth.oauth.is_subscription, provider.auth.oauth.login_label)
        for provider in builtin_providers()
        if provider.auth.oauth is not None
    }
    assert actual == expected
    for provider in builtin_providers():
        assert provider.auth.api_key is not None or provider.auth.oauth is not None
        if provider.auth.oauth is not None:
            # Flow implementations load lazily.
            assert isinstance(provider.auth.oauth, LazyOAuthAuth)
    codex = next(provider for provider in builtin_providers() if provider.id == "openai-codex")
    assert codex.auth.api_key is None


def _oauth_credential(**extra: Any) -> dict[str, Any]:
    return {"type": "oauth", "access": "access", "refresh": "refresh", "expires": now_ms() + 3_600_000, **extra}


async def test_lists_openai_decisions_models_only_for_api_key_credentials() -> None:
    api_key_store = InMemoryCredentialStore({"openai": {"type": "api_key", "key": "secret"}})
    oauth_store = InMemoryCredentialStore({"openai": _oauth_credential()})

    with_api_key = builtin_models(credentials=api_key_store)
    with_oauth = builtin_models(credentials=oauth_store)

    assert [model.id for model in await with_api_key.get_available_of_type("classifier", "openai")] == ["gpt-6-luna"]
    assert await with_oauth.get_available_of_type("classifier", "openai") == []
    # Chat models stay available with ChatGPT OAuth.
    assert any(model.id == "gpt-6-luna" for model in await with_oauth.get_available("openai"))


def test_openai_filter_all_models_hides_classifiers_under_oauth() -> None:
    provider = openai_provider()
    assert provider.filter_all_models is not None and provider.get_all_models is not None
    all_models = list(provider.get_all_models())
    assert any(model.type == "classifier" for model in all_models)
    assert provider.filter_all_models(all_models, None) == all_models
    assert provider.filter_all_models(all_models, {"type": "api_key", "key": "k"}) == all_models
    filtered = provider.filter_all_models(all_models, _oauth_credential())
    assert filtered and all(model.type != "classifier" for model in filtered)


def test_github_copilot_filters_models_by_the_oauth_plan() -> None:
    provider = github_copilot_provider()
    assert provider.filter_models is not None
    catalog = list(provider.get_models())
    chosen = [catalog[0].id, catalog[1].id]
    assert provider.filter_models(catalog, None) == catalog
    assert provider.filter_models(catalog, {"type": "api_key", "key": "k"}) == catalog
    assert provider.filter_models(catalog, _oauth_credential()) == catalog
    assert provider.filter_models(catalog, _oauth_credential(availableModelIds=[chosen[0], 3])) == catalog
    assert [
        model.id for model in provider.filter_models(catalog, _oauth_credential(availableModelIds=chosen))
    ] == chosen


async def test_github_copilot_available_models_follow_the_stored_plan() -> None:
    catalog = list(github_copilot_provider().get_models())
    store = InMemoryCredentialStore({"github-copilot": _oauth_credential(availableModelIds=[catalog[2].id])})
    models = builtin_models(credentials=store)
    assert [model.id for model in await models.get_available("github-copilot")] == [catalog[2].id]
