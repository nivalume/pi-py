"""Port of upstream ``test/providers.test.ts``: built-in providers, env-key auth, create_provider dispatch."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Mapping
from typing import Any

import pytest

from pi_ai import (
    AbortSignal,
    AssistantMessageEventStream,
    AuthEvent,
    AuthResult,
    Context,
    DeferredHandle,
    DoneEvent,
    InMemoryModelsStore,
    Model,
    ModelAuth,
    ModelCost,
    ProviderAuth,
    StartEvent,
    TextContent,
    UserMessage,
    create_models,
    create_provider,
    env_api_key_auth,
    faux_assistant_message,
    faux_provider,
    get_supported_thinking_levels,
    lazy_api,
    normalize_context,
)
from pi_ai.auth.types import ApiKeyAuth, AuthPrompt
from pi_ai.compat import get_model as get_compat_model
from pi_ai.compat import get_models as get_compat_models
from pi_ai.providers.all import (
    builtin_models,
    builtin_providers,
    get_all_builtin_models,
    get_builtin_classifier_model,
    get_builtin_classifier_models,
    get_builtin_image_model,
    get_builtin_image_models,
    get_builtin_model,
    get_builtin_models,
    get_builtin_providers,
)
from pi_ai.providers.amazon_bedrock import amazon_bedrock_provider
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.cloudflare_ai_gateway import cloudflare_ai_gateway_provider
from pi_ai.providers.cloudflare_workers_ai import cloudflare_workers_ai_provider
from pi_ai.providers.google_vertex import google_vertex_provider


class FakeAuthContext:
    def __init__(self, env: Mapping[str, str] | None = None, files: list[str] | None = None) -> None:
        self._env = dict(env or {})
        self._files = list(files or [])

    async def env(self, name: str) -> str | None:
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

CONTEXT = normalize_context(Context(messages=[UserMessage("hi")]))

DEFAULT_IMAGE_RESIZE = {"max_width": 2000, "max_height": 2000, "max_bytes": int(4.5 * 1024 * 1024), "jpeg_quality": 80}


def _resize_dict(model: Any) -> dict[str, Any] | None:
    limits = model.input_limits
    if limits is None or limits.images is None or limits.images.resize is None:
        return None
    resize = limits.images.resize
    return {
        "max_width": resize.max_width,
        "max_height": resize.max_height,
        "max_bytes": resize.max_bytes,
        "jpeg_quality": resize.jpeg_quality,
    }


# ---------------------------------------------------------------------------
# builtin providers
# ---------------------------------------------------------------------------


def test_builtin_models_registers_every_builtin_provider_with_models() -> None:
    models = builtin_models()
    providers = models.get_providers()
    assert len(providers) == len(builtin_providers())
    assert "anthropic" in [provider.id for provider in providers]

    anthropic = models.get_model("anthropic", "claude-haiku-4-5")
    assert anthropic is not None and anthropic.api == "anthropic-messages"

    assert len(models.get_models()) > 500

    for provider in providers:
        listed = models.get_all_models(provider.id)
        assert listed, provider.id
        assert all(model.provider == provider.id for model in listed)
    radius = get_builtin_model("radius", "balanced")
    assert radius is not None and radius.api == "pi-messages" and radius.provider == "radius"


def test_builtin_provider_ids_match_the_generated_catalog() -> None:
    assert sorted(provider.id for provider in builtin_providers()) == sorted(get_builtin_providers())


def test_returns_empty_results_for_unknown_provider_ids() -> None:
    assert get_builtin_model("not-a-provider", "x") is None
    assert get_builtin_image_model("not-a-provider", "x") is None
    assert get_builtin_classifier_model("not-a-provider", "x") is None
    assert get_builtin_models("not-a-provider") == []
    assert get_builtin_image_models("not-a-provider") == []
    assert get_builtin_classifier_models("not-a-provider") == []
    assert get_all_builtin_models("not-a-provider") == []
    assert get_compat_model("not-a-provider", "x") is None
    assert get_compat_models("not-a-provider") == []


def test_stores_native_constrained_sampling_capabilities_in_model_metadata() -> None:
    gpt4o = get_builtin_model("openai", "gpt-4o")
    assert gpt4o is not None and gpt4o.compat is not None
    assert gpt4o.compat.get("supportsStrictMode") is True
    assert "supportsOpenAIGrammarTools" not in gpt4o.compat
    gpt54 = get_builtin_model("openai", "gpt-5.4")
    assert gpt54 is not None and gpt54.compat is not None
    assert gpt54.compat.get("supportsStrictMode") is True
    assert gpt54.compat.get("supportsOpenAIGrammarTools") is True
    haiku = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert haiku is not None and haiku.compat is not None
    assert haiku.compat.get("supportsStrictTools") is True


def test_keeps_the_conservative_resize_profile_on_every_vision_model() -> None:
    vision_models = [
        model
        for provider in get_builtin_providers()
        for model in get_builtin_models(provider)
        if "image" in model.input
    ]
    assert vision_models
    for model in vision_models:
        assert _resize_dict(model) == DEFAULT_IMAGE_RESIZE, f"{model.provider}/{model.id}"


def test_records_known_direct_provider_image_request_limits() -> None:
    haiku = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert haiku is not None and haiku.input_limits is not None and haiku.input_limits.images is not None
    assert haiku.input_limits.max_request_bytes == 32 * 1024 * 1024
    assert haiku.input_limits.images.max_per_request == 100
    opus5 = get_builtin_model("anthropic", "claude-opus-5")
    assert opus5 is not None and opus5.input_limits is not None and opus5.input_limits.images is not None
    assert opus5.input_limits.images.max_per_request == 600
    bedrock = get_builtin_model("amazon-bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0")
    assert bedrock is not None and bedrock.input_limits is not None and bedrock.input_limits.images is not None
    assert bedrock.input_limits.images.max_per_message == 20
    gpt4o = get_builtin_model("openai", "gpt-4o")
    assert gpt4o is not None and gpt4o.input_limits is not None and gpt4o.input_limits.images is not None
    assert gpt4o.input_limits.max_request_bytes == 512 * 1024 * 1024
    assert gpt4o.input_limits.images.max_per_request == 1500
    flash = get_builtin_model("google", "gemini-2.5-flash")
    assert flash is not None and flash.input_limits is not None and flash.input_limits.images is not None
    assert flash.input_limits.max_request_bytes == 20 * 1024 * 1024
    assert flash.input_limits.images.max_per_request == 3600


def test_does_not_infer_image_limits_from_gateway_api_compatibility() -> None:
    model = next(model for model in get_builtin_models("openrouter") if "image" in model.input)
    assert model.input_limits is not None and model.input_limits.images is not None
    assert model.input_limits.max_request_bytes is None
    assert model.input_limits.images.max_per_request is None
    assert model.input_limits.images.max_per_message is None
    assert _resize_dict(model) == DEFAULT_IMAGE_RESIZE


def _levels(provider: str, model_id: str) -> list[str]:
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"{provider}/{model_id}"
    return list(get_supported_thinking_levels(model))


def test_uses_models_dev_effort_levels_for_google_thinking_models() -> None:
    # Regression test for https://github.com/earendil-works/pi/issues/9455
    for provider in ("google", "google-vertex"):
        assert "minimal" in _levels(provider, "gemini-3.6-flash")
        assert _levels(provider, "gemini-3.8-flash") == ["low", "medium", "high"]
        assert _levels(provider, "gemini-3.1-pro-preview") == ["low", "medium", "high"]
    assert _levels("opencode", "gemini-3.8-flash") == ["low", "medium", "high"]
    assert _levels("google", "gemma-4-31b-it") == ["minimal", "high"]


MID_CONVO_SUPPORTED = [
    ("moonshotai", "kimi-k2.6"),
    ("moonshotai", "kimi-k2.7-code"),
    ("moonshotai", "kimi-k2.7-code-highspeed"),
    ("moonshotai", "kimi-k3"),
    ("moonshotai-cn", "kimi-k2.6"),
    ("moonshotai-cn", "kimi-k2.7-code"),
    ("moonshotai-cn", "kimi-k2.7-code-highspeed"),
    ("moonshotai-cn", "kimi-k3"),
    ("fireworks", "accounts/fireworks/models/kimi-k3"),
    ("fireworks", "accounts/fireworks/routers/kimi-k3-fast"),
    ("openai", "gpt-5.4"),
    ("openai", "gpt-5.5"),
    ("openai", "gpt-6-astra"),
    ("openai-codex", "gpt-5.5"),
    ("anthropic", "claude-opus-5"),
    ("opencode", "gpt-5.4"),
    ("opencode", "gpt-5.6-terra"),
    ("opencode-go", "gpt-5.6-luna"),
    ("opencode", "claude-opus-4-8"),
    ("opencode", "claude-opus-5"),
    ("opencode", "kimi-k3"),
    ("opencode-go", "kimi-k3"),
    ("github-copilot", "gpt-5.6-terra"),
    ("github-copilot", "claude-opus-5"),
    ("github-copilot", "claude-opus-4.8"),
    ("github-copilot", "kimi-k3"),
    ("deepseek", "deepseek-v4-pro"),
    ("openrouter", "openai/gpt-5.6-terra"),
]

MID_CONVO_UNSUPPORTED = [
    ("fireworks", "accounts/fireworks/models/nemotron-3-ultra-nvfp4"),
    ("openai", "gpt-4.1"),
    ("openai", "gpt-5.2"),
    ("anthropic", "claude-sonnet-4-5"),
    ("google", "gemini-2.5-pro"),
    ("opencode", "gpt-5.2"),
    ("opencode", "claude-sonnet-4-5"),
    ("github-copilot", "claude-sonnet-4.6"),
    ("deepseek", "deepseek-flash"),
    ("openrouter", "anthropic/claude-opus-5"),
    ("openrouter", "moonshotai/kimi-k3"),
    ("openrouter", "openai/gpt-5.6-terra:batch"),
]


def test_enables_mid_conversation_system_messages_only_for_verified_models() -> None:
    models = builtin_models()
    for provider, model_id in MID_CONVO_SUPPORTED:
        model = models.get_model(provider, model_id)
        assert model is not None, f"{provider}/{model_id}"
        assert (model.compat or {}).get("supportsMidConvoSystemMessages") is True, f"{provider}/{model_id}"
    for provider, model_id in MID_CONVO_UNSUPPORTED:
        model = models.get_model(provider, model_id)
        assert model is not None, f"{provider}/{model_id}"
        assert "supportsMidConvoSystemMessages" not in (model.compat or {}), f"{provider}/{model_id}"


def _compat(models: Any, provider: str, model_id: str) -> dict[str, Any]:
    model = models.get_model(provider, model_id)
    assert model is not None, f"{provider}/{model_id}"
    return model.compat or {}


def test_routes_proxied_tool_changes_through_verified_transports_only() -> None:
    models = builtin_models()
    for provider, model_id in (("opencode", "gpt-5.6-terra"), ("github-copilot", "gpt-5.6-terra")):
        # Proxies pass `additional_tools` through to OpenAI but are not verified for tool search.
        compat = _compat(models, provider, model_id)
        assert compat.get("supportsAdditionalTools") is True
        assert "supportsToolSearch" not in compat
    # Proxied Anthropic endpoints reject `tool_addition`/`tool_removal` blocks.
    for provider in ("opencode", "github-copilot"):
        assert "supportsMidConvoToolChanges" not in _compat(models, provider, "claude-opus-5")
    assert _compat(models, "anthropic", "claude-opus-5").get("supportsMidConvoToolChanges") is True
    # Kimi-style tool-bearing system messages survive Moonshot and OpenCode but not Copilot.
    for provider in ("moonshotai", "moonshotai-cn", "opencode", "opencode-go"):
        assert _compat(models, provider, "kimi-k3").get("supportsMidConvoToolAdditions") is True
    for provider in ("moonshotai", "moonshotai-cn"):
        for model_id in ("kimi-k2.6", "kimi-k2.7-code", "kimi-k2.7-code-highspeed"):
            assert "supportsMidConvoToolAdditions" not in _compat(models, provider, model_id)
    assert "supportsMidConvoToolAdditions" not in _compat(models, "github-copilot", "kimi-k3")
    assert "supportsMidConvoToolAdditions" not in _compat(models, "openrouter", "openai/gpt-5.6-terra")


def test_uses_official_kimi_k3_pricing_for_moonshot_providers() -> None:
    models = builtin_models()
    for provider in ("moonshotai", "moonshotai-cn"):
        model = models.get_model(provider, "kimi-k3")
        assert model is not None
        assert model.cost == ModelCost(input=3, output=15, cache_read=0.3, cache_write=0)


def test_uses_api_equivalent_implied_pricing_for_kimi_coding_subscription_models() -> None:
    models = builtin_models()
    expected = {
        "k3": ModelCost(input=3, output=15, cache_read=0.3, cache_write=0),
        "kimi-for-coding-highspeed": ModelCost(input=1.9, output=8, cache_read=0.38, cache_write=0),
    }
    for model_id, cost in expected.items():
        model = models.get_model("kimi-coding", model_id)
        assert model is not None and model.cost == cost


async def test_resolves_anthropic_bearer_auth_from_env_with_auth_token_precedence() -> None:
    models = create_models(
        auth_context=FakeAuthContext(
            {
                "ANTHROPIC_AUTH_TOKEN": "auth-token",
                "ANTHROPIC_OAUTH_TOKEN": "oauth-token",
                "ANTHROPIC_API_KEY": "api-key",
            }
        )
    )
    models.set_provider(anthropic_provider())

    assert await models.get_auth("anthropic") == AuthResult(
        ModelAuth(headers={"Authorization": "Bearer auth-token"}), source="ANTHROPIC_AUTH_TOKEN"
    )


async def test_preserves_anthropic_oauth_token_precedence_over_the_api_key() -> None:
    models = create_models(
        auth_context=FakeAuthContext({"ANTHROPIC_API_KEY": "key", "ANTHROPIC_OAUTH_TOKEN": "oauth-token"})
    )
    models.set_provider(anthropic_provider())

    result = await models.get_auth("anthropic")
    assert result is not None
    assert result.auth.api_key == "oauth-token"
    assert result.source == "ANTHROPIC_OAUTH_TOKEN"


async def test_runs_provider_owned_bedrock_bearer_token_and_aws_profile_login_flows() -> None:
    auth = amazon_bedrock_provider().auth.api_key
    assert auth is not None and auth.login is not None
    assert await auth.login(ScriptedInteraction(["bearer-token", "bedrock-token"])) == {  # type: ignore[arg-type]
        "type": "api_key",
        "key": "bedrock-token",
    }

    profile = ScriptedInteraction(["aws-profile", "work"])
    assert await auth.login(profile) == {"type": "api_key", "env": {"AWS_PROFILE": "work"}}  # type: ignore[arg-type]
    assert len(profile.events) == 1
    assert profile.events[0].type == "info"
    assert [link.label for link in profile.events[0].links or []] == ["AWS credential provider chain"]
    result = await auth.resolve(
        ctx=FakeAuthContext({}),
        credential={"type": "api_key", "env": {"AWS_PROFILE": "work"}},
        signal=NEVER_ABORTED,
    )
    assert result is not None
    assert result.auth == ModelAuth()
    assert result.env == {"AWS_PROFILE": "work"}


async def test_reports_bedrock_as_configured_from_ambient_aws_credentials_without_an_api_key() -> None:
    models = create_models(auth_context=FakeAuthContext({"AWS_PROFILE": "dev"}))
    models.set_provider(amazon_bedrock_provider())
    model = models.get_models("amazon-bedrock")[0]

    result = await models.get_auth(model.provider)
    assert result is not None
    assert result.auth == ModelAuth()
    assert result.source == "AWS_PROFILE"

    unconfigured = create_models(auth_context=FakeAuthContext({}))
    unconfigured.set_provider(amazon_bedrock_provider())
    assert await unconfigured.get_auth(model.provider) is None


async def test_requires_cloudflare_workers_ai_account_config_and_returns_scoped_env() -> None:
    missing_account = create_models(auth_context=FakeAuthContext({"CLOUDFLARE_API_KEY": "cf-key"}))
    missing_account.set_provider(cloudflare_workers_ai_provider())
    model = missing_account.get_models("cloudflare-workers-ai")[0]
    assert await missing_account.get_auth(model.provider) is None

    configured = create_models(
        auth_context=FakeAuthContext({"CLOUDFLARE_API_KEY": "cf-key", "CLOUDFLARE_ACCOUNT_ID": "account-id"})
    )
    configured.set_provider(cloudflare_workers_ai_provider())
    result = await configured.get_auth(model.provider)
    assert result is not None
    assert result.auth == ModelAuth(api_key="cf-key")
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "account-id"}


async def test_requires_cloudflare_ai_gateway_account_and_gateway_config_and_returns_scoped_env_headers() -> None:
    missing_gateway = create_models(
        auth_context=FakeAuthContext({"CLOUDFLARE_API_KEY": "cf-key", "CLOUDFLARE_ACCOUNT_ID": "account-id"})
    )
    missing_gateway.set_provider(cloudflare_ai_gateway_provider())
    model = missing_gateway.get_models("cloudflare-ai-gateway")[0]
    assert await missing_gateway.get_auth(model.provider) is None

    configured = create_models(
        auth_context=FakeAuthContext(
            {
                "CLOUDFLARE_API_KEY": "cf-key",
                "CLOUDFLARE_ACCOUNT_ID": "account-id",
                "CLOUDFLARE_GATEWAY_ID": "gateway-id",
            }
        )
    )
    configured.set_provider(cloudflare_ai_gateway_provider())
    result = await configured.get_auth(model.provider)
    assert result is not None
    assert result.auth == ModelAuth(
        headers={"cf-aig-authorization": "Bearer cf-key", "Authorization": None, "x-api-key": None}
    )
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "account-id", "CLOUDFLARE_GATEWAY_ID": "gateway-id"}


async def test_runs_provider_owned_vertex_api_key_and_adc_login_flows() -> None:
    auth = google_vertex_provider().auth.api_key
    assert auth is not None and auth.login is not None
    assert await auth.login(ScriptedInteraction(["api-key", "vertex-key"])) == {  # type: ignore[arg-type]
        "type": "api_key",
        "key": "vertex-key",
    }

    adc = ScriptedInteraction(["adc", "project-id", "us-central1"])
    assert await auth.login(adc) == {  # type: ignore[arg-type]
        "type": "api_key",
        "env": {"GOOGLE_CLOUD_PROJECT": "project-id", "GOOGLE_CLOUD_LOCATION": "us-central1"},
    }
    assert len(adc.events) == 1
    assert adc.events[0].type == "info"
    assert [link.label for link in adc.events[0].links or []] == ["Application Default Credentials"]
    result = await auth.resolve(
        ctx=FakeAuthContext({}, ["~/.config/gcloud/application_default_credentials.json"]),
        credential={
            "type": "api_key",
            "env": {"GOOGLE_CLOUD_PROJECT": "project-id", "GOOGLE_CLOUD_LOCATION": "us-central1"},
        },
        signal=NEVER_ABORTED,
    )
    assert result is not None
    assert result.auth == ModelAuth()
    assert result.env == {"GOOGLE_CLOUD_PROJECT": "project-id", "GOOGLE_CLOUD_LOCATION": "us-central1"}


async def test_resolves_vertex_via_adc_file_plus_project_and_location() -> None:
    adc = "~/.config/gcloud/application_default_credentials.json"
    configured = create_models(
        auth_context=FakeAuthContext({"GOOGLE_CLOUD_PROJECT": "proj", "GOOGLE_CLOUD_LOCATION": "us-central1"}, [adc])
    )
    configured.set_provider(google_vertex_provider())
    model = configured.get_models("google-vertex")[0]

    result = await configured.get_auth(model.provider)
    assert result is not None
    assert result.auth == ModelAuth()
    assert result.source is not None and "application default" in result.source

    # ADC without project/location is not configured
    partial = create_models(auth_context=FakeAuthContext({"GOOGLE_CLOUD_PROJECT": "proj"}, [adc]))
    partial.set_provider(google_vertex_provider())
    assert await partial.get_auth(model.provider) is None

    # explicit key wins over ADC
    keyed = create_models(auth_context=FakeAuthContext({"GOOGLE_CLOUD_API_KEY": "vertex-key"}))
    keyed.set_provider(google_vertex_provider())
    keyed_result = await keyed.get_auth(model.provider)
    assert keyed_result is not None and keyed_result.auth.api_key == "vertex-key"


# ---------------------------------------------------------------------------
# env_api_key_auth
# ---------------------------------------------------------------------------


async def test_env_api_key_auth_prefers_the_stored_credential_key_and_falls_back_through_env_vars() -> None:
    auth = env_api_key_auth("Test key", ["FIRST_KEY", "SECOND_KEY"])

    stored = await auth.resolve(
        ctx=FakeAuthContext({"FIRST_KEY": "env"}),
        credential={"type": "api_key", "key": "stored"},
        signal=NEVER_ABORTED,
    )
    assert stored is not None
    assert stored.auth.api_key == "stored"
    assert stored.source == "stored credential"

    second = await auth.resolve(ctx=FakeAuthContext({"SECOND_KEY": "second"}), credential=None, signal=NEVER_ABORTED)
    assert second is not None
    assert second.auth.api_key == "second"
    assert second.source == "SECOND_KEY"

    assert await auth.resolve(ctx=FakeAuthContext({}), credential=None, signal=NEVER_ABORTED) is None


async def test_env_api_key_auth_login_prompts_for_a_secret_and_returns_an_api_key_credential() -> None:
    auth = env_api_key_auth("Test key", ["TEST_KEY"])
    interaction = ScriptedInteraction(["entered-key"])
    assert auth.login is not None
    credential = await auth.login(interaction)  # type: ignore[arg-type]
    assert interaction.prompts[0].type == "secret"
    assert credential == {"type": "api_key", "key": "entered-key"}


# ---------------------------------------------------------------------------
# create_provider
# ---------------------------------------------------------------------------


def _completed_stream() -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = faux_assistant_message("ok")
    stream.push(StartEvent(message))
    stream.push(DoneEvent("stop", message))
    stream.end(message)
    return stream


class RecordingStreams:
    def __init__(self, label: str, calls: list[str]) -> None:
        self.label = label
        self.calls = calls

    def _respond(self, model: Model) -> AssistantMessageEventStream:
        self.calls.append(f"{self.label}:{model.id}")
        return _completed_stream()

    def stream(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        return self._respond(model)

    def stream_simple(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        return self._respond(model)


def _test_model(api: str, model_id: str, provider: str = "mixed") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10000,
        max_tokens=1000,
    )


def _static_auth(result: AuthResult | None = None) -> ProviderAuth:
    async def resolve(**_: Any) -> AuthResult | None:
        return result if result is not None else AuthResult(ModelAuth())

    return ProviderAuth(api_key=ApiKeyAuth("Test", resolve))


async def test_lazily_exposes_only_declared_deferred_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = 0
    streams = RecordingStreams("deferred", [])

    def fetch_deferred(model: Model, handle: DeferredHandle, options: Any = None) -> Any:
        return streams.stream_simple(model, CONTEXT)

    module = types.ModuleType("pi_ai_test_fake_deferred_api")
    module.stream = streams.stream  # type: ignore[attr-defined]
    module.stream_simple = streams.stream_simple  # type: ignore[attr-defined]
    module.fetch_deferred = fetch_deferred  # type: ignore[attr-defined]

    import importlib

    original_import = importlib.import_module

    def counting_import(name: str, package: str | None = None) -> Any:
        nonlocal loads
        if name == module.__name__:
            loads += 1
            return module
        return original_import(name, package)

    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(importlib, "import_module", counting_import)

    api = lazy_api(module.__name__, fetch_deferred=True)
    model = _test_model("api-a", "model-a")
    handle = DeferredHandle(model.provider, model.id, model.api, "response-1")

    assert loads == 0
    assert api.cancel_deferred is None
    assert api.fetch_deferred is not None
    assert (await api.fetch_deferred(model, handle).result()).stop_reason == "stop"
    assert loads == 1


async def test_dispatches_on_model_api_for_mixed_api_providers() -> None:
    calls: list[str] = []
    provider = create_provider(
        id="mixed",
        auth=_static_auth(),
        models=[_test_model("api-a", "model-a"), _test_model("api-b", "model-b")],
        api={"api-a": RecordingStreams("a", calls), "api-b": RecordingStreams("b", calls)},
    )
    models = create_models()
    models.set_provider(provider)

    await models.complete_simple(_test_model("api-a", "model-a"), CONTEXT)
    await models.complete_simple(_test_model("api-b", "model-b"), CONTEXT)
    assert calls == ["a:model-a", "b:model-b"]


async def test_merges_provider_resolved_env_into_stream_options() -> None:
    captured: dict[str, Any] = {}
    env_model = _test_model("api-a", "model-a", provider="env-provider")

    class CapturingStreams(RecordingStreams):
        def stream_simple(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
            captured["env"] = (options or {}).get("env")
            captured["api_key"] = (options or {}).get("api_key")
            return super().stream_simple(model, context, options)

    provider = create_provider(
        id="env-provider",
        auth=_static_auth(
            AuthResult(ModelAuth(api_key="provider-key"), env={"PROVIDER_ONLY": "provider", "SHARED": "provider"})
        ),
        models=[env_model],
        api=CapturingStreams("a", []),
    )
    models = create_models()
    models.set_provider(provider)

    await models.complete_simple(
        env_model, CONTEXT, {"api_key": "request-key", "env": {"REQUEST_ONLY": "request", "SHARED": "request"}}
    )

    assert captured["api_key"] == "request-key"
    assert captured["env"] == {"PROVIDER_ONLY": "provider", "REQUEST_ONLY": "request", "SHARED": "request"}


async def test_applies_resolved_request_options_to_deferred_fetch_and_cancellation() -> None:
    fetched: dict[str, Any] = {}
    cancelled: dict[str, Any] = {}
    deferred_model = _test_model("api-a", "model-a", provider="deferred-provider")

    class DeferredStreams(RecordingStreams):
        def fetch_deferred(self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None) -> Any:
            fetched["model"] = model
            fetched["options"] = dict(options or {})
            return self.stream_simple(model, CONTEXT)

        async def cancel_deferred(
            self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None
        ) -> None:
            cancelled["options"] = dict(options or {})

    provider = create_provider(
        id="deferred-provider",
        auth=_static_auth(
            AuthResult(
                ModelAuth(
                    api_key="provider-key",
                    base_url="https://resolved.test/v1",
                    headers={"Authorization": "Bearer provider", "X-Shared": "provider"},
                ),
                env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
            )
        ),
        models=[deferred_model],
        api=DeferredStreams("deferred", []),
    )
    models = create_models()
    models.set_provider(provider)
    handle = DeferredHandle(deferred_model.provider, deferred_model.id, deferred_model.api, "response-1")

    await models.fetch_deferred(
        deferred_model,
        handle,
        {
            "wait": 50,
            "timeout_ms": 100,
            "api_key": "request-key",
            "headers": {"X-Request": "request", "x-shared": "request"},
            "env": {"REQUEST_ONLY": "request", "SHARED": "request"},
            "transform_headers": lambda headers: {**headers, "X-Transformed": "yes"},
        },
    )
    await models.cancel_deferred(
        deferred_model,
        handle,
        {"timeout_ms": 200, "transform_headers": lambda headers: {**headers, "X-Cancel": "yes"}},
    )

    assert fetched["model"].base_url == "https://resolved.test/v1"
    options = fetched["options"]
    assert options["wait"] == 50
    assert options["timeout_ms"] == 100
    assert options["api_key"] == "request-key"
    assert options["headers"] == {
        "Authorization": "Bearer provider",
        "X-Request": "request",
        "x-shared": "request",
        "X-Transformed": "yes",
    }
    assert options["env"] == {"PROVIDER_ONLY": "provider", "REQUEST_ONLY": "request", "SHARED": "request"}
    cancel_options = cancelled["options"]
    assert cancel_options["timeout_ms"] == 200
    assert cancel_options["api_key"] == "provider-key"
    assert cancel_options["headers"] == {"Authorization": "Bearer provider", "X-Shared": "provider", "X-Cancel": "yes"}
    assert cancel_options["env"] == {"PROVIDER_ONLY": "provider", "SHARED": "provider"}


async def test_produces_a_stream_error_for_a_model_whose_api_has_no_implementation() -> None:
    provider = create_provider(
        id="mixed",
        auth=_static_auth(),
        models=[_test_model("api-a", "model-a")],
        api={"api-a": RecordingStreams("a", [])},
    )
    result = await provider.stream_simple(_test_model("api-ghost", "model-x"), CONTEXT).result()
    assert result.stop_reason == "error"
    assert result.error_message is not None and "no API implementation" in result.error_message


async def test_lets_a_newer_dynamic_refresh_bypass_and_supersede_older_network_work() -> None:
    fetches = 0
    first_started = asyncio.Event()
    first_blocked = asyncio.Event()

    async def fetch_models(_context: Any) -> list[Model]:
        nonlocal fetches
        fetches += 1
        current = fetches
        if current == 1:
            first_started.set()
            await first_blocked.wait()
        return [_test_model("api-a", f"listed-{current}", provider="dynamic")]

    provider = create_provider(
        id="dynamic",
        auth=_static_auth(),
        models=[],
        fetch_models=fetch_models,
        api=RecordingStreams("a", []),
    )
    store = InMemoryModelsStore()
    models = create_models(models_store=store)
    models.set_provider(provider)
    assert list(provider.get_models()) == []

    first = asyncio.ensure_future(models.refresh(providers=["dynamic"]))
    await first_started.wait()
    second = await models.refresh(providers=["dynamic"])
    assert second.aborted is False
    assert (await first).aborted is False
    assert fetches == 2
    assert [model.id for model in provider.get_models()] == ["listed-2"]
    stored = await store.read("dynamic")
    assert stored is not None and [model.id for model in stored.models] == ["listed-2"]

    first_blocked.set()
    await asyncio.sleep(0)
    assert [model.id for model in provider.get_models()] == ["listed-2"]
    stored = await store.read("dynamic")
    assert stored is not None and [model.id for model in stored.models] == ["listed-2"]


# ---------------------------------------------------------------------------
# faux_provider
# ---------------------------------------------------------------------------


async def test_faux_streams_queued_responses_through_a_models_collection() -> None:
    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)
    faux.set_responses([faux_assistant_message("hello from faux")])

    model = models.get_models(faux.provider.id)[0]
    result = await models.complete_simple(model, CONTEXT)
    assert result.stop_reason == "stop"
    assert result.content == [TextContent("hello from faux")]
    assert faux.state.call_count == 1


async def test_faux_submits_polls_and_redeems_deferred_responses() -> None:
    faux = faux_provider(deferred={"pending_fetches": 1, "poll_after_ms": 25})
    models = create_models()
    models.set_provider(faux.provider)
    faux.set_responses([faux_assistant_message("ready")])
    model = faux.get_model()

    submission = models.stream_simple(model, CONTEXT, {"deferred": {"window": "1h"}})
    event_types = [event.type async for event in submission]
    deferred = await submission.result()
    assert event_types == ["start", "done"]
    assert deferred.stop_reason == "deferred"
    assert deferred.content == []
    assert deferred.deferred is not None
    assert deferred.deferred.provider == model.provider
    assert deferred.deferred.model_id == model.id
    assert deferred.deferred.api == model.api
    assert isinstance(deferred.deferred.id, str)
    assert deferred.deferred.poll_after_ms == 25

    pending = await models.fetch_deferred(model, deferred.deferred)
    assert pending.stop_reason == "deferred"
    assert pending.deferred == deferred.deferred

    ready = await models.fetch_deferred(model, deferred.deferred, {"wait": 0})
    assert ready.stop_reason == "stop"
    assert ready.content == [TextContent("ready")]
    assert ready.usage.total_tokens > 0
    assert faux.state.call_count == 1
    assert faux.state.deferred_fetch_count == 2


async def test_faux_records_cancellation_and_returns_deferred_fetch_failures_in_band() -> None:
    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)

    async def failing(*_: Any) -> Any:
        raise RuntimeError("deferred failed")

    faux.set_responses([failing, faux_assistant_message("cancelled")])
    model = faux.get_model()

    failed_submission = await models.complete_simple(model, CONTEXT, {"deferred": True})
    assert failed_submission.deferred is not None
    failed = await models.fetch_deferred(model, failed_submission.deferred)
    assert failed.stop_reason == "error"
    assert failed.error_message == "deferred failed"

    cancelled_submission = await models.complete_simple(model, CONTEXT, {"deferred": True})
    assert cancelled_submission.deferred is not None
    await models.cancel_deferred(model, cancelled_submission.deferred)
    assert faux.state.cancelled_deferred == [cancelled_submission.deferred]
    cancelled = await models.fetch_deferred(model, cancelled_submission.deferred)
    assert cancelled.stop_reason == "error"
    assert cancelled.error_message is not None and "was cancelled" in cancelled.error_message
