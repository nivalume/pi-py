"""Provider request wrappers: Cloudflare placeholders, OpenCode session header, Azure endpoint/deployment, dispatch.

Ports upstream ``cloudflare-stream.test.ts``, ``opencode-provider-headers.test.ts`` and the
provider-level parts of ``azure-openai-completions.test.ts``. API implementation modules are
replaced with fakes in ``sys.modules``, so nothing here depends on the real adapters.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from typing import Any

import pytest

from pi_ai import (
    AssistantMessageEventStream,
    ClassifierContext,
    ClassifierModel,
    ClassifierResult,
    Context,
    DoneEvent,
    Model,
    ModelCost,
    StartEvent,
    UserMessage,
    faux_assistant_message,
    normalize_context,
)
from pi_ai.api.registry import API_MODULES
from pi_ai.providers.all import builtin_providers, get_builtin_model
from pi_ai.providers.azure import azure_provider, azure_streams
from pi_ai.providers.cloudflare_ai_gateway import cloudflare_ai_gateway_provider
from pi_ai.providers.cloudflare_stream import cloudflare_classifier, cloudflare_streams, resolve_cloudflare_model
from pi_ai.providers.opencode import opencode_provider
from pi_ai.providers.opencode_headers import with_opencode_session_header

CONTEXT = normalize_context(Context(messages=[UserMessage("hi")]))


def _completed_stream() -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = faux_assistant_message("ok")
    stream.push(StartEvent(message))
    stream.push(DoneEvent("stop", message))
    stream.end(message)
    return stream


class RecordingStreams:
    def __init__(self, label: str = "streams") -> None:
        self.label = label
        self.calls: list[tuple[str, Model, dict[str, Any] | None]] = []

    def stream(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        self.calls.append(("stream", model, dict(options) if options is not None else None))
        return _completed_stream()

    def stream_simple(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        self.calls.append(("stream_simple", model, dict(options) if options is not None else None))
        return _completed_stream()


def _fake_api_module(monkeypatch: pytest.MonkeyPatch, api: str) -> RecordingStreams:
    """Replace the implementation module of ``api`` with a recording fake."""
    recording = RecordingStreams(api)
    module = types.ModuleType(API_MODULES[api])
    module.stream = recording.stream  # type: ignore[attr-defined]
    module.stream_simple = recording.stream_simple  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, API_MODULES[api], module)
    return recording


def _model(api: str, provider: str, base_url: str, model_id: str = "model") -> Model:
    return Model(
        id=model_id,
        api=api,
        provider=provider,
        base_url=base_url,
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )


# ---------------------------------------------------------------------------
# Cloudflare
# ---------------------------------------------------------------------------

GATEWAY_MODEL = _model(
    "openai-completions",
    "cloudflare-ai-gateway",
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai",
)


def test_cloudflare_materializes_the_model_endpoint_before_dispatch() -> None:
    recording = RecordingStreams()
    streams = cloudflare_streams(recording)
    env = {"CLOUDFLARE_ACCOUNT_ID": "account", "CLOUDFLARE_GATEWAY_ID": "gateway"}

    streams.stream(GATEWAY_MODEL, CONTEXT, {"env": env})
    streams.stream_simple(GATEWAY_MODEL, CONTEXT, {"env": env})

    assert [call[1].base_url for call in recording.calls] == [
        "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
        "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
    ]
    assert [call[0] for call in recording.calls] == ["stream", "stream_simple"]


def test_cloudflare_keeps_placeholders_when_the_provider_env_does_not_resolve_them() -> None:
    recording = RecordingStreams()
    streams = cloudflare_streams(recording)

    streams.stream_simple(GATEWAY_MODEL, CONTEXT, {})
    streams.stream_simple(GATEWAY_MODEL, CONTEXT, None)

    assert recording.calls[0][1] is GATEWAY_MODEL
    assert recording.calls[1][1] is GATEWAY_MODEL


def test_resolve_cloudflare_model_replaces_only_known_placeholders() -> None:
    partial = resolve_cloudflare_model(GATEWAY_MODEL, {"CLOUDFLARE_ACCOUNT_ID": "acct"})
    assert partial.base_url == "https://gateway.ai.cloudflare.com/v1/acct/{CLOUDFLARE_GATEWAY_ID}/openai"
    assert partial.id == GATEWAY_MODEL.id
    unchanged = resolve_cloudflare_model(GATEWAY_MODEL, {"OTHER": "x"})
    assert unchanged is GATEWAY_MODEL


async def test_cloudflare_classifier_materializes_the_rest_endpoint() -> None:
    seen: list[str] = []

    class Classifier:
        async def classify(self, model: ClassifierModel, context: Any, options: Any = None) -> ClassifierResult:
            seen.append(model.base_url)
            return ClassifierResult(api=model.api, provider=model.provider, model=model.id)

    model = ClassifierModel(
        id="c",
        api="cloudflare-workers-ai-system-one",
        provider="cloudflare-workers-ai",
        base_url="https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai",
        context_window=1000,
    )
    result = await cloudflare_classifier(Classifier()).classify(
        model, ClassifierContext(state={}, questions={}), {"env": {"CLOUDFLARE_ACCOUNT_ID": "acct"}}
    )
    assert result.stop_reason == "stop"
    assert seen == ["https://api.cloudflare.com/client/v4/accounts/acct/ai"]


def test_cloudflare_catalog_base_urls_use_the_placeholders() -> None:
    gateway = cloudflare_ai_gateway_provider()
    for model in gateway.get_models():
        assert "{CLOUDFLARE_ACCOUNT_ID}" in model.base_url and "{CLOUDFLARE_GATEWAY_ID}" in model.base_url
    workers = next(provider for provider in builtin_providers() if provider.id == "cloudflare-workers-ai")
    assert workers.get_all_models is not None
    for model in workers.get_all_models():
        assert "{CLOUDFLARE_ACCOUNT_ID}" in model.base_url


async def test_cloudflare_gateway_provider_dispatches_all_three_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = {
        api: _fake_api_module(monkeypatch, api)
        for api in ("anthropic-messages", "openai-completions", "openai-responses")
    }
    provider = cloudflare_ai_gateway_provider()
    env = {"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_GATEWAY_ID": "gw"}
    for api, fake in fakes.items():
        model = _model(
            api,
            "cloudflare-ai-gateway",
            "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/x",
        )
        await provider.stream_simple(model, CONTEXT, {"env": env}).result()
        assert fake.calls[0][1].base_url == "https://gateway.ai.cloudflare.com/v1/acct/gw/x"


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------

OPENCODE_MODEL = _model("test-api", "opencode", "https://opencode.ai/zen/v1", "test-model")


@pytest.mark.parametrize("method", ["stream", "stream_simple"])
def test_opencode_maps_session_id_even_without_cache_retention(method: str) -> None:
    # Regression test for https://github.com/earendil-works/pi/issues/9326
    recording = RecordingStreams()
    streams = with_opencode_session_header(recording)

    getattr(streams, method)(OPENCODE_MODEL, CONTEXT, {"session_id": "conversation-1", "cache_retention": "none"})

    options = recording.calls[0][2]
    assert options is not None
    assert options["headers"] == {"x-opencode-session": "conversation-1"}


@pytest.mark.parametrize(
    "headers",
    [{"X-OpenCode-Session": "caller-value"}, {"X-OpenCode-Session": None}],
)
def test_opencode_preserves_a_case_insensitive_caller_override(headers: dict[str, str | None]) -> None:
    recording = RecordingStreams()
    streams = with_opencode_session_header(recording)

    streams.stream_simple(OPENCODE_MODEL, CONTEXT, {"session_id": "generated-value", "headers": headers})

    options = recording.calls[0][2]
    assert options is not None and options["headers"] == headers


def test_opencode_does_not_fabricate_a_session_header_when_session_id_is_absent() -> None:
    recording = RecordingStreams()
    streams = with_opencode_session_header(recording)

    streams.stream_simple(OPENCODE_MODEL, CONTEXT, {"headers": {"x-custom": "value"}})
    streams.stream_simple(OPENCODE_MODEL, CONTEXT)

    first = recording.calls[0][2]
    assert first is not None and first["headers"] == {"x-custom": "value"}
    assert recording.calls[1][2] is None


def test_opencode_wrapper_keeps_deferred_capabilities() -> None:
    class Deferred(RecordingStreams):
        def fetch_deferred(self, *args: Any) -> Any:
            return _completed_stream()

    wrapped = with_opencode_session_header(Deferred())
    assert wrapped.fetch_deferred is not None
    assert wrapped.cancel_deferred is None
    assert with_opencode_session_header(RecordingStreams()).fetch_deferred is None


async def test_opencode_provider_adds_the_header_for_every_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = {
        api: _fake_api_module(monkeypatch, api)
        for api in ("anthropic-messages", "google-generative-ai", "openai-completions", "openai-responses")
    }
    provider = opencode_provider()
    for api, fake in fakes.items():
        model = _model(api, "opencode", "https://opencode.ai/zen/v1")
        await provider.stream(model, CONTEXT, {"session_id": "s-1"}).result()
        options = fake.calls[0][2]
        assert options is not None and options["headers"] == {"x-opencode-session": "s-1"}, api


# ---------------------------------------------------------------------------
# Azure
# ---------------------------------------------------------------------------

AZURE_ENV = {"AZURE_OPENAI_BASE_URL": "https://my-resource.services.ai.azure.com"}


def _azure_deepseek() -> Model:
    model = get_builtin_model("azure", "deepseek-v4-pro")
    assert model is not None and model.api == "openai-completions"
    return model


@pytest.fixture
def no_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_RESOURCE_NAME",
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


async def test_azure_normalizes_the_endpoint_the_completions_api_receives(no_azure_env: None) -> None:
    recording = RecordingStreams()
    await azure_streams(recording).stream(_azure_deepseek(), CONTEXT, {"api_key": "k", "env": AZURE_ENV}).result()

    assert recording.calls[0][1].base_url == "https://my-resource.services.ai.azure.com/openai/v1"
    assert recording.calls[0][1].id == "deepseek-v4-pro"


async def test_azure_surfaces_an_unconfigured_endpoint_as_an_error_event(no_azure_env: None) -> None:
    recording = RecordingStreams()
    stream = azure_streams(recording).stream(_azure_deepseek(), CONTEXT, {"api_key": "k"})
    result = await stream.result()

    assert result.stop_reason == "error"
    assert result.error_message is not None and "Azure OpenAI base URL is required" in result.error_message
    assert recording.calls == []


async def test_azure_keeps_options_when_the_deployment_is_the_model_id(no_azure_env: None) -> None:
    recording = RecordingStreams()
    options = {"api_key": "k", "env": AZURE_ENV}
    await azure_streams(recording).stream_simple(_azure_deepseek(), CONTEXT, options).result()

    assert recording.calls[0][2] == options


async def test_azure_sends_the_mapped_deployment_name_through_on_payload(no_azure_env: None) -> None:
    recording = RecordingStreams()
    env = {**AZURE_ENV, "AZURE_OPENAI_DEPLOYMENT_NAME_MAP": "deepseek-v4-pro=my-deepseek"}
    await azure_streams(recording).stream_simple(_azure_deepseek(), CONTEXT, {"api_key": "k", "env": env}).result()

    options = recording.calls[0][2]
    assert options is not None
    payload = await options["on_payload"]({"model": "deepseek-v4-pro", "messages": []}, recording.calls[0][1])
    assert payload == {"model": "my-deepseek", "messages": []}


async def test_azure_passes_the_deployment_name_through_a_callers_on_payload(no_azure_env: None) -> None:
    recording = RecordingStreams()
    seen: list[Any] = []

    def on_payload(payload: dict[str, Any], _model: Any) -> dict[str, Any]:
        seen.append(payload["model"])
        return {**payload, "temperature": 0.1}

    await (
        azure_streams(recording)
        .stream(
            _azure_deepseek(),
            CONTEXT,
            {
                "api_key": "k",
                "env": AZURE_ENV,
                "azure_deployment_name": "explicit-deployment",
                "on_payload": on_payload,
            },
        )
        .result()
    )

    options = recording.calls[0][2]
    assert options is not None
    payload = await options["on_payload"]({"model": "deepseek-v4-pro"}, recording.calls[0][1])
    assert seen == ["explicit-deployment"]
    assert payload == {"model": "explicit-deployment", "temperature": 0.1}

    # A callback returning None keeps the rewritten payload.
    await (
        azure_streams(recording)
        .stream(
            _azure_deepseek(),
            CONTEXT,
            {"env": AZURE_ENV, "azure_deployment_name": "d", "on_payload": lambda payload, model: None},
        )
        .result()
    )
    options = recording.calls[1][2]
    assert options is not None
    assert await options["on_payload"]({"model": "deepseek-v4-pro"}, None) == {"model": "d"}


async def test_azure_api_map_routes_responses_and_completions(
    monkeypatch: pytest.MonkeyPatch, no_azure_env: None
) -> None:
    responses = _fake_api_module(monkeypatch, "azure-openai-responses")
    completions = _fake_api_module(monkeypatch, "openai-completions")
    provider = azure_provider()
    gpt = get_builtin_model("azure", "gpt-4o-mini")
    assert gpt is not None and gpt.api == "azure-openai-responses"

    await provider.stream(gpt, CONTEXT, {"api_key": "k"}).result()
    await provider.stream(_azure_deepseek(), CONTEXT, {"api_key": "k", "env": AZURE_ENV}).result()

    assert [call[1].id for call in responses.calls] == ["gpt-4o-mini"]
    assert [call[1].id for call in completions.calls] == ["deepseek-v4-pro"]


# ---------------------------------------------------------------------------
# Mixed-API dispatch of built-in providers
# ---------------------------------------------------------------------------


async def test_mixed_api_builtin_providers_dispatch_catalog_models_to_their_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = {
        api: _fake_api_module(monkeypatch, api)
        for api in ("anthropic-messages", "openai-completions", "openai-responses")
    }
    for provider in builtin_providers():
        if provider.id not in ("fireworks", "github-copilot", "openrouter"):
            continue
        apis_seen: set[str] = set()
        for model in provider.get_models():
            if model.api in apis_seen:
                continue
            apis_seen.add(model.api)
            before = len(fakes[model.api].calls)
            result = await provider.stream_simple(model, CONTEXT, {"api_key": "k"}).result()
            assert result.stop_reason == "stop", f"{provider.id}/{model.id}"
            assert len(fakes[model.api].calls) == before + 1
        assert len(apis_seen) >= 2, provider.id


async def test_builtin_providers_report_a_stream_error_for_unknown_apis() -> None:
    provider = next(provider for provider in builtin_providers() if provider.id == "fireworks")
    result = await provider.stream(_model("unknown-api", "fireworks", "https://x"), CONTEXT).result()
    assert result.stop_reason == "error"
    assert result.error_message is not None and 'no API implementation for "unknown-api"' in result.error_message
