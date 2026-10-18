from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from pi_ai import (
    AbortController,
    AbortError,
    AbortSignal,
    ApiKeyAuth,
    AssistantMessage,
    AssistantMessageEventStream,
    AuthCheck,
    AuthResult,
    Context,
    DoneEvent,
    InMemoryCredentialStore,
    InMemoryModelsStore,
    Model,
    ModelAuth,
    ModelCost,
    ModelCostTier,
    ModelsError,
    ModelsPublication,
    ModelsStoreEntry,
    OAuthAuth,
    Provider,
    ProviderAuth,
    StartEvent,
    TextContent,
    TranscriptContext,
    Usage,
    UsageCost,
    UserMessage,
    calculate_cost,
    create_models,
    create_provider,
    has_api,
    now_ms,
)


def make_model(provider: str, model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="test-api",
        provider=provider,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10000,
        max_tokens=1000,
    )


def done_message(model: Model, text: str) -> AssistantMessage:
    return AssistantMessage([TextContent(text)], api=model.api, provider=model.provider, model=model.id, usage=Usage())


async def _ambient_resolve(**_: Any) -> AuthResult:
    return AuthResult(ModelAuth())


AMBIENT_AUTH = ApiKeyAuth("Ambient", _ambient_resolve)


class FakeProvider(Provider):
    def __init__(
        self,
        id: str,
        *,
        models: list[Model] | None = None,
        auth: ProviderAuth | None = None,
        get_models: Callable[[], Sequence[Model]] | None = None,
        refresh_models: Any = None,
        calls: list[tuple[Model, dict[str, Any]]] | None = None,
        base_url: str | None = None,
    ) -> None:
        self.id = id
        self.name = id
        self.base_url = base_url
        self.auth = auth or ProviderAuth(api_key=AMBIENT_AUTH)
        self._models = models if models is not None else [make_model(id, "model-a")]
        self._get_models = get_models
        if refresh_models is not None:
            self.refresh_models = refresh_models
        self.calls = calls

    def get_models(self) -> Sequence[Model]:
        return self._get_models() if self._get_models is not None else self._models

    def _respond(self, model: Model, options: Any) -> AssistantMessageEventStream:
        if self.calls is not None:
            self.calls.append((model, dict(options or {})))
        stream = AssistantMessageEventStream()
        message = done_message(model, "ok")
        stream.push(StartEvent(message))
        stream.push(DoneEvent("stop", message))
        stream.end(message)
        return stream

    def stream(self, model: Model, context: TranscriptContext, options: Any = None) -> AssistantMessageEventStream:
        return self._respond(model, options)

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Any = None
    ) -> AssistantMessageEventStream:
        return self._respond(model, options)


CONTEXT = Context(messages=[UserMessage("hi")])


def env_key_auth(key: str | None) -> ApiKeyAuth:
    async def resolve(*, credential: Any, **_: Any) -> AuthResult | None:
        resolved = (credential or {}).get("key") or key
        if not resolved:
            return None
        return AuthResult(ModelAuth(api_key=resolved), source="stored" if credential else "env")

    return ApiKeyAuth("Test API key", resolve)


class FakeOAuth(OAuthAuth):
    name = "Test OAuth"

    def __init__(self, refresh: Callable[..., Any] | None = None) -> None:
        self._refresh = refresh

    async def login(self, interaction: Any, options: Any = None) -> Any:
        raise RuntimeError("not used")

    async def refresh(self, credential: Any, signal: AbortSignal) -> Any:
        if self._refresh is None:
            return credential
        return await self._refresh(credential)

    async def to_auth(self, credential: Any) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


def _streams_api() -> Any:
    class Streams:
        def stream(self, *_args: Any) -> AssistantMessageEventStream:
            return AssistantMessageEventStream()

        def stream_simple(self, *_args: Any) -> AssistantMessageEventStream:
            return AssistantMessageEventStream()

    return Streams()


async def wait_for(predicate: Callable[[], Any], timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if await predicate() if asyncio.iscoroutinefunction(predicate) else predicate():
            return
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.005)


async def test_enumerates_credential_metadata_without_secrets() -> None:
    credentials = InMemoryCredentialStore()

    async def api_key(_current: Any) -> Any:
        return {"type": "api_key", "key": "secret"}

    async def oauth(_current: Any) -> Any:
        return {"type": "oauth", "access": "access", "refresh": "refresh", "expires": now_ms() + 60_000}

    await credentials.modify("api-provider", api_key)
    await credentials.modify("oauth-provider", oauth)
    listed = await credentials.list()
    assert [(info.provider_id, info.type) for info in listed] == [
        ("api-provider", "api_key"),
        ("oauth-provider", "oauth"),
    ]


def test_applies_request_wide_pricing_tiers() -> None:
    model = make_model("openai", "gpt-5.6-sol")
    model.cost = ModelCost(5, 30, 0.5, 6.25, tiers=[ModelCostTier(272000, 10, 45, 1, 12.5)])

    def usage(cache_write: int) -> Usage:
        return Usage(200000, 100000, 72000, cache_write, 372000 + cache_write, UsageCost())

    short = calculate_cost(model, usage(0))
    assert (short.input, short.output, short.cache_read, short.cache_write) == pytest.approx((1, 3, 0.036, 0))
    long = calculate_cost(model, usage(1))
    assert (long.input, long.output, long.cache_read, long.cache_write) == pytest.approx((2, 4.5, 0.072, 0.0000125))


def test_cost_charges_twice_input_for_1h_cache_writes() -> None:
    model = make_model("anthropic", "claude")
    model.cost = ModelCost(3, 15, 0.3, 3.75)
    cost = calculate_cost(model, Usage(0, 0, 0, 1_000_000, 1_000_000, UsageCost(), cache_write_1h=400_000))
    assert cost.cache_write == pytest.approx(3.75 * 0.6 + 3 * 2 * 0.4)


def test_registers_replaces_and_deletes_providers() -> None:
    models = create_models()
    models.set_provider(FakeProvider("p1"))
    models.set_provider(FakeProvider("p2"))
    assert [p.id for p in models.get_providers()] == ["p1", "p2"]
    replacement = FakeProvider("p1")
    models.set_provider(replacement)
    assert models.get_provider("p1") is replacement
    assert len(models.get_providers()) == 2
    models.delete_provider("p1")
    assert models.get_provider("p1") is None
    models.clear_providers()
    assert models.get_providers() == []


def test_lists_and_finds_models_per_provider() -> None:
    models = create_models()
    models.set_provider(FakeProvider("p1", models=[make_model("p1", "m1"), make_model("p1", "m2")]))
    models.set_provider(FakeProvider("p2", models=[make_model("p2", "m3")]))
    assert [m.id for m in models.get_models()] == ["m1", "m2", "m3"]
    assert [m.id for m in models.get_models("p1")] == ["m1", "m2"]
    assert models.get_models("nope") == []
    assert models.get_model("p2", "m3").id == "m3"  # type: ignore[union-attr]
    assert models.get_model("p2", "missing") is None
    found = models.get_model("p2", "m3")
    assert found is not None and not has_api(found, "openai-completions") and has_api(found, "test-api")


async def test_chat_reads_independent_from_all_model_catalog() -> None:
    provider = FakeProvider("chat-only")

    def broken() -> Sequence[Any]:
        raise RuntimeError("all models unavailable")

    provider.get_all_models = broken
    models = create_models()
    models.set_provider(provider)
    assert [m.id for m in models.get_models("chat-only")] == ["model-a"]
    assert models.get_model("chat-only", "model-a").id == "model-a"  # type: ignore[union-attr]
    assert [m.id for m in await models.get_available("chat-only")] == ["model-a"]
    assert models.get_all_models("chat-only") == []


def test_swallows_provider_source_failures() -> None:
    def boom() -> Sequence[Model]:
        raise RuntimeError("boom")

    models = create_models()
    models.set_provider(FakeProvider("broken", get_models=boom))
    models.set_provider(FakeProvider("ok", models=[make_model("ok", "m1")]))
    assert [m.id for m in models.get_models()] == ["m1"]
    assert models.get_models("broken") == []
    with pytest.raises(RuntimeError, match="boom"):
        models.get_provider("broken").get_models()  # type: ignore[union-attr]


async def test_refresh_updates_dynamic_providers_and_reports_failures() -> None:
    state = {"list": [make_model("dyn", "before")], "refreshes": 0}

    async def refresh(context: Any) -> None:
        if not context.allow_network:
            return
        state["refreshes"] += 1

        def update() -> None:
            state["list"] = [make_model("dyn", "after")]

        await context.publish(ModelsPublication(update=update))

    models = create_models()
    models.set_provider(FakeProvider("dyn", get_models=lambda: state["list"], refresh_models=refresh))
    models.set_provider(FakeProvider("static", models=[make_model("static", "s1")]))
    assert models.get_model("dyn", "before") is not None
    first = await models.refresh()
    assert first.errors == {}
    assert state["refreshes"] == 1
    assert models.get_model("dyn", "after") is not None
    assert models.get_model("dyn", "before") is None

    async def flaky(context: Any) -> None:
        if context.allow_network:
            raise RuntimeError("fetch failed")

    models.set_provider(FakeProvider("flaky", refresh_models=flaky))
    second = await models.refresh()
    assert state["refreshes"] == 2
    assert str(second.errors["flaky"]) == "fetch failed"


async def test_restricts_refresh_to_selected_providers() -> None:
    calls: list[str] = []
    models = create_models()
    for provider_id in ("one", "two"):

        async def refresh(context: Any, provider_id: str = provider_id) -> None:
            calls.append(f"{provider_id}:{'network' if context.allow_network else 'cache'}")

        models.set_provider(FakeProvider(provider_id, refresh_models=refresh))
    result = await models.refresh(providers=["two", "unknown"])
    assert result.errors == {}
    assert calls == ["two:cache", "two:network"]


async def test_restores_cached_models_before_waiting_for_network_auth() -> None:
    store = InMemoryModelsStore()
    await store.write("dynamic", ModelsStoreEntry(models=[make_model("dynamic", "cached")]))
    auth_started = asyncio.Event()
    finish_auth = asyncio.Event()

    async def resolve(**_: Any) -> AuthResult:
        auth_started.set()
        await finish_auth.wait()
        return AuthResult(ModelAuth(api_key="key"))

    async def fetch_models(_context: Any) -> list[Any]:
        raise RuntimeError("must not fetch")

    provider = create_provider(
        id="dynamic",
        auth=ProviderAuth(api_key=ApiKeyAuth("Blocked auth", resolve)),
        models=[],
        fetch_models=fetch_models,
        api=_streams_api(),
    )
    models = create_models(models_store=store)
    models.set_provider(provider)
    controller = AbortController()
    pending = asyncio.ensure_future(models.refresh(providers=["dynamic"], signal=controller.signal))
    await auth_started.wait()
    assert models.get_model("dynamic", "cached") is not None
    controller.abort()
    assert (await pending).aborted is True
    finish_auth.set()


async def test_persistent_deletion_and_ephemeral_publication() -> None:
    entry: dict[str, Any] = {"value": ModelsStoreEntry(models=[make_model("dynamic", "stored")])}

    class Store:
        async def read(self, provider_id: str, *, signal: Any = None) -> Any:
            return entry["value"]

        async def write(self, provider_id: str, value: Any, *, signal: Any = None) -> None:
            entry["value"] = value

        async def delete(self, provider_id: str, *, signal: Any = None) -> None:
            entry["value"] = None

    state = {"value": "initial"}

    async def refresh(context: Any) -> None:
        assert context.stored.models[0].id == "stored"

        def deleted() -> None:
            assert entry["value"] is None
            state["value"] = "deleted"

        await context.publish(ModelsPublication(persist=None, update=deleted))
        await context.publish(ModelsPublication(update=lambda: state.update(value="ephemeral")))

    models = create_models(models_store=Store())  # type: ignore[arg-type]
    models.set_provider(FakeProvider("dynamic", refresh_models=refresh))
    result = await models.refresh(allow_network=False)
    assert result.errors == {}
    assert entry["value"] is None
    assert state["value"] == "ephemeral"


async def test_persists_dynamic_catalogs_and_restores_offline() -> None:
    credentials = InMemoryCredentialStore()
    store = InMemoryModelsStore()

    async def stored(_current: Any) -> Any:
        return {"type": "api_key", "key": "key"}

    await credentials.modify("dynamic", stored)

    def dynamic_provider(fetch: Any) -> Provider:
        return create_provider(
            id="dynamic",
            auth=ProviderAuth(api_key=env_key_auth(None)),
            models=[],
            fetch_models=fetch,
            api=_streams_api(),
        )

    async def fetched(_context: Any) -> list[Any]:
        return [make_model("dynamic", "fetched")]

    online = create_models(credentials=credentials, models_store=store)
    online.set_provider(dynamic_provider(fetched))
    assert (await online.refresh()).errors == {}
    assert online.get_model("dynamic", "fetched") is not None

    async def must_not(_context: Any) -> list[Any]:
        raise RuntimeError("must not fetch")

    offline = create_models(credentials=credentials, models_store=store)
    offline.set_provider(dynamic_provider(must_not))
    assert (await offline.refresh(allow_network=False)).errors == {}
    assert offline.get_model("dynamic", "fetched") is not None


async def test_passes_effective_credentials_and_skips_unconfigured() -> None:
    seen: dict[str, Any] = {"unconfigured": 0}

    async def configured(context: Any) -> None:
        if not context.allow_network:
            return
        seen["credential"] = context.credential
        seen["force"] = context.force

    async def unconfigured(context: Any) -> None:
        if context.allow_network:
            seen["unconfigured"] += 1

    models = create_models()
    models.set_provider(
        FakeProvider("configured", auth=ProviderAuth(api_key=env_key_auth("ambient-key")), refresh_models=configured)
    )
    models.set_provider(
        FakeProvider("unconfigured", auth=ProviderAuth(api_key=env_key_auth(None)), refresh_models=unconfigured)
    )
    await models.refresh(force=True)
    assert seen["credential"] == {"type": "api_key", "key": "ambient-key"}
    assert seen["force"] is True
    assert seen["unconfigured"] == 0


async def test_passes_registered_provider_base_url_to_refresh() -> None:
    base_urls: list[str | None] = []

    async def refresh(context: Any) -> None:
        base_urls.append(context.base_url)

    models = create_models()
    models.set_provider(FakeProvider("dynamic", refresh_models=refresh, base_url="https://regional.example.test/v1"))
    await models.refresh()
    assert base_urls == ["https://regional.example.test/v1", "https://regional.example.test/v1"]


async def test_refreshes_expired_oauth_before_refreshing_models() -> None:
    credentials = InMemoryCredentialStore()

    async def expired(_current: Any) -> Any:
        return {"type": "oauth", "access": "expired", "refresh": "refresh", "expires": 0}

    await credentials.modify("oauth-dynamic", expired)
    seen: dict[str, Any] = {}

    async def rotate(_credential: Any) -> Any:
        return {"type": "oauth", "access": "fresh", "refresh": "rotated", "expires": now_ms() + 60_000}

    async def refresh(context: Any) -> None:
        if context.allow_network:
            seen["credential"] = context.credential

    models = create_models(credentials=credentials)
    models.set_provider(
        FakeProvider("oauth-dynamic", auth=ProviderAuth(oauth=FakeOAuth(rotate)), refresh_models=refresh)
    )
    assert (await models.refresh()).errors == {}
    assert seen["credential"]["access"] == "fresh" and seen["credential"]["refresh"] == "rotated"
    stored = await credentials.read("oauth-dynamic")
    assert stored["access"] == "fresh" and stored["refresh"] == "rotated"  # type: ignore[index]


async def test_persists_oauth_refresh_started_before_model_refresh_was_cancelled() -> None:
    credentials = InMemoryCredentialStore()

    async def old(_current: Any) -> Any:
        return {"type": "oauth", "access": "old", "refresh": "old-refresh", "expires": 0}

    await credentials.modify("p1", old)
    controller = AbortController()

    async def rotate(credential: Any) -> Any:
        controller.abort()
        return {**credential, "access": "new", "refresh": "new-refresh", "expires": now_ms() + 60_000}

    async def refresh(_context: Any) -> None:
        return None

    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth(rotate)), refresh_models=refresh))
    assert (await models.refresh(signal=controller.signal)).aborted is True

    async def rotated() -> bool:
        stored = await credentials.read("p1")
        return bool(stored and stored["refresh"] == "new-refresh")

    await wait_for(rotated)


async def test_always_gives_providers_a_concrete_signal() -> None:
    received: list[Any] = []

    async def refresh(context: Any) -> None:
        received.append(context.signal)

    models = create_models()
    models.set_provider(FakeProvider("dynamic", refresh_models=refresh))
    result = await models.refresh()
    assert result.aborted is False
    assert isinstance(received[0], AbortSignal) and not received[0].aborted


async def test_binds_model_store_waits_to_provider_refresh_signal() -> None:
    storage_signals: list[Any] = []

    class Store:
        async def read(self, provider_id: str, *, signal: Any = None) -> Any:
            storage_signals.append(signal)
            return None

        async def write(self, provider_id: str, entry: Any, *, signal: Any = None) -> None:
            storage_signals.append(signal)

        async def delete(self, provider_id: str, *, signal: Any = None) -> None:
            storage_signals.append(signal)

    provider_signal: dict[str, Any] = {}

    async def refresh(context: Any) -> None:
        provider_signal["value"] = context.signal
        if not context.allow_network:
            return
        await context.publish(ModelsPublication(persist=ModelsStoreEntry(models=[make_model("dynamic", "fresh")])))

    models = create_models(models_store=Store())  # type: ignore[arg-type]
    models.set_provider(FakeProvider("dynamic", auth=ProviderAuth(api_key=env_key_auth("key")), refresh_models=refresh))
    result = await models.refresh(providers=["dynamic"])
    assert result.errors == {}
    assert len(storage_signals) == 3
    assert all(signal is provider_signal["value"] for signal in storage_signals)


async def test_returns_aborted_state_without_reporting_errors() -> None:
    controller = AbortController()

    async def refresh(context: Any) -> None:
        controller.abort()

    models = create_models()
    models.set_provider(FakeProvider("dynamic", refresh_models=refresh))
    result = await models.refresh(signal=controller.signal)
    assert result.aborted is True
    assert result.errors == {}


async def test_stops_waiting_when_provider_ignores_signal() -> None:
    controller = AbortController()
    started = asyncio.Event()
    stalled: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    calls = {"count": 0}

    async def refresh(_context: Any) -> None:
        calls["count"] += 1
        if calls["count"] != 1:
            return
        started.set()
        await stalled

    models = create_models()
    models.set_provider(FakeProvider("dynamic", refresh_models=refresh))
    pending = asyncio.ensure_future(models.refresh(signal=controller.signal))
    await started.wait()
    controller.abort()
    result = await pending
    assert result.aborted is True
    assert result.errors == {}
    stalled.set_exception(RuntimeError("late provider failure"))
    await asyncio.sleep(0)
    assert result.errors == {}


async def test_rejects_late_publication_from_superseded_provider() -> None:
    store = InMemoryModelsStore()
    state = {"value": "initial", "calls": 0}
    first_started = asyncio.Event()
    finish_first = asyncio.Event()

    async def refresh(context: Any) -> None:
        if not context.allow_network:
            return
        state["calls"] += 1
        current = state["calls"]
        if current == 1:
            first_started.set()
            await finish_first.wait()
        value = f"generation-{current}"
        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=[make_model("dynamic", value)]),
                update=lambda: state.update(value=value),
            )
        )

    models = create_models(models_store=store)
    models.set_provider(FakeProvider("dynamic", refresh_models=refresh))
    first = asyncio.ensure_future(models.refresh(providers=["dynamic"]))
    await first_started.wait()
    await models.refresh(providers=["dynamic"])
    await first
    finish_first.set()
    await asyncio.sleep(0.01)
    assert state["value"] == "generation-2"
    stored = await store.read("dynamic")
    assert stored is not None and stored.models[0].id == "generation-2"


async def test_passes_caller_signals_to_auth_callbacks() -> None:
    controller = AbortController()
    received: list[Any] = []

    async def login(interaction: Any) -> Any:
        received.append(interaction.signal)
        return {"type": "api_key", "key": "saved"}

    async def check(*, signal: Any, **_: Any) -> AuthCheck:
        received.append(signal)
        return AuthCheck("api_key")

    async def resolve(*, signal: Any, **_: Any) -> AuthResult:
        received.append(signal)
        return AuthResult(ModelAuth(api_key="resolved"))

    models = create_models()
    models.set_provider(
        FakeProvider("p1", auth=ProviderAuth(api_key=ApiKeyAuth("Signal auth", resolve, login=login, check=check)))
    )

    class Interaction:
        signal = controller.signal

        async def prompt(self, _prompt: Any) -> str:
            return "unused"

        def notify(self, _event: Any) -> None:
            return None

    await models.check_auth("p1", signal=controller.signal)
    await models.get_auth("p1", signal=controller.signal)
    await models.login("p1", "api_key", Interaction())
    assert received == [controller.signal, controller.signal, controller.signal]


async def test_stops_waiting_for_non_cooperative_auth_callbacks() -> None:
    check_started = asyncio.Event()
    resolve_started = asyncio.Event()
    blocked = asyncio.Event()

    async def check(**_: Any) -> AuthCheck:
        check_started.set()
        await blocked.wait()
        return AuthCheck("api_key")

    async def resolve(**_: Any) -> AuthResult:
        resolve_started.set()
        await blocked.wait()
        return AuthResult(ModelAuth(api_key="key"))

    models = create_models()
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=ApiKeyAuth("Blocked", resolve, check=check))))

    available_controller = AbortController()
    available = asyncio.ensure_future(models.get_available(signal=available_controller.signal))
    await check_started.wait()
    available_controller.abort()
    with pytest.raises(AbortError):
        await available

    auth_controller = AbortController()
    auth = asyncio.ensure_future(models.get_auth("p1", signal=auth_controller.signal))
    await resolve_started.wait()
    auth_controller.abort()
    with pytest.raises(AbortError):
        await auth
    blocked.set()


async def test_cancels_queued_credential_mutations() -> None:
    credentials = InMemoryCredentialStore()
    finish_first = asyncio.Event()
    second_ran = {"value": False}

    async def first_fn(_current: Any) -> Any:
        await finish_first.wait()
        return {"type": "api_key", "key": "first"}

    async def second_fn(_current: Any) -> Any:
        second_ran["value"] = True
        return {"type": "api_key", "key": "second"}

    first = asyncio.ensure_future(credentials.modify("p1", first_fn))
    controller = AbortController()
    second = asyncio.ensure_future(credentials.modify("p1", second_fn, signal=controller.signal))
    await asyncio.sleep(0)
    controller.abort()
    with pytest.raises(AbortError):
        await second
    finish_first.set()
    await first
    await asyncio.sleep(0.01)
    assert second_ran["value"] is False
    assert await credentials.read("p1") == {"type": "api_key", "key": "first"}


async def test_persists_oauth_refresh_started_before_request_was_cancelled() -> None:
    credentials = InMemoryCredentialStore()

    async def old(_current: Any) -> Any:
        return {"type": "oauth", "access": "old", "refresh": "old-refresh", "expires": 0}

    await credentials.modify("p1", old)
    controller = AbortController()

    async def rotate(credential: Any) -> Any:
        controller.abort()
        return {**credential, "access": "new", "refresh": "new-refresh", "expires": now_ms() + 60_000}

    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth(rotate))))
    with pytest.raises(AbortError):
        await models.get_auth("p1", signal=controller.signal)

    async def rotated() -> bool:
        stored = await credentials.read("p1")
        return bool(stored and stored["refresh"] == "new-refresh")

    await wait_for(rotated)


async def test_stored_credential_owns_provider() -> None:
    credentials = InMemoryCredentialStore()
    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=env_key_auth("env-key"), oauth=FakeOAuth())))
    model = make_model("p1", "model-a")
    assert (await models.get_auth(model)).auth.api_key == "env-key"  # type: ignore[union-attr]
    assert (await models.get_auth(model.provider)).auth.api_key == "env-key"  # type: ignore[union-attr]
    assert (await models.get_auth(model, api_key="explicit-key")).auth.api_key == "explicit-key"  # type: ignore[union-attr]

    async def oauth(_current: Any) -> Any:
        return {"type": "oauth", "access": "oauth-token", "refresh": "r", "expires": now_ms() + 10 * 60_000}

    await credentials.modify("p1", oauth)
    resolution = await models.get_auth(model.provider)
    assert resolution is not None and resolution.auth.api_key == "oauth-token" and resolution.source == "OAuth"

    async def api_key(_current: Any) -> Any:
        return {"type": "api_key", "key": "stored-key"}

    await credentials.modify("p1", api_key)
    resolution = await models.get_auth(model.provider)
    assert resolution is not None and resolution.auth.api_key == "stored-key" and resolution.source == "stored"


async def test_checks_auth_without_refreshing_oauth_and_filters_available() -> None:
    credentials = InMemoryCredentialStore()
    refreshes = {"count": 0}

    async def count_refresh(credential: Any) -> Any:
        refreshes["count"] += 1
        return credential

    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("ambient", auth=ProviderAuth(api_key=env_key_auth("env-key"))))
    models.set_provider(FakeProvider("missing", auth=ProviderAuth(api_key=env_key_auth(None))))
    models.set_provider(FakeProvider("oauth", auth=ProviderAuth(oauth=FakeOAuth(count_refresh))))

    async def expired(_current: Any) -> Any:
        return {"type": "oauth", "access": "expired", "refresh": "refresh", "expires": 0}

    await credentials.modify("oauth", expired)
    assert await models.check_auth("ambient") == AuthCheck("api_key", "env")
    assert await models.check_auth("missing") is None
    assert await models.check_auth("oauth") == AuthCheck("oauth", "OAuth")
    assert refreshes["count"] == 0
    assert [m.provider for m in await models.get_available()] == ["ambient", "oauth"]
    assert [m.provider for m in await models.get_available("ambient")] == ["ambient"]


async def test_login_and_logout_through_credential_store() -> None:
    credentials = InMemoryCredentialStore()
    api_key = env_key_auth(None)

    async def login(_interaction: Any) -> Any:
        return {"type": "api_key", "key": "logged-in"}

    api_key.login = login
    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=api_key)))

    class Interaction:
        signal = None

        async def prompt(self, _prompt: Any) -> str:
            return "unused"

        def notify(self, _event: Any) -> None:
            return None

    credential = await models.login("p1", "api_key", Interaction())
    assert credential == {"type": "api_key", "key": "logged-in"}
    assert await credentials.read("p1") == credential
    await models.logout("p1")
    assert await credentials.read("p1") is None


async def test_login_rejects_unknown_provider_and_missing_method() -> None:
    models = create_models()
    models.set_provider(FakeProvider("p1"))

    class Interaction:
        signal = None

        async def prompt(self, _prompt: Any) -> str:
            return ""

        def notify(self, _event: Any) -> None:
            return None

    with pytest.raises(ModelsError) as unknown:
        await models.login("ghost", "api_key", Interaction())
    assert unknown.value.code == "provider"
    with pytest.raises(ModelsError, match="does not support oauth login"):
        await models.login("p1", "oauth", Interaction())


async def test_stored_credential_without_handler_blocks_ambient_fallback() -> None:
    credentials = InMemoryCredentialStore()
    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=env_key_auth("env-key"))))

    async def oauth(_current: Any) -> Any:
        return {"type": "oauth", "access": "a", "refresh": "r", "expires": 0}

    await credentials.modify("p1", oauth)
    assert await models.get_auth("p1") is None


async def test_refreshes_expired_oauth_and_persists_rotated_credential() -> None:
    credentials = InMemoryCredentialStore()

    async def rotate(credential: Any) -> Any:
        return {**credential, "access": "new-token", "expires": now_ms() + 60 * 60_000}

    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth(rotate))))

    async def old(_current: Any) -> Any:
        return {"type": "oauth", "access": "old-token", "refresh": "r", "expires": 0}

    await credentials.modify("p1", old)
    assert (await models.get_auth("p1")).auth.api_key == "new-token"  # type: ignore[union-attr]
    assert (await credentials.read("p1"))["access"] == "new-token"  # type: ignore[index]


@pytest.mark.parametrize(("expires_in_ms", "min_validity"), [(60_000, None), (10 * 60_000, 30 * 60_000)])
async def test_refreshes_tokens_close_to_expiry(expires_in_ms: int, min_validity: int | None) -> None:
    credentials = InMemoryCredentialStore()
    calls = {"count": 0}

    async def rotate(credential: Any) -> Any:
        calls["count"] += 1
        return {**credential, "access": "new-token", "expires": now_ms() + 60 * 60_000}

    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth(rotate))))

    async def soon(_current: Any) -> Any:
        return {"type": "oauth", "access": "old-token", "refresh": "r", "expires": now_ms() + expires_in_ms}

    await credentials.modify("p1", soon)
    resolution = await models.get_auth("p1", min_oauth_validity_ms=min_validity)
    assert resolution is not None and resolution.auth.api_key == "new-token"
    assert calls["count"] == 1


async def test_oauth_refresh_failure_has_code_oauth_and_keeps_credential() -> None:
    credentials = InMemoryCredentialStore()

    async def fail(_credential: Any) -> Any:
        raise RuntimeError("token refresh failed (400): invalid_grant")

    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth(fail))))

    async def old(_current: Any) -> Any:
        return {"type": "oauth", "access": "old", "refresh": "r", "expires": 0}

    await credentials.modify("p1", old)
    with pytest.raises(ModelsError) as raised:
        await models.get_auth("p1")
    assert raised.value.code == "oauth"
    assert "OAuth refresh failed for p1: token refresh failed (400): invalid_grant" in str(raised.value)
    assert (await credentials.read("p1"))["access"] == "old"  # type: ignore[index]


async def test_serializes_concurrent_oauth_refreshes() -> None:
    credentials = InMemoryCredentialStore()

    async def old(_current: Any) -> Any:
        return {"type": "oauth", "access": "old", "refresh": "r1", "expires": 0}

    await credentials.modify("p1", old)
    calls = {"count": 0}

    async def rotate(_credential: Any) -> Any:
        calls["count"] += 1
        await asyncio.sleep(0.01)
        return {"type": "oauth", "access": f"new-{calls['count']}", "refresh": "r2", "expires": now_ms() + 3_600_000}

    models = create_models(credentials=credentials)
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth(rotate))))
    first, second = await asyncio.gather(models.get_auth("p1"), models.get_auth("p1"))
    assert calls["count"] == 1
    assert first.auth.api_key == "new-1" and second.auth.api_key == "new-1"  # type: ignore[union-attr]


async def test_valid_oauth_tokens_resolve_without_modify() -> None:
    base = InMemoryCredentialStore()
    modifies = {"count": 0}

    class CountingStore:
        async def read(self, provider_id: str, *, signal: Any = None) -> Any:
            return await base.read(provider_id)

        async def list(self, *, signal: Any = None) -> Any:
            return await base.list()

        async def modify(self, provider_id: str, fn: Any, *, signal: Any = None) -> Any:
            modifies["count"] += 1
            return await base.modify(provider_id, fn)

        async def delete(self, provider_id: str, *, signal: Any = None) -> None:
            await base.delete(provider_id)

    async def valid(_current: Any) -> Any:
        return {"type": "oauth", "access": "valid", "refresh": "r", "expires": now_ms() + 10 * 60_000}

    await base.modify("p1", valid)
    models = create_models(credentials=CountingStore())  # type: ignore[arg-type]
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth())))
    assert (await models.get_auth("p1")).auth.api_key == "valid"  # type: ignore[union-attr]
    assert modifies["count"] == 0


async def test_wraps_credential_store_failures() -> None:
    class ReadFailing:
        async def read(self, *_args: Any, **_kwargs: Any) -> Any:
            raise OSError("disk on fire")

        async def list(self, **_kwargs: Any) -> Any:
            return []

        async def modify(self, *_args: Any, **_kwargs: Any) -> Any:
            return None

        async def delete(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    models = create_models(credentials=ReadFailing())  # type: ignore[arg-type]
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=env_key_auth("env-key"))))
    with pytest.raises(ModelsError) as read_error:
        await models.get_auth("p1")
    assert read_error.value.code == "auth"

    class ModifyFailing(ReadFailing):
        async def read(self, *_args: Any, **_kwargs: Any) -> Any:
            return {"type": "oauth", "access": "old", "refresh": "r", "expires": 0}

        async def modify(self, *_args: Any, **_kwargs: Any) -> Any:
            raise OSError("disk on fire")

    oauth_models = create_models(credentials=ModifyFailing())  # type: ignore[arg-type]
    oauth_models.set_provider(FakeProvider("p1", auth=ProviderAuth(oauth=FakeOAuth())))
    with pytest.raises(ModelsError) as modify_error:
        await oauth_models.get_auth("p1")
    assert modify_error.value.code == "auth"


async def test_wraps_api_key_auth_failures() -> None:
    async def fail(**_: Any) -> Any:
        raise RuntimeError("nope")

    models = create_models()
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=ApiKeyAuth("Failing", fail))))
    with pytest.raises(ModelsError) as raised:
        await models.get_auth("p1")
    assert raised.value.code == "auth"


async def test_uses_explicit_api_key_and_env_during_auth_resolution() -> None:
    calls: list[tuple[Model, dict[str, Any]]] = []

    async def resolve(*, credential: Any, ctx: Any, **_: Any) -> AuthResult | None:
        account = ((credential or {}).get("env") or {}).get("ACCOUNT_ID") or await ctx.env("ACCOUNT_ID")
        if not (credential or {}).get("key") or not account:
            return None
        return AuthResult(
            ModelAuth(api_key=credential["key"], base_url=f"https://example.test/{account}"),
            env={"ACCOUNT_ID": account},
        )

    models = create_models()
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=ApiKeyAuth("Scoped", resolve)), calls=calls))
    await models.complete_simple(
        make_model("p1", "model-a"), CONTEXT, api_key="explicit-key", env={"ACCOUNT_ID": "acct"}
    )
    model, options = calls[0]
    assert model.base_url == "https://example.test/acct"
    assert options["api_key"] == "explicit-key"
    assert options["env"] == {"ACCOUNT_ID": "acct"}


async def test_merges_resolved_auth_into_stream_options() -> None:
    calls: list[tuple[Model, dict[str, Any]]] = []

    async def resolve(**_: Any) -> AuthResult:
        return AuthResult(
            ModelAuth(
                api_key="resolved-key",
                headers={"Authorization": "Bearer resolved-key", "x-a": "auth", "x-b": "auth"},
                base_url="https://auth.test/v1",
            )
        )

    models = create_models()
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=ApiKeyAuth("Test", resolve)), calls=calls))
    model = make_model("p1", "model-a")
    result = await models.complete_simple(
        model, CONTEXT, {"api_key": "explicit-key", "headers": {"authorization": "Explicit token", "x-b": "explicit"}}
    )
    assert result.stop_reason == "stop"
    request_model, options = calls[0]
    assert options["api_key"] == "explicit-key"
    assert options["headers"] == {"authorization": "Explicit token", "x-a": "auth", "x-b": "explicit"}
    assert request_model.base_url == "https://auth.test/v1"
    await models.complete_simple(model, CONTEXT)
    assert calls[1][1]["api_key"] == "resolved-key"


async def test_model_headers_and_transform_headers() -> None:
    calls: list[tuple[Model, dict[str, Any]]] = []
    models = create_models()
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=env_key_auth("key")), calls=calls))
    model = make_model("p1", "model-a")
    model.headers = {"x-model": "model", "x-shared": "model"}
    assert (await models.get_auth("p1")).auth.headers is None  # type: ignore[union-attr]
    assert (await models.get_auth(model)).auth.headers == {"x-model": "model", "x-shared": "model"}  # type: ignore[union-attr]
    transforms = {"count": 0}

    async def transform(headers: dict[str, Any]) -> dict[str, Any]:
        transforms["count"] += 1
        assert headers == {"x-model": "model", "x-explicit": "explicit", "X-Shared": "explicit"}
        return {**headers, "x-transformed": "yes"}

    await models.complete_simple(
        model, CONTEXT, headers={"x-explicit": "explicit", "X-Shared": "explicit"}, transform_headers=transform
    )
    assert transforms["count"] == 1
    assert calls[0][1]["headers"] == {
        "x-model": "model",
        "x-explicit": "explicit",
        "X-Shared": "explicit",
        "x-transformed": "yes",
    }
    assert "transform_headers" not in calls[0][1]


async def test_error_stream_for_unknown_providers() -> None:
    result = await create_models().complete_simple(make_model("ghost", "model-a"), CONTEXT)
    assert result.stop_reason == "error"
    assert "Unknown provider: ghost" in (result.error_message or "")


async def test_unconfigured_provider_surfaces_as_stream_error() -> None:
    models = create_models()
    models.set_provider(FakeProvider("p1", auth=ProviderAuth(api_key=env_key_auth(None))))
    result = await models.complete(make_model("p1", "model-a"), CONTEXT)
    assert result.stop_reason == "error"
    assert result.error_message == "Provider is not configured: p1"


async def test_streams_through_the_provider() -> None:
    models = create_models()
    models.set_provider(FakeProvider("p1"))
    stream = models.stream_simple(make_model("p1", "model-a"), CONTEXT)
    assert [event.type async for event in stream] == ["start", "done"]
    assert (await stream.result()).stop_reason == "stop"


async def test_create_provider_requires_an_implementation_and_dispatches_by_api() -> None:
    with pytest.raises(ValueError, match="at least one of"):
        create_provider(id="empty", auth=ProviderAuth(api_key=AMBIENT_AUTH), models=[])

    model = make_model("mixed", "m")
    provider = create_provider(
        id="mixed", auth=ProviderAuth(api_key=AMBIENT_AUTH), models=[model], api={"other-api": _streams_api()}
    )
    models = create_models()
    models.set_provider(provider)
    result = await models.complete(model, CONTEXT)
    assert result.stop_reason == "error"
    assert 'has no API implementation for "test-api"' in (result.error_message or "")


async def test_image_and_classifier_operations_never_raise() -> None:
    from pi_ai import ClassifierContext, ClassifierModel, ImageModel, ImagesContext

    models = create_models()
    image_model = ImageModel(id="img", api="x", provider="ghost")
    result = await models.generate_images(image_model, ImagesContext([TextContent("draw")]))
    assert result.stop_reason == "error" and "Unknown provider: ghost" in (result.error_message or "")
    classifier = ClassifierModel(id="cls", api="x", provider="ghost")
    classified = await models.classify(classifier, ClassifierContext({"a": 1}, {}))
    assert classified.stop_reason == "error"
