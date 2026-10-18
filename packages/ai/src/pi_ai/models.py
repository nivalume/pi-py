"""The model runtime: providers, the ``Models`` collection, and model helpers.

A :class:`Provider` owns its model catalog, auth, and request behavior. A
:class:`Models` collection holds providers, resolves auth through the owning
provider and routes every request to it::

    models = create_models()
    models.set_provider(anthropic_provider())
    model = models.get_model("anthropic", "claude-sonnet-4-5")
    message = await models.complete(model, Context(messages=[UserMessage("Hi")]))
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .abort import (
    AbortController,
    AbortError,
    AbortSignal,
    operation_signal,
    race_with_abort_signal,
)
from .api.lazy import lazy_stream
from .auth.context import default_provider_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.resolve import AuthResolutionOverrides, refresh_stored_oauth_credential, resolve_provider_auth
from .auth.types import (
    AuthCheck,
    AuthContext,
    AuthInteraction,
    AuthResult,
    Credential,
    CredentialStore,
    LoginOptions,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from .event_stream import AssistantMessageEventStream
from .models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry
from .types import (
    AnyModel,
    AssistantImages,
    AssistantMessage,
    ClassifierContext,
    ClassifierModel,
    ClassifierResult,
    Context,
    DeferredHandle,
    ImageModel,
    ImagesContext,
    Model,
    ModelThinkingLevel,
    TranscriptContext,
    Usage,
    UsageCost,
    now_ms,
)
from .utils.callbacks import maybe_await
from .utils.misc import merge_headers
from .utils.model_operations import (
    assert_chat_model,
    assert_classifier_input_supported,
    assert_classifier_model,
    assert_image_model,
    classifier_error_result,
    get_model_type,
    image_error_result,
    is_model_type,
)
from .utils.models_error import ModelsError, ModelsErrorCode
from .utils.transcript import normalize_context

__all__ = [
    "ModelsError",
    "ModelsErrorCode",
    "get_model_type",
    "is_model_type",
    "UNCHANGED",
    "ModelsPublication",
    "RefreshModelsContext",
    "ModelsRefreshResult",
    "Provider",
    "Models",
    "MutableModels",
    "create_models",
    "create_provider",
    "has_api",
    "calculate_cost",
    "get_supported_thinking_levels",
    "clamp_thinking_level",
    "models_are_equal",
]

_KNOWN_MODEL_TYPES = frozenset({"chat", "image", "classifier"})

#: ``ModelsPublication.persist`` default: leave persisted storage unchanged.
UNCHANGED: Any = ...


def _has_known_model_type(model: Any) -> bool:
    return get_model_type(model) in _KNOWN_MODEL_TYPES


@dataclass(slots=True)
class ModelsPublication:
    #: Provider-selected persisted catalog. ``UNCHANGED`` leaves storage as is; ``None`` deletes it.
    persist: ModelsStoreEntry | None = UNCHANGED
    #: Optional synchronous update of provider-private in-memory catalog state.
    update: Callable[[], None] | None = None


@dataclass(slots=True)
class RefreshModelsContext:
    #: Generation-checked publication; the update runs only after the persistence mutation.
    publish: Callable[[ModelsPublication], Awaitable[bool]]
    #: Always present, including when the public refresh caller omits a signal.
    signal: AbortSignal
    #: False during offline/cache-only initialization.
    allow_network: bool = True
    #: Effective configured credential. OAuth credentials are refreshed before network access.
    credential: Credential | None = None
    #: Provider-scoped catalog snapshot captured before this refresh phase.
    stored: ModelsStoreEntry | None = None
    #: Bypass provider freshness checks.
    force: bool | None = None
    #: Base URL of the registered provider.
    base_url: str | None = None


@dataclass(slots=True)
class ModelsRefreshResult:
    aborted: bool
    errors: dict[str, Exception] = field(default_factory=dict)


class Provider:
    """The concrete runtime unit: identity, auth, model listing, and operations.

    Subclass it to implement a provider by hand, or use :func:`create_provider`.
    Optional capabilities (``get_all_models``, ``refresh_models``, ``filter_models``,
    ``filter_all_models``, ``fetch_deferred``, ``cancel_deferred``, ``generate_images``,
    ``classify``) are ``None`` unless implemented.
    """

    id: str
    name: str
    base_url: str | None = None
    headers: dict[str, str | None] | None = None
    auth: ProviderAuth

    get_all_models: Callable[[], Sequence[AnyModel]] | None = None
    refresh_models: Callable[[RefreshModelsContext], Awaitable[None]] | None = None
    filter_models: Callable[[Sequence[Model], Credential | None], Sequence[Model]] | None = None
    filter_all_models: Callable[[Sequence[AnyModel], Credential | None], Sequence[AnyModel]] | None = None
    fetch_deferred: Callable[..., AssistantMessageEventStream] | None = None
    cancel_deferred: Callable[..., Awaitable[None]] | None = None
    generate_images: Callable[..., Awaitable[AssistantImages]] | None = None
    classify: Callable[..., Awaitable[ClassifierResult]] | None = None

    def get_models(self) -> Sequence[Model]:
        """Current known chat models. Must not raise."""
        raise NotImplementedError

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        raise NotImplementedError

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        raise NotImplementedError

    def find_model(self, type: str, model_id: str) -> AnyModel | None:
        """Look up a model by type and id (providers may override with an index)."""
        # Chat reads stay independent of get_all_models, which may fail on its own.
        source = self.get_all_models() if type != "chat" and self.get_all_models is not None else self.get_models()
        for model in source:
            if model.id == model_id and get_model_type(model) == type:
                return model
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"


def _merge_options(options: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if options is None:
        return dict(kwargs)
    if not kwargs:
        return dict(options)
    return {**options, **kwargs}


def _safe_models(provider: Provider, all_types: bool) -> Sequence[AnyModel]:
    try:
        if all_types and provider.get_all_models is not None:
            return provider.get_all_models()
        return provider.get_models()
    except Exception:
        return []


class Models:
    """Runtime collection of providers plus auth application and request convenience.

    Reads are synchronous and return last-known lists: the unqualified accessors
    (``get_models``, ``get_model``, ``get_available``) return chat models, the
    ``*_of_type`` accessors one model type, and ``get_all_models``/``get_all_available``
    every type.

    Request methods accept options as a mapping and/or keyword arguments::

        await models.complete(model, context, api_key="sk-...", max_tokens=512)
    """

    def __init__(
        self,
        *,
        credentials: CredentialStore | None = None,
        models_store: ModelsStore | None = None,
        auth_context: AuthContext | None = None,
    ) -> None:
        self._providers: dict[str, Provider] = {}
        self._credentials: CredentialStore = credentials or InMemoryCredentialStore()
        self._models_store: ModelsStore = models_store or InMemoryModelsStore()
        self._auth_context: AuthContext = auth_context or default_provider_auth_context()
        self._refresh_generations: dict[str, int] = {}
        self._refresh_controllers: dict[str, AbortController] = {}
        self._publication_locks: dict[str, asyncio.Lock] = {}

    @property
    def credentials(self) -> CredentialStore:
        return self._credentials

    @property
    def models_store(self) -> ModelsStore:
        return self._models_store

    # -- providers ----------------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        """Upsert/replace by ``provider.id``."""
        self._supersede_provider_refresh(provider.id)
        self._providers[provider.id] = provider

    def delete_provider(self, provider_id: str) -> None:
        self._supersede_provider_refresh(provider_id)
        self._providers.pop(provider_id, None)

    def clear_providers(self) -> None:
        for provider_id in {*self._providers, *self._refresh_controllers}:
            self._supersede_provider_refresh(provider_id)
        self._providers.clear()

    def get_providers(self) -> list[Provider]:
        return list(self._providers.values())

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    # -- sync reads ---------------------------------------------------------

    def get_models(self, provider: str | None = None) -> list[Model]:
        """Last-known chat models from one provider or all providers (best-effort)."""
        if provider is not None:
            entry = self._providers.get(provider)
            return list(_safe_models(entry, False)) if entry is not None else []  # type: ignore[arg-type]
        models: list[Model] = []
        for entry in self._providers.values():
            models.extend(_safe_models(entry, False))  # type: ignore[arg-type]
        return models

    def get_all_models(self, provider: str | None = None) -> list[AnyModel]:
        if provider is not None:
            entry = self._providers.get(provider)
            return list(_safe_models(entry, True)) if entry is not None else []
        models: list[AnyModel] = []
        for entry in self._providers.values():
            models.extend(_safe_models(entry, True))
        return models

    def get_models_of_type(self, type: str, provider: str | None = None) -> list[AnyModel]:
        return [model for model in self.get_all_models(provider) if get_model_type(model) == type]

    def get_model(self, provider: str, model_id: str) -> Model | None:
        """Look up a chat model by provider and id."""
        return self.get_model_of_type("chat", provider, model_id)  # type: ignore[return-value]

    def get_model_of_type(self, type: str, provider: str, model_id: str) -> AnyModel | None:
        entry = self._providers.get(provider)
        if entry is None:
            return None
        try:
            return entry.find_model(type, model_id)
        except Exception:
            return None

    # -- dynamic catalogs -----------------------------------------------------

    def _supersede_provider_refresh(self, provider_id: str) -> int:
        generation = self._refresh_generations.get(provider_id, 0) + 1
        self._refresh_generations[provider_id] = generation
        previous = self._refresh_controllers.pop(provider_id, None)
        if previous is not None:
            previous.abort()
        return generation

    def _begin_provider_refresh(self, provider_id: str) -> tuple[int, AbortController]:
        generation = self._supersede_provider_refresh(provider_id)
        controller = AbortController()
        self._refresh_controllers[provider_id] = controller
        return generation, controller

    async def _publish(
        self, provider_id: str, generation: int, signal: AbortSignal, publication: ModelsPublication
    ) -> bool:
        lock = self._publication_locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._publication_locks[provider_id] = lock

        async def queued() -> bool:
            async with lock:
                if signal.aborted or self._refresh_generations.get(provider_id) != generation:
                    return False
                if publication.persist is None:
                    await self._models_store.delete(provider_id, signal=signal)
                elif publication.persist is not UNCHANGED:
                    await self._models_store.write(provider_id, copy.deepcopy(publication.persist), signal=signal)
                if signal.aborted or self._refresh_generations.get(provider_id) != generation:
                    return False
                if publication.update is not None:
                    publication.update()
                return True

        return await race_with_abort_signal(queued(), signal)

    async def _run_refresh_phase(
        self,
        provider: Provider,
        credential: Credential | None,
        allow_network: bool,
        force: bool | None,
        generation: int,
        signal: AbortSignal,
    ) -> None:
        stored = await self._models_store.read(provider.id, signal=signal)
        if stored is not None:
            stored = dataclasses.replace(stored, models=[m for m in stored.models if _has_known_model_type(m)])

        async def publish(publication: ModelsPublication) -> bool:
            return await self._publish(provider.id, generation, signal, publication)

        assert provider.refresh_models is not None
        await provider.refresh_models(
            RefreshModelsContext(
                publish=publish,
                signal=signal,
                allow_network=allow_network,
                credential=credential,
                stored=stored,
                force=force if allow_network else None,
                base_url=provider.base_url,
            )
        )

    async def refresh(
        self,
        *,
        allow_network: bool = True,
        providers: Iterable[str] | None = None,
        force: bool | None = None,
        signal: AbortSignal | None = None,
    ) -> ModelsRefreshResult:
        """Refresh configured dynamic providers concurrently (all when ``providers`` is omitted).

        Provider errors and cancellation are returned, never raised; static, unknown
        and unconfigured providers are skipped.
        """
        caller_signal = operation_signal(signal)
        errors: dict[str, Exception] = {}
        if caller_signal.aborted:
            return ModelsRefreshResult(True, errors)
        selected = set(providers) if providers is not None else None
        refreshable = [
            provider
            for provider in self._providers.values()
            if provider.refresh_models is not None and (selected is None or provider.id in selected)
        ]

        async def refresh_one(provider: Provider) -> None:
            generation, controller = self._begin_provider_refresh(provider.id)
            provider_signal = AbortSignal.any([caller_signal, controller.signal])

            async def operation() -> None:
                stored_credential: Credential | None = None
                credential_error: Exception | None = None
                try:
                    stored_credential = await self._read_credential(provider.id, provider_signal)
                except Exception as error:
                    credential_error = error
                # Restore cached provider state before auth resolution or network access.
                await self._run_refresh_phase(provider, stored_credential, False, None, generation, provider_signal)
                if credential_error is not None:
                    raise credential_error
                if not allow_network or provider_signal.aborted:
                    return
                credential = await self._resolve_refresh_credential(provider, stored_credential, provider_signal)
                if credential is None:
                    return
                await self._run_refresh_phase(provider, credential, True, force, generation, provider_signal)

            try:
                await race_with_abort_signal(operation(), provider_signal)
            except Exception as error:
                if not provider_signal.aborted:
                    errors[provider.id] = error
            finally:
                if self._refresh_controllers.get(provider.id) is controller:
                    del self._refresh_controllers[provider.id]

        try:
            await race_with_abort_signal(asyncio.gather(*(refresh_one(p) for p in refreshable)), caller_signal)
        except AbortError:
            if not caller_signal.aborted:
                raise
        return ModelsRefreshResult(caller_signal.aborted, dict(errors))

    async def _resolve_refresh_credential(
        self, provider: Provider, stored: Credential | None, signal: AbortSignal
    ) -> Credential | None:
        if stored is not None and stored.get("type") == "oauth":
            oauth = provider.auth.oauth
            if oauth is None:
                return None
            if now_ms() < int(stored.get("expires") or 0):
                return stored
            if signal.aborted:
                return None
            return await refresh_stored_oauth_credential(
                self._credentials,
                provider.id,
                oauth,
                lambda current: now_ms() >= int(current.get("expires") or 0),
                signal,
            )
        api_key = provider.auth.api_key
        if api_key is None:
            return None
        credential = stored if stored is not None and stored.get("type") == "api_key" else None
        result = await api_key.resolve(ctx=self._auth_context, credential=credential, signal=signal)
        if result is None:
            return None
        resolved: Credential = {"type": "api_key"}
        if result.auth.api_key is not None:
            resolved["key"] = result.auth.api_key
        if result.env:
            resolved["env"] = result.env
        return resolved

    # -- auth -----------------------------------------------------------------

    async def _read_credential(self, provider_id: str, signal: AbortSignal) -> Credential | None:
        try:
            return await self._credentials.read(provider_id, signal=signal)
        except AbortError:
            raise
        except Exception as error:
            raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error) from error

    async def _check_provider_auth(
        self, provider: Provider, credential: Credential | None, signal: AbortSignal
    ) -> AuthCheck | None:
        if credential is not None and credential.get("type") == "oauth":
            return AuthCheck("oauth", "OAuth") if provider.auth.oauth is not None else None
        api_key = provider.auth.api_key
        if api_key is None:
            return None
        if api_key.check is not None:
            try:
                return await api_key.check(
                    ctx=self._auth_context,
                    credential=credential if credential is not None and credential.get("type") == "api_key" else None,
                    signal=signal,
                )
            except AbortError:
                raise
            except Exception as error:
                raise ModelsError(
                    "auth", f"API key auth check failed for provider {provider.id}", cause=error
                ) from error
        resolution = await resolve_provider_auth(
            provider, self._credentials, self._auth_context, AuthResolutionOverrides(signal=signal)
        )
        return AuthCheck("api_key", resolution.source) if resolution is not None else None

    async def check_auth(self, provider_id: str, *, signal: AbortSignal | None = None) -> AuthCheck | None:
        """Whether a provider has complete auth configuration (without refreshing OAuth)."""
        operation = operation_signal(signal)

        async def check() -> AuthCheck | None:
            operation.throw_if_aborted()
            provider = self._providers.get(provider_id)
            if provider is None:
                return None
            return await self._check_provider_auth(
                provider, await self._read_credential(provider_id, operation), operation
            )

        return await race_with_abort_signal(check(), operation)

    async def _authenticated_providers(
        self, provider_id: str | None, signal: AbortSignal
    ) -> list[tuple[Provider, Credential | None]]:
        signal.throw_if_aborted()
        if provider_id is not None:
            providers = [p for p in (self._providers.get(provider_id),) if p is not None]
        else:
            providers = self.get_providers()

        async def check(provider: Provider) -> tuple[Provider, Credential | None, AuthCheck | None]:
            credential = await self._read_credential(provider.id, signal)
            return provider, credential, await self._check_provider_auth(provider, credential, signal)

        checks = await asyncio.gather(*(check(provider) for provider in providers))
        return [(provider, credential) for provider, credential, auth in checks if auth is not None]

    async def get_available(self, provider_id: str | None = None, *, signal: AbortSignal | None = None) -> list[Model]:
        """Chat models whose providers have complete auth configuration."""
        operation = operation_signal(signal)

        async def available() -> list[Model]:
            result: list[Model] = []
            for provider, credential in await self._authenticated_providers(provider_id, operation):
                models = list(_safe_models(provider, False))
                if provider.filter_models is not None:
                    models = list(provider.filter_models(models, credential))  # type: ignore[arg-type]
                result.extend(models)  # type: ignore[arg-type]
            return result

        return await race_with_abort_signal(available(), operation)

    async def get_available_of_type(
        self, type: str, provider_id: str | None = None, *, signal: AbortSignal | None = None
    ) -> list[AnyModel]:
        return [
            model for model in await self.get_all_available(provider_id, signal=signal) if get_model_type(model) == type
        ]

    async def get_all_available(
        self, provider_id: str | None = None, *, signal: AbortSignal | None = None
    ) -> list[AnyModel]:
        operation = operation_signal(signal)

        async def available() -> list[AnyModel]:
            result: list[AnyModel] = []
            for provider, credential in await self._authenticated_providers(provider_id, operation):
                models = list(_safe_models(provider, True))
                if provider.filter_all_models is not None:
                    result.extend(provider.filter_all_models(models, credential))
                    continue
                if provider.filter_models is None:
                    result.extend(models)
                    continue
                chat_ids = {
                    model.id
                    for model in provider.filter_models(list(_safe_models(provider, False)), credential)  # type: ignore[arg-type]
                }
                result.extend(model for model in models if not is_model_type(model, "chat") or model.id in chat_ids)
            return result

        return await race_with_abort_signal(available(), operation)

    async def get_auth(
        self,
        provider_or_model: str | AnyModel,
        *,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
        min_oauth_validity_ms: int | None = None,
        signal: AbortSignal | None = None,
    ) -> AuthResult | None:
        """Resolve provider-scoped auth (plus static ``model.headers`` when given a model).

        Returns ``None`` when the provider is unknown or unconfigured. Raises
        :class:`ModelsError` with code ``"oauth"`` when a token refresh fails and
        ``"auth"`` when key resolution or the credential store fails.
        """
        operation = operation_signal(signal)
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        result = await resolve_provider_auth(
            provider,
            self._credentials,
            self._auth_context,
            AuthResolutionOverrides(
                api_key=api_key,
                env=dict(env) if env else None,
                min_oauth_validity_ms=min_oauth_validity_ms,
                signal=operation,
            ),
        )
        if result is None or isinstance(provider_or_model, str) or not provider_or_model.headers:
            return result
        return AuthResult(
            ModelAuth(
                api_key=result.auth.api_key,
                headers=merge_headers(result.auth.headers, provider_or_model.headers),
                base_url=result.auth.base_url,
            ),
            env=result.env,
            source=result.source,
        )

    async def login(
        self,
        provider_id: str,
        type: str,
        interaction: AuthInteraction,
        options: LoginOptions | None = None,
    ) -> Credential:
        """Run a provider-owned login flow and persist the returned credential."""
        signal = operation_signal(getattr(interaction, "signal", None))
        signal.throw_if_aborted()
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        method: Any = provider.auth.oauth if type == "oauth" else provider.auth.api_key
        login = getattr(method, "login", None) if method is not None else None
        if login is None:
            raise ModelsError("auth", f"{provider.name} does not support {type} login")
        provider_interaction = ProviderAuthInteraction(interaction, signal)
        operation = login(provider_interaction, options) if type == "oauth" else login(provider_interaction)
        credential: Credential = await race_with_abort_signal(operation, signal)
        credential = {"type": type, **{k: v for k, v in credential.items() if k != "type"}}

        started = asyncio.Event()

        async def write(_current: Credential | None) -> Credential | None:
            started.set()
            return credential

        loop = asyncio.get_running_loop()
        mutation = asyncio.ensure_future(self._credentials.modify(provider_id, write, signal=signal))
        started_wait = asyncio.ensure_future(started.wait())
        abort_wait: asyncio.Future[None] = loop.create_future()
        remove = signal.add_listener(
            lambda: loop.call_soon_threadsafe(lambda: abort_wait.done() or abort_wait.set_result(None))
        )
        try:
            done, _ = await asyncio.wait({mutation, started_wait, abort_wait}, return_when=asyncio.FIRST_COMPLETED)
            if abort_wait in done and not started.is_set() and not mutation.done():
                mutation.add_done_callback(lambda task: task.cancelled() or task.exception())
                signal.throw_if_aborted()
            await mutation
        except AbortError:
            raise
        except Exception as error:
            signal.throw_if_aborted()
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error) from error
        finally:
            remove()
            started_wait.cancel()
            if not abort_wait.done():
                abort_wait.cancel()
        return credential

    async def logout(self, provider_id: str, *, signal: AbortSignal | None = None) -> None:
        """Remove the stored credential for a provider."""
        operation = operation_signal(signal)
        operation.throw_if_aborted()
        try:
            await self._credentials.delete(provider_id, signal=operation)
        except AbortError:
            raise
        except Exception as error:
            operation.throw_if_aborted()
            raise ModelsError("auth", f"Credential store delete failed for {provider_id}", cause=error) from error

    # -- requests ---------------------------------------------------------------

    def _require_provider(self, model: AnyModel) -> Provider:
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    def _require_chat_provider(self, model: Model) -> Provider:
        assert_chat_model(model)
        return self._require_provider(model)

    async def _apply_auth(self, model: Any, options: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self._require_provider(model)
        resolution = await self.get_auth(
            model, api_key=options.get("api_key"), env=options.get("env"), signal=options.get("signal")
        )
        if resolution is None:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")
        auth = resolution.auth
        explicit_key = options.get("api_key")
        api_key = explicit_key if explicit_key is not None else auth.api_key
        headers = merge_headers(auth.headers, options.get("headers"))
        transform = options.get("transform_headers")
        if transform is not None:
            headers = await maybe_await(transform(dict(headers or {})))
        env = {**(resolution.env or {}), **(options.get("env") or {})} if resolution.env or options.get("env") else None
        request_model = dataclasses.replace(model, base_url=auth.base_url) if auth.base_url else model
        request_options = {key: value for key, value in options.items() if key != "transform_headers"}
        if api_key is not None:
            request_options["api_key"] = api_key
        else:
            request_options.pop("api_key", None)
        if headers is not None:
            request_options["headers"] = headers
        if env is not None:
            request_options["env"] = env
        return request_model, request_options

    def stream(
        self,
        model: Model,
        context: Context | Mapping[str, Any] | TranscriptContext,
        options: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> AssistantMessageEventStream:
        """Stream a response with the model API's full option set.

        Request failures never raise: they end the stream with an ``error`` event and
        an assistant message whose ``stop_reason`` is ``"error"`` or ``"aborted"``.
        """
        transcript = normalize_context(context)
        opts = _merge_options(options, kwargs)

        async def setup() -> AssistantMessageEventStream:
            provider = self._require_chat_provider(model)
            request_model, request_options = await self._apply_auth(model, opts)
            return provider.stream(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete(
        self,
        model: Model,
        context: Context | Mapping[str, Any] | TranscriptContext,
        options: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> AssistantMessage:
        """Like :meth:`stream` but returns only the final assistant message."""
        return await self.stream(model, context, options, **kwargs).result()

    def stream_simple(
        self,
        model: Model,
        context: Context | Mapping[str, Any] | TranscriptContext,
        options: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> AssistantMessageEventStream:
        """Stream with provider-neutral options (``reasoning="medium"``, ``tool_choice``...)."""
        transcript = normalize_context(context)
        opts = _merge_options(options, kwargs)

        async def setup() -> AssistantMessageEventStream:
            provider = self._require_chat_provider(model)
            request_model, request_options = await self._apply_auth(model, opts)
            return provider.stream_simple(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete_simple(
        self,
        model: Model,
        context: Context | Mapping[str, Any] | TranscriptContext,
        options: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> AssistantMessage:
        return await self.stream_simple(model, context, options, **kwargs).result()

    def stream_deferred(
        self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> AssistantMessageEventStream:
        opts = _merge_options(options, kwargs)

        async def setup() -> AssistantMessageEventStream:
            provider = self._require_chat_provider(model)
            if provider.fetch_deferred is None:
                raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
            request_model, request_options = await self._apply_auth(model, opts)
            return provider.fetch_deferred(request_model, handle, request_options)

        return lazy_stream(model, setup)

    async def fetch_deferred(
        self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> AssistantMessage:
        return await self.stream_deferred(model, handle, options, **kwargs).result()

    async def cancel_deferred(
        self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> None:
        provider = self._require_chat_provider(model)
        if provider.cancel_deferred is None:
            raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
        request_model, request_options = await self._apply_auth(model, _merge_options(options, kwargs))
        await provider.cancel_deferred(request_model, handle, request_options)

    async def generate_images(
        self, model: ImageModel, context: ImagesContext, options: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> AssistantImages:
        """Generate images through the owning provider. Never raises."""
        opts = _merge_options(options, kwargs)
        try:
            assert_image_model(model)
            provider = self._require_provider(model)
            if provider.generate_images is None:
                raise ModelsError("provider", f"Provider {model.provider} does not support image generation")
            request_model, request_options = await self._apply_auth(model, opts)
            return await provider.generate_images(request_model, context, request_options)
        except Exception as error:
            signal = opts.get("signal")
            return image_error_result(model, error, bool(signal is not None and signal.aborted))

    async def classify(
        self,
        model: ClassifierModel,
        context: ClassifierContext,
        options: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> ClassifierResult:
        """Classify structured state through the owning provider. Never raises."""
        opts = _merge_options(options, kwargs)
        try:
            assert_classifier_model(model)
            assert_classifier_input_supported(model, context)
            provider = self._require_provider(model)
            if provider.classify is None:
                raise ModelsError("provider", f"Provider {model.provider} does not support classification")
            request_model, request_options = await self._apply_auth(model, opts)
            return await provider.classify(request_model, context, request_options)
        except Exception as error:
            signal = opts.get("signal")
            return classifier_error_result(model, error, bool(signal is not None and signal.aborted))


#: ``Models`` is mutable in Python (``set_provider``/``delete_provider``/``clear_providers``).
MutableModels = Models


def create_models(
    *,
    credentials: CredentialStore | None = None,
    models_store: ModelsStore | None = None,
    auth_context: AuthContext | None = None,
) -> Models:
    """Create an empty :class:`Models` collection."""
    return Models(credentials=credentials, models_store=models_store, auth_context=auth_context)


# ---------------------------------------------------------------------------
# create_provider
# ---------------------------------------------------------------------------


class _ComposedProvider(Provider):
    """Provider assembled by :func:`create_provider`."""

    def __init__(
        self,
        *,
        id: str,
        name: str,
        base_url: str | None,
        headers: dict[str, str | None] | None,
        auth: ProviderAuth,
        models: Sequence[AnyModel] | Callable[[], Sequence[AnyModel]],
        api_single: Any,
        api_by_name: Mapping[str, Any] | None,
        images: Mapping[str, Any] | None,
        classifiers: Mapping[str, Any] | None,
        fetch_models: Callable[[RefreshModelsContext], Awaitable[Sequence[AnyModel]]] | None,
        filter_models: Any,
        filter_all_models: Any,
    ) -> None:
        self.id = id
        self.name = name
        self.base_url = base_url
        self.headers = headers
        self.auth = auth
        self._baseline_source = models
        self._baseline: list[AnyModel] | None = None if callable(models) else list(models)
        self._dynamic: list[AnyModel] = []
        self._merged: list[AnyModel] | None = None
        self._chat: list[Model] | None = None
        self._index: dict[tuple[str, str], AnyModel] | None = None
        self._api_single = api_single
        self._api_by_name = dict(api_by_name) if api_by_name else None
        self._images = dict(images) if images else None
        self._classifiers = dict(classifiers) if classifiers else None
        self._fetch_models = fetch_models
        self.filter_models = filter_models
        self.filter_all_models = filter_all_models
        self.get_all_models = self._get_all_models
        if fetch_models is not None:
            self.refresh_models = self._refresh_models
        streams = [api_single] if api_single is not None else list((api_by_name or {}).values())
        if any(getattr(entry, "fetch_deferred", None) is not None for entry in streams):
            self.fetch_deferred = self._fetch_deferred
        if any(getattr(entry, "cancel_deferred", None) is not None for entry in streams):
            self.cancel_deferred = self._cancel_deferred
        if self._images:
            self.generate_images = self._generate_images
        if self._classifiers:
            self.classify = self._classify

    # catalog ------------------------------------------------------------------

    def _current(self) -> list[AnyModel]:
        merged = self._merged
        if merged is None:
            if self._baseline is None:
                self._baseline = list(self._baseline_source())  # type: ignore[operator]
            merged = list(self._baseline)
            positions = {(get_model_type(model), model.id): index for index, model in enumerate(merged)}
            for model in self._dynamic:
                key = (get_model_type(model), model.id)
                index = positions.get(key)
                if index is not None:
                    merged[index] = model
                else:
                    positions[key] = len(merged)
                    merged.append(model)
            self._merged = merged
            self._chat = [model for model in merged if get_model_type(model) == "chat"]  # type: ignore[misc]
            index_map: dict[tuple[str, str], AnyModel] = {}
            for model in merged:
                index_map.setdefault((get_model_type(model), model.id), model)
            self._index = index_map
        return merged

    def _set_dynamic(self, models: list[AnyModel]) -> None:
        self._dynamic = models
        self._merged = None
        self._chat = None
        self._index = None

    def get_models(self) -> Sequence[Model]:
        self._current()
        return list(self._chat or [])

    def _get_all_models(self) -> Sequence[AnyModel]:
        return list(self._current())

    def find_model(self, type: str, model_id: str) -> AnyModel | None:
        self._current()
        assert self._index is not None
        return self._index.get((type, model_id))

    async def _refresh_models(self, context: RefreshModelsContext) -> None:
        assert self._fetch_models is not None
        if context.stored is not None:
            restored = [model for model in context.stored.models if model.provider == self.id]
            if not await context.publish(ModelsPublication(update=lambda: self._set_dynamic(restored))):
                return
        if not context.allow_network or context.signal.aborted:
            return
        fetched = await self._fetch_models(context)
        if context.signal.aborted:
            return
        refreshed = [model for model in fetched if _has_known_model_type(model)]
        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=refreshed, checked_at=now_ms()),
                update=lambda: self._set_dynamic(refreshed),
            )
        )

    # operations ---------------------------------------------------------------

    def _api_for(self, model: Model) -> Any:
        if self._api_single is not None:
            return self._api_single
        return (self._api_by_name or {}).get(model.api)

    def _missing_api(self, model: Model) -> AssistantMessageEventStream:
        async def fail() -> AssistantMessageEventStream:
            raise ModelsError("stream", f'Provider {self.id} has no API implementation for "{model.api}"')

        return lazy_stream(model, fail)

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        implementation = self._api_for(model)
        if implementation is None:
            return self._missing_api(model)
        return implementation.stream(model, context, options)

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        implementation = self._api_for(model)
        if implementation is None:
            return self._missing_api(model)
        return implementation.stream_simple(model, context, options)

    def _fetch_deferred(
        self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        implementation = self._api_for(model)
        fetch = getattr(implementation, "fetch_deferred", None) if implementation is not None else None
        if fetch is None:

            async def fail() -> AssistantMessageEventStream:
                raise ModelsError(
                    "provider", f'Provider {self.id} does not support deferred responses for "{model.api}"'
                )

            return lazy_stream(model, fail)
        return fetch(model, handle, options)

    async def _cancel_deferred(
        self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None
    ) -> None:
        implementation = self._api_for(model)
        cancel = getattr(implementation, "cancel_deferred", None) if implementation is not None else None
        if cancel is None:
            raise ModelsError("provider", f'Provider {self.id} cannot cancel deferred responses for "{model.api}"')
        await cancel(model, handle, options)

    async def _generate_images(
        self, model: ImageModel, context: ImagesContext, options: Mapping[str, Any] | None = None
    ) -> AssistantImages:
        implementation = (self._images or {}).get(model.api)
        if implementation is None:
            return image_error_result(
                model,
                ModelsError("provider", f'Provider {self.id} has no image generation implementation for "{model.api}"'),
            )
        return await implementation.generate_images(model, context, options)

    async def _classify(
        self, model: ClassifierModel, context: ClassifierContext, options: Mapping[str, Any] | None = None
    ) -> ClassifierResult:
        implementation = (self._classifiers or {}).get(model.api)
        if implementation is None:
            return classifier_error_result(
                model, ModelsError("provider", f'Provider {self.id} has no classifier implementation for "{model.api}"')
            )
        return await implementation.classify(model, context, options)


def create_provider(
    *,
    id: str,
    auth: ProviderAuth,
    models: Sequence[AnyModel] | Callable[[], Sequence[AnyModel]] = (),
    api: Any = None,
    images: Mapping[str, Any] | None = None,
    classifiers: Mapping[str, Any] | None = None,
    name: str | None = None,
    base_url: str | None = None,
    headers: dict[str, str | None] | None = None,
    fetch_models: Callable[[RefreshModelsContext], Awaitable[Sequence[AnyModel]]] | None = None,
    filter_models: Callable[[Sequence[Model], Credential | None], Sequence[Model]] | None = None,
    filter_all_models: Callable[[Sequence[AnyModel], Credential | None], Sequence[AnyModel]] | None = None,
) -> Provider:
    """Build a provider from parts.

    ``models`` is the static baseline catalog (a sequence, or a zero-argument callable
    evaluated on first access so catalogs load lazily). ``api`` is a single
    ``ProviderStreams`` implementation for all chat models or a mapping keyed by
    ``model.api`` for mixed-API providers. ``images`` and
    ``classifiers`` map ``model.api`` to one-shot implementations. At least one
    concrete implementation is required.
    """
    single = api if api is not None and callable(getattr(api, "stream", None)) else None
    by_name = api if single is None and isinstance(api, Mapping) else None
    stream_impls = (
        [single] if single is not None else [entry for entry in (by_name or {}).values() if entry is not None]
    )
    image_impls = [entry for entry in (images or {}).values() if entry is not None]
    classifier_impls = [entry for entry in (classifiers or {}).values() if entry is not None]
    if not stream_impls and not image_impls and not classifier_impls:
        raise ValueError(f'Provider {id}: at least one of "api", "images", or "classifiers" is required.')
    return _ComposedProvider(
        id=id,
        name=name or id,
        base_url=base_url,
        headers=headers,
        auth=auth,
        models=models,
        api_single=single,
        api_by_name=by_name,
        images=images,
        classifiers=classifiers,
        fetch_models=fetch_models,
        filter_models=filter_models,
        filter_all_models=filter_all_models,
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def has_api(model: AnyModel, api: str) -> bool:
    """True when ``model`` is a chat model using ``api``. Non-chat models never match."""
    return is_model_type(model, "chat") and model.api == api


def calculate_cost(model: AnyModel, usage: Usage) -> UsageCost:
    """Fill in ``usage.cost`` from the model's pricing (per million tokens) and return it.

    Request-wide pricing tiers apply when total input exceeds their threshold.
    Anthropic charges twice the base input rate for 1h cache writes.
    """
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates: Any = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers or ():
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above
    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    cost = usage.cost
    cost.input = (rates.input / 1_000_000) * usage.input
    cost.output = (rates.output / 1_000_000) * usage.output
    cost.cache_read = (rates.cache_read / 1_000_000) * usage.cache_read
    cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write
    return cost


_EXTENDED_THINKING_LEVELS: tuple[ModelThinkingLevel, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    """Thinking levels the model exposes. ``xhigh``/``max`` need an explicit map entry."""
    if not model.reasoning:
        return ["off"]
    level_map = model.thinking_level_map or {}
    levels: list[ModelThinkingLevel] = []
    for level in _EXTENDED_THINKING_LEVELS:
        present = level in level_map
        mapped = level_map.get(level)
        if present and mapped is None:
            continue
        if level in ("xhigh", "max") and not present:
            continue
        levels.append(level)
    return levels


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    """The closest supported level, preferring higher levels first."""
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    if level not in _EXTENDED_THINKING_LEVELS:
        return available[0] if available else "off"
    requested = _EXTENDED_THINKING_LEVELS.index(level)
    for candidate in _EXTENDED_THINKING_LEVELS[requested:]:
        if candidate in available:
            return candidate
    for candidate in reversed(_EXTENDED_THINKING_LEVELS[:requested]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def models_are_equal(a: AnyModel | None, b: AnyModel | None) -> bool:
    """Whether two models are the same catalog entry (type, id and provider)."""
    if a is None or b is None:
        return False
    return get_model_type(a) == get_model_type(b) and a.id == b.id and a.provider == b.provider
