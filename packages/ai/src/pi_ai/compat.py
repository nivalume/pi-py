"""Compatibility entry point preserving the old global pi-ai API surface.

API-dispatch :func:`stream`/:func:`complete` with env API key injection, the API
registry, generated catalog reads (:func:`get_model`/:func:`get_models`/
:func:`get_providers`), per-API lazy stream wrappers, and image generation.

Existing apps switch imports from ``pi_ai`` to ``pi_ai.compat`` unchanged (this
module re-exports the whole root API); new code uses :func:`pi_ai.create_models`
and the provider factories in :mod:`pi_ai.providers`.
"""

from __future__ import annotations

import random
import string
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from . import *  # noqa: F403  (the compat entry point re-exports the root API)
from . import __all__ as _root_all
from .api.lazy import LazyApi
from .api.registry import (
    anthropic_messages_api,
    azure_openai_responses_api,
    bedrock_converse_stream_api,
    google_generative_ai_api,
    google_vertex_api,
    mistral_conversations_api,
    openai_codex_responses_api,
    openai_completions_api,
    openai_responses_api,
    openrouter_images_api,
    pi_messages_api,
)
from .env_api_keys import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_FEDERATION_RULE_ID_ENV,
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
    ANTHROPIC_OAUTH_TOKEN_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV,
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV,
    ANTHROPIC_WORKSPACE_ID_ENV,
    PROVIDER_API_KEY_ENV_VARS,
    find_env_keys,
    get_env_api_key,
)
from .event_stream import AssistantMessageEventStream
from .models import Models, Provider
from .providers.all import (
    BuiltinProvider,
    builtin_models,
    get_builtin_image_model,
    get_builtin_image_models,
    get_builtin_model,
    get_builtin_models,
    get_builtin_providers,
)
from .providers.catalog import BUILTIN_PROVIDERS, load_catalog
from .providers.faux import FauxModelDefinition, FauxProviderState, FauxResponseStep, create_faux_core
from .types import (
    AssistantImages,
    AssistantMessage,
    Context,
    ImageModel,
    ImagesContext,
    Model,
    TranscriptContext,
    now_ms,
)
from .utils.diagnostics import format_thrown_value
from .utils.transcript import normalize_context

__all__ = [
    *_root_all,
    # lazy API wrappers
    "anthropic_messages_api",
    "azure_openai_responses_api",
    "bedrock_converse_stream_api",
    "google_generative_ai_api",
    "google_vertex_api",
    "mistral_conversations_api",
    "openai_codex_responses_api",
    "openai_completions_api",
    "openai_responses_api",
    "pi_messages_api",
    # env API keys
    "ANTHROPIC_AUTH_TOKEN_ENV",
    "ANTHROPIC_OAUTH_TOKEN_ENV",
    "ANTHROPIC_API_KEY_ENV",
    "ANTHROPIC_FEDERATION_RULE_ID_ENV",
    "ANTHROPIC_ORGANIZATION_ID_ENV",
    "ANTHROPIC_SERVICE_ACCOUNT_ID_ENV",
    "ANTHROPIC_IDENTITY_TOKEN_FILE_ENV",
    "ANTHROPIC_WORKSPACE_ID_ENV",
    "PROVIDER_API_KEY_ENV_VARS",
    "find_env_keys",
    "get_env_api_key",
    # static catalog reads
    "BuiltinProvider",
    "BuiltinImageProvider",
    "get_model",
    "get_models",
    "get_providers",
    "get_image_model",
    "get_image_models",
    "get_image_providers",
    # API registry
    "ApiStreamFunction",
    "ApiStreamSimpleFunction",
    "ApiProvider",
    "register_api_provider",
    "get_api_provider",
    "get_api_providers",
    "unregister_api_providers",
    "register_builtin_api_providers",
    "reset_api_providers",
    "FauxProviderRegistration",
    "register_faux_provider",
    # api dispatch
    "stream",
    "complete",
    "stream_simple",
    "complete_simple",
    # images
    "ImagesApiFunction",
    "ImagesApiProvider",
    "register_images_api_provider",
    "get_images_api_provider",
    "generate_images",
    "generate_images_openrouter",
    "register_builtin_images_api_providers",
    # legacy per-API aliases
    "stream_anthropic",
    "stream_simple_anthropic",
    "stream_azure_openai_responses",
    "stream_simple_azure_openai_responses",
    "stream_google",
    "stream_simple_google",
    "stream_google_vertex",
    "stream_simple_google_vertex",
    "stream_mistral",
    "stream_simple_mistral",
    "stream_openai_codex_responses",
    "stream_simple_openai_codex_responses",
    "stream_openai_completions",
    "stream_simple_openai_completions",
    "stream_openai_responses",
    "stream_simple_openai_responses",
]


def _merge_options(options: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> dict[str, Any] | None:
    if not kwargs:
        return dict(options) if options is not None else None
    return {**(options or {}), **kwargs}


# ---------------------------------------------------------------------------
# Static catalog reads
# ---------------------------------------------------------------------------

#: Deprecated static catalog read. Use ``pi_ai.providers.all.get_builtin_model`` or ``Models.get_model()``.
get_model = get_builtin_model

#: Deprecated static catalog read. Use ``pi_ai.providers.all.get_builtin_models`` or ``Models.get_models()``.
get_models = get_builtin_models

#: Deprecated static catalog read. Use ``pi_ai.providers.all.get_builtin_providers`` or ``Models.get_providers()``.
get_providers = get_builtin_providers

#: Built-in providers with at least one image model in the generated catalog.
BuiltinImageProvider: TypeAlias = str

_image_providers: list[str] | None = None


def get_image_model(provider: str, model_id: str) -> ImageModel | None:
    """Deprecated static catalog read. Use ``get_builtin_image_model`` or ``Models.get_model_of_type("image", ...)``."""
    return get_builtin_image_model(provider, model_id)


def get_image_providers() -> list[str]:
    """Deprecated static catalog read. Use ``Models.get_providers()``."""
    global _image_providers
    if _image_providers is None:
        _image_providers = [provider for provider in BUILTIN_PROVIDERS if load_catalog(provider)["image"]]
    return list(_image_providers)


def get_image_models(provider: str) -> list[ImageModel]:
    """Deprecated static catalog read. Use ``get_builtin_image_models`` or ``Models.get_models_of_type("image")``."""
    return get_builtin_image_models(provider)


# ---------------------------------------------------------------------------
# API registry
# ---------------------------------------------------------------------------

ApiStreamFunction: TypeAlias = Callable[
    [Model, TranscriptContext, "Mapping[str, Any] | None"], AssistantMessageEventStream
]
ApiStreamSimpleFunction: TypeAlias = ApiStreamFunction


@dataclass(slots=True)
class ApiProvider:
    """A chat API implementation registered under its ``api`` id."""

    api: str
    stream: ApiStreamFunction
    stream_simple: ApiStreamSimpleFunction


@dataclass(slots=True)
class _RegisteredApiProvider:
    provider: ApiProvider
    source_id: str | None


_api_provider_registry: dict[str, _RegisteredApiProvider] = {}


def _field(provider: Any, name: str) -> Any:
    return provider[name] if isinstance(provider, Mapping) else getattr(provider, name)


def _wrap_stream(api: str, stream_fn: Callable[..., AssistantMessageEventStream]) -> ApiStreamFunction:
    def wrapped(
        model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return stream_fn(model, context, options)

    return wrapped


def register_api_provider(provider: ApiProvider | Mapping[str, Any] | Any, source_id: str | None = None) -> None:
    """Register (or replace) the implementation for ``provider.api``.

    ``provider`` is an :class:`ApiProvider`, any object with ``api``/``stream``/``stream_simple``
    attributes, or a mapping with those keys. ``source_id`` groups registrations for
    :func:`unregister_api_providers`.
    """
    api = _field(provider, "api")
    _api_provider_registry[api] = _RegisteredApiProvider(
        ApiProvider(
            api=api,
            stream=_wrap_stream(api, _field(provider, "stream")),
            stream_simple=_wrap_stream(api, _field(provider, "stream_simple")),
        ),
        source_id,
    )


def get_api_provider(api: str) -> ApiProvider | None:
    entry = _api_provider_registry.get(api)
    return entry.provider if entry is not None else None


def get_api_providers() -> list[ApiProvider]:
    return [entry.provider for entry in _api_provider_registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """Remove every registration made with ``source_id``."""
    for api in [api for api, entry in _api_provider_registry.items() if entry.source_id == source_id]:
        del _api_provider_registry[api]


def _clear_api_providers() -> None:
    _api_provider_registry.clear()


class FauxProviderRegistration:
    """Handle returned by :func:`register_faux_provider`."""

    def __init__(self, core: Any, source_id: str) -> None:
        self._core = core
        self._source_id = source_id
        self.api: str = core.api
        self.models: list[Model] = core.models
        self.state: FauxProviderState = core.state

    def get_model(self, model_id: str | None = None) -> Model | None:
        """The first faux model, or the one with ``model_id`` (``None`` if unknown)."""
        return self._core.get_model(model_id)

    def set_responses(self, responses: Sequence[FauxResponseStep]) -> None:
        self._core.set_responses(responses)

    def append_responses(self, responses: Sequence[FauxResponseStep]) -> None:
        self._core.append_responses(responses)

    def get_pending_response_count(self) -> int:
        return self._core.get_pending_response_count()

    def unregister(self) -> None:
        unregister_api_providers(self._source_id)


def _random_base36(length: int) -> str:
    return "".join(random.choices(string.digits + string.ascii_lowercase, k=length))


def register_faux_provider(
    *,
    api: str | None = None,
    provider: str | None = None,
    models: Sequence[FauxModelDefinition | Mapping[str, Any]] | None = None,
    deferred: Mapping[str, Any] | None = None,
    tokens_per_second: float | None = None,
    token_size: Mapping[str, int] | None = None,
) -> FauxProviderRegistration:
    """Register a scripted in-memory API implementation in the global API registry."""
    core = create_faux_core(
        api=api,
        provider=provider,
        models=models,
        deferred=deferred,
        tokens_per_second=tokens_per_second,
        token_size=token_size,
    )
    source_id = f"faux-provider-{_random_base36(8)}"
    register_api_provider(ApiProvider(api=core.api, stream=core.stream, stream_simple=core.stream_simple), source_id)
    return FauxProviderRegistration(core, source_id)


_BUILTIN_APIS: tuple[tuple[str, LazyApi], ...] = (
    ("anthropic-messages", anthropic_messages_api()),
    ("openai-completions", openai_completions_api()),
    ("openai-responses", openai_responses_api()),
    ("openai-codex-responses", openai_codex_responses_api()),
    ("azure-openai-responses", azure_openai_responses_api()),
    ("google-generative-ai", google_generative_ai_api()),
    ("google-vertex", google_vertex_api()),
    ("mistral-conversations", mistral_conversations_api()),
    ("bedrock-converse-stream", bedrock_converse_stream_api()),
    ("pi-messages", pi_messages_api()),
)

_builtin_api_provider_instances: dict[str, ApiProvider | None] = {}


def register_builtin_api_providers() -> None:
    """Register the built-in API implementations without clobbering existing entries.

    Compat may load after a test or extension has already registered an override for
    a built-in api id.
    """
    for api, streams in _BUILTIN_APIS:
        if get_api_provider(api) is None:
            register_api_provider(ApiProvider(api=api, stream=streams.stream, stream_simple=streams.stream_simple))
        _builtin_api_provider_instances[api] = get_api_provider(api)


def reset_api_providers() -> None:
    """Drop every registration and re-register the built-in API implementations."""
    _clear_api_providers()
    _builtin_api_provider_instances.clear()
    register_builtin_api_providers()


register_builtin_api_providers()


# ---------------------------------------------------------------------------
# API dispatch with env API key injection
# ---------------------------------------------------------------------------

_AMBIENT_AUTH_MARKER = "<authenticated>"
_compat_models_instance: Models | None = None


def _compat_models() -> Models:
    global _compat_models_instance
    if _compat_models_instance is None:
        _compat_models_instance = builtin_models()
    return _compat_models_instance


def _has_explicit_api_key(api_key: Any) -> bool:
    return isinstance(api_key, str) and len(api_key.strip()) > 0


def _with_env_api_key(model: Model, options: dict[str, Any] | None) -> dict[str, Any] | None:
    if options is not None and _has_explicit_api_key(options.get("api_key")):
        return options
    api_key = get_env_api_key(model.provider, options.get("env") if options else None)
    if not api_key or api_key == _AMBIENT_AUTH_MARKER:
        return options
    return {**(options or {}), "api_key": api_key}


def _has_resolved_cloudflare_auth(options: Mapping[str, Any] | None) -> bool:
    if not options:
        return False
    headers = options.get("headers") or {}
    return _has_explicit_api_key(options.get("api_key")) or isinstance(headers.get("cf-aig-authorization"), str)


def _get_builtin_provider_for_model(model: Model) -> Provider | None:
    if get_api_provider(model.api) is not _builtin_api_provider_instances.get(model.api):
        return None
    provider = _compat_models().get_provider(model.provider)
    if provider is None:
        return None
    return provider if any(candidate.api == model.api for candidate in provider.get_models()) else None


def _resolve_api_provider(api: str) -> ApiProvider:
    provider = get_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


def stream(
    model: Model,
    context: Context | Mapping[str, Any] | TranscriptContext,
    options: Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> AssistantMessageEventStream:
    """Stream through the implementation registered for ``model.api``.

    Built-in providers' models go through their provider (so provider request shaping
    applies); the API key comes from ``options["api_key"]`` or the provider's env var.
    Raises :class:`ValueError` when no implementation is registered for ``model.api``.
    """
    transcript = normalize_context(context)
    opts = _merge_options(options, kwargs)
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(opts):
            return _compat_models().stream(model, transcript, opts)
        return builtin_provider.stream(model, transcript, _with_env_api_key(model, opts))
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, transcript, _with_env_api_key(model, opts))


async def complete(
    model: Model,
    context: Context | Mapping[str, Any] | TranscriptContext,
    options: Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> AssistantMessage:
    return await stream(model, context, options, **kwargs).result()


def stream_simple(
    model: Model,
    context: Context | Mapping[str, Any] | TranscriptContext,
    options: Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> AssistantMessageEventStream:
    """Like :func:`stream` with provider-neutral options (``reasoning="medium"``, ``tool_choice``...)."""
    transcript = normalize_context(context)
    opts = _merge_options(options, kwargs)
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(opts):
            return _compat_models().stream_simple(model, transcript, opts)
        return builtin_provider.stream_simple(model, transcript, _with_env_api_key(model, opts))
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, transcript, _with_env_api_key(model, opts))


async def complete_simple(
    model: Model,
    context: Context | Mapping[str, Any] | TranscriptContext,
    options: Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> AssistantMessage:
    return await stream_simple(model, context, options, **kwargs).result()


# ---------------------------------------------------------------------------
# Image generation registry
# ---------------------------------------------------------------------------

ImagesApiFunction: TypeAlias = Callable[
    [ImageModel, ImagesContext, "Mapping[str, Any] | None"], Awaitable[AssistantImages]
]


@dataclass(slots=True)
class ImagesApiProvider:
    """An image-generation implementation registered under its ``api`` id."""

    api: str
    generate_images: ImagesApiFunction


@dataclass(slots=True)
class _RegisteredImagesApiProvider:
    provider: ImagesApiProvider
    source_id: str | None


_images_api_provider_registry: dict[str, _RegisteredImagesApiProvider] = {}


def _wrap_generate_images(api: str, generate: Callable[..., Awaitable[AssistantImages]]) -> ImagesApiFunction:
    async def wrapped(
        model: ImageModel, context: ImagesContext, options: Mapping[str, Any] | None = None
    ) -> AssistantImages:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return await generate(model, context, options)

    return wrapped


def register_images_api_provider(
    provider: ImagesApiProvider | Mapping[str, Any] | Any, source_id: str | None = None
) -> None:
    """Register (or replace) the image implementation for ``provider.api``."""
    api = _field(provider, "api")
    _images_api_provider_registry[api] = _RegisteredImagesApiProvider(
        ImagesApiProvider(api=api, generate_images=_wrap_generate_images(api, _field(provider, "generate_images"))),
        source_id,
    )


def get_images_api_provider(api: str) -> ImagesApiProvider | None:
    entry = _images_api_provider_registry.get(api)
    return entry.provider if entry is not None else None


_openrouter_images = openrouter_images_api()


async def generate_images_openrouter(
    model: ImageModel, context: ImagesContext, options: Mapping[str, Any] | None = None
) -> AssistantImages:
    """OpenRouter image generation, loaded on first use. Load and request failures become error results."""
    try:
        return await _openrouter_images.generate_images(model, context, options)
    except Exception as error:
        return AssistantImages(
            api=model.api,
            provider=model.provider,
            model=model.id,
            output=[],
            stop_reason="error",
            error_message=format_thrown_value(error),
            timestamp=now_ms(),
        )


def register_builtin_images_api_providers() -> None:
    register_images_api_provider(ImagesApiProvider(api="openrouter-images", generate_images=generate_images_openrouter))


register_builtin_images_api_providers()


async def generate_images(
    model: ImageModel,
    context: ImagesContext,
    options: Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> AssistantImages:
    """Global image generation dispatched on ``model.api`` through the images API registry.

    Auth must be passed explicitly via ``api_key``; prefer ``Models.generate_images()``,
    which resolves provider auth. Raises :class:`ValueError` for an unregistered api.
    """
    provider = get_images_api_provider(model.api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {model.api}")
    return await provider.generate_images(model, context, _merge_options(options, kwargs))


# ---------------------------------------------------------------------------
# Legacy per-API stream aliases
# ---------------------------------------------------------------------------


def _legacy_alias(streams: LazyApi, simple: bool, name: str, module: str) -> ApiStreamFunction:
    method = streams.stream_simple if simple else streams.stream
    target = "stream_simple" if simple else "stream"

    def alias(
        model: Model,
        context: Context | Mapping[str, Any] | TranscriptContext,
        options: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> AssistantMessageEventStream:
        return method(model, normalize_context(context), _merge_options(options, kwargs))

    alias.__name__ = alias.__qualname__ = name
    alias.__doc__ = f"Deprecated: use ``pi_ai.api.{module}.{target}`` or ``{module}_api().{target}``."
    return alias


_anthropic_messages_streams = anthropic_messages_api()
_azure_openai_responses_streams = azure_openai_responses_api()
_google_generative_ai_streams = google_generative_ai_api()
_google_vertex_streams = google_vertex_api()
_mistral_conversations_streams = mistral_conversations_api()
_openai_codex_responses_streams = openai_codex_responses_api()
_openai_completions_streams = openai_completions_api()
_openai_responses_streams = openai_responses_api()

stream_anthropic = _legacy_alias(_anthropic_messages_streams, False, "stream_anthropic", "anthropic_messages")
stream_simple_anthropic = _legacy_alias(
    _anthropic_messages_streams, True, "stream_simple_anthropic", "anthropic_messages"
)
stream_azure_openai_responses = _legacy_alias(
    _azure_openai_responses_streams, False, "stream_azure_openai_responses", "azure_openai_responses"
)
stream_simple_azure_openai_responses = _legacy_alias(
    _azure_openai_responses_streams, True, "stream_simple_azure_openai_responses", "azure_openai_responses"
)
stream_google = _legacy_alias(_google_generative_ai_streams, False, "stream_google", "google_generative_ai")
stream_simple_google = _legacy_alias(
    _google_generative_ai_streams, True, "stream_simple_google", "google_generative_ai"
)
stream_google_vertex = _legacy_alias(_google_vertex_streams, False, "stream_google_vertex", "google_vertex")
stream_simple_google_vertex = _legacy_alias(
    _google_vertex_streams, True, "stream_simple_google_vertex", "google_vertex"
)
stream_mistral = _legacy_alias(_mistral_conversations_streams, False, "stream_mistral", "mistral_conversations")
stream_simple_mistral = _legacy_alias(
    _mistral_conversations_streams, True, "stream_simple_mistral", "mistral_conversations"
)
stream_openai_codex_responses = _legacy_alias(
    _openai_codex_responses_streams, False, "stream_openai_codex_responses", "openai_codex_responses"
)
stream_simple_openai_codex_responses = _legacy_alias(
    _openai_codex_responses_streams, True, "stream_simple_openai_codex_responses", "openai_codex_responses"
)
stream_openai_completions = _legacy_alias(
    _openai_completions_streams, False, "stream_openai_completions", "openai_completions"
)
stream_simple_openai_completions = _legacy_alias(
    _openai_completions_streams, True, "stream_simple_openai_completions", "openai_completions"
)
stream_openai_responses = _legacy_alias(_openai_responses_streams, False, "stream_openai_responses", "openai_responses")
stream_simple_openai_responses = _legacy_alias(
    _openai_responses_streams, True, "stream_simple_openai_responses", "openai_responses"
)
