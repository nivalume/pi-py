"""Azure OpenAI provider."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from ..api.lazy import lazy_stream
from ..api.registry import azure_openai_responses_api, openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..event_stream import AssistantMessageEventStream
from ..models import Provider, create_provider
from ..types import Model, TranscriptContext
from ..utils.callbacks import maybe_await
from .catalog import catalog_models

__all__ = ["azure_provider"]


def _resolve_azure_model(model: Model, options: Mapping[str, Any] | None) -> Model:
    from ..api.azure_openai_config import resolve_azure_base_url

    return dataclasses.replace(model, base_url=resolve_azure_base_url(model, options))


def _with_deployment_name(model: Model, options: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Send the deployment name as the request's model, keeping ``model.id`` as the catalog id."""
    from ..api.azure_openai_config import resolve_deployment_name

    deployment_name = resolve_deployment_name(model, options)
    if deployment_name == model.id:
        return options
    user_on_payload = options.get("on_payload") if options else None

    async def on_payload(payload: Any, payload_model: Any) -> Any:
        params = {**payload, "model": deployment_name}
        if user_on_payload is not None:
            replaced = await maybe_await(user_on_payload(params, payload_model))
            if replaced is not None:
                return replaced
        return params

    return {**(options or {}), "on_payload": on_payload}


class _AzureStreams:
    """Resolve the Azure endpoint and deployment before dispatch.

    Resolution runs inside ``lazy_stream`` so an unconfigured endpoint errors on the
    stream instead of raising out of ``stream()``.
    """

    __slots__ = ("_streams",)

    def __init__(self, streams: Any) -> None:
        self._streams = streams

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        async def setup() -> AssistantMessageEventStream:
            return self._streams.stream(
                _resolve_azure_model(model, options), context, _with_deployment_name(model, options)
            )

        return lazy_stream(model, setup)

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        async def setup() -> AssistantMessageEventStream:
            return self._streams.stream_simple(
                _resolve_azure_model(model, options), context, _with_deployment_name(model, options)
            )

        return lazy_stream(model, setup)

    def __repr__(self) -> str:
        return f"azure_streams({self._streams!r})"


def azure_streams(streams: Any) -> Any:
    return _AzureStreams(streams)


def azure_provider() -> Provider:
    """Azure OpenAI (Responses and Chat Completions), authenticated with ``AZURE_OPENAI_API_KEY``.

    Catalog models ship without a base URL: the endpoint comes from ``AZURE_OPENAI_BASE_URL``,
    ``AZURE_OPENAI_RESOURCE_NAME`` or the matching request options.
    """
    return create_provider(
        id="azure",
        name="Azure",
        auth=ProviderAuth(api_key=env_api_key_auth("Azure OpenAI API key", ["AZURE_OPENAI_API_KEY"])),
        models=lambda: catalog_models("azure"),
        api={
            "azure-openai-responses": azure_openai_responses_api(),
            "openai-completions": azure_streams(openai_completions_api()),
        },
    )
