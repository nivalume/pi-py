"""Radius gateway provider with a persisted, dynamically refreshed catalog."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from ..api.registry import pi_messages_api
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_radius_oauth
from ..auth.types import ProviderAuth
from ..event_stream import AssistantMessageEventStream
from ..models import ModelsPublication, Provider, RefreshModelsContext
from ..models_store import ModelsStoreEntry
from ..types import Model, TranscriptContext, now_ms
from .catalog import get_builtin_models
from .radius_config import (
    DEFAULT_RADIUS_GATEWAY,
    get_radius_models,
    get_radius_models_from_config,
    load_radius_gateway_config,
    normalize_radius_gateway_url,
)

__all__ = ["RadiusProvider", "radius_provider"]


class RadiusProvider(Provider):
    """Radius gateway provider.

    The gateway catalog for the account replaces the shipped baseline once known: Radius
    organization owners can disable models, so disabled models must not reappear from the
    baseline. The baseline (default gateway only) covers the time before any catalog exists.
    """

    def __init__(
        self,
        *,
        id: str = "radius",
        name: str = "Radius",
        gateway: str = DEFAULT_RADIUS_GATEWAY,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.gateway = normalize_radius_gateway_url(gateway)
        self._http_client = http_client
        self._use_public_catalog = self.gateway == normalize_radius_gateway_url(DEFAULT_RADIUS_GATEWAY)
        self._baseline: list[Model] | None = None
        self._dynamic: list[Model] | None = None
        self._streams = pi_messages_api()
        gateway_url = self.gateway
        self.auth = ProviderAuth(
            api_key=env_api_key_auth("Radius API key", ["RADIUS_API_KEY"]),
            oauth=lazy_oauth(name, lambda: load_radius_oauth(name=name, gateway=gateway_url)),
        )
        self.refresh_models = self._refresh_models

    def _baseline_models(self) -> list[Model]:
        baseline = self._baseline
        if baseline is None:
            baseline = (
                [dataclasses.replace(model, provider=self.id) for model in get_builtin_models("radius")]
                if self._use_public_catalog
                else []
            )
            self._baseline = baseline
        return baseline

    def get_models(self) -> Sequence[Model]:
        dynamic = self._dynamic
        return list(dynamic if dynamic is not None else self._baseline_models())

    def _set_dynamic(self, models: list[Model]) -> None:
        self._dynamic = models

    async def _refresh_models(self, context: RefreshModelsContext) -> None:
        stored = context.stored
        if stored is not None:
            restored: list[Model] = [model for model in stored.models if model.provider == self.id]
            if not await context.publish(ModelsPublication(update=lambda: self._set_dynamic(restored))):
                return

        credential = context.credential
        is_oauth = credential is not None and credential.get("type") == "oauth"
        # Import catalogs cached by the pre-ModelsStore Radius implementation.
        if stored is None and is_oauth:
            legacy = get_radius_models(self.id, credential)
            if legacy:
                if not await context.publish(
                    ModelsPublication(
                        persist=ModelsStoreEntry(models=list(legacy), checked_at=now_ms()),
                        update=lambda: self._set_dynamic(legacy),
                    )
                ):
                    return

        if not context.allow_network or context.signal.aborted:
            return
        if is_oauth:
            api_key = credential.get("access")  # type: ignore[union-attr]
        else:
            api_key = credential.get("key") if credential is not None else None
        config = await load_radius_gateway_config(self.gateway, api_key, context.signal, http_client=self._http_client)
        if context.signal.aborted:
            return
        refreshed = get_radius_models_from_config(self.id, config)
        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=list(refreshed), checked_at=now_ms()),
                update=lambda: self._set_dynamic(refreshed),
            )
        )

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        return self._streams.stream(model, context, options)

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        return self._streams.stream_simple(model, context, options)


def radius_provider(
    *,
    id: str | None = None,
    name: str | None = None,
    gateway: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider:
    """Radius gateway provider (``RADIUS_API_KEY`` or Radius OAuth).

    ``gateway`` defaults to ``https://radius.pi.dev``; only the default gateway ships a
    static baseline catalog. ``http_client`` is used for ``/v1/config`` catalog requests.
    """
    return RadiusProvider(
        id=id if id is not None else "radius",
        name=name if name is not None else "Radius",
        gateway=gateway if gateway is not None else DEFAULT_RADIUS_GATEWAY,
        http_client=http_client,
    )
