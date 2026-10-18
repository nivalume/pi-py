"""Materialize Cloudflare account/gateway placeholders in model base URLs before dispatch."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, TypeVar

from ..event_stream import AssistantMessageEventStream
from ..types import ClassifierContext, ClassifierModel, ClassifierResult, Model, TranscriptContext

__all__ = ["resolve_cloudflare_model", "cloudflare_streams", "cloudflare_classifier"]

CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"

_ModelT = TypeVar("_ModelT")


def resolve_cloudflare_model(model: _ModelT, env: Mapping[str, str] | None) -> _ModelT:
    """Replace ``{CLOUDFLARE_ACCOUNT_ID}``/``{CLOUDFLARE_GATEWAY_ID}`` in ``model.base_url`` from ``env``.

    Unresolved placeholders are kept; the model is returned unchanged when nothing changes.
    """
    if not env:
        return model
    base_url: str = model.base_url  # type: ignore[attr-defined]
    resolved = base_url
    for name in (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_GATEWAY_ID):
        value = env.get(name)
        placeholder = "{" + name + "}"
        resolved = resolved.replace(placeholder, value if value is not None else placeholder)
    if resolved == base_url:
        return model
    return dataclasses.replace(model, base_url=resolved)  # type: ignore[type-var]


def _env(options: Mapping[str, Any] | None) -> Mapping[str, str] | None:
    return options.get("env") if options else None


class _CloudflareStreams:
    __slots__ = ("_streams",)

    def __init__(self, streams: Any) -> None:
        self._streams = streams

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        return self._streams.stream(resolve_cloudflare_model(model, _env(options)), context, options)

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        return self._streams.stream_simple(resolve_cloudflare_model(model, _env(options)), context, options)

    def __repr__(self) -> str:
        return f"cloudflare_streams({self._streams!r})"


class _CloudflareClassifier:
    __slots__ = ("_classifier",)

    def __init__(self, classifier: Any) -> None:
        self._classifier = classifier

    async def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: Mapping[str, Any] | None = None
    ) -> ClassifierResult:
        return await self._classifier.classify(resolve_cloudflare_model(model, _env(options)), context, options)

    def __repr__(self) -> str:
        return f"cloudflare_classifier({self._classifier!r})"


def cloudflare_streams(streams: Any) -> Any:
    """Wrap an API implementation so Cloudflare endpoint placeholders materialize from the resolved provider env."""
    return _CloudflareStreams(streams)


def cloudflare_classifier(classifier: Any) -> Any:
    """Classifier counterpart of :func:`cloudflare_streams`."""
    return _CloudflareClassifier(classifier)
