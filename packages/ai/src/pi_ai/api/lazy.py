"""Deferred stream setup and lazily imported API implementations."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from types import ModuleType
from typing import Any

from ..abort import AbortError
from ..event_stream import AssistantMessageEventStream, EventStream
from ..types import AssistantMessage, ErrorEvent, Model, TranscriptContext, Usage, now_ms
from ..utils.diagnostics import format_thrown_value

__all__ = ["lazy_stream", "error_stream", "setup_error_message", "LazyApi", "lazy_api"]


def setup_error_message(model: Any, error: Any, timestamp: int | None = None) -> AssistantMessage:
    aborted = isinstance(error, AbortError)
    return AssistantMessage(
        [],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="aborted" if aborted else "error",
        error_message=format_thrown_value(error),
        timestamp=timestamp if timestamp is not None else now_ms(),
    )


def error_stream(model: Any, error: Any, timestamp: int | None = None) -> AssistantMessageEventStream:
    """A stream that terminates immediately with an error event."""
    stream = AssistantMessageEventStream()
    message = setup_error_message(model, error, timestamp)
    stream.push(ErrorEvent(message.stop_reason, message))  # type: ignore[arg-type]
    stream.end(message)
    return stream


def lazy_stream(
    model: Any,
    setup: Callable[[], Awaitable[AsyncIterable[Any]]],
) -> AssistantMessageEventStream:
    """Return a stream synchronously while async setup (auth, loading) runs behind it.

    Setup failures terminate the stream with an error event (``aborted`` when the
    failure is an :class:`AbortError`).
    """
    started_at = now_ms()
    outer = AssistantMessageEventStream()

    async def run() -> None:
        try:
            inner = await setup()
        except Exception as error:
            message = setup_error_message(model, error, started_at)
            outer.push(ErrorEvent(message.stop_reason, message))  # type: ignore[arg-type]
            outer.end(message)
            return
        if isinstance(inner, EventStream):
            inner.forward_to(outer)
            return
        try:
            async for event in inner:
                outer.push(event)
        except Exception as error:
            message = setup_error_message(model, error, started_at)
            outer.push(ErrorEvent(message.stop_reason, message))  # type: ignore[arg-type]
        result: Callable[[], Awaitable[Any]] | None = getattr(inner, "result", None)
        outer.end(await result() if result is not None else None)

    outer.start(run())
    return outer


class LazyApi:
    """``ProviderStreams`` that imports its implementation module on first use.

    The module must expose ``stream`` and ``stream_simple`` (and optionally
    ``fetch_deferred``/``cancel_deferred``). Import failures, for example a missing
    optional dependency, surface as stream errors.
    """

    def __init__(self, module: str, *, fetch_deferred: bool = False, cancel_deferred: bool = False) -> None:
        self.module = module
        self._loaded: ModuleType | None = None
        self.fetch_deferred: Callable[..., AssistantMessageEventStream] | None = (
            self._fetch_deferred if fetch_deferred else None
        )
        self.cancel_deferred: Callable[..., Awaitable[None]] | None = self._cancel_deferred if cancel_deferred else None

    def load(self) -> ModuleType:
        module = self._loaded
        if module is None:
            module = importlib.import_module(self.module)
            self._loaded = module
        return module

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        try:
            module = self.load()
        except Exception as error:
            return error_stream(model, error)
        return module.stream(model, context, options)

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        try:
            module = self.load()
        except Exception as error:
            return error_stream(model, error)
        return module.stream_simple(model, context, options)

    def _fetch_deferred(self, model: Model, handle: Any, options: Mapping[str, Any] | None = None) -> Any:
        try:
            module = self.load()
            implementation = getattr(module, "fetch_deferred", None)
            if implementation is None:
                raise RuntimeError("API does not support deferred responses")
        except Exception as error:
            return error_stream(model, error)
        return implementation(model, handle, options)

    async def _cancel_deferred(self, model: Model, handle: Any, options: Mapping[str, Any] | None = None) -> None:
        implementation = getattr(self.load(), "cancel_deferred", None)
        if implementation is None:
            raise RuntimeError("API cannot cancel deferred responses")
        await implementation(model, handle, options)

    def __repr__(self) -> str:
        return f"LazyApi({self.module!r})"


def lazy_api(module: str, *, fetch_deferred: bool = False, cancel_deferred: bool = False) -> LazyApi:
    return LazyApi(module, fetch_deferred=fetch_deferred, cancel_deferred=cancel_deferred)


class LazyOperation:
    """Lazily imported image or classifier implementation (``generate_images``/``classify``)."""

    def __init__(self, module: str, attribute: str) -> None:
        self.module = module
        self.attribute = attribute

    def _get(self) -> Callable[..., Awaitable[Any]]:
        return getattr(importlib.import_module(self.module), self.attribute)

    async def generate_images(self, model: Any, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        return await self._get()(model, context, options)

    async def classify(self, model: Any, context: Any, options: Mapping[str, Any] | None = None) -> Any:
        return await self._get()(model, context, options)
