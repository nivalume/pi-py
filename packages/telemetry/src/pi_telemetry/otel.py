"""OpenTelemetry adapter for the pi telemetry contract (needs the ``otel`` extra: ``opentelemetry-api``).

:class:`OpenTelemetryContext` turns pi spans into OpenTelemetry spans:

- a pi span becomes an OTel span whose parent is the pi parent span; a root pi span's parent is ``parent``
  (an OTel ``Context``) or, by default, the OTel context current when the span starts, so pi spans join an
  application trace;
- attributes keep their ``pi.*`` names; events map to span events; a pi ``error`` status maps to
  ``StatusCode.ERROR`` with the error name as ``error.type`` and the message as the description, an explicit
  ``ok`` to ``StatusCode.OK``, and an automatic ``ok`` leaves the status unset;
- a raise sets ``StatusCode.ERROR`` with ``error.type`` (the exception class) but no description: exception
  messages are free-form error details, recorded only with ``record_exception=True``;
- ``pi.ai.request`` spans are ``CLIENT`` spans that additionally carry the OpenTelemetry GenAI semantic-convention
  attributes (``gen_ai.operation.name``, ``gen_ai.provider.name``, ``gen_ai.request.model``,
  ``gen_ai.response.model``, ``gen_ai.usage.*``, ``gen_ai.response.finish_reasons``, ...), and feed GenAI client
  metrics plus pi's request, error, token and cost counters, all labeled by provider and model only.

While a callback runs, its OTel span is the current OTel span, so automatic instrumentation (for example of
``httpx``) nests under it. pi code itself still passes parents explicitly.

Importing :mod:`pi_telemetry` never imports OpenTelemetry; only this module does.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from opentelemetry.context import Context as OtelContext
from opentelemetry.trace import SpanKind, Status, StatusCode

from ._util import Settled, copy_attributes, copy_status, read_span_options
from .noop import NOOP_TELEMETRY_CONTEXT
from .types import AttributeValue, SpanAttributes, SpanOptions, SpanStatus, TelemetrySpan

__all__ = ["OpenTelemetryContext", "GEN_AI_FINISH_REASONS", "AI_REQUEST_SPAN"]

T = TypeVar("T")

_INSTRUMENTATION_NAME: Final = "pi_telemetry"

#: The span that gets GenAI semantic-convention attributes and metrics.
AI_REQUEST_SPAN: Final = "pi.ai.request"

#: pi stop reasons as ``gen_ai.response.finish_reasons`` values.
GEN_AI_FINISH_REASONS: Final[Mapping[str, str]] = {
    "stop": "stop",
    "length": "length",
    "toolUse": "tool_call",
    "error": "error",
    "aborted": "cancelled",
    "deferred": "deferred",
}

_DURATION_BUCKETS: Final = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92)
_FIRST_CHUNK_BUCKETS: Final = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0)
_TOKEN_BUCKETS: Final = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)


class _Metrics:
    """GenAI client metrics and pi's counters, labeled only by low-cardinality values."""

    __slots__ = ("cost", "duration", "errors", "first_chunk", "requests", "token_usage", "tokens")

    def __init__(self, meter: metrics.Meter) -> None:
        self.duration = meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="Duration of pi model requests",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS,
        )
        self.first_chunk = meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk",
            unit="s",
            description="Time from the start of a pi model request to its first streamed content",
            explicit_bucket_boundaries_advisory=_FIRST_CHUNK_BUCKETS,
        )
        self.token_usage = meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Input and output tokens per pi model request",
            explicit_bucket_boundaries_advisory=_TOKEN_BUCKETS,
        )
        self.requests = meter.create_counter("pi.ai.requests", unit="{request}", description="pi model requests")
        self.errors = meter.create_counter(
            "pi.ai.errors", unit="{request}", description="pi model requests that ended with an error or abort"
        )
        self.tokens = meter.create_counter(
            "pi.ai.tokens", unit="{token}", description="Tokens by type: input, output, cache_read, cache_write"
        )
        self.cost = meter.create_counter("pi.ai.cost", unit="USD", description="Estimated model request cost")


def _number(value: Any) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _gen_ai_attributes(attributes: Mapping[str, AttributeValue], error_type: str | None) -> dict[str, Any]:
    """The GenAI semantic-convention attributes of a ``pi.ai.request`` span's final pi attributes."""
    mapped: dict[str, Any] = {"gen_ai.operation.name": "chat"}
    provider = attributes.get("pi.ai.provider")
    if isinstance(provider, str):
        mapped["gen_ai.provider.name"] = provider
        # The attribute older GenAI conventions (and many backends) still read.
        mapped["gen_ai.system"] = provider
    model = attributes.get("pi.ai.model")
    if isinstance(model, str):
        mapped["gen_ai.request.model"] = model
    response_model = attributes.get("pi.ai.response.model")
    if isinstance(response_model, str):
        mapped["gen_ai.response.model"] = response_model
    session = attributes.get("pi.ai.session_id")
    if isinstance(session, str):
        mapped["gen_ai.conversation.id"] = session
    for pi_name, gen_ai_name in (
        ("pi.ai.request.max_tokens", "gen_ai.request.max_tokens"),
        ("pi.ai.request.temperature", "gen_ai.request.temperature"),
        ("pi.ai.usage.input_tokens", "gen_ai.usage.input_tokens"),
        ("pi.ai.usage.output_tokens", "gen_ai.usage.output_tokens"),
        ("pi.ai.usage.cache_read_tokens", "gen_ai.usage.cache_read.input_tokens"),
        ("pi.ai.usage.cache_write_tokens", "gen_ai.usage.cache_creation.input_tokens"),
        ("pi.ai.usage.reasoning_tokens", "gen_ai.usage.reasoning.output_tokens"),
    ):
        value = _number(attributes.get(pi_name))
        if value is not None:
            mapped[gen_ai_name] = value
    first_token_ms = _number(attributes.get("pi.ai.time_to_first_token_ms"))
    if first_token_ms is not None:
        mapped["gen_ai.response.time_to_first_chunk"] = first_token_ms / 1000
    stop_reason = attributes.get("pi.ai.stop_reason")
    if isinstance(stop_reason, str):
        mapped["gen_ai.response.finish_reasons"] = [GEN_AI_FINISH_REASONS.get(stop_reason, stop_reason)]
    if error_type is not None:
        mapped["error.type"] = error_type
    return mapped


def _metric_labels(gen_ai: Mapping[str, Any], attributes: Mapping[str, AttributeValue]) -> dict[str, Any]:
    labels = {
        key: gen_ai[key]
        for key in ("gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model", "gen_ai.response.model")
        if key in gen_ai
    }
    api = attributes.get("pi.ai.api")
    if isinstance(api, str):
        labels["pi.ai.api"] = api
    return labels


class _OtelSpan:
    __slots__ = ("_ai", "_explicit", "_native", "_owner", "_parent", "_settled", "_started", "_status")

    def __init__(
        self, owner: OpenTelemetryContext, native: trace.Span, parent: OtelContext, ai: dict[str, Any] | None
    ) -> None:
        self._owner = owner
        self._native = native
        #: The OTel context holding this span: the parent of its children.
        self._parent = parent
        self._settled = False
        self._status: SpanStatus | None = None
        self._explicit = False
        #: Merged pi attributes of a ``pi.ai.request`` span, for the GenAI mapping at settlement.
        self._ai = ai
        self._started = time.perf_counter()

    @property
    def native_span(self) -> trace.Span:
        """The OpenTelemetry span (for linking or application-specific attributes)."""
        return self._native

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        if self._settled:
            return NOOP_TELEMETRY_CONTEXT.start_span(options, callback)
        return self._owner._start(self._parent, options, callback)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        if self._settled:
            return
        try:
            if not isinstance(name, str):
                raise TypeError("Telemetry event names must be strings")
            self._native.add_event(name, copy_attributes(attributes))
        except Exception:
            pass

    def set_attributes(self, attributes: SpanAttributes) -> None:
        if self._settled:
            return
        try:
            copied = copy_attributes(attributes)
            self._native.set_attributes(copied)
            if self._ai is not None:
                self._ai.update(copied)
        except Exception:
            pass

    def set_status(self, status: SpanStatus) -> None:
        if self._settled:
            return
        try:
            self._status = copy_status(status)
            self._explicit = True
        except Exception:
            pass

    def _settle(self, failed: bool, error: BaseException | None) -> None:
        if self._settled:
            return
        self._settled = True
        native = self._native
        owner = self._owner
        try:
            error_type: str | None = None
            status = self._status if self._explicit else None
            if status is not None:
                if status["status"] == "ok":
                    native.set_status(Status(StatusCode.OK))
                else:
                    info = status.get("error")
                    error_type = info["name"] if info is not None else "error"
                    native.set_attribute("error.type", error_type)
                    native.set_status(Status(StatusCode.ERROR, (info["message"] or None) if info is not None else None))
            elif failed:
                error_type = type(error).__qualname__ if error is not None else "error"
                native.set_attribute("error.type", error_type)
                native.set_status(Status(StatusCode.ERROR))
            if failed and error is not None and owner._record_exception:
                native.record_exception(error)
            if self._ai is not None:
                owner._finish_ai_request(native, self._ai, error_type, time.perf_counter() - self._started)
        except Exception:
            pass
        try:
            native.end()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"OpenTelemetrySpan({self._native!r})"


class OpenTelemetryContext:
    """A root pi :class:`~pi_telemetry.TelemetryContext` backed by OpenTelemetry.

    Args:
        tracer_provider: Where spans go; default the global tracer provider.
        meter_provider: Where metrics go; default the global meter provider.
        parent: OTel parent context of root pi spans; default the OTel context current at each span start.
        record_metrics: Record GenAI and pi metrics for ``pi.ai.request`` spans.
        record_exception: Also record raised exceptions (message and stack trace) as OTel exception events. Off by
            default: exception messages are free-form error details.
        activate: Make each callback's span the current OTel span while the callback runs, for automatic
            instrumentation.
    """

    __slots__ = ("_activate", "_metrics", "_parent", "_record_exception", "_tracer")

    def __init__(
        self,
        *,
        tracer_provider: trace.TracerProvider | None = None,
        meter_provider: metrics.MeterProvider | None = None,
        parent: OtelContext | None = None,
        record_metrics: bool = True,
        record_exception: bool = False,
        activate: bool = True,
    ) -> None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            package_version = version("pi-ai-python")
        except PackageNotFoundError:  # pragma: no cover - source checkouts
            package_version = ""
        self._tracer = trace.get_tracer(_INSTRUMENTATION_NAME, package_version, tracer_provider)
        self._metrics = (
            _Metrics(metrics.get_meter(_INSTRUMENTATION_NAME, package_version, meter_provider))
            if record_metrics
            else None
        )
        self._parent = parent
        self._record_exception = record_exception
        self._activate = activate

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        parent = self._parent if self._parent is not None else otel_context.get_current()
        return self._start(parent, options, callback)

    def _start(
        self, parent: OtelContext, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]
    ) -> Awaitable[T]:
        try:
            name, attributes = read_span_options(options)
            ai = dict(attributes) if name == AI_REQUEST_SPAN else None
            native = self._tracer.start_span(
                name,
                context=parent,
                kind=SpanKind.CLIENT if ai is not None else SpanKind.INTERNAL,
                attributes=attributes,
                record_exception=False,
                set_status_on_exception=False,
            )
            current = trace.set_span_in_context(native, parent)
            span = _OtelSpan(self, native, current, ai)
        except Exception:
            return NOOP_TELEMETRY_CONTEXT.start_span(options, callback)
        token = self._attach(current)
        try:
            result = callback(span)
        except BaseException as error:
            self._detach(token)
            span._settle(True, error)
            return Settled(None, error)
        self._detach(token)
        if inspect.isawaitable(result):
            return self._settle_after(result, span, current)
        span._settle(False, None)
        return Settled(result)

    async def _settle_after(self, result: Awaitable[T], span: _OtelSpan, current: OtelContext) -> T:
        token = self._attach(current)
        try:
            value = await result
        except BaseException as error:
            span._settle(True, error)
            raise
        finally:
            self._detach(token)
        span._settle(False, None)
        return value

    def _attach(self, current: OtelContext) -> object | None:
        if not self._activate:
            return None
        try:
            return otel_context.attach(current)
        except Exception:
            return None

    def _detach(self, token: object | None) -> None:
        if token is None:
            return
        try:
            otel_context.detach(token)  # type: ignore[arg-type]
        except Exception:
            pass

    def _finish_ai_request(
        self, native: trace.Span, attributes: Mapping[str, AttributeValue], error_type: str | None, elapsed: float
    ) -> None:
        gen_ai = _gen_ai_attributes(attributes, error_type)
        native.set_attributes(gen_ai)
        recorder = self._metrics
        if recorder is None:
            return
        labels = _metric_labels(gen_ai, attributes)
        duration_ms = _number(attributes.get("pi.ai.duration_ms"))
        outcome = dict(labels)
        if error_type is not None:
            outcome["error.type"] = error_type
        recorder.duration.record(duration_ms / 1000 if duration_ms is not None else elapsed, outcome)
        first_token_ms = _number(attributes.get("pi.ai.time_to_first_token_ms"))
        if first_token_ms is not None:
            recorder.first_chunk.record(first_token_ms / 1000, labels)
        recorder.requests.add(1, outcome)
        if error_type is not None:
            recorder.errors.add(1, outcome)
        for pi_name, token_type, semconv in (
            ("pi.ai.usage.input_tokens", "input", True),
            ("pi.ai.usage.output_tokens", "output", True),
            ("pi.ai.usage.cache_read_tokens", "cache_read", False),
            ("pi.ai.usage.cache_write_tokens", "cache_write", False),
        ):
            value = _number(attributes.get(pi_name))
            if value is None:
                continue
            if semconv:
                recorder.token_usage.record(value, {**labels, "gen_ai.token.type": token_type})
            if value > 0:
                recorder.tokens.add(value, {**labels, "pi.ai.token.type": token_type})
        cost = _number(attributes.get("pi.ai.cost"))
        if cost is not None and cost > 0:
            recorder.cost.add(cost, labels)

    def __repr__(self) -> str:
        return "OpenTelemetryContext()"
