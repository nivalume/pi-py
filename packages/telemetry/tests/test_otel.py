"""The OpenTelemetry adapter against the conformance suite and the GenAI mapping, with the SDK's in-memory exporters."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.context import Context as OtelContext
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.trace import SpanKind, StatusCode

from pi_telemetry import RecordedTelemetryEvent, RecordedTelemetrySpan, SpanStatus, TelemetryContext, TelemetrySpan
from pi_telemetry.otel import OpenTelemetryContext
from pi_telemetry.testing import TelemetryAdapterFixture, create_telemetry_adapter_conformance


class _Recorder(SpanProcessor):
    """Keeps spans in start order and their end order, as the conformance normalization needs."""

    def __init__(self) -> None:
        self.started: list[Span] = []
        self.ended: list[ReadableSpan] = []

    def on_start(self, span: Span, parent_context: OtelContext | None = None) -> None:
        self.started.append(span)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.append(span)


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _id(span: ReadableSpan) -> int:
    context = span.context
    assert context is not None
    return context.span_id


def normalize(recorder: _Recorder) -> list[RecordedTelemetrySpan]:
    ids = {_id(span): index + 1 for index, span in enumerate(recorder.started)}
    ends = {_id(span): index + 1 for index, span in enumerate(recorder.ended)}
    normalized: list[RecordedTelemetrySpan] = []
    for span in recorder.started:
        span_id = _id(span)
        attributes = {key: _plain(value) for key, value in (span.attributes or {}).items()}
        error_type = attributes.pop("error.type", None)
        code = span.status.status_code
        status: SpanStatus
        if code == StatusCode.ERROR:
            status = (
                {"status": "error", "error": {"name": str(error_type), "message": span.status.description or ""}}
                if error_type is not None
                else {"status": "error"}
            )
        else:
            status = {"status": "ok"}
        normalized.append(
            RecordedTelemetrySpan(
                id=ids[span_id],
                parent_id=None if span.parent is None else ids.get(span.parent.span_id),
                name=span.name,
                attributes=attributes,
                events=[
                    RecordedTelemetryEvent(event.name, {k: _plain(v) for k, v in (event.attributes or {}).items()})
                    for event in span.events
                ],
                status=status,
                settled=span.end_time is not None,
                end_sequence=ends.get(span_id),
            )
        )
    return normalized


class _Fixture:
    def __init__(self) -> None:
        self.recorder = _Recorder()
        self.provider = TracerProvider()
        self.provider.add_span_processor(self.recorder)
        self._context = OpenTelemetryContext(tracer_provider=self.provider, record_metrics=False)

    @property
    def context(self) -> TelemetryContext:
        return self._context

    async def get_spans(self) -> Sequence[RecordedTelemetrySpan]:
        return normalize(self.recorder)

    async def aclose(self) -> None:
        self.provider.shutdown()


async def _factory() -> TelemetryAdapterFixture:
    return _Fixture()


CASES = create_telemetry_adapter_conformance(_factory)


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.group}: {case.name}")
async def test_open_telemetry_conformance(case: Any) -> None:
    await case.run()


def _setup(**options: Any) -> tuple[OpenTelemetryContext, _Recorder, InMemoryMetricReader, TracerProvider]:
    recorder = _Recorder()
    provider = TracerProvider()
    provider.add_span_processor(recorder)
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    return OpenTelemetryContext(tracer_provider=provider, meter_provider=meters, **options), recorder, reader, provider


def _metrics(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    data = reader.get_metrics_data()
    found: dict[str, list[Any]] = {}
    assert data is not None
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                found[metric.name] = list(metric.data.data_points)
    return found


async def test_ai_requests_get_gen_ai_attributes_and_metrics() -> None:
    context, recorder, reader, _provider = _setup()

    async def request(span: TelemetrySpan) -> None:
        span.add_event("pi.ai.retry", {"pi.ai.retry.attempt": 1, "pi.ai.retry.reason": "http_status"})
        span.set_attributes(
            {
                "pi.ai.usage.input_tokens": 120,
                "pi.ai.usage.output_tokens": 30,
                "pi.ai.usage.cache_read_tokens": 100,
                "pi.ai.usage.cache_write_tokens": 0,
                "pi.ai.cost": 0.25,
                "pi.ai.stop_reason": "toolUse",
                "pi.ai.time_to_first_token_ms": 50,
                "pi.ai.duration_ms": 400,
                "pi.ai.response.model": "model-2026",
            }
        )

    await context.start_span(
        {
            "name": "pi.ai.request",
            "attributes": {
                "pi.ai.provider": "anthropic",
                "pi.ai.api": "anthropic-messages",
                "pi.ai.model": "claude",
                "pi.ai.session_id": "session-1",
                "pi.ai.request.max_tokens": 1000,
            },
        },
        request,
    )
    [span] = recorder.ended
    assert span.kind == SpanKind.CLIENT
    attributes = dict(span.attributes or {})
    assert attributes["pi.ai.provider"] == "anthropic"
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.provider.name"] == "anthropic"
    assert attributes["gen_ai.system"] == "anthropic"
    assert attributes["gen_ai.request.model"] == "claude"
    assert attributes["gen_ai.response.model"] == "model-2026"
    assert attributes["gen_ai.usage.input_tokens"] == 120
    assert attributes["gen_ai.usage.output_tokens"] == 30
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 100
    assert attributes["gen_ai.response.finish_reasons"] == ("tool_call",)
    assert attributes["gen_ai.conversation.id"] == "session-1"
    assert attributes["gen_ai.request.max_tokens"] == 1000
    assert attributes["gen_ai.response.time_to_first_chunk"] == 0.05
    assert "error.type" not in attributes
    assert span.status.status_code == StatusCode.UNSET
    assert [event.name for event in span.events] == ["pi.ai.retry"]

    found = _metrics(reader)
    [duration] = found["gen_ai.client.operation.duration"]
    assert duration.sum == pytest.approx(0.4)
    assert dict(duration.attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.model": "claude",
        "gen_ai.response.model": "model-2026",
        "pi.ai.api": "anthropic-messages",
    }
    assert found["gen_ai.client.operation.time_to_first_chunk"][0].sum == pytest.approx(0.05)
    usage = {point.attributes["gen_ai.token.type"]: point.sum for point in found["gen_ai.client.token.usage"]}
    assert usage == {"input": 120, "output": 30}
    tokens = {point.attributes["pi.ai.token.type"]: point.value for point in found["pi.ai.tokens"]}
    assert tokens == {"input": 120, "output": 30, "cache_read": 100}
    assert found["pi.ai.requests"][0].value == 1
    assert found["pi.ai.cost"][0].value == pytest.approx(0.25)
    assert "pi.ai.errors" not in found
    # Labels are low-cardinality: no session, no request-specific values.
    for points in found.values():
        for point in points:
            assert "gen_ai.conversation.id" not in point.attributes
            assert "pi.ai.session_id" not in point.attributes


async def test_errors_carry_the_kind_but_no_free_form_details() -> None:
    context, recorder, reader, _provider = _setup()

    def failed(span: TelemetrySpan) -> None:
        span.set_status({"status": "error", "error": {"name": "rate_limited", "message": ""}})

    await context.start_span(
        {"name": "pi.ai.request", "attributes": {"pi.ai.provider": "openai", "pi.ai.model": "gpt"}}, failed
    )

    def raises(_span: TelemetrySpan) -> None:
        raise RuntimeError("secret prompt text in an error message")

    with pytest.raises(RuntimeError):
        await context.start_span({"name": "example.operation"}, raises)
    request, operation = recorder.ended
    assert request.status.status_code == StatusCode.ERROR
    assert request.status.description is None
    assert dict(request.attributes or {})["error.type"] == "rate_limited"
    assert operation.status.status_code == StatusCode.ERROR
    assert operation.status.description is None
    assert dict(operation.attributes or {})["error.type"] == "RuntimeError"
    assert operation.events == ()
    found = _metrics(reader)
    assert found["pi.ai.errors"][0].attributes["error.type"] == "rate_limited"

    recording, recorder2, _reader2, _provider2 = _setup(record_exception=True, record_metrics=False)
    with pytest.raises(RuntimeError):
        await recording.start_span({"name": "example.operation"}, raises)
    assert [event.name for event in recorder2.ended[0].events] == ["exception"]


async def test_parents_come_from_the_pi_context_and_callbacks_see_their_span_as_current() -> None:
    context, recorder, _reader, provider = _setup(record_metrics=False)
    tracer = provider.get_tracer("application")
    seen: list[Any] = []

    async def child(span: TelemetrySpan) -> None:
        seen.append(trace.get_current_span())

    async def parent(span: TelemetrySpan) -> None:
        seen.append(trace.get_current_span())
        await span.start_span({"name": "child"}, child)

    with tracer.start_as_current_span("application.request") as outer:
        await context.start_span({"name": "parent"}, parent)
        assert trace.get_current_span() is outer
    application, *_ = [span for span in recorder.started if span.name == "application.request"]
    pi_parent = next(span for span in recorder.started if span.name == "parent")
    pi_child = next(span for span in recorder.started if span.name == "child")
    assert pi_parent.parent is not None and pi_parent.parent.span_id == _id(application)
    assert pi_child.parent is not None and pi_child.parent.span_id == _id(pi_parent)
    assert seen == [pi_parent, pi_child]
    assert pi_parent.parent.trace_id == pi_parent.get_span_context().trace_id

    explicit_root, recorder3, _reader3, _provider3 = _setup(record_metrics=False, parent=OtelContext(), activate=False)
    with tracer.start_as_current_span("ignored"):
        await explicit_root.start_span({"name": "root"}, lambda span: seen.append(trace.get_current_span()))
    assert recorder3.started[0].parent is None
    assert seen[-1].name == "ignored"  # not activated


def test_importing_pi_telemetry_does_not_import_opentelemetry() -> None:
    code = (
        "import sys, pi_telemetry, pi_telemetry.testing\n"
        "import pi_ai, pi_agent, pi_durable\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('opentelemetry'))\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
