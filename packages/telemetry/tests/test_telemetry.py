"""Port of ``test/telemetry.test.ts``, plus the runtime schema checks Python needs instead of compile-time ones.

Dropped: the ``expectTypeOf`` and ``@ts-expect-error`` assertions (TypeScript-only); each one has a runtime
counterpart below. ``createTypedSpanStarter(context, unreadable(schemas))`` not throwing does not apply: Python
starters index their schemas at creation to check names and attributes.
"""

from __future__ import annotations

import copy
import json
import warnings
from typing import Any

import pytest

from pi_telemetry import (
    NOOP_TELEMETRY_CONTEXT,
    InMemoryTelemetryContext,
    SchemaTelemetrySpan,
    SpanAttributes,
    SpanOptions,
    SpanStatus,
    TelemetrySchemaDefinition,
    TelemetrySchemaError,
    TelemetrySchemaWarning,
    TelemetrySpan,
    TypedSpanStarter,
    bind_typed_span_starter,
    check_attribute_value,
    create_typed_span_starter,
    define_telemetry_schema,
    is_telemetry_enabled,
    schema_span_names,
)
from pi_telemetry.testing import assert_spans_match_schemas, check_spans_against_schemas, unreadable_mapping

OPERATION_SCHEMA = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            "operation": {
                "description": "Operation",
                "parents": {"kind": "root_or_external"},
                "startAttributes": {
                    "kind": {"type": "string", "required": True, "values": ["read", "write"], "description": "Kind"},
                },
                "endAttributes": {},
                "events": {
                    "result": {
                        "description": "Result",
                        "attributes": {
                            "outcome": {
                                "type": "string",
                                "required": True,
                                "values": ["ok", "error"],
                                "description": "Outcome",
                            },
                        },
                    },
                },
                "status": {"default": "ok", "errorWhen": "The operation fails"},
            },
        },
    }
)

REQUEST_SCHEMA = define_telemetry_schema(
    {
        "version": 3,
        "spans": {
            "request": {
                "description": "Request",
                "parents": {"kind": "spans", "spans": ["operation"]},
                "startAttributes": {"provider": {"type": "string", "required": True, "description": "Provider"}},
                "endAttributes": {
                    "response": {"type": "string", "description": "Response kind"},
                    "count": {"type": "number", "description": "Items"},
                    "tags": {"type": "string[]", "elementValues": ["a", "b"], "description": "Tags"},
                },
                "status": {"default": "ok", "errorWhen": "The request fails"},
            },
        },
    }
)


# ─── telemetry schemas ──────────────────────────────────────────────────────


def test_preserves_serializable_definitions() -> None:
    definition: TelemetrySchemaDefinition = {
        "version": 1,
        "spans": dict(OPERATION_SCHEMA["spans"]),
    }
    schema = define_telemetry_schema(definition)
    assert schema is definition
    assert json.loads(json.dumps(schema)) == schema
    assert schema_span_names([OPERATION_SCHEMA, REQUEST_SCHEMA]) == ["operation", "request"]


async def test_schema_span_checks_events_and_end_attributes() -> None:
    telemetry = InMemoryTelemetryContext()
    start = create_typed_span_starter(telemetry, [OPERATION_SCHEMA])
    failures: list[str] = []

    def callback(span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> None:
        span.add_event("result", {"outcome": "ok"})
        for attempt in (
            lambda: span.add_event("result"),  # required event attributes cannot be omitted
            lambda: span.add_event("result", {"outcome": "other"}),  # closed-set event values are exact
            lambda: span.add_event("unknown", {}),  # undeclared events are rejected
            lambda: span.set_attributes({"unknown": True}),  # empty end schemas reject every attribute
        ):
            try:
                attempt()
            except TelemetrySchemaError as error:
                failures.append(str(error))

    await start("operation", {"kind": "read"}, callback)
    assert len(failures) == 4
    assert "requires attribute 'outcome'" in failures[0]
    assert "'other' is not one of ['ok', 'error']" in failures[1]
    assert "does not declare event 'unknown'" in failures[2]
    assert "does not declare attribute 'unknown'" in failures[3]
    [span] = telemetry.get_spans()
    assert [event.name for event in span.events] == ["result"]


async def test_combines_schema_vocabularies_and_binds_child_starters_to_their_parent_spans() -> None:
    telemetry = InMemoryTelemetryContext()
    start_span = create_typed_span_starter(telemetry, [OPERATION_SCHEMA, REQUEST_SCHEMA])

    def request(span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> int:
        span.set_attributes({"response": "cached"})
        return 42

    async def operation(_span: SchemaTelemetrySpan, start_child_span: TypedSpanStarter) -> int:
        return await start_child_span("request", {"provider": "example"}, request)

    result = await start_span("operation", {"kind": "read"}, operation)
    assert result == 42
    spans = telemetry.get_spans()
    operation_span = next(span for span in spans if span.name == "operation")
    request_span = next(span for span in spans if span.name == "request")
    assert operation_span.parent_id is None
    assert request_span.parent_id == operation_span.id
    assert request_span.attributes == {"provider": "example", "response": "cached"}
    assert_spans_match_schemas(spans, [OPERATION_SCHEMA, REQUEST_SCHEMA])

    sync_error = RuntimeError("sync")

    def raise_sync(_span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> None:
        raise sync_error

    with pytest.raises(RuntimeError) as raised:
        await start_span("operation", {"kind": "write"}, raise_sync)
    assert raised.value is sync_error

    async_error = ValueError("async")

    async def raise_async(_span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> None:
        raise async_error

    with pytest.raises(ValueError) as raised_async:
        await start_span("request", {"provider": "example"}, raise_async)
    assert raised_async.value is async_error


def test_rejects_what_typescript_rejects_at_compile_time() -> None:
    telemetry = InMemoryTelemetryContext()
    start_span = create_typed_span_starter(telemetry, [OPERATION_SCHEMA, REQUEST_SCHEMA])
    with pytest.raises(TelemetrySchemaError, match="does not declare attribute 'unknown'"):
        start_span("request", {"provider": "example", "unknown": True}, lambda _span, _child: None)
    with pytest.raises(TelemetrySchemaError, match="Unknown telemetry span 'unknown'"):
        start_span("unknown", {}, lambda _span, _child: None)
    # Attributes are selected from the schema that owns the span.
    with pytest.raises(TelemetrySchemaError, match="requires attribute 'provider'"):
        start_span("request", {"kind": "read"}, lambda _span, _child: None)
    with pytest.raises(TelemetrySchemaError, match="Duplicate telemetry span name 'operation'"):
        create_typed_span_starter(telemetry, [OPERATION_SCHEMA, OPERATION_SCHEMA])
    with pytest.raises(TelemetrySchemaError, match="expected string, got int"):
        start_span("request", {"provider": 3}, lambda _span, _child: None)
    with pytest.raises(TelemetrySchemaError):
        create_typed_span_starter(telemetry, [])
    # Nothing was recorded and no callback ran.
    assert telemetry.get_spans() == []


async def test_warn_mode_drops_invalid_attributes_and_keeps_running() -> None:
    telemetry = InMemoryTelemetryContext()
    start_span = create_typed_span_starter(telemetry, [OPERATION_SCHEMA, REQUEST_SCHEMA], on_invalid="warn")

    def callback(span: SchemaTelemetrySpan, _child: TypedSpanStarter) -> str:
        span.set_attributes({"count": "many", "response": "fresh", "tags": ["a", "c"]})
        span.add_event("unknown")
        return "done"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await start_span("request", {"provider": "example", "extra": 1}, callback)
        unknown = await start_span("missing", {}, lambda _span, _child: "ran")
    assert result == "done"
    assert unknown == "ran"
    assert all(issubclass(warning.category, TelemetrySchemaWarning) for warning in caught)
    assert len(caught) == 4
    [span] = telemetry.get_spans()
    assert span.attributes == {"provider": "example", "response": "fresh"}
    assert span.events == []

    quiet = create_typed_span_starter(InMemoryTelemetryContext(), [REQUEST_SCHEMA], on_invalid="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert await quiet("request", {}, lambda _span, _child: 1) == 1


def test_attribute_value_checks() -> None:
    assert check_attribute_value({"type": "number", "description": ""}, True) == "expected number, got bool"
    assert check_attribute_value({"type": "number", "description": ""}, 1.5) is None
    assert check_attribute_value({"type": "boolean", "description": ""}, 0) == "expected boolean, got int"
    assert check_attribute_value({"type": "number[]", "description": ""}, [1, 2.5]) is None
    assert check_attribute_value({"type": "number[]", "description": ""}, [1, "2"]) is not None
    assert check_attribute_value({"type": "boolean[]", "description": "", "elementValues": [True]}, [False]) is not None
    assert check_attribute_value({"type": "string", "description": ""}, ["a"]) == "expected string, got list"


async def test_bound_starters_reuse_the_index_and_schema_checks_find_parent_violations() -> None:
    telemetry = InMemoryTelemetryContext()
    starter = create_typed_span_starter(NOOP_TELEMETRY_CONTEXT, [OPERATION_SCHEMA, REQUEST_SCHEMA])
    bound = bind_typed_span_starter(telemetry, starter)
    assert bound.span_names == ["operation", "request"]
    await bound("request", {"provider": "root"}, lambda _span, _child: None)
    problems = check_spans_against_schemas(telemetry.get_spans(), [OPERATION_SCHEMA, REQUEST_SCHEMA])
    assert problems == ["Span 'request' (id 1) must be a child of ['operation'], not None"]
    assert check_spans_against_schemas(telemetry.get_spans(), [OPERATION_SCHEMA]) == [
        "Span 'request' (id 1) is not declared by any schema"
    ]


# ─── NOOP_TELEMETRY_CONTEXT ─────────────────────────────────────────────────


async def test_noop_admits_callbacks_synchronously_and_reuses_one_inert_span() -> None:
    admitted = False
    first_span: list[TelemetrySpan] = []

    async def callback(span: TelemetrySpan) -> int:
        first_span.append(span)
        child = await span.start_span({"name": "child"}, lambda child_span: child_span)
        assert child is span
        return 42

    def sync_callback(span: TelemetrySpan) -> TelemetrySpan:
        nonlocal admitted
        admitted = True
        return span

    pending = NOOP_TELEMETRY_CONTEXT.start_span({"name": "first"}, sync_callback)
    assert admitted is True
    assert (await pending) is NOOP_TELEMETRY_CONTEXT
    assert await NOOP_TELEMETRY_CONTEXT.start_span({"name": "first"}, callback) == 42
    with pytest.raises(AttributeError):
        first_span[0].frozen = True  # type: ignore[attr-defined]
    assert copy.deepcopy(NOOP_TELEMETRY_CONTEXT) is NOOP_TELEMETRY_CONTEXT
    assert not is_telemetry_enabled(NOOP_TELEMETRY_CONTEXT)
    assert not is_telemetry_enabled(None)
    assert is_telemetry_enabled(InMemoryTelemetryContext())


async def test_noop_preserves_synchronous_and_asynchronous_rejection_values() -> None:
    sync_error = RuntimeError("sync")

    def raise_sync(_span: TelemetrySpan) -> None:
        raise sync_error

    pending = NOOP_TELEMETRY_CONTEXT.start_span({"name": "sync"}, raise_sync)
    with pytest.raises(RuntimeError) as raised:
        await pending
    assert raised.value is sync_error

    async_error = ValueError("async")

    async def raise_async(_span: TelemetrySpan) -> None:
        raise async_error

    with pytest.raises(ValueError) as raised_async:
        await NOOP_TELEMETRY_CONTEXT.start_span({"name": "async"}, raise_async)
    assert raised_async.value is async_error


async def test_noop_does_not_inspect_or_retain_telemetry_payloads() -> None:
    options: SpanOptions = unreadable_mapping()

    def callback(span: TelemetrySpan) -> None:
        attributes: SpanAttributes = unreadable_mapping()
        status: SpanStatus = unreadable_mapping()
        span.add_event("event", attributes)
        span.set_attributes(attributes)
        span.set_status(status)

    await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)


async def test_settled_results_can_be_dropped_without_warnings() -> None:
    # A synchronous callback's result is already settled: dropping it leaks no coroutine.
    telemetry = InMemoryTelemetryContext()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        telemetry.start_span({"name": "dropped"}, lambda _span: 1)
        NOOP_TELEMETRY_CONTEXT.start_span({"name": "dropped"}, lambda _span: 1)
    assert telemetry.get_spans()[0].settled is True


async def test_in_memory_records_arrays_as_lists_and_rejects_non_primitives() -> None:
    telemetry = InMemoryTelemetryContext()
    value: Any = {"nested": 1}

    def callback(span: TelemetrySpan) -> None:
        span.set_attributes({"tuple": ("a", "b"), "mixed": value})
        span.set_attributes({"tuple": ("a", "b"), "numbers": [1, 2.5], "flag": False})
        mixed: Any = [1, "a"]
        span.set_attributes({"mixed_array": mixed})
        span.add_event(3, {})  # type: ignore[arg-type]

    await telemetry.start_span({"name": "values"}, callback)
    [span] = telemetry.get_spans()
    assert span.attributes == {"tuple": ["a", "b"], "numbers": [1, 2.5], "flag": False}
    assert span.events == []
    telemetry.clear()
    assert telemetry.get_spans() == []
