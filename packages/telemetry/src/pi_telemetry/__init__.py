"""pi_telemetry: vendor-neutral telemetry contracts and typed schema utilities for pi packages.

Python port of ``@earendil-works/pi-telemetry``:

- an explicit, callback-based :class:`TelemetryContext` / :class:`TelemetrySpan` contract (no ambient current
  span, no ``contextvars``);
- the shared :data:`NOOP_TELEMETRY_CONTEXT`;
- the reference :class:`InMemoryTelemetryContext`;
- serializable schemas (:func:`define_telemetry_schema`) and :func:`create_typed_span_starter`, which checks
  span names and attributes at runtime;
- an explicit content-capture opt-in (:func:`with_content_capture`);
- the adapter conformance suite in :mod:`pi_telemetry.testing`.

No exporter and no backend dependency: :mod:`pi_telemetry.otel` adapts OpenTelemetry (the ``otel`` extra) and is
only imported when used.
"""

from .content import (
    CONTENT_EVENT,
    CONTENT_EVENT_NAME,
    ContentCapture,
    ContentRedactor,
    content_capture_of,
    record_content,
    with_content_capture,
)
from .memory import InMemoryTelemetryContext, RecordedTelemetryEvent, RecordedTelemetrySpan
from .noop import NOOP_TELEMETRY_CONTEXT, is_telemetry_enabled
from .schema import (
    OnInvalid,
    SchemaTelemetrySpan,
    TelemetryAttributeDefinition,
    TelemetryAttributeType,
    TelemetryEventAttributeDefinition,
    TelemetryEventDefinition,
    TelemetryParentAny,
    TelemetryParentDefinition,
    TelemetryParentRootOrExternal,
    TelemetryParentSpans,
    TelemetrySchemaDefinition,
    TelemetrySchemaError,
    TelemetrySchemaWarning,
    TelemetrySpanDefinition,
    TelemetrySpanStatusRule,
    TelemetryStartAttributeDefinition,
    TypedSpanStarter,
    bind_typed_span_starter,
    check_attribute_value,
    check_end_attributes,
    check_event,
    check_start_attributes,
    create_typed_span_starter,
    define_telemetry_schema,
    schema_span_names,
)
from .types import (
    AttributeScalar,
    AttributeValue,
    ErrorSpanStatus,
    OkSpanStatus,
    SpanAttributes,
    SpanErrorInfo,
    SpanOptions,
    SpanStatus,
    TelemetryContext,
    TelemetrySpan,
)

__all__ = [
    # core contract
    "TelemetryContext",
    "TelemetrySpan",
    "SpanOptions",
    "SpanAttributes",
    "AttributeValue",
    "AttributeScalar",
    "SpanStatus",
    "OkSpanStatus",
    "ErrorSpanStatus",
    "SpanErrorInfo",
    "NOOP_TELEMETRY_CONTEXT",
    "is_telemetry_enabled",
    "InMemoryTelemetryContext",
    "RecordedTelemetrySpan",
    "RecordedTelemetryEvent",
    # schemas
    "define_telemetry_schema",
    "create_typed_span_starter",
    "bind_typed_span_starter",
    "TypedSpanStarter",
    "SchemaTelemetrySpan",
    "TelemetrySchemaDefinition",
    "TelemetrySpanDefinition",
    "TelemetryAttributeType",
    "TelemetryAttributeDefinition",
    "TelemetryStartAttributeDefinition",
    "TelemetryEventAttributeDefinition",
    "TelemetryEventDefinition",
    "TelemetryParentDefinition",
    "TelemetryParentAny",
    "TelemetryParentRootOrExternal",
    "TelemetryParentSpans",
    "TelemetrySpanStatusRule",
    "OnInvalid",
    "TelemetrySchemaError",
    "TelemetrySchemaWarning",
    "schema_span_names",
    "check_attribute_value",
    "check_start_attributes",
    "check_end_attributes",
    "check_event",
    # content capture
    "with_content_capture",
    "content_capture_of",
    "record_content",
    "ContentCapture",
    "ContentRedactor",
    "CONTENT_EVENT",
    "CONTENT_EVENT_NAME",
]
