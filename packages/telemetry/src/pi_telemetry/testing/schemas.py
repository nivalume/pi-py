"""Check recorded spans against schemas (Python addition: what TypeScript's types check at compile time)."""

from __future__ import annotations

from collections.abc import Sequence

from ..memory import RecordedTelemetrySpan
from ..schema import TelemetrySchemaDefinition, TelemetrySpanDefinition, check_attribute_value, check_event

__all__ = ["check_spans_against_schemas", "assert_spans_match_schemas"]


def _check_span(
    span: RecordedTelemetrySpan,
    definition: TelemetrySpanDefinition,
    names: dict[int, str],
    index: dict[str, TelemetrySpanDefinition],
) -> list[str]:
    problems: list[str] = []
    where = f"Span {span.name!r} (id {span.id})"
    start = definition["startAttributes"]
    end = definition["endAttributes"]
    for name, value in span.attributes.items():
        attribute = start.get(name) or end.get(name)
        if attribute is None:
            problems.append(f"{where} has undeclared attribute {name!r}")
            continue
        problem = check_attribute_value(attribute, value)
        if problem is not None:
            problems.append(f"{where} attribute {name!r}: {problem}")
    for name, attribute in start.items():
        if attribute.get("required") is True and name not in span.attributes:
            problems.append(f"{where} lacks required start attribute {name!r}")
    for event in span.events:
        problems.extend(check_event(span.name, definition, event.name, event.attributes))
    parents = definition["parents"]
    parent_name = None if span.parent_id is None else names.get(span.parent_id)
    if parents["kind"] == "spans":
        if parent_name not in parents["spans"]:
            problems.append(f"{where} must be a child of {list(parents['spans'])!r}, not {parent_name!r}")
    elif parents["kind"] == "root_or_external":
        if parent_name is not None and parent_name in index:
            problems.append(f"{where} must be a root or external child, not a child of schema span {parent_name!r}")
    return problems


def check_spans_against_schemas(
    spans: Sequence[RecordedTelemetrySpan],
    schemas: Sequence[TelemetrySchemaDefinition],
    *,
    allow_unknown: bool = False,
) -> list[str]:
    """Every way ``spans`` break ``schemas``: unknown names (unless ``allow_unknown``), undeclared or mistyped
    attributes, missing required start attributes, undeclared or invalid events, and parent rules."""
    index: dict[str, TelemetrySpanDefinition] = {}
    for schema in schemas:
        index.update(schema["spans"])
    names = {span.id: span.name for span in spans}
    problems: list[str] = []
    for span in spans:
        definition = index.get(span.name)
        if definition is None:
            if not allow_unknown:
                problems.append(f"Span {span.name!r} (id {span.id}) is not declared by any schema")
            continue
        problems.extend(_check_span(span, definition, names, index))
    return problems


def assert_spans_match_schemas(
    spans: Sequence[RecordedTelemetrySpan],
    schemas: Sequence[TelemetrySchemaDefinition],
    *,
    allow_unknown: bool = False,
) -> None:
    """Raise ``AssertionError`` listing every problem :func:`check_spans_against_schemas` finds."""
    problems = check_spans_against_schemas(spans, schemas, allow_unknown=allow_unknown)
    if problems:
        raise AssertionError("Spans do not match their schemas:\n- " + "\n- ".join(problems))
