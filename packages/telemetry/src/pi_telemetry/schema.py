"""Typed, serializable telemetry schemas (port of the schema half of ``src/index.ts``).

A schema is plain JSON-serializable data with upstream's keys (``startAttributes``, ``endAttributes``,
``elementValues``, ``errorWhen``), so one schema document describes the same vocabulary in TypeScript and Python.
The ``TypedDict`` s below let a type checker verify a schema literal's structure.

TypeScript checks span names and attributes against a schema at compile time. Python cannot, so
:func:`create_typed_span_starter` checks them at runtime: unknown span names, missing required start attributes,
undeclared attributes or events, wrong value types, and values outside a closed set. ``on_invalid`` picks what a
violation does: ``"raise"`` (the default, the analog of a compile error) raises :class:`TelemetrySchemaError`
before anything is recorded; ``"warn"`` drops the offending attributes or event and emits a
:class:`TelemetrySchemaWarning`; ``"ignore"`` drops them silently. Instrumentation that must never change the
behavior of the code it observes (pi's own packages) uses ``"warn"``.
"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, TypedDict, TypeVar

from .noop import NOOP_TELEMETRY_CONTEXT
from .types import AttributeValue, SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan

if TYPE_CHECKING:
    from .content import ContentCapture

__all__ = [
    "TelemetryAttributeType",
    "TelemetryAttributeDefinition",
    "TelemetryStartAttributeDefinition",
    "TelemetryEventAttributeDefinition",
    "TelemetryEventDefinition",
    "TelemetryParentAny",
    "TelemetryParentRootOrExternal",
    "TelemetryParentSpans",
    "TelemetryParentDefinition",
    "TelemetrySpanStatusRule",
    "TelemetrySpanDefinition",
    "TelemetrySchemaDefinition",
    "OnInvalid",
    "TelemetrySchemaError",
    "TelemetrySchemaWarning",
    "define_telemetry_schema",
    "create_typed_span_starter",
    "bind_typed_span_starter",
    "TypedSpanStarter",
    "SchemaTelemetrySpan",
    "schema_span_names",
    "check_attribute_value",
    "check_start_attributes",
    "check_end_attributes",
    "check_event",
]

T = TypeVar("T")

TelemetryAttributeType: TypeAlias = Literal["string", "number", "boolean", "string[]", "number[]", "boolean[]"]


class TelemetryAttributeDefinition(TypedDict):
    """An attribute's type, documentation and constraints.

    ``values`` closes the set of a scalar attribute; ``elementValues`` closes the set of an array's elements.
    ``sensitive`` marks data that needs special handling; ``cardinality`` records whether a value set is small
    (``"low"``, safe as a metric label) or unbounded (``"high"``).
    """

    type: TelemetryAttributeType
    description: str
    sensitive: NotRequired[bool]
    cardinality: NotRequired[Literal["low", "high"]]
    values: NotRequired[Sequence[str | int | float | bool]]
    elementValues: NotRequired[Sequence[str | int | float | bool]]
    examples: NotRequired[Sequence[Any]]


class TelemetryStartAttributeDefinition(TelemetryAttributeDefinition):
    """A start attribute: passed when the span starts; ``required`` is explicit."""

    required: bool


class TelemetryEventAttributeDefinition(TelemetryAttributeDefinition):
    required: bool


class TelemetryEventDefinition(TypedDict):
    description: str
    attributes: Mapping[str, TelemetryEventAttributeDefinition]


class TelemetryParentAny(TypedDict):
    """Root or any caller span."""

    kind: Literal["any"]


class TelemetryParentRootOrExternal(TypedDict):
    """Root, or a caller-owned span outside the schema."""

    kind: Literal["root_or_external"]


class TelemetryParentSpans(TypedDict):
    """Only the listed schema spans."""

    kind: Literal["spans"]
    spans: Sequence[str]


TelemetryParentDefinition: TypeAlias = TelemetryParentAny | TelemetryParentRootOrExternal | TelemetryParentSpans


class TelemetrySpanStatusRule(TypedDict):
    default: Literal["ok"]
    errorWhen: str


class TelemetrySpanDefinition(TypedDict):
    """One span: description, parent rule, start and end attributes, events and status rule.

    ``startAttributes`` are known when the span starts; ``endAttributes`` are completion enrichment set later with
    ``set_attributes()``, always optional. Both become ordinary attributes of the same span.
    """

    description: str
    parents: TelemetryParentDefinition
    startAttributes: Mapping[str, TelemetryStartAttributeDefinition]
    endAttributes: Mapping[str, TelemetryAttributeDefinition]
    events: NotRequired[Mapping[str, TelemetryEventDefinition]]
    status: TelemetrySpanStatusRule


class TelemetrySchemaDefinition(TypedDict):
    version: int
    spans: Mapping[str, TelemetrySpanDefinition]


OnInvalid: TypeAlias = Literal["raise", "warn", "ignore"]


class TelemetrySchemaError(TypeError):
    """A span name, attribute or event that its schema does not allow (the analog of a TypeScript type error)."""


class TelemetrySchemaWarning(UserWarning):
    """Emitted by ``on_invalid="warn"`` starters for a dropped attribute or event."""


def define_telemetry_schema(schema: TelemetrySchemaDefinition) -> TelemetrySchemaDefinition:
    """Typed identity helper for serializable schema data: returns ``schema`` itself, unvalidated."""
    return schema


def schema_span_names(schemas: Sequence[TelemetrySchemaDefinition]) -> list[str]:
    """Every span name the schemas declare, in declaration order."""
    return [name for schema in schemas for name in schema["spans"]]


# ─── value checks ───────────────────────────────────────────────────────────


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


_SCALAR_CHECKS: dict[str, Callable[[object], bool]] = {
    "string": lambda value: isinstance(value, str),
    "number": _is_number,
    "boolean": lambda value: isinstance(value, bool),
}


def check_attribute_value(definition: TelemetryAttributeDefinition, value: object) -> str | None:
    """Why ``value`` breaks ``definition``, or ``None`` when it fits."""
    kind = definition["type"]
    if kind.endswith("[]"):
        if not isinstance(value, list | tuple):
            return f"expected {kind}, got {type(value).__name__}"
        check = _SCALAR_CHECKS[kind[:-2]]
        allowed = definition.get("elementValues")
        for item in value:
            if not check(item):
                return f"expected {kind}, got an element of type {type(item).__name__}"
            if allowed is not None and item not in allowed:
                return f"element {item!r} is not one of {list(allowed)!r}"
        return None
    if not _SCALAR_CHECKS[kind](value):
        return f"expected {kind}, got {type(value).__name__}"
    allowed = definition.get("values")
    if allowed is not None and value not in allowed:
        return f"{value!r} is not one of {list(allowed)!r}"
    return None


def _check_attributes(
    definitions: Mapping[str, TelemetryAttributeDefinition],
    attributes: SpanAttributes | None,
    required: bool,
    what: str,
) -> dict[str, str]:
    """Problems by attribute name (``""`` for a missing required one)."""
    problems: dict[str, str] = {}
    present = attributes or {}
    for name, value in present.items():
        if value is None:
            continue
        definition = definitions.get(name)
        if definition is None:
            problems[name] = f"{what} does not declare attribute {name!r}"
            continue
        problem = check_attribute_value(definition, value)
        if problem is not None:
            problems[name] = f"{what} attribute {name!r}: {problem}"
    if required:
        missing = [
            name
            for name, definition in definitions.items()
            if definition.get("required") is True and present.get(name) is None
        ]
        if missing:
            problems[""] = f"{what} requires attribute{'s' if len(missing) > 1 else ''} {', '.join(map(repr, missing))}"
    return problems


def check_start_attributes(
    name: str, definition: TelemetrySpanDefinition, attributes: SpanAttributes | None
) -> list[str]:
    """Problems with a span's start attributes: undeclared, mistyped, outside a closed set, or missing."""
    return list(_check_attributes(definition["startAttributes"], attributes, True, f"Span {name!r}").values())


def check_end_attributes(
    name: str, definition: TelemetrySpanDefinition, attributes: SpanAttributes | None
) -> list[str]:
    """Problems with completion attributes; all of them are optional."""
    return list(_check_attributes(definition["endAttributes"], attributes, False, f"Span {name!r}").values())


def check_event(
    name: str, definition: TelemetrySpanDefinition, event: str, attributes: SpanAttributes | None
) -> list[str]:
    """Problems with an event: undeclared, or attributes that break its definition."""
    events = definition.get("events") or {}
    event_definition = events.get(event)
    if event_definition is None:
        return [f"Span {name!r} does not declare event {event!r}"]
    return list(
        _check_attributes(
            event_definition["attributes"], attributes, True, f"Event {event!r} of span {name!r}"
        ).values()
    )


# ─── typed starters ─────────────────────────────────────────────────────────


def _report(on_invalid: OnInvalid, problems: Sequence[str]) -> None:
    if on_invalid == "raise":
        raise TelemetrySchemaError("; ".join(problems))
    if on_invalid == "warn":
        warnings.warn("; ".join(problems), TelemetrySchemaWarning, stacklevel=4)


def _without(attributes: SpanAttributes | None, names: Sequence[str]) -> dict[str, AttributeValue | None]:
    return {key: value for key, value in (attributes or {}).items() if key not in names}


_EMPTY_SPAN: TelemetrySpanDefinition = {
    "description": "",
    "parents": {"kind": "any"},
    "startAttributes": {},
    "endAttributes": {},
    "status": {"default": "ok", "errorWhen": ""},
}


class SchemaTelemetrySpan:
    """A span view restricted to one schema span: ``add_event`` and ``set_attributes`` accept only what the span
    declares (its events, its end attributes). ``start_span`` (an open child) and ``set_status`` pass through."""

    __slots__ = ("_definition", "_name", "_on_invalid", "_span")

    def __init__(
        self, span: TelemetrySpan, name: str, definition: TelemetrySpanDefinition, on_invalid: OnInvalid
    ) -> None:
        self._span = span
        self._name = name
        self._definition = definition
        self._on_invalid: OnInvalid = on_invalid

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> TelemetrySpanDefinition:
        return self._definition

    @property
    def span(self) -> TelemetrySpan:
        """The adapter's span, an open parent context."""
        return self._span

    @property
    def content_capture(self) -> ContentCapture | None:
        return getattr(self._span, "content_capture", None)

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        return self._span.start_span(options, callback)

    def set_status(self, status: SpanStatus) -> None:
        self._span.set_status(status)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        problems = check_event(self._name, self._definition, name, attributes)
        if problems:
            _report(self._on_invalid, problems)
            return
        self._span.add_event(name, attributes)

    def set_attributes(self, attributes: SpanAttributes) -> None:
        problems = _check_attributes(self._definition["endAttributes"], attributes, False, f"Span {self._name!r}")
        if problems:
            _report(self._on_invalid, list(problems.values()))
            attributes = _without(attributes, list(problems))
        self._span.set_attributes(attributes)

    def __repr__(self) -> str:
        return f"SchemaTelemetrySpan({self._name!r}, {self._span!r})"


class TypedSpanStarter:
    """Starts spans of a fixed vocabulary from one explicit parent context.

    ``starter(name, attributes, callback)`` checks ``name`` and ``attributes`` against the span's definition and
    calls ``callback(span, start_child_span)`` inside the new span: ``span`` is a :class:`SchemaTelemetrySpan`
    and ``start_child_span`` a starter over the same schemas, bound to that span.
    """

    __slots__ = ("_context", "_index", "_on_invalid")

    def __init__(
        self, context: TelemetryContext, index: Mapping[str, TelemetrySpanDefinition], on_invalid: OnInvalid
    ) -> None:
        self._context = context
        self._index = index
        self._on_invalid: OnInvalid = on_invalid

    @property
    def context(self) -> TelemetryContext:
        """The parent context."""
        return self._context

    @property
    def span_names(self) -> list[str]:
        return list(self._index)

    def __call__(
        self,
        name: str,
        attributes: SpanAttributes,
        callback: Callable[[SchemaTelemetrySpan, TypedSpanStarter], T | Awaitable[T]],
    ) -> Awaitable[T]:
        index, on_invalid = self._index, self._on_invalid
        definition = index.get(name)
        context = self._context
        if definition is None:
            _report(on_invalid, [f"Unknown telemetry span {name!r}"])
            definition = _EMPTY_SPAN
            context = NOOP_TELEMETRY_CONTEXT
        else:
            problems = _check_attributes(definition["startAttributes"], attributes, True, f"Span {name!r}")
            if problems:
                _report(on_invalid, list(problems.values()))
                attributes = _without(attributes, [key for key in problems if key])
        bound = definition

        def run(span: TelemetrySpan) -> T | Awaitable[T]:
            return callback(
                SchemaTelemetrySpan(span, name, bound, on_invalid), TypedSpanStarter(span, index, on_invalid)
            )

        return context.start_span({"name": name, "attributes": attributes}, run)

    def __repr__(self) -> str:
        return f"TypedSpanStarter({self._context!r}, spans={self.span_names!r})"


def _index(schemas: Sequence[TelemetrySchemaDefinition]) -> dict[str, TelemetrySpanDefinition]:
    index: dict[str, TelemetrySpanDefinition] = {}
    for schema in schemas:
        for name, definition in schema["spans"].items():
            if name in index:
                raise TelemetrySchemaError(f"Duplicate telemetry span name {name!r} across schemas")
            index[name] = definition
    return index


def create_typed_span_starter(
    context: TelemetryContext,
    schemas: Sequence[TelemetrySchemaDefinition],
    *,
    on_invalid: OnInvalid = "raise",
) -> TypedSpanStarter:
    """Bind an explicit parent context to the combined span vocabulary of one or more schemas.

    Raises :class:`TelemetrySchemaError` when two schemas declare the same span name. The schemas are indexed once;
    reuse a starter (or its children) rather than creating one per span on hot paths.
    """
    if not schemas:
        raise TelemetrySchemaError("create_typed_span_starter() needs at least one schema")
    return TypedSpanStarter(context, _index(schemas), on_invalid)


def bind_typed_span_starter(context: TelemetryContext, starter: TypedSpanStarter) -> TypedSpanStarter:
    """A starter over ``starter`` 's vocabulary bound to another parent context, without re-indexing schemas."""
    return TypedSpanStarter(context, starter._index, starter._on_invalid)
