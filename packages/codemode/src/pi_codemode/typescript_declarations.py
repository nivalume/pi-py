"""TypeScript declarations of the script-visible API for JavaScript scripts (port of ``declarations.ts``).

Tools become members of ``declare const tools``, globals ``declare function`` statements, and ``ns.member`` globals
members of ``declare const ns``. Descriptions become doc comments; JSON Schemas become types:

- objects on one line (``{ a: string; b?: number; }``) with properties sorted by name, or one property per line with
  ``//`` comments when a property has a description or constraints the type cannot express
  (``// @integer @minimum 1 @default 20``); ``Array<T>`` for arrays, tuples for ``prefixItems``;
- unions from ``anyOf``/``oneOf``/``type`` arrays, ``const`` and ``enum`` as literal types, ``allOf`` as an
  intersection;
- local references (``#/$defs/...``, ``#/definitions/...``) expanded at most 32 times per schema; recursive and remote
  references render as ``unknown``.

The output is upstream's, character for character. Schemas only shape the declarations; values are not validated
against them.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TypeGuard
from urllib.parse import unquote

from .identifier import to_codemode_identifier
from .types import CodemodeJsonSchema, CodemodeToolLike

__all__ = [
    "DEFAULT_INPUT_SCHEMA_MAX_CHARS",
    "render_declarations",
    "render_tool_output_type",
    "render_tool_sample",
    "render_tool_signature",
    "schema_to_type",
]

_IDENTIFIER: Final = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_INDENT: Final = "  "
#: Largest rendered input type, in characters, before it becomes ``unknown``.
DEFAULT_INPUT_SCHEMA_MAX_CHARS: Final = 16_000
#: Local ``$ref`` expansions per rendered schema, so shared definitions cannot blow up the output.
_MAX_REF_EXPANSIONS: Final = 32
#: JSON Schema keywords with a number or string value shown as ``@keyword value`` property tags.
_TAG_KEYWORDS: Final = (
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
)
#: Tag keywords whose values all hold at once, so only the largest (lower) or smallest (upper) matters.
_LOWER_BOUNDS: Final = frozenset({"minimum", "exclusiveMinimum", "minLength", "minItems"})
_UPPER_BOUNDS: Final = frozenset({"maximum", "exclusiveMaximum", "maxLength", "maxItems"})
#: Longest property tag; longer ones (large defaults, long patterns) are left out.
_MAX_TAG_CHARS: Final = 80
#: Characters that end a ``//`` comment.
_LINE_TERMINATORS: Final = re.compile("[\r\n\u2028\u2029]")


def render_declarations(*, tools: Iterable[CodemodeToolLike] = (), globals: Iterable[CodemodeToolLike] = ()) -> str:
    """TypeScript declarations of the script-visible API."""
    sections: list[str] = []
    tools = list(tools)
    if tools:
        members = [f"{_doc_comment(tool.description, _INDENT)}{_INDENT}{render_tool_signature(tool)}" for tool in tools]
        sections.append("declare const tools: {\n" + "\n".join(members) + "\n};")
    namespaces: dict[str, list[str]] = {}
    for global_ in globals:
        namespace, dot, member = global_.name.partition(".")
        if not dot:
            sections.append(_render_global(f"declare function {global_.name}", global_, ""))
            continue
        namespaces.setdefault(namespace, []).append(_render_global(member, global_, _INDENT))
    for namespace, members in namespaces.items():
        sections.append(f"declare const {namespace}: {{\n" + "\n".join(members) + "\n};")
    return "\n\n".join(sections)


def render_tool_signature(tool: CodemodeToolLike, *, input_max_chars: int | None = None) -> str:
    """One tool as a member of the ``tools`` object: ``name(args: T): Promise<R>;`` with the name as the identifier
    scripts use. Input types longer than ``input_max_chars`` (default 16,000) render as ``unknown``."""
    input_schema = tool.input_schema
    input = (
        "unknown"
        if input_schema is None
        else schema_to_type(
            input_schema, max_chars=DEFAULT_INPUT_SCHEMA_MAX_CHARS if input_max_chars is None else input_max_chars
        )
    )
    return (
        f"{to_codemode_identifier(tool.name)}(args: {input}): Promise<{render_tool_output_type(tool.output_schema)}>;"
    )


def render_tool_sample(tool: CodemodeToolLike, *, input_max_chars: int | None = None) -> str:
    """A tool's sample: the description followed by the tool's declaration, for tool listings."""
    declaration = f"declare const tools: {{ {render_tool_signature(tool, input_max_chars=input_max_chars)} }};"
    description = (tool.description or "").strip()
    return f"{description}\n\ncodemode tool declaration:\n```ts\n{declaration}\n```"


def render_tool_output_type(schema: CodemodeJsonSchema | None) -> str:
    """The type a tool call resolves to: the output schema's type, and ``unknown`` without a schema."""
    return "unknown" if schema is None else schema_to_type(schema)


def schema_to_type(schema: CodemodeJsonSchema, *, max_chars: int | None = None) -> str:
    """A JSON Schema as a TypeScript type expression.

    A result longer than ``max_chars`` is rendered again without the constraint tags, and renders as ``unknown`` if
    it is still too long.
    """
    rendered = _Renderer(schema, tags=True).type(schema)
    if max_chars is None or len(rendered) <= max_chars:
        return rendered
    untagged = _Renderer(schema, tags=False).type(schema)
    return untagged if len(untagged) <= max_chars else "unknown"


def _render_global(head: str, global_: CodemodeToolLike, indent: str) -> str:
    if global_.signature is not None:
        return f"{_doc_comment(global_.description, indent)}{indent}{head}{global_.signature};"
    input = "unknown" if global_.input_schema is None else schema_to_type(global_.input_schema)
    output = "unknown" if global_.output_schema is None else schema_to_type(global_.output_schema)
    return f"{_doc_comment(global_.description, indent)}{indent}{head}(args: {input}): Promise<{output}>;"


def _doc_comment(description: str | None, indent: str) -> str:
    text = (description or "").strip()
    if not text:
        return ""
    lines = re.split(r"\r?\n", text.replace("*/", "*\\/"))
    if len(lines) == 1:
        return f"{indent}/** {lines[0]} */\n"
    body = "\n".join(f"{indent} *{f' {line}' if line else ''}" for line in lines)
    return f"{indent}/**\n{body}\n{indent} */\n"


def _js_number(value: float) -> str:
    """``String(value)`` of a finite number: ``5`` rather than ``5.0``, ``1e+21``, ``1.5e-7``."""
    if isinstance(value, int) and abs(value) < 10**21:
        return str(value)
    value = float(value)
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    text = repr(value)
    if "e" not in text:
        return text
    if 1e-6 <= abs(value) < 1e21:
        # Python switches to exponent notation earlier than JavaScript does.
        mantissa, exponent = text.split("e")
        digits = mantissa.replace("-", "").replace(".", "")
        point = mantissa.replace("-", "").find(".")
        point = len(mantissa.replace("-", "")) if point == -1 else point
        position = point + int(exponent)
        sign = "-" if value < 0 else ""
        if position <= 0:
            return f"{sign}0.{'0' * -position}{digits}".rstrip("0").rstrip(".")
        return f"{sign}{digits[:position]}.{digits[position:]}".rstrip("0").rstrip(".")
    mantissa, exponent = text.split("e")
    exponent_sign = "-" if exponent.startswith("-") else "+"
    return f"{mantissa}e{exponent_sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def _js_json(value: Any) -> str | None:
    """``JSON.stringify(value)``: compact, non-ASCII kept, numbers as JavaScript prints them; ``None`` for
    ``undefined``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _js_number(value) if math.isfinite(value) else "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            text = _js_json(item)
            if text is not None:
                parts.append(f"{json.dumps(str(key), ensure_ascii=False)}:{text}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_js_json(item) or "null" for item in value) + "]"
    return None


def _property_key(name: str) -> str:
    return name if _IDENTIFIER.match(name) else json.dumps(name, ensure_ascii=False)


def _union(types: Iterable[str]) -> str:
    unique = list(dict.fromkeys(types))
    if "unknown" in unique:
        return "unknown"
    return "never" if not unique else " | ".join(unique)


def _utf16_order(name: str) -> bytes:
    """JavaScript's default sort order: by UTF-16 code units."""
    return name.encode("utf-16-be", "surrogatepass")


def _is_number(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _description_of(schema: Any) -> str:
    description = schema.get("description") if isinstance(schema, Mapping) else None
    return description.strip() if isinstance(description, str) else ""


def _constraint_tags(schema: Any) -> str:
    """Constraints of a property that its TypeScript type cannot express, as JSDoc-style tags:
    ``@integer @minimum 1 @maximum 100 @default 20``. Empty when there are none."""
    if not isinstance(schema, Mapping):
        return ""
    prop: Mapping[str, Any] = schema
    # Optional fields such as Pydantic's `int | None` keep their constraints in the non-null branch.
    # The branch's constraints apply together with the property's, so both are read.
    variants = prop.get("anyOf") if isinstance(prop.get("anyOf"), list) else prop.get("oneOf")
    non_null = (
        [variant for variant in variants if not (isinstance(variant, Mapping) and variant.get("type") == "null")]
        if isinstance(variants, list)
        else []
    )
    branch = non_null[0] if len(non_null) == 1 else None
    branches: list[Mapping[str, Any]] = [prop, branch] if isinstance(branch, Mapping) else [prop]
    # Draft-04 and OpenAPI 3.0 make `minimum`/`maximum` exclusive with a boolean `exclusiveMinimum`/`exclusiveMaximum`.
    schemas: list[dict[str, Any]] = []
    for entry in branches:
        normalized = dict(entry)
        if entry.get("exclusiveMinimum") is True:
            normalized["exclusiveMinimum"] = entry.get("minimum")
            normalized.pop("minimum", None)
        if entry.get("exclusiveMaximum") is True:
            normalized["exclusiveMaximum"] = entry.get("maximum")
            normalized.pop("maximum", None)
        schemas.append(normalized)

    def is_integer(kind: Any) -> bool:
        return kind == "integer" or (isinstance(kind, list) and "integer" in kind)

    tags: list[str] = []
    if any(is_integer(entry.get("type")) for entry in schemas):
        tags.append("@integer")
    for keyword in _TAG_KEYWORDS:
        values = [entry.get(keyword) for entry in schemas]
        if keyword in _LOWER_BOUNDS or keyword in _UPPER_BOUNDS:
            numbers = [value for value in values if _is_number(value)]
            if not numbers:
                continue
            bound = max(numbers) if keyword in _LOWER_BOUNDS else min(numbers)
            tags.append(f"@{keyword} {_js_number(bound) if math.isfinite(bound) else _js_json(bound)}")
            continue
        shown: list[Any] = []
        for value in values:
            if (isinstance(value, str) or (_is_number(value) and math.isfinite(value))) and value not in shown:
                shown.append(value)
        tags.extend(f"@{keyword} {value if isinstance(value, str) else _js_number(value)}" for value in shown)
    if any(entry.get("uniqueItems") is True for entry in schemas):
        tags.append("@uniqueItems")
    # Annotations, unlike constraints, come from the property itself.
    default = prop.get("default")
    if default is not None:
        text = _js_json(default)
        if text is not None:
            tags.append(f"@default {text}")
    if prop.get("deprecated") is True:
        tags.append("@deprecated")
    return " ".join(
        _LINE_TERMINATORS.sub(lambda match: f"\\u{ord(match.group(0)):04x}", tag)
        for tag in tags
        if len(tag) <= _MAX_TAG_CHARS
    )


@dataclass(slots=True)
class _Renderer:
    root: CodemodeJsonSchema
    #: Whether property comments include constraint tags.
    tags: bool
    #: References being expanded on the current path, to stop at recursive types.
    resolving: set[str] = field(default_factory=lambda: set[str]())
    expansions: int = 0

    def resolve(self, ref: str) -> CodemodeJsonSchema | None:
        if ref != "#" and not ref.startswith("#/"):
            return None
        current: Any = self.root
        for segment in (part for part in ref[2:].split("/") if part):
            key = unquote(segment).replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        if isinstance(current, (bool, Mapping)):
            return current
        return None

    def type(self, schema: Any) -> str:
        if schema is True:
            return "unknown"
        if schema is False:
            return "never"
        if not isinstance(schema, Mapping):
            return "unknown"
        if isinstance(schema.get("$ref"), str):
            ref: str = schema["$ref"]
            if ref in self.resolving or self.expansions >= _MAX_REF_EXPANSIONS:
                return "unknown"
            target = self.resolve(ref)
            if target is None:
                return "unknown"
            self.expansions += 1
            self.resolving.add(ref)
            try:
                return self.type(target)
            finally:
                self.resolving.discard(ref)
        if "const" in schema:
            return _js_json(schema["const"]) or "unknown"
        if isinstance(schema.get("enum"), list):
            return _union(_js_json(value) or "unknown" for value in schema["enum"])
        variants = (
            schema["anyOf"]
            if isinstance(schema.get("anyOf"), list)
            else schema["oneOf"]
            if isinstance(schema.get("oneOf"), list)
            else None
        )
        if variants is not None:
            return _union(self.type(variant) for variant in variants)
        if isinstance(schema.get("allOf"), list):
            parts = [part for part in (self.type(entry) for entry in schema["allOf"]) if part != "unknown"]
            if not parts:
                return "unknown"
            return " & ".join(f"({part})" if " | " in part else part for part in parts)
        if "type" not in schema:
            if "properties" in schema or "additionalProperties" in schema or "required" in schema:
                return self.object(schema)
            if "items" in schema or "prefixItems" in schema:
                return self.array(schema)
            return "unknown"
        kind = schema["type"]
        if isinstance(kind, list):
            return _union(self.type({**schema, "type": entry}) for entry in kind)
        if kind == "string":
            return "string"
        if kind in ("number", "integer"):
            return "number"
        if kind == "boolean":
            return "boolean"
        if kind == "null":
            return "null"
        if kind == "array":
            return self.array(schema)
        if kind == "object":
            return self.object(schema)
        return "unknown"

    def array(self, schema: Mapping[str, Any]) -> str:
        items = schema.get("items")
        if "items" in schema and not isinstance(items, list):
            return f"Array<{self.type(items)}>"
        prefix = schema.get("prefixItems")
        tuple_items: Sequence[Any] = prefix if isinstance(prefix, list) else items if isinstance(items, list) else []
        if tuple_items:
            return "[" + ", ".join(self.type(item) for item in tuple_items) + "]"
        return "unknown[]"

    def object(self, schema: Mapping[str, Any]) -> str:
        raw_properties = schema.get("properties")
        properties: Mapping[str, Any] = raw_properties if isinstance(raw_properties, Mapping) else {}
        raw_required = schema.get("required")
        required = set(raw_required) if isinstance(raw_required, list) else set()
        names = sorted(properties, key=_utf16_order)
        members = [
            f"{_property_key(name)}{'' if name in required else '?'}: {self.type(properties[name])};" for name in names
        ]
        additional = schema.get("additionalProperties", ...)
        if additional is not ... and additional is not False:
            members.append(f"[key: string]: {'unknown' if additional is True else self.type(additional)};")
        elif additional is ... and not names:
            members.append("[key: string]: unknown;")
        if not members:
            return "{}"
        comments: list[list[str]] = []
        for name in names:
            lines = [
                line.strip() for line in _LINE_TERMINATORS.split(_description_of(properties[name])) if line.strip()
            ]
            tags = _constraint_tags(properties[name]) if self.tags else ""
            comments.append([*lines, tags] if tags else lines)
        if not any(comments):
            return "{ " + " ".join(members) + " }"
        lines = ["{"]
        for index, comment in enumerate(comments):
            lines.extend(f"{_INDENT}// {line}" for line in comment)
            lines.append(_INDENT + members[index].replace("\n", f"\n{_INDENT}"))
        lines.extend(f"{_INDENT}{member}" for member in members[len(names) :])
        lines.append("}")
        return "\n".join(lines)
