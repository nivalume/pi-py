"""Python declarations of the script-visible API for Python scripts, rendered from JSON Schemas (port of
``declarations.ts``).

Upstream renders TypeScript (:mod:`pi_codemode.typescript_declarations`); Python scripts read tools as a stub class::

    class tools:
        async def read(self, *, path: str) -> str:
            \"\"\"Read a file\"\"\"

- An object input whose property names are identifiers becomes keyword-only parameters
  (``path: str``; optional ones ``= ...``), any other input one positional parameter
  (``args: T, /``). Scripts may pass either form: ``tools.read(path="x")`` or
  ``tools.read({"path": "x"})``.
- Nested objects are inline ``TypedDict[{...}]`` types (PEP 764 notation) with
  ``NotRequired[...]`` for optional keys; ``**dict[str, T]`` stands for extra keys.
- Property descriptions and the constraints a type cannot express (``# @minimum 1 @default 20``)
  become ``#`` comments; tags are dropped first when a type exceeds its budget. (Upstream's
  ``@integer`` tag is left out: ``int`` says it.)
- Local references (``#/$defs/...``, ``#/definitions/...``) are expanded; recursive and remote
  references render as ``Any``.

Schemas only shape the declarations; values are not validated against them.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.parse import unquote

from .identifier import is_codemode_identifier, to_codemode_identifier
from .types import CodemodeJsonSchema, CodemodeToolLike

__all__ = [
    "DEFAULT_INPUT_SCHEMA_MAX_CHARS",
    "render_declarations",
    "render_tool_output_type",
    "render_tool_sample",
    "render_tool_signature",
    "schema_to_type",
]

INDENT: Final = "    "
#: Largest rendered input (parameter list), in characters, before it falls back to ``Any``.
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
#: Characters that end a line (a ``#`` comment ends at the first two; upstream escapes all four).
_LINE_TERMINATORS: Final = re.compile("[\r\n\u2028\u2029]")
_FALLBACK_PARAMETERS: Final = "args: Any = None, /, **kwargs: Any"


def render_declarations(*, tools: Iterable[CodemodeToolLike] = (), globals: Iterable[CodemodeToolLike] = ()) -> str:
    """Python declarations of the script-visible API.

    Tools become methods of ``class tools``, globals ``async def`` functions, and
    ``ns.member`` globals methods of ``class ns``. Descriptions become docstrings and
    schemas become parameter and return types.
    """
    sections: list[str] = []
    members = [_method(_tool_head(tool), tool.description, INDENT) for tool in tools]
    if members:
        sections.append("class tools:\n" + "\n".join(members))
    namespaces: dict[str, list[str]] = {}
    for global_ in globals:
        namespace, dot, member = global_.name.partition(".")
        if not dot:
            sections.append(_method(_global_head(global_.name, global_, method=False), global_.description, ""))
            continue
        namespaces.setdefault(namespace, []).append(
            _method(_global_head(member, global_, method=True), global_.description, INDENT)
        )
    for namespace, rendered in namespaces.items():
        sections.append(f"class {namespace}:\n" + "\n".join(rendered))
    return "\n\n".join(sections)


def render_tool_signature(tool: CodemodeToolLike, *, input_max_chars: int | None = None) -> str:
    """One tool as a method of ``class tools``: ``async def name(self, ...) -> R: ...``.

    The name is the identifier scripts use. Parameter lists longer than ``input_max_chars``
    (default :data:`DEFAULT_INPUT_SCHEMA_MAX_CHARS`) are rendered again without constraint
    tags, and as ``args: Any = None, /, **kwargs: Any`` if still too long.
    """
    return f"{_tool_head(tool, input_max_chars)}: ..."


def render_tool_sample(tool: CodemodeToolLike, *, input_max_chars: int | None = None) -> str:
    """A tool's sample: its description followed by its declaration, for tool listings."""
    signature = render_tool_signature(tool, input_max_chars=input_max_chars).replace("\n", "\n" + INDENT)
    description = (tool.description or "").strip()
    return f"{description}\n\ncodemode tool declaration:\n```python\nclass tools:\n{INDENT}{signature}\n```"


def render_tool_output_type(schema: CodemodeJsonSchema | None) -> str:
    """The type a tool call resolves to: the output schema's type, ``Any`` without a schema."""
    return "Any" if schema is None else schema_to_type(schema)


def schema_to_type(schema: CodemodeJsonSchema, *, max_chars: int | None = None) -> str:
    """Convert a JSON Schema to a Python type expression.

    Objects with properties render as ``TypedDict[{"a": str, "b": NotRequired[float]}]`` with
    keys sorted, on one line, or one key per line with ``#`` comments when a property has a
    description or constraints the type cannot express. Arrays are ``list[T]``, tuples
    ``tuple[A, B]``, maps ``dict[str, T]``, constants and enums ``Literal[...]``. A result
    longer than ``max_chars`` is rendered again without the constraint tags, and renders as
    ``Any`` if it is still too long.
    """
    rendered = _Renderer(schema, tags=True).type(schema)
    if max_chars is None or len(rendered) <= max_chars:
        return rendered
    untagged = _Renderer(schema, tags=False).type(schema)
    return untagged if len(untagged) <= max_chars else "Any"


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def _tool_head(tool: CodemodeToolLike, input_max_chars: int | None = None) -> str:
    limit = DEFAULT_INPUT_SCHEMA_MAX_CHARS if input_max_chars is None else input_max_chars
    parameters = _parameters(tool.input_schema, limit)
    returns = render_tool_output_type(tool.output_schema)
    return _head(f"async def {to_codemode_identifier(tool.name, 'python')}", ["self", *parameters], returns)


def _global_head(name: str, global_: CodemodeToolLike, *, method: bool) -> str:
    if global_.signature is not None:
        signature = global_.signature.strip()
        if method and signature.startswith("("):
            signature = "(self)" + signature[2:] if signature.startswith("()") else "(self, " + signature[1:]
        return f"async def {name}{signature}"
    parameters = _parameters(global_.input_schema, None)
    return _head(
        f"async def {name}",
        ["self", *parameters] if method else parameters,
        render_tool_output_type(global_.output_schema),
    )


def _head(start: str, parameters: Sequence[str], returns: str) -> str:
    if any("\n" in parameter for parameter in parameters):
        lines = [f"{start}("]
        lines.extend(f"{INDENT}{parameter.replace(chr(10), chr(10) + INDENT)}," for parameter in parameters)
        lines.append(f") -> {returns}")
        return "\n".join(lines)
    return f"{start}({', '.join(parameters)}) -> {returns}"


def _method(head: str, description: str | None, indent: str) -> str:
    """``head`` with a docstring body, or ``: ...`` without a description."""
    head = head.replace("\n", "\n" + indent)
    text = (description or "").strip()
    if not text:
        return f"{indent}{head}: ..."
    escaped = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    lines = re.split(r"\r?\n", escaped)
    body = indent + INDENT
    if len(lines) == 1:
        return f'{indent}{head}:\n{body}"""{lines[0]}"""'
    inner = "\n".join(f"{body}{line}" if line else "" for line in lines)
    return f'{indent}{head}:\n{body}"""\n{inner}\n{body}"""'


def _parameters(schema: CodemodeJsonSchema | None, max_chars: int | None) -> list[str]:
    """The parameters after ``self``: keyword-only for an object of identifier keys, else one positional."""
    if schema is None:
        return ["args: Any = None", "/"]
    for tags in (True, False):
        parameters = _Renderer(schema, tags=tags).parameters(schema)
        if max_chars is None or sum(len(parameter) for parameter in parameters) <= max_chars:
            return parameters
    return [_FALLBACK_PARAMETERS]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _union(types: Iterable[str]) -> str:
    unique = list(dict.fromkeys(types))
    if "Any" in unique:
        return "Any"
    return "Never" if not unique else " | ".join(unique)


def _json(value: Any) -> str:
    """Compact JSON like ``JSON.stringify``, or ``None`` for values JSON cannot hold."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _literal(value: Any) -> str | None:
    """A ``Literal[...]`` member for a JSON value, or ``None`` when Python's ``Literal`` cannot hold it."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _json(value)
    return None


def _literal_type(values: Sequence[Any]) -> str:
    """``Literal[...]`` for the values Python's ``Literal`` can hold, unioned with types for the others."""
    members: list[str] = []
    others: list[str] = []
    for value in values:
        literal = _literal(value)
        if literal is not None:
            if literal not in members:
                members.append(literal)
        elif isinstance(value, float):
            others.append("float" if math.isfinite(value) else "Any")
        elif isinstance(value, list):
            others.append("list[Any]")
        elif isinstance(value, dict):
            others.append("dict[str, Any]")
        else:
            others.append("Any")
    parts = ([f"Literal[{', '.join(members)}]"] if members else []) + others
    return _union(parts)


def _description_of(schema: Any) -> str:
    description = schema.get("description") if isinstance(schema, Mapping) else None
    return description.strip() if isinstance(description, str) else ""


def _escape_terminators(text: str) -> str:
    return _LINE_TERMINATORS.sub(lambda match: f"\\u{ord(match.group(0)):04x}", text)


def _number_text(value: float) -> str:
    """A number as JavaScript prints it: ``1`` for ``1.0``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _constraint_tags(schema: Any) -> str:
    """Constraints a Python type cannot express, as tags: ``@minimum 1 @default 20``."""
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

    # Upstream's `@integer` tag is left out: Python's `int` says it.
    tags: list[str] = []
    for keyword in _TAG_KEYWORDS:
        values = [entry.get(keyword) for entry in schemas]
        if keyword in _LOWER_BOUNDS or keyword in _UPPER_BOUNDS:
            numbers = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if not numbers:
                continue
            bound = max(numbers) if keyword in _LOWER_BOUNDS else min(numbers)
            tags.append(f"@{keyword} {_number_text(bound)}")
            continue
        for value in dict.fromkeys(values):
            if isinstance(value, str):
                tags.append(f"@{keyword} {value}")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                tags.append(f"@{keyword} {_number_text(value)}")
    if any(entry.get("uniqueItems") is True for entry in schemas):
        tags.append("@uniqueItems")
    # Annotations, unlike constraints, come from the property itself.
    default = prop.get("default")
    if default is not None:
        try:
            tags.append(f"@default {_json(default)}")
        except ValueError:
            pass
    if prop.get("deprecated") is True:
        tags.append("@deprecated")
    return " ".join(_escape_terminators(tag) for tag in tags if len(tag) <= _MAX_TAG_CHARS)


@dataclass(slots=True)
class _Renderer:
    root: CodemodeJsonSchema
    #: Whether property comments include constraint tags.
    tags: bool
    #: References being expanded on the current path, to stop at recursive types.
    resolving: set[str] = field(default_factory=set)
    expansions: int = 0

    def resolve(self, ref: str) -> CodemodeJsonSchema | None:
        if ref != "#" and not ref.startswith("#/"):
            return None
        current: Any = self.root
        for segment in [part for part in ref[2:].split("/") if part]:
            key = unquote(segment).replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        if isinstance(current, bool):
            return current
        return current if isinstance(current, Mapping) else None

    def dereference(self, schema: Any) -> tuple[Any, str | None]:
        """``schema`` with a top-level ``$ref`` followed, and the ref now being expanded."""
        if not isinstance(schema, Mapping) or not isinstance(schema.get("$ref"), str):
            return schema, None
        ref: str = schema["$ref"]
        if ref in self.resolving or self.expansions >= _MAX_REF_EXPANSIONS:
            return True, None
        target = self.resolve(ref)
        if target is None:
            return True, None
        self.expansions += 1
        return target, ref

    def type(self, schema: Any) -> str:
        if schema is True:
            return "Any"
        if schema is False:
            return "Never"
        if not isinstance(schema, Mapping):
            return "Any"
        if isinstance(schema.get("$ref"), str):
            target, ref = self.dereference(schema)
            if ref is None:
                return "Any"
            self.resolving.add(ref)
            try:
                return self.type(target)
            finally:
                self.resolving.discard(ref)

        if "const" in schema:
            return _literal_type([schema["const"]])
        if isinstance(schema.get("enum"), list):
            return _literal_type(schema["enum"])

        variants = schema.get("anyOf") if isinstance(schema.get("anyOf"), list) else schema.get("oneOf")
        if isinstance(variants, list):
            return _union(self.type(variant) for variant in variants)
        if isinstance(schema.get("allOf"), list):
            return self.intersection(schema["allOf"])

        kind = schema.get("type")
        if isinstance(kind, list):
            return _union(self.type({**schema, "type": entry}) for entry in kind)
        if kind == "string":
            return "str"
        if kind == "number":
            return "float"
        if kind == "integer":
            return "int"
        if kind == "boolean":
            return "bool"
        if kind == "null":
            return "None"
        if kind == "array":
            return self.array(schema)
        if kind == "object":
            return self.object(schema)
        if kind is None:
            if "properties" in schema or "additionalProperties" in schema or "required" in schema:
                return self.object(schema)
            if "items" in schema or "prefixItems" in schema:
                return self.array(schema)
        return "Any"

    def intersection(self, parts: list[Any]) -> str:
        """``allOf``. Python has no intersection types: objects merge, otherwise the most specific part wins."""
        resolved = [self.dereference(part)[0] for part in parts]
        if resolved and all(_is_object_schema(part) for part in resolved):
            merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for part in resolved:
                properties = part.get("properties")
                if isinstance(properties, Mapping):
                    merged["properties"].update(properties)
                required = part.get("required")
                if isinstance(required, list):
                    merged["required"].extend(required)
                if "additionalProperties" in part:
                    merged["additionalProperties"] = part["additionalProperties"]
            return self.object(merged)
        rendered = [entry for entry in (self.type(part) for part in parts) if entry != "Any"]
        if not rendered:
            return "Any"
        literal = next((entry for entry in rendered if entry.startswith("Literal[")), None)
        return literal if literal is not None else rendered[0]

    def array(self, schema: Mapping[str, Any]) -> str:
        items = schema.get("items")
        if items is not None and not isinstance(items, list):
            return f"list[{self.type(items)}]"
        prefix = schema.get("prefixItems")
        tuple_items = prefix if isinstance(prefix, list) else items if isinstance(items, list) else []
        if tuple_items:
            return f"tuple[{', '.join(self.type(item) for item in tuple_items)}]"
        return "list[Any]"

    def comment_lines(self, schema: Any) -> list[str]:
        lines = [line.strip() for line in _LINE_TERMINATORS.split(_description_of(schema)) if line.strip()]
        tags = _constraint_tags(schema) if self.tags else ""
        return [*lines, tags] if tags else lines

    def object(self, schema: Mapping[str, Any]) -> str:
        raw_properties = schema.get("properties")
        properties: Mapping[str, Any] = raw_properties if isinstance(raw_properties, Mapping) else {}
        raw_required = schema.get("required")
        required = set(raw_required) if isinstance(raw_required, list) else set()
        names = sorted(properties)
        additional = schema.get("additionalProperties")
        extra = None
        if additional is not None and additional is not False:
            extra = "Any" if additional is True else self.type(additional)
        if not names:
            if additional is False:
                return "TypedDict[{}]"
            return f"dict[str, {extra or 'Any'}]"
        members = []
        for name in names:
            value = self.type(properties[name])
            members.append(f"{_json(name)}: {value if name in required else f'NotRequired[{value}]'}")
        if extra is not None:
            members.append(f"**dict[str, {extra}]")
        comments = [self.comment_lines(properties[name]) for name in names]
        if not any(comments):
            return "TypedDict[{" + ", ".join(members) + "}]"
        lines = ["TypedDict[{"]
        for index, member in enumerate(members):
            for line in comments[index] if index < len(comments) else []:
                lines.append(f"{INDENT}# {line}")
            lines.append(f"{INDENT}{member.replace(chr(10), chr(10) + INDENT)},")
        lines.append("}]")
        return "\n".join(lines)

    def parameters(self, schema: Any) -> list[str]:
        """Keyword-only parameters for an object input of identifier keys, else one positional ``args``."""
        target, _ref = self.dereference(schema)
        if _is_object_schema(target) and target.get("type", "object") == "object":
            raw_properties = target.get("properties")
            properties: Mapping[str, Any] = raw_properties if isinstance(raw_properties, Mapping) else {}
            names = sorted(properties)
            if all(is_codemode_identifier(name, "python") and name not in ("self", "args", "kwargs") for name in names):
                return self.keyword_parameters(target, properties, names)
        return [f"args: {self.type(schema)}", "/"]

    def keyword_parameters(
        self, schema: Mapping[str, Any], properties: Mapping[str, Any], names: list[str]
    ) -> list[str]:
        raw_required = schema.get("required")
        required = set(raw_required) if isinstance(raw_required, list) else set()
        additional = schema.get("additionalProperties")
        parameters: list[str] = []
        if names:
            parameters.append("*")
        for name in names:
            comments = "".join(f"# {line}\n" for line in self.comment_lines(properties[name]))
            default = "" if name in required else " = ..."
            parameters.append(f"{comments}{name}: {self.type(properties[name])}{default}")
        if additional is True or (additional is None and not names):
            parameters.append("**kwargs: Any")
        elif additional is not None and additional is not False:
            parameters.append(f"**kwargs: {self.type(additional)}")
        return parameters


def _is_object_schema(schema: Any) -> bool:
    if not isinstance(schema, Mapping):
        return False
    kind = schema.get("type")
    if kind == "object":
        return True
    return kind is None and ("properties" in schema or "additionalProperties" in schema or "required" in schema)
