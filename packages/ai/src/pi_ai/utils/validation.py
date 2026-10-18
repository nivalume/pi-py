"""Tool argument validation.

Arguments are coerced and validated against the tool's JSON Schema with a small
compiled validator (schemas compile once into closures and are cached), so no
third-party JSON Schema library is needed. Pydantic model parameters are validated
with ``model_validate`` and returned as model instances.

Supported keywords: ``type``, ``enum``, ``const``, ``properties``, ``required``,
``additionalProperties``, ``patternProperties``, ``propertyNames``,
``minProperties``/``maxProperties``, ``items``/``prefixItems``, ``contains``,
``minItems``/``maxItems``, ``uniqueItems``, ``minLength``/``maxLength``,
``pattern``, ``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum``,
``multipleOf``, ``allOf``/``anyOf``/``oneOf``/``not``, ``if``/``then``/``else``,
``dependentRequired`` and local ``$ref`` (``#``, ``#/$defs/...``, ``#/definitions/...``).
``format`` is treated as an annotation.
"""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from ..types import Tool, ToolCall
from .schema import is_pydantic_model, parameters_schema

__all__ = [
    "ToolValidationError",
    "validate_tool_call",
    "validate_tool_arguments",
    "compile_schema",
    "check_schema",
    "schema_errors",
    "coerce_with_json_schema",
]


class ToolValidationError(ValueError):
    """Raised when tool arguments do not match the tool's schema."""


# A compiled validator: (value, path, errors) -> ok. ``errors`` is None in check-only mode.
_Validator = Callable[[Any, tuple, "list[tuple[tuple, str, str]] | None"], bool]

_CACHE_LIMIT = 512
_validator_cache: OrderedDict[int, tuple[Any, _Validator]] = OrderedDict()


def compile_schema(schema: Any) -> _Validator:
    """Compile (and cache) a validator for ``schema``."""
    key = id(schema)
    cached = _validator_cache.get(key)
    if cached is not None and cached[0] is schema:
        _validator_cache.move_to_end(key)
        return cached[1]
    validator = _Compiler(schema).compile(schema)
    _validator_cache[key] = (schema, validator)
    if len(_validator_cache) > _CACHE_LIMIT:
        _validator_cache.popitem(last=False)
    return validator


def check_schema(schema: Any, value: Any) -> bool:
    return compile_schema(schema)(value, (), None)


def schema_errors(schema: Any, value: Any) -> list[tuple[str, str]]:
    """Validation errors as ``(path, message)`` pairs; empty when valid."""
    errors: list[tuple[tuple, str, str]] = []
    compile_schema(schema)(value, (), errors)
    return [(_format_path(path, keyword_extra), message) for path, keyword_extra, message in errors]


def _format_path(path: tuple, extra: str) -> str:
    parts = [str(part) for part in path]
    if extra:
        parts.append(extra)
    return ".".join(parts) if parts else "root"


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_type(value: Any, json_type: str) -> bool:
    match json_type:
        case "string":
            return isinstance(value, str)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "integer":
            if isinstance(value, bool):
                return False
            if isinstance(value, int):
                return True
            return isinstance(value, float) and value.is_integer()
        case "boolean":
            return isinstance(value, bool)
        case "null":
            return value is None
        case "array":
            return isinstance(value, (list, tuple))
        case "object":
            return isinstance(value, dict)
    return False


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


class _Compiler:
    def __init__(self, root: Any) -> None:
        self.root = root
        self.refs: dict[str, _Validator] = {}

    def compile(self, schema: Any) -> _Validator:
        if schema is True or schema is None:
            return _accept
        if schema is False:
            return _reject
        if not isinstance(schema, dict):
            return _accept

        checks: list[_Validator] = []
        ref = schema.get("$ref")
        if isinstance(ref, str):
            checks.append(self._ref(ref))

        types = schema.get("type")
        if types is not None:
            checks.append(_type_check(types if isinstance(types, list) else [types]))

        if "const" in schema:
            checks.append(_const_check(schema["const"]))
        enum = schema.get("enum")
        if isinstance(enum, list):
            checks.append(_enum_check(enum))

        # strings
        if any(key in schema for key in ("minLength", "maxLength", "pattern")):
            checks.append(_string_check(schema.get("minLength"), schema.get("maxLength"), schema.get("pattern")))
        # numbers
        if any(key in schema for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")):
            checks.append(_number_check(schema))
        # objects
        if any(
            key in schema
            for key in (
                "properties",
                "required",
                "additionalProperties",
                "patternProperties",
                "propertyNames",
                "minProperties",
                "maxProperties",
                "dependentRequired",
            )
        ):
            checks.append(self._object_check(schema))
        # arrays
        if any(key in schema for key in ("items", "prefixItems", "minItems", "maxItems", "uniqueItems", "contains")):
            checks.append(self._array_check(schema))
        # combinators
        all_of = schema.get("allOf")
        if isinstance(all_of, list):
            subs = [self.compile(sub) for sub in all_of]

            def all_of_check(value: Any, path: tuple, errors: Any, subs: list[_Validator] = subs) -> bool:
                ok = True
                for sub in subs:
                    if not sub(value, path, errors):
                        ok = False
                        if errors is None:
                            return False
                return ok

            checks.append(all_of_check)
        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            checks.append(self._any_of(any_of))
        one_of = schema.get("oneOf")
        if isinstance(one_of, list):
            checks.append(self._one_of(one_of))
        not_schema = schema.get("not")
        if not_schema is not None:
            negated = self.compile(not_schema)

            def not_check(value: Any, path: tuple, errors: Any) -> bool:
                if negated(value, path, None):
                    if errors is not None:
                        errors.append((path, "", "must NOT be valid"))
                    return False
                return True

            checks.append(not_check)
        if "if" in schema:
            condition = self.compile(schema["if"])
            then = self.compile(schema["then"]) if "then" in schema else _accept
            otherwise = self.compile(schema["else"]) if "else" in schema else _accept

            def if_check(value: Any, path: tuple, errors: Any) -> bool:
                return (then if condition(value, path, None) else otherwise)(value, path, errors)

            checks.append(if_check)

        if not checks:
            return _accept
        if len(checks) == 1:
            return checks[0]

        def combined(value: Any, path: tuple, errors: Any, checks: list[_Validator] = checks) -> bool:
            ok = True
            for check in checks:
                if not check(value, path, errors):
                    ok = False
                    if errors is None:
                        return False
            return ok

        return combined

    def _ref(self, ref: str) -> _Validator:
        compiled = self.refs.get(ref)
        if compiled is not None:
            return compiled
        holder: list[_Validator] = []

        def deferred(value: Any, path: tuple, errors: Any) -> bool:
            return holder[0](value, path, errors)

        self.refs[ref] = deferred
        target = _resolve_pointer(self.root, ref)
        holder.append(self.compile(target) if target is not None else _accept)
        return deferred

    def _object_check(self, schema: dict[str, Any]) -> _Validator:
        properties: dict[str, Any] = schema.get("properties") or {}
        compiled_props = {name: self.compile(sub) for name, sub in properties.items()}
        required: Sequence[str] = schema.get("required") or ()
        additional = schema.get("additionalProperties", True)
        additional_validator = self.compile(additional) if isinstance(additional, dict) else None
        pattern_props = [
            (re.compile(pattern), self.compile(sub)) for pattern, sub in (schema.get("patternProperties") or {}).items()
        ]
        property_names = self.compile(schema["propertyNames"]) if "propertyNames" in schema else None
        min_props = schema.get("minProperties")
        max_props = schema.get("maxProperties")
        dependent_required: dict[str, list[str]] = schema.get("dependentRequired") or {}

        def object_check(value: Any, path: tuple, errors: Any) -> bool:
            if not isinstance(value, dict):
                return True
            ok = True
            for name in required:
                if name not in value:
                    ok = False
                    if errors is None:
                        return False
                    errors.append((path, name, f"must have required property '{name}'"))
            for name, item in value.items():
                validator = compiled_props.get(name)
                matched = validator is not None
                if validator is not None and not validator(item, (*path, name), errors):
                    ok = False
                    if errors is None:
                        return False
                for pattern, sub in pattern_props:
                    if pattern.search(name):
                        matched = True
                        if not sub(item, (*path, name), errors):
                            ok = False
                            if errors is None:
                                return False
                if property_names is not None and not property_names(name, (*path, name), errors):
                    ok = False
                    if errors is None:
                        return False
                if not matched:
                    if additional is False:
                        ok = False
                        if errors is None:
                            return False
                        errors.append((path, "", f"must NOT have additional properties ('{name}')"))
                    elif additional_validator is not None and not additional_validator(item, (*path, name), errors):
                        ok = False
                        if errors is None:
                            return False
            if min_props is not None and len(value) < min_props:
                ok = False
                if errors is None:
                    return False
                errors.append((path, "", f"must NOT have fewer than {min_props} properties"))
            if max_props is not None and len(value) > max_props:
                ok = False
                if errors is None:
                    return False
                errors.append((path, "", f"must NOT have more than {max_props} properties"))
            for trigger, names in dependent_required.items():
                if trigger in value:
                    for name in names:
                        if name not in value:
                            ok = False
                            if errors is None:
                                return False
                            errors.append((path, name, f"must have property '{name}' when '{trigger}' is present"))
            return ok

        return object_check

    def _array_check(self, schema: dict[str, Any]) -> _Validator:
        items = schema.get("items")
        prefix = schema.get("prefixItems")
        if isinstance(items, list):  # draft-07 tuple form
            prefix, items = items, schema.get("additionalItems", True)
        prefix_validators = [self.compile(sub) for sub in prefix] if isinstance(prefix, list) else []
        rest_validator = self.compile(items) if items is not None and items is not True else None
        rest_forbidden = items is False
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        unique = schema.get("uniqueItems") is True
        contains = self.compile(schema["contains"]) if "contains" in schema else None

        def array_check(value: Any, path: tuple, errors: Any) -> bool:
            if not isinstance(value, (list, tuple)):
                return True
            ok = True
            for index, item in enumerate(value):
                if index < len(prefix_validators):
                    validator: _Validator | None = prefix_validators[index]
                elif rest_forbidden:
                    ok = False
                    if errors is None:
                        return False
                    errors.append((path, "", f"must NOT have more than {len(prefix_validators)} items"))
                    break
                else:
                    validator = rest_validator
                if validator is not None and not validator(item, (*path, index), errors):
                    ok = False
                    if errors is None:
                        return False
            if min_items is not None and len(value) < min_items:
                ok = False
                if errors is None:
                    return False
                errors.append((path, "", f"must NOT have fewer than {min_items} items"))
            if max_items is not None and len(value) > max_items:
                ok = False
                if errors is None:
                    return False
                errors.append((path, "", f"must NOT have more than {max_items} items"))
            if unique:
                seen: list[str] = []
                for item in value:
                    key = json.dumps(item, sort_keys=True, default=str)
                    if key in seen:
                        ok = False
                        if errors is None:
                            return False
                        errors.append((path, "", "must NOT have duplicate items"))
                        break
                    seen.append(key)
            if contains is not None and not any(contains(item, path, None) for item in value):
                ok = False
                if errors is None:
                    return False
                errors.append((path, "", "must contain at least 1 valid item"))
            return ok

        return array_check

    def _any_of(self, schemas: list[Any]) -> _Validator:
        subs = [self.compile(sub) for sub in schemas]

        def any_of_check(value: Any, path: tuple, errors: Any) -> bool:
            for sub in subs:
                if sub(value, path, None):
                    return True
            if errors is not None:
                errors.append((path, "", "must match a schema in anyOf"))
            return False

        return any_of_check

    def _one_of(self, schemas: list[Any]) -> _Validator:
        subs = [self.compile(sub) for sub in schemas]

        def one_of_check(value: Any, path: tuple, errors: Any) -> bool:
            matches = sum(1 for sub in subs if sub(value, path, None))
            if matches == 1:
                return True
            if errors is not None:
                errors.append((path, "", "must match exactly one schema in oneOf"))
            return False

        return one_of_check


def _accept(value: Any, path: tuple, errors: Any) -> bool:
    return True


def _reject(value: Any, path: tuple, errors: Any) -> bool:
    if errors is not None:
        errors.append((path, "", "boolean schema is false"))
    return False


def _type_check(types: list[str]) -> _Validator:
    if len(types) == 1:
        json_type = types[0]

        def single_type_check(value: Any, path: tuple, errors: Any) -> bool:
            if _is_type(value, json_type):
                return True
            if errors is not None:
                errors.append((path, "", f"must be {json_type}"))
            return False

        return single_type_check

    label = ",".join(types)

    def type_check(value: Any, path: tuple, errors: Any) -> bool:
        for json_type in types:
            if _is_type(value, json_type):
                return True
        if errors is not None:
            errors.append((path, "", f"must be {label}"))
        return False

    return type_check


def _const_check(expected: Any) -> _Validator:
    def const_check(value: Any, path: tuple, errors: Any) -> bool:
        if _json_equal(value, expected):
            return True
        if errors is not None:
            errors.append((path, "", f"must be equal to constant {json.dumps(expected, default=str)}"))
        return False

    return const_check


def _enum_check(options: list[Any]) -> _Validator:
    def enum_check(value: Any, path: tuple, errors: Any) -> bool:
        for option in options:
            if _json_equal(value, option):
                return True
        if errors is not None:
            allowed = ", ".join(json.dumps(option, default=str) for option in options)
            errors.append((path, "", f"must be equal to one of the allowed values: {allowed}"))
        return False

    return enum_check


def _string_check(min_length: int | None, max_length: int | None, pattern: str | None) -> _Validator:
    compiled = re.compile(pattern) if pattern else None

    def string_check(value: Any, path: tuple, errors: Any) -> bool:
        if not isinstance(value, str):
            return True
        ok = True
        if min_length is not None and len(value) < min_length:
            ok = False
            if errors is None:
                return False
            errors.append((path, "", f"must NOT have fewer than {min_length} characters"))
        if max_length is not None and len(value) > max_length:
            ok = False
            if errors is None:
                return False
            errors.append((path, "", f"must NOT have more than {max_length} characters"))
        if compiled is not None and compiled.search(value) is None:
            ok = False
            if errors is None:
                return False
            errors.append((path, "", f'must match pattern "{pattern}"'))
        return ok

    return string_check


def _number_check(schema: dict[str, Any]) -> _Validator:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_minimum = schema.get("exclusiveMinimum")
    exclusive_maximum = schema.get("exclusiveMaximum")
    multiple_of = schema.get("multipleOf")
    # draft-04 boolean form
    if exclusive_minimum is True:
        exclusive_minimum, minimum = minimum, None
    elif exclusive_minimum is False:
        exclusive_minimum = None
    if exclusive_maximum is True:
        exclusive_maximum, maximum = maximum, None
    elif exclusive_maximum is False:
        exclusive_maximum = None

    def number_check(value: Any, path: tuple, errors: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return True
        failures: list[str] = []
        if minimum is not None and value < minimum:
            failures.append(f"must be >= {minimum}")
        if maximum is not None and value > maximum:
            failures.append(f"must be <= {maximum}")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            failures.append(f"must be > {exclusive_minimum}")
        if exclusive_maximum is not None and value >= exclusive_maximum:
            failures.append(f"must be < {exclusive_maximum}")
        if multiple_of:
            quotient = value / multiple_of
            if not math.isclose(quotient, round(quotient), rel_tol=0, abs_tol=1e-9):
                failures.append(f"must be multiple of {multiple_of}")
        if not failures:
            return True
        if errors is not None:
            errors.extend((path, "", failure) for failure in failures)
        return False

    return number_check


def _resolve_pointer(root: Any, ref: str) -> Any:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return None
    node = root
    for raw in ref[2:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


# ---------------------------------------------------------------------------
# Coercion (mirrors pi's coerceWithJsonSchema / TypeBox Value.Convert)
# ---------------------------------------------------------------------------


def _schema_types(schema: dict[str, Any]) -> list[str]:
    types = schema.get("type")
    if isinstance(types, str):
        return [types]
    if isinstance(types, list):
        return [t for t in types if isinstance(t, str)]
    return []


def _coerce_primitive(value: Any, json_type: str) -> Any:
    match json_type:
        case "number":
            if value is None:
                return 0
            if isinstance(value, str) and value.strip():
                try:
                    parsed = float(value)
                except ValueError:
                    return value
                if math.isfinite(parsed):
                    return int(parsed) if parsed.is_integer() and _looks_integral(value) else parsed
            if isinstance(value, bool):
                return 1 if value else 0
            return value
        case "integer":
            if value is None:
                return 0
            if isinstance(value, str) and value.strip():
                try:
                    parsed = float(value)
                except ValueError:
                    return value
                if math.isfinite(parsed) and parsed.is_integer():
                    return int(parsed)
            if isinstance(value, bool):
                return 1 if value else 0
            return value
        case "boolean":
            if value is None:
                return False
            if value == "true":
                return True
            if value == "false":
                return False
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                if value == 1:
                    return True
                if value == 0:
                    return False
            return value
        case "string":
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return _number_to_string(value)
            return value
        case "null":
            if value == "" or (not isinstance(value, bool) and value == 0) or value is False:
                return None
            return value
    return value


def _looks_integral(text: str) -> bool:
    stripped = text.strip()
    return "." not in stripped and "e" not in stripped.lower()


def _number_to_string(value: float) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    return json.dumps(value)


def _coerce_union(value: Any, schemas: list[Any]) -> Any:
    for schema in schemas:
        if check_schema(schema, value):
            return value
    for schema in schemas:
        candidate = coerce_with_json_schema(_json_copy(value), schema)
        if check_schema(schema, candidate):
            return candidate
    return value


def coerce_with_json_schema(value: Any, schema: Any) -> Any:
    """Coerce primitive values toward the types a JSON Schema expects (in place for containers)."""
    if not isinstance(schema, dict):
        return value
    next_value = value
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            next_value = coerce_with_json_schema(next_value, nested)
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        next_value = _coerce_union(next_value, any_of)
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        next_value = _coerce_union(next_value, one_of)

    types = _schema_types(schema)
    matches_union_member = len(types) > 1 and any(_is_type(next_value, t) for t in types)
    if types and not matches_union_member and not (len(types) == 1 and _is_type(next_value, types[0])):
        for json_type in types:
            candidate = _coerce_primitive(next_value, json_type)
            if type(candidate) is not type(next_value) or candidate != next_value:
                next_value = candidate
                break

    if "object" in types and isinstance(next_value, dict):
        properties = schema.get("properties")
        defined = set(properties) if isinstance(properties, dict) else set()
        if isinstance(properties, dict):
            for key, sub in properties.items():
                if key in next_value:
                    next_value[key] = coerce_with_json_schema(next_value[key], sub)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key in list(next_value):
                if key not in defined:
                    next_value[key] = coerce_with_json_schema(next_value[key], additional)
    if "array" in types and isinstance(next_value, list):
        items = schema.get("items")
        prefix = schema.get("prefixItems")
        if isinstance(items, list):
            prefix = items
        if isinstance(prefix, list):
            for index in range(min(len(prefix), len(next_value))):
                next_value[index] = coerce_with_json_schema(next_value[index], prefix[index])
        elif isinstance(items, dict):
            for index in range(len(next_value)):
                next_value[index] = coerce_with_json_schema(next_value[index], items)
    return next_value


def _normalize_optional_nulls(value: Any, schema: Any) -> None:
    """Drop ``null`` values for optional properties whose schema does not accept null."""
    if not isinstance(schema, dict):
        return
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, list):
            for item, sub in zip(value, items, strict=False):
                _normalize_optional_nulls(item, sub)
        elif isinstance(items, dict):
            for item in value:
                _normalize_optional_nulls(item, items)
        return
    properties = schema.get("properties")
    if not isinstance(value, dict) or not isinstance(properties, dict):
        return
    required = set(schema.get("required") or ())
    for key, sub in properties.items():
        if key not in value:
            continue
        if (
            value[key] is None
            and key not in required
            and not (isinstance(sub, dict) and isinstance(sub.get("$ref"), str))
            and not check_schema(sub, None)
        ):
            del value[key]
        else:
            _normalize_optional_nulls(value[key], sub)


def _json_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Public tool validation
# ---------------------------------------------------------------------------


def validate_tool_call(tools: Sequence[Tool], tool_call: ToolCall) -> Any:
    """Find the tool by name and validate the call's arguments against its schema.

    Raises :class:`ToolValidationError` when the tool is unknown or validation fails.
    """
    for tool in tools:
        if tool.name == tool_call.name:
            return validate_tool_arguments(tool, tool_call)
    raise ToolValidationError(f'Tool "{tool_call.name}" not found')


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    """Validate (and coerce) tool call arguments against the tool's parameter schema.

    Returns the validated arguments: a ``dict`` for JSON Schema parameters, or a model
    instance for Pydantic parameters. Raises :class:`ToolValidationError` with a
    readable message listing every problem.
    """
    if tool_call.arguments_error:
        raise ToolValidationError(f'Invalid arguments for tool "{tool_call.name}": {tool_call.arguments_error}')

    parameters = tool.parameters
    if is_pydantic_model(parameters):
        return _validate_pydantic(parameters, tool_call)

    schema = parameters_schema(parameters)
    args = _json_copy(tool_call.arguments)
    _normalize_optional_nulls(args, schema)
    coerced = coerce_with_json_schema(args, schema)
    validator = compile_schema(schema)
    if validator(coerced, (), None):
        return coerced
    if coerced is not args and validator(args, (), None):
        return args

    raw_errors: list[tuple[tuple, str, str]] = []
    validator(coerced, (), raw_errors)
    lines = [f"  - {_format_path(path, extra)}: {message}" for path, extra, message in raw_errors]
    details = "\n".join(lines) or "Unknown validation error"
    received = json.dumps(tool_call.arguments, indent=2, ensure_ascii=False, default=str)
    raise ToolValidationError(
        f'Validation failed for tool "{tool_call.name}":\n{details}\n\nReceived arguments:\n{received}'
    )


def _validate_pydantic(model_class: Any, tool_call: ToolCall) -> Any:
    try:
        return model_class.model_validate(tool_call.arguments)
    except Exception as error:  # pydantic.ValidationError
        errors_fn = getattr(error, "errors", None)
        if not callable(errors_fn):
            raise
        lines = []
        items: Any = errors_fn()
        for item in items:
            location = ".".join(str(part) for part in item.get("loc", ())) or "root"
            lines.append(f"  - {location}: {item.get('msg', 'invalid')}")
        received = json.dumps(tool_call.arguments, indent=2, ensure_ascii=False, default=str)
        raise ToolValidationError(
            f'Validation failed for tool "{tool_call.name}":\n'
            + ("\n".join(lines) or "Unknown validation error")
            + f"\n\nReceived arguments:\n{received}"
        ) from error
