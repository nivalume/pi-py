"""TypeScript declarations for JavaScript scripts (``pi_codemode.typescript_declarations``), and the dispatcher.

Ports upstream ``test/declarations.test.ts`` (by way of delta's ``tests/declarations.rs``). Every expected string is
upstream's output: the parity cases at the end were recorded from upstream's ``declarations.js`` under Node and are
embedded here, so the suite needs neither Node nor the network. ``test_declarations.py`` covers the Python stubs.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from pi_codemode import (
    CodemodeTool,
    CodemodeToolDeclaration,
    is_codemode_identifier,
    python_declarations,
    to_codemode_identifier,
)
from pi_codemode import declarations as dispatcher
from pi_codemode.typescript_declarations import (
    DEFAULT_INPUT_SCHEMA_MAX_CHARS,
    render_declarations,
    render_tool_output_type,
    render_tool_sample,
    render_tool_signature,
    schema_to_type,
)


def ty(schema: Any) -> str:
    return schema_to_type(schema)


def lines(*parts: str) -> str:
    return "\n".join(parts)


def declaration(case: dict[str, Any]) -> CodemodeToolDeclaration:
    """A tool from an upstream-shaped case (``inputSchema``, ``outputSchema``)."""
    return CodemodeToolDeclaration(
        case["name"], case.get("description"), case.get("inputSchema"), case.get("outputSchema")
    )


class TestSchemaToType:
    def test_primitives_literals_and_unions(self) -> None:
        assert ty({"type": "string"}) == "string"
        assert ty({"type": "integer"}) == "number"
        assert ty({"type": "number"}) == "number"
        assert ty({"type": "boolean"}) == "boolean"
        assert ty({"type": "null"}) == "null"
        assert ty({"type": ["string", "null"]}) == "string | null"
        assert ty({"type": ["integer", "null"]}) == "number | null"
        assert ty({"const": "a"}) == '"a"'
        assert ty({"enum": ["a", 1, None]}) == '"a" | 1 | null'
        assert ty({"anyOf": [{"type": "string"}, {"type": "number"}]}) == "string | number"
        assert ty({"oneOf": [{"type": "string"}, {"type": "string"}]}) == "string"
        assert ty({"anyOf": [{"type": "string"}, {}]}) == "unknown"
        assert ty({"anyOf": []}) == "never"
        assert ty({"allOf": [{"anyOf": [{"type": "string"}, {"type": "number"}]}, {"const": 1}]}) == (
            "(string | number) & 1"
        )
        assert ty({"allOf": [{}]}) == "unknown"
        assert ty({"$ref": "#/defs/x"}) == "unknown"
        assert ty({"type": "strange"}) == "unknown"
        assert ty({"type": None}) == "unknown"
        assert ty(True) == "unknown"
        assert ty(False) == "never"

    def test_objects_on_one_line_with_sorted_properties(self) -> None:
        assert (
            ty(
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "max-lines": {"type": "number"}},
                    "required": ["city"],
                    "additionalProperties": False,
                }
            )
            == '{ city: string; "max-lines"?: number; }'
        )
        assert ty({"type": "object", "additionalProperties": {"type": "number"}}) == "{ [key: string]: number; }"
        assert ty({"type": "object"}) == "{ [key: string]: unknown; }"
        assert ty({"type": "object", "properties": {}, "additionalProperties": False}) == "{}"
        assert ty({"required": ["a"]}) == "{ [key: string]: unknown; }"
        assert (
            ty({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True})
            == "{ a?: string; [key: string]: unknown; }"
        )
        # JavaScript's sort order: by UTF-16 code units, so upper case before lower case before non-ASCII.
        assert (
            ty({"type": "object", "properties": {"ä": {"type": "string"}, "Z": {"type": "string"}, "a": {}}})
            == '{ Z?: string; a?: unknown; "ä"?: string; }'
        )

    def test_property_descriptions_on_comment_lines(self) -> None:
        assert ty(
            {
                "type": "object",
                "properties": {
                    "weather": {
                        "type": "array",
                        "description": "look up weather for a given list of locations",
                        "items": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                },
                "required": ["weather"],
            }
        ) == lines(
            "{",
            "  // look up weather for a given list of locations",
            "  weather: Array<{ location: string; }>;",
            "}",
        )
        assert ty(
            {
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "description": "Outer",
                        "properties": {"inner": {"type": "string", "description": "Inner"}},
                    }
                },
            }
        ) == lines("{", "  // Outer", "  outer?: {", "    // Inner", "    inner?: string;", "  };", "}")
        assert ty(
            {
                "type": "object",
                "properties": {"a": {"type": "string", "description": "A"}},
                "additionalProperties": {"type": "number"},
            }
        ) == lines("{", "  // A", "  a?: string;", "  [key: string]: number;", "}")

    def test_constraints_on_property_comment_lines(self) -> None:
        assert ty(
            {
                "type": "object",
                "properties": {
                    "timeoutMs": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 30000,
                        "default": 0,
                        "description": "Wait in milliseconds.",
                    },
                    "id": {"type": "string", "pattern": "^[a-z]+$", "maxLength": 8},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                        "uniqueItems": True,
                        "deprecated": True,
                    },
                    "name": {"type": "string"},
                },
                "required": ["id"],
            }
        ) == lines(
            "{",
            "  // @maxLength 8 @pattern ^[a-z]+$",
            "  id: string;",
            "  name?: string;",
            "  // @maxItems 5 @uniqueItems @deprecated",
            "  tags?: Array<string>;",
            "  // Wait in milliseconds.",
            "  // @integer @minimum 0 @maximum 30000 @default 0",
            "  timeoutMs?: number;",
            "}",
        )

    def test_constraints_of_optional_fields_from_the_non_null_branch(self) -> None:
        assert ty(
            {
                "type": "object",
                "properties": {
                    "limit": {
                        "anyOf": [{"type": "integer", "minimum": 1, "maximum": 100}, {"type": "null"}],
                        "default": None,
                        "title": "Limit",
                    }
                },
            }
        ) == lines("{", "  // @integer @minimum 1 @maximum 100", "  limit?: number | null;", "}")
        assert ty(
            {
                "type": "object",
                "properties": {
                    "n": {
                        "anyOf": [
                            {"type": "integer", "minimum": 10, "maximum": 50, "format": "int32"},
                            {"type": "null"},
                        ],
                        "minimum": 0,
                        "maximum": 20,
                        "format": "port",
                    }
                },
            }
        ) == lines("{", "  // @integer @minimum 10 @maximum 20 @format port @format int32", "  n?: number | null;", "}")

    def test_draft_04_boolean_exclusive_bounds(self) -> None:
        assert ty(
            {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "minimum": 0,
                        "exclusiveMinimum": True,
                        "maximum": 10,
                        "exclusiveMaximum": False,
                    },
                    "m": {
                        "anyOf": [{"type": "number", "maximum": 5, "exclusiveMaximum": True}, {"type": "null"}],
                        "maximum": 8,
                    },
                },
            }
        ) == lines(
            "{",
            "  // @maximum 8 @exclusiveMaximum 5",
            "  m?: number | null;",
            "  // @integer @exclusiveMinimum 0 @maximum 10",
            "  n?: number;",
            "}",
        )

    def test_long_tags_left_out_and_line_terminators_escaped(self) -> None:
        assert (
            ty({"type": "object", "properties": {"config": {"type": "object", "default": {"text": "x" * 100}}}})
            == "{ config?: { [key: string]: unknown; }; }"
        )
        assert ty(
            {
                "type": "object",
                "properties": {
                    "sep": {
                        "type": "string",
                        "pattern": "^[\r\n\u2028\u2029]$",
                        "default": "\u2028",
                        "description": "a\u2028b",
                    }
                },
            }
        ) == lines(
            "{",
            "  // a",
            "  // b",
            '  // @pattern ^[\\u000d\\u000a\\u2028\\u2029]$ @default "\\u2028"',
            "  sep?: string;",
            "}",
        )

    def test_numbers_print_as_javascript_prints_them(self) -> None:
        assert ty({"const": 1.0}) == "1"
        assert ty({"const": 1e21}) == "1e+21"
        assert ty({"const": 1e-7}) == "1e-7"
        assert ty({"const": 0.000001}) == "0.000001"
        assert ty({"enum": [1.5, 2, 1e300, -0.5, 123456789012345680000]}) == (
            "1.5 | 2 | 1e+300 | -0.5 | 123456789012345680000"
        )
        assert ty({"const": {"k": [1, 2.0, 0.000001, 1e21, 1e-7]}}) == '{"k":[1,2,0.000001,1e+21,1e-7]}'
        assert ty(
            {
                "type": "object",
                "properties": {
                    "n": {"type": "number", "minimum": 1.0, "maximum": 1e21, "multipleOf": 1e-7, "default": 0.000001}
                },
            }
        ) == lines("{", "  // @minimum 1 @maximum 1e+21 @multipleOf 1e-7 @default 0.000001", "  n?: number;", "}")
        # JSON.stringify keeps non-ASCII and leaves out no spaces.
        assert ty({"const": {"x": "é", "y": [None, True]}}) == '{"x":"é","y":[null,true]}'

    def test_constraint_tags_dropped_before_unknown(self) -> None:
        schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}
        assert ty(schema) == lines("{", "  // @integer @minimum 1", "  n?: number;", "}")
        assert schema_to_type(schema, max_chars=20) == "{ n?: number; }"
        assert schema_to_type(schema, max_chars=10) == "unknown"

    def test_local_references_resolve_and_recursion_stops(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
                "legacy": {"$ref": "#/definitions/Legacy"},
                "remote": {"$ref": "https://example.com/schema.json"},
            },
            "required": ["item"],
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "parent": {"$ref": "#/$defs/Item"}},
                    "required": ["id"],
                }
            },
            "definitions": {"Legacy": {"enum": ["a", "b"]}},
        }
        assert ty(schema) == '{ item: { id: string; parent?: unknown; }; legacy?: "a" | "b"; remote?: unknown; }'
        assert ty({"$ref": "#"}) == "unknown"
        assert ty({"$ref": "#/$defs/missing", "$defs": {}}) == "unknown"

    def test_reference_pointers_are_unescaped(self) -> None:
        assert (
            ty(
                {
                    "$defs": {"a/b": {"type": "string"}, "c~d": {"type": "number"}, "e f": {"type": "boolean"}},
                    "type": "object",
                    "properties": {
                        "x": {"$ref": "#/$defs/a~1b"},
                        "y": {"$ref": "#/$defs/c~0d"},
                        "z": {"$ref": "#/$defs/e%20f"},
                    },
                }
            )
            == "{ x?: string; y?: number; z?: boolean; }"
        )

    def test_reference_expansions_are_limited_per_schema(self) -> None:
        properties = {f"p{index:02d}": {"$ref": "#/$defs/S"} for index in range(40)}
        rendered = ty({"type": "object", "properties": properties, "$defs": {"S": {"type": "string"}}})
        assert (
            rendered
            == "{ " + " ".join(f"p{index:02d}?: {'string' if index < 32 else 'unknown'};" for index in range(40)) + " }"
        )

    def test_arrays_tuples_and_budgets(self) -> None:
        assert ty({"type": "array", "items": {"type": "string"}}) == "Array<string>"
        assert ty({"items": {"type": "string"}}) == "Array<string>"
        assert ty({"type": "array", "prefixItems": [{"type": "string"}, {"type": "number"}]}) == "[string, number]"
        assert ty({"type": "array", "items": [{"type": "string"}]}) == "[string]"
        assert ty({"type": "array"}) == "unknown[]"
        assert ty(
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"a": {"type": "string", "description": "Deep"}}},
                },
            }
        ) == lines("Array<Array<{", "  // Deep", "  a?: string;", "}>>")
        schema = {"type": "object", "properties": {f"field{index}": {"type": "string"} for index in range(50)}}
        assert schema_to_type(schema, max_chars=100) == "unknown"
        assert "field49?: string;" in ty(schema)


class TestRendering:
    def test_tool_signatures(self) -> None:
        tool = CodemodeToolDeclaration(
            "hidden-dynamic-tool",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        )
        assert render_tool_signature(tool) == "hidden_dynamic_tool(args: { city: string; }): Promise<{ ok: boolean; }>;"
        assert render_tool_signature(CodemodeToolDeclaration("free")) == "free(args: unknown): Promise<unknown>;"
        plain = CodemodeToolDeclaration(
            "plain",
            input_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "array", "items": {"type": "object"}},
                    "isError": {"type": "boolean"},
                    "_meta": {"type": "object"},
                },
                "required": ["content"],
            },
        )
        assert render_tool_signature(plain) == (
            "plain(args: { [key: string]: unknown; }): Promise<{ _meta?: { [key: string]: unknown; }; "
            "content: Array<{ [key: string]: unknown; }>; isError?: boolean; }>;"
        )
        assert render_tool_output_type(None) == "unknown"
        assert render_tool_output_type({"type": "string"}) == "string"

    def test_the_input_budget_drops_tags_then_renders_unknown(self) -> None:
        tool = CodemodeToolDeclaration(
            "t", input_schema={"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}
        )
        assert render_tool_signature(tool) == lines(
            "t(args: {", "  // @integer @minimum 1", "  n?: number;", "}): Promise<unknown>;"
        )
        assert render_tool_signature(tool, input_max_chars=20) == "t(args: { n?: number; }): Promise<unknown>;"
        assert render_tool_signature(tool, input_max_chars=10) == "t(args: unknown): Promise<unknown>;"
        assert DEFAULT_INPUT_SCHEMA_MAX_CHARS == 16_000
        huge = CodemodeToolDeclaration(
            "huge",
            input_schema={
                "type": "object",
                "properties": {f"f{index:05d}": {"type": "string"} for index in range(2000)},
            },
        )
        assert render_tool_signature(huge) == "huge(args: unknown): Promise<unknown>;"

    def test_tool_samples(self) -> None:
        foo = CodemodeToolDeclaration("foo", "bar", {"type": "string"})
        assert render_tool_sample(foo) == lines(
            "bar",
            "",
            "codemode tool declaration:",
            "```ts",
            "declare const tools: { foo(args: string): Promise<unknown>; };",
            "```",
        )
        assert render_tool_sample(CodemodeToolDeclaration("bare", "  padded  \n")).startswith("padded\n\n")

    def test_declarations_of_tools_globals_and_namespaces(self) -> None:
        tools = [
            CodemodeToolDeclaration(
                "read",
                "Read a file.\nSecond line.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                {"type": "string"},
            ),
            CodemodeToolDeclaration("remote-api"),
        ]
        attach = CodemodeTool(
            "attach", lambda _args, _context: None, description="Attach it.", input_schema={"type": "string"}
        )
        assert render_declarations(tools=tools, globals=[attach]) == lines(
            "declare const tools: {",
            "  /**",
            "   * Read a file.",
            "   * Second line.",
            "   */",
            "  read(args: { path: string; }): Promise<string>;",
            "  remote_api(args: unknown): Promise<unknown>;",
            "};",
            "",
            "/** Attach it. */",
            "declare function attach(args: string): Promise<unknown>;",
        )
        globals = [
            CodemodeToolDeclaration("models.list", "List models.", signature="(type: string): Promise<string[]>"),
            CodemodeToolDeclaration("models.get", input_schema={"type": "string"}),
            CodemodeToolDeclaration("plain", signature="(): void"),
        ]
        assert render_declarations(globals=globals) == lines(
            "declare function plain(): void;",
            "",
            "declare const models: {",
            "  /** List models. */",
            "  list(type: string): Promise<string[]>;",
            "  get(args: string): Promise<unknown>;",
            "};",
        )
        assert render_declarations() == ""

    def test_doc_comments_cannot_end_early(self) -> None:
        assert "/** a *\\/ b */" in render_declarations(tools=[CodemodeToolDeclaration("x", "a */ b")])
        assert render_declarations(tools=[CodemodeToolDeclaration("x", "one\r\n\r\nthree")]) == lines(
            "declare const tools: {",
            "  /**",
            "   * one",
            "   *",
            "   * three",
            "   */",
            "  x(args: unknown): Promise<unknown>;",
            "};",
        )
        assert render_declarations(tools=[CodemodeToolDeclaration("x", "   ")]) == lines(
            "declare const tools: {", "  x(args: unknown): Promise<unknown>;", "};"
        )


class TestDispatcher:
    def test_javascript_is_the_default(self) -> None:
        tool = CodemodeToolDeclaration(
            "read", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}}
        )
        assert dispatcher.render_declarations(tools=[tool]) == render_declarations(tools=[tool])
        assert dispatcher.render_tool_signature(tool) == render_tool_signature(tool)
        assert dispatcher.render_tool_sample(tool, input_max_chars=5) == render_tool_sample(tool, input_max_chars=5)
        assert dispatcher.render_tool_output_type({"type": "integer"}) == "number"
        assert dispatcher.schema_to_type({"type": "integer"}) == "number"
        assert (
            dispatcher.schema_to_type(
                {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}, max_chars=20
            )
            == "{ n?: number; }"
        )
        assert dispatcher.DEFAULT_INPUT_SCHEMA_MAX_CHARS == DEFAULT_INPUT_SCHEMA_MAX_CHARS

    def test_python_delegates_to_the_python_stubs(self) -> None:
        tool = CodemodeToolDeclaration("my-tool", "Do it", {"type": "object", "properties": {"n": {"type": "integer"}}})
        attach = CodemodeToolDeclaration("attach", input_schema={"type": "string"})
        assert dispatcher.render_declarations(tools=[tool], globals=[attach], language="python") == (
            python_declarations.render_declarations(tools=[tool], globals=[attach])
        )
        assert dispatcher.render_declarations(tools=[tool], language="python").startswith("class tools:\n")
        assert dispatcher.render_tool_signature(tool, language="python") == python_declarations.render_tool_signature(
            tool
        )
        assert dispatcher.render_tool_sample(tool, language="python") == python_declarations.render_tool_sample(tool)
        assert dispatcher.render_tool_output_type(None, language="python") == "Any"
        assert dispatcher.schema_to_type({"type": "integer"}, language="python") == "int"

    def test_unknown_languages_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown code mode language"):
            dispatcher.schema_to_type({"type": "string"}, language="ruby")  # type: ignore[arg-type]


class TestIdentifiers:
    def test_javascript_identifiers(self) -> None:
        assert to_codemode_identifier("mcp__docs__search") == "mcp__docs__search"
        assert to_codemode_identifier("my-tool") == "my_tool"
        assert to_codemode_identifier("2fa") == "_fa"
        assert to_codemode_identifier("$el") == "$el"
        assert to_codemode_identifier("a.b c") == "a_b_c"
        assert to_codemode_identifier("class") == "class"
        assert to_codemode_identifier("") == "_"
        assert to_codemode_identifier("é") == "_"
        assert is_codemode_identifier("$x1")
        assert is_codemode_identifier("class")
        assert not is_codemode_identifier("my-tool")
        assert not is_codemode_identifier("")

    def test_python_identifiers(self) -> None:
        assert to_codemode_identifier("my-tool", "python") == "my_tool"
        assert to_codemode_identifier("2fa", "python") == "_fa"
        assert to_codemode_identifier("$el", "python") == "_el"
        assert to_codemode_identifier("class", "python") == "class_"
        assert to_codemode_identifier("", "python") == "_"
        assert is_codemode_identifier("x1", "python")
        assert not is_codemode_identifier("$x1", "python")
        assert not is_codemode_identifier("class", "python")

    def test_unknown_languages_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown code mode language"):
            to_codemode_identifier("x", "ruby")  # type: ignore[arg-type]


#: Recorded from upstream pi-codemode 1.3.0's declarations.js under Node: a tool (upstream-shaped), its signature, its
#: sample, and its input rendered with ``maxChars: 60`` (``None`` without an input).
PARITY_CASES: Final[list[tuple[dict[str, Any], str, str, str | None]]] = [
    (
        {
            "name": "read",
            "description": "Read a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace"},
                    "limit": {"type": "integer", "minimum": 1, "default": 2000},
                },
                "required": ["path"],
            },
            "outputSchema": {"type": "string"},
        },
        "read(args: {\n"
        "  // @integer @minimum 1 @default 2000\n"
        "  limit?: number;\n"
        "  // Path relative to the workspace\n"
        "  path: string;\n"
        "}): Promise<string>;",
        "Read a file\n"
        "\n"
        "codemode tool declaration:\n"
        "```ts\n"
        "declare const tools: { read(args: {\n"
        "  // @integer @minimum 1 @default 2000\n"
        "  limit?: number;\n"
        "  // Path relative to the workspace\n"
        "  path: string;\n"
        "}): Promise<string>; };\n"
        "```",
        "unknown",
    ),
    (
        {
            "name": "my-tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a b": {"type": ["string", "null"]},
                    "x": {"enum": ["a", 1, 1.5, None, True]},
                    "y": {"const": {"k": [1, 2.0, 1e-06, 1e21, 1e-07]}},
                },
            },
        },
        'my_tool(args: { "a b"?: string | null; x?: "a" | 1 | 1.5 | null | true; y?: '
        '{"k":[1,2,0.000001,1e+21,1e-7]}; }): Promise<unknown>;',
        "\n"
        "\n"
        "codemode tool declaration:\n"
        "```ts\n"
        'declare const tools: { my_tool(args: { "a b"?: string | null; x?: "a" | 1 | 1.5 | null | true; y?: '
        '{"k":[1,2,0.000001,1e+21,1e-7]}; }): Promise<unknown>; };\n'
        "```",
        "unknown",
    ),
    (
        {
            "name": "refs",
            "description": "Line one\nline two */ end",
            "inputSchema": {
                "$defs": {
                    "N": {
                        "type": "object",
                        "properties": {
                            "next": {"$ref": "#/$defs/N"},
                            "v": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
                        },
                    }
                },
                "$ref": "#/$defs/N",
            },
        },
        "refs(args: {\n  next?: unknown;\n  // @exclusiveMinimum 0 @maximum 10\n  v?: number;\n}): Promise<unknown>;",
        "Line one\n"
        "line two */ end\n"
        "\n"
        "codemode tool declaration:\n"
        "```ts\n"
        "declare const tools: { refs(args: {\n"
        "  next?: unknown;\n"
        "  // @exclusiveMinimum 0 @maximum 10\n"
        "  v?: number;\n"
        "}): Promise<unknown>; };\n"
        "```",
        "{ next?: unknown; v?: number; }",
    ),
    (
        {
            "name": "arrs",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "t": {"prefixItems": [{"type": "string"}, {"type": "number"}]},
                    "u": {"type": "array"},
                    "v": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 3}]},
                        "uniqueItems": True,
                    },
                    "w": {
                        "allOf": [
                            {"type": "object", "properties": {"a": {"type": "string"}}},
                            {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        ]
                    },
                    "z": {"type": "object", "additionalProperties": {"type": "boolean"}},
                    "e": {"type": "object", "additionalProperties": False},
                    "o": {
                        "anyOf": [{"type": "integer", "minimum": 5, "maximum": 9}, {"type": "null"}],
                        "default": None,
                        "deprecated": True,
                    },
                    "p": {"type": "string", "pattern": "^a\nb$", "format": "email", "minLength": 2, "maxLength": 5},
                },
                "required": ["t", "u"],
            },
        },
        "arrs(args: {\n"
        "  e?: {};\n"
        "  // @integer @minimum 5 @maximum 9 @deprecated\n"
        "  o?: number | null;\n"
        "  // @minLength 2 @maxLength 5 @pattern ^a\\u000ab$ @format email\n"
        "  p?: string;\n"
        "  t: [string, number];\n"
        "  u: unknown[];\n"
        "  // @uniqueItems\n"
        "  v?: Array<string | number>;\n"
        "  w?: { a?: string; } & (string | number);\n"
        "  z?: { [key: string]: boolean; };\n"
        "}): Promise<unknown>;",
        "\n"
        "\n"
        "codemode tool declaration:\n"
        "```ts\n"
        "declare const tools: { arrs(args: {\n"
        "  e?: {};\n"
        "  // @integer @minimum 5 @maximum 9 @deprecated\n"
        "  o?: number | null;\n"
        "  // @minLength 2 @maxLength 5 @pattern ^a\\u000ab$ @format email\n"
        "  p?: string;\n"
        "  t: [string, number];\n"
        "  u: unknown[];\n"
        "  // @uniqueItems\n"
        "  v?: Array<string | number>;\n"
        "  w?: { a?: string; } & (string | number);\n"
        "  z?: { [key: string]: boolean; };\n"
        "}): Promise<unknown>; };\n"
        "```",
        "unknown",
    ),
    (
        {
            "name": "plain",
            "inputSchema": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
                "required": ["a"],
            },
            "outputSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        },
        "plain(args: { a: string; b?: number; }): Promise<{ ok?: boolean; }>;",
        "\n"
        "\n"
        "codemode tool declaration:\n"
        "```ts\n"
        "declare const tools: { plain(args: { a: string; b?: number; }): Promise<{ ok?: boolean; }>; };\n"
        "```",
        "{ a: string; b?: number; }",
    ),
    (
        {"name": "none"},
        "none(args: unknown): Promise<unknown>;",
        "\n\ncodemode tool declaration:\n```ts\ndeclare const tools: { none(args: unknown): Promise<unknown>; };\n```",
        None,
    ),
    (
        {
            "name": "draft4",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "number",
                        "minimum": 1,
                        "exclusiveMinimum": True,
                        "maximum": 2,
                        "exclusiveMaximum": True,
                        "multipleOf": 0.5,
                        "default": {"x": "é"},
                    }
                },
            },
        },
        "draft4(args: {\n"
        '  // @exclusiveMinimum 1 @exclusiveMaximum 2 @multipleOf 0.5 @default {"x":"é"}\n'
        "  n?: number;\n"
        "}): Promise<unknown>;",
        "\n"
        "\n"
        "codemode tool declaration:\n"
        "```ts\n"
        "declare const tools: { draft4(args: {\n"
        '  // @exclusiveMinimum 1 @exclusiveMaximum 2 @multipleOf 0.5 @default {"x":"é"}\n'
        "  n?: number;\n"
        "}): Promise<unknown>; };\n"
        "```",
        "{ n?: number; }",
    ),
    (
        {
            "name": "weird",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ä": {"type": "string"},
                    "Z": {"type": "string"},
                    "a": {"type": "strange"},
                    "n": {"type": None},
                },
            },
        },
        'weird(args: { Z?: string; a?: unknown; n?: unknown; "ä"?: string; }): Promise<unknown>;',
        "\n"
        "\n"
        "codemode tool declaration:\n"
        "```ts\n"
        'declare const tools: { weird(args: { Z?: string; a?: unknown; n?: unknown; "ä"?: string; }): '
        "Promise<unknown>; };\n"
        "```",
        '{ Z?: string; a?: unknown; n?: unknown; "ä"?: string; }',
    ),
]


@pytest.mark.parametrize(
    ("case", "signature", "sample", "budgeted"), PARITY_CASES, ids=[row[0]["name"] for row in PARITY_CASES]
)
def test_matches_upstream_output(case: dict[str, Any], signature: str, sample: str, budgeted: str | None) -> None:
    tool = declaration(case)
    assert render_tool_signature(tool) == signature
    assert render_tool_sample(tool) == sample
    if budgeted is not None:
        assert schema_to_type(case["inputSchema"], max_chars=60) == budgeted


def test_matches_upstream_declarations_of_every_parity_case() -> None:
    tools = [declaration(case) for case, *_ in PARITY_CASES]
    rendered = render_declarations(tools=tools)
    assert rendered.startswith("declare const tools: {\n  /** Read a file */\n  read(args: {\n")
    for _case, signature, *_ in PARITY_CASES:
        assert "  " + signature in rendered
