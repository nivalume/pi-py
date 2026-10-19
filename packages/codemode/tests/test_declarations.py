"""Port of upstream ``test/declarations.test.ts``.

Declarations are Python here: the same schemas and the same rules (sorted keys, comment lines for
descriptions and constraint tags, the budget, ``$ref`` handling), rendered as Python types and stubs.
"""

from __future__ import annotations

from typing import Any

from pi_codemode import (
    CodemodeTool,
    CodemodeToolDeclaration,
    render_declarations,
    render_tool_sample,
    render_tool_signature,
    schema_to_type,
)


def execute(args: Any, context: Any) -> None:
    return None


class TestSchemaToType:
    def test_leaves_out_the_integer_tag_python_types_express(self) -> None:
        assert schema_to_type({"type": "object", "properties": {"n": {"type": ["integer", "null"]}}}) == (
            'TypedDict[{"n": NotRequired[int | None]}]'
        )

    def test_renders_primitives_literals_and_unions(self) -> None:
        assert schema_to_type({"type": "string"}) == "str"
        assert schema_to_type({"type": "integer"}) == "int"
        assert schema_to_type({"type": "number"}) == "float"
        assert schema_to_type({"type": ["string", "null"]}) == "str | None"
        assert schema_to_type({"const": "a"}) == 'Literal["a"]'
        assert schema_to_type({"enum": ["a", 1, None]}) == 'Literal["a", 1, None]'
        assert schema_to_type({"anyOf": [{"type": "string"}, {"type": "number"}]}) == "str | float"
        assert schema_to_type({"anyOf": [{"type": "string"}, {}]}) == "Any"
        # No intersection types in Python: the most specific part (a literal) wins.
        assert schema_to_type({"allOf": [{"anyOf": [{"type": "string"}, {"type": "number"}]}, {"const": 1}]}) == (
            "Literal[1]"
        )
        assert schema_to_type({"$ref": "#/defs/x"}) == "Any"
        assert schema_to_type(True) == "Any"
        assert schema_to_type(False) == "Never"

    def test_python_specific_literals(self) -> None:
        assert (
            schema_to_type({"enum": [True, 1.5, [1], {"a": 1}]}) == "Literal[True] | float | list[Any] | dict[str, Any]"
        )
        assert schema_to_type({"const": 'say "hi"'}) == 'Literal["say \\"hi\\""]'

    def test_merges_object_parts_of_all_of(self) -> None:
        assert (
            schema_to_type(
                {
                    "allOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                        {"type": "object", "properties": {"b": {"type": "integer"}}},
                    ]
                }
            )
            == 'TypedDict[{"a": str, "b": NotRequired[int]}]'
        )

    def test_renders_objects_on_one_line_with_sorted_properties(self) -> None:
        assert (
            schema_to_type(
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "max-lines": {"type": "number"}},
                    "required": ["city"],
                    "additionalProperties": False,
                }
            )
            == 'TypedDict[{"city": str, "max-lines": NotRequired[float]}]'
        )
        assert schema_to_type({"type": "object", "additionalProperties": {"type": "number"}}) == "dict[str, float]"
        assert schema_to_type({"type": "object"}) == "dict[str, Any]"
        assert schema_to_type({"type": "object", "properties": {}, "additionalProperties": False}) == "TypedDict[{}]"
        assert (
            schema_to_type(
                {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": {"type": "number"}}
            )
            == 'TypedDict[{"a": NotRequired[str], **dict[str, float]}]'
        )

    def test_puts_property_descriptions_on_comment_lines(self) -> None:
        assert (
            schema_to_type(
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
            )
            == 'TypedDict[{\n    # look up weather for a given list of locations\n    "weather": list[TypedDict[{"location": str}]],\n}]'
        )
        assert (
            schema_to_type(
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
            )
            == 'TypedDict[{\n    # Outer\n    "outer": NotRequired[TypedDict[{\n        # Inner\n        "inner": NotRequired[str],\n    }]],\n}]'
        )

    # https://github.com/earendil-works/pi/issues/10707
    def test_puts_constraints_the_type_cannot_express_on_property_comment_lines(self) -> None:
        assert schema_to_type(
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
        ) == "\n".join(
            [
                "TypedDict[{",
                "    # @maxLength 8 @pattern ^[a-z]+$",
                '    "id": str,',
                '    "name": NotRequired[str],',
                "    # @maxItems 5 @uniqueItems @deprecated",
                '    "tags": NotRequired[list[str]],',
                "    # Wait in milliseconds.",
                "    # @minimum 0 @maximum 30000 @default 0",
                '    "timeoutMs": NotRequired[int],',
                "}]",
            ]
        )

    def test_reads_constraints_of_optional_fields_from_the_non_null_branch(self) -> None:
        # Pydantic's `limit: int | None = Field(None, ge=1, le=100)`.
        assert (
            schema_to_type(
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
            )
            == 'TypedDict[{\n    # @minimum 1 @maximum 100\n    "limit": NotRequired[int | None],\n}]'
        )
        # Constraints of the property and the branch both apply.
        assert (
            schema_to_type(
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
            )
            == 'TypedDict[{\n    # @minimum 10 @maximum 20 @format port @format int32\n    "n": NotRequired[int | None],\n}]'
        )

    def test_reads_draft_04_boolean_exclusive_bounds(self) -> None:
        assert schema_to_type(
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
        ) == "\n".join(
            [
                "TypedDict[{",
                "    # @maximum 8 @exclusiveMaximum 5",
                '    "m": NotRequired[float | None],',
                "    # @exclusiveMinimum 0 @maximum 10",
                '    "n": NotRequired[int],',
                "}]",
            ]
        )

    def test_leaves_out_long_tags_and_escapes_line_terminators(self) -> None:
        assert (
            schema_to_type(
                {"type": "object", "properties": {"config": {"type": "object", "default": {"text": "x" * 100}}}}
            )
            == 'TypedDict[{"config": NotRequired[dict[str, Any]]}]'
        )
        assert schema_to_type(
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
        ) == "\n".join(
            [
                "TypedDict[{",
                "    # a",
                "    # b",
                '    # @pattern ^[\\u000d\\u000a\\u2028\\u2029]$ @default "\\u2028"',
                '    "sep": NotRequired[str],',
                "}]",
            ]
        )

    def test_drops_constraint_tags_before_rendering_a_type_over_the_budget_as_any(self) -> None:
        schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}
        assert schema_to_type(schema) == 'TypedDict[{\n    # @minimum 1\n    "n": NotRequired[int],\n}]'
        assert schema_to_type(schema, max_chars=40) == 'TypedDict[{"n": NotRequired[int]}]'
        assert schema_to_type(schema, max_chars=10) == "Any"

    def test_resolves_local_references_and_stops_at_recursive_ones(self) -> None:
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
        assert schema_to_type(schema) == (
            'TypedDict[{"item": TypedDict[{"id": str, "parent": NotRequired[Any]}], '
            '"legacy": NotRequired[Literal["a", "b"]], "remote": NotRequired[Any]}]'
        )

    def test_renders_arrays_and_tuples(self) -> None:
        assert schema_to_type({"type": "array", "items": {"type": "string"}}) == "list[str]"
        assert schema_to_type({"type": "array", "prefixItems": [{"type": "string"}, {"type": "number"}]}) == (
            "tuple[str, float]"
        )
        assert schema_to_type({"type": "array"}) == "list[Any]"

    def test_renders_types_over_the_budget_as_any(self) -> None:
        properties = {f"field{index}": {"type": "string"} for index in range(50)}
        schema = {"type": "object", "properties": properties}
        assert schema_to_type(schema, max_chars=100) == "Any"
        assert '"field49": NotRequired[str]' in schema_to_type(schema)


class TestToolDeclarations:
    def test_renders_signatures_with_normalized_identifiers(self) -> None:
        assert (
            render_tool_signature(
                CodemodeToolDeclaration(
                    "hidden-dynamic-tool",
                    input_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                )
            )
            == 'async def hidden_dynamic_tool(self, *, city: str) -> TypedDict[{"ok": bool}]: ...'
        )
        assert render_tool_signature(CodemodeToolDeclaration("free")) == (
            "async def free(self, args: Any = None, /) -> Any: ..."
        )

    def test_renders_keyword_parameters_with_comments(self) -> None:
        assert render_tool_signature(
            CodemodeToolDeclaration(
                "read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File to read"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                },
                output_schema={"type": "string"},
            )
        ) == "\n".join(
            [
                "async def read(",
                "    self,",
                "    *,",
                "    # @minimum 1",
                "    limit: int = ...,",
                "    # File to read",
                "    path: str,",
                ") -> str: ...",
            ]
        )

    def test_renders_non_identifier_keys_and_non_objects_as_one_positional_argument(self) -> None:
        assert (
            render_tool_signature(
                CodemodeToolDeclaration(
                    "t", input_schema={"type": "object", "properties": {"max-lines": {"type": "number"}}}
                )
            )
            == 'async def t(self, args: TypedDict[{"max-lines": NotRequired[float]}], /) -> Any: ...'
        )
        assert render_tool_signature(CodemodeToolDeclaration("t", input_schema={"type": "string"})) == (
            "async def t(self, args: str, /) -> Any: ..."
        )
        assert render_tool_signature(
            CodemodeToolDeclaration(
                "t", input_schema={"type": "object", "properties": {}, "additionalProperties": False}
            )
        ) == ("async def t(self) -> Any: ...")

    def test_falls_back_when_parameters_exceed_the_budget(self) -> None:
        tool = CodemodeToolDeclaration(
            "big", input_schema={"type": "object", "properties": {f"field{i}": {"type": "string"} for i in range(20)}}
        )
        assert render_tool_signature(tool, input_max_chars=50) == (
            "async def big(self, args: Any = None, /, **kwargs: Any) -> Any: ..."
        )

    def test_renders_output_schemas_shaped_like_an_mcp_call_tool_result_as_plain_types(self) -> None:
        output_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "array", "items": {"type": "object"}},
                "isError": {"type": "boolean"},
                "_meta": {"type": "object"},
            },
            "required": ["content"],
        }
        assert render_tool_signature(
            CodemodeToolDeclaration(
                "plain", input_schema={"type": "object", "properties": {}}, output_schema=output_schema
            )
        ) == (
            'async def plain(self, **kwargs: Any) -> TypedDict[{"_meta": NotRequired[dict[str, Any]], '
            '"content": list[dict[str, Any]], "isError": NotRequired[bool]}]: ...'
        )

    def test_renders_the_per_tool_sample(self) -> None:
        assert render_tool_sample(
            CodemodeToolDeclaration("foo", description="bar", input_schema={"type": "string"})
        ) == (
            "bar\n\ncodemode tool declaration:\n```python\nclass tools:\n    async def foo(self, args: str, /) -> Any: ...\n```"
        )


class TestRenderDeclarations:
    def test_renders_tools_and_globals(self) -> None:
        text = render_declarations(
            tools=[
                CodemodeTool(
                    "read",
                    execute,
                    description="Read a file.\nSecond line.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                    output_schema={"type": "string"},
                ),
                CodemodeTool("remote-api", execute),
            ],
            globals=[CodemodeTool("attach", execute, description="Attach it.", input_schema={"type": "string"})],
        )
        assert text == "\n".join(
            [
                "class tools:",
                "    async def read(self, *, path: str) -> str:",
                '        """',
                "        Read a file.",
                "        Second line.",
                '        """',
                "    async def remote_api(self, args: Any = None, /) -> Any: ...",
                "",
                "async def attach(args: str, /) -> Any:",
                '    """Attach it."""',
            ]
        )

    def test_renders_namespaced_globals_and_explicit_signatures(self) -> None:
        text = render_declarations(
            globals=[
                CodemodeTool("models.list", execute, description="List models.", signature="(type: str) -> list[str]"),
                CodemodeTool("models.get", execute, input_schema={"type": "string"}),
                CodemodeTool("plain", execute, signature="() -> None"),
            ]
        )
        assert text == "\n".join(
            [
                "async def plain() -> None: ...",
                "",
                "class models:",
                "    async def list(self, type: str) -> list[str]:",
                '        """List models."""',
                "    async def get(self, args: str, /) -> Any: ...",
            ]
        )

    def test_escapes_docstring_terminators_in_descriptions(self) -> None:
        text = render_declarations(tools=[CodemodeTool("x", execute, description='a """ b \\ c')])
        assert '"""a \\"\\"\\" b \\\\ c"""' in text

    def test_the_declarations_are_valid_python(self) -> None:
        import ast

        text = render_declarations(
            tools=[
                CodemodeTool(
                    "read",
                    execute,
                    description="Read.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Path"}, "n": {"type": "integer"}},
                        "required": ["path"],
                    },
                ),
                CodemodeTool(
                    "odd-keys",
                    execute,
                    input_schema={"type": "object", "properties": {"a-b": {"type": "string", "description": "x"}}},
                ),
            ],
            globals=[CodemodeTool("models.list", execute, signature="(type: str) -> list[str]")],
        )
        ast.parse(text)
