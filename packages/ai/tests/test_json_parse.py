from __future__ import annotations

import json
import random

import pytest

from pi_ai import StreamingJsonParser, parse_streaming_json, parse_tool_call_arguments, repair_json
from pi_ai.utils.json_parse import complete_partial_json


class TestParseStreamingJson:
    # #10868
    def test_keeps_invalid_u_escapes_as_literal_text(self) -> None:
        assert parse_streaming_json(r'{"text":"\url{x}"}') == {"text": "\\url{x}"}
        assert parse_streaming_json(r'{"path":"a.tex","content":"\usepackage{x}","mode":"w"}') == {
            "path": "a.tex",
            "content": "\\usepackage{x}",
            "mode": "w",
        }

    def test_decodes_valid_u_escapes(self) -> None:
        assert parse_streaming_json(r'{"text":"café \url"}') == {"text": "café \\url"}

    def test_omits_cut_off_u_escape(self) -> None:
        assert parse_streaming_json(r'{"text":"ab\u00') == {"text": "ab"}

    def test_empty_and_whitespace(self) -> None:
        assert parse_streaming_json(None) == {}
        assert parse_streaming_json("") == {}
        assert parse_streaming_json("   ") == {}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"a": 1', {"a": 1}),
            ('{"a": "hel', {"a": "hel"}),
            ('{"a"', {}),
            ('{"a":', {}),
            ('{"a": 1,', {"a": 1}),
            ('{"a": [1, 2,', {"a": [1, 2]}),
            ('{"a": {"b": "c", "d"', {"a": {"b": "c"}}),
            ('{"a": tr', {"a": True}),
            ('{"a": nul', {"a": None}),
            ('{"a": -', {}),
            ('{"a": 1.', {"a": 1}),
            ('{"a": "x\\', {"a": "x"}),
            ('{"a": "line\nbreak', {"a": "line\nbreak"}),
            ("[1, 2", [1, 2]),
        ],
    )
    def test_partial_documents(self, text: str, expected: object) -> None:
        assert parse_streaming_json(text) == expected


class TestParseToolCallArguments:
    # #10897
    def test_parses_and_repairs_invalid_escapes(self) -> None:
        assert parse_tool_call_arguments(r'{"kind":"seek","note":"\url"}') == ({"kind": "seek", "note": "\\url"}, None)

    def test_empty_text(self) -> None:
        assert parse_tool_call_arguments("") == ({}, None)

    def test_does_not_keep_prefix_of_malformed_json(self) -> None:
        raw = '{"action":"apply","actions":[{"kind":"seek","time":.5}]}'
        arguments, error = parse_tool_call_arguments(raw)
        assert arguments == {}
        assert error is not None and "\n\nReceived arguments:\n" in error
        assert error.endswith(raw)

    def test_does_not_complete_cut_off_json(self) -> None:
        raw = '{"kind":"seek","clipId":"clip-fl'
        arguments, error = parse_tool_call_arguments(raw)
        assert arguments == {}
        assert error is not None and error.endswith(raw)

    def test_rejects_non_objects(self) -> None:
        assert parse_tool_call_arguments("[1,2]") == (
            {},
            "Tool arguments must be a JSON object\n\nReceived arguments:\n[1,2]",
        )


def test_repair_json_escapes_control_characters_and_invalid_escapes() -> None:
    assert repair_json('{"a":"x\ny"}') == '{"a":"x\\ny"}'
    assert repair_json('{"a":"\\d"}') == '{"a":"\\\\d"}'
    assert repair_json('{"a":"\\u00e9"}') == '{"a":"\\u00e9"}'


def _documents() -> list[object]:
    return [
        {"path": "src/main.py", "content": "def f():\n    return 'x'\n", "mode": "w"},
        {"nested": {"list": [1, 2.5, -3, True, False, None], "deep": {"k": "vé\\"}}, "empty": {}},
        {"unicode": "emoji 🙈 and 中文", "escapes": 'quote " slash / tab \t'},
        [{"a": 1}, {"b": [2, 3]}],
    ]


@pytest.mark.parametrize("document", _documents())
def test_incremental_parser_matches_full_parse_at_every_prefix(document: object) -> None:
    text = json.dumps(document, ensure_ascii=False)
    rng = random.Random(1)
    parser = StreamingJsonParser()
    position = 0
    while position < len(text):
        size = rng.randint(1, 7)
        delta = text[position : position + size]
        position += size
        parser.feed(delta)
        value = parser.value()
        # Every intermediate value must be a JSON value (never raises) ...
        assert value is not None
        # ... and agree with the stateless completion of the same prefix when that parses.
        completed = complete_partial_json(parser.text)
        if completed is not None:
            try:
                expected = json.loads(completed, strict=False)
            except ValueError:
                continue
            assert value == expected
    assert parser.value() == document


def test_incremental_parser_keeps_last_good_value_on_unparseable_prefix() -> None:
    parser = StreamingJsonParser()
    parser.feed('{"a": 1, "b": ')
    assert parser.value() == {"a": 1}
    parser.feed("@@@")
    assert parser.value() == {"a": 1}
