"""Port of upstream ``test/source.test.ts``, for both languages.

The cases are written for Python, whose options line is ``# @options:``; for JavaScript, upstream's language, the
comments become ``//`` comments and the messages name JavaScript.
"""

from __future__ import annotations

import re

import pytest

from pi_codemode import CodemodeLanguage
from pi_codemode.source import (
    CODEMODE_OPTIONS_PREFIX,
    CODEMODE_SOURCE_GRAMMAR,
    PYTHON_CODEMODE_OPTIONS_PREFIX,
    PYTHON_CODEMODE_SOURCE_GRAMMAR,
    CodemodeSourceError,
    CodemodeSourceOptions,
    ParsedCodemodeSource,
    codemode_options_prefix,
    codemode_source_grammar,
)
from pi_codemode.source import parse_codemode_source as parse

_NAMES = {"javascript": "JavaScript", "python": "Python"}


@pytest.fixture(params=["javascript", "python"])
def language(request: pytest.FixtureRequest) -> CodemodeLanguage:
    return request.param


def adapt(text: str, language: CodemodeLanguage) -> str:
    """A Python case in ``language``: ``#`` comments become ``//`` comments in JavaScript."""
    if language == "python":
        return text
    return re.sub(r"(^|\n)([ \t]*)# ", r"\1\2// ", text.replace("# @options", "// @options"))


def parse_codemode_source(source: str, language: CodemodeLanguage) -> ParsedCodemodeSource:
    return parse(adapt(source, language), language)


def message(text: str, language: CodemodeLanguage) -> str:
    return text.replace("Python", _NAMES[language]).replace("# @options:", codemode_options_prefix(language))


def test_returns_plain_code_unchanged(language: CodemodeLanguage) -> None:
    assert parse_codemode_source("text('hi')", language) == ParsedCodemodeSource("text('hi')", CodemodeSourceOptions())
    assert parse_codemode_source("# just a comment\nreturn 1", language) == ParsedCodemodeSource(
        adapt("# just a comment\nreturn 1", language), CodemodeSourceOptions()
    )


def test_parses_the_options_line_and_keeps_line_numbers(language: CodemodeLanguage) -> None:
    assert parse_codemode_source('# @options: {"timeout_ms": 10}\na = 1\ntext(a)', language) == ParsedCodemodeSource(
        "\na = 1\ntext(a)", CodemodeSourceOptions(timeout_ms=10)
    )
    assert parse_codemode_source(
        '  # @options:{"max_output_tokens":0,"timeout_ms":1500}\r\ntext(1)', language
    ).options == CodemodeSourceOptions(max_output_tokens=0, timeout_ms=1500)
    assert parse_codemode_source("# @options: {}\ntext(1)", language) == ParsedCodemodeSource(
        "\ntext(1)", CodemodeSourceOptions()
    )


def test_only_treats_the_first_line_as_an_options_line(language: CodemodeLanguage) -> None:
    assert parse_codemode_source("# @optionsx {}\ntext(1)", language).options == CodemodeSourceOptions()
    assert parse_codemode_source('text("# @options: {}")', language).options == CodemodeSourceOptions()


def test_reads_only_the_options_line_of_its_own_language() -> None:
    assert parse('# @options: {"timeout_ms": 1}\ntext(1)', "javascript").options == CodemodeSourceOptions()
    assert parse('// @options: {"timeout_ms": 1}\ntext(1)', "python").options == CodemodeSourceOptions()
    assert parse('// @options: {"timeout_ms": 1}\ntext(1)').options == CodemodeSourceOptions(timeout_ms=1)


# #10631: an options line after the first line used to be ignored, so its timeout never applied.
@pytest.mark.parametrize(
    ("source", "line"),
    [
        ('# count things\n# @options: {"timeout_ms": 1000}\nwhile True: pass', 2),
        ('\n  # @options: {"timeout_ms": 1000}\nwhile True: pass', 2),
        ('text(1)\r\n# @options: {"timeout_ms": 1}', 2),
        ('# @options: {"timeout_ms": 1}\ntext(1)\n# @options: {"timeout_ms": 2}', 3),
    ],
)
def test_rejects_an_options_line_that_is_not_the_first_line(source: str, line: int, language: CodemodeLanguage) -> None:
    with pytest.raises(
        CodemodeSourceError,
        match=re.escape(
            message(f"The # @options: line must be the first line of the input; found one on line {line}", language)
        ),
    ):
        parse_codemode_source(source, language)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", r"Expected Python source text \(non-empty\)"),
        ("  \n", r"Expected Python source text \(non-empty\)"),
        ("# @options:\ntext(1)", "@options must be a JSON object with supported fields"),
        ("# @options: {timeout_ms: 1}\ntext(1)", "@options must be valid JSON with supported fields"),
        ("# @options: [1]\ntext(1)", "@options must be a JSON object with supported fields"),
        (
            '# @options: {"yield": 1}\ntext(1)',
            re.escape("@options only supports `max_output_tokens` and `timeout_ms`; got `yield`"),
        ),
        (
            '# @options: {"max_output_tokens": 1.5}\ntext(1)',
            re.escape("@options field `max_output_tokens` must be a non-negative safe integer"),
        ),
        ('# @options: {"timeout_ms": 0}\ntext(1)', "@options field `timeout_ms` must be a positive integer"),
        (
            '# @options: {"timeout_ms": 1}',
            "The @options line must be followed by Python source on subsequent lines",
        ),
        (
            '# @options: {"timeout_ms": 1}\n  \n',
            "The @options line must be followed by Python source on subsequent lines",
        ),
        # Python additions: JSON's `null`, booleans and non-finite numbers are not integers.
        ('# @options: {"timeout_ms": null}\ntext(1)', "@options field `timeout_ms` must be a positive integer"),
        ('# @options: {"max_output_tokens": true}\ntext(1)', "must be a non-negative safe integer"),
        ('# @options: {"timeout_ms": NaN}\ntext(1)', "@options must be valid JSON"),
    ],
)
def test_rejects_empty_input_and_invalid_options(source: str, expected: str, language: CodemodeLanguage) -> None:
    with pytest.raises(CodemodeSourceError, match=message(expected, language)):
        parse_codemode_source(source, language)


def test_names_the_language_in_the_empty_input_message() -> None:
    with pytest.raises(CodemodeSourceError) as javascript:
        parse("")
    assert str(javascript.value) == (
        "Expected JavaScript source text (non-empty). Provide JS only, optionally with a first line "
        '`// @options: {"max_output_tokens": 1000}`.'
    )
    with pytest.raises(CodemodeSourceError) as python:
        parse("", "python")
    assert str(python.value) == (
        "Expected Python source text (non-empty). Provide Python only, optionally with a first line "
        '`# @options: {"max_output_tokens": 1000}`.'
    )


def test_accepts_integral_json_numbers_like_javascript(language: CodemodeLanguage) -> None:
    assert parse_codemode_source('# @options: {"timeout_ms": 10.0}\ntext(1)', language).options.timeout_ms == 10


def test_grammar_fixes_the_options_line_shape() -> None:
    assert r"OPTIONS_LINE: /[ \t]*\/\/ @options:[^\r\n]*/" in CODEMODE_SOURCE_GRAMMAR
    assert r"OPTIONS_LINE: /[ \t]*# @options:[^\r\n]*/" in PYTHON_CODEMODE_SOURCE_GRAMMAR
    for grammar in (CODEMODE_SOURCE_GRAMMAR, PYTHON_CODEMODE_SOURCE_GRAMMAR):
        assert grammar.strip().startswith("start: options_source | plain_source")
    assert codemode_source_grammar() is CODEMODE_SOURCE_GRAMMAR
    assert codemode_source_grammar("python") is PYTHON_CODEMODE_SOURCE_GRAMMAR
    assert (CODEMODE_OPTIONS_PREFIX, PYTHON_CODEMODE_OPTIONS_PREFIX) == ("// @options:", "# @options:")
