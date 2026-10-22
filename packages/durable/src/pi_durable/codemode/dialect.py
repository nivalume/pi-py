"""What code mode tells the model and scripts, in the script's language.

The code mode tool follows its sandbox's language (:attr:`pi_codemode.CodemodeSandboxBase.language`), so a
conversation offers JavaScript or Python scripts, never both. JavaScript is upstream's language: its globals
(``searchTools``, ``models.generateImages``, ...), descriptions and error messages are upstream's, word for word.
Python has the same environment with snake_case globals, keyword arguments, and messages that name Python's types.
The internal tools (``pi.codemode.models.*``) and what is stored keep upstream's names in both.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from pi_codemode.language import CodemodeLanguage, check_codemode_language

__all__ = ["JAVASCRIPT", "PYTHON", "Dialect", "MAX_CONCURRENT_MODEL_CALLS", "dialect_of"]

#: ``models.classify()`` and ``models.generateImages()`` calls one script may have in flight; the rest wait.
MAX_CONCURRENT_MODEL_CALLS: Final = 4

_VOWEL: Final = re.compile(r"^[aeiou]")


def with_article(word: str) -> str:
    """``an image``, ``a classifier``."""
    return f"{'an' if _VOWEL.match(word) else 'a'} {word}"


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _js_json_text(value: Any) -> str:
    """``JSON.stringify(value)`` in a message: compact."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _describe_js(value: Any) -> str:
    """A script value in an error message: ``null``, ``a string``, ``an array``, or its keys (``{ prompt }``)."""
    if value is None:
        return "null"
    if isinstance(value, list):
        return "an empty array" if len(cast("list[Any]", value)) == 0 else "an array"
    if isinstance(value, dict):
        keys = [str(key) for key in cast("dict[str, Any]", value)]
        if len(keys) == 0:
            return "{}"
        return "{ " + ", ".join(keys[:6]) + (", ..." if len(keys) > 6 else "") + " }"
    if isinstance(value, str):
        return "a string"
    return "a boolean" if isinstance(value, bool) else "a number"


def _describe_python(value: Any) -> str:
    """A script value in an error message: ``None``, ``a str``, ``a list``, or its keys (``{"prompt": ...}``)."""
    if value is None:
        return "None"
    if isinstance(value, list):
        return "an empty list" if len(cast("list[Any]", value)) == 0 else "a list"
    if isinstance(value, dict):
        keys = [str(key) for key in cast("dict[str, Any]", value)]
        if len(keys) == 0:
            return "{}"
        shown = ", ".join(f'"{key}": ...' for key in keys[:6])
        return "{" + shown + (", ..." if len(keys) > 6 else "") + "}"
    return with_article(type(value).__name__)


def _bind_positional(args: Any, names: Sequence[str]) -> list[Any]:
    """JavaScript: a spread global's arguments by position, ``None`` for missing ones."""
    values: list[Any] = list(args) if isinstance(args, list) else []  # pyright: ignore[reportUnknownArgumentType]
    return [values[index] if index < len(values) else None for index in range(len(names))]


def _bind_keywords(args: Any, names: Sequence[str]) -> list[Any]:
    """Python: the arguments of a spread global's call by parameter, ``None`` for missing ones.

    A spread global receives the positional arguments as a list, and keyword arguments as one trailing dict. That
    dict is taken as the keyword arguments when every key names a parameter that no positional argument filled, so
    ``f(a, b)``, ``f(a, b=b)`` and ``f(a=a, b=b)`` bind alike, while a dict passed positionally stays a value.
    """
    values: list[Any] = list(args) if isinstance(args, list) else []  # pyright: ignore[reportUnknownArgumentType]
    keywords: dict[str, Any] = {}
    if values and isinstance(values[-1], dict):
        last = cast("dict[str, Any]", values[-1])
        free = set(names[len(values) - 1 :])
        if last and all(key in free for key in last):
            keywords = last
            values = values[:-1]
    bound = [values[index] if index < len(values) else None for index in range(len(names))]
    for index, name in enumerate(names):
        if name in keywords:
            bound[index] = keywords[name]
    return bound


@dataclass(frozen=True, slots=True)
class Dialect:
    """Names and text of code mode in one script language."""

    language: CodemodeLanguage
    #: Discovery globals: ``searchTools``, ``describeTool``, ``describeNamespace``.
    search_tools: str
    describe_tool: str
    describe_namespace: str
    #: Catalog globals: ``models.getModelsOfType``, ``models.getAvailableOfType``, ``models.getModelOfType``.
    get_models_of_type: str
    get_available_of_type: str
    get_model_of_type: str
    #: Model call globals: ``models.classify``, ``models.generateImages``.
    classify: str
    generate_images: str
    #: The code mode tool's description, before its globals.
    description_intro: str
    #: The globals lines of the description, after "Globals:".
    globals_lines: tuple[str, ...]
    models_description: str
    #: The ``code`` parameter's description.
    code_description: str
    #: The first line of the ``codemode`` section.
    section_header: str
    #: The section's note for tools without a namespace that do not fit.
    not_listed: str
    #: How messages name a JSON object, a JSON array, and a missing value.
    object_noun: str
    array_noun: str
    missing: str
    classifier_context_shape: str
    images_context_shape: str
    #: ``criteria`` of a bool question, as messages show it.
    bool_criteria: str
    #: ``describe_value``: a script value in an error message.
    describe_value: Callable[[Any], str]
    #: ``bind(args, names)``: a spread global's argument list by parameter.
    bind: Callable[[Any, Sequence[str]], list[Any]]
    #: ``json_text``: a value as messages quote it.
    json_text: Callable[[Any], str]

    @property
    def discovery_globals(self) -> tuple[str, str, str]:
        return (self.search_tools, self.describe_tool, self.describe_namespace)

    @property
    def catalog_globals(self) -> tuple[str, str, str]:
        return (self.get_models_of_type, self.get_available_of_type, self.get_model_of_type)

    def field(self, base: str, key: str) -> str:
        """``context.state`` (JavaScript) or ``context["state"]`` (Python)."""
        return f"{base}.{key}" if self.language == "javascript" else f'{base}["{key}"]'

    def entry(self, base: str, key: str) -> str:
        """A question by its ID: ``context.questions.<id>`` or ``context["questions"]["<id>"]``."""
        return f"{base}.{key}" if self.language == "javascript" else f"{base}[{_json_text(key)}]"

    def generated_images_note(self, count: int) -> str:
        plural = "" if count == 1 else "s"
        return (
            f"Note: {self.generate_images}() returned {count} image{plural} that the script did not show. "
            f"Show each image block of {self.field('result', 'output')} with image(block)."
        )


JAVASCRIPT: Final = Dialect(
    language="javascript",
    search_tools="searchTools",
    describe_tool="describeTool",
    describe_namespace="describeNamespace",
    get_models_of_type="models.getModelsOfType",
    get_available_of_type="models.getAvailableOfType",
    get_model_of_type="models.getModelOfType",
    classify="models.classify",
    generate_images="models.generateImages",
    description_intro="""Run JavaScript that calls other tools. The input is raw JavaScript (not JSON, no code fence), run as an async function body in a QuickJS sandbox: top-level `await` and `return` work. No Node, file system, network, or timers.
- `await tools.<name>({ ...args })` resolves to a string, or an object if the tool's declaration says so, and rejects with an Error on failure. A tool whose declaration returns an object resolves on failure too when it returned that object; check its fields. Calls still running when the script ends are cancelled.
- The script survives a restart: it resumes where it was, without running finished calls again.
- Optional first line: `// @options: {"max_output_tokens": 10000, "timeout_ms": 60000}`""",
    globals_lines=(
        "- `text(value)`, `image(dataUrlOrImageBlock)`, `console.log(...)`, and top-level `return` add output; `exit()` "
        "ends the script. With several text items, each starts with a `==> text N/M <==` line, and `console` lines "
        "follow the other output in one `<console_output>` block. `image()` also saves the image to a temp file and "
        "the result names its path.",
        "- `store(key, value)` and `load(key)` keep JSON values across codemode calls.",
        "- `ALL_TOOLS`, `await searchTools(query, { limit?, namespace? })`, `await describeTool(name)`, "
        "`await describeNamespace(name)`: find tools the `codemode` section does not list.",
    ),
    models_description="\n".join(
        [
            "`models` (all functions return promises):",
            "- `models.getModelsOfType(type, provider?)`, `models.getAvailableOfType(type, provider?)` (configured "
            'auth only), `models.getModelOfType(type, provider, id)`: the catalog; `type` is "chat", "image", or '
            '"classifier".',
            '- `models.classify(model, { state: { ... }, images?: [{ type: "image", data, mimeType }], questions: { '
            '<id>: { type: "choice", instructions, criteria: { <label>: <meaning> } } | { type: "score", instructions, '
            'criteria: [<lowest level>, ..., <highest level>] } | { type: "bool", instructions, criteria: { true: '
            "<meaning>, false: <meaning> } } } })` resolves to `{ answers: { <id>: ... }, stopReason, errorMessage?, "
            "usage? }`.",
            '- `models.generateImages(model, { input: [{ type: "text", text: <prompt> }, ...optional { type: "image", '
            "data, mimeType }] })` resolves to `{ output: [image blocks], stopReason, errorMessage?, usage? }`; show "
            "each image with `image(block)`.",
            f"At most {MAX_CONCURRENT_MODEL_CALLS} classify/generateImages calls run at once; more wait.",
        ]
    ),
    code_description="Raw JavaScript source.",
    section_header="Tools scripts can call (`await tools.<name>(args)`):",
    not_listed="Some tools are not listed: find them with `searchTools()`.",
    object_noun="object",
    array_noun="array",
    missing="undefined",
    classifier_context_shape=(
        '{ state: { ... }, images?: [{ type: "image", data: <base64>, mimeType }], questions: { <id>: { type: '
        '"choice", instructions, criteria: { <label>: <meaning> } } | { type: "score", instructions, criteria: '
        '[<lowest level>, ..., <highest level>] } | { type: "bool", instructions, criteria: { true: <meaning>, '
        "false: <meaning> } } } }"
    ),
    images_context_shape=(
        '{ input: [{ type: "text", text: <prompt> }, ...optional { type: "image", data: <base64>, mimeType } '
        "references] }"
    ),
    bool_criteria="{ true: string, false: string }",
    describe_value=_describe_js,
    bind=_bind_positional,
    json_text=_js_json_text,
)

PYTHON: Final = Dialect(
    language="python",
    search_tools="search_tools",
    describe_tool="describe_tool",
    describe_namespace="describe_namespace",
    get_models_of_type="models.get_models_of_type",
    get_available_of_type="models.get_available_of_type",
    get_model_of_type="models.get_model_of_type",
    classify="models.classify",
    generate_images="models.generate_images",
    description_intro="""Run Python that calls other tools. The input is raw Python (not JSON, no code fence), run as the body of an async function in a Monty sandbox: top-level `await` and `return` work. No file system, network, subprocesses, or host modules; standard modules such as `json`, `re`, `math`, `datetime`, and `asyncio` can be imported.
- `await tools.<name>(key=value, ...)` returns a string, or a dict if the tool's declaration says so, and raises an Exception with the error text on failure. A tool whose declaration returns a dict returns on failure too when it returned that dict; check its fields. Calls start when made, so `await asyncio.gather(...)` (after `import asyncio`) runs several at once. Calls still running when the script ends are cancelled.
- The script survives a restart: it resumes where it was, without running finished calls again.
- Optional first line: `# @options: {"max_output_tokens": 10000, "timeout_ms": 60000}`""",
    globals_lines=(
        "- `text(value)`, `image(data_url_or_image_block)`, `print(...)`, and top-level `return` add output; `exit()` "
        "ends the script. With several text items, each starts with a `==> text N/M <==` line, and `print` lines "
        "follow the other output in one `<console_output>` block. `image()` also saves the image to a temp file and "
        "the result names its path.",
        "- `store(key, value)` and `load(key)` keep JSON values across codemode calls.",
        "- `ALL_TOOLS`, `await search_tools(query, limit=8, namespace=None)`, `await describe_tool(name)`, "
        "`await describe_namespace(name)`: find tools the `codemode` section does not list.",
    ),
    models_description="\n".join(
        [
            "`models` (all functions are async: `await` them):",
            "- `models.get_models_of_type(type, provider=None)`, `models.get_available_of_type(type, provider=None)` "
            "(configured auth only), `models.get_model_of_type(type, provider, id)`: the catalog, as dicts with "
            'camelCase keys; `type` is "chat", "image", or "classifier".',
            '- `models.classify(model, {"state": {...}, "images": [{"type": "image", "data": ..., "mimeType": ...}] '
            '(optional), "questions": {<id>: {"type": "choice", "instructions": ..., "criteria": {<label>: <meaning>}} '
            '| {"type": "score", "instructions": ..., "criteria": [<lowest level>, ..., <highest level>]} | '
            '{"type": "bool", "instructions": ..., "criteria": {"true": <meaning>, "false": <meaning>}}}})` returns '
            '`{"answers": {<id>: ...}, "stopReason": ..., "errorMessage"?: ..., "usage"?: ...}`.',
            '- `models.generate_images(model, {"input": [{"type": "text", "text": <prompt>}, ...optional {"type": '
            '"image", "data": ..., "mimeType": ...}]})` returns `{"output": [image blocks], "stopReason": ..., '
            '"errorMessage"?: ..., "usage"?: ...}`; show each image with `image(block)`.',
            f"At most {MAX_CONCURRENT_MODEL_CALLS} classify/generate_images calls run at once; more wait.",
        ]
    ),
    code_description="Raw Python source.",
    section_header="Tools scripts can call (`await tools.<name>(...)`):",
    not_listed="Some tools are not listed: find them with `search_tools()`.",
    object_noun="dict",
    array_noun="list",
    missing="None",
    classifier_context_shape=(
        '{"state": {...}, "images": [{"type": "image", "data": <base64>, "mimeType": ...}] (optional), "questions": '
        '{<id>: {"type": "choice", "instructions": ..., "criteria": {<label>: <meaning>}} | {"type": "score", '
        '"instructions": ..., "criteria": [<lowest level>, ..., <highest level>]} | {"type": "bool", "instructions": '
        '..., "criteria": {"true": <meaning>, "false": <meaning>}}}}'
    ),
    images_context_shape=(
        '{"input": [{"type": "text", "text": <prompt>}, ...optional {"type": "image", "data": <base64>, '
        '"mimeType": ...} references]}'
    ),
    bool_criteria='{"true": str, "false": str}',
    describe_value=_describe_python,
    bind=_bind_keywords,
    json_text=_json_text,
)


def dialect_of(language: CodemodeLanguage) -> Dialect:
    return JAVASCRIPT if check_codemode_language(language) == "javascript" else PYTHON
