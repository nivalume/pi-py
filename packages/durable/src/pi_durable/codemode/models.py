"""``models.*`` for code mode scripts (port of ``codemode/models.ts``): the model catalog, and classifier and
image model calls.

The catalog functions are host globals, journaled like any global. ``models.classify()`` and
``models.generate_images()`` are paid effects, so each runs as an internal tool (``callers=["tools"]``, a reserved
name, hidden from scripts' tool listings): a nested durable call with its own task, live row, abort, usage
accounting, and replay policy. The code mode tool turns a script's call of the global into a call of that tool.

Scripts are Python, so the globals have snake_case names (``models.get_models_of_type``,
``models.generate_images``, ...); the internal tools keep upstream's names (``pi.codemode.models.classify``,
``pi.codemode.models.generateImages``). Catalog entries, contexts, and results are pi-ai's JSON (camelCase keys:
``mimeType``, ``stopReason``, ``contextWindow``), as upstream's scripts see them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pi_ai import ClassifierContext, ImagesContext, TextContent, Type
from pi_ai.serialization import from_dict, to_json_value

from ..harness.define import define_tool

if TYPE_CHECKING:
    from pi_ai import Models
    from pi_chord.context import Context

    from ..harness.types import ToolExecutionApi, ToolExecutionResult, ToolRegistration

__all__ = [
    "CATALOG_GLOBALS",
    "CLASSIFY_TOOL",
    "CODEMODE_INTERNAL_PREFIX",
    "GENERATE_IMAGES_TOOL",
    "MAX_CONCURRENT_MODEL_CALLS",
    "MODELS_DESCRIPTION",
    "MODEL_CALLS",
    "bind_arguments",
    "catalog_call",
    "create_model_tools",
]

#: Reserved names of code mode's internal tools; they are never listed to scripts or the model.
CODEMODE_INTERNAL_PREFIX: Final = "pi.codemode."
CLASSIFY_TOOL: Final = f"{CODEMODE_INTERNAL_PREFIX}models.classify"
GENERATE_IMAGES_TOOL: Final = f"{CODEMODE_INTERNAL_PREFIX}models.generateImages"

#: The internal tool, by the global a script calls.
MODEL_CALLS: Final[Mapping[str, str]] = {
    "models.classify": CLASSIFY_TOOL,
    "models.generate_images": GENERATE_IMAGES_TOOL,
}

#: The catalog functions, host globals of the ``models`` namespace.
CATALOG_GLOBALS: Final = (
    "models.get_models_of_type",
    "models.get_available_of_type",
    "models.get_model_of_type",
)

#: ``models.classify()`` and ``models.generate_images()`` calls one script may have in flight; the rest wait.
MAX_CONCURRENT_MODEL_CALLS: Final = 4

_MODEL_TYPES: Final = ("chat", "image", "classifier")

MODELS_DESCRIPTION: Final = "\n".join(
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
)

_VOWEL: Final = re.compile(r"^[aeiou]")


def bind_arguments(args: Any, names: Sequence[str]) -> list[Any]:
    """The arguments of a spread global's call by parameter, ``None`` for missing ones.

    A spread global receives the positional arguments as a list, and keyword arguments as one trailing dict. That
    dict is taken as the keyword arguments when every key names a parameter that no positional argument filled, so
    ``f(a, b)``, ``f(a, b=b)`` and ``f(a=a, b=b)`` bind alike, while a dict passed positionally stays a value.
    """
    values: list[Any] = list(args) if isinstance(args, list) else []
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


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def describe_value(value: Any) -> str:
    """A script value in an error message: ``None``, ``a str``, ``a list``, or its keys (``{"prompt": ...}``)."""
    if value is None:
        return "None"
    if isinstance(value, list):
        return "an empty list" if len(value) == 0 else "a list"  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(value, dict):
        keys = [str(key) for key in cast("dict[str, Any]", value)]
        if len(keys) == 0:
            return "{}"
        shown = ", ".join(f'"{key}": ...' for key in keys[:6])
        return "{" + shown + (", ..." if len(keys) > 6 else "") + "}"
    return _with_article(type(value).__name__)


def _with_article(word: str) -> str:
    """``an image``, ``a classifier``."""
    return f"{'an' if _VOWEL.match(word) else 'a'} {word}"


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _to_model_type(value: Any) -> str:
    if isinstance(value, str) and value in _MODEL_TYPES:
        return value
    raise RuntimeError(f'Unknown model type {_json_text(value)}. Use "chat", "image", or "classifier".')


def _to_provider(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError("provider must be a string")
    return value


def _to_model_info(model: Any) -> Any:
    """Catalog entry for scripts: pi-ai's JSON of the model. ``headers`` is dropped because configured headers can
    carry credentials."""
    info = dict(to_json_value(model))
    info.pop("headers", None)
    return info


async def catalog_call(name: str, args: Any, models: Models, context: Context) -> Any:
    """The catalog functions: what a script's call of ``name`` with ``args`` (the argument list) returns."""
    if name == "models.get_models_of_type":
        type_, provider = bind_arguments(args, ("type", "provider"))
        return [
            _to_model_info(model) for model in models.get_models_of_type(_to_model_type(type_), _to_provider(provider))
        ]
    if name == "models.get_available_of_type":
        type_, provider = bind_arguments(args, ("type", "provider"))
        available = await models.get_available_of_type(
            _to_model_type(type_), _to_provider(provider), signal=context.abort_signal
        )
        return [_to_model_info(model) for model in available]
    type_, provider, id = bind_arguments(args, ("type", "provider", "id"))
    if not isinstance(provider, str) or not isinstance(id, str):
        given = list(args) if isinstance(args, list) else []  # pyright: ignore[reportUnknownArgumentType]
        raise RuntimeError(
            "models.get_model_of_type(type, provider, id) expects three strings, got "
            f"({', '.join(describe_value(value) for value in given)}). The provider and the id are separate arguments, "
            'for example models.get_model_of_type("classifier", "typesafe", "jev-latest").'
        )
    model = models.get_model_of_type(_to_model_type(type_), provider, id)
    return None if model is None else _to_model_info(model)


_CLASSIFIER_CONTEXT_SHAPE: Final = (
    '{"state": {...}, "images": [{"type": "image", "data": <base64>, "mimeType": ...}] (optional), "questions": '
    '{<id>: {"type": "choice", "instructions": ..., "criteria": {<label>: <meaning>}} | {"type": "score", '
    '"instructions": ..., "criteria": [<lowest level>, ..., <highest level>]} | {"type": "bool", "instructions": '
    '..., "criteria": {"true": <meaning>, "false": <meaning>}}}}'
)


def _is_image_block(value: Any) -> bool:
    return (
        _is_record(value)
        and value.get("type") == "image"
        and isinstance(value.get("data"), str)
        and isinstance(value.get("mimeType"), str)
    )


def _check_classifier_context(context: Any) -> ClassifierContext:
    """Check a script's classifier context, so mistakes fail with the expected shape instead of a provider error."""

    def fail(problem: str) -> RuntimeError:
        return RuntimeError(f"models.classify() {problem}. Expected context: {_CLASSIFIER_CONTEXT_SHAPE}.")

    if not _is_record(context):
        raise fail(f"expects a context dict as its second argument, got {describe_value(context)}")
    if not _is_record(context.get("state")):
        raise fail(f'context["state"] must be a dict, got {describe_value(context.get("state"))}')
    images = context.get("images")
    if images is not None:
        if not isinstance(images, list):
            raise fail(f'context["images"] must be a list, got {describe_value(images)}')
        for index, image in enumerate(cast("list[Any]", images)):
            if not _is_image_block(image):
                raise fail(f'context["images"][{index}] must be an image block, got {describe_value(image)}')
    questions = context.get("questions")
    if not _is_record(questions) or len(questions) == 0:
        raise fail(f'context["questions"] must map question IDs to questions, got {describe_value(questions)}')

    def is_strings(values: list[Any]) -> bool:
        return len(values) > 0 and all(isinstance(value, str) for value in values)

    for id, question in cast("dict[str, Any]", questions).items():
        at = f'context["questions"][{_json_text(id)}]'
        if not _is_record(question):
            raise fail(f"{at} must be a question dict, got {describe_value(question)}")
        if not isinstance(question.get("instructions"), str):
            raise fail(f'{at}["instructions"] must be a string')
        criteria = question.get("criteria")
        kind = question.get("type")
        if kind == "choice":
            if not _is_record(criteria) or not is_strings(list(criteria.values())):
                raise fail(f'{at} is a "choice" question, so criteria must map each label to its meaning')
        elif kind == "score":
            if not isinstance(criteria, list) or not is_strings(cast("list[Any]", criteria)):
                raise fail(f'{at} is a "score" question, so criteria must list the levels as strings, lowest first')
        elif kind == "bool":
            if (
                not _is_record(criteria)
                or not isinstance(criteria.get("true"), str)
                or not isinstance(criteria.get("false"), str)
            ):
                raise fail(f'{at} is a "bool" question, so criteria must be {{"true": str, "false": str}}')
        else:
            raise fail(f'{at}["type"] must be "choice", "score", or "bool", got {_json_text(kind)}')
    return from_dict(ClassifierContext, context)


def _check_images_context(context: Any) -> ImagesContext:
    """Check a script's image context, so mistakes such as ``{"prompt": ...}`` fail with the expected shape."""

    def fail(problem: str) -> RuntimeError:
        return RuntimeError(
            f"models.generate_images() {problem}. Expected context: "
            '{"input": [{"type": "text", "text": <prompt>}, ...optional {"type": "image", "data": <base64>, '
            '"mimeType": ...} references]}.'
        )

    if not _is_record(context):
        raise fail(f"expects a context dict as its second argument, got {describe_value(context)}")
    blocks = context.get("input")
    if not isinstance(blocks, list) or len(cast("list[Any]", blocks)) == 0:
        raise fail(f'context["input"] must be a non-empty list of blocks, got {describe_value(blocks)}')
    for index, block in enumerate(cast("list[Any]", blocks)):
        if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            continue
        if _is_image_block(block):
            continue
        raise fail(f'context["input"][{index}] must be a text or image block, got {describe_value(block)}')
    return from_dict(ImagesContext, context)


def _resolve_model(name: str, type: Literal["classifier", "image"], model: Any, models: Models) -> Any:
    """Resolve the script's model by provider and id only, so a script-supplied ``baseUrl`` or ``headers`` never
    receive the credentials, with errors that say what to do."""
    list_hint = f'List the {type} models you can use with models.get_available_of_type("{type}").'
    if not _is_record(model) or not isinstance(model.get("provider"), str) or not isinstance(model.get("id"), str):
        none_hint = " models.get_model_of_type() returns None for an unknown provider or id." if model is None else ""
        raise RuntimeError(
            f"{name}() expects {_with_article(type)} model as its first argument, got "
            f"{describe_value(model)}.{none_hint} {list_hint}"
        )
    provider: str = model["provider"]
    id: str = model["id"]
    resolved = models.get_model_of_type(type, provider, id)
    if resolved is not None:
        return resolved
    actual = next(
        (
            other
            for other in _MODEL_TYPES
            if other != type and models.get_model_of_type(other, provider, id) is not None
        ),
        None,
    )
    raise RuntimeError(
        f'"{provider}/{id}" is {_with_article(actual)} model, not {_with_article(type)} model. {list_hint}'
        if actual is not None
        else f'Unknown {type} model "{provider}/{id}". {list_hint}'
    )


_MODEL_CALL_PARAMETERS: Final = Type.Object({"model": Type.Unknown(), "context": Type.Unknown()})


def _model_result(label: str, result: Any) -> ToolExecutionResult:
    """The result of a model call as the tool returns it: the whole result for the script (pi-ai's JSON), usage
    counted by the Harness."""
    ok = result.stop_reason == "stop"
    text = f"{label} done" if ok else f"{label} {result.stop_reason}: {result.error_message or ''}"
    tool: dict[str, Any] = {"output": [TextContent(text)], "structuredOutput": to_json_value(result)}
    if not ok:
        tool["isError"] = True
    if result.usage is not None:
        tool["usage"] = result.usage
    return cast("ToolExecutionResult", tool)


def create_model_tools() -> list[ToolRegistration]:
    """The internal tools behind ``models.classify()`` and ``models.generate_images()``."""

    async def classify(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        model = _resolve_model("models.classify", "classifier", args.get("model"), api.models)
        checked = _check_classifier_context(args.get("context"))
        result = await api.models.classify(model, checked, signal=context.abort_signal)
        return _model_result(f"classify {model.provider}/{model.id}", result)

    async def generate_images(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        model = _resolve_model("models.generate_images", "image", args.get("model"), api.models)
        checked = _check_images_context(args.get("context"))
        result = await api.models.generate_images(model, checked, signal=context.abort_signal)
        return _model_result(f"generateImages {model.provider}/{model.id}", result)

    return [
        define_tool(
            name=CLASSIFY_TOOL,
            description="models.classify() of code mode scripts.",
            parameters=_MODEL_CALL_PARAMETERS,
            structured_output_schema=Type.Unknown(),
            callers=["tools"],
            execute=classify,
        ),
        define_tool(
            name=GENERATE_IMAGES_TOOL,
            description="models.generate_images() of code mode scripts.",
            parameters=_MODEL_CALL_PARAMETERS,
            structured_output_schema=Type.Unknown(),
            callers=["tools"],
            execute=generate_images,
        ),
    ]
