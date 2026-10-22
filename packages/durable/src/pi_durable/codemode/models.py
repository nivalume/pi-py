"""``models.*`` for code mode scripts (port of ``codemode/models.ts``): the model catalog, and classifier and
image model calls.

The catalog functions are host globals, journaled like any global. ``models.classify()`` and
``models.generateImages()`` are paid effects, so each runs as an internal tool (``callers=["tools"]``, a reserved
name, hidden from scripts' tool listings): a nested durable call with its own task, live row, abort, usage
accounting, and replay policy. The code mode tool turns a script's call of the global into a call of that tool.

The globals are named in the script's language (:mod:`.dialect`): upstream's camelCase names for JavaScript, snake_case
for Python (``models.get_models_of_type``, ``models.generate_images``, ...); so are the error messages. The internal
tools keep upstream's names (``pi.codemode.models.classify``, ``pi.codemode.models.generateImages``). Catalog
entries, contexts, and results are pi-ai's JSON (camelCase keys: ``mimeType``, ``stopReason``, ``contextWindow``), as
upstream's scripts see them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pi_ai import ClassifierContext, ImagesContext, TextContent, Type
from pi_ai.serialization import from_dict, to_json_value

from ..harness.define import define_tool
from .dialect import MAX_CONCURRENT_MODEL_CALLS, Dialect, with_article

if TYPE_CHECKING:
    from pi_ai import Models
    from pi_chord.context import Context

    from ..harness.types import ToolExecutionApi, ToolExecutionResult, ToolRegistration

__all__ = [
    "CLASSIFY_TOOL",
    "CODEMODE_INTERNAL_PREFIX",
    "GENERATE_IMAGES_TOOL",
    "MAX_CONCURRENT_MODEL_CALLS",
    "catalog_call",
    "create_model_tools",
    "model_calls",
]

#: Reserved names of code mode's internal tools; they are never listed to scripts or the model.
CODEMODE_INTERNAL_PREFIX: Final = "pi.codemode."
CLASSIFY_TOOL: Final = f"{CODEMODE_INTERNAL_PREFIX}models.classify"
GENERATE_IMAGES_TOOL: Final = f"{CODEMODE_INTERNAL_PREFIX}models.generateImages"

_MODEL_TYPES: Final = ("chat", "image", "classifier")


def model_calls(dialect: Dialect) -> Mapping[str, str]:
    """The internal tool, by the global a script calls."""
    return {dialect.classify: CLASSIFY_TOOL, dialect.generate_images: GENERATE_IMAGES_TOOL}


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _to_model_type(value: Any, dialect: Dialect) -> str:
    if isinstance(value, str) and value in _MODEL_TYPES:
        return value
    shown = dialect.missing if value is None and dialect.language == "javascript" else dialect.json_text(value)
    raise RuntimeError(f'Unknown model type {shown}. Use "chat", "image", or "classifier".')


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


async def catalog_call(name: str, args: Any, models: Models, context: Context, dialect: Dialect) -> Any:
    """The catalog functions: what a script's call of ``name`` with ``args`` (the argument list) returns."""
    if name == dialect.get_models_of_type:
        type_, provider = dialect.bind(args, ("type", "provider"))
        return [
            _to_model_info(model)
            for model in models.get_models_of_type(_to_model_type(type_, dialect), _to_provider(provider))
        ]
    if name == dialect.get_available_of_type:
        type_, provider = dialect.bind(args, ("type", "provider"))
        available = await models.get_available_of_type(
            _to_model_type(type_, dialect), _to_provider(provider), signal=context.abort_signal
        )
        return [_to_model_info(model) for model in available]
    type_, provider, id = dialect.bind(args, ("type", "provider", "id"))
    if not isinstance(provider, str) or not isinstance(id, str):
        given = list(args) if isinstance(args, list) else []  # pyright: ignore[reportUnknownArgumentType]
        call = dialect.get_model_of_type
        raise RuntimeError(
            f"{call}(type, provider, id) expects three strings, got "
            f"({', '.join(dialect.describe_value(value) for value in given)}). The provider and the id are separate "
            f'arguments, for example {call}("classifier", "typesafe", "jev-latest").'
        )
    model = models.get_model_of_type(_to_model_type(type_, dialect), provider, id)
    return None if model is None else _to_model_info(model)


def _describe_field(record: dict[str, Any], key: str, dialect: Dialect) -> str:
    """A context field in an error message; in JavaScript a missing one is ``undefined``, not ``null``."""
    if dialect.language == "javascript" and key not in record:
        return dialect.missing
    return dialect.describe_value(record.get(key))


def _is_image_block(value: Any) -> bool:
    return (
        _is_record(value)
        and value.get("type") == "image"
        and isinstance(value.get("data"), str)
        and isinstance(value.get("mimeType"), str)
    )


def _check_classifier_context(context: Any, dialect: Dialect) -> ClassifierContext:
    """Check a script's classifier context, so mistakes fail with the expected shape instead of a provider error."""
    describe, field = dialect.describe_value, dialect.field

    def fail(problem: str) -> RuntimeError:
        return RuntimeError(f"{dialect.classify}() {problem}. Expected context: {dialect.classifier_context_shape}.")

    if not _is_record(context):
        raise fail(f"expects a context {dialect.object_noun} as its second argument, got {describe(context)}")
    if not _is_record(context.get("state")):
        raise fail(
            f"{field('context', 'state')} must be {with_article(dialect.object_noun)}, got "
            f"{_describe_field(context, 'state', dialect)}"
        )
    images = context.get("images")
    if images is not None:
        if not isinstance(images, list):
            raise fail(
                f"{field('context', 'images')} must be {with_article(dialect.array_noun)}, got {describe(images)}"
            )
        for index, image in enumerate(cast("list[Any]", images)):
            if not _is_image_block(image):
                raise fail(f"{field('context', 'images')}[{index}] must be an image block, got {describe(image)}")
    questions = context.get("questions")
    if not _is_record(questions) or len(questions) == 0:
        raise fail(
            f"{field('context', 'questions')} must map question IDs to questions, got "
            f"{_describe_field(context, 'questions', dialect)}"
        )

    def is_strings(values: list[Any]) -> bool:
        return len(values) > 0 and all(isinstance(value, str) for value in values)

    for id, question in cast("dict[str, Any]", questions).items():
        at = dialect.entry(field("context", "questions"), id)
        if not _is_record(question):
            raise fail(f"{at} must be a question {dialect.object_noun}, got {describe(question)}")
        if not isinstance(question.get("instructions"), str):
            raise fail(f"{field(at, 'instructions')} must be a string")
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
                raise fail(f'{at} is a "bool" question, so criteria must be {dialect.bool_criteria}')
        else:
            shown = (
                dialect.missing
                if "type" not in question and dialect.language == "javascript"
                else dialect.json_text(kind)
            )
            raise fail(f'{field(at, "type")} must be "choice", "score", or "bool", got {shown}')
    return from_dict(ClassifierContext, context)


def _check_images_context(context: Any, dialect: Dialect) -> ImagesContext:
    """Check a script's image context, so mistakes such as a ``prompt`` field fail with the expected shape."""
    describe, field = dialect.describe_value, dialect.field

    def fail(problem: str) -> RuntimeError:
        return RuntimeError(f"{dialect.generate_images}() {problem}. Expected context: {dialect.images_context_shape}.")

    if not _is_record(context):
        raise fail(f"expects a context {dialect.object_noun} as its second argument, got {describe(context)}")
    blocks = context.get("input")
    if not isinstance(blocks, list) or len(cast("list[Any]", blocks)) == 0:
        raise fail(
            f"{field('context', 'input')} must be a non-empty {dialect.array_noun} of blocks, got "
            f"{_describe_field(context, 'input', dialect)}"
        )
    for index, block in enumerate(cast("list[Any]", blocks)):
        if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            continue
        if _is_image_block(block):
            continue
        raise fail(f"{field('context', 'input')}[{index}] must be a text or image block, got {describe(block)}")
    return from_dict(ImagesContext, context)


def _resolve_model(
    name: str, type: Literal["classifier", "image"], model: Any, models: Models, dialect: Dialect
) -> Any:
    """Resolve the script's model by provider and id only, so a script-supplied ``baseUrl`` or ``headers`` never
    receive the credentials, with errors that say what to do."""
    list_hint = f'List the {type} models you can use with {dialect.get_available_of_type}("{type}").'
    if not _is_record(model) or not isinstance(model.get("provider"), str) or not isinstance(model.get("id"), str):
        none_hint = (
            f" {dialect.get_model_of_type}() returns {dialect.missing} for an unknown provider or id."
            if model is None
            else ""
        )
        raise RuntimeError(
            f"{name}() expects {with_article(type)} model as its first argument, got "
            f"{dialect.describe_value(model)}.{none_hint} {list_hint}"
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
        f'"{provider}/{id}" is {with_article(actual)} model, not {with_article(type)} model. {list_hint}'
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


def create_model_tools(dialect: Dialect) -> list[ToolRegistration]:
    """The internal tools behind ``models.classify()`` and ``models.generateImages()`` (``generate_images()``)."""

    async def classify(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        model = _resolve_model(dialect.classify, "classifier", args.get("model"), api.models, dialect)
        checked = _check_classifier_context(args.get("context"), dialect)
        result = await api.models.classify(model, checked, signal=context.abort_signal)
        return _model_result(f"classify {model.provider}/{model.id}", result)

    async def generate_images(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        model = _resolve_model(dialect.generate_images, "image", args.get("model"), api.models, dialect)
        checked = _check_images_context(args.get("context"), dialect)
        result = await api.models.generate_images(model, checked, signal=context.abort_signal)
        return _model_result(f"generateImages {model.provider}/{model.id}", result)

    return [
        define_tool(
            name=CLASSIFY_TOOL,
            description=f"{dialect.classify}() of code mode scripts.",
            parameters=_MODEL_CALL_PARAMETERS,
            structured_output_schema=Type.Unknown(),
            callers=["tools"],
            execute=classify,
        ),
        define_tool(
            name=GENERATE_IMAGES_TOOL,
            description=f"{dialect.generate_images}() of code mode scripts.",
            parameters=_MODEL_CALL_PARAMETERS,
            structured_output_schema=Type.Unknown(),
            callers=["tools"],
            execute=generate_images,
        ),
    ]
