"""Model-type narrowing and operation result helpers."""

from __future__ import annotations

from typing import Any

from ..types import AssistantImages, ClassifierContext, ClassifierModel, ClassifierResult, ImageModel, now_ms
from .diagnostics import format_thrown_value
from .models_error import ModelsError

__all__ = [
    "get_model_type",
    "is_model_type",
    "assert_chat_model",
    "assert_image_model",
    "assert_classifier_model",
    "assert_classifier_input_supported",
    "image_error_result",
    "classifier_error_result",
]


def get_model_type(model: Any) -> str:
    """The type of a model. Models without ``type`` are chat models."""
    return getattr(model, "type", None) or "chat"


def is_model_type(model: Any, type: str) -> bool:
    return get_model_type(model) == type


def assert_chat_model(model: Any) -> None:
    if not is_model_type(model, "chat"):
        raise ModelsError("provider", f"Model {model.provider}/{model.id} is not a chat model")


def assert_image_model(model: Any) -> None:
    if not is_model_type(model, "image"):
        raise ModelsError("provider", f"Model {model.provider}/{model.id} is not an image model")


def assert_classifier_model(model: Any) -> None:
    if not is_model_type(model, "classifier"):
        raise ModelsError("provider", f"Model {model.provider}/{model.id} is not a classifier model")


def assert_classifier_input_supported(model: ClassifierModel, context: ClassifierContext) -> None:
    if context.images and "image" not in model.input:
        raise ModelsError("provider", f"Model {model.provider}/{model.id} does not accept image input")


def image_error_result(model: ImageModel, error: Any, aborted: bool = False) -> AssistantImages:
    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="aborted" if aborted else "error",
        error_message=format_thrown_value(error),
        timestamp=now_ms(),
    )


def classifier_error_result(model: ClassifierModel, error: Any, aborted: bool = False) -> ClassifierResult:
    return ClassifierResult(
        api=model.api,
        provider=model.provider,
        model=model.id,
        answers={},
        stop_reason="aborted" if aborted else "error",
        error_message=format_thrown_value(error),
        timestamp=now_ms(),
    )
