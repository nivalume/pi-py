"""TypeSafe System One protocol shared by the services that serve System One models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from ..serialization import to_json_value
from ..types import (
    ClassifierAnswer,
    ClassifierBoolAnswer,
    ClassifierChoiceAnswer,
    ClassifierContext,
    ClassifierModel,
    ClassifierQuestion,
    ClassifierResult,
    ClassifierScoreAnswer,
    now_ms,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.model_operations import classifier_error_result
from .classifier_shared import (
    classifier_questions,
    is_record,
    parse_classifier_usage,
    post_classifier_request,
    required_number,
)

__all__ = ["SystemOneWireRequest", "SystemOneTransport", "wire_request", "classify_system_one"]


class SystemOneWireRequest(TypedDict):
    """TypeSafe System One request body without the transport-specific envelope."""

    state: dict[str, Any]
    questions: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SystemOneTransport:
    """Differences between services that serve System One models."""

    #: Classifier API implemented by this transport.
    api: str
    #: Service name used in error messages.
    label: str
    #: Absolute request URL.
    url: Callable[[ClassifierModel], str]
    #: Wraps the System One request in the service's request envelope.
    payload: Callable[[ClassifierModel, SystemOneWireRequest], Any]
    #: Extracts the System One output (``{answers, usage}``) from the service's response envelope.
    output: Callable[[Any], Mapping[str, Any]]


def _probabilities(label: str, value: Any, question_id: str) -> dict[str, float]:
    if not is_record(value):
        raise ValueError(f"{label} returned invalid probabilities for {question_id}")
    return {
        key: required_number(label, probability, f"probability for {question_id}.{key}")
        for key, probability in value.items()
    }


def _parse_answers(label: str, value: Any, questions: Mapping[str, ClassifierQuestion]) -> dict[str, ClassifierAnswer]:
    if not is_record(value):
        raise ValueError(f"{label} returned an unexpected response")
    answers: dict[str, ClassifierAnswer] = {}
    for question_id, question in questions.items():
        answer = value.get(question_id)
        if not is_record(answer):
            raise ValueError(f"{label} did not return an answer for {question_id}")
        if question.type == "choice":
            choice = answer.get("choice")
            if answer.get("type") != "choice" or not isinstance(choice, str):
                raise ValueError(f"{label} did not return a choice answer for {question_id}")
            answers[question_id] = ClassifierChoiceAnswer(
                choice=choice,
                probabilities=_probabilities(label, answer.get("probabilities"), question_id),
                confidence=required_number(label, answer.get("confidence"), f"confidence for {question_id}"),
            )
        elif question.type == "score":
            if answer.get("type") != "score":
                raise ValueError(f"{label} did not return a score answer for {question_id}")
            answers[question_id] = ClassifierScoreAnswer(
                score=required_number(label, answer.get("score"), f"score for {question_id}"),
                confidence=required_number(label, answer.get("confidence"), f"confidence for {question_id}"),
            )
        else:
            if answer.get("type") != "noul":
                raise ValueError(f"{label} did not return a bool answer for {question_id}")
            answers[question_id] = ClassifierBoolAnswer(
                probability=required_number(label, answer.get("noul"), f"probability for {question_id}")
            )
    return answers


def wire_request(state: dict[str, Any], questions: Mapping[str, ClassifierQuestion]) -> SystemOneWireRequest:
    """Map public ``bool`` questions to TypeSafe's wire-level ``noul`` type."""
    wire_questions: dict[str, Any] = {}
    for question_id, question in questions.items():
        encoded = to_json_value(question)
        wire_questions[question_id] = {**encoded, "type": "noul"} if question.type == "bool" else encoded
    return {"state": state, "questions": wire_questions}


async def classify_system_one(
    transport: SystemOneTransport,
    model: ClassifierModel,
    context: ClassifierContext,
    options: Mapping[str, Any] | None,
) -> ClassifierResult:
    """Run one System One classification over ``transport``. Never raises."""
    options = options or {}
    output = ClassifierResult(
        api=model.api, provider=model.provider, model=model.id, answers={}, stop_reason="stop", timestamp=now_ms()
    )
    try:
        if model.api != transport.api:
            raise ValueError(f"Unsupported classifier API: {model.api}")
        if context.images:
            raise ValueError(f"{transport.label} does not support image input")
        questions = classifier_questions(context)
        body = await post_classifier_request(
            transport.label,
            transport.url(model),
            model,
            transport.payload(model, wire_request(context.state, questions)),
            options,
        )
        result = transport.output(body)
        # Set before parsing answers: a request with malformed answers was still billed.
        usage = parse_classifier_usage(result.get("usage"), model)
        if usage is not None:
            output.usage = usage
        output.answers = _parse_answers(transport.label, result.get("answers"), questions)
        return output
    except Exception as error:
        signal = options.get("signal")
        failed = classifier_error_result(
            model,
            format_provider_error(normalize_provider_error(error), f"{transport.label} error"),
            aborted=signal is not None and signal.aborted,
        )
        failed.usage = output.usage
        failed.timestamp = output.timestamp
        return failed
