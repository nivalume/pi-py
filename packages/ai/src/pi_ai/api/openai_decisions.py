"""OpenAI's Decisions API: ``POST /v1/decisions`` with ``{model, input, questions}``.

https://developers.openai.com/api/docs/guides/decisions

The state is sent as JSON text. With images, the input becomes one user message
with the state as ``input_text`` followed by ``input_image`` data URLs. Questions
map to Decisions types: ``choice`` to ``choice``, ``score`` to ``score``, and
``bool`` to ``predicate``. Predicates have no criteria field, so the meanings of
true and false are appended to the instructions.

Only OpenAI API keys work: Sign in with ChatGPT tokens are rejected on this route.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

from ..types import (
    ClassifierAnswer,
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
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

__all__ = ["classify"]

LABEL = "OpenAI Decisions"

#: The endpoint accepts at most this many image parts per request.
MAX_IMAGES = 128

#: Cloudflare in front of api.openai.com answers 504 with an HTML page when a request runs
#: longer than about five seconds. Large inputs, currently above roughly 600K tokens, hit
#: this limit, and retrying the same input runs into it again, so 504 is not retried.
NO_RETRY_STATUSES = (504,)


def _predicate_instructions(question: ClassifierBoolQuestion) -> str:
    meanings = [
        f"True means: {question.criteria['true']}" if question.criteria.get("true") else "",
        f"False means: {question.criteria['false']}" if question.criteria.get("false") else "",
    ]
    meanings = [meaning for meaning in meanings if meaning]
    return question.instructions + "\n\n" + "\n".join(meanings) if meanings else question.instructions


def _wire_question(name: str, question: ClassifierQuestion) -> dict[str, Any]:
    if question.type == "choice":
        return {
            "type": "choice",
            "name": name,
            "instructions": question.instructions,
            "choices": [
                {"value": value, "description": description} if description else {"value": value}
                for value, description in question.criteria.items()
            ],
        }
    if question.type == "score":
        return {
            "type": "score",
            "name": name,
            "instructions": question.instructions,
            "levels": [{"label": label} for label in question.criteria],
        }
    return {"type": "predicate", "name": name, "instructions": _predicate_instructions(question)}


def _wire_input(context: ClassifierContext) -> Any:
    state = json.dumps(context.state, ensure_ascii=False, separators=(",", ":"))
    images = context.images or []
    if not images:
        return state
    if len(images) > MAX_IMAGES:
        raise ValueError(f"{LABEL} accepts at most {MAX_IMAGES} images, got {len(images)}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": state},
                *(
                    {"type": "input_image", "image_url": f"data:{image.mime_type};base64,{image.data}"}
                    for image in images
                ),
            ],
        }
    ]


def _choice_probabilities(value: Any, question_id: str) -> dict[str, float]:
    if not isinstance(value, list):
        raise ValueError(f"{LABEL} returned invalid probabilities for {question_id}")
    probabilities: dict[str, float] = {}
    for entry in value:
        if not is_record(entry) or not isinstance(entry.get("value"), str):
            raise ValueError(f"{LABEL} returned invalid probabilities for {question_id}")
        choice = entry["value"]
        probabilities[choice] = required_number(
            LABEL, entry.get("probability"), f"probability for {question_id}.{choice}"
        )
    return probabilities


def _parse_answer(question_id: str, question: ClassifierQuestion, answer: Mapping[str, Any]) -> ClassifierAnswer:
    kind = answer.get("type")
    if kind == "refusal":
        raise ValueError(f"{LABEL} refused to answer {question_id}")
    if question.type == "choice":
        choice = answer.get("choice")
        if kind != "choice" or not isinstance(choice, str):
            raise ValueError(f"{LABEL} did not return a choice answer for {question_id}")
        return ClassifierChoiceAnswer(
            choice=choice,
            probabilities=_choice_probabilities(answer.get("probabilities"), question_id),
            confidence=required_number(LABEL, answer.get("confidence"), f"confidence for {question_id}"),
        )
    if question.type == "score":
        if kind != "score":
            raise ValueError(f"{LABEL} did not return a score answer for {question_id}")
        return ClassifierScoreAnswer(
            score=required_number(LABEL, answer.get("score"), f"score for {question_id}"),
            confidence=required_number(LABEL, answer.get("confidence"), f"confidence for {question_id}"),
        )
    if kind != "predicate":
        raise ValueError(f"{LABEL} did not return a predicate answer for {question_id}")
    return ClassifierBoolAnswer(
        probability=required_number(LABEL, answer.get("probability"), f"probability for {question_id}")
    )


def _parse_answers(value: Any, questions: Mapping[str, ClassifierQuestion]) -> dict[str, ClassifierAnswer]:
    if not isinstance(value, list):
        raise ValueError(f"{LABEL} returned an unexpected response")
    by_name: dict[str, Mapping[str, Any]] = {}
    for answer in value:
        if is_record(answer) and isinstance(answer.get("name"), str):
            by_name[answer["name"]] = answer
    answers: dict[str, ClassifierAnswer] = {}
    for question_id, question in questions.items():
        answer = by_name.get(question_id)
        if answer is None:
            raise ValueError(f"{LABEL} did not return an answer for {question_id}")
        answers[question_id] = _parse_answer(question_id, question, answer)
    return answers


def _error_message(error: Exception) -> str:
    if getattr(error, "status", None) == 504:
        return (
            f"{LABEL} error (504): the request timed out at the gateway. Very large inputs "
            "(above roughly 600K tokens) currently exceed its time limit."
        )
    return format_provider_error(normalize_provider_error(error), f"{LABEL} error")


async def classify(
    model: ClassifierModel, context: ClassifierContext, options: Mapping[str, Any] | None = None
) -> ClassifierResult:
    """Classification through OpenAI's Decisions API. Never raises."""
    options = options or {}
    output = ClassifierResult(
        api=model.api, provider=model.provider, model=model.id, answers={}, stop_reason="stop", timestamp=now_ms()
    )
    try:
        if model.api != "openai-decisions":
            raise ValueError(f"Unsupported classifier API: {model.api}")
        questions = classifier_questions(context)
        body = await post_classifier_request(
            LABEL,
            urljoin(f"{model.base_url.rstrip('/')}/", "decisions"),
            model,
            {
                "model": model.id,
                "input": _wire_input(context),
                "questions": [_wire_question(name, question) for name, question in questions.items()],
            },
            options,
            NO_RETRY_STATUSES,
        )
        if not is_record(body):
            raise ValueError(f"{LABEL} returned an unexpected response")
        # Set before parsing answers: a request with malformed or refused answers was still billed.
        usage = parse_classifier_usage(body.get("usage"), model)
        if usage is not None:
            output.usage = usage
        output.answers = _parse_answers(body.get("answers"), questions)
        return output
    except Exception as error:
        signal = options.get("signal")
        failed = classifier_error_result(model, _error_message(error), aborted=signal is not None and signal.aborted)
        failed.usage = output.usage
        failed.timestamp = output.timestamp
        return failed
