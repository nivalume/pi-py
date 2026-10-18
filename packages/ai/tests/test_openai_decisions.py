"""Offline tests for OpenAI's Decisions API (port of upstream ``test/openai-decisions.test.ts``)."""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from pi_ai.api.openai_decisions import classify
from pi_ai.types import (
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierChoiceAnswer,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierScoreAnswer,
    ClassifierScoreQuestion,
    ImageContent,
    ModelCost,
    ModelCostTier,
)

MODEL = ClassifierModel(
    id="gpt-6-luna",
    name="GPT-6 Luna",
    api="openai-decisions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    input=["text", "image"],
    cost=ModelCost(input=0.1, tiers=[ModelCostTier(input_tokens_above=272000, input=0.2)]),
    context_window=922000,
)

CONTEXT = ClassifierContext(
    state={"text": "The deployment succeeded, thank you."},
    questions={
        "category": ClassifierChoiceQuestion("Classify the message", {"success": "Successful", "failure": ""}),
        "satisfaction": ClassifierScoreQuestion("Score satisfaction", ["low", "neutral", "high"]),
        "approved": ClassifierBoolQuestion("Does the user approve?", {"true": "Approval", "false": "No approval"}),
    },
)

#: Response shape from the API reference and live ``gpt-6-luna`` requests.
WIRE_ANSWERS = [
    {
        "type": "choice",
        "name": "category",
        "choice": "success",
        "probabilities": [{"value": "success", "probability": 0.9}, {"value": "failure", "probability": 0.1}],
        "confidence": 0.8,
    },
    {
        "type": "score",
        "name": "satisfaction",
        "score": 1.8,
        "probabilities": [
            {"value": 0, "label": "low", "probability": 0.05},
            {"value": 1, "label": "neutral", "probability": 0.1},
            {"value": 2, "label": "high", "probability": 0.85},
        ],
        "confidence": 0.7,
    },
    {"type": "predicate", "name": "approved", "probability": 0.95},
]

WIRE_USAGE = {
    "input_tokens": 164,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 164,
}

IMAGE = ImageContent("aW1hZ2U=", "image/png")
STATE_JSON = json.dumps(CONTEXT.state, separators=(",", ":"))


async def test_maps_questions_to_decisions_types_and_answers_back_by_name(mock_server):
    # Answers out of question order: they are matched by name.
    mock_server.add_json({"model": "gpt-6-luna", "answers": list(reversed(WIRE_ANSWERS)), "usage": WIRE_USAGE})

    result = await classify(
        MODEL, CONTEXT, {"api_key": "secret", "temperature": 1.5, "http_client": mock_server.client()}
    )

    assert len(mock_server.requests) == 1
    request = mock_server.last
    assert request.url == "https://api.openai.com/v1/decisions"
    assert request.headers["authorization"] == "Bearer secret"
    assert request.json() == {
        "model": "gpt-6-luna",
        "input": STATE_JSON,
        "questions": [
            {
                "type": "choice",
                "name": "category",
                "instructions": "Classify the message",
                # Empty descriptions are omitted.
                "choices": [{"value": "success", "description": "Successful"}, {"value": "failure"}],
            },
            {
                "type": "score",
                "name": "satisfaction",
                "instructions": "Score satisfaction",
                "levels": [{"label": "low"}, {"label": "neutral"}, {"label": "high"}],
            },
            {
                "type": "predicate",
                "name": "approved",
                "instructions": "Does the user approve?\n\nTrue means: Approval\nFalse means: No approval",
            },
        ],
    }
    assert result.stop_reason == "stop"
    assert result.answers == {
        "category": ClassifierChoiceAnswer("success", {"success": 0.9, "failure": 0.1}, 0.8),
        "satisfaction": ClassifierScoreAnswer(1.8, 0.7),
        "approved": ClassifierBoolAnswer(0.95),
    }
    assert result.usage is not None
    assert (result.usage.input, result.usage.output, result.usage.cache_read, result.usage.total_tokens) == (
        164,
        0,
        0,
        164,
    )
    assert result.usage.cost.total == pytest.approx(0.0000164, abs=1e-12)


async def test_prices_long_context_requests_at_the_long_context_input_rate(mock_server):
    mock_server.add_json({"answers": WIRE_ANSWERS, "usage": {"input_tokens": 300000, "output_tokens": 0}})

    result = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert result.usage is not None
    assert result.usage.cost.total == pytest.approx(0.06, abs=1e-12)


async def test_sends_images_after_the_state_in_one_user_message(mock_server):
    mock_server.add_json({"answers": WIRE_ANSWERS})

    result = await classify(
        MODEL,
        dataclasses.replace(CONTEXT, images=[IMAGE, ImageContent(IMAGE.data, "image/jpeg")]),
        {"api_key": "secret", "http_client": mock_server.client()},
    )

    assert result.stop_reason == "stop"
    assert mock_server.last.json()["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": STATE_JSON},
                {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,aW1hZ2U="},
            ],
        }
    ]


async def test_rejects_more_than_128_images_before_sending(mock_server):
    result = await classify(
        MODEL,
        dataclasses.replace(CONTEXT, images=[IMAGE] * 129),
        {"api_key": "secret", "http_client": mock_server.client()},
    )

    assert mock_server.requests == []
    assert result.stop_reason == "error"
    assert "at most 128 images, got 129" in (result.error_message or "")


async def test_fails_the_result_when_a_question_is_refused_and_keeps_the_billed_usage(mock_server):
    mock_server.add_json(
        {"answers": [WIRE_ANSWERS[0], WIRE_ANSWERS[1], {"type": "refusal", "name": "approved"}], "usage": WIRE_USAGE}
    )

    result = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert result.stop_reason == "error"
    assert result.answers == {}
    assert result.error_message == "OpenAI Decisions refused to answer approved"
    assert result.usage is not None
    assert result.usage.input == 164


async def test_returns_missing_and_mistyped_answers_as_classifier_errors(mock_server):
    mock_server.add_json({"answers": WIRE_ANSWERS[:2]})
    mock_server.add_json({"answers": [WIRE_ANSWERS[0], WIRE_ANSWERS[1], {"type": "score", "name": "approved"}]})

    missing = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})
    mistyped = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert missing.stop_reason == "error"
    assert "did not return an answer for approved" in (missing.error_message or "")
    assert mistyped.stop_reason == "error"
    assert "did not return a predicate answer for approved" in (mistyped.error_message or "")


async def test_preserves_prototype_sensitive_question_ids_in_answers(mock_server):
    mock_server.add_json({"answers": [{"type": "predicate", "name": "__proto__", "probability": 0.75}]})
    context = ClassifierContext(
        state={}, questions={"__proto__": ClassifierBoolQuestion("Is this true?", {"true": "Yes", "false": "No"})}
    )

    result = await classify(MODEL, context, {"api_key": "secret", "http_client": mock_server.client()})

    assert result.stop_reason == "stop"
    assert result.answers["__proto__"] == ClassifierBoolAnswer(0.75)


async def test_does_not_retry_gateway_timeouts_and_explains_them_instead_of_returning_the_html_page(mock_server):
    for _ in range(3):
        mock_server.add(
            httpx.Response(504, text="<!DOCTYPE html><html>Gateway time-out</html>", headers={"retry-after-ms": "0"})
        )

    # Default retries: the same input would time out again.
    result = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert len(mock_server.requests) == 1
    assert result.stop_reason == "error"
    assert "OpenAI Decisions error (504): the request timed out at the gateway" in (result.error_message or "")
    assert "<html>" not in (result.error_message or "")


async def test_still_retries_other_server_errors(mock_server):
    mock_server.add(httpx.Response(503, text="busy", headers={"retry-after-ms": "0"}))
    mock_server.add_json({"answers": WIRE_ANSWERS})

    result = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert len(mock_server.requests) == 2
    assert result.stop_reason == "stop"


async def test_includes_the_api_error_body_for_other_http_failures(mock_server):
    mock_server.add_json(
        {"error": {"message": "Decision input exceeds the token limit.", "type": "invalid_request_error"}}, status=400
    )

    result = await classify(
        MODEL, CONTEXT, {"api_key": "secret", "max_retries": 0, "http_client": mock_server.client()}
    )

    assert result.stop_reason == "error"
    assert "OpenAI Decisions error (400)" in (result.error_message or "")
    assert "Decision input exceeds the token limit." in (result.error_message or "")


async def test_rejects_models_for_other_classifier_apis_and_missing_api_keys(mock_server):
    other_api = await classify(
        dataclasses.replace(MODEL, api="typesafe-system-one"),
        CONTEXT,
        {"api_key": "secret", "http_client": mock_server.client()},
    )
    no_key = await classify(MODEL, CONTEXT, {"http_client": mock_server.client()})

    assert mock_server.requests == []
    assert "Unsupported classifier API: typesafe-system-one" in (other_api.error_message or "")
    assert "No API key for provider: openai" in (no_key.error_message or "")
