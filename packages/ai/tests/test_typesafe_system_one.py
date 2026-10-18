"""Offline tests for the TypeSafe System One classifier (port of upstream ``test/typesafe-system-one.test.ts``)."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import httpx
import pytest

from pi_ai.abort import AbortController
from pi_ai.api.typesafe_system_one import classify
from pi_ai.serialization import to_json_value
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
)

MODEL = ClassifierModel(
    id="jev-latest",
    name="Jev",
    api="typesafe-system-one",
    provider="typesafe",
    base_url="https://api.typesafe.ai/v1/",
    input=["text"],
    cost=ModelCost(),
    context_window=64000,
)

CONTEXT = ClassifierContext(
    state={"text": "The deployment succeeded, thank you."},
    questions={
        "category": ClassifierChoiceQuestion("Classify the message", {"success": "Successful", "failure": "Failed"}),
        "satisfaction": ClassifierScoreQuestion("Score satisfaction", ["low", "neutral", "high"]),
        "approved": ClassifierBoolQuestion("Does the user approve?", {"true": "Approval", "false": "No approval"}),
    },
)

WIRE_ANSWERS = {
    "category": {
        "type": "choice",
        "choice": "success",
        "probabilities": {"success": 0.9, "failure": 0.1},
        "confidence": 0.8,
    },
    "satisfaction": {"type": "score", "score": 2, "confidence": 0.7},
    "approved": {"type": "noul", "noul": 0.95},
}


async def test_maps_public_bool_questions_and_answers_to_typesafe_noul_values(mock_server):
    mock_server.add_json({"answers": WIRE_ANSWERS})
    mock_server.add_json({"answers": WIRE_ANSWERS, "usage": {"input_tokens": 308, "output_tokens": 23}})

    result = await classify(
        MODEL, CONTEXT, {"api_key": "secret", "temperature": 1.5, "http_client": mock_server.client()}
    )
    priced = await classify(
        dataclasses.replace(MODEL, cost=ModelCost(input=0.042)),
        CONTEXT,
        {"api_key": "secret", "http_client": mock_server.client()},
    )

    request = mock_server.requests[0]
    payload = request.json()
    assert request.url == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["content-type"] == "application/json"
    assert payload == {
        "model": "jev-latest",
        "state": CONTEXT.state,
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Classify the message",
                "criteria": {"success": "Successful", "failure": "Failed"},
            },
            "satisfaction": {
                "type": "score",
                "instructions": "Score satisfaction",
                "criteria": ["low", "neutral", "high"],
            },
            "approved": {
                "type": "noul",
                "instructions": "Does the user approve?",
                "criteria": {"true": "Approval", "false": "No approval"},
            },
        },
    }
    # System One has no temperature field; the option is ignored.
    assert "temperature" not in payload

    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert result.answers["approved"] == ClassifierBoolAnswer(0.95)
    assert result.answers["category"] == ClassifierChoiceAnswer("success", {"success": 0.9, "failure": 0.1}, 0.8)
    assert result.answers["satisfaction"] == ClassifierScoreAnswer(2, 0.7)
    assert result.usage is None
    assert priced.usage is not None
    assert (priced.usage.input, priced.usage.output, priced.usage.total_tokens) == (308, 23, 331)
    assert priced.usage.cost.total == pytest.approx(0.000012936, abs=1e-12)
    assert to_json_value(result.answers["approved"]) == {"type": "bool", "probability": 0.95}


async def test_posts_openrouter_system_one_requests_to_its_typesafe_compatible_endpoint(mock_server):
    # Response shape observed from the live OpenRouter endpoint.
    mock_server.add_json(
        {
            "id": "gen-dec-1",
            "provider": "TypeSafe",
            "answers": WIRE_ANSWERS,
            "usage": {"input_tokens": 308, "output_tokens": 23, "cost": 0.000012936},
        }
    )
    open_router_model = dataclasses.replace(
        MODEL,
        id="typesafe/jev-1.13",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        cost=ModelCost(input=0.042),
    )

    result = await classify(open_router_model, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert mock_server.last.url == "https://openrouter.ai/api/v1/systemone"
    body = mock_server.last.json()
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == CONTEXT.state
    assert result.stop_reason == "stop"
    assert result.answers["approved"] == ClassifierBoolAnswer(0.95)
    # Priced from the catalog like chat usage; matches OpenRouter's reported cost.
    assert result.usage is not None
    assert result.usage.cost.total == pytest.approx(0.000012936, abs=1e-12)


async def test_rejects_models_for_other_classifier_apis(mock_server):
    result = await classify(
        dataclasses.replace(MODEL, api="cloudflare-workers-ai-system-one"),
        CONTEXT,
        {"api_key": "secret", "http_client": mock_server.client()},
    )

    assert mock_server.requests == []
    assert result.stop_reason == "error"
    assert "Unsupported classifier API: cloudflare-workers-ai-system-one" in (result.error_message or "")
    assert result.answers == {}


async def test_rejects_image_input_before_sending(mock_server):
    result = await classify(
        dataclasses.replace(MODEL, input=["text", "image"]),
        dataclasses.replace(CONTEXT, images=[ImageContent("aW1hZ2U=", "image/png")]),
        {"api_key": "secret", "http_client": mock_server.client()},
    )

    assert mock_server.requests == []
    assert result.stop_reason == "error"
    assert result.error_message == "System One API does not support image input"


async def test_requires_an_api_key(mock_server):
    result = await classify(MODEL, CONTEXT, {"http_client": mock_server.client()})

    assert mock_server.requests == []
    assert result.error_message == "No API key for provider: typesafe"


async def test_merges_headers_case_insensitively_and_supports_null_suppression(mock_server):
    mock_server.add_json({"answers": WIRE_ANSWERS})
    mock_server.add_json({"answers": WIRE_ANSWERS})
    model = dataclasses.replace(MODEL, headers={"authorization": "Bearer model", "X-Source": "model"})

    await classify(
        model,
        CONTEXT,
        {
            "api_key": "secret",
            "headers": {"Authorization": "Bearer request", "x-source": "request"},
            "http_client": mock_server.client(),
        },
    )
    await classify(
        model, CONTEXT, {"api_key": "secret", "headers": {"Authorization": None}, "http_client": mock_server.client()}
    )

    first, second = mock_server.requests
    assert first.headers["authorization"] == "Bearer request"
    assert first.headers["x-source"] == "request"
    assert len(first.headers.get_list("authorization")) == 1
    assert "authorization" not in second.headers
    assert second.headers["x-source"] == "model"


async def test_preserves_prototype_sensitive_question_ids_in_answers(mock_server):
    mock_server.add_json({"answers": {"__proto__": {"type": "noul", "noul": 0.75}}})
    context = ClassifierContext(
        state={},
        questions={"__proto__": ClassifierBoolQuestion("Is this true?", {"true": "Yes", "false": "No"})},
    )

    result = await classify(MODEL, context, {"api_key": "secret", "http_client": mock_server.client()})

    assert result.stop_reason == "stop"
    assert list(result.answers) == ["__proto__"]
    assert result.answers["__proto__"] == ClassifierBoolAnswer(0.75)
    assert json.loads(json.dumps(to_json_value(result.answers)))["__proto__"] == {"type": "bool", "probability": 0.75}


async def test_accepts_questions_in_pi_json_form(mock_server):
    mock_server.add_json({"answers": {"ok": {"type": "noul", "noul": 0.6}}})
    context = ClassifierContext(
        state={}, questions={"ok": {"type": "bool", "instructions": "OK?", "criteria": {"true": "", "false": ""}}}
    )

    result = await classify(MODEL, context, {"api_key": "secret", "http_client": mock_server.client()})

    assert result.answers == {"ok": ClassifierBoolAnswer(0.6)}
    assert mock_server.last.json()["questions"]["ok"]["type"] == "noul"


async def test_reports_request_timeouts_separately_from_caller_cancellation(mock_server):
    async def hang(_request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    mock_server.add(hang)
    result = await classify(
        MODEL, CONTEXT, {"api_key": "secret", "timeout_ms": 5, "max_retries": 0, "http_client": mock_server.client()}
    )

    assert result.stop_reason == "error"
    assert result.error_message == "Request timed out after 5ms"


async def test_reports_caller_cancellation_as_aborted(mock_server):
    controller = AbortController()

    async def hang(_request: httpx.Request) -> httpx.Response:
        controller.abort()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    mock_server.add(hang)
    result = await classify(
        MODEL,
        CONTEXT,
        {"api_key": "secret", "timeout_ms": 5000, "signal": controller.signal, "http_client": mock_server.client()},
    )

    assert result.stop_reason == "aborted"
    assert result.error_message == "Request aborted"


async def test_creates_a_fresh_timeout_for_every_retry_attempt(mock_server):
    mock_server.add(httpx.Response(500, text="retry", headers={"retry-after-ms": "0"}))
    mock_server.add_json({"answers": WIRE_ANSWERS})

    result = await classify(
        MODEL, CONTEXT, {"api_key": "secret", "timeout_ms": 1000, "max_retries": 1, "http_client": mock_server.client()}
    )

    assert result.stop_reason == "stop"
    assert len(mock_server.requests) == 2


async def test_retries_twice_by_default_and_reports_the_http_failure(mock_server):
    for _ in range(3):
        mock_server.add(httpx.Response(503, text="overloaded", headers={"retry-after-ms": "0"}))

    result = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert len(mock_server.requests) == 3
    assert result.stop_reason == "error"
    assert result.error_message == "System One API error (503): overloaded"


async def test_returns_malformed_responses_as_classifier_errors(mock_server):
    mock_server.add_json({"answers": {}, "usage": {"input_tokens": 10, "output_tokens": 2}})

    result = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert result.stop_reason == "error"
    assert result.answers == {}
    assert "did not return an answer for category" in (result.error_message or "")
    # The request was billed, so its usage is kept.
    assert result.usage is not None
    assert (result.usage.input, result.usage.output) == (10, 2)


async def test_ignores_malformed_usage(mock_server):
    mock_server.add_json({"answers": WIRE_ANSWERS, "usage": {"input_tokens": "many", "output_tokens": 3}})
    mock_server.add_json({"answers": WIRE_ANSWERS, "usage": {"cost": 0.1}})

    result = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})
    without_tokens = await classify(MODEL, CONTEXT, {"api_key": "secret", "http_client": mock_server.client()})

    assert result.stop_reason == "stop"
    assert result.usage is not None
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (0, 3, 3)
    assert without_tokens.stop_reason == "stop"
    assert without_tokens.usage is None


async def test_payload_and_response_hooks(mock_server):
    mock_server.add_json({"answers": WIRE_ANSWERS}, headers={"x-request-id": "req-1"})
    payloads = []
    responses = []

    async def on_payload(payload, _model):
        payloads.append(payload)
        return {**payload, "trace": True}

    result = await classify(
        MODEL,
        CONTEXT,
        {
            "api_key": "secret",
            "on_payload": on_payload,
            "on_response": lambda response, _model: responses.append(response),
            "http_client": mock_server.client(),
        },
    )

    assert result.stop_reason == "stop"
    assert payloads[0]["model"] == "jev-latest"
    assert mock_server.last.json()["trace"] is True
    assert responses[0].status == 200
    assert responses[0].headers["x-request-id"] == "req-1"
