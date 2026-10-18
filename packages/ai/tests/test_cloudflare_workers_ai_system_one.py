"""Offline tests for System One on Cloudflare Workers AI.

Port of upstream ``test/cloudflare-workers-ai-system-one.test.ts``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from pi_ai.api import cloudflare
from pi_ai.api.cloudflare_workers_ai_system_one import classify
from pi_ai.providers.catalog import get_builtin_classifier_model, get_builtin_model
from pi_ai.types import (
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierModel,
)

CONTEXT = ClassifierContext(
    state={"message": "Help! My payouts have been failing for 3 days."},
    questions={
        "is_urgent": ClassifierBoolQuestion(
            "Does this convey urgency?", {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
        ),
        "department": ClassifierChoiceQuestion(
            "Which team should handle this?", {"billing": "Payments", "technical": "Bugs"}
        ),
    },
)

# Model output from https://developers.cloudflare.com/ai/models/typesafe/jev/
JEV_OUTPUT = {
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.95},
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.87, "technical": 0.13},
        },
    },
    "usage": {"input_tokens": 426, "output_tokens": 73},
}

# Cloudflare-hosted output observed from a live /ai/run call, question ids renamed to match CONTEXT.
# The envelope carries the output directly, without a run record.
CLEF_OUTPUT = {
    "model": "clef",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.9912},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.1632, "technical": 0.8368},
            "confidence": 0.4538,
        },
    },
    "usage": {"input_tokens": 222, "output_tokens": 0},
}

RUN_URL = "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run"


def rest_response(state: str, result: Any = JEV_OUTPUT) -> dict[str, Any]:
    """REST envelope observed from the live /ai/run endpoint."""
    return {
        "result": {"state": state, "result": result, "gatewayMetadata": {"keySource": "Unified"}},
        "success": True,
        "errors": [],
        "messages": [],
    }


def resolved(model_id: str) -> ClassifierModel:
    """A catalog model with the account placeholder filled, as the Cloudflare provider does before dispatch."""
    model = get_builtin_classifier_model("cloudflare-workers-ai", model_id)
    assert model is not None
    return dataclasses.replace(model, base_url=model.base_url.replace("{CLOUDFLARE_ACCOUNT_ID}", "account-id"))


def test_exposes_jev_only_through_classifier_catalog_accessors():
    jev = get_builtin_classifier_model("cloudflare-workers-ai", "typesafe/jev")
    assert jev is not None
    assert jev.type == "classifier"
    assert jev.api == "cloudflare-workers-ai-system-one"
    assert jev.base_url == cloudflare.CLOUDFLARE_WORKERS_AI_REST_BASE_URL
    assert get_builtin_model("cloudflare-workers-ai", "typesafe/jev") is None


async def test_runs_jev_through_the_account_scoped_run_endpoint(mock_server):
    mock_server.add_json(rest_response("Completed"))

    result = await classify(
        resolved("typesafe/jev"), CONTEXT, {"api_key": "cf-key", "http_client": mock_server.client()}
    )

    request = mock_server.last
    payload = request.json()
    assert request.url == RUN_URL
    assert payload["model"] == "typesafe/jev"
    assert payload["input"]["state"] == CONTEXT.state
    assert payload["input"]["questions"]["is_urgent"]["type"] == "noul"
    assert payload["input"]["questions"]["department"]["type"] == "choice"
    assert request.headers["authorization"] == "Bearer cf-key"
    assert result.stop_reason == "stop"
    assert result.answers["is_urgent"] == ClassifierBoolAnswer(0.95)
    department = result.answers["department"]
    assert (department.type, department.choice, department.confidence) == ("choice", "billing", 0.8)
    assert result.usage is not None
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (426, 73, 499)


@pytest.mark.parametrize(
    ("model_id", "input_price"), [("@cf/cloudflare/clef", 0.24), ("@cf/cloudflare/clef-flash", 0.09)]
)
async def test_runs_clef_through_run_and_parses_its_direct_output(mock_server, model_id, input_price):
    mock_server.add_json({"result": CLEF_OUTPUT, "success": True, "errors": [], "messages": []})

    result = await classify(resolved(model_id), CONTEXT, {"api_key": "cf-key", "http_client": mock_server.client()})

    payload = mock_server.last.json()
    assert mock_server.last.url == RUN_URL
    assert payload["model"] == model_id
    assert payload["input"]["state"] == CONTEXT.state
    assert payload["input"]["questions"]["is_urgent"]["type"] == "noul"
    assert result.stop_reason == "stop"
    assert result.answers["is_urgent"] == ClassifierBoolAnswer(0.9912)
    department = result.answers["department"]
    assert (department.choice, department.confidence) == ("technical", 0.4538)
    assert result.usage is not None
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (222, 0, 222)
    assert result.usage.cost.input == pytest.approx(222 * input_price / 1_000_000)


async def test_reports_runs_that_did_not_complete(mock_server):
    mock_server.add_json(rest_response("Queued", None))

    result = await classify(
        resolved("typesafe/jev"), CONTEXT, {"api_key": "cf-key", "http_client": mock_server.client()}
    )

    assert result.stop_reason == "error"
    assert "run did not complete (state: Queued)" in (result.error_message or "")


async def test_reports_cloudflare_envelope_errors(mock_server):
    mock_server.add_json({"success": False, "errors": [{"code": 5007, "message": "No such model"}], "result": None})
    mock_server.add_json({"success": False, "errors": []})

    result = await classify(
        resolved("typesafe/jev"), CONTEXT, {"api_key": "cf-key", "http_client": mock_server.client()}
    )
    generic = await classify(
        resolved("typesafe/jev"), CONTEXT, {"api_key": "cf-key", "http_client": mock_server.client()}
    )

    assert result.stop_reason == "error"
    assert result.error_message == "Cloudflare Workers AI error: No such model"
    assert generic.error_message == "Cloudflare Workers AI request failed"


async def test_rejects_unexpected_envelopes(mock_server):
    mock_server.add_json({"success": True, "result": {"state": "Completed", "result": "nope"}})
    mock_server.add_json(["not", "an", "object"])

    nested = await classify(
        resolved("typesafe/jev"), CONTEXT, {"api_key": "cf-key", "http_client": mock_server.client()}
    )
    top = await classify(resolved("typesafe/jev"), CONTEXT, {"api_key": "cf-key", "http_client": mock_server.client()})

    assert nested.error_message == "Cloudflare Workers AI returned an unexpected response"
    assert top.error_message == "Cloudflare Workers AI returned an unexpected response"


def test_endpoint_constants():
    assert cloudflare.CLOUDFLARE_WORKERS_AI_BASE_URL.endswith("/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1")
    assert cloudflare.CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL == (
        "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat"
    )
    assert cloudflare.CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL.endswith("/openai")
    assert cloudflare.CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL.endswith("/anthropic")


async def test_runs_jev_through_the_cloudflare_provider(mock_server):
    providers = pytest.importorskip("pi_ai.providers.cloudflare_workers_ai")
    from pi_ai.models import create_models

    models = create_models()
    models.set_provider(providers.cloudflare_workers_ai_provider())
    jev = models.get_model_of_type("classifier", "cloudflare-workers-ai", "typesafe/jev")
    assert jev is not None
    assert models.get_model("cloudflare-workers-ai", "typesafe/jev") is None
    mock_server.add_json(rest_response("Completed"))

    result = await models.classify(
        jev,
        CONTEXT,
        {"api_key": "cf-key", "env": {"CLOUDFLARE_ACCOUNT_ID": "account-id"}, "http_client": mock_server.client()},
    )

    assert mock_server.last.url == RUN_URL
    assert result.stop_reason == "stop"
    assert result.answers["is_urgent"] == ClassifierBoolAnswer(0.95)
