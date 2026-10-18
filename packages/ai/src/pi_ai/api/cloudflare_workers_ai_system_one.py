"""System One models on the Cloudflare Workers AI REST endpoint.

``POST /accounts/{account}/ai/run`` with ``{model, input}``. The REST API wraps the
model output in Cloudflare's API envelope. Third-party models such as
``typesafe/jev`` add a run record:
``{success, result: {state: "Completed", result: {answers, usage}}}``
(https://developers.cloudflare.com/ai/models/typesafe/jev/). Cloudflare-hosted
models such as ``@cf/cloudflare/clef`` return the output directly:
``{success, result: {model, answers, usage}}``
(https://developers.cloudflare.com/workers-ai/models/clef/).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

from ..types import ClassifierContext, ClassifierModel, ClassifierResult
from .classifier_shared import is_record
from .system_one_shared import SystemOneTransport, SystemOneWireRequest, classify_system_one

__all__ = ["classify"]

LABEL = "Cloudflare Workers AI"


def _cloudflare_error_message(errors: Any) -> str:
    if isinstance(errors, list):
        messages = [error["message"] for error in errors if is_record(error) and isinstance(error.get("message"), str)]
        if messages:
            return f"{LABEL} error: {'; '.join(messages)}"
    return f"{LABEL} request failed"


def _js_string(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _url(model: ClassifierModel) -> str:
    return urljoin(f"{model.base_url.rstrip('/')}/", "run")


def _payload(model: ClassifierModel, request: SystemOneWireRequest) -> Any:
    return {"model": model.id, "input": request}


def _output(body: Any) -> Mapping[str, Any]:
    if not is_record(body):
        raise ValueError(f"{LABEL} returned an unexpected response")
    if body.get("success") is False:
        raise ValueError(_cloudflare_error_message(body.get("errors")))
    result = body.get("result")
    if not is_record(result):
        raise ValueError(f"{LABEL} returned an unexpected response")
    if "answers" in result:
        return result
    if result.get("state") != "Completed":
        state = _js_string(result["state"]) if "state" in result else "undefined"
        raise ValueError(f"{LABEL} run did not complete (state: {state})")
    run_result = result.get("result")
    if not is_record(run_result):
        raise ValueError(f"{LABEL} returned an unexpected response")
    return run_result


_TRANSPORT = SystemOneTransport(
    api="cloudflare-workers-ai-system-one", label=LABEL, url=_url, payload=_payload, output=_output
)


async def classify(
    model: ClassifierModel, context: ClassifierContext, options: Mapping[str, Any] | None = None
) -> ClassifierResult:
    """Cloudflare Workers AI System One classification with public ``bool`` values mapped to ``noul``."""
    return await classify_system_one(_TRANSPORT, model, context, options)
