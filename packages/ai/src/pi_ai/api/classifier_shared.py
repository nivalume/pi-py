"""HTTP and parsing helpers shared by the classifier APIs."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

import httpx

from ..abort import AbortError, abort_scope
from ..models import calculate_cost
from ..serialization import classifier_question_from_dict
from ..types import ClassifierContext, ClassifierModel, ClassifierQuestion, ProviderResponse, Usage
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.http import DEFAULT_TIMEOUT_MS, ProviderHttpError, get_http_client, retry_provider_request
from ..utils.misc import headers_to_record, provider_headers_to_record

__all__ = [
    "ClassifierHttpError",
    "is_record",
    "is_number",
    "required_number",
    "format_js_number",
    "classifier_questions",
    "coerce_question",
    "post_json",
    "post_classifier_request",
    "parse_classifier_usage",
]

#: Default client-side retries for classifier requests.
DEFAULT_CLASSIFIER_MAX_RETRIES = 2


class ClassifierHttpError(ProviderHttpError):
    """An HTTP failure in the shape ``retry_provider_request`` and ``normalize_provider_error`` understand.

    ``body`` holds the raw response text, so error formatting can show the provider's reason.
    """

    def __init__(self, message: str, *, status: int | None, headers: httpx.Headers | None, body: str) -> None:
        super().__init__(message, status=status, headers=headers, body=body, include_body=True)


def _http_error(label: str, response: httpx.Response) -> ClassifierHttpError:
    return ClassifierHttpError(
        f"{label} returned {response.status_code}",
        status=response.status_code,
        headers=response.headers,
        body=response.text,
    )


def _timeout_error(timeout_ms: float) -> ClassifierHttpError:
    return ClassifierHttpError(
        f"Request timed out after {format_js_number(timeout_ms)}ms", status=None, headers=None, body=""
    )


def is_record(value: Any) -> TypeGuard[Mapping[str, Any]]:
    """Whether ``value`` is a JSON object."""
    return isinstance(value, Mapping)


def is_number(value: Any) -> bool:
    """Whether ``value`` is a finite JSON number (booleans excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def required_number(label: str, value: Any, field: str) -> float:
    if not is_number(value):
        raise ValueError(f"{label} returned an invalid {field}")
    return value


def format_js_number(value: float) -> str:
    """Render a number like JavaScript's ``String(n)`` for integral values (``5`` rather than ``5.0``)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def coerce_question(question: Any) -> ClassifierQuestion:
    """Accept a question dataclass or its pi JSON form (``{"type": "bool", ...}``)."""
    if isinstance(question, Mapping):
        return classifier_question_from_dict(question)
    return question


def classifier_questions(context: ClassifierContext) -> dict[str, ClassifierQuestion]:
    """The context's questions as dataclasses, in request order."""
    return {name: coerce_question(question) for name, question in context.questions.items()}


async def post_json(
    label: str,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Any,
    options: Mapping[str, Any],
    no_retry_statuses: Sequence[int] = (),
) -> tuple[httpx.Response, Any]:
    """POST ``payload`` as JSON and return the response with its parsed body.

    Each attempt gets a fresh ``timeout_ms`` deadline covering the whole exchange.
    Retryable failures are retried ``max_retries`` times (default 2). Caller
    cancellation raises :class:`AbortError` ("Request aborted").
    """
    client = get_http_client(url, options)
    content = fastjson.dumps_bytes(payload)
    timeout_ms = options.get("timeout_ms")
    signal = options.get("signal")
    http_timeout = httpx.Timeout((timeout_ms if timeout_ms is not None else DEFAULT_TIMEOUT_MS) / 1000)

    async def attempt() -> tuple[httpx.Response, Any]:
        request = client.build_request("POST", url, headers=headers, content=content, timeout=http_timeout)
        try:
            async with asyncio.timeout(timeout_ms / 1000 if timeout_ms is not None else None):
                response = await client.send(request)
        except (TimeoutError, httpx.TimeoutException) as error:
            if timeout_ms is not None and not (signal is not None and signal.aborted):
                raise _timeout_error(timeout_ms) from error
            raise
        if not response.is_success:
            raise _http_error(label, response)
        return response, fastjson.loads(response.content)

    max_retries = options.get("max_retries")
    try:
        async with abort_scope(signal):
            return await retry_provider_request(
                attempt,
                max_retries=max_retries if max_retries is not None else DEFAULT_CLASSIFIER_MAX_RETRIES,
                max_retry_delay_ms=options.get("max_retry_delay_ms"),
                signal=signal,
                no_retry_statuses=no_retry_statuses,
                telemetry_context=options.get("telemetry_context"),
            )
    except AbortError as error:
        raise AbortError("Request aborted") from error


def _request_headers(model: ClassifierModel, api_key: str, options_headers: Any) -> dict[str, str]:
    return (
        provider_headers_to_record(
            {"authorization": f"Bearer {api_key}", "content-type": "application/json"},
            model.headers,
            options_headers,
        )
        or {}
    )


async def post_classifier_request(
    label: str,
    url: str,
    model: ClassifierModel,
    body: Any,
    options: Mapping[str, Any] | None,
    no_retry_statuses: Sequence[int] = (),
) -> Any:
    """POST one JSON classifier request with bearer auth, hooks, per-attempt timeouts and retries.

    Returns the parsed response body; raises on failure. ``no_retry_statuses`` lists
    HTTP statuses that fail at once although they are normally retried.
    """
    options = options or {}
    api_key = options.get("api_key")
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")
    payload = body
    on_payload = options.get("on_payload")
    if on_payload is not None:
        transformed = await maybe_await(on_payload(payload, model))
        if transformed is not None:
            payload = transformed
    response, json_body = await post_json(
        label,
        url,
        headers=_request_headers(model, api_key, options.get("headers")),
        payload=payload,
        options=options,
        no_retry_statuses=no_retry_statuses,
    )
    on_response = options.get("on_response")
    if on_response is not None:
        await maybe_await(
            on_response(ProviderResponse(response.status_code, headers_to_record(response.headers)), model)
        )
    return json_body


def _token_count(value: Any) -> Any:
    return value if is_number(value) and value > 0 else 0


def parse_classifier_usage(value: Any, model: ClassifierModel) -> Usage | None:
    """Usage from an ``{input_tokens, output_tokens}`` object, priced from the model catalog like chat usage.

    A missing or malformed usage object leaves the result without usage instead of failing it.
    """
    if not is_record(value) or ("input_tokens" not in value and "output_tokens" not in value):
        return None
    input_tokens = _token_count(value.get("input_tokens"))
    output_tokens = _token_count(value.get("output_tokens"))
    usage = Usage(input_tokens, output_tokens, 0, 0, input_tokens + output_tokens)
    calculate_cost(model, usage)
    return usage
