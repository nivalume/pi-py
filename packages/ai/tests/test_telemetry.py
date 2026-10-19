"""Telemetry in pi_ai: ``telemetry_context`` propagation (port of ``test/telemetry-options.test.ts``) and the
``pi.ai.request`` span that ``Models`` emits (a Python addition)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import httpx

from pi_ai import (
    AbortController,
    ApiKeyAuth,
    AssistantImages,
    AssistantMessage,
    AssistantMessageEventStream,
    AuthResult,
    Context,
    DeferredHandle,
    DoneEvent,
    ImageModel,
    ImagesContext,
    Model,
    ModelAuth,
    ModelCost,
    ProviderAuth,
    TextContent,
    Tool,
    Type,
    Usage,
    UserMessage,
    create_models,
    create_provider,
    faux_assistant_message,
    faux_provider,
    faux_tool_call,
    normalize_context,
)
from pi_ai.api import openai_completions
from pi_ai.api.simple_options import build_base_options
from pi_ai.telemetry import AI_TELEMETRY_SCHEMAS, AiRequestSpan, ai_request_span_of
from pi_telemetry import (
    NOOP_TELEMETRY_CONTEXT,
    InMemoryTelemetryContext,
    RecordedTelemetrySpan,
    TelemetryContext,
    with_content_capture,
)
from pi_telemetry.testing import assert_spans_match_schemas

TELEMETRY: TelemetryContext = NOOP_TELEMETRY_CONTEXT
CONTEXT = normalize_context({"messages": []})

MODEL = Model(
    id="model",
    name="Model",
    api="telemetry-test",
    provider="telemetry-provider",
    base_url="https://example.test",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=1000,
    max_tokens=100,
)

IMAGE_MODEL = ImageModel(
    id="image-model",
    name="Image Model",
    api="telemetry-test-images",
    provider="telemetry-image-provider",
    base_url="https://example.test",
    input=["text"],
    output=["image"],
)


async def _resolve(**_: Any) -> AuthResult:
    return AuthResult(ModelAuth())


AUTH = ProviderAuth(api_key=ApiKeyAuth("Test", _resolve))


def completed_stream(model: Model) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = AssistantMessage([], api=model.api, provider=model.provider, model=model.id, usage=Usage(), timestamp=0)
    stream.push(DoneEvent("stop", message))
    stream.end(message)
    return stream


# ─── propagation (upstream test) ────────────────────────────────────────────


def test_is_inherited_by_every_request_option_surface_and_simple_stream_conversion() -> None:
    assert build_base_options(MODEL, CONTEXT, {"telemetry_context": TELEMETRY})["telemetry_context"] is TELEMETRY
    assert "telemetry_context" not in build_base_options(MODEL, CONTEXT, {})


async def test_survives_provider_and_models_stream_and_deferred_dispatch() -> None:
    observed: list[Any] = []
    handle = DeferredHandle(provider=MODEL.provider, model_id=MODEL.id, api=MODEL.api, id="response")

    class Streams:
        def stream(self, model: Model, _context: Any, options: Mapping[str, Any] | None = None) -> Any:
            observed.append((options or {}).get("telemetry_context"))
            return completed_stream(model)

        def stream_simple(self, model: Model, _context: Any, options: Mapping[str, Any] | None = None) -> Any:
            observed.append((options or {}).get("telemetry_context"))
            return completed_stream(model)

        def fetch_deferred(self, model: Model, _handle: Any, options: Mapping[str, Any] | None = None) -> Any:
            observed.append((options or {}).get("telemetry_context"))
            return completed_stream(model)

        async def cancel_deferred(self, _model: Model, _handle: Any, options: Mapping[str, Any] | None = None) -> None:
            observed.append((options or {}).get("telemetry_context"))

    provider = create_provider(id=MODEL.provider, auth=AUTH, models=[MODEL], api=Streams())
    await provider.stream(MODEL, CONTEXT, {"telemetry_context": TELEMETRY}).result()
    await provider.stream_simple(MODEL, CONTEXT, {"telemetry_context": TELEMETRY}).result()
    assert provider.fetch_deferred is not None and provider.cancel_deferred is not None
    await provider.fetch_deferred(MODEL, handle, {"telemetry_context": TELEMETRY}).result()
    await provider.cancel_deferred(MODEL, handle, {"telemetry_context": TELEMETRY})

    models = create_models()
    models.set_provider(provider)
    await models.stream(MODEL, CONTEXT, {"telemetry_context": TELEMETRY}).result()
    await models.stream_simple(MODEL, CONTEXT, telemetry_context=TELEMETRY).result()
    await models.fetch_deferred(MODEL, handle, {"telemetry_context": TELEMETRY})
    await models.cancel_deferred(MODEL, handle, {"telemetry_context": TELEMETRY})

    assert len(observed) == 8
    assert all(value is TELEMETRY for value in observed)


async def test_survives_direct_and_models_image_dispatch() -> None:
    from pi_ai import compat

    observed: list[Any] = []

    async def generate(model: ImageModel, _context: Any, options: Mapping[str, Any] | None = None) -> AssistantImages:
        observed.append((options or {}).get("telemetry_context"))
        return AssistantImages(api=model.api, provider=model.provider, model=model.id, timestamp=0)

    compat.register_images_api_provider({"api": IMAGE_MODEL.api, "generate_images": generate})
    context = ImagesContext([TextContent("circle")])
    await compat.generate_images(IMAGE_MODEL, context, {"telemetry_context": TELEMETRY})

    class Images:
        async def generate_images(self, model: ImageModel, context: Any, options: Any = None) -> AssistantImages:
            return await generate(model, context, options)

    models = create_models()
    models.set_provider(
        create_provider(id=IMAGE_MODEL.provider, auth=AUTH, models=[IMAGE_MODEL], images={IMAGE_MODEL.api: Images()})
    )
    await models.generate_images(IMAGE_MODEL, context, {"telemetry_context": TELEMETRY})
    assert observed == [TELEMETRY, TELEMETRY]


# ─── the pi.ai.request span ─────────────────────────────────────────────────

PROMPT = "SECRET-PROMPT the user wrote"
ANSWER = "SECRET-ANSWER the model wrote"


def _faux() -> tuple[Any, Any]:
    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)
    return faux, models


def _request_span(telemetry: InMemoryTelemetryContext) -> RecordedTelemetrySpan:
    [span] = [span for span in telemetry.get_spans() if span.name == "pi.ai.request"]
    return span


def _all_text(spans: list[RecordedTelemetrySpan]) -> str:
    return json.dumps(
        [[span.attributes, [[event.name, event.attributes] for event in span.events], span.status] for span in spans]
    )


async def test_models_emit_one_request_span_with_start_and_end_attributes() -> None:
    faux, models = _faux()
    faux.set_responses([faux_assistant_message(ANSWER)])
    telemetry = InMemoryTelemetryContext()
    tool = Tool("read", "Read", Type.Object({"path": Type.String()}))
    events = []
    stream = models.stream_simple(
        faux.get_model(),
        Context(messages=[UserMessage(PROMPT)], system_prompt="be brief", tools=[tool]),
        telemetry_context=telemetry,
        reasoning="low",
        session_id="session-7",
        max_tokens=64,
    )
    async for event in stream:
        events.append(event.type)
    message = await stream.result()
    assert message.stop_reason == "stop"
    assert events[0] == "start" and events[-1] == "done"

    span = _request_span(telemetry)
    assert span.parent_id is None
    assert span.settled is True
    assert span.status == {"status": "ok"}
    attributes = dict(span.attributes)
    assert attributes.pop("pi.ai.duration_ms") >= 0  # type: ignore[operator]
    assert attributes.pop("pi.ai.time_to_first_token_ms") >= 0  # type: ignore[operator]
    assert attributes == {
        "pi.ai.provider": "faux",
        "pi.ai.api": faux.get_model().api,
        "pi.ai.model": "faux-1",
        "pi.ai.operation": "stream_simple",
        "pi.ai.streaming": True,
        "pi.ai.reasoning": "low",
        "pi.ai.tool_count": 1,
        "pi.ai.message_count": 2,
        "pi.ai.session_id": "session-7",
        "pi.ai.request.max_tokens": 64,
        "pi.ai.usage.input_tokens": message.usage.input,
        "pi.ai.usage.output_tokens": message.usage.output,
        "pi.ai.usage.cache_read_tokens": message.usage.cache_read,
        "pi.ai.usage.cache_write_tokens": message.usage.cache_write,
        "pi.ai.cost": message.usage.cost.total,
        "pi.ai.stop_reason": "stop",
        "pi.ai.http.status": 200,
    }
    assert span.events == []
    assert_spans_match_schemas(telemetry.get_spans(), AI_TELEMETRY_SCHEMAS)
    # The default data policy: no prompt, completion or system prompt anywhere.
    text = _all_text(telemetry.get_spans())
    assert "SECRET" not in text and "be brief" not in text


async def test_complete_marks_non_streaming_and_providers_receive_the_request_span() -> None:
    seen: list[Any] = []
    telemetry = InMemoryTelemetryContext()

    class Streams:
        def stream(self, model: Model, _context: Any, options: Mapping[str, Any] | None = None) -> Any:
            context = (options or {}).get("telemetry_context")
            seen.append(context)

            async def child(span: Any) -> None:
                span.set_attributes({"provider.detail": 1})

            async def produce() -> AssistantMessageEventStream:
                await context.start_span({"name": "provider.call"}, child)
                return completed_stream(model)

            from pi_ai import lazy_stream

            return lazy_stream(model, produce)

        def stream_simple(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
            return self.stream(model, context, options)

    models = create_models()
    models.set_provider(create_provider(id=MODEL.provider, auth=AUTH, models=[MODEL], api=Streams()))
    await models.complete(MODEL, Context(messages=[UserMessage("hi")]), telemetry_context=telemetry)
    [request] = seen
    assert isinstance(request, AiRequestSpan) and ai_request_span_of(request) is request
    request_span, provider_span = telemetry.get_spans()
    assert request_span.name == "pi.ai.request"
    assert request_span.attributes["pi.ai.streaming"] is False
    assert request_span.attributes["pi.ai.operation"] == "stream"
    assert "pi.ai.reasoning" not in request_span.attributes
    assert provider_span.name == "provider.call" and provider_span.parent_id == request_span.id
    assert ai_request_span_of(telemetry) is None


async def test_spans_nest_under_the_callers_span() -> None:
    faux, models = _faux()
    faux.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
    telemetry = InMemoryTelemetryContext()

    async def turn(span: Any) -> None:
        for _ in range(2):
            await models.complete(faux.get_model(), Context(messages=[UserMessage("hi")]), telemetry_context=span)

    await telemetry.start_span({"name": "app.turn"}, turn)
    parent, first, second = telemetry.get_spans()
    assert parent.name == "app.turn"
    assert (first.parent_id, second.parent_id) == (parent.id, parent.id)
    assert first.end_sequence is not None and second.end_sequence is not None and parent.end_sequence is not None
    assert first.end_sequence < second.end_sequence < parent.end_sequence


async def test_failed_requests_record_the_error_kind_only() -> None:
    faux, models = _faux()
    model = faux.get_model()
    faux.set_responses(
        [
            faux_assistant_message([], stop_reason="error", error_message="prompt is too long: SECRET details"),
            faux_assistant_message([], stop_reason="error", error_message="upstream exploded with SECRET"),
        ]
    )
    telemetry = InMemoryTelemetryContext()
    overflow = await models.complete(model, Context(messages=[UserMessage("x")]), telemetry_context=telemetry)
    failed = await models.complete(model, Context(messages=[UserMessage("x")]), telemetry_context=telemetry)
    assert overflow.stop_reason == "error" and failed.stop_reason == "error"
    first, second = telemetry.get_spans()
    assert first.status == {"status": "error", "error": {"name": "context_overflow", "message": ""}}
    assert first.attributes["pi.ai.error_kind"] == "context_overflow"
    assert [event.name for event in first.events] == ["pi.ai.context_overflow"]
    assert first.events[0].attributes == {"pi.ai.context_window": model.context_window}
    assert second.status == {"status": "error", "error": {"name": "provider_error", "message": ""}}
    assert "SECRET" not in _all_text(telemetry.get_spans())
    assert_spans_match_schemas(telemetry.get_spans(), AI_TELEMETRY_SCHEMAS)


async def test_aborted_requests_record_an_abort_event() -> None:
    faux, models = _faux()
    faux.set_responses([faux_assistant_message("never")])
    telemetry = InMemoryTelemetryContext()
    controller = AbortController()
    controller.abort()
    message = await models.complete(
        faux.get_model(), Context(messages=[UserMessage("x")]), signal=controller.signal, telemetry_context=telemetry
    )
    assert message.stop_reason == "aborted"
    span = _request_span(telemetry)
    assert span.status == {"status": "error", "error": {"name": "aborted", "message": ""}}
    assert [event.name for event in span.events] == ["pi.ai.abort"]


async def test_setup_failures_and_closed_streams_settle_the_span() -> None:
    telemetry = InMemoryTelemetryContext()
    models = create_models()
    message = await models.complete(MODEL, Context(messages=[UserMessage("x")]), telemetry_context=telemetry)
    assert message.stop_reason == "error"  # unknown provider
    span = _request_span(telemetry)
    assert span.settled and span.status["status"] == "error"

    _faux_fast, models = _faux()
    faux_slow = faux_provider(tokens_per_second=5)
    models.set_provider(faux_slow.provider)
    faux_slow.set_responses([faux_assistant_message("a long answer " * 20)])
    telemetry = InMemoryTelemetryContext()
    stream = models.stream(faux_slow.get_model(), Context(messages=[UserMessage("x")]), telemetry_context=telemetry)
    async for event in stream:
        if event.type == "text_delta":
            break
    await stream.aclose()
    span = _request_span(telemetry)
    assert span.settled
    assert span.status == {"status": "error", "error": {"name": "cancelled", "message": ""}}
    assert span.attributes["pi.ai.error_kind"] == "cancelled"
    assert "pi.ai.time_to_first_token_ms" in span.attributes


def _openai_sse(text: str) -> bytes:
    events = [
        {"id": "c1", "model": "gpt-4o-mini-2024-07-18", "choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"


async def test_http_retries_are_events_on_the_request_span() -> None:
    from pi_ai.providers.catalog import get_builtin_model

    base = get_builtin_model("openai", "gpt-4o-mini")
    assert base is not None
    model = dataclasses.replace(base, api="openai-completions")
    statuses = [429, 503, 200]

    def handler(_request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status != 200:
            return httpx.Response(status, headers={"retry-after-ms": "1"}, json={"error": {"message": "SECRET busy"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_openai_sse(ANSWER))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    models = create_models()
    models.set_provider(create_provider(id=model.provider, auth=AUTH, models=[model], api=openai_completions))
    telemetry = InMemoryTelemetryContext()
    responses: list[int] = []
    message = await models.complete(
        model,
        Context(messages=[UserMessage("hi")]),
        api_key="k",
        http_client=client,
        max_retries=3,
        telemetry_context=telemetry,
        on_response=lambda response, _model: responses.append(response.status),
    )
    await client.aclose()
    assert message.stop_reason == "stop", message.error_message
    assert responses == [200]  # the caller's on_response still runs
    span = _request_span(telemetry)
    assert [(event.name, event.attributes) for event in span.events] == [
        (
            "pi.ai.retry",
            {
                "pi.ai.retry.attempt": 1,
                "pi.ai.retry.reason": "http_status",
                "pi.ai.retry.delay_ms": 1,
                "pi.ai.retry.http_status": 429,
            },
        ),
        (
            "pi.ai.retry",
            {
                "pi.ai.retry.attempt": 2,
                "pi.ai.retry.reason": "http_status",
                "pi.ai.retry.delay_ms": 1,
                "pi.ai.retry.http_status": 503,
            },
        ),
    ]
    assert span.attributes["pi.ai.retry_count"] == 2
    assert span.attributes["pi.ai.http.status"] == 200
    assert span.attributes["pi.ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert span.attributes["pi.ai.usage.input_tokens"] == 7
    assert "SECRET" not in _all_text(telemetry.get_spans())
    assert_spans_match_schemas(telemetry.get_spans(), AI_TELEMETRY_SCHEMAS)

    # A final failure records its status and the rate-limit kind.
    statuses[:] = [429, 429]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    telemetry = InMemoryTelemetryContext()
    failed = await models.complete(
        model,
        Context(messages=[UserMessage("hi")]),
        api_key="k",
        http_client=client,
        max_retries=1,
        telemetry_context=telemetry,
    )
    await client.aclose()
    assert failed.stop_reason == "error"
    span = _request_span(telemetry)
    assert span.attributes["pi.ai.http.status"] == 429
    assert span.status == {"status": "error", "error": {"name": "rate_limited", "message": ""}}
    assert len(span.events) == 1


async def test_content_is_recorded_only_with_an_explicit_opt_in() -> None:
    faux, models = _faux()
    faux.set_responses(
        [faux_assistant_message([faux_tool_call("read", {"path": "secret.txt"})], stop_reason="toolUse")]
    )
    redacted: list[tuple[str, str]] = []

    def redact(kind: str, text: str) -> str | None:
        redacted.append((kind, text))
        return None if kind == "system" else text.replace("SECRET", "[redacted]")

    telemetry = InMemoryTelemetryContext()
    await models.complete(
        faux.get_model(),
        Context(messages=[UserMessage(PROMPT + " " + "x" * 50)], system_prompt="system rules"),
        telemetry_context=with_content_capture(telemetry, max_length=30, redact=redact),
    )
    span = _request_span(telemetry)
    content = [event.attributes for event in span.events if event.name == "pi.content"]
    assert content == [
        {
            "pi.content.kind": "user",
            "pi.content.text": "[redacted]-PROMPT the user wro",
            "pi.content.length": 83,
            "pi.content.truncated": True,
            "pi.content.index": 1,
        },
        {
            "pi.content.kind": "tool_call",
            "pi.content.text": '{"path": "secret.txt"}',
            "pi.content.length": 22,
            "pi.content.truncated": False,
            "pi.content.name": "read",
        },
    ]
    assert [kind for kind, _text in redacted] == ["system", "user", "tool_call"]
    assert_spans_match_schemas(telemetry.get_spans(), AI_TELEMETRY_SCHEMAS)


async def test_without_telemetry_providers_see_the_options_unchanged() -> None:
    seen: list[Mapping[str, Any]] = []

    class Streams:
        def stream(self, model: Model, _context: Any, options: Mapping[str, Any] | None = None) -> Any:
            seen.append(dict(options or {}))
            return completed_stream(model)

        def stream_simple(self, model: Model, context: Any, options: Mapping[str, Any] | None = None) -> Any:
            return self.stream(model, context, options)

    models = create_models()
    models.set_provider(create_provider(id=MODEL.provider, auth=AUTH, models=[MODEL], api=Streams()))
    await models.complete(MODEL, Context(messages=[UserMessage("x")]))
    await models.complete(MODEL, Context(messages=[UserMessage("x")]), telemetry_context=NOOP_TELEMETRY_CONTEXT)
    assert "telemetry_context" not in seen[0] and "on_response" not in seen[0]
    assert seen[1]["telemetry_context"] is NOOP_TELEMETRY_CONTEXT and "on_response" not in seen[1]


async def test_compat_dispatch_is_traced_too() -> None:
    from pi_ai import compat

    faux = compat.register_faux_provider()
    try:
        faux.set_responses([faux_assistant_message("hi")])
        telemetry = InMemoryTelemetryContext()
        await compat.complete(faux.get_model(), Context(messages=[UserMessage("x")]), telemetry_context=telemetry)
        span = _request_span(telemetry)
        assert span.attributes["pi.ai.streaming"] is False
        assert span.status == {"status": "ok"}
    finally:
        faux.unregister()


async def test_concurrent_requests_record_separate_spans() -> None:
    faux, models = _faux()
    faux.set_responses([faux_assistant_message(f"answer {index}") for index in range(5)])
    telemetry = InMemoryTelemetryContext()
    await asyncio.gather(
        *(
            models.complete(faux.get_model(), Context(messages=[UserMessage("x")]), telemetry_context=telemetry)
            for _ in range(5)
        )
    )
    spans = telemetry.get_spans()
    assert len(spans) == 5 and all(span.settled and span.status == {"status": "ok"} for span in spans)
