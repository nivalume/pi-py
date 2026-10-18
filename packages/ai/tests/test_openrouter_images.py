"""Offline tests for OpenRouter image generation (port of upstream ``test/openrouter-images.test.ts``).

``test/images.test.ts`` is end-to-end only (it needs ``OPENROUTER_API_KEY``) and is not ported.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import httpx
import pytest

from pi_ai.abort import AbortController
from pi_ai.api.openrouter_images import generate_images
from pi_ai.types import ImageContent, ImageModel, ImagesContext, ModelCost, TextContent

TEXT_AND_IMAGE_MODEL = ImageModel(
    id="google/gemini-3.1-flash-image-preview",
    name="Gemini 3.1 Flash Image Preview",
    api="openrouter-images",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    input=["text", "image"],
    output=["text", "image"],
    cost=ModelCost(input=0.015, output=0.03),
    headers={"HTTP-Referer": "https://example.com"},
)

IMAGE_ONLY_MODEL = ImageModel(
    id="black-forest-labs/flux.2-pro",
    name="FLUX.2 Pro",
    api="openrouter-images",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    input=["text", "image"],
    output=["image"],
    cost=ModelCost(input=0.015, output=0.03),
)

CONTEXT = ImagesContext(input=[TextContent("Generate a dog")])

RESPONSE = {
    "id": "img-1",
    "usage": {"prompt_tokens": 12, "completion_tokens": 34, "prompt_tokens_details": {"cached_tokens": 0}},
    "choices": [
        {
            "message": {
                "content": "Here is your image.",
                "images": [{"image_url": "data:image/png;base64,ZmFrZS1wbmc="}],
            }
        }
    ],
}


async def test_returns_text_plus_images_in_final_output(mock_server):
    mock_server.add_json(RESPONSE)

    output = await generate_images(
        TEXT_AND_IMAGE_MODEL, CONTEXT, {"api_key": "test", "http_client": mock_server.client()}
    )

    assert output.stop_reason == "stop"
    assert output.error_message is None
    assert output.response_id == "img-1"
    assert output.output == [TextContent("Here is your image."), ImageContent("ZmFrZS1wbmc=", "image/png")]
    assert output.usage is not None
    assert (output.usage.input, output.usage.output, output.usage.total_tokens) == (12, 34, 46)
    assert output.usage.cost.input == pytest.approx(0.015 * 12 / 1_000_000)
    assert output.usage.cost.total == pytest.approx((0.015 * 12 + 0.03 * 34) / 1_000_000)

    request = mock_server.last
    assert request.url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test"
    assert request.headers["http-referer"] == "https://example.com"
    assert request.headers["content-type"] == "application/json"
    params = request.json()
    assert params == {
        "model": "google/gemini-3.1-flash-image-preview",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Generate a dog"}]}],
        "stream": False,
        "modalities": ["image", "text"],
    }


async def test_returns_an_aborted_result_for_an_aborted_signal(mock_server):
    controller = AbortController()
    controller.abort()

    output = await generate_images(
        IMAGE_ONLY_MODEL, CONTEXT, {"api_key": "test", "signal": controller.signal, "http_client": mock_server.client()}
    )

    assert output.stop_reason == "aborted"
    assert output.error_message == "Request aborted"
    assert output.output == []
    assert mock_server.requests == []


async def test_image_only_models_do_not_request_text_output(mock_server):
    mock_server.add_json(RESPONSE)

    output = await generate_images(IMAGE_ONLY_MODEL, CONTEXT, {"api_key": "test", "http_client": mock_server.client()})

    assert any(item.type == "image" for item in output.output)
    # Image-only models must not request text output.
    assert mock_server.last.json()["modalities"] == ["image"]


async def test_sends_image_input_as_data_urls_and_sanitizes_text(mock_server):
    mock_server.add_json(RESPONSE)
    context = ImagesContext(input=[TextContent("Vary this \ud83d"), ImageContent("aW1n", "image/jpeg")])

    await generate_images(IMAGE_ONLY_MODEL, context, {"api_key": "test", "http_client": mock_server.client()})

    assert mock_server.last.json()["messages"][0]["content"] == [
        {"type": "text", "text": "Vary this "},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1n"}},
    ]


async def test_parses_object_image_urls_and_skips_non_data_urls(mock_server):
    mock_server.add_json(
        {
            "id": "img-2",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "images": [
                            {"image_url": {"url": "data:image/webp;base64,d2VicA=="}},
                            {"image_url": {"url": "https://example.com/a.png"}},
                            {"image_url": "data:image/png,not-base64"},
                            {},
                        ],
                    }
                }
            ],
        }
    )

    output = await generate_images(IMAGE_ONLY_MODEL, CONTEXT, {"api_key": "test", "http_client": mock_server.client()})

    assert output.stop_reason == "stop"
    assert output.output == [ImageContent("d2VicA==", "image/webp")]
    assert output.usage is None


async def test_splits_cached_and_cache_write_prompt_tokens(mock_server):
    mock_server.add_json(
        {
            "id": "img-3",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
            },
            "choices": [],
        }
    )
    model = dataclasses.replace(IMAGE_ONLY_MODEL, cost=ModelCost(input=1, output=2, cache_read=0.5, cache_write=3))

    output = await generate_images(model, CONTEXT, {"api_key": "test", "http_client": mock_server.client()})

    usage = output.usage
    assert usage is not None
    assert (usage.input, usage.output, usage.cache_read, usage.cache_write, usage.total_tokens) == (70, 5, 20, 10, 105)
    assert usage.cost.total == pytest.approx((70 * 1 + 5 * 2 + 20 * 0.5 + 10 * 3) / 1_000_000)
    assert output.output == []


async def test_reports_http_errors_with_the_provider_body(mock_server):
    mock_server.add_json({"error": {"message": "Model not found", "code": 404}}, status=404)

    output = await generate_images(IMAGE_ONLY_MODEL, CONTEXT, {"api_key": "test", "http_client": mock_server.client()})

    assert output.stop_reason == "error"
    assert output.error_message == '404: {"message":"Model not found","code":404}'


async def test_retries_only_when_asked(mock_server):
    mock_server.add(httpx.Response(503, json={"error": {"message": "busy"}}, headers={"retry-after-ms": "0"}))
    mock_server.add_json(RESPONSE)

    output = await generate_images(
        IMAGE_ONLY_MODEL, CONTEXT, {"api_key": "test", "max_retries": 1, "http_client": mock_server.client()}
    )

    assert output.stop_reason == "stop"
    assert len(mock_server.requests) == 2


async def test_requires_an_api_key(mock_server):
    output = await generate_images(IMAGE_ONLY_MODEL, CONTEXT, {"http_client": mock_server.client()})

    assert output.stop_reason == "error"
    assert output.error_message == "No API key for provider: openrouter"
    assert mock_server.requests == []


async def test_payload_and_response_hooks(mock_server):
    mock_server.add_json(RESPONSE, headers={"x-generation-id": "gen-1"})
    observed: list[Any] = []

    async def on_payload(params: Any, _model: Any) -> Any:
        return {**params, "image_config": {"aspect_ratio": "16:9"}}

    await generate_images(
        IMAGE_ONLY_MODEL,
        CONTEXT,
        {
            "api_key": "test",
            "headers": {"HTTP-Referer": None, "X-Title": "pi"},
            "on_payload": on_payload,
            "on_response": lambda response, _model: observed.append(response),
            "http_client": mock_server.client(),
        },
    )

    assert mock_server.last.json()["image_config"] == {"aspect_ratio": "16:9"}
    assert mock_server.last.headers["x-title"] == "pi"
    assert observed[0].status == 200
    assert observed[0].headers["x-generation-id"] == "gen-1"


async def test_generate_images_resolves_through_the_images_registry(mock_server):
    compat = pytest.importorskip("pi_ai.compat")
    mock_server.add_json(RESPONSE)

    output = await compat.generate_images(
        IMAGE_ONLY_MODEL, CONTEXT, {"api_key": "test", "http_client": mock_server.client()}
    )

    assert any(item.type == "image" for item in output.output)
