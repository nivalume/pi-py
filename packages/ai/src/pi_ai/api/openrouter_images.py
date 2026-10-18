"""Image generation over OpenRouter's chat completions endpoint."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..abort import AbortError, abort_scope
from ..types import (
    AssistantImages,
    ImageContent,
    ImageModel,
    ImagesContext,
    ProviderResponse,
    TextContent,
    Usage,
    UsageCost,
    now_ms,
)
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.misc import headers_to_record, provider_headers_to_record, sanitize_surrogates
from ..utils.model_operations import image_error_result
from .openai_sdk import create_openai_client, openai_request, sdk_body_arguments

__all__ = ["generate_images"]

_DATA_URL = re.compile(r"data:([^;]+);base64,(.+)")


async def generate_images(
    model: ImageModel, context: ImagesContext, options: Mapping[str, Any] | None = None
) -> AssistantImages:
    """Generate images with an OpenRouter image model. Never raises."""
    options = options or {}
    output = AssistantImages(
        api=model.api, provider=model.provider, model=model.id, output=[], stop_reason="stop", timestamp=now_ms()
    )
    signal = options.get("signal")
    try:
        api_key = options.get("api_key")
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")
        client = create_openai_client(
            api_key=api_key,
            base_url=model.base_url,
            headers=provider_headers_to_record(model.headers, options.get("headers")) or {},
            options=options,
        )
        params: Any = _build_params(model, context)
        on_payload = options.get("on_payload")
        if on_payload is not None:
            next_params = await maybe_await(on_payload(params, model))
            if next_params is not None:
                params = next_params
        create = client.chat.completions.with_raw_response.create
        body_arguments = sdk_body_arguments(params, "model", "messages")
        try:
            async with abort_scope(signal):
                response = await openai_request(lambda: create(**body_arguments), options)
        except AbortError as error:
            raise AbortError("Request aborted") from error
        on_response = options.get("on_response")
        if on_response is not None:
            await maybe_await(
                on_response(ProviderResponse(response.status_code, headers_to_record(response.headers)), model)
            )

        body = fastjson.loads(response.content)
        if not isinstance(body, Mapping):
            raise ValueError("OpenRouter returned an unexpected image generation response")
        output.response_id = body.get("id")
        raw_usage = body.get("usage")
        if isinstance(raw_usage, Mapping):
            output.usage = _parse_usage(raw_usage, model)

        choices = body.get("choices")
        if not isinstance(choices, list):
            raise ValueError("OpenRouter returned an image generation response without choices")
        choice = choices[0] if choices else None
        if isinstance(choice, Mapping):
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                output.output.append(TextContent(content))
            for image in message.get("images") or ():
                image_url = image.get("image_url") if isinstance(image, Mapping) else None
                url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
                if not isinstance(url, str) or not url.startswith("data:"):
                    continue
                matches = _DATA_URL.fullmatch(url)
                if matches is None:
                    continue
                output.output.append(ImageContent(data=matches.group(2), mime_type=matches.group(1)))
        return output
    except Exception as error:
        failed = image_error_result(
            model,
            format_provider_error(normalize_provider_error(error)),
            aborted=signal is not None and signal.aborted,
        )
        failed.response_id = output.response_id
        failed.usage = output.usage
        failed.timestamp = output.timestamp
        return failed


def _build_params(model: ImageModel, context: ImagesContext) -> dict[str, Any]:
    """The chat completions request body for an image generation."""
    content: list[dict[str, Any]] = []
    for item in context.input:
        if item.type == "text":
            content.append({"type": "text", "text": sanitize_surrogates(item.text)})  # type: ignore[union-attr]
        else:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}}  # type: ignore[union-attr]
            )
    return {
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": ["image", "text"] if "text" in model.output else ["image"],
    }


def _parse_usage(raw_usage: Mapping[str, Any], model: ImageModel) -> Usage:
    """Usage from an OpenAI-style ``usage`` object, priced from the model's base rates."""
    details = raw_usage.get("prompt_tokens_details") or {}
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    reported_cached_tokens = details.get("cached_tokens") or 0
    cache_write_tokens = details.get("cache_write_tokens") or 0
    cache_read_tokens = (
        max(0, reported_cached_tokens - cache_write_tokens) if cache_write_tokens > 0 else reported_cached_tokens
    )
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = raw_usage.get("completion_tokens") or 0
    cost = UsageCost(
        input=(model.cost.input / 1000000) * input_tokens,
        output=(model.cost.output / 1000000) * output_tokens,
        cache_read=(model.cost.cache_read / 1000000) * cache_read_tokens,
        cache_write=(model.cost.cache_write / 1000000) * cache_write_tokens,
    )
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write
    return Usage(
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        cost,
    )
