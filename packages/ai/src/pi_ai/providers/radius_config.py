"""Radius gateway configuration (``GET /v1/config``) and its model list."""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, TypedDict

import httpx

from ..abort import AbortSignal, abort_scope
from ..auth.types import Credential
from ..serialization import model_from_dict
from ..types import Model
from ..utils.http import get_http_client

__all__ = [
    "DEFAULT_RADIUS_GATEWAY",
    "RadiusGatewayModel",
    "RadiusGatewayConfig",
    "normalize_radius_gateway_url",
    "get_radius_credential_config",
    "get_radius_models_from_config",
    "get_radius_models",
    "load_radius_gateway_config",
]

DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"


class RadiusGatewayModel(TypedDict, total=False):
    """One gateway model in pi's JSON shape (camelCase keys)."""

    id: str
    name: str
    reasoning: bool
    thinkingLevelMap: dict[str, str | None]
    input: list[str]
    cost: dict[str, Any]
    contextWindow: int
    maxTokens: int


class RadiusGatewayConfig(TypedDict):
    baseUrl: str
    models: list[RadiusGatewayModel]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_radius_gateway_model(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("id"), str)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("reasoning"), bool)
        and isinstance(value.get("input"), list)
        and isinstance(value.get("cost"), dict)
        and _is_number(value.get("contextWindow"))
        and _is_number(value.get("maxTokens"))
    )


def _sanitize_radius_gateway_config(config: Any) -> RadiusGatewayConfig | None:
    if not isinstance(config, dict):
        return None
    base_url = config.get("baseUrl")
    models = config.get("models")
    if not isinstance(base_url, str) or not isinstance(models, list):
        return None
    return {"baseUrl": base_url, "models": [dict(model) for model in models if _is_radius_gateway_model(model)]}  # type: ignore[misc]


_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_TRAILING_SLASHES = re.compile(r"/+$")


def normalize_radius_gateway_url(value: str) -> str:
    """Add ``https://`` when no scheme is given and drop trailing slashes."""
    with_scheme = value if _SCHEME.match(value) else f"https://{value}"
    return _TRAILING_SLASHES.sub("", with_scheme)


def get_radius_credential_config(credential: Credential | None) -> RadiusGatewayConfig | None:
    """The gateway config cached in an OAuth credential by the pre-ModelsStore Radius implementation."""
    return _sanitize_radius_gateway_config(credential.get("gatewayConfig") if credential else None)


def get_radius_models_from_config(provider_id: str, config: RadiusGatewayConfig) -> list[Model]:
    models: list[Model] = []
    for entry in config["models"]:
        data = {key: value for key, value in entry.items() if key != "type"}
        data.update(api="pi-messages", provider=provider_id, baseUrl=config["baseUrl"])
        models.append(model_from_dict(data))
    return models


def get_radius_models(provider_id: str, credential: Credential | None) -> list[Model]:
    config = get_radius_credential_config(credential)
    return get_radius_models_from_config(provider_id, config) if config is not None else []


def _truncate_http_body(body: str) -> str:
    trimmed = body.strip()
    return f"{trimmed[:512]}…" if len(trimmed) > 512 else trimmed


async def load_radius_gateway_config(
    gateway: str,
    api_key: str | None = None,
    signal: AbortSignal | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> RadiusGatewayConfig:
    """Fetch and validate ``<gateway>/v1/config``. Raises on HTTP errors and invalid configs."""
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    url = urllib.parse.urljoin(gateway, "/v1/config")
    client = http_client if http_client is not None else get_http_client(url)
    async with abort_scope(signal):
        response = await client.get(url, headers=headers)
        raw = await response.aread()
    if not response.is_success:
        text = raw.decode("utf-8", "replace")
        raise RuntimeError(
            f"Could not load Radius config from {gateway}: {response.status_code}: {_truncate_http_body(text)}"
        )
    config = _sanitize_radius_gateway_config(json.loads(raw))
    if config is None:
        raise ValueError(f"Invalid Radius config from {gateway}")
    return config
