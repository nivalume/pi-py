"""Azure OpenAI endpoint resolution (base URL, API version, deployment names)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

from ..types import StreamOptions
from ..utils.misc import get_provider_env_value

__all__ = [
    "DEFAULT_AZURE_API_VERSION",
    "AzureEndpointOptions",
    "AzureConfig",
    "resolve_deployment_name",
    "resolve_azure_base_url",
    "resolve_azure_config",
]

DEFAULT_AZURE_API_VERSION = "v1"

_TRAILING_SLASHES = re.compile(r"/+$")
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*$")
_SPECIAL_SCHEMES = frozenset({"http", "https", "ws", "wss", "ftp", "file"})


class AzureEndpointOptions(StreamOptions, total=False):
    """Azure models ship without a base URL: one resource per user, resolved per request."""

    azure_api_version: str
    azure_resource_name: str
    azure_base_url: str
    azure_deployment_name: str


class AzureConfig(NamedTuple):
    base_url: str
    api_version: str


def _parse_deployment_name_map(value: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not value:
        return mapping
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        model_id, _, deployment_name = trimmed.partition("=")
        # JavaScript's split("=", 2) keeps only the text up to a second "=".
        deployment_name = deployment_name.split("=", 1)[0]
        if not model_id or not deployment_name:
            continue
        mapping[model_id.strip()] = deployment_name.strip()
    return mapping


def resolve_deployment_name(model: Any, options: Mapping[str, Any] | None = None) -> str:
    """Deployment for ``model.id``: ``azure_deployment_name``, ``AZURE_OPENAI_DEPLOYMENT_NAME_MAP``, or the id."""
    options = options or {}
    explicit = options.get("azure_deployment_name")
    if explicit:
        return explicit
    mapped = _parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.get("env"))
    ).get(model.id)
    return mapped or model.id


def _normalize_azure_base_url(base_url: str) -> str:
    trimmed = _TRAILING_SLASHES.sub("", base_url.strip())
    try:
        parts = urlsplit(trimmed)
    except ValueError as error:
        raise ValueError(f"Invalid Azure OpenAI base URL: {base_url}") from error
    scheme = parts.scheme.lower()
    if not scheme or not _URL_SCHEME.match(scheme) or (scheme in _SPECIAL_SCHEMES and not parts.netloc):
        raise ValueError(f"Invalid Azure OpenAI base URL: {base_url}")

    hostname = parts.hostname or ""
    is_azure_host = (
        hostname.endswith(".openai.azure.com")
        or hostname.endswith(".cognitiveservices.azure.com")
        or hostname.endswith(".ai.azure.com")
    )
    normalized_path = _TRAILING_SLASHES.sub("", parts.path)
    path = parts.path
    query = parts.query

    # Ensure Azure hosts have /openai/v1 as base path so /responses and ?api-version=v1
    # are appended correctly.
    if is_azure_host and normalized_path in ("", "/", "/openai", "/openai/v1/responses"):
        path = "/openai/v1"
        query = ""

    netloc = parts.netloc if "@" in parts.netloc else parts.netloc.lower()
    if not path and scheme in _SPECIAL_SCHEMES:
        path = "/"
    url = urlunsplit((scheme, netloc, path, query, parts.fragment))
    return _TRAILING_SLASHES.sub("", url)


def _build_default_base_url(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def resolve_azure_base_url(model: Any, options: Mapping[str, Any] | None = None) -> str:
    """Base URL from ``azure_base_url``, ``AZURE_OPENAI_BASE_URL``, the resource name, or ``model.base_url``."""
    options = options or {}
    env = options.get("env")
    explicit = (options.get("azure_base_url") or "").strip()
    base_url = explicit or (get_provider_env_value("AZURE_OPENAI_BASE_URL", env) or "").strip() or None
    resource_name = options.get("azure_resource_name") or get_provider_env_value("AZURE_OPENAI_RESOURCE_NAME", env)

    resolved_base_url = base_url
    if not resolved_base_url and resource_name:
        resolved_base_url = _build_default_base_url(resource_name)
    if not resolved_base_url and getattr(model, "base_url", None):
        resolved_base_url = model.base_url
    if not resolved_base_url:
        raise ValueError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, "
            "or pass azure_base_url, azure_resource_name, or model.base_url."
        )
    return _normalize_azure_base_url(resolved_base_url)


def resolve_azure_config(model: Any, options: Mapping[str, Any] | None = None) -> AzureConfig:
    """Base URL and API version (``azure_api_version``, ``AZURE_OPENAI_API_VERSION``, or ``"v1"``)."""
    options = options or {}
    return AzureConfig(
        base_url=resolve_azure_base_url(model, options),
        api_version=options.get("azure_api_version")
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", options.get("env"))
        or DEFAULT_AZURE_API_VERSION,
    )
