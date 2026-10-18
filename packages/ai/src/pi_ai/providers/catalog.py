"""Built-in model catalog data.

Catalog files (``data/<provider>.json``) are the generated model data shipped by
``@earendil-works/pi-ai``. Each file groups entries by API id and keys them as
``"<type>:<model id>"``. Files load on first access and are cached per process.

Returned models are shared instances, as in pi. Derive variants with
``dataclasses.replace(model, ...)`` instead of mutating them.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from functools import lru_cache
from importlib import resources
from typing import Any

from ..serialization import model_from_dict
from ..types import AnyModel, ClassifierModel, ImageModel, Model

__all__ = [
    "BUILTIN_PROVIDERS",
    "load_catalog",
    "catalog_models",
    "get_builtin_providers",
    "get_builtin_model",
    "get_builtin_image_model",
    "get_builtin_classifier_model",
    "get_builtin_models",
    "get_builtin_image_models",
    "get_builtin_classifier_models",
    "get_all_builtin_models",
    "get_builtin_model_data_generated_at",
]

#: Providers present in the generated catalog.
BUILTIN_PROVIDERS: tuple[str, ...] = (
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "azure",
    "baseten",
    "cerebras",
    "cloudflare-ai-gateway",
    "cloudflare-workers-ai",
    "deepseek",
    "fireworks",
    "github-copilot",
    "google",
    "google-vertex",
    "groq",
    "huggingface",
    "kimi-coding",
    "meta",
    "minimax",
    "minimax-cn",
    "mistral",
    "moonshotai",
    "moonshotai-cn",
    "nvidia",
    "openai",
    "openai-codex",
    "opencode",
    "opencode-go",
    "openrouter",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "qwen-token-plan-individual",
    "radius",
    "together",
    "typesafe",
    "vercel-ai-gateway",
    "xai",
    "xiaomi",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-sgp",
    "zai",
    "zai-coding-cn",
)

_lock = threading.Lock()
_cache: dict[str, dict[str, dict[str, AnyModel]]] = {}


def _read_json(name: str) -> Any:
    data_dir = resources.files("pi_ai.providers").joinpath("data")
    with data_dir.joinpath(name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_catalog(provider: str) -> dict[str, dict[str, AnyModel]]:
    """Catalog of ``provider`` as ``{"chat": {id: model}, "image": {...}, "classifier": {...}}``."""
    cached = _cache.get(provider)
    if cached is not None:
        return cached
    with _lock:
        cached = _cache.get(provider)
        if cached is not None:
            return cached
        groups: dict[str, dict[str, AnyModel]] = {"chat": {}, "image": {}, "classifier": {}}
        try:
            raw = _read_json(f"{provider}.json")
        except FileNotFoundError:
            raw = {}
        for entries in raw.values():
            for key, value in entries.items():
                model_type, _, _ = key.partition(":")
                group = groups.get(model_type or "chat")
                if group is None:
                    continue
                model = model_from_dict(value)
                group[model.id] = model
        _cache[provider] = groups
        return groups


def catalog_models(provider: str) -> list[AnyModel]:
    """Every built-in model of ``provider`` (chat, then image, then classifier)."""
    groups = load_catalog(provider)
    return [*groups["chat"].values(), *groups["image"].values(), *groups["classifier"].values()]


def get_builtin_providers() -> list[str]:
    return list(BUILTIN_PROVIDERS)


def get_builtin_model(provider: str, model_id: str) -> Model | None:
    """One generated built-in chat model, independent of any ``Models`` collection."""
    return load_catalog(provider)["chat"].get(model_id)  # type: ignore[return-value]


def get_builtin_image_model(provider: str, model_id: str) -> ImageModel | None:
    return load_catalog(provider)["image"].get(model_id)  # type: ignore[return-value]


def get_builtin_classifier_model(provider: str, model_id: str) -> ClassifierModel | None:
    return load_catalog(provider)["classifier"].get(model_id)  # type: ignore[return-value]


def get_builtin_models(provider: str) -> list[Model]:
    return list(load_catalog(provider)["chat"].values())  # type: ignore[arg-type]


def get_builtin_image_models(provider: str) -> list[ImageModel]:
    return list(load_catalog(provider)["image"].values())  # type: ignore[arg-type]


def get_builtin_classifier_models(provider: str) -> list[ClassifierModel]:
    return list(load_catalog(provider)["classifier"].values())  # type: ignore[arg-type]


def get_all_builtin_models(provider: str) -> list[AnyModel]:
    return catalog_models(provider)


@lru_cache(maxsize=1)
def get_builtin_model_data_generated_at() -> int | None:
    """Generation timestamp (ms) shared by all built-in provider catalogs."""
    try:
        manifest = _read_json("manifest.json")
        stamp = datetime.fromisoformat(manifest["generatedAt"].replace("Z", "+00:00"))
    except (FileNotFoundError, KeyError, ValueError):
        return None
    return int(stamp.timestamp() * 1000)
