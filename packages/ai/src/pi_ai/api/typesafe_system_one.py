"""TypeSafe's native System One protocol.

OpenRouter (and other gateways) serve the same protocol, so those providers use
this API with different base URLs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

from ..types import ClassifierContext, ClassifierModel, ClassifierResult
from .classifier_shared import is_record
from .system_one_shared import SystemOneTransport, SystemOneWireRequest, classify_system_one

__all__ = ["classify"]


def _url(model: ClassifierModel) -> str:
    return urljoin(f"{model.base_url.rstrip('/')}/", "systemone")


def _payload(model: ClassifierModel, request: SystemOneWireRequest) -> Any:
    return {"model": model.id, **request}


def _output(body: Any) -> Mapping[str, Any]:
    if not is_record(body):
        raise ValueError("System One API returned an unexpected response")
    return body


_TRANSPORT = SystemOneTransport(
    api="typesafe-system-one", label="System One API", url=_url, payload=_payload, output=_output
)


async def classify(
    model: ClassifierModel, context: ClassifierContext, options: Mapping[str, Any] | None = None
) -> ClassifierResult:
    """TypeSafe System One classification with public ``bool`` values mapped to wire-level ``noul``."""
    return await classify_system_one(_TRANSPORT, model, context, options)
