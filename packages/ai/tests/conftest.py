"""Shared test helpers: an in-process HTTP mock for provider adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest


def sse_body(events: Iterable[Any]) -> bytes:
    """Encode SSE events.

    Each item is a dict (sent as ``data: <json>``), a ``(event_name, dict | str)`` tuple,
    or a raw string (sent as a ``data:`` line verbatim, e.g. ``"[DONE]"``).
    """
    chunks: list[str] = []
    for item in events:
        if isinstance(item, tuple):
            name, data = item
            payload = data if isinstance(data, str) else json.dumps(data)
            chunks.append(f"event: {name}\ndata: {payload}\n\n")
        elif isinstance(item, str):
            chunks.append(f"data: {item}\n\n")
        else:
            chunks.append(f"data: {json.dumps(item)}\n\n")
    return "".join(chunks).encode()


@dataclass
class RecordedRequest:
    method: str
    url: str
    headers: httpx.Headers
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body) if self.body else None


@dataclass
class MockServer:
    """Queue of canned responses; records every request.

    ``responses`` items are ``httpx.Response`` objects or callables taking the
    ``httpx.Request`` and returning a response.
    """

    responses: list[Any] = field(default_factory=list)
    requests: list[RecordedRequest] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(RecordedRequest(request.method, str(request.url), request.headers, request.read()))
        if not self.responses:
            return httpx.Response(500, json={"error": {"message": "no mock response queued"}})
        response = self.responses.pop(0)
        return response(request) if callable(response) else response

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def add_sse(self, events: Iterable[Any], status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.responses.append(
            httpx.Response(
                status,
                content=sse_body(events),
                headers={"content-type": "text/event-stream", **(headers or {})},
            )
        )

    def add_json(self, body: Any, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.responses.append(httpx.Response(status, json=body, headers=headers))

    def add(self, response: httpx.Response | Callable[[httpx.Request], httpx.Response]) -> None:
        self.responses.append(response)

    @property
    def last(self) -> RecordedRequest:
        return self.requests[-1]


@pytest.fixture
def mock_server() -> MockServer:
    return MockServer()
