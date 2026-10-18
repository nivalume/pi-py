"""Loopback OAuth redirect handler shared by the browser sign-in flows.

The server is a minimal asyncio HTTP/1.1 listener (one request per connection)
running on the caller's event loop, so it is cancellable together with the login.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, ClassVar, Generic, TypeVar

from ...abort import AbortController, AbortSignal
from ...utils.oauth_page import oauth_error_html, oauth_success_html
from ..types import AuthPrompt, ProviderAuthInteraction
from ._common import OAuthError, ParsedUrl, login_cancelled, parse_url

__all__ = [
    "LoopbackHttpServer",
    "OAuthCallbackServer",
    "CallbackResult",
    "ManualInput",
    "start_oauth_callback_server",
    "wait_for_callback_or_manual_input",
]

T = TypeVar("T")

#: ``(method, url) -> (status, html)`` where ``url`` is the request target resolved against the server origin.
RequestHandler = Callable[[str, ParsedUrl | None], Awaitable[tuple[int, str]]]

_MAX_HEAD_BYTES = 64 * 1024


class _Connection:
    __slots__ = ("active", "writer")

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.active = False


class LoopbackHttpServer:
    """Tiny HTTP/1.1 server that answers every request with an HTML page and closes the connection."""

    def __init__(self, handler: RequestHandler, *, headers: Mapping[str, str] | None = None) -> None:
        self._handler = handler
        self._headers = dict(headers or {"content-type": "text/html; charset=utf-8"})
        self._server: asyncio.Server | None = None
        self._connections: set[_Connection] = set()
        self.port = 0
        self._origin = "http://localhost"

    async def listen(self, host: str, port: int) -> None:
        """Bind and start serving. Raises ``OSError`` (e.g. ``EADDRINUSE``) when the port is taken."""
        loop = asyncio.get_running_loop()
        # Bind the first resolved address only (like Node's `server.listen(port, host)`), so a
        # free port picked with `port=0` is the same for every client of this server.
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
        family, sock_type, proto, _, address = infos[0]
        sock = socket.socket(family, sock_type, proto)
        try:
            if os.name != "nt":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(address)
            sock.listen(511)
            sock.setblocking(False)
            self._server = await asyncio.start_server(self._serve, sock=sock, limit=_MAX_HEAD_BYTES)
        except BaseException:
            sock.close()
            raise
        self.port = sock.getsockname()[1]
        self._origin = f"http://{f'[{host}]' if ':' in host else host}:{self.port}"

    def close(self) -> None:
        """Stop accepting connections and drop idle ones; requests being handled complete."""
        if self._server is not None:
            self._server.close()
        for connection in list(self._connections):
            if not connection.active:
                self._drop(connection)

    def close_all_connections(self) -> None:
        for connection in list(self._connections):
            self._drop(connection)

    def _drop(self, connection: _Connection) -> None:
        self._connections.discard(connection)
        with contextlib.suppress(Exception):
            connection.writer.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = _Connection(writer)
        self._connections.add(connection)
        try:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                return
            connection.active = True
            request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
            parts = request_line.split(" ")
            if len(parts) != 3 or not parts[2].startswith("HTTP/"):
                status, html = 400, oauth_error_html("Bad request.")
            else:
                method, target = parts[0], parts[1]
                try:
                    status, html = await self._handler(method, parse_url(target, self._origin))
                except Exception:
                    status, html = 500, oauth_error_html("Internal error while processing the callback.")
            body = html.encode("utf-8")
            try:
                reason = HTTPStatus(status).phrase
            except ValueError:
                reason = ""
            lines = [f"HTTP/1.1 {status} {reason}"]
            lines.extend(f"{name}: {value}" for name, value in self._headers.items())
            lines.append(f"content-length: {len(body)}")
            lines.append("connection: close")
            writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
            await writer.drain()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            self._connections.discard(connection)
            with contextlib.suppress(Exception):
                writer.close()


class OAuthCallbackServer(Generic[T]):
    """A running loopback redirect handler. Create with :func:`start_oauth_callback_server`."""

    def __init__(
        self,
        *,
        provider_name: str,
        path: str,
        state: str | None,
        complete: Callable[[str], Awaitable[T]],
    ) -> None:
        self.redirect_uri = ""
        self._provider_name = provider_name
        self._path = path
        self._state = state
        self._complete = complete
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[T | None] = self._loop.create_future()
        # A cancelled or closed wait may never be observed.
        self._future.add_done_callback(_consume_future)
        self._claimed = False
        self._settled = False
        self._timer: asyncio.TimerHandle | None = None
        self._remove_abort: Callable[[], None] | None = None
        self._http = LoopbackHttpServer(
            self._handle, headers={"content-type": "text/html; charset=utf-8", "cache-control": "no-store"}
        )

    def _finish(self, *, value: T | None = None, error: BaseException | None = None) -> None:
        if self._settled:
            return
        self._settled = True
        if self._timer is not None:
            self._timer.cancel()
        if self._remove_abort is not None:
            self._remove_abort()
        if self._future.done():
            return
        if error is not None:
            self._future.set_exception(error)
        else:
            self._future.set_result(value)

    def _on_abort(self) -> None:
        def finish() -> None:
            self._finish(error=login_cancelled())

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            finish()
        elif not self._loop.is_closed():
            self._loop.call_soon_threadsafe(finish)

    async def _handle(self, method: str, url: ParsedUrl | None) -> tuple[int, str]:
        if method != "GET" or url is None or url.pathname != self._path:
            return 404, oauth_error_html("Callback route not found.")
        if self._state is not None and url.param("state") != self._state:
            return 400, oauth_error_html("State mismatch.")
        if self._claimed or self._settled:
            return 409, oauth_error_html("This sign-in has already been handled.")
        error = url.param("error")
        if error:
            description = url.param("error_description")
            if description is None:
                description = error
            self._finish(error=OAuthError(f"{self._provider_name} authorization failed: {description}"))
            return 400, oauth_error_html(f"{self._provider_name} authorization failed.", description)
        code = url.param("code")
        if not code:
            return 400, oauth_error_html("Missing authorization code.")
        self._claimed = True
        try:
            value = await self._complete(code)
        except Exception as failure:
            self._finish(error=failure)
            return 502, oauth_error_html(f"{self._provider_name} sign-in failed.", str(failure))
        self._finish(value=value)
        return 200, oauth_success_html(f"Signed in to {self._provider_name}. You may now close this page.")

    async def wait(self) -> T | None:
        """The result of ``complete``, or ``None`` after :meth:`cancel`.

        Raises when the provider redirects with an error, ``complete`` fails, the signal
        aborts, or the timeout elapses.
        """
        return await asyncio.shield(self._future)

    def cancel(self) -> None:
        """Stop waiting for the browser unless a callback is already being completed."""
        if not self._claimed:
            self._finish(value=None)

    def close(self) -> None:
        self._finish(error=OAuthError("OAuth callback server closed"))
        self._http.close()


async def start_oauth_callback_server(
    *,
    provider_name: str,
    host: str,
    port: int,
    path: str,
    complete: Callable[[str], Awaitable[T]],
    redirect_host: str | None = None,
    state: str | None = None,
    signal: AbortSignal | None = None,
    timeout_ms: float | None = None,
) -> OAuthCallbackServer[T]:
    """Listen for the OAuth redirect on ``host:port`` (``0`` picks a free port).

    ``provider_name`` is shown on the browser page. ``redirect_host`` replaces ``host``
    in ``redirect_uri`` (for example ``localhost``). ``state`` is the expected ``state``
    parameter (omit when the provider sends none). ``complete`` finishes the sign-in with
    the received code before the browser page is sent, so the page can show exchange
    failures; pass ``lambda code: code`` (async) to exchange the code later.
    """
    if signal is not None and signal.aborted:
        raise login_cancelled()
    server: OAuthCallbackServer[T] = OAuthCallbackServer(
        provider_name=provider_name, path=path, state=state, complete=complete
    )
    await server._http.listen(host, port)
    if signal is not None:
        server._remove_abort = signal.add_listener(server._on_abort)
    if timeout_ms is not None:
        server._timer = server._loop.call_later(
            timeout_ms / 1000, lambda: server._finish(error=OAuthError(f"{provider_name} sign-in timed out"))
        )
    effective_host = redirect_host if redirect_host is not None else host
    formatted = f"[{effective_host}]" if ":" in effective_host else effective_host
    server.redirect_uri = f"http://{formatted}:{server._http.port}{path}"
    return server


@dataclass(frozen=True, slots=True)
class CallbackResult:
    """The browser callback completed with ``value``."""

    value: Any
    type: ClassVar[str] = "callback"


@dataclass(frozen=True, slots=True)
class ManualInput:
    """The user pasted ``input`` (a code or redirect URL)."""

    input: str
    type: ClassVar[str] = "manual"


async def wait_for_callback_or_manual_input(
    interaction: ProviderAuthInteraction,
    callback: OAuthCallbackServer[Any] | None,
    *,
    message: str,
    placeholder: str,
) -> CallbackResult | ManualInput:
    """Wait for the browser callback, or for the user to paste the code or redirect URL.

    The paste prompt covers browsers that cannot reach the loopback server (for example
    over SSH). Without a callback server only the manual prompt is used. Once one side
    wins, the manual prompt's signal aborts so UIs can dismiss it.
    """
    manual_abort = AbortController()
    manual_errors: list[BaseException] = []

    async def manual() -> str | None:
        try:
            value = await interaction.prompt(
                AuthPrompt(type="manual_code", message=message, placeholder=placeholder, signal=manual_abort.signal)
            )
        except Exception as error:
            manual_errors.append(error)
            if callback is not None:
                callback.cancel()
            return None
        if callback is not None:
            callback.cancel()
        return value

    manual_task = asyncio.ensure_future(manual())
    try:
        value = await callback.wait() if callback is not None else None
        if manual_errors:
            raise manual_errors[0]
        if value is not None:
            return CallbackResult(value)
        entered = await manual_task
        if manual_errors:
            raise manual_errors[0]
        return ManualInput(entered or "")
    finally:
        manual_abort.abort()
        if not manual_task.done():
            # Prompts that ignore their signal would otherwise stay pending forever.
            manual_task.cancel()


def _consume_future(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()
