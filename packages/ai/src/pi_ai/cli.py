"""``pi-ai`` command line: log in to the built-in OAuth providers.

``pi-ai login [provider]`` runs a provider's OAuth flow and saves the credential to
``auth.json`` in the current directory (the same format pi uses); ``pi-ai list``
lists the providers that support OAuth.
"""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import os
import re
import sys
import threading
import uuid
import webbrowser
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TextIO

from .abort import AbortController, AbortError, AbortSignal, abort_scope
from .auth.credential_store import FileCredentialStore
from .auth.types import AuthEvent, AuthPrompt, LoginOptions

if TYPE_CHECKING:
    from .models import Models, Provider

__all__ = ["main", "run"]

AUTH_FILE = "auth.json"


class _LineReader:
    """Reads stdin lines on a daemon thread so a dismissed prompt never blocks exit.

    A line goes to the oldest prompt still waiting; lines typed while no prompt waits
    are dropped. (``asyncio.to_thread(input)`` would keep the interpreter alive until
    Enter is pressed after the browser callback already finished the login.)
    """

    def __init__(self, stream: Callable[[], TextIO] = lambda: sys.stdin) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._waiters: deque[tuple[asyncio.AbstractEventLoop, asyncio.Future[str]]] = deque()
        self._reading = False

    async def readline(self) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        with self._lock:
            self._waiters.append((loop, future))
            start = not self._reading
            self._reading = True
        if start:
            threading.Thread(target=self._read_loop, name="pi-ai-stdin", daemon=True).start()
        return await future

    def _read_loop(self) -> None:
        while True:
            try:
                line: str | None = self._stream().readline()
            except Exception:
                line = ""
            with self._lock:
                while self._waiters and self._waiters[0][1].done():
                    self._waiters.popleft()
                if not line:
                    waiters = list(self._waiters)
                    self._waiters.clear()
                    self._reading = False
                else:
                    waiters = [self._waiters.popleft()] if self._waiters else []
                    if not self._waiters:
                        self._reading = False
                reading = self._reading
            for loop, future in waiters:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(_settle_line, future, line)
            if not reading:
                return


def _settle_line(future: asyncio.Future[str], line: str | None) -> None:
    if future.done():
        return
    if not line:
        future.set_exception(EOFError("No input (stdin closed)"))
    else:
        future.set_result(line.rstrip("\r\n"))


async def _read_secret(question: str) -> str:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def read() -> None:
        try:
            value = getpass.getpass(question)
        except BaseException as error:
            failure = error if isinstance(error, Exception) else EOFError("No input")
            loop.call_soon_threadsafe(lambda: future.done() or future.set_exception(failure))
            return
        loop.call_soon_threadsafe(lambda: future.done() or future.set_result(value))

    threading.Thread(target=read, name="pi-ai-getpass", daemon=True).start()
    return await future


def _parse_int(value: str) -> int | None:
    """``Number.parseInt(value, 10)``."""
    match = re.match(r"^\s*([+-]?\d+)", value)
    return int(match.group(1)) if match else None


def _can_open_browser() -> bool:
    # Without a display, `webbrowser` may start a text-mode browser inside this terminal.
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _open_browser(url: str) -> None:
    if not _can_open_browser():
        return
    with contextlib.suppress(Exception):
        webbrowser.open(url)


class _ConsoleInteraction:
    """Login interaction on stdin/stdout."""

    def __init__(self, lines: _LineReader, *, open_browser: Callable[[str], None] = _open_browser) -> None:
        self.signal: AbortSignal | None = AbortController().signal
        self._lines = lines
        self._open_browser = open_browser

    async def _ask(self, question: str, *, secret: bool = False, signal: Any = None) -> str:
        if secret:
            async with abort_scope(signal):
                return await _read_secret(question)
        print(question, end="", flush=True)
        try:
            async with abort_scope(signal):
                return await self._lines.readline()
        except (AbortError, asyncio.CancelledError):
            # The prompt was dismissed (for example the browser callback finished the login).
            print()
            raise

    async def prompt(self, prompt: AuthPrompt) -> str:
        if prompt.type == "select":
            options = prompt.options or []
            print(f"\n{prompt.message}")
            for index, option in enumerate(options):
                print(f"  {index + 1}. {option.label}")
            answer = await self._ask(f"Enter number (1-{len(options)}): ", signal=prompt.signal)
            choice = _parse_int(answer)
            if choice is None or not 1 <= choice <= len(options):
                raise ValueError("Invalid selection")
            return options[choice - 1].id
        suffix = f" ({prompt.placeholder})" if prompt.placeholder else ""
        return await self._ask(f"{prompt.message}{suffix}: ", secret=prompt.type == "secret", signal=prompt.signal)

    def notify(self, event: AuthEvent) -> None:
        if event.type == "auth_url":
            print(f"\nOpen this URL in your browser:\n{event.url}")
            if event.instructions:
                print(event.instructions)
            if event.url:
                self._open_browser(event.url)
        elif event.type == "device_code":
            print(f"\nOpen this URL in your browser:\n{event.verification_uri}")
            print(f"Enter code: {event.user_code}")
            if event.verification_uri:
                self._open_browser(event.verification_uri)
        elif event.message:
            print(event.message)


def _oauth_providers(models: Models) -> list[Provider]:
    return [provider for provider in models.get_providers() if provider.auth.oauth is not None]


def _usage(providers: Sequence[Provider]) -> str:
    provider_list = "\n".join(f"  {provider.id:<20} {provider.name}" for provider in providers)
    return (
        "Usage: pi-ai <command> [provider]\n\n"
        "Commands:\n"
        "  login [provider]  Login to an OAuth provider\n"
        "  list              List available providers\n\n"
        f"Providers:\n{provider_list}"
    )


async def _login(models: Models, provider_id: str, interaction: _ConsoleInteraction) -> None:
    # This CLI does not persist an installation ID; apps should reuse one across logins.
    await models.login(provider_id, "oauth", interaction, LoginOptions(get_device_id=lambda: str(uuid.uuid4())))
    print(f"\nCredentials saved to {AUTH_FILE}")


async def run(
    argv: Sequence[str],
    *,
    models: Models | None = None,
    lines: _LineReader | None = None,
    open_browser: Callable[[str], None] = _open_browser,
) -> None:
    """Run the CLI with ``argv`` (without the program name). Raises on errors."""
    if models is None:
        from .providers.all import builtin_models

        models = builtin_models(credentials=FileCredentialStore(AUTH_FILE))
    providers = _oauth_providers(models)
    reader = lines if lines is not None else _LineReader()

    command = argv[0] if argv else None
    if not command or command in ("help", "--help", "-h"):
        print(_usage(providers))
        return
    if command == "list":
        for provider in providers:
            print(f"{provider.id:<20} {provider.name}")
        return
    if command == "login":
        interaction = _ConsoleInteraction(reader, open_browser=open_browser)
        provider_id = argv[1] if len(argv) > 1 else None
        if not provider_id:
            for index, provider in enumerate(providers):
                print(f"  {index + 1}. {provider.name}")
            print(f"Enter number (1-{len(providers)}): ", end="", flush=True)
            choice = _parse_int(await reader.readline())
            provider_id = providers[choice - 1].id if choice is not None and 1 <= choice <= len(providers) else None
        if not provider_id or not any(provider.id == provider_id for provider in providers):
            raise ValueError(f"Unknown provider: {provider_id or ''}")
        await _login(models, provider_id, interaction)
        return
    raise ValueError(f"Unknown command: {command}")


async def _main(argv: Sequence[str]) -> None:
    from .utils.http import aclose_http_clients

    try:
        await run(argv)
    finally:
        await aclose_http_clients()


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``pi-ai`` script."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\nLogin cancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
