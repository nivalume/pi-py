"""Tests for the ``pi-ai`` command line (pi_ai.cli)."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest

from pi_ai import (
    AuthEvent,
    AuthPrompt,
    AuthSelectOption,
    Credential,
    FileCredentialStore,
    LoginOptions,
    ModelAuth,
    OAuthAuth,
    ProviderAuth,
    ProviderAuthInteraction,
    create_models,
    create_provider,
    env_api_key_auth,
)
from pi_ai.abort import AbortController
from pi_ai.cli import _ConsoleInteraction, _LineReader, main, run


class _NoApi:
    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    stream_simple = stream


class ScriptedLines:
    def __init__(self, *lines: str) -> None:
        self.lines = list(lines)

    async def readline(self) -> str:
        if not self.lines:
            await asyncio.Event().wait()
        return self.lines.pop(0)


class FakeOAuth(OAuthAuth):
    name = "Fake OAuth"

    def __init__(self) -> None:
        self.options: LoginOptions | None = None

    async def login(self, interaction: ProviderAuthInteraction, options: LoginOptions | None = None) -> Credential:
        self.options = options
        method = await interaction.prompt(
            AuthPrompt(
                type="select",
                message="Pick:",
                options=[AuthSelectOption("a", "Option A"), AuthSelectOption("b", "Option B")],
            )
        )
        interaction.notify(AuthEvent(type="auth_url", url="https://example.com/auth", instructions="Do it."))
        interaction.notify(AuthEvent(type="device_code", user_code="ABCD", verification_uri="https://example.com/dev"))
        interaction.notify(AuthEvent(type="progress", message="Working..."))
        code = await interaction.prompt(AuthPrompt(type="text", message="Code", placeholder="xyz"))
        return {"type": "oauth", "access": f"{method}:{code}", "refresh": "r", "expires": 1}

    async def refresh(self, credential: Credential, signal: Any) -> Credential:
        return credential

    async def to_auth(self, credential: Credential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


def make_models(tmp_path: Path, oauth: OAuthAuth) -> Any:
    models = create_models(credentials=FileCredentialStore(tmp_path / "auth.json"))
    models.set_provider(create_provider(id="fake", name="Fake Provider", auth=ProviderAuth(oauth=oauth), api=_NoApi()))
    models.set_provider(
        create_provider(id="keyonly", auth=ProviderAuth(api_key=env_api_key_auth("Key", ["KEY"])), api=_NoApi())
    )
    return models


async def test_help_and_list_show_only_oauth_providers(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    models = make_models(tmp_path, FakeOAuth())
    await run([], models=models)
    out = capsys.readouterr().out
    assert out.startswith("Usage: pi-ai <command> [provider]")
    assert "  login [provider]  Login to an OAuth provider" in out
    assert f"  {'fake':<20} Fake Provider" in out
    assert "keyonly" not in out

    await run(["list"], models=models)
    assert capsys.readouterr().out == f"{'fake':<20} Fake Provider\n"


async def test_login_prompts_and_saves_credentials(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    oauth = FakeOAuth()
    opened: list[str] = []
    models = make_models(tmp_path, oauth)
    await run(["login", "fake"], models=models, lines=ScriptedLines("2", "the-code"), open_browser=opened.append)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "\nPick:\n  1. Option A\n  2. Option B\nEnter number (1-2): " in out
    assert "Open this URL in your browser:\nhttps://example.com/auth\nDo it." in out
    assert "Enter code: ABCD" in out
    assert "Working..." in out
    assert "Code (xyz): " in out
    assert "Credentials saved to auth.json" in out
    assert opened == ["https://example.com/auth", "https://example.com/dev"]
    saved = json.loads((tmp_path / "auth.json").read_text())
    assert saved["fake"] == {"type": "oauth", "access": "b:the-code", "refresh": "r", "expires": 1}
    assert oauth.options is not None and oauth.options.get_device_id is not None
    assert len(oauth.options.get_device_id()) == 36


async def test_login_interactive_provider_selection(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    models = make_models(tmp_path, FakeOAuth())
    await run(["login"], models=models, lines=ScriptedLines("1", "1", "c"), open_browser=lambda url: None)  # type: ignore[arg-type]
    assert "  1. Fake Provider\nEnter number (1-1): " in capsys.readouterr().out
    assert json.loads((tmp_path / "auth.json").read_text())["fake"]["access"] == "a:c"


async def test_unknown_provider_command_and_selection(tmp_path: Path):
    models = make_models(tmp_path, FakeOAuth())
    with pytest.raises(ValueError, match="Unknown provider: nope"):
        await run(["login", "nope"], models=models)
    with pytest.raises(ValueError, match="Unknown provider: keyonly"):
        await run(["login", "keyonly"], models=models)
    with pytest.raises(ValueError, match="Unknown command: frobnicate"):
        await run(["frobnicate"], models=models)
    with pytest.raises(ValueError, match="Invalid selection"):
        await run(["login", "fake"], models=models, lines=ScriptedLines("9"))  # type: ignore[arg-type]


async def test_dismissed_prompt_stops_waiting(capsys: pytest.CaptureFixture[str]):
    console = _ConsoleInteraction(ScriptedLines())  # type: ignore[arg-type]
    controller = AbortController()
    task = asyncio.ensure_future(
        console.prompt(AuthPrompt(type="manual_code", message="Paste", signal=controller.signal))
    )
    await asyncio.sleep(0)
    controller.abort()
    with pytest.raises(Exception, match="aborted"):
        await task
    assert capsys.readouterr().out == "Paste: \n"


async def test_line_reader_hands_lines_to_live_waiters():
    stream = io.StringIO("first\nsecond\n")
    reader = _LineReader(lambda: stream)
    assert await reader.readline() == "first"
    assert await reader.readline() == "second"
    with pytest.raises(EOFError):
        await reader.readline()


def test_main_reports_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    async def failing(argv: Any) -> None:
        raise ValueError("Unknown command: x")

    monkeypatch.setattr("pi_ai.cli._main", failing)
    with pytest.raises(SystemExit) as info:
        main(["x"])
    assert info.value.code == 1
    assert capsys.readouterr().err == "Error: Unknown command: x\n"


def test_main_lists_builtin_oauth_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(tmp_path)
    main(["list"])
    out = capsys.readouterr().out
    for provider_id in ("anthropic", "github-copilot", "openai-codex", "openrouter", "xai"):
        assert provider_id in out
