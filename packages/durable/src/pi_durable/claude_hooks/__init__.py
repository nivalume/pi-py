"""Claude Code's command and HTTP hooks for the Harness (a port of delta's ``claude-hooks``).

:class:`ClaudeHooks` reads the ``hooks`` object of Claude Code's settings (:class:`HooksConfig`) and runs its handlers
from the Harness's hook seams: an ordinary extension with tool, generation and compaction hooks, plus an approval
policy decorator for ``PermissionRequest``. Commands run through the conversation's execution environment with the
event's JSON on standard input; HTTP handlers get it as a POST body. Every event's outcomes are memoized in the hook's
task, so a phase that runs again after a restart does not run its handlers again::

    import json
    from pi_durable.approval import AskAll
    from pi_durable.claude_hooks import ClaudeHooks, HooksConfig

    hooks = ClaudeHooks(HooksConfig.from_settings(json.load(open(".claude/settings.json"))))
    registry.install(hooks.extension())
    options["approval"] = hooks.permission_policy(AskAll())  # PermissionRequest answers what the policy would ask

Not imported by :mod:`pi_durable`.
"""

from __future__ import annotations

from .config import (
    DEFAULT_TIMEOUT_SECONDS,
    SUPPORTED_EVENTS,
    CommandHook,
    ConfigError,
    HookCommand,
    HooksConfig,
    HttpHook,
    Matcher,
    MatcherGroup,
)
from .extension import CLAUDE_HOOKS, ClaudeHooks, ClaudeHooksOptions
from .protocol import HookOutput, claude_tool_name
from .runner import Outcome

__all__ = [
    "CLAUDE_HOOKS",
    "DEFAULT_TIMEOUT_SECONDS",
    "SUPPORTED_EVENTS",
    "ClaudeHooks",
    "ClaudeHooksOptions",
    "CommandHook",
    "ConfigError",
    "HookCommand",
    "HookOutput",
    "HooksConfig",
    "HttpHook",
    "Matcher",
    "MatcherGroup",
    "Outcome",
    "claude_tool_name",
]
