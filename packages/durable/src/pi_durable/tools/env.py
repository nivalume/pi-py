"""The execution environment of a tool call (port of ``tools/env.ts``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..env.types import ExecutionEnv
    from ..harness.types import ToolExecutionApi

__all__ = ["require_env"]


def require_env(api: ToolExecutionApi) -> ExecutionEnv:
    """The call's execution environment; a tool without one fails with an ordinary error result."""
    env = api.env
    if env is None:
        raise RuntimeError("No execution environment is configured")
    return env
