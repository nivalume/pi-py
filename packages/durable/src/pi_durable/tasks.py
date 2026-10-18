"""Task definitions (port of ``tasks.ts``)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from .types import PhaseHandler, Task, TaskDefinition, TaskMigration

__all__ = ["define_task"]

I = TypeVar("I")  # noqa: E741 - upstream's name for task inputs
S = TypeVar("S")
R = TypeVar("R")
H = TypeVar("H")


def define_task(
    *,
    name: str,
    version: int,
    initial: Callable[[I], S],
    phases: Mapping[str, PhaseHandler[I, S, R, H]],
    abort: PhaseHandler[I, S, R, H],
    migrate: Callable[[Any, Any, int], TaskMigration[I, S]] | None = None,
    hooks: H | None = None,
) -> Task[I, S, R, H]:
    """Define an executable task. Register it in the registry so a Harness can run tasks of its kind.

    The keyword arguments are upstream's ``TaskDefinition`` fields. Annotate the result to fix the
    result type ``R``, which no argument names::

        Reminder: Task[ReminderInput, ReminderCheckpoint, None, Any] = define_task(
            name="app.reminder", version=1, initial=..., phases={"wait": wait, "deliver": deliver}, abort=abort
        )
    """
    return Task(
        TaskDefinition(
            name=name, version=version, initial=initial, phases=phases, abort=abort, migrate=migrate, hooks=hooks
        )
    )
