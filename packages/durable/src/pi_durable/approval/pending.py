"""``PendingApprovals``: the in-memory fast path from a deciding commit to the approval phase that waits for it
(Codex's ``pending_approvals`` map)."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from .document import ReviewDecision

__all__ = ["ApprovalWaiter", "PendingApprovals"]


class PendingApprovals:
    """The waiters of a Harness's approval phases, keyed by approval id. Never persisted: losing it loses nothing,
    because a waiting phase reads the committed decision after it registers."""

    __slots__ = ("_next", "_waiters")

    def __init__(self) -> None:
        self._waiters: dict[int, tuple[int, asyncio.Future[ReviewDecision | None]]] = {}
        self._next = itertools.count()

    def register(self, id: int) -> ApprovalWaiter:
        """A waiter for ``id``; replaces an older one (a rerun of the phase), whose wait ends with ``None``."""
        future: asyncio.Future[ReviewDecision | None] = asyncio.get_running_loop().create_future()
        token = next(self._next)
        previous = self._waiters.get(id)
        self._waiters[id] = (token, future)
        if previous is not None and not previous[1].done():
            previous[1].set_result(None)
        return ApprovalWaiter(self, id, token, future)

    def notify(self, id: int, decision: ReviewDecision) -> bool:
        """Hand ``decision`` to the waiter of ``id``, after the deciding commit settled; ``False`` when nobody waits."""
        waiter = self._waiters.pop(id, None)
        if waiter is None or waiter[1].done():
            return False
        waiter[1].set_result(decision)
        return True

    def waiting(self) -> list[int]:
        return sorted(self._waiters)

    def _release(self, id: int, token: int) -> None:
        waiter = self._waiters.get(id)
        if waiter is not None and waiter[0] == token:
            del self._waiters[id]

    def __repr__(self) -> str:
        return f"PendingApprovals(waiting={self.waiting()})"


class ApprovalWaiter:
    """A registered waiter; :meth:`close` removes its entry unless a newer one replaced it."""

    __slots__ = ("_future", "_pending", "_token", "id")

    def __init__(
        self, pending: PendingApprovals, id: int, token: int, future: asyncio.Future[ReviewDecision | None]
    ) -> None:
        self._pending = pending
        self.id = id
        self._token = token
        self._future = future

    def decision(self) -> asyncio.Future[ReviewDecision | None]:
        """The notified decision; ``None`` when the waiter was replaced."""
        return self._future

    def close(self) -> None:
        self._pending._release(self.id, self._token)
        if not self._future.done():
            self._future.cancel()

    def __enter__(self) -> ApprovalWaiter:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
