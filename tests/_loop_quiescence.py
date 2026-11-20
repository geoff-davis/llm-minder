"""Run an event loop until nothing on it can make progress (#327).

The stateful machines check invariants such as "no waiter is stranded" between
rules. A fixed number of ``sleep(0)`` turns is not enough for that: a cascade of
cancellations and lock hand-offs can need more turns than any fixed bound, and an
invariant checked mid-cascade reports a stall that the unchanged code would have
resolved a few turns later.

``run_until_quiescent`` yields until the loop has no ready callback and no timer
already due, for a few consecutive turns. Nothing can then change until the test
acts again (resolves a future, advances a fake clock, cancels a task), so an
invariant failure is a real stall. A bound turns a genuine busy loop into a
diagnostic failure instead of a hang.
"""

from __future__ import annotations

import asyncio
from typing import Any

_STABLE_TURNS = 3
_MAX_TURNS = 10_000


def _runnable(loop: asyncio.AbstractEventLoop) -> list[Any]:
    # BaseEventLoop internals, present on every supported CPython (3.11 to 3.15).
    ready = list(loop._ready)  # type: ignore[attr-defined]
    now = loop.time()
    due = [
        handle
        for handle in loop._scheduled  # type: ignore[attr-defined]
        if not handle.cancelled() and handle.when() <= now
    ]
    return ready + due


async def run_until_quiescent(
    *, stable_turns: int = _STABLE_TURNS, max_turns: int = _MAX_TURNS
) -> None:
    """Yield until no callback is ready and no timer is due for ``stable_turns``
    consecutive turns; fail with the pending callbacks after ``max_turns``."""
    loop = asyncio.get_running_loop()
    stable = 0
    pending: list[Any] = []
    for _ in range(max_turns):
        await asyncio.sleep(0)
        pending = _runnable(loop)
        stable = stable + 1 if not pending else 0
        if stable >= stable_turns:
            return
    shown = ", ".join(repr(handle) for handle in pending[:5])
    raise AssertionError(
        f"event loop did not quiesce within {max_turns} turns; still runnable: {shown}"
    )
