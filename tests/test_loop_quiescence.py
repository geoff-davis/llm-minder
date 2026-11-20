"""The stateful machines' settling helper (#327)."""

from __future__ import annotations

import asyncio

import pytest

from tests._loop_quiescence import run_until_quiescent


async def test_waits_out_a_cascade_longer_than_any_fixed_drain() -> None:
    done = asyncio.Event()

    async def chain(depth: int) -> None:
        for _ in range(depth):
            await asyncio.sleep(0)
        done.set()

    task = asyncio.create_task(chain(200))
    await run_until_quiescent()
    assert done.is_set()
    await task


async def test_a_busy_loop_fails_with_a_diagnostic() -> None:
    async def spin() -> None:
        while True:
            await asyncio.sleep(0)

    task = asyncio.create_task(spin())
    try:
        with pytest.raises(AssertionError, match="did not quiesce within 50 turns"):
            await run_until_quiescent(max_turns=50)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
