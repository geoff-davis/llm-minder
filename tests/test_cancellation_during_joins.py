"""Caller cancellation must survive joins on child tasks the framework cancelled.

Every interleaving here is built from events, never from racing sleeps: the
child signals when it is handling its own cancellation, and the test cancels
the caller at exactly that point.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_minder import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from llm_minder._internal.capacity import CapacityLimiter
from llm_minder._internal.cleanup import cancel_and_join
from llm_minder._internal.guardrails import AbortCause, AbortController, await_with_guardrails
from llm_minder.core import AbortMode
from llm_minder.llm_strategies import LLMCallStrategy

GUARD = 5.0  # hang guard only; never part of an interleaving


class _SlowToCancel:
    """A child that keeps running for a while after it is cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelling = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled_again = asyncio.Event()

    async def run(self) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelling.set()
            # Cleanup that must finish (e.g. a transport closing on cancel);
            # further cancellations do not cut it short.
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled_again.set()
            raise
        return "unreachable"


@pytest.fixture
def child():
    slow = _SlowToCancel()
    yield slow
    slow.release.set()  # never leave a cancellation-resistant child running


# --------------------------------------------------------------------------- #
# await_with_guardrails
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("trigger", ["operation_timer", "item_deadline", "batch_abort"])
async def test_guardrail_join_propagates_caller_cancellation(trigger: str, child) -> None:
    controller = AbortController(AbortMode.CANCEL_ACTIVE) if trigger == "batch_abort" else None

    async def caller() -> str:
        return await await_with_guardrails(
            child.run(),
            item_deadline=time.perf_counter() + 0.05 if trigger == "item_deadline" else None,
            item_id="x",
            operation_timeout=0.01 if trigger == "operation_timer" else None,
            abort_controller=controller,
        )

    task = asyncio.create_task(caller())
    await asyncio.wait_for(child.started.wait(), GUARD)
    if controller is not None:
        controller.trip_now(AbortCause(kind="fail_fast", reason="stop"))
    # The control fired and the guardrail is now joining the cancelled child.
    await asyncio.wait_for(child.cancelling.wait(), GUARD)
    task.cancel()
    child.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, GUARD)
    assert task.cancelled()


async def test_guardrail_join_waits_for_child_even_when_cancelled_twice(child) -> None:
    async def caller() -> str:
        return await await_with_guardrails(
            child.run(), item_deadline=None, item_id="x", operation_timeout=0.01
        )

    task = asyncio.create_task(caller())
    await asyncio.wait_for(child.cancelling.wait(), GUARD)
    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    assert not task.done()  # still joining: the child has not settled
    child.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, GUARD)


# --------------------------------------------------------------------------- #
# cancel_and_join
# --------------------------------------------------------------------------- #


async def test_cancel_and_join_consumes_child_cancellation() -> None:
    child_task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    await cancel_and_join(child_task)  # does not raise: not the caller's cancel
    assert child_task.cancelled()


async def test_cancel_and_join_propagates_caller_cancellation(child) -> None:
    child_task = asyncio.create_task(child.run())
    try:
        await asyncio.wait_for(child.started.wait(), GUARD)
        joiner = asyncio.create_task(cancel_and_join(child_task))
        await asyncio.wait_for(child.cancelling.wait(), GUARD)
        joiner.cancel()
        child.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(joiner, GUARD)
        assert child_task.done()
    finally:
        child.release.set()
        child_task.cancel()
        await asyncio.gather(child_task, return_exceptions=True)


async def test_cancel_and_join_propagates_cancellation_when_child_swallows_it() -> None:
    """A child that returns normally after cancellation must not absorb ours."""
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def swallowing_child() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
        return "late value"

    child_task = asyncio.create_task(swallowing_child())
    try:
        await asyncio.wait_for(started.wait(), GUARD)
        joiner = asyncio.create_task(cancel_and_join(child_task))
        await asyncio.wait_for(cancelling.wait(), GUARD)
        joiner.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(joiner, GUARD)
        assert child_task.result() == "late value"
    finally:
        release.set()
        child_task.cancel()
        await asyncio.gather(child_task, return_exceptions=True)


async def test_cancel_and_join_error_propagation() -> None:
    async def failing() -> None:
        raise ValueError("boom")

    failed = asyncio.create_task(failing())
    await asyncio.sleep(0)
    with pytest.raises(ValueError, match="boom"):
        await cancel_and_join(failed, propagate_errors=True)

    failed_again = asyncio.create_task(failing())
    await asyncio.sleep(0)
    await cancel_and_join(failed_again)  # discarded by default


class _Fatal(BaseException):
    """A non-Exception outcome that no join may discard."""


async def test_guardrail_join_propagates_fatal_base_exception_from_child() -> None:
    async def child() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            raise _Fatal("fatal during cancellation")

    with pytest.raises(_Fatal, match="fatal during cancellation"):
        await await_with_guardrails(
            child(), item_deadline=None, item_id="x", operation_timeout=0.01
        )


@pytest.mark.parametrize("propagate_errors", [False, True])
async def test_cancel_and_join_raises_process_control_outcome(propagate_errors: bool) -> None:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_exception(SystemExit(7))
    with pytest.raises(SystemExit) as raised:
        await cancel_and_join(future, propagate_errors=propagate_errors)
    assert raised.value.code == 7


def _child_ending_with(error: BaseException, cancelling: asyncio.Event, release: asyncio.Event):
    async def child() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            raise error from None

    return child()


@pytest.mark.parametrize("propagate_errors", [False, True])
async def test_cancel_and_join_fatal_outcome_outranks_caller_cancellation(
    propagate_errors: bool,
) -> None:
    cancelling, release = asyncio.Event(), asyncio.Event()
    child_task = asyncio.create_task(_child_ending_with(_Fatal("fatal"), cancelling, release))
    try:
        await asyncio.sleep(0)
        joiner = asyncio.create_task(cancel_and_join(child_task, propagate_errors=propagate_errors))
        await asyncio.wait_for(cancelling.wait(), GUARD)
        joiner.cancel()
        release.set()
        with pytest.raises(_Fatal) as raised:
            await asyncio.wait_for(joiner, GUARD)
        assert isinstance(raised.value.__context__, asyncio.CancelledError)
    finally:
        release.set()
        child_task.cancel()
        await asyncio.gather(child_task, return_exceptions=True)


@pytest.mark.parametrize("propagate_errors", [False, True])
async def test_cancel_and_join_caller_cancellation_outranks_ordinary_exception(
    propagate_errors: bool,
) -> None:
    cancelling, release = asyncio.Event(), asyncio.Event()
    child_task = asyncio.create_task(_child_ending_with(ValueError("late"), cancelling, release))
    try:
        await asyncio.sleep(0)
        joiner = asyncio.create_task(cancel_and_join(child_task, propagate_errors=propagate_errors))
        await asyncio.wait_for(cancelling.wait(), GUARD)
        joiner.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(joiner, GUARD)
    finally:
        release.set()
        child_task.cancel()
        await asyncio.gather(child_task, return_exceptions=True)


# --------------------------------------------------------------------------- #
# End to end: cancelling process_all() must stop retries
# --------------------------------------------------------------------------- #


async def test_cancelled_process_all_starts_no_further_provider_calls(fast_retry, child) -> None:
    calls = 0

    class Strategy(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            nonlocal calls
            calls += 1
            await child.run()
            return "x", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None

    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, attempt_timeout=0.01, retry=fast_retry)
    )
    await processor.add_work(LLMWorkItem(item_id="a", strategy=Strategy(), prompt="p"))
    run = asyncio.create_task(processor.process_all())
    # Attempt 1 timed out; the worker is joining its cancelled provider call.
    await asyncio.wait_for(child.cancelling.wait(), GUARD)
    run.cancel()
    # The worker is cancelled while still joining: the cancellation reaches
    # the provider call it is joining.
    await asyncio.wait_for(child.cancelled_again.wait(), GUARD)
    child.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, GUARD)
    await asyncio.wait_for(processor.shutdown(), GUARD)
    assert calls == 1


# --------------------------------------------------------------------------- #
# Capacity gate release
# --------------------------------------------------------------------------- #


class _Capacity1:
    max_concurrency = 1


async def test_capacity_release_cancelled_while_queued_for_lock_frees_slot() -> None:
    strategy = _Capacity1()
    limiter = CapacityLimiter(None, max_workers=4)
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def holder() -> None:
        async with limiter.admit(strategy):
            entered.set()
            await leave.wait()

    async def waiter() -> str:
        async with limiter.admit(strategy):
            return "admitted"

    holding = asyncio.create_task(holder())
    await asyncio.wait_for(entered.wait(), GUARD)
    gate = next(iter(limiter._entries.values()))[2]
    waiting = asyncio.create_task(waiter())
    for _ in range(5):
        await asyncio.sleep(0)
    assert not waiting.done()  # parked in condition.wait(), lock released

    # Contend the Condition lock so the holder's release() has to queue for it.
    await gate._condition.acquire()
    leave.set()
    for _ in range(3):
        await asyncio.sleep(0)
    assert not holding.done()  # queued for the lock inside release()
    holding.cancel()
    await asyncio.gather(holding, return_exceptions=True)
    gate._condition.release()
    assert gate._active == 0

    # The parked waiter must still be woken for the freed slot.
    assert await asyncio.wait_for(waiting, GUARD) == "admitted"
    assert gate._active == 0
