"""Caller cancellation must survive joins on child tasks the framework cancelled.

Every interleaving here is built from events, never from racing sleeps: the
child signals when it is handling its own cancellation, and the test cancels
the caller at exactly that point.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

import pytest

from llm_minder import (
    AdmissionRefused,
    GuardrailConfig,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
)
from llm_minder._internal.capacity import CapacityLimiter
from llm_minder._internal.cleanup import CleanupInterruptedError, cancel_and_join
from llm_minder._internal.guardrails import AbortCause, AbortController, await_with_guardrails
from llm_minder.core import AbortMode
from llm_minder.llm_strategies import LLMCallStrategy
from llm_minder.observers import BaseObserver, ProcessingEvent
from llm_minder.testing import FakeStrategy

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


async def test_guardrail_cancel_branch_join_waits_for_child_when_cancelled_twice(child) -> None:
    # Issue #384: the caller-cancellation branch joins the cancelled provider
    # call. A second caller cancel during that join must not let the caller
    # finish, or release the active call, before the call settles.
    controller = AbortController(AbortMode.CANCEL_ACTIVE)

    async def caller() -> str:
        return await await_with_guardrails(
            child.run(),
            item_deadline=None,
            item_id="x",
            abort_controller=controller,
            active_provider=True,
        )

    task = asyncio.create_task(caller())
    await asyncio.wait_for(child.started.wait(), GUARD)
    task.cancel()  # lands in the control wait: the cancellation branch
    await asyncio.wait_for(child.cancelling.wait(), GUARD)
    task.cancel()  # during the join; forwarded to the still-running call
    await asyncio.wait_for(child.cancelled_again.wait(), GUARD)
    for _ in range(3):
        await asyncio.sleep(0)
    assert not task.done()  # still joining: the child has not settled
    assert controller._active_provider_calls == 1
    child.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, GUARD)
    assert controller._active_provider_calls == 0


class _SlowToCancelEvent(asyncio.Event):
    """An abort event whose ``wait()`` keeps running for a while after cancellation.

    It stands in for the guardrail's abort-watch control task, so the cleanup
    ``gather`` over the controls stays pending until the test releases it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cancelling = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> Literal[True]:
        try:
            return await super().wait()
        except asyncio.CancelledError:
            self.cancelling.set()
            await self.release.wait()
            raise


@pytest.mark.parametrize("outcome", ["success", "error"])
async def test_cancel_during_control_cleanup_releases_provider_call(outcome: str) -> None:
    # Issue #379: a cancellation while the finally block joins the control tasks
    # must not leave the provider call counted as active.
    controller = AbortController(AbortMode.CANCEL_ACTIVE)
    slow_event = _SlowToCancelEvent()
    controller.event = slow_event

    async def provider() -> str:
        if outcome == "error":
            raise RuntimeError("provider failed")
        return "ok"

    async def caller() -> str:
        return await await_with_guardrails(
            provider(),
            item_deadline=None,
            item_id="x",
            abort_controller=controller,
            active_provider=True,
        )

    task = asyncio.create_task(caller())
    try:
        # The provider call finished; the guardrail is joining the abort watcher.
        await asyncio.wait_for(slow_event.cancelling.wait(), GUARD)
        assert controller.provider_calls_started == 1
        task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
    finally:
        slow_event.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, GUARD)
    assert controller._active_provider_calls == 0
    assert controller.provider_calls_started == 1


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


# --------------------------------------------------------------------------- #
# Abort announcements (#384)
# --------------------------------------------------------------------------- #

_USAGE = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _BlockOnAbort(BaseObserver):
    """Holds the first ``BATCH_ABORTED`` emit open until released."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def on_event(self, event: ProcessingEvent, data: dict) -> None:
        if event is ProcessingEvent.BATCH_ABORTED and not self.entered.is_set():
            self.entered.set()
            await self.release.wait()


class _RecordAborts(BaseObserver):
    def __init__(self) -> None:
        self.reasons: list[str] = []

    async def on_event(self, event: ProcessingEvent, data: dict) -> None:
        if event is ProcessingEvent.BATCH_ABORTED:
            self.reasons.append(data["reason"])


async def _wait_until_announced_or_done(blocker: _BlockOnAbort, run: asyncio.Task) -> None:
    entered = asyncio.create_task(blocker.entered.wait())
    try:
        await asyncio.wait({entered, run}, timeout=GUARD, return_when=asyncio.FIRST_COMPLETED)
    finally:
        entered.cancel()


class _SettleFails:
    """Admits everything; ``settle`` raises, which stops the run after the call."""

    def admit(self, request: object) -> object:
        return None

    def settle(self, ticket: object, usage: object, success: bool) -> None:
        raise RuntimeError("settle failed")


class _RefuseAll:
    def admit(self, request: object) -> object:
        raise AdmissionRefused("no")

    def settle(self, ticket: object, usage: object, success: bool) -> None:
        pass


class _BlockingStrategy(LLMCallStrategy[str]):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute(self, prompt, attempt, timeout, state=None):
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.parametrize("stop", ["budget", "call_cap", "admission_refused", "settle_failed"])
async def test_cancel_during_abort_announcement_still_emits_it(stop: str) -> None:
    # One cause per run (the first stop wins), announced from three places:
    # the budget step of the attempt's finally, the BatchAbortedError handler
    # (call cap, admission refusal) and the controller step of the finally
    # (settle failure). A cancellation while an observer handles the event
    # must not stop the remaining observers from receiving it.
    guardrails = {
        "budget": GuardrailConfig(max_total_tokens=1),
        "call_cap": GuardrailConfig(max_provider_calls=1),
        "admission_refused": GuardrailConfig(admission=_RefuseAll()),
        "settle_failed": GuardrailConfig(admission=_SettleFails()),
    }[stop]
    blocker, recorder = _BlockOnAbort(), _RecordAborts()
    strategy = FakeStrategy("ok", token_usage=_USAGE)
    async with ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, guardrails=guardrails),
        observers=[blocker, recorder],
    ) as processor:
        for item_id in ("a", "b"):
            await processor.add_work(LLMWorkItem(item_id=item_id, strategy=strategy, prompt="p"))
        run = asyncio.create_task(processor.process_all())
        try:
            await _wait_until_announced_or_done(blocker, run)
            assert blocker.entered.is_set(), "the stop was never announced"
            run.cancel()
            for _ in range(10):
                await asyncio.sleep(0)
            assert not run.done()  # the cancellation waits for the emit
            assert recorder.reasons == []
        finally:
            blocker.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, GUARD)
    assert len(recorder.reasons) == 1


async def test_cancel_during_call_then_second_cancel_abandons_announcement() -> None:
    # The attempt is cancelled mid-call, so its finally announces the settle
    # failure with a cancellation already propagating: that counts as the
    # first delivery, and the next cancellation abandons the emit.
    blocker, recorder = _BlockOnAbort(), _RecordAborts()
    strategy = _BlockingStrategy()
    async with ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, guardrails=GuardrailConfig(admission=_SettleFails())),
        observers=[blocker, recorder],
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="p"))
        run = asyncio.create_task(processor.process_all())
        try:
            await asyncio.wait_for(strategy.started.wait(), GUARD)
            run.cancel()
            await asyncio.wait_for(blocker.entered.wait(), GUARD)
            run.cancel()
            await asyncio.wait({run}, timeout=GUARD)  # hang guard only
            assert run.done(), "the second cancellation did not abandon the emit"
            with pytest.raises(asyncio.CancelledError):
                await run
        finally:
            blocker.release.set()
    assert recorder.reasons == []


async def test_nothing_to_announce_starts_no_cleanup(monkeypatch) -> None:
    from llm_minder._internal import item_executor

    calls: list[object] = []

    async def fail(*args: object, **kwargs: object) -> None:
        calls.append(args)
        raise AssertionError("run_cleanup_steps called with nothing to announce")

    monkeypatch.setattr(item_executor, "run_cleanup_steps", fail)
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, guardrails=GuardrailConfig(max_total_tokens=100))
    )
    await processor.add_work(
        LLMWorkItem(item_id="a", strategy=FakeStrategy("ok", token_usage=_USAGE), prompt="p")
    )
    await processor.process_all()
    stats = await processor.get_stats()
    await processor.shutdown()
    assert calls == []
    assert stats["succeeded"] == 1


async def test_announce_stops_emits_every_cause_despite_a_cancellation() -> None:
    # Only one cause can win a run, so a processor never hands this helper more
    # than one. Drive it directly with all three to pin the step ordering.
    blocker, recorder = _BlockOnAbort(), _RecordAborts()
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(
            max_workers=1,
            guardrails=GuardrailConfig(max_total_tokens=1, min_cache_hit_rate=0.5),
        ),
        observers=[blocker, recorder],
    )
    controller = processor._abort_controller
    assert controller is not None
    controller._unannounced = AbortCause(kind="admission_refused", reason="controller")
    announce = asyncio.create_task(
        processor._executor._announce_stops(
            budget_cause=AbortCause(kind="budget_exceeded", reason="budget"),
            cache_cause=AbortCause(kind="fail_fast", reason="cache"),
            controller=controller,
            primary_exception=None,
        )
    )
    try:
        await asyncio.wait_for(blocker.entered.wait(), GUARD)
        announce.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
        assert not announce.done()
    finally:
        blocker.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(announce, GUARD)
    assert recorder.reasons == ["budget", "cache", "controller"]
    assert controller.take_pending_announcement() is None
    await processor.shutdown()


class _FatalObserverError(BaseException):
    """Not an ``Exception``: the event dispatcher does not swallow it."""


async def test_announce_stops_still_raises_an_emit_failure() -> None:
    class Raises(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict) -> None:
            if event is ProcessingEvent.BATCH_ABORTED:
                raise _FatalObserverError

    recorder = _RecordAborts()
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, guardrails=GuardrailConfig(max_total_tokens=1)),
        observers=[Raises(), recorder],
    )
    controller = processor._abort_controller
    assert controller is not None
    controller._unannounced = AbortCause(kind="admission_refused", reason="controller")
    with pytest.raises(_FatalObserverError):
        await processor._executor._announce_stops(
            budget_cause=AbortCause(kind="budget_exceeded", reason="budget"),
            controller=controller,
            primary_exception=None,
        )
    # The later cause is still attempted (its emit fails the same way).
    assert controller.take_pending_announcement() is None
    await processor.shutdown()


async def test_announce_stops_reports_an_observer_cancellation_as_interrupted() -> None:
    """An observer's own ``CancelledError`` is not the worker's cancellation.

    Under the cleanup contract it surfaces as ``CleanupInterruptedError`` after
    the remaining causes are emitted, instead of making the worker look cancelled.
    """

    class CancelsOnce(BaseObserver):
        def __init__(self) -> None:
            self.raised = False

        async def on_event(self, event: ProcessingEvent, data: dict) -> None:
            if event is ProcessingEvent.BATCH_ABORTED and not self.raised:
                self.raised = True
                raise asyncio.CancelledError

    recorder = _RecordAborts()
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, guardrails=GuardrailConfig(max_total_tokens=1)),
        observers=[CancelsOnce(), recorder],
    )
    controller = processor._abort_controller
    assert controller is not None
    controller._unannounced = AbortCause(kind="admission_refused", reason="controller")
    with pytest.raises(CleanupInterruptedError):
        await processor._executor._announce_stops(
            budget_cause=AbortCause(kind="budget_exceeded", reason="budget"),
            controller=controller,
            primary_exception=None,
        )
    assert asyncio.current_task().cancelling() == 0  # type: ignore[union-attr]
    # The cancelled emit was abandoned; the next cause was still emitted to everyone.
    assert "controller" in recorder.reasons
    assert controller.take_pending_announcement() is None
    await processor.shutdown()
