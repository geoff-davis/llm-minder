"""Unit tests for RateLimitCoordinator cooldown duration logic.

Exercises the coordinator's state machine in isolation (no processor), in
particular the ``suggested_wait`` floor added in v0.10.0: a server-suggested
wait (e.g. a parsed ``Retry-After``) raises the cooldown but never lowers it.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_minder import BaseObserver, ProcessingEvent
from llm_minder._internal.event_dispatcher import EventDispatcher
from llm_minder._internal.rate_limit_coordinator import RateLimitCoordinator
from llm_minder.strategies.rate_limit import FixedDelayStrategy


def _make_coordinator(cooldown: float) -> RateLimitCoordinator:
    events: EventDispatcher = EventDispatcher(observers=[], middlewares=[])
    return RateLimitCoordinator(FixedDelayStrategy(cooldown=cooldown), events)


async def _captured_cooldown(coord: RateLimitCoordinator, monkeypatch, **kwargs) -> float:
    """Run one cooldown cycle and return the duration passed to asyncio.sleep."""
    slept: list[float] = []

    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *a, **kw):
        slept.append(delay)
        await real_sleep(0)  # yield without actually waiting

    monkeypatch.setattr("llm_minder._internal.rate_limit_coordinator.asyncio.sleep", fake_sleep)
    await coord.handle_rate_limit(worker_id=0, observed_generation=0, **kwargs)
    return slept[0] if slept else 0.0


@pytest.mark.asyncio
async def test_suggested_wait_raises_cooldown(monkeypatch):
    coord = _make_coordinator(cooldown=5.0)
    duration = await _captured_cooldown(coord, monkeypatch, suggested_wait=30.0)
    assert duration == 30.0


@pytest.mark.asyncio
async def test_suggested_wait_below_cooldown_is_ignored(monkeypatch):
    coord = _make_coordinator(cooldown=20.0)
    duration = await _captured_cooldown(coord, monkeypatch, suggested_wait=5.0)
    assert duration == 20.0


@pytest.mark.asyncio
async def test_no_suggested_wait_uses_strategy_cooldown(monkeypatch):
    coord = _make_coordinator(cooldown=8.0)
    duration = await _captured_cooldown(coord, monkeypatch)
    assert duration == 8.0


# ── Issue #88: caller cancellation must not finish the shared cooldown ──


@pytest.mark.asyncio
async def test_cancelled_reporter_does_not_finish_shared_cooldown():
    """A gateway submit timeout cancels the caller that reported the 429;
    the shared cooldown must survive and other waiters stay paused."""
    coord = _make_coordinator(cooldown=0.5)

    reporter = asyncio.create_task(coord.handle_rate_limit(worker_id=0, observed_generation=0))
    await asyncio.sleep(0.05)  # cooldown underway
    assert coord._in_cooldown

    waiter = asyncio.create_task(coord.handle_rate_limit(worker_id=1, observed_generation=0))
    await asyncio.sleep(0.02)

    # Simulate the submit timeout: cancel only the reporting caller.
    reporter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reporter

    # The shared pause survives the caller's cancellation.
    await asyncio.sleep(0.1)
    assert coord._in_cooldown
    assert not coord._rate_limit_event.is_set()
    assert not waiter.done()

    # The waiter resumes only once the real cooldown expires.
    await asyncio.wait_for(waiter, timeout=1.0)
    assert not coord._in_cooldown
    assert coord._rate_limit_event.is_set()
    await coord.shutdown()


@pytest.mark.asyncio
async def test_two_waiters_survive_reporter_cancellation():
    """Regression per the issue: two waiters + a cancelled reporter."""
    coord = _make_coordinator(cooldown=0.4)
    started = asyncio.get_running_loop().time()

    reporter = asyncio.create_task(coord.handle_rate_limit(worker_id=0, observed_generation=0))
    await asyncio.sleep(0.02)
    waiters = [
        asyncio.create_task(coord.handle_rate_limit(worker_id=i, observed_generation=0))
        for i in (1, 2)
    ]
    await asyncio.sleep(0.02)
    reporter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reporter

    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)
    elapsed = asyncio.get_running_loop().time() - started
    # Both waiters waited out the REAL cooldown, not the 0.04s to cancellation.
    assert elapsed >= 0.35
    await coord.shutdown()


@pytest.mark.asyncio
async def test_shutdown_stops_cooldown_and_wakes_waiters():
    """Host teardown stops the owned task without leaking it; waiters wake."""
    coord = _make_coordinator(cooldown=30.0)

    reporter = asyncio.create_task(coord.handle_rate_limit(worker_id=0, observed_generation=0))
    await asyncio.sleep(0.05)
    task = next(iter(coord._owned_cooldowns), None)
    assert task is not None and not task.done()

    await asyncio.wait_for(coord.shutdown(), timeout=1.0)
    assert task.done()
    assert next(iter(coord._owned_cooldowns), None) is None
    # Waiters (including the reporter) are woken by teardown finalization.
    await asyncio.wait_for(reporter, timeout=1.0)
    assert coord._rate_limit_event.is_set()
    # Idempotent.
    await coord.shutdown()


@pytest.mark.asyncio
async def test_shutdown_without_cooldown_is_noop():
    coord = _make_coordinator(cooldown=1.0)
    await coord.shutdown()
    assert next(iter(coord._owned_cooldowns), None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suggestion,expected", [(86400, 0.1), (float("inf"), 0.01), (float("nan"), 0.01)]
)
async def test_prov5_coordinator_caps_untrusted_suggestion(monkeypatch, suggestion, expected):
    events = EventDispatcher(observers=[], middlewares=[])
    coord = RateLimitCoordinator(
        FixedDelayStrategy(cooldown=0.01), events, max_cooldown_seconds=0.1
    )
    assert await _captured_cooldown(coord, monkeypatch, suggested_wait=suggestion) == expected
    await coord.shutdown()


@pytest.mark.asyncio
async def test_adm5_failed_strategy_uses_fallback_and_suggested_wait(monkeypatch):
    class Broken(FixedDelayStrategy):
        async def on_rate_limit(self, worker_id, consecutive_limit_count):
            raise RuntimeError("broken policy")

    coord = RateLimitCoordinator(Broken(), EventDispatcher([], []))
    coord._fallback_cooldown_seconds = 5
    assert await _captured_cooldown(coord, monkeypatch, suggested_wait=8) == 8
    await coord.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -float("inf"), -5.0, 0.0, 0.25])
@pytest.mark.parametrize("suggested_wait", [None, 0.0, 8.0])
async def test_sd1_invalid_cooldown_return_uses_fallback(
    duration, suggested_wait, monkeypatch, caplog
):
    import math

    coord = RateLimitCoordinator(
        FixedDelayStrategy(cooldown=duration),
        EventDispatcher([], []),
        fallback_cooldown_seconds=5,
    )
    invalid = not math.isfinite(duration) or duration < 0
    expected = max(5 if invalid else duration, suggested_wait or 0)
    try:
        actual = await _captured_cooldown(coord, monkeypatch, suggested_wait=suggested_wait)
        assert actual == expected
        assert ("Using the configured fallback cooldown" in caplog.text) is invalid
        assert not coord.is_paused
    finally:
        await coord.shutdown()
    assert not coord._owned_cooldowns


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host_kind", ["direct", "batch_scope", "batch_compat", "executor_scope", "executor_compat"]
)
@pytest.mark.parametrize("strategy_result", ["raises", "nan", "valid"])
@pytest.mark.parametrize(
    "suggestion,clamped",
    [(None, 0), (-5, 0), (2, 2), (7, 7), (86400, 10), (float("nan"), 0), (float("inf"), 0)],
)
async def test_c4_d5_hint_clamp_and_fallback_across_hosts(
    host_kind, strategy_result, suggestion, clamped, monkeypatch
):
    from llm_minder import ParallelBatchProcessor, ProcessorConfig, RateLimitConfig
    from llm_minder._internal.executor_host import ExecutorHost
    from llm_minder.llm_strategies import LLMCallStrategy

    class CooldownPolicy(FixedDelayStrategy):
        async def on_rate_limit(self, worker_id, consecutive_limit_count):
            if strategy_result == "raises":
                raise RuntimeError("policy failed")
            return float("nan") if strategy_result == "nan" else 30.0

    class Strategy(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            return "ok", {}, None

    config = ProcessorConfig(
        rate_limit=RateLimitConfig(cooldown_seconds=5, max_cooldown_seconds=10)
    )
    policy = CooldownPolicy()
    strategy = Strategy()
    if host_kind == "direct":
        coord = RateLimitCoordinator(
            policy,
            EventDispatcher([], []),
            max_cooldown_seconds=10,
            fallback_cooldown_seconds=5,
        )
        close = coord.shutdown
    elif host_kind.startswith("batch"):
        processor = ParallelBatchProcessor(config=config, rate_limit_strategy=policy)
        coord = (
            processor._admission_registry.resolve(strategy).cooldown
            if host_kind == "batch_scope"
            else processor._compatibility_rate_limit_coord
        )
        close = processor.shutdown
    else:
        host = ExecutorHost(
            config,
            strategy=strategy if host_kind == "executor_scope" else None,
            rate_limit_strategy=policy,
        )
        coord = host._rate_limit_coord
        close = host.aclose
    try:
        duration = await _captured_cooldown(coord, monkeypatch, suggested_wait=suggestion)
        # Only the server hint is capped. Valid custom cooldowns may exceed it.
        assert duration == max(30 if strategy_result == "valid" else 5, clamped)
    finally:
        await close()


# ── Injected clock and cooldown sleep (#327) ────────────────────────────


class _CooldownEvents(BaseObserver):
    _abl_fast_observer = True

    def __init__(self) -> None:
        self.events: list[tuple[ProcessingEvent, dict]] = []

    async def on_event(self, event: ProcessingEvent, data: dict) -> None:
        if event in (ProcessingEvent.COOLDOWN_STARTED, ProcessingEvent.COOLDOWN_ENDED):
            self.events.append((event, data))


async def test_injected_sleep_and_clock_drive_the_cooldown() -> None:
    """The cooldown sleeps through the injected sleep (a 30 s cooldown finishes at
    once) and COOLDOWN_ENDED reports the injected clock's duration."""
    now = [100.0]
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)
        now[0] += delay

    observer = _CooldownEvents()
    coord = RateLimitCoordinator(
        FixedDelayStrategy(cooldown=30.0),
        EventDispatcher(observers=[observer], middlewares=[]),
        clock=lambda: now[0],
        sleep=fake_sleep,
    )
    await asyncio.wait_for(coord.handle_rate_limit(worker_id=0, observed_generation=0), 1.0)
    await coord.shutdown()
    assert slept == [30.0]
    ended = [data for event, data in observer.events if event is ProcessingEvent.COOLDOWN_ENDED]
    assert ended == [{"duration": 30.0}]


async def test_default_clock_is_looked_up_at_call_time(monkeypatch) -> None:
    """Without an injected clock, COOLDOWN_ENDED's duration comes from the module's
    ``time.time`` read when the cooldown runs (the default sleep is covered by the
    tests above, which replace ``asyncio.sleep`` after construction)."""
    from types import SimpleNamespace

    from llm_minder._internal import rate_limit_coordinator as rlc_module

    now = [500.0]

    async def fake_sleep(delay: float, *args, **kwargs) -> None:
        now[0] += delay

    observer = _CooldownEvents()
    coord = RateLimitCoordinator(
        FixedDelayStrategy(cooldown=7.0), EventDispatcher(observers=[observer], middlewares=[])
    )
    monkeypatch.setattr(rlc_module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(rlc_module.asyncio, "sleep", fake_sleep)
    await asyncio.wait_for(coord.handle_rate_limit(worker_id=0, observed_generation=0), 1.0)
    await coord.shutdown()
    ended = [data for event, data in observer.events if event is ProcessingEvent.COOLDOWN_ENDED]
    assert ended == [{"duration": 7.0}]
