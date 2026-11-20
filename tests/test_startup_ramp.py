"""Startup concurrency ramp behavior and timeout boundaries."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from llm_minder import (
    LLMCallPool,
    LLMCallStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    StartupRampConfig,
    TokenUsage,
)


class _RampStrategy(LLMCallStrategy[str]):
    def __init__(self, delays: dict[str, float], capacity: int | None = None) -> None:
        self.delays = delays
        self.capacity = capacity
        self.active = 0
        self.peak = 0
        self.started_at: list[float] = []
        self.epoch = time.perf_counter()

    @property
    def max_concurrency(self) -> int | None:
        return self.capacity

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: Any = None
    ) -> tuple[str, TokenUsage, None]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started_at.append(time.perf_counter() - self.epoch)
        try:
            await asyncio.sleep(self.delays[prompt])
            return prompt, {"total_tokens": 0}, None
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_startup_ramp_increases_allowed_concurrency_over_time() -> None:
    strategy = _RampStrategy({str(i): 0.09 for i in range(5)})
    config = ProcessorConfig(
        max_workers=5,
        attempt_timeout=1.0,
        startup_ramp=StartupRampConfig(
            initial_concurrency=1,
            concurrency_step=1,
            ramp_interval_seconds=0.04,
            max_concurrency=3,
        ),
    )
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(5):
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
        result = await processor.process_all()

    assert result.succeeded == 5
    assert strategy.peak == 3
    starts = sorted(strategy.started_at)
    assert starts[1] >= 0.025
    assert starts[2] >= 0.06
    assert any(item.timing.startup_ramp_wait_seconds > 0 for item in result.results)


@pytest.mark.asyncio
async def test_ramp_wait_is_outside_execution_timeout_in_gateway() -> None:
    # "second" waits ~0.6 s at the ramp, then runs 0.3 s. Each part fits the
    # 0.6 s timeout with 0.2 s or more to spare, but together (a timeout that
    # counted the ramp wait) they exceed it by about 0.3 s.
    strategy = _RampStrategy({"first": 1.5, "second": 0.3})
    config = ProcessorConfig(
        max_workers=2,
        attempt_timeout=0.6,
        retry=RetryConfig(max_attempts=1),
        startup_ramp=StartupRampConfig(
            initial_concurrency=1,
            concurrency_step=1,
            ramp_interval_seconds=1.2,
            max_concurrency=2,
        ),
    )
    gateway = LLMCallPool(strategy, config=config)
    first = asyncio.create_task(gateway.submit_result("first"))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(gateway.submit_result("second"))
    first_result, second_result = await asyncio.gather(first, second)
    await gateway.aclose()

    assert not first_result.success  # execution itself exceeds 0.6 s
    assert second_result.success
    # "second" waits at the ramp until "first" times out at about 0.6 s.
    assert second_result.timing.startup_ramp_wait_seconds >= 0.35
    assert second_result.timing.attempts[0].execution_seconds < 0.6


@pytest.mark.asyncio
async def test_ramp_composes_with_lower_advertised_capacity() -> None:
    strategy = _RampStrategy({str(i): 0.05 for i in range(4)}, capacity=2)
    config = ProcessorConfig(
        max_workers=4,
        attempt_timeout=1.0,
        startup_ramp=StartupRampConfig(
            initial_concurrency=1,
            concurrency_step=4,
            ramp_interval_seconds=0.01,
            max_concurrency=4,
        ),
    )
    processor = ParallelBatchProcessor[str, str, None](config=config)
    with pytest.warns(UserWarning):
        for i in range(4):
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
    result = await processor.process_all()
    await processor.cleanup()

    assert result.succeeded == 4
    assert strategy.peak == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_concurrency": 0}, "initial_concurrency"),
        ({"concurrency_step": 0}, "concurrency_step"),
        ({"ramp_interval_seconds": 0}, "ramp_interval_seconds"),
        ({"max_concurrency": 0}, "max_concurrency"),
        (
            {"initial_concurrency": 3, "max_concurrency": 2},
            "initial_concurrency must be <= max_concurrency",
        ),
        ({"jitter_seconds": -1}, "jitter_seconds"),
    ],
)
def test_startup_ramp_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StartupRampConfig(**kwargs)


# ── Injected clock and jitter sleep (#327) ──────────────────────────────


class _FrozenClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_injected_clock_sets_the_ramp_limit() -> None:
    """The ramp limit follows the injected clock, not ``time.perf_counter``: after
    one interval of injected time, a second slot is free at once even though no
    real time has passed (the real clock would keep the limit at 1 for 60 s)."""
    from llm_minder._internal.capacity import _CapacityGate

    clock = _FrozenClock()
    gate = _CapacityGate(
        3, StartupRampConfig(initial_concurrency=1, ramp_interval_seconds=60.0), clock=clock
    )
    await gate.acquire()
    clock.now = 60.0
    wait_seconds, _ = await asyncio.wait_for(gate.acquire(), timeout=1.0)
    assert gate._active == 2
    assert wait_seconds == 0.0


async def test_default_clock_is_looked_up_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an injected clock the gate reads the module's ``time.perf_counter``
    when it runs, so replacing it after construction still takes effect."""
    from types import SimpleNamespace

    from llm_minder._internal import capacity as capacity_module

    gate = capacity_module._CapacityGate(
        3, StartupRampConfig(initial_concurrency=1, ramp_interval_seconds=60.0)
    )
    clock = _FrozenClock(gate._started)
    monkeypatch.setattr(capacity_module, "time", SimpleNamespace(perf_counter=clock))
    await gate.acquire()
    clock.now += 60.0
    await asyncio.wait_for(gate.acquire(), timeout=1.0)
    assert gate._active == 2


async def test_injected_sleep_serves_the_ramp_jitter() -> None:
    from llm_minder._internal.capacity import CapacityLimiter

    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    limiter = CapacityLimiter(
        None,
        max_workers=3,
        startup_ramp=StartupRampConfig(
            initial_concurrency=1, ramp_interval_seconds=60.0, jitter_seconds=30.0
        ),
        clock=_FrozenClock(),
        sleep=fake_sleep,
    )

    class _Strategy:
        max_concurrency = 3

    async with asyncio.timeout(1.0), limiter.admit(_Strategy()) as admission:
        assert len(slept) == 1 and 0.0 <= slept[0] <= 30.0
        assert admission.startup_ramp_wait_seconds == slept[0]


async def test_default_jitter_sleep_is_looked_up_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_minder._internal import capacity as capacity_module

    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    gate = capacity_module._CapacityGate(
        3,
        StartupRampConfig(initial_concurrency=1, ramp_interval_seconds=60.0, jitter_seconds=30.0),
    )
    monkeypatch.setattr(capacity_module.asyncio, "sleep", fake_sleep)
    _, ramp_wait = await asyncio.wait_for(gate.acquire(), timeout=1.0)
    assert len(slept) == 1 and ramp_wait == slept[0]
