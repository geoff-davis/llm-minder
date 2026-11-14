"""RetryConfig.jitter modes: False, True/"equal" (50-100%) and "full" (0-100%)."""

import asyncio
import random

import pytest

from llm_minder import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from llm_minder._internal.backoff import apply_jitter
from llm_minder.core import RetryConfig
from llm_minder.testing import FakeStrategy


@pytest.mark.parametrize("mode", [True, False, "equal", "full"])
def test_accepted_jitter_values(mode):
    assert RetryConfig(jitter=mode).jitter == mode


@pytest.mark.parametrize("mode", ["Full", "none", "", "true"])
def test_unknown_jitter_string_is_rejected(mode):
    with pytest.raises(ValueError, match="jitter must be True, False, 'equal' or 'full'"):
        RetryConfig(jitter=mode)


@pytest.mark.parametrize(
    ("mode", "low_roll", "high_roll"),
    [
        (False, 10.0, 10.0),
        (True, 5.0, 10.0),
        ("equal", 5.0, 10.0),
        ("full", 0.0, 10.0),
    ],
)
def test_apply_jitter_range(monkeypatch, mode, low_roll, high_roll):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    assert apply_jitter(10.0, mode) == pytest.approx(low_roll)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    assert apply_jitter(10.0, mode) == pytest.approx(high_roll)


async def _backoff_sleeps(monkeypatch, mode, roll: float) -> list[float]:
    """Run one item that fails once, and return the backoff delays it slept for."""
    monkeypatch.setattr(random, "random", lambda: roll)
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def record_sleep(delay, *args, **kwargs):
        slept.append(delay)
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    strategy = FakeStrategy("ok", failure_schedule=[ConnectionError("transient")])
    config = ProcessorConfig(
        max_workers=1,
        retry=RetryConfig(max_attempts=2, initial_wait=5.0, max_wait=5.0, jitter=mode),
    )
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="p"))
        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.calls == 2
    # Other framework timers sleep for longer than max_wait (5s); backoff never does.
    return [delay for delay in slept if 0 < delay <= 5.0]


@pytest.mark.asyncio
async def test_full_jitter_can_retry_at_once(monkeypatch):
    # A roll of 0 gives no wait under full jitter; equal jitter would still wait
    # half of the 5s backoff.
    assert await _backoff_sleeps(monkeypatch, "full", 0.0) == []


@pytest.mark.asyncio
async def test_full_jitter_scales_the_whole_backoff(monkeypatch):
    assert await _backoff_sleeps(monkeypatch, "full", 0.25) == [pytest.approx(1.25)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [True, "equal"])
async def test_equal_jitter_keeps_half_the_backoff(monkeypatch, mode):
    assert await _backoff_sleeps(monkeypatch, mode, 0.0) == [pytest.approx(2.5)]


@pytest.mark.asyncio
async def test_no_jitter_waits_the_full_backoff(monkeypatch):
    assert await _backoff_sleeps(monkeypatch, False, 0.0) == [pytest.approx(5.0)]
