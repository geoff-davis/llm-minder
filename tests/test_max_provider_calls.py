"""GuardrailConfig.max_provider_calls: a hard cap on physical provider attempts (#272)."""

from __future__ import annotations

from typing import Any

import pytest

from llm_minder import (
    AbortMode,
    ArtifactIdentity,
    BatchBudgetExceeded,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallPool,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    call_result,
    process_prompts,
)
from llm_minder._internal.guardrails import AbortController
from llm_minder.observers import BaseObserver, ProcessingEvent
from llm_minder.testing import FakeRateLimitError, FakeStrategy

_FAST_RETRY = RetryConfig(max_attempts=5, initial_wait=0.001, max_wait=0.001, jitter=False)


def _config(cap: int | None, **kwargs: Any) -> ProcessorConfig:
    kwargs.setdefault("retry", _FAST_RETRY)
    return ProcessorConfig(guardrails=GuardrailConfig(max_provider_calls=cap), **kwargs)


class _Recorder(BaseObserver):
    def __init__(self) -> None:
        self.events: list[tuple[ProcessingEvent, dict[str, Any]]] = []

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.events.append((event, dict(data)))


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_rejects_non_positive_int(value):
    with pytest.raises(ValueError, match="max_provider_calls must be a positive int or None"):
        GuardrailConfig(max_provider_calls=value)


def test_accepts_positive_int_or_none():
    assert GuardrailConfig(max_provider_calls=3).max_provider_calls == 3
    assert GuardrailConfig().max_provider_calls is None


@pytest.mark.asyncio
async def test_run_within_the_cap_completes_normally():
    strategy = FakeStrategy("ok")
    batch = await process_prompts(strategy, ["a", "b", "c"], config=_config(3))
    assert batch.succeeded == 3
    assert strategy.calls == 3
    assert batch.termination.kind == "completed"


@pytest.mark.asyncio
async def test_retries_count_and_the_next_call_is_refused():
    strategy = FakeStrategy("ok", failure_schedule=[ConnectionError("x"), ConnectionError("y")])
    batch = await process_prompts(strategy, ["a"], config=_config(2))
    assert strategy.calls == 2
    result = batch.results[0]
    assert not result.success
    assert result.error_category == "batch_budget_exceeded"
    assert isinstance(result.exception, BatchBudgetExceeded)
    assert batch.termination.kind == "budget_exceeded"
    assert batch.termination.error_category == "batch_budget_exceeded"
    assert batch.termination.reason == "Provider call cap reached: 2 of 2 calls started"
    assert batch.termination.triggering_item_id == "item_0"


@pytest.mark.asyncio
async def test_rate_limit_retries_count(fast_rate_limit):
    strategy = FakeStrategy("ok", failure_schedule=[FakeRateLimitError()])
    batch = await process_prompts(strategy, ["a"], config=_config(1, rate_limit=fast_rate_limit))
    assert strategy.calls == 1
    assert batch.results[0].error_category == "batch_budget_exceeded"
    assert batch.termination.kind == "budget_exceeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_mode", [AbortMode.DRAIN_ACTIVE, AbortMode.CANCEL_ACTIVE])
async def test_concurrent_workers_never_overshoot(abort_mode):
    strategy = FakeStrategy("ok", latency=0.02)
    config = ProcessorConfig(
        max_workers=10,
        retry=_FAST_RETRY,
        guardrails=GuardrailConfig(max_provider_calls=5, abort_mode=abort_mode),
    )
    batch = await process_prompts(strategy, [f"p{i}" for i in range(30)], config=config)
    assert strategy.calls == 5
    # Prompts not yet accepted when the run stopped are never admitted.
    assert batch.total_items < 30
    assert batch.termination.kind == "budget_exceeded"
    categories = {r.error_category for r in batch.results if not r.success}
    assert categories <= {"batch_budget_exceeded"}
    if abort_mode is AbortMode.DRAIN_ACTIVE:
        # The five admitted calls were already running and are allowed to finish.
        assert batch.succeeded == 5


@pytest.mark.asyncio
async def test_stop_is_announced_once():
    recorder = _Recorder()
    strategy = FakeStrategy("ok", latency=0.01)
    config = ProcessorConfig(max_workers=4, retry=_FAST_RETRY)
    config.guardrails = GuardrailConfig(max_provider_calls=2)
    from llm_minder import ParallelBatchProcessor
    from llm_minder.base import LLMWorkItem

    async with ParallelBatchProcessor[None, str, None](
        config=config, observers=[recorder]
    ) as processor:
        for i in range(8):
            await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="p"))
        await processor.process_all()
        stats = await processor.get_stats()

    aborted = [data for event, data in recorder.events if event is ProcessingEvent.BATCH_ABORTED]
    assert len(aborted) == 1
    assert aborted[0]["kind"] == "budget_exceeded"
    assert aborted[0]["reason"] == "Provider call cap reached: 2 of 2 calls started"
    assert stats["provider_calls_started"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
async def test_replayed_results_do_not_count(tmp_path, store_type):
    path = tmp_path / "run.store"
    identity = ArtifactIdentity(provider="fake", model="fake")
    await process_prompts(
        FakeStrategy("ok"),
        ["a", "b"],
        config=_config(None),
        artifact_store=store_type(path, identity=identity),
    )
    strategy = FakeStrategy("ok")
    batch = await process_prompts(
        strategy,
        ["a", "b", "c"],
        config=_config(1),
        artifact_store=store_type(path, identity=identity),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert strategy.calls == 1
    assert batch.succeeded == 3
    assert batch.termination.kind == "completed"


@pytest.mark.asyncio
async def test_single_call_and_pool_reject_the_cap():
    config = _config(1)
    with pytest.raises(ValueError, match="max_provider_calls is supported only"):
        await call_result(FakeStrategy("ok"), "p", config=config)
    with pytest.raises(ValueError, match="max_provider_calls is supported only"):
        LLMCallPool(FakeStrategy("ok"), config=config)


@pytest.mark.asyncio
async def test_unstarted_call_is_refunded():
    controller = AbortController(AbortMode.DRAIN_ACTIVE, max_provider_calls=1)
    await controller.begin_provider_call("a")
    controller.refund_unstarted_provider_call()
    await controller.end_provider_call()
    assert controller.provider_calls_started == 0
    await controller.begin_provider_call("a")
    assert controller.provider_calls_started == 1
    with pytest.raises(BatchBudgetExceeded):
        await controller.begin_provider_call("b")
    assert controller.cause is not None and controller.cause.triggering_item_id == "b"
