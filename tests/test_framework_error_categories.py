"""Framework-owned failures keep framework categories under a custom classifier.

A classifier that subclasses ``ErrorClassifier`` directly (the shape used by
``examples/example_batch_benchmark.py``) never reaches the built-in framework
prelude. Guardrail failures must still get the framework's ``ErrorInfo``, since
replay eligibility, deadline events, abort metrics and fail-fast key on it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_minder import (
    ArtifactIdentity,
    CallableStrategy,
    CallOutcome,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallPool,
    LLMWorkItem,
    MetricsObserver,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    call_result,
    process_prompts,
)
from llm_minder.observers import BaseObserver, ProcessingEvent
from llm_minder.strategies import (
    BatchAbortedError,
    BatchBudgetExceeded,
    BatchDeadlineExceeded,
    ErrorClassifier,
    ErrorInfo,
    FrameworkTimeoutError,
    ItemDeadlineExceeded,
    RateLimitRetriesExceeded,
)


class PlainClassifier(ErrorClassifier):
    """Labels everything ``unknown``; never calls the framework prelude."""

    def classify(self, exception: Exception) -> ErrorInfo:
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
            hint="custom hint",
        )


class AuthError(Exception):
    pass


class AuthClassifier(ErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, AuthError):
            return ErrorInfo(False, False, False, "authentication")
        return ErrorInfo(True, False, False, "unknown")


async def _slow(prompt: str, *, attempt: int, timeout: float, state: Any) -> CallOutcome[str]:
    await asyncio.sleep(5)
    return CallOutcome(prompt, {})


def _deadline_config() -> ProcessorConfig:
    return ProcessorConfig(
        guardrails=GuardrailConfig(total_timeout_per_item=0.05),
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.01, jitter=False),
    )


async def test_call_result_item_deadline_ignores_custom_classifier() -> None:
    result = await call_result(
        CallableStrategy(_slow), "x", config=_deadline_config(), error_classifier=PlainClassifier()
    )
    assert isinstance(result.exception, ItemDeadlineExceeded)
    assert result.error_category == "framework_total_item_timeout"


async def test_pool_item_deadline_ignores_custom_classifier() -> None:
    async with LLMCallPool(
        CallableStrategy(_slow), config=_deadline_config(), error_classifier=PlainClassifier()
    ) as pool:
        result = await pool.submit_result("x")
    assert result.error_category == "framework_total_item_timeout"


class _Events(BaseObserver):
    def __init__(self) -> None:
        self.events: list[ProcessingEvent] = []

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.events.append(event)


async def test_processor_item_deadline_event_with_custom_classifier() -> None:
    events = _Events()
    strategy = CallableStrategy(_slow)
    async with ParallelBatchProcessor[str, str, None](
        config=_deadline_config(), error_classifier=PlainClassifier(), observers=[events]
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="a"))
        batch = await processor.process_all()
    assert batch.results[0].error_category == "framework_total_item_timeout"
    assert ProcessingEvent.ITEM_DEADLINE_EXCEEDED in events.events


def _abort_config() -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=2,
        guardrails=GuardrailConfig(
            abort_on_error_categories=frozenset({"authentication"}),
            abort_mode="cancel_active",
        ),
        retry=RetryConfig(max_attempts=1),
    )


async def _slow_or_bad(
    prompt: str, *, attempt: int, timeout: float, state: Any
) -> CallOutcome[str]:
    if prompt == "bad":
        await asyncio.sleep(0.02)
        raise AuthError("401")
    await asyncio.sleep(5)
    return CallOutcome(prompt, {})


async def test_pool_cancel_active_victim_is_batch_aborted() -> None:
    async with LLMCallPool(
        CallableStrategy(_slow_or_bad), config=_abort_config(), error_classifier=AuthClassifier()
    ) as pool:
        slow = asyncio.create_task(pool.submit_result("slow"))
        await asyncio.sleep(0.01)
        bad = await pool.submit_result("bad")
        victim = await slow
    assert bad.error_category == "authentication"
    assert isinstance(victim.exception, BatchAbortedError)
    assert victim.error_category == "batch_aborted"


async def test_processor_cancel_active_victim_counts_as_aborted() -> None:
    metrics = MetricsObserver()
    strategy = CallableStrategy(_slow_or_bad)
    async with ParallelBatchProcessor[str, str, None](
        config=_abort_config(), error_classifier=AuthClassifier(), observers=[metrics]
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="slow", strategy=strategy, prompt="slow"))
        await processor.add_work(LLMWorkItem(item_id="bad", strategy=strategy, prompt="bad"))
        batch = await processor.process_all()
    by_id = {r.item_id: r for r in batch.results}
    assert by_id["slow"].error_category == "batch_aborted"
    assert (await metrics.get_metrics())["items_aborted"] == 1


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
async def test_item_deadline_does_not_replay_with_custom_classifier(
    tmp_path: Path, backend: str
) -> None:
    slow = {"on": True}
    calls = {"n": 0}

    async def fn(prompt: str, *, attempt: int, timeout: float, state: Any) -> CallOutcome[str]:
        calls["n"] += 1
        if slow["on"]:
            await asyncio.sleep(5)
        return CallOutcome(prompt, {})

    def store() -> Any:
        if backend == "jsonl":
            return JsonlArtifactStore(tmp_path / "a.jsonl")
        return SqliteArtifactStore(tmp_path / "a.sqlite")

    strategy = CallableStrategy(fn, identity=ArtifactIdentity(provider="fake", model="m"))
    config = ProcessorConfig(
        guardrails=GuardrailConfig(total_timeout_per_item=0.05),
        retry=RetryConfig(max_attempts=1),
    )
    first = await process_prompts(
        strategy, ["a"], config=config, artifact_store=store(), error_classifier=PlainClassifier()
    )
    assert first.results[0].error_category == "framework_total_item_timeout"

    slow["on"] = False
    calls["n"] = 0
    second = await process_prompts(
        strategy,
        ["a"],
        config=config,
        artifact_store=store(),
        resume=ResumePolicy.REUSE_ALL,
        error_classifier=PlainClassifier(),
    )
    result = second.results[0]
    assert result.success
    assert not result.replayed_from_artifact
    assert calls["n"] == 1


@pytest.mark.parametrize(
    ("exception", "category"),
    [
        (ItemDeadlineExceeded("x"), "framework_total_item_timeout"),
        (BatchDeadlineExceeded("x"), "batch_deadline_exceeded"),
        (BatchBudgetExceeded("x"), "batch_budget_exceeded"),
        (BatchAbortedError("x"), "batch_aborted"),
        (RateLimitRetriesExceeded("x"), "rate_limit_retries_exceeded"),
    ],
)
def test_executor_classification_bypasses_custom_classifier(
    exception: Exception, category: str
) -> None:
    from llm_minder._internal.item_executor import _classify_error

    info = _classify_error(exception, PlainClassifier())
    assert info.error_category == category
    assert not info.is_retryable
    assert info.hint is None


def test_attempt_timeout_still_reaches_custom_classifier() -> None:
    """Attempt timeouts are a provider condition: users may set their policy."""
    from llm_minder._internal.item_executor import _classify_error

    info = _classify_error(FrameworkTimeoutError("t"), PlainClassifier())
    assert info.error_category == "unknown"
