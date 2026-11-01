"""Public support for strategies that override execute() (issue #267)."""

from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

import pytest

import async_batch_llm
from async_batch_llm import (
    LLMCallStrategy,
    LLMResponse,
    LLMWorkItem,
    ModelStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    RetryState,
    TokenUsage,
    attach_token_usage,
    provider_call,
)

_USAGE: TokenUsage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


async def _run(strategy: LLMCallStrategy[Any], *, max_attempts: int = 1) -> Any:
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=5.0,
        retry=RetryConfig(max_attempts=max_attempts, initial_wait=0.01, jitter=False),
    )
    async with ParallelBatchProcessor[str, Any, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="x", strategy=strategy, prompt="hi"))
        result = await processor.process_all()
    return result.results[0]


class _TimedStrategy(LLMCallStrategy[str]):
    """Overrides execute() and times its provider call with provider_call."""

    def __init__(self, *, fail: bool = False, use_async: bool = True) -> None:
        self.fail = fail
        self.use_async = use_async

    async def _provider(self) -> None:
        await asyncio.sleep(0.03)
        if self.fail:
            raise ConnectionError("provider down")

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, dict[str, Any] | None]:
        if self.use_async:
            async with provider_call(state):
                await self._provider()
        else:
            with provider_call(state):
                await self._provider()
        await asyncio.sleep(0.03)  # parsing: not provider time
        return "ok", dict(_USAGE), None  # type: ignore[return-value]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False], ids=["async-with", "with"])
async def test_provider_call_records_provider_time(use_async):
    item = await _run(_TimedStrategy(use_async=use_async))
    assert item.success
    attempt = item.timing.attempts[0]
    assert attempt.provider_seconds is not None
    assert 0.025 <= attempt.provider_seconds < attempt.execution_seconds
    assert item.timing.provider_seconds == attempt.provider_seconds


@pytest.mark.asyncio
async def test_provider_call_records_provider_time_when_the_call_raises():
    item = await _run(_TimedStrategy(fail=True))
    assert not item.success
    assert item.timing.attempts[0].provider_seconds is not None
    assert item.timing.attempts[0].provider_seconds >= 0.025


@pytest.mark.asyncio
async def test_provider_call_without_state_records_nothing_and_propagates():
    async with provider_call(None):
        pass
    with pytest.raises(KeyError):
        with provider_call(None):
            raise KeyError("x")


class _ValidationError(ValueError):
    pass


class _ParseFailingStrategy(LLMCallStrategy[str]):
    """A billed response whose parsing fails; tokens attached with the helper."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[BaseException] = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, dict[str, Any] | None]:
        self.calls += 1
        try:
            raise _ValidationError("bad JSON")
        except _ValidationError as e:
            attach_token_usage(e, MappingProxyType(dict(_USAGE)))
            raise

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        self.seen.append(exception)


@pytest.mark.asyncio
async def test_attach_token_usage_counts_tokens_and_keeps_the_exception_type():
    strategy = _ParseFailingStrategy()
    item = await _run(strategy, max_attempts=2)

    assert not item.success
    # Classified from the original ValueError subclass: a non-retried logic error.
    assert item.error_category == "logic_error"
    assert strategy.calls == 1
    # on_error sees the original type, not a wrapper, so escalation logic works.
    assert [type(seen) for seen in strategy.seen] == [_ValidationError]
    assert item.timing.attempts[0].error_type == "_ValidationError"
    assert {key: item.token_usage.get(key) for key in _USAGE} == _USAGE


def test_attach_token_usage_stamps_a_copy_in_place():
    usage = {"total_tokens": 5}
    error = ValueError("x")
    assert attach_token_usage(error, usage) is None
    usage["total_tokens"] = 99
    assert error.__dict__["_failed_token_usage"] == {"total_tokens": 5}


def _model() -> AsyncMock:
    model = AsyncMock()
    model.generate.return_value = LLMResponse(
        text="ok", input_tokens=1, output_tokens=1, total_tokens=2
    )
    return model


class _Growing(ModelStrategy[str]):
    def attempt_generation_config(
        self, attempt: int, state: RetryState | None
    ) -> dict[str, Any] | None:
        return {"max_tokens": 100 * 2 ** (attempt - 1)}


@pytest.mark.asyncio
async def test_attempt_generation_config_merges_over_generation_config_per_attempt():
    model = _model()
    base = {"max_tokens": 1, "response_format": {"type": "json_object"}}
    strategy = _Growing(model, generation_config=base)

    await strategy.execute("p", attempt=1, timeout=1.0)
    await strategy.execute("p", attempt=3, timeout=1.0)

    configs = [call.kwargs["config"] for call in model.generate.await_args_list]
    assert configs == [
        {"max_tokens": 100, "response_format": {"type": "json_object"}},
        {"max_tokens": 400, "response_format": {"type": "json_object"}},
    ]
    assert base == {"max_tokens": 1, "response_format": {"type": "json_object"}}


@pytest.mark.asyncio
async def test_attempt_generation_config_works_without_generation_config():
    model = _model()
    await _Growing(model).execute("p", attempt=2, timeout=1.0)
    assert model.generate.await_args.kwargs["config"] == {"max_tokens": 200}


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_config", [None, {"max_tokens": 5}])
async def test_default_hook_leaves_the_call_unchanged(generation_config):
    model = _model()
    await ModelStrategy(model, generation_config=generation_config).execute(
        "p", attempt=2, timeout=1.0
    )
    kwargs = model.generate.await_args.kwargs
    if generation_config is None:
        assert "config" not in kwargs
    else:
        assert kwargs["config"] is generation_config


@pytest.mark.asyncio
async def test_model_strategy_records_provider_time_with_the_hook():
    model = _model()

    async def slow_generate(*args: Any, **kwargs: Any) -> LLMResponse:
        await asyncio.sleep(0.03)
        return LLMResponse(text="ok", input_tokens=1, output_tokens=1, total_tokens=2)

    model.generate.side_effect = slow_generate
    item = await _run(_Growing(model))
    assert item.success
    assert item.timing.attempts[0].provider_seconds >= 0.025


def test_helpers_are_exported():
    assert {"provider_call", "attach_token_usage"} <= set(async_batch_llm.__all__)
