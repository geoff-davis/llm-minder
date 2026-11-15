"""describe_limits() and gate-wait reports (#319)."""

from __future__ import annotations

import asyncio
import logging
import warnings

import pytest

from llm_minder import (
    AttemptTiming,
    BatchResult,
    EffectiveLimits,
    LLMCallPool,
    LLMCallStrategy,
    LLMWorkItem,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    StartupRampConfig,
    WorkItemResult,
    WorkItemTiming,
    describe_limits,
    process_prompts,
    summarize_gate_waits,
)


class EchoStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


class CappedStrategy(EchoStrategy):
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity

    @property
    def max_concurrency(self) -> int:
        return self._capacity


def _row(limits: EffectiveLimits, name: str, scope: str = "all strategies"):
    matches = [row for row in limits.rows if row.name == name and row.scope == scope]
    assert len(matches) == 1, (name, scope, limits.rows)
    return matches[0]


class TestConfigurationRows:
    def test_defaults(self):
        limits = describe_limits(ProcessorConfig())
        assert _row(limits, "max_workers").value == 5
        assert _row(limits, "max_workers").source == "default"
        assert _row(limits, "concurrency").source == "not set"
        assert _row(limits, "max_provider_concurrency").value is None
        assert limits.in_effect

    def test_derived_from_concurrency(self):
        limits = describe_limits(ProcessorConfig(concurrency=12))
        assert _row(limits, "concurrency").source == "explicit"
        assert _row(limits, "max_workers").source == "derived from concurrency"
        assert _row(limits, "max_provider_concurrency").source == "derived from concurrency"

    def test_explicit_values_win(self):
        limits = describe_limits(ProcessorConfig(concurrency=12, max_workers=3))
        assert (_row(limits, "max_workers").value, _row(limits, "max_workers").source) == (
            3,
            "explicit",
        )

    def test_assignment_after_construction_reads_as_explicit(self):
        config = ProcessorConfig(concurrency=12)
        config.max_workers = 4
        assert _row(describe_limits(config), "max_workers").source == "explicit"

    def test_quota_rows_are_per_quota_scope(self):
        limits = describe_limits(
            ProcessorConfig(max_requests_per_minute=600, max_tokens_per_minute=90_000)
        )
        assert _row(limits, "max_requests_per_minute").source == "explicit, per quota scope"
        assert _row(limits, "max_tokens_per_minute").value == 90_000
        assert _row(limits, "quota_burst_seconds").source == "not set"

    async def test_concurrency_shorthand_keeps_defaults_as_defaults(self):
        # process_prompts(config=..., concurrency=N) copies the config with
        # dataclasses.replace(), which passes resolved values back in.
        from llm_minder.streaming import _apply_concurrency_shorthand

        derived = _apply_concurrency_shorthand(ProcessorConfig(), 9)
        assert derived is not None
        limits = describe_limits(derived)
        assert _row(limits, "max_workers").source == "derived from concurrency"
        assert _row(limits, "max_provider_concurrency").source == "derived from concurrency"


class TestStrategyRows:
    def test_capacity_is_the_lowest_bound(self):
        config = ProcessorConfig(max_workers=10, max_provider_concurrency=6)
        limits = describe_limits(config, CappedStrategy(4))
        capacity = _row(limits, "provider capacity", "CappedStrategy")
        assert capacity.value == 4
        assert "strategy max_concurrency=4" in capacity.source
        assert "max_provider_concurrency=6" in capacity.source
        attempts = _row(limits, "concurrent attempts", "CappedStrategy")
        assert (attempts.value, attempts.source) == (4, "bound by provider capacity")

    def test_workers_bind_without_a_capacity_gate(self):
        limits = describe_limits(ProcessorConfig(max_workers=7), EchoStrategy())
        assert _row(limits, "provider capacity", "EchoStrategy").value is None
        attempts = _row(limits, "concurrent attempts", "EchoStrategy")
        assert (attempts.value, attempts.source) == (7, "bound by max_workers")

    def test_startup_ramp_caps_capacity(self):
        config = ProcessorConfig(max_workers=10, startup_ramp=StartupRampConfig(max_concurrency=3))
        limits = describe_limits(config, EchoStrategy())
        assert _row(limits, "startup_ramp.max_concurrency").value == 3
        assert _row(limits, "provider capacity", "EchoStrategy").value == 3

    def test_matches_the_capacity_limiter(self):
        # The rows must agree with what admission actually enforces.
        from llm_minder._internal.capacity import CapacityLimiter

        cases = [
            (ProcessorConfig(max_workers=10, max_provider_concurrency=6), CappedStrategy(4)),
            (ProcessorConfig(concurrency=5), CappedStrategy(9)),
            (ProcessorConfig(max_workers=8, startup_ramp=StartupRampConfig()), EchoStrategy()),
            (ProcessorConfig(), EchoStrategy()),
        ]
        for config, strategy in cases:
            limiter = CapacityLimiter(
                config.max_provider_concurrency,
                max_workers=config.max_workers,
                startup_ramp=config.startup_ramp,
            )
            label = type(strategy).__name__
            row = _row(describe_limits(config, strategy), "provider capacity", label)
            assert row.value == limiter.effective_capacity(strategy), config

    def test_duplicate_strategy_classes_get_distinct_labels(self):
        limits = describe_limits(ProcessorConfig(), CappedStrategy(2), CappedStrategy(3))
        assert _row(limits, "provider capacity", "CappedStrategy").value == 2
        assert _row(limits, "provider capacity", "CappedStrategy #2").value == 3

    async def test_pool_resize_is_flagged_before_start(self):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        try:
            limits = describe_limits(ProcessorConfig(concurrency=8), OpenAIStrategy(model))
            assert not limits.in_effect
            assert (
                "may resize it to concurrency=8"
                in _row(limits, "strategy max_concurrency", "OpenAIStrategy").source
            )
            assert "before batch start" in str(limits)
        finally:
            await model.cleanup()

    def test_str_lists_every_row(self):
        text = str(describe_limits(ProcessorConfig(concurrency=4), CappedStrategy(2)))
        assert text.startswith("Effective limits")
        assert "[CappedStrategy]" in text
        assert "derived from concurrency" in text


class TestSurfaces:
    async def test_processor_reports_in_effect_after_resize(self, caplog):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        strategy = OpenAIStrategy(model)
        config = ProcessorConfig(concurrency=8)
        processor = ParallelBatchProcessor[None, str, None](config=config)
        try:
            before = processor.describe_limits(strategy)
            assert not before.in_effect
            assert _row(before, "strategy max_concurrency", "OpenAIStrategy").value is None
            with caplog.at_level(logging.INFO, logger="llm_minder.parallel"):
                await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="p"))
                await processor.add_work(LLMWorkItem(item_id="b", strategy=strategy, prompt="q"))
            after = processor.describe_limits()
            assert after.in_effect
            assert _row(after, "strategy max_concurrency", "OpenAIStrategy").value == 8
            logged = [r for r in caplog.records if "Effective limits" in r.getMessage()]
            # Once per strategy, with the post-resize capacity.
            assert len(logged) == 1
            assert logged[0].levelno == logging.INFO
            assert str(after) in logged[0].getMessage()
        finally:
            await model.cleanup()

    async def test_per_item_strategies_with_the_same_limits_log_once(self, caplog):
        processor = ParallelBatchProcessor[None, str, None](config=ProcessorConfig(max_workers=3))
        with caplog.at_level(logging.INFO, logger="llm_minder.parallel"):
            for i in range(5):
                await processor.add_work(
                    LLMWorkItem(item_id=str(i), strategy=CappedStrategy(2), prompt="p")
                )
            await processor.add_work(
                LLMWorkItem(item_id="other", strategy=CappedStrategy(1), prompt="p")
            )
        logged = [r for r in caplog.records if "Effective limits" in r.getMessage()]
        assert len(logged) == 2

    async def test_processor_without_work_reports_configuration_only(self):
        processor = ParallelBatchProcessor[None, str, None](config=ProcessorConfig(max_workers=3))
        limits = processor.describe_limits()
        assert {row.scope for row in limits.rows} == {"all strategies"}

    async def test_pool_reports_its_strategy_and_logs_once(self, caplog):
        with caplog.at_level(logging.INFO, logger="llm_minder.call_pool"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pool = LLMCallPool(CappedStrategy(2), config=ProcessorConfig(max_workers=6))
        try:
            limits = pool.describe_limits()
            assert limits.in_effect
            assert _row(limits, "concurrent attempts", "CappedStrategy").value == 2
            logged = [r for r in caplog.records if "Effective limits" in r.getMessage()]
            assert len(logged) == 1
            assert str(limits) in logged[0].getMessage()
        finally:
            await pool.aclose()

    async def test_pool_never_flags_a_resize(self):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        pool = LLMCallPool(OpenAIStrategy(model), config=ProcessorConfig(concurrency=8))
        try:
            assert pool.describe_limits().in_effect
        finally:
            await pool.aclose()
            await model.cleanup()

    def test_function_covers_call(self):
        # call() builds its host from (config, strategy); the function gives the
        # same rows for that pair.
        config = ProcessorConfig(max_provider_concurrency=2)
        limits = describe_limits(config, EchoStrategy())
        assert _row(limits, "concurrent attempts", "EchoStrategy").value == 2


def _result(item_id: str, total: float, *attempts: AttemptTiming, replayed: bool = False):
    result = WorkItemResult[str, None](
        item_id=item_id,
        success=True,
        output="x",
        timing=WorkItemTiming(total_seconds=total, attempts=list(attempts)),
    )
    result.replayed_from_artifact = replayed
    return result


class TestGateWaits:
    def test_totals_and_order(self):
        results = [
            _result("a", 4.0, AttemptTiming(attempt=1, try_number=1, quota_wait_seconds=2.0)),
            _result(
                "b",
                6.0,
                AttemptTiming(
                    attempt=1, try_number=1, admission_wait_seconds=1.0, quota_wait_seconds=1.0
                ),
                AttemptTiming(attempt=2, try_number=2, retry_backoff_seconds=0.5),
            ),
        ]
        report = summarize_gate_waits(results)
        assert report.items == 2
        assert report.item_seconds == 10.0
        first = report.gates[0]
        assert (first.gate, first.total_seconds, first.attempts_waited) == (
            "quota (RPM/TPM)",
            3.0,
            2,
        )
        assert first.share == pytest.approx(0.3)
        by_gate = {gate.gate: gate for gate in report.gates}
        assert by_gate["provider capacity"].total_seconds == 1.0
        assert by_gate["retry backoff"].attempts_waited == 1
        assert by_gate["rate-limit cooldown"].total_seconds == 0.0
        assert "quota (RPM/TPM)" in str(report)

    def test_replayed_results_are_skipped(self):
        replayed = _result(
            "old",
            9.0,
            AttemptTiming(attempt=1, try_number=1, quota_wait_seconds=9.0),
            replayed=True,
        )
        report = summarize_gate_waits([replayed])
        assert report.items == 0
        assert all(gate.total_seconds == 0 for gate in report.gates)
        assert "no attempt waited" in str(report)

    def test_sub_millisecond_waits_do_not_count_as_waiting(self):
        report = summarize_gate_waits(
            [_result("a", 1.0, AttemptTiming(attempt=1, try_number=1, cooldown_wait_seconds=1e-5))]
        )
        by_gate = {gate.gate: gate for gate in report.gates}
        assert by_gate["rate-limit cooldown"].attempts_waited == 0

    async def test_batch_result_method_on_a_real_run(self):
        class SlowCapped(CappedStrategy):
            async def execute(self, prompt, attempt, timeout, state=None):
                await asyncio.sleep(0.02)
                return await super().execute(prompt, attempt, timeout, state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = await process_prompts(
                SlowCapped(1), ["a", "b", "c", "d"], config=ProcessorConfig(max_workers=4)
            )
        report = result.gate_waits()
        assert report == summarize_gate_waits(result.results)
        # Four workers share one capacity slot, so three attempts queue there.
        assert report.gates[0].gate == "provider capacity"
        assert report.gates[0].attempts_waited == 3
        assert report.items == 4

    def test_restored_batch(self):
        batch = BatchResult[str, None](
            results=[
                _result("a", 2.0, AttemptTiming(attempt=1, try_number=1, quota_wait_seconds=1.0))
            ]
        )
        restored = BatchResult.from_json(batch.to_json())
        assert restored.gate_waits() == batch.gate_waits()
