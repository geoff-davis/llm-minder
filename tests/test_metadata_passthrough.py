"""Tests for metadata pass-through from strategy.execute() to WorkItemResult.

Added in v0.10.0 alongside the 3-tuple ``LLMCallStrategy.execute()``
contract. Verifies:

- Built-in strategies surface ``LLMResponse.metadata`` into
  ``WorkItemResult.metadata``.
- Custom strategies returning the pre-1.0 2-tuple fail with a pointer to the
  fix, without a retry, and their token usage is still recorded.
- The ``_unpack_strategy_result`` helper rejects malformed shapes.
- ``gemini_safety_ratings`` is gone (removed in 1.0); ratings live in
  ``metadata['safety_ratings']``.
"""

from __future__ import annotations

import dataclasses
from collections import UserDict
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_minder import (
    ArtifactIdentity,
    JsonlArtifactStore,
    LLMWorkItem,
    OpenAIStrategy,
    OpenRouterStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    process_prompts,
)
from llm_minder.base import (
    LLMResponse,
    RetryState,
    TokenUsage,
    WorkItemResult,
    _unpack_strategy_result,
)
from llm_minder.llm_strategies import LLMCallStrategy


def _mock_model_with_metadata(metadata: dict | None) -> AsyncMock:
    m = AsyncMock()
    m.generate = AsyncMock(
        return_value=LLMResponse(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            metadata=metadata,
        )
    )
    return m


# ── _unpack_strategy_result ───────────────────────────────────────────


class TestUnpackStrategyResult:
    def test_three_tuple_passes_through(self):
        out, tokens, meta = _unpack_strategy_result(("hi", {"total_tokens": 5}, {"x": 1}))
        assert out == "hi"
        assert tokens == {"total_tokens": 5}
        assert meta == {"x": 1}

    def test_three_tuple_with_none_metadata(self):
        out, tokens, meta = _unpack_strategy_result(("hi", {"total_tokens": 5}, None))
        assert meta is None

    def test_two_tuple_rejected_with_its_usage_attached(self):
        with pytest.raises(ValueError, match=r"returned a 2-tuple.*Since 1\.0") as caught:
            _unpack_strategy_result(("hi", {"total_tokens": 5}))
        assert caught.value.__dict__["_failed_token_usage"] == {"total_tokens": 5}

    def test_non_tuple_rejected(self):
        with pytest.raises(ValueError, match="must return a tuple"):
            _unpack_strategy_result("not a tuple")
        with pytest.raises(ValueError, match="must return a tuple"):
            _unpack_strategy_result({"a": 1})

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError, match="must return a 3-tuple"):
            _unpack_strategy_result(("a",))
        with pytest.raises(ValueError, match="must return a 3-tuple"):
            _unpack_strategy_result(("a", "b", "c", "d"))


# ── Built-in strategies forward LLMResponse.metadata ─────────────────


class TestBuiltinStrategiesForwardMetadata:
    @pytest.mark.asyncio
    async def test_openai_strategy_forwards_metadata(self):
        meta = {"finish_reason": "stop", "model": "gpt-4o-mini"}
        model = _mock_model_with_metadata(meta)
        strategy = OpenAIStrategy(model)

        out, tokens, returned_meta = await strategy.execute("p", 1, 10.0)

        assert out == "hello"
        assert returned_meta == meta

    @pytest.mark.asyncio
    async def test_openrouter_strategy_forwards_provider_metadata(self):
        meta = {
            "finish_reason": "stop",
            "model": "anthropic/claude-haiku-4-5",
            "provider": "Anthropic",
        }
        model = _mock_model_with_metadata(meta)
        strategy = OpenRouterStrategy(model)

        out, tokens, returned_meta = await strategy.execute("p", 1, 10.0)

        assert returned_meta == meta
        assert returned_meta["provider"] == "Anthropic"

    @pytest.mark.asyncio
    async def test_metadata_is_none_when_model_has_none(self):
        model = _mock_model_with_metadata(None)
        strategy = OpenAIStrategy(model)

        _, _, meta = await strategy.execute("p", 1, 10.0)

        assert meta is None


# ── End-to-end: WorkItemResult.metadata is populated ─────────────────


class _ThreeTupleStrategy(LLMCallStrategy[str]):
    """Test helper: returns the configured metadata in a 3-tuple."""

    def __init__(self, metadata: dict | None) -> None:
        self.metadata = metadata

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage, dict | None]:
        return (
            f"out:{prompt}",
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            self.metadata,
        )


class _TwoTupleLegacyStrategy(LLMCallStrategy[str]):
    """Test helper: returns the pre-1.0 2-tuple shape (no metadata)."""

    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage]:
        self.calls += 1
        return (
            f"legacy:{prompt}",
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


_BILLED_USAGE = {
    "input_tokens": 7,
    "output_tokens": 3,
    "total_tokens": 10,
    "cached_input_tokens": 2,
    "cache_write_input_tokens": 1,
}


class _TwoTupleMappingStrategy(LLMCallStrategy[str]):
    """Returns the pre-1.0 2-tuple with token usage in any Mapping type."""

    def __init__(self, usage_type: Callable[[dict[str, int]], Mapping[str, int]]) -> None:
        self.usage_type = usage_type
        self.calls = 0

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> Any:
        self.calls += 1
        return f"legacy:{prompt}", self.usage_type(dict(_BILLED_USAGE))


class TestWorkItemResultMetadata:
    @pytest.mark.asyncio
    async def test_three_tuple_strategy_populates_workitem_metadata(self):
        strategy = _ThreeTupleStrategy(metadata={"provider": "test", "model": "x"})
        config = ProcessorConfig(max_workers=1, attempt_timeout=2.0)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="i1", strategy=strategy, prompt="hi"))
            result = await processor.process_all()

        assert result.results[0].success
        assert result.results[0].metadata == {"provider": "test", "model": "x"}

    @pytest.mark.asyncio
    async def test_two_tuple_legacy_strategy_fails_without_retry(self):
        """A 2-tuple fails as a logic error naming the strategy, isn't retried
        (the provider already billed the call), and keeps its token usage."""
        strategy = _TwoTupleLegacyStrategy()
        config = ProcessorConfig(max_workers=1, attempt_timeout=2.0)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="i1", strategy=strategy, prompt="hi"))
            result = await processor.process_all()

        item = result.results[0]
        assert not item.success
        assert item.error_category == "logic_error"
        assert "_TwoTupleLegacyStrategy.execute() returned a 2-tuple" in (item.error or "")
        assert strategy.calls == 1
        assert item.token_usage["total_tokens"] == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "usage_type",
        [dict, MappingProxyType, UserDict],
        ids=["dict", "mappingproxy", "userdict"],
    )
    @pytest.mark.parametrize(
        ("store_type", "suffix"),
        [(JsonlArtifactStore, ".jsonl"), (SqliteArtifactStore, ".sqlite")],
        ids=["jsonl", "sqlite"],
    )
    async def test_two_tuple_usage_in_any_mapping_is_counted_and_replayed(
        self, tmp_path: Path, usage_type: Any, store_type: Any, suffix: str
    ):
        """Every Mapping keeps the billed usage, and both stores replay it as stored."""
        path = tmp_path / f"two-tuple{suffix}"
        identity = ArtifactIdentity(provider="test", model="m", prompt_version="v1")
        strategy = _TwoTupleMappingStrategy(usage_type)
        first = await process_prompts(
            strategy, [("x", "hi")], artifact_store=store_type(path, identity=identity)
        )
        item = first.results[0]
        assert item.error_category == "logic_error"
        assert strategy.calls == 1
        assert {k: item.token_usage.get(k) for k in _BILLED_USAGE} == _BILLED_USAGE

        rerun = _TwoTupleMappingStrategy(usage_type)
        replayed = await process_prompts(
            rerun,
            [("x", "hi")],
            artifact_store=store_type(path, identity=identity),
            resume=ResumePolicy.REUSE_ALL,
        )
        again = replayed.results[0]
        assert rerun.calls == 0
        assert again.replayed_from_artifact
        assert again.error_category == "logic_error"
        assert {k: again.token_usage.get(k) for k in _BILLED_USAGE} == _BILLED_USAGE

    @pytest.mark.asyncio
    async def test_three_tuple_with_none_metadata(self):
        strategy = _ThreeTupleStrategy(metadata=None)
        config = ProcessorConfig(max_workers=1, attempt_timeout=2.0)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="i1", strategy=strategy, prompt="hi"))
            result = await processor.process_all()

        assert result.results[0].metadata is None

    @pytest.mark.asyncio
    async def test_reserved_keys_reach_typed_views_end_to_end(self):
        """The reserved provider-output keys flow through the processor and
        come out readable via the typed views (issue #52 Phase 2)."""
        strategy = _ThreeTupleStrategy(
            metadata={
                "grounding": {"sources": [{"uri": "https://a", "title": "A"}], "queries": ["q"]},
                "reasoning": "thought",
                "tool_calls": [{"id": "c1", "name": "f", "arguments": "{}"}],
            }
        )
        config = ProcessorConfig(max_workers=1, attempt_timeout=2.0)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="i1", strategy=strategy, prompt="hi"))
            result = await processor.process_all()

        item = result.results[0]
        assert item.success
        assert item.grounding is not None
        assert item.grounding.sources[0].uri == "https://a"
        assert item.grounding.queries == ["q"]
        assert item.reasoning == "thought"
        assert item.tool_calls is not None and item.tool_calls[0].name == "f"
        assert item.logprobs is None


# ── gemini_safety_ratings removed in 1.0 ─────────────────────────────


def test_gemini_safety_ratings_field_is_removed():
    assert "gemini_safety_ratings" not in {f.name for f in dataclasses.fields(WorkItemResult)}
    result = WorkItemResult(
        item_id="x",
        success=True,
        metadata={"safety_ratings": {"HARM_HATE": "LOW"}},
    )
    assert not hasattr(result, "gemini_safety_ratings")
    assert result.metadata == {"safety_ratings": {"HARM_HATE": "LOW"}}
    with pytest.raises(TypeError, match="gemini_safety_ratings"):
        WorkItemResult(item_id="x", success=True, gemini_safety_ratings={})  # type: ignore[call-arg]
