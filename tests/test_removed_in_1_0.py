"""Everything deprecated in v0.27 and v0.28 is gone in 1.0, with a pointer to the fix."""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import sys

import pytest

import async_batch_llm
from async_batch_llm import (
    BatchResult,
    GuardrailConfig,
    LLMCallPool,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    StartupRampConfig,
    WorkItemResult,
    process_prompts,
    process_stream,
)
from async_batch_llm.testing import FakeStrategy

REMOVED_ROOT_NAMES = {
    "LLMGateway": "use LLMCallPool",
    "BatchProcessor": "use ParallelBatchProcessor",
    "ProcessingStats": "get_stats()",
    "grounding_metadata_extractor": "metadata['grounding']",
}


@pytest.mark.parametrize(("name", "hint"), sorted(REMOVED_ROOT_NAMES.items()))
def test_removed_root_names_raise_with_a_hint(name, hint):
    assert name not in async_batch_llm.__all__
    with pytest.raises(AttributeError, match="removed") as caught:
        getattr(async_batch_llm, name)
    assert hint in str(caught.value)
    namespace: dict[str, object] = {}
    with pytest.raises(ImportError):
        exec(f"from async_batch_llm import {name}", namespace)  # noqa: S102


def test_internal_modules_keep_their_names():
    # Leaving the public API is not a deletion: the library still uses them.
    base = importlib.import_module("async_batch_llm.base")
    models = importlib.import_module("async_batch_llm.models")
    assert base.BatchProcessor is not None and base.ProcessingStats is not None
    assert callable(models.grounding_metadata_extractor)


def test_gateway_module_keeps_llm_call_pool_only():
    gateway = importlib.import_module("async_batch_llm.gateway")
    assert gateway.LLMCallPool is LLMCallPool
    assert gateway.__all__ == ["LLMCallPool"]
    with pytest.raises(AttributeError, match="LLMGateway was removed in 1.0"):
        gateway.LLMGateway  # noqa: B018
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        gateway.Nope  # noqa: B018
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        async_batch_llm.Nope  # noqa: B018


def test_plain_and_star_imports_are_silent():
    code = (
        "import warnings\n"
        "warnings.simplefilter('error', DeprecationWarning)\n"
        "warnings.filterwarnings('ignore', category=DeprecationWarning, module='google')\n"
        "import async_batch_llm\n"
        "from async_batch_llm import *\n"
        "from async_batch_llm.gateway import *\n"
        "assert 'LLMGateway' not in dir()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


CONFIGS = [
    (ProcessorConfig, 7),
    (RetryConfig, 4),
    (RateLimitConfig, 30.0),
    (StartupRampConfig, 2),
    (GuardrailConfig, 10.0),
]


@pytest.mark.parametrize(("cls", "value"), CONFIGS)
def test_config_classes_are_keyword_only(cls, value):
    with pytest.raises(TypeError, match="positional argument"):
        cls(value)
    assert all(f.kw_only for f in dataclasses.fields(cls))


@pytest.mark.parametrize(
    ("keyword", "hint"),
    [("timeout_per_item", "use attempt_timeout"), ("enable_detailed_logging", "logger")],
)
def test_removed_processor_config_keywords_point_to_the_replacement(keyword, hint):
    with pytest.raises(TypeError, match=f"'{keyword}'.*removed in 1.0") as caught:
        ProcessorConfig(**{keyword: 1})
    assert hint in str(caught.value)
    assert not hasattr(ProcessorConfig(), keyword)


def test_processor_config_keyword_typos_still_get_suggestions():
    with pytest.raises(TypeError, match="Did you mean 'max_workers'"):
        ProcessorConfig(max_worker=3)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("keyword", "hint"),
    [
        ("max_workers", r"ProcessorConfig\(max_workers=\.\.\.\)"),
        ("timeout_per_item", r"ProcessorConfig\(attempt_timeout=\.\.\.\)"),
        ("rate_limit_cooldown", r"RateLimitConfig\(cooldown_seconds=\.\.\.\)"),
    ],
)
def test_legacy_processor_parameters_point_to_the_config(keyword, hint):
    with pytest.raises(TypeError, match=rf"Unexpected processor keyword '{keyword}'.*{hint}"):
        ParallelBatchProcessor(**{keyword: 1})


def test_processor_keyword_typos_still_get_suggestions():
    with pytest.raises(TypeError, match="Did you mean 'post_processor'"):
        ParallelBatchProcessor(postprocessor=None)  # type: ignore[call-arg]


def test_processor_arguments_are_keyword_only():
    with pytest.raises(TypeError, match="positional argument"):
        ParallelBatchProcessor(None)  # type: ignore[misc]
    processor = ParallelBatchProcessor(config=ProcessorConfig(max_workers=3))
    assert processor.config.max_workers == 3
    assert ParallelBatchProcessor().config.max_workers == 5


def test_cache_hit_rate_is_a_plain_float():
    batch = BatchResult(
        results=[
            WorkItemResult(
                item_id="a",
                success=True,
                token_usage={
                    "input_tokens": 100,
                    "output_tokens": 1,
                    "total_tokens": 101,
                    "cached_input_tokens": 25,
                },
            )
        ]
    )
    assert type(batch.cache_hit_rate) is float
    assert batch.cache_hit_rate == 25.0
    with pytest.raises(TypeError, match="not callable"):
        batch.cache_hit_rate()  # type: ignore[operator]
    assert type(BatchResult(results=[]).cache_hit_rate) is float


def test_cached_token_rate_is_required():
    batch = BatchResult(results=[])
    with pytest.raises(TypeError, match="cached_token_rate"):
        batch.effective_input_tokens()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="cached_token_rate"):
        batch.estimated_cost(1.0, 1.0)  # type: ignore[call-arg]
    assert batch.effective_input_tokens(0.5) == 0
    assert batch.estimated_cost(1.0, 1.0, 0.5) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["process_prompts", "process_stream"])
async def test_integer_prompts_are_rejected(api):
    if api == "process_prompts":
        with pytest.raises(TypeError, match="Prompt at index 1 must be a string"):
            await process_prompts(FakeStrategy("ok"), ["a", 7])  # type: ignore[list-item]
    else:
        with pytest.raises(TypeError, match="Prompt at index 0 must be a string"):
            async for _ in process_stream(FakeStrategy("ok"), [7]):  # type: ignore[list-item]
                pass
