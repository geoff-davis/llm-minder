"""The streaming helpers reach the processor only through its private interface (#175).

``streaming.py`` used to set and read seven private processor attributes. It now
calls a small set of private methods on ``ParallelBatchProcessor``; these tests
keep it that way and pin the methods' own contracts. The stream's behaviour
(acceptance wins, finish idempotency, cleanup precedence, ...) stays covered by
``test_streaming.py`` and ``test_cleanup_contract.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import llm_minder.streaming as streaming_module
from llm_minder import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from llm_minder._internal.cleanup import CleanupStep
from llm_minder.testing import FakeStrategy

STREAM_INTERFACE = {
    "_own_stream",
    "_completion_counts",
    "_end_stream_production",
    "_close_after_run",
}


def _private_processor_accesses() -> set[str]:
    tree = ast.parse(Path(streaming_module.__file__).read_text())
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "processor"
        and node.attr.startswith("_")
    }


def test_streaming_uses_only_the_private_interface() -> None:
    accesses = _private_processor_accesses()
    assert accesses <= STREAM_INTERFACE, (
        "streaming.py reaches processor internals outside the #175 interface: "
        f"{sorted(accesses - STREAM_INTERFACE)}. Add a method on ParallelBatchProcessor instead."
    )
    # Guard the guard: the scan must still see the calls it is meant to police.
    assert {"_own_stream", "_end_stream_production", "_close_after_run"} <= accesses


def test_interface_methods_exist_on_the_processor() -> None:
    for name in STREAM_INTERFACE:
        assert callable(getattr(ParallelBatchProcessor, name)), name


def _processor() -> ParallelBatchProcessor:
    return ParallelBatchProcessor(config=ProcessorConfig(max_workers=1))


async def test_own_stream_configures_ownership_before_start() -> None:
    processor = _processor()
    calls: list[str] = []

    async def step() -> None:
        calls.append("extra")

    source = ("caller.py", 7, "caller")
    processor._own_stream(
        capacity_warning_source=source, cleanup_steps=[CleanupStep("extra", step)]
    )
    assert processor._preserve_completed_result is True
    assert processor._capacity_warning_source == source

    processor.start()
    await processor.finish()
    async for _ in processor.results():
        pass
    report = await processor._close_after_run()
    report.raise_first()
    assert calls == ["extra"]


async def test_own_stream_refuses_a_started_processor() -> None:
    processor = _processor()
    processor.start()
    try:
        with pytest.raises(RuntimeError, match="before start"):
            processor._own_stream(capacity_warning_source=("caller.py", 1, "caller"))
    finally:
        await processor._close_after_run()


async def test_completion_counts_track_processed_and_total() -> None:
    processor = _processor()
    processor._own_stream(capacity_warning_source=("caller.py", 1, "caller"))
    processor.start()
    strategy = FakeStrategy("ok")
    for i in range(2):
        await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="p"))
    assert processor._completion_counts() == (0, 2)
    await processor._end_stream_production()
    async for _ in processor.results():
        pass
    assert processor._completion_counts() == (2, 2)
    (await processor._close_after_run()).raise_first()


async def test_end_stream_production_keeps_the_producer_error_primary() -> None:
    processor = _processor()
    processor._own_stream(capacity_warning_source=("caller.py", 1, "caller"))
    processor.start()
    error = ValueError("source failed")
    await processor._end_stream_production(error)
    with pytest.raises(ValueError, match="source failed"):
        async for _ in processor.results():
            pass
    await processor._close_after_run(primary_exception=error)


async def test_end_stream_production_is_a_no_op_once_the_stream_moved_on() -> None:
    processor = _processor()
    processor._own_stream(capacity_warning_source=("caller.py", 1, "caller"))
    processor.start()
    await processor._end_stream_production()
    # A second call (e.g. a producer finishing after finish()) must not raise.
    await processor._end_stream_production()
    async for _ in processor.results():
        pass
    (await processor._close_after_run()).raise_first()
