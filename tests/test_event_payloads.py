"""Every ProcessingEvent carries the payload keys docs/api/observers.md documents.

The documented keys are part of the stability promise (and of the API snapshot); this
runs small batches that emit all of the events and checks each payload against them.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from llm_minder import (
    BaseObserver,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessingEvent,
    ProcessorConfig,
    ResumePolicy,
    TokenEstimate,
)
from llm_minder.testing import FakeRateLimitError

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "api_snapshot.py"
_spec = importlib.util.spec_from_file_location("api_snapshot", SCRIPT)
assert _spec is not None and _spec.loader is not None
api_snapshot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api_snapshot)


def _documented() -> dict[str, tuple[set[str], set[str]]]:
    """Map each event name to its (required, optional) documented keys."""
    documented = {}
    for line in api_snapshot._documented_payloads():
        match = re.fullmatch(r"event (\w+): \{(.*)\}", line)
        assert match, line
        keys = [key.strip() for key in match.group(2).split(",")]
        expanded = []
        for key in keys:
            # "admission_wait_p50/p95/p99_seconds" stands for three keys.
            if parts := re.fullmatch(r"(\w+?)_(p\d+(?:/p\d+)+)_(\w+)", key.rstrip("?")):
                prefix, percentiles, suffix = parts.groups()
                optional = "?" if key.endswith("?") else ""
                expanded.extend(f"{prefix}_{p}_{suffix}{optional}" for p in percentiles.split("/"))
            else:
                expanded.append(key)
        required = {key for key in expanded if not key.endswith("?")}
        optional = {key.removesuffix("?") for key in expanded if key.endswith("?")}
        documented[match.group(1)] = (required, optional)
    return documented


class _Recorder(BaseObserver):
    def __init__(self) -> None:
        self.payloads: dict[ProcessingEvent, list[dict[str, Any]]] = defaultdict(list)

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.payloads[event].append(dict(data))


class _Strategy(LLMCallStrategy[str]):
    def __init__(self) -> None:
        self.rate_limited = False

    async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
        if prompt == "rate-limit" and not self.rate_limited:
            self.rate_limited = True
            raise FakeRateLimitError()
        if prompt == "fail":
            raise ValueError("terminal failure")
        if prompt == "slow":
            await asyncio.sleep(5)
        return f"ok:{prompt}", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


async def _run(
    recorder: _Recorder,
    prompts: list[str],
    *,
    store: JsonlArtifactStore | None = None,
    **config: Any,
) -> None:
    strategy = _Strategy()
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=2, **config),
        observers=[recorder],
        artifact_store=store,
        resume=ResumePolicy.REUSE_ALL if store is not None else ResumePolicy.NONE,
    )
    async with processor:
        for index, prompt in enumerate(prompts):
            await processor.add_work(LLMWorkItem(f"{prompt}-{index}", strategy, prompt))
        await processor.process_all()


@pytest.mark.asyncio
async def test_every_event_carries_its_documented_keys(tmp_path, fast_retry, fast_rate_limit):
    recorder = _Recorder()
    quota = {
        "max_requests_per_minute": 6000,
        "max_tokens_per_minute": 600_000,
        "token_estimator": lambda *args, **kwargs: TokenEstimate(1, 1),
    }
    # Successes, a terminal failure, a rate limit with its cooldown, and quota admission.
    await _run(
        recorder,
        ["ok", "fail", "rate-limit"],
        retry=fast_retry,
        rate_limit=fast_rate_limit,
        **quota,
    )
    # Replay from a checkpoint.
    path = tmp_path / "artifacts.jsonl"
    for _ in range(2):
        store = JsonlArtifactStore(path)
        try:
            await _run(recorder, ["ok"], store=store, retry=fast_retry)
        finally:
            await store.close()
    # An item deadline.
    await _run(
        recorder,
        ["slow"],
        retry=fast_retry,
        guardrails=GuardrailConfig(total_timeout_per_item=0.05),
    )
    # A fail-fast abort on the failure's category.
    category = next(
        data["error_category"] for data in recorder.payloads[ProcessingEvent.ITEM_FAILED]
    )
    await _run(
        recorder,
        ["fail", "ok"],
        retry=fast_retry,
        guardrails=GuardrailConfig(abort_on_error_categories=frozenset({category})),
    )

    documented = _documented()
    assert set(documented) == {event.name for event in ProcessingEvent}
    missing_events = [event.name for event in ProcessingEvent if not recorder.payloads[event]]
    assert missing_events == []
    problems = []
    for event, payloads in recorder.payloads.items():
        required, _optional = documented[event.name]
        for data in payloads:
            if absent := required - set(data):
                problems.append(f"{event.name} is missing {sorted(absent)}: {sorted(data)}")
    assert problems == []
