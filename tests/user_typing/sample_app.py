"""A small application written only against llm_minder's top-level public API.

CI type-checks this file with ``mypy --strict`` (``make typecheck-user``), as a user's
own code would be checked, and tests/test_user_typing.py runs the offline parts. If a
change to the library breaks it, users' typed code breaks the same way.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_type

from llm_minder import (
    BaseMiddleware,
    BaseObserver,
    BatchResult,
    CachedTokenRates,
    CallableStrategy,
    CallOutcome,
    EmptyResponseError,
    ErrorCategory,
    GeminiStrategy,
    Grounding,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallPool,
    LLMCallStrategy,
    LLMMinderError,
    LLMResponse,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessingEvent,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    RetryState,
    TokenUsage,
    ToolCall,
    WorkItemResult,
    call_result,
    llm,
    process_prompts,
    process_stream,
    pydantic_json_parser,
)


@dataclass(frozen=True)
class Ticket:
    ticket_id: int
    customer: str


class EchoStrategy(LLMCallStrategy[str]):
    """A custom strategy returning the three-tuple contract."""

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, dict[str, Any] | None]:
        tokens: TokenUsage = {
            "input_tokens": len(prompt),
            "output_tokens": 1,
            "total_tokens": len(prompt) + 1,
        }
        return prompt.upper(), tokens, {"attempt": attempt}


async def _invoke(
    prompt: str, *, attempt: int, timeout: float, state: RetryState | None
) -> CallOutcome[dict[str, int]]:
    return CallOutcome({"length": len(prompt)}, token_usage={"total_tokens": 1})


class Counter(BaseObserver):
    def __init__(self) -> None:
        self.completed = 0

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        if event is ProcessingEvent.ITEM_COMPLETED:
            self.completed += 1


class Tag(BaseMiddleware[str, str, Ticket]):
    async def after_process(
        self, result: WorkItemResult[str, Ticket]
    ) -> WorkItemResult[str, Ticket]:
        return result


async def run_processor(tmp_dir: Path) -> BatchResult[str, Ticket]:
    strategy = EchoStrategy()
    counter = Counter()
    config = ProcessorConfig(
        max_workers=2,
        attempt_timeout=5.0,
        retry=RetryConfig(max_attempts=2, initial_wait=0.01, max_wait=0.02, jitter=False),
        guardrails=GuardrailConfig(
            abort_on_error_categories=frozenset({ErrorCategory.AUTHENTICATION})
        ),
    )
    store = JsonlArtifactStore(tmp_dir / "artifacts.jsonl")
    try:
        async with ParallelBatchProcessor[str, str, Ticket](
            config=config,
            observers=[counter],
            middlewares=[Tag()],
            artifact_store=store,
            resume=ResumePolicy.REUSE_SUCCESSES,
        ) as processor:
            for ticket in (Ticket(1, "ada"), Ticket(2, "grace")):
                await processor.add_work(
                    LLMWorkItem(
                        f"t{ticket.ticket_id}", strategy, f"hello {ticket.customer}", ticket
                    )
                )
            result = await processor.process_all()
    finally:
        await store.close()

    for item in result.results:
        assert_type(item.output, str | None)
        assert_type(item.context, Ticket | None)
        if item.success and item.context is not None:
            print(item.context.customer, item.output)
    assert_type(result.by_id(), dict[str, WorkItemResult[str, Ticket]])
    print(result.estimated_cost(1.0, 2.0, cached_token_rate=CachedTokenRates.OPENAI))
    return result


async def run_high_level() -> list[dict[str, int]]:
    strategy = CallableStrategy(_invoke)
    batch = await process_prompts(strategy, ["a", "bb"], concurrency=2)
    outputs = [r.output for r in batch.results if r.output is not None]
    async for streamed in process_stream(strategy, ["ccc"]):
        if streamed.output is not None:
            outputs.append(streamed.output)
    single = await call_result(strategy, "dddd")
    if single.success and single.output is not None:
        outputs.append(single.output)
    async with LLMCallPool(strategy) as pool:
        outputs.append(await pool.submit("eeeee"))
    return outputs


async def run_callable_processor() -> dict[str, int] | None:
    """The context manager and a CallableStrategy keep the processor's output type."""
    async with ParallelBatchProcessor[str, dict[str, int], None]() as processor:
        assert_type(processor, ParallelBatchProcessor[str, dict[str, int], None])
        await processor.add_work(
            LLMWorkItem(item_id="x", prompt="xyz", strategy=CallableStrategy(_invoke))
        )
        result = await processor.process_all()
    output = result.results[0].output
    assert_type(output, dict[str, int] | None)
    return output


def describe(response: LLMResponse | WorkItemResult[str, Any]) -> str:
    """The typed provider-output views, without touching raw metadata dicts."""
    parts: list[str] = []
    grounding: Grounding | None = response.grounding
    if grounding is not None:
        parts.extend(source.uri for source in grounding.sources)
    if response.reasoning:
        parts.append(response.reasoning)
    calls: list[ToolCall] = response.tool_calls or []
    parts.extend(f"{c.name}({json.loads(c.arguments)})" for c in calls)
    return "; ".join(parts)


def handle(result: WorkItemResult[Any, Any]) -> str:
    error = result.exception
    if isinstance(error, EmptyResponseError):
        return f"empty: {error.reason} ({error.provider_reason})"
    if isinstance(error, LLMMinderError):
        return f"framework: {type(error).__name__}"
    return result.error_category or "ok"


@dataclass
class Summary:
    title: str


def provider_strategies() -> list[LLMCallStrategy[Any]]:
    """Type-checked only; building these needs provider credentials."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    parsed: LLMCallStrategy[Answer] = llm(
        "openai:gpt-6-luna", response_parser=pydantic_json_parser(Answer)
    )
    plain = llm("deepseek:deepseek-v4.1-flash", generation_config={"thinking": False})
    gemini: Mapping[str, GeminiStrategy[str]] = {}
    return [parsed, plain, *gemini.values()]
