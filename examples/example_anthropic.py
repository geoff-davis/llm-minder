"""Example of using llm-minder with Anthropic via the built-in AnthropicModel.

This example uses the native ``AnthropicModel`` and ``AnthropicStrategy`` classes
(added in v1.2, issue #314), which call the Messages API directly. For writing your
own ``LLMCallStrategy`` against a provider SDK, see
``examples/example_llm_strategies.py``.

## Installation

```bash
pip install 'llm-minder[anthropic]'
```

## Setup

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Features demonstrated

1. ``llm("anthropic:...")`` with the high-level ``process_prompts`` API.
2. ``AnthropicModel.from_api_key`` with a system instruction, a lower effort, and
   automatic prompt caching (``cache_control``), on the cheaper Haiku model.
3. A structured-output ``response_parser``.
4. Cache read/write token tracking and cache-aware billing estimates.
5. Truncated and refused responses surfacing as categorized failures.
"""

import asyncio
import os

from pydantic import BaseModel

from llm_minder import (
    AnthropicModel,
    AnthropicStrategy,
    CachedTokenRates,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    llm,
    process_prompts,
    pydantic_json_parser,
)

# The default examples use the most capable model; the batch classifier uses the
# cheaper Haiku model.
MODEL = "claude-opus-5-5"
CHEAP_MODEL = "claude-haiku-5-5"


class SentimentOutput(BaseModel):
    """Structured output for sentiment classification."""

    sentiment: str  # "positive" | "negative" | "neutral"
    confidence: float


async def example_simple_text() -> None:
    """Basic text generation through the factory and process_prompts."""
    print("\n=== Example 1: Simple text generation ===\n")

    strategy = llm(f"anthropic:{MODEL}", max_tokens=1_024)  # reads ANTHROPIC_API_KEY
    result = await process_prompts(
        strategy,
        [
            "What is the capital of France? Answer in one word.",
            "Explain quantum computing in one sentence.",
            "Name the largest planet in the solar system.",
        ],
        concurrency=3,
    )
    for r in result.results:
        if r.success:
            print(f"{r.item_id}: {r.output}")
        else:
            print(f"{r.item_id} FAILED ({r.error_category}): {r.error}")
    print(f"\nTokens: input={result.total_input_tokens} output={result.total_output_tokens}")


async def example_cached_batch_classifier() -> None:
    """A cheap, cached batch classifier with structured output."""
    print("\n=== Example 2: Cached batch classification ===\n")

    model = AnthropicModel.from_api_key(
        CHEAP_MODEL,
        max_tokens=256,
        max_connections=4,
        system_instruction=(
            "You classify user-supplied text as positive, negative, or neutral. "
            "Respond with ONLY a JSON object of the shape "
            '{"sentiment": "positive"|"negative"|"neutral", "confidence": <0.0-1.0>}.'
        ),
        # Cache the shared prefix (the system instruction) across calls. It only
        # caches once the prefix is long enough for the model's minimum.
        cache_control={"type": "ephemeral"},
        # Thinking and effort have no defaults here; ask for a quick answer.
        extra_body={"output_config": {"effort": "low"}},
    )
    strategy = AnthropicStrategy(model, response_parser=pydantic_json_parser(SentimentOutput))
    config = ProcessorConfig(max_workers=4, attempt_timeout=60.0)

    reviews = [
        "This product is amazing! Highly recommend.",
        "Not worth the money. Very disappointed.",
        "It's okay. Does the job.",
        "Arrived broken, but support replaced it quickly.",
    ]
    async with ParallelBatchProcessor[None, SentimentOutput, None](config=config) as processor:
        for i, text in enumerate(reviews):
            await processor.add_work(
                LLMWorkItem(item_id=f"review_{i}", strategy=strategy, prompt=text)
            )
        result = await processor.process_all()

    for r in result.results:
        if r.success and r.output is not None:
            print(
                f"{r.item_id}: {r.output.sentiment} ({r.output.confidence:.2f}) "
                f"stop_reason={(r.metadata or {}).get('finish_reason')}"
            )
        else:
            print(f"{r.item_id} FAILED ({r.error_category}): {r.error}")

    billable = result.effective_input_tokens(
        CachedTokenRates.ANTHROPIC_READ,
        cache_write_rate=CachedTokenRates.ANTHROPIC_WRITE,
    )
    print(
        f"\nInput tokens: {result.total_input_tokens} "
        f"(cache reads {result.total_cached_tokens}, "
        f"cache writes {result.total_cache_write_tokens}); "
        f"billable-equivalent input: {billable}"
    )


async def example_truncation() -> None:
    """A response cut off before any text fails as response_truncated."""
    print("\n=== Example 3: Truncated response ===\n")

    strategy = llm(f"anthropic:{CHEAP_MODEL}", max_tokens=1)
    result = await process_prompts(strategy, ["Write a long poem about the sea."])
    r = result.results[0]
    if r.success:
        # With some text before the bound, the call succeeds; finish_reason says so.
        finish_reason = (r.metadata or {}).get("finish_reason")
        print(f"Partial text: {r.output!r} (finish_reason={finish_reason})")
    else:
        print(f"Failed as {r.error_category}: {r.error}")


async def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Set ANTHROPIC_API_KEY before running this example.")
        return
    await example_simple_text()
    await example_cached_batch_classifier()
    await example_truncation()


if __name__ == "__main__":
    asyncio.run(main())
