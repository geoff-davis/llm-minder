# llm-minder

**Run thousands of LLM calls concurrently without babysitting them.** When one
call hits a rate limit, every worker sharing that quota holds off on new calls
until the cooldown ends. Failures retry with the right strategy for their cause.
After a crash, a rerun resumes from its checkpoint instead of starting over.
Deadlines and spend caps stop a run cleanly, and token counts include the
attempts that failed.

It works with any async client: wrap your own, or use the built-in OpenAI,
Anthropic, Gemini, OpenRouter, DeepSeek, or PydanticAI support. It's for anyone running LLM
calls in bulk from one Python process: dataset labeling, extraction pipelines,
model evaluations, nightly jobs, and services that share one quota. For
latency-tolerant jobs, a provider's native batch API may be cheaper.

> Formerly `async-batch-llm`: releases through v0.32 used that name. See
> [Renamed in 1.0](stability.md#renamed-in-10).

## First batch

Install a provider and the optional terminal progress bar:

```bash
pip install 'llm-minder[openai,progress]'
export OPENAI_API_KEY='...'
```

```python
import asyncio
from llm_minder import llm, process_prompts

async def main():
    batch = await process_prompts(
        llm("openai:gpt-6-luna"), ["Summarize A", "Summarize B"],
        concurrency=10, progress=True,
    )
    print(batch.summary())

asyncio.run(main())
```

The same execution engine powers collection, completion-order streaming,
single calls, and the in-process `LLMCallPool`. Built-in provider wrappers are
conveniences, not requirements.

## Why not just `asyncio.gather`?

A semaphore plus `asyncio.gather()` caps concurrency, and for a small script
that is enough. Production batches need more:

| Problem | What llm-minder does |
| --- | --- |
| A 429 hits one call while 50 more are in flight | Workers sharing the quota pause together, then ramp back up |
| Rate limits and bad outputs need different handling | Separate retry budgets; `on_error()` can escalate to a stronger model |
| The process dies at item 80,000 | JSONL or SQLite checkpoints; a rerun replays compatible results |
| A run must not exceed a time or money budget | Item and batch deadlines, token/cost caps, a hard call cap, fail-fast |
| "How many tokens did that cost?" | Token counts and timing include retries and failed attempts |

On the GSM8K benchmark at the same concurrency, llm-minder processed 72–108
items/s against 55–58 for a hand-written semaphore pool
([Benchmarks](benchmarks.md)). [Compare alternatives](comparison.md) covers when
gather, Curator, a gateway, native batch, or a workflow engine is a better fit.

## Bring an existing client

`CallableStrategy` adapts an async SDK, gateway client, PydanticAI agent, or
internal service without introducing another executor:

```python
from llm_minder import ArtifactIdentity, CallOutcome, CallableStrategy

async def invoke(prompt, *, attempt, timeout, state):
    response = await client.generate(prompt, timeout=timeout)
    return CallOutcome(
        response.text,
        # token_usage must be a str -> int mapping, not an SDK usage object
        token_usage={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )

strategy = CallableStrategy(
    invoke,
    identity=ArtifactIdentity(provider="internal", model="summary-route"),
)
```

The [credential-free application example](https://github.com/geoff-davis/llm-minder/blob/main/examples/example_callable_application.py)
demonstrates a lazy source, stateful recovery, failed-attempt token accounting,
bounded result handoff, transactional writes, and checkpoint replay.

## Where to go next

- [Getting Started](getting-started.md) — one batch through streaming and an
  existing client
- [Choosing Your Limits](choosing-your-limits.md) — concurrency, pools,
  admission, retries, and deadlines, and how to probe limits you don't know
- [Deadlines, Budgets and Fail-Fast Guardrails](guardrails.md) — stopping a run
  on time, spend, call count, or a fatal error
- [Use Your Existing Async Client](callable-integration.md) — `CallableStrategy`
  and `LLMCallPool` in an application
- [Large Runs (100k–1M)](large-runs.md) — SQLite checkpoints, streaming, and
  restarts at scale
- [Bounded Work and Backpressure](bounded-work.md) — memory behavior for large
  lazy sources
- [Results, Artifacts, and Resume](results-and-artifacts.md) — terminal results
  and durable replay
- [Compare alternatives](comparison.md) — when gather, Curator, a gateway,
  native batch, or a workflow engine is a better fit
- [Benchmarks](benchmarks.md) — throughput, wall time, and a model bake-off on
  GSM8K
- [Troubleshooting and FAQ](troubleshooting.md) — operational symptoms and
  fixes
- [API Reference](api/core.md) — public classes and functions
- [API Stability](stability.md) — the compatibility promise and what it covers
- [1.0 migration guide](migration/v1.0.md) — upgrading from 0.x

## Project status

llm-minder is stable. Since 1.0.0, the public API is covered by the compatibility
promise on the [API stability](stability.md) page: minor and patch releases don't
break code that uses the stable API, and anything slated for removal warns for at
least one minor release before a major release removes it. Provisional names, listed on the same
page, may still change in a minor release. Run your tests with
`-W error::DeprecationWarning` to catch deprecations early
([Testing](testing.md#4-catching-deprecations-early) covers a google-genai warning
to filter on Python 3.14). Contributions and production feedback are welcome.

Upgrading from 0.x? Read the [1.0 migration guide](migration/v1.0.md); it starts
from v0.32, the last 0.x release and the last to support Python 3.10.

## License

MIT License — see [LICENSE](https://github.com/geoff-davis/llm-minder/blob/main/LICENSE).
