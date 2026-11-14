# llm-minder

Run independent LLM calls concurrently with bounded input and result handoffs,
coordinated retries, deadlines, checkpoint/replay, and terminal outcomes for
every accepted item. Use a built-in provider convenience or wrap an async
client you already have.

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
  admission, retries, and deadlines
- [Bounded Work and Backpressure](bounded-work.md) — memory behavior for large
  lazy sources
- [Results, Artifacts, and Resume](results-and-artifacts.md) — terminal results
  and durable replay
- [Compare alternatives](comparison.md) — when gather, Curator, a gateway,
  native batch, or a workflow engine is a better fit
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
