# Anthropic Integration

Native Anthropic support arrived in v1.2 via `AnthropicModel`, `AnthropicStrategy`,
and `AnthropicErrorClassifier`. It calls the Messages API directly, so cache reads
and writes, refusals and thinking blocks come through as Anthropic reports them.
All three names are provisional (see [API stability](stability.md)).

## Installation

```bash
pip install 'llm-minder[anthropic]'
```

The extra needs `anthropic` 1.0.0 or later; CI tests the 1.x and 2.x lines. These
SDK releases run on `httpx2`, not `httpx`, so a custom HTTP client must be built with
`anthropic.DefaultAsyncHttpxClient` (see [Connection pool sizing](#connection-pool-sizing-max_connections)).

## Authentication

Set the `ANTHROPIC_API_KEY` environment variable, or pass `api_key=` to
`AnthropicModel.from_api_key()`. With no `api_key`, the SDK resolves credentials
itself, including its other supported mechanisms.

## Quick start

```python
import asyncio

from llm_minder import process_prompts, llm


async def main() -> None:
    strategy = llm("anthropic:claude-opus-5-5")  # reads ANTHROPIC_API_KEY
    result = await process_prompts(strategy, ["Summarize A", "Summarize B"])
    for item in result.results:
        print(item.item_id, item.output)


asyncio.run(main())
```

The explicit form gives full control:

```python
from llm_minder import (
    AnthropicModel,
    AnthropicStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
)

model = AnthropicModel.from_api_key("claude-opus-5-5", max_connections=10)
strategy = AnthropicStrategy(model)
config = ProcessorConfig(max_workers=10, attempt_timeout=120.0)

async with ParallelBatchProcessor[None, str, None](config=config) as processor:
    await processor.add_work(LLMWorkItem(item_id="hello", strategy=strategy, prompt="Hi!"))
    result = await processor.process_all()
```

`AnthropicStrategy` recommends `AnthropicErrorClassifier`, so the processor, the
streaming API and `call()` select it automatically.

## Retry ownership

`from_api_key()` builds the SDK client with `max_retries=0`. The framework owns
retries, so its attempt counts, timeouts, cooldowns and quota accounting reflect
real provider requests. An explicit `max_retries=N` overrides this and is kept when
an owned client reopens. If you pass your own client, build it with `max_retries=0`.

## Choosing a model

`AnthropicModel` takes any Claude model id. Use `claude-opus-5-5` for the hardest
work and `claude-haiku-5-5` for cheap, high-volume batches:

```python
from llm_minder import llm

strategy = llm(
    "anthropic:claude-haiku-5-5",
    max_tokens=1_024,
    system_instruction="Classify the sentiment as positive, negative or neutral.",
)
```

## Requests

- A string prompt becomes one user message. For several turns, or
  `cache_control` on individual content blocks, give the work item a message list
  with `messages=` (see [Structured message prompts](structured-prompts.md)).
  Leading `system` messages become the top-level `system` field, as text blocks
  that keep their `cache_control`; `user` and `assistant` messages pass through.
  Other roles, non-text parts, and a system message after a user or assistant one
  raise `ValueError` before the request is sent.
- `system_instruction` becomes the top-level `system` field. A per-call
  `system_instruction` wins over the model's default, and a system message in a
  message list wins over both. The cache group is labelled by the system actually
  sent.
- `max_tokens` is required by the API and defaults to 16,000. Calls are
  non-streaming, and the SDK refuses a non-streaming call whose `max_tokens` could
  take longer than 10 minutes (above about 21,000) unless the client has an explicit
  `timeout`, for example `from_api_key(..., timeout=900.0)`.
- `temperature` is sent only when you set it. Current Claude models reject
  sampling parameters, and that 400 fails fast as a non-retryable `client_error`,
  so leave the strategy's `temperature` at `None`.
- Thinking and effort have no defaults here; the model's own defaults apply. Set
  them through `extra_body` or per-call `config`:

```python
from llm_minder import AnthropicModel, AnthropicStrategy

model = AnthropicModel.from_api_key(
    "claude-opus-5-5",
    extra_body={"output_config": {"effort": "low"}},
)
strategy = AnthropicStrategy(model)

# Or per strategy, on every call:
strategy = AnthropicStrategy(model, generation_config={"output_config": {"effort": "high"}})
```

`extra_body` and `config` merge into the request, so any Messages API field works
(`tools`, `tool_choice`, `stop_sequences`, `metadata`, `service_tier`, ...).
Fields the installed SDK's `messages.create` doesn't name are sent through the
SDK's own `extra_body`. Per-call `config` wins over the model's `extra_body`, which
makes `ModelStrategy.attempt_generation_config()` a way to raise effort or
`max_tokens` on a retry.

## Responses

- `text` is the concatenation of the response's text blocks.
- With text and `stop_reason="max_tokens"`, the call succeeds and
  `metadata["finish_reason"]` says `max_tokens` (the same policy as OpenAI's
  `length`).
- With no text, the call raises `EmptyResponseError` with the tokens already
  billed attached, and `reason` says why:

  | `stop_reason` | Exception | `reason` | Category |
  | --- | --- | --- | --- |
  | `max_tokens`, `model_context_window_exceeded` | `TruncatedResponseError` | `max_tokens` | `response_truncated` |
  | `refusal` | `EmptyResponseError` | `content_filter` | `empty_response` |
  | `tool_use` | `EmptyResponseError` | `tool_call` | `empty_response` |
  | `end_turn`, `stop_sequence`, `pause_turn` | `EmptyResponseError` | `no_content` | `empty_response` |

  `provider_reason` holds the raw `stop_reason`. A refusal's `stop_details.category`
  is included in the exception message.

Metadata keys: `provider` (`"anthropic"`), `finish_reason` (the `stop_reason`),
`model`, `request_id` (when the SDK reports one) and `stop_details` (on a
refusal). These are provisional diagnostics. The reserved keys read by the typed
views are:

- **`reasoning`**: the text of the response's `thinking` blocks, joined. It's
  often empty, because current models omit the thinking text unless you ask for
  it. Access via `item_result.reasoning`.
- **`tool_calls`**: the `tool_use` blocks as `[{"id", "name", "arguments"}]`, with
  `arguments` the JSON-encoded `input`. Access via `item_result.tool_calls`.
  Visibility only: the framework never executes tools.

## Prompt caching

`cache_control` turns on automatic prompt caching for every call: it is sent as the
top-level `cache_control` request field.

```python
from llm_minder import AnthropicModel, AnthropicStrategy, CachedTokenRates

model = AnthropicModel.from_api_key(
    "claude-haiku-5-5",
    system_instruction=LONG_SHARED_INSTRUCTIONS,
    cache_control={"type": "ephemeral"},  # or {"type": "ephemeral", "ttl": "1h"}
)
strategy = AnthropicStrategy(model)

result = await processor.process_all()
billable = result.effective_input_tokens(
    CachedTokenRates.ANTHROPIC_READ,  # cache reads cost 10% of normal input
    cache_write_rate=CachedTokenRates.ANTHROPIC_WRITE,  # 1.25x; ANTHROPIC_WRITE_1H is 2.0x
)
```

To place breakpoints yourself, put `cache_control` on individual text blocks of
a message list:

```python
from llm_minder import LLMWorkItem

item = LLMWorkItem(
    item_id="q1",
    strategy=strategy,
    messages=[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": LONG_SHARED_INSTRUCTIONS, "cache_control": {"type": "ephemeral"}},
            ],
        },
        {"role": "user", "content": "First question"},
    ],
)
```

Anthropic's own `usage.input_tokens` excludes cache reads and writes. llm-minder
adds them back so `input_tokens` counts every prompt token, as it does for the other
providers:

| `LLMResponse` field | From `usage` |
| --- | --- |
| `input_tokens` | `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` |
| `cached_input_tokens` | `cache_read_input_tokens` |
| `cache_write_input_tokens` | `cache_creation_input_tokens` |
| `output_tokens` | `output_tokens` |

Cache writes cost 1.25× normal input with the default 5-minute TTL
(`CachedTokenRates.ANTHROPIC_WRITE`) and 2× with the 1-hour TTL
(`CachedTokenRates.ANTHROPIC_WRITE_1H`); pass the matching constant as
`cache_write_rate` to `effective_input_tokens()` or `estimated_cost()`. Calls are
grouped for the per-group cache statistics by a hash of the system instruction, or
by `cache_group=` when you set one.

## Error handling

`AnthropicErrorClassifier` dispatches on the SDK's exception types and status codes
first, and on message text only for exceptions without either:

- 429 → retryable rate limit with a coordinated cooldown. A `retry-after-ms` or
  `retry-after` header becomes `ErrorInfo.suggested_wait`, a floor on the
  cooldown. A 429 whose `error_code` is `enforced_spend_limit_reached` is
  `insufficient_balance` instead: not retried.
- 529 (overloaded) → retryable, category `server_overload`, and treated like a rate
  limit: the overload is provider-wide, so every worker pauses in the shared
  cooldown rather than each item retrying into it.
- 500/502/503/504 (and 408) → retryable `server_error`.
- `APITimeoutError` → retryable `api_timeout`; `APIConnectionError` → retryable
  `network_error`.
- 401 → `authentication`; 403 → `permission_denied`; 400, 404, 413 and 422 →
  `client_error`. None are retried.
- Empty and truncated responses, framework timeouts, Pydantic `ValidationError`
  and programming errors keep the categories every built-in classifier gives them.

## Convenience constructor

```python
from llm_minder import AnthropicModel

model = AnthropicModel.from_api_key(
    "claude-opus-5-5",
    api_key=None,                 # None: the SDK resolves ANTHROPIC_API_KEY
    base_url=None,                # override the SDK default if needed
    max_connections=50,           # size the connection pool to match max_workers
    max_tokens=16_000,            # required by the API
    system_instruction="...",     # default system prompt
    cache_control=None,           # e.g. {"type": "ephemeral"}
    extra_headers=None,           # forwarded on every request
    extra_body=None,              # merged into every request
    metadata_extractors=None,     # extra metadata keys (see api/core.md)
    cache_group=None,             # label for the cache-group statistics
    timeout=120.0,                # other kwargs go to AsyncAnthropic
)
```

The model owns a client built this way and closes it when the framework cleans up
the strategy; a later call reopens it. A client you pass to `AnthropicModel(...)`
yourself is never closed by llm-minder.

## Platform clients (Vertex AI, Bedrock, Foundry)

`AnthropicModel` accepts any async Anthropic client with `messages.create`, so the
SDK's platform clients work unchanged. Build them with `max_retries=0`; you own
their lifecycle.

```python
from anthropic import AsyncAnthropicBedrockMantle, AsyncAnthropicVertex

from llm_minder import AnthropicModel, AnthropicStrategy

vertex = AsyncAnthropicVertex(region="us-east5", project_id="my-project", max_retries=0)
strategy = AnthropicStrategy(AnthropicModel("claude-opus-5-5", vertex))

bedrock = AsyncAnthropicBedrockMantle(aws_region="us-east-1", max_retries=0)
strategy = AnthropicStrategy(AnthropicModel("claude-opus-5-5", bedrock))
```

Platforms may name models differently; use the model id your platform documents.
Close the client yourself when you're done (`await vertex.close()`).

## Connection pool sizing (`max_connections`)

Without `max_connections`, the SDK's default pool allows 1,000 connections but keeps
only 100 alive between requests, and llm-minder can't see it. Pass
`max_connections` to size the pool to your worker count; it is also exposed as
`model.max_concurrency`, which the processor uses to warn about and gate attempts
above the pool size. `ProcessorConfig(concurrency=N)` resizes an owned default-pool
client before the first request.

For finer control, build the client yourself with the SDK's `httpx2`-based
`DefaultAsyncHttpxClient` (a plain `httpx` client is rejected) and declare the limit
on the processor:

```python
import anthropic

from llm_minder import AnthropicModel, ProcessorConfig

http_client = anthropic.DefaultAsyncHttpxClient(
    limits=type(anthropic.DEFAULT_CONNECTION_LIMITS)(
        max_connections=64, max_keepalive_connections=64
    )
)
client = anthropic.AsyncAnthropic(http_client=http_client, max_retries=0)
model = AnthropicModel("claude-opus-5-5", client)  # caller owns and closes client
config = ProcessorConfig(max_workers=100, max_provider_concurrency=64)
```

## Out of scope for now

Image and document content blocks, streaming mode, the refusal `fallbacks` beta,
Message Batches and server tools.

## See also

- [`examples/example_anthropic.py`](https://github.com/geoff-davis/llm-minder/blob/main/examples/example_anthropic.py)
  — runnable example.
- [`docs/OPENROUTER_INTEGRATION.md`](OPENROUTER_INTEGRATION.md) — Claude through
  OpenRouter, with the same `messages=` and `cache_control` pattern.
- [Prompt caching guide](prompt-caching.md) — cache groups and hit-rate floors.
