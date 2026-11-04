# Core API Reference

## ParallelBatchProcessor

::: llm_minder.ParallelBatchProcessor
    options:
      show_bases: false

## LLMWorkItem

::: llm_minder.LLMWorkItem

## WorkItemResult

::: llm_minder.WorkItemResult

## AttemptTiming

::: llm_minder.AttemptTiming

## WorkItemTiming

::: llm_minder.WorkItemTiming

## ProcessorConfig

::: llm_minder.ProcessorConfig

## RetryConfig

::: llm_minder.core.RetryConfig

## RateLimitConfig

::: llm_minder.core.RateLimitConfig

## GuardrailConfig

::: llm_minder.GuardrailConfig

## AbortMode

::: llm_minder.AbortMode

## AttemptUsage

::: llm_minder.AttemptUsage

## CacheGroupTracker

::: llm_minder.CacheGroupTracker

## CacheFloorBreach

::: llm_minder.CacheFloorBreach

## BatchBudgetExceeded

::: llm_minder.BatchBudgetExceeded

## StartupRampConfig

::: llm_minder.StartupRampConfig

## BatchResult

::: llm_minder.BatchResult

## BatchTermination

::: llm_minder.BatchTermination

## Grounding

::: llm_minder.Grounding

## GroundingSource

::: llm_minder.GroundingSource

## ToolCall

::: llm_minder.ToolCall

## ErrorCategory

::: llm_minder.ErrorCategory

## TimeoutCategory

::: llm_minder.TimeoutCategory

## AsyncBatchLLMError

::: llm_minder.AsyncBatchLLMError

Every exception type the library exports subclasses it, alongside its original
built-in base, so `except AsyncBatchLLMError` catches the library's own errors
and existing `except TimeoutError` (or `RuntimeError`, `ValueError`) clauses
keep working.

## Lifecycle exceptions

::: llm_minder.StreamFinalizationError

::: llm_minder.CleanupInterruptedError

## Admission closure

::: llm_minder.BatchAdmissionClosedError

`add_work()` raises `BatchAdmissionClosedError` after finish, shutdown, or abort.
Rejected submissions receive no submission index. Use a new processor for further work.

## Cleanup and shutdown

### `async def cleanup() -> None` / `async def shutdown() -> None`

Release every owned resource in dependency order. `cleanup()`, `shutdown()`,
and `async with` exit all run the same ordered close:

1. Runtime tasks and callbacks: the stream finalizer, workers, queued work,
   progress callbacks, background post-processors, and callback threads.
2. Admission resources (quota scopes and the rate-limit coordinator).
3. Prepared strategies (`strategy.cleanup()`).
4. The artifact store, including after an early stream exit or a failed run.

```python
await processor.shutdown()
```

Behavior (see `docs/cleanup-lifecycle-contract.md` for the full contract):

- **Every step is attempted.** A failing strategy `cleanup()` does not stop
  sibling strategies or the artifact store from closing. Every failure is
  logged with a traceback; with no body exception, the first failure is
  raised after all steps have run. Inside `async with`, a body exception
  stays primary and cleanup failures are only logged.
- **Repeat calls are safe.** A step that succeeded is never repeated. A step
  that failed or was interrupted is retried by the next explicit close.
  Concurrent calls share one in-flight attempt. Because a retry re-invokes a
  strategy's whole `cleanup()`, user `cleanup()` implementations must be
  idempotent and safe after a partially completed earlier attempt.
- **No deadline.** Cleanup waits for the artifact store and the call-pool drain
  to finish. The two-second worker and progress thresholds only log a warning
  and keep waiting; any step still running after 30 seconds logs one warning.
  A dependent resource is never closed while the resource it depends on is
  still running, including synchronous callback threads.
- **Cancellation.** Cancelling the task that is closing once defers that
  cancellation until cleanup finishes, then re-raises it (cleanup errors are
  logged, not raised). Cancelling it a second time force-aborts: the running
  step is abandoned, later steps are skipped and logged, and the cancellation
  propagates immediately. This is the operator's escape from a cleanup that
  never returns and deliberately sacrifices durability.
- **Interruptions.** A strategy `cleanup()` that raises `CancelledError`
  itself, or whose private task is cancelled by something else, surfaces as
  `CleanupInterruptedError` (an ordinary `Exception`); the caller's task is
  not cancelled and the next close retries the step.
- `KeyboardInterrupt` and `SystemExit` raised by a cleanup step keep their
  type and take precedence over a deferred cancellation.
- Once a close has started, preparing a new strategy raises `RuntimeError`.

## Provider metadata

### Response metadata (`WorkItemResult.metadata`)

Provider metadata (Gemini safety ratings and finish reason, OpenRouter
provider/routed model, etc.) flows into `WorkItemResult.metadata` — a plain
`dict[str, Any] | None`. The parsed output stays in `WorkItemResult.output`;
you no longer wrap it in a separate response object.

Conservative structured-output recovery uses three reserved metadata keys:
`structured_output_recovered`, `structured_output_recovery_reason`, and
`structured_output_retries_avoided`. Read them through the corresponding typed
properties on `LLMResponse` or `WorkItemResult`.

> **Removed:** the old `GeminiResponse` wrapper and the `include_metadata`
> opt-in were removed in v0.6.0. Read metadata off `result.metadata` instead.
> For Gemini safety ratings specifically, `result.metadata["safety_ratings"]`
> carries them (the `result.gemini_safety_ratings` field was removed in 1.0).

**Usage:**

```python
result = await processor.process_all()
first = result.results[0]
ratings = (first.metadata or {}).get("safety_ratings")
if ratings and ratings.get("HARM_CATEGORY_HATE_SPEECH") == "HIGH":
    log_flagged_content(first.output)
```

Two more reserved keys feed the per-group cache statistics (provisional, new in
0.30): `cache_group`, a string naming the prompt-cache group an attempt belongs to,
and `cache_miss_reason`, a provider's explanation of a cache miss. Built-in models
set `cache_group`; see [Cache hit-rate floor](../guardrails.md#cache-hit-rate-floor).

---

### Typed auxiliary output (grounding, reasoning, tool calls, logprobs)

> **Provisional.** The shapes were adjusted in v0.27 for OpenAI's Responses API.
> The views and the reserved keys' shapes may be promoted to stable at 1.0.0rc1 if
> real-provider runs fill them in correctly; until then they can change in a minor
> release, and `logprobs` is likely to stay provisional after 1.0. The `metadata`
> dict channel itself is stable. If you persist these shapes, read them back
> defensively. See [API stability](../stability.md#provisional-through-10).

Provider-specific structured output travels through `metadata` under four
**reserved keys** with documented plain-dict shapes (JSON-serializable, so
persisting `metadata` as-is works):

| Key | Shape | Emitted by |
| --- | ----- | ---------- |
| `grounding` | `{"sources": [{"uri", "title"}], "queries": [str], "supports": [dict]}` | Gemini models, when the response carries `google_search` grounding |
| `reasoning` | `str`: the model's reasoning/thinking trace | Chat Completions: `reasoning_content` (e.g. DeepSeek), falling back to `reasoning` (e.g. OpenRouter). Responses API: reasoning-item text, else the joined reasoning summaries (OpenAI, when requested) |
| `tool_calls` | `[{"id": str\|None, "name": str, "arguments": str}]`, with `arguments` as the raw JSON string | OpenAI-compatible models: Chat `tool_calls`, or Responses `function_call` items (`id` is the `call_id`) |
| `logprobs` | provider-shaped `dict`/`list` (plain JSON) | OpenAI-compatible models, when requested: the Chat logprobs object, or a list of Responses per-token entries |

Both `LLMResponse` and `WorkItemResult` expose **lazy read-only typed
views** over these keys — parsed from `metadata` on each access, never
cached, never stored twice:

```python
result = await processor.process_all()
item = result.results[0]

if item.grounding:                       # Grounding | None
    for source in item.grounding.sources:  # list[GroundingSource]
        print(source.uri, source.title)
    print(item.grounding.queries)          # list[str]

print(item.reasoning)                    # str | None
for call in item.tool_calls or []:       # list[ToolCall] | None
    print(call.name, call.arguments)     # arguments: raw JSON string
print(item.logprobs)                     # Any | None (provider-shaped)
```

The view classes (`Grounding`, `GroundingSource`, `ToolCall`) are exported
at the top level. Parsing is lenient: malformed metadata yields `None` (or
drops the bad entry) rather than raising — which also means a future shape
change degrades to `None` views rather than errors.

**Boundaries:**

- `tool_calls` is **visibility only** — the framework never executes tools.
  Feed them to your own dispatch loop (or use an agent framework via
  `PydanticAIStrategy`). Extracted for OpenAI-compatible models only; Gemini
  function-call parts are not extracted.
- On Chat Completions, a response whose `content` is `null` (for example a
  pure tool-call turn) raises `EmptyResponseError` before any result exists,
  so `tool_calls` surfaces only when the model returned text alongside the
  calls. On `OpenAIModel`'s Responses surface, a function-call-only reply
  succeeds with empty text and `tool_calls` populated.
- Auxiliary output does not survive empty/safety-blocked responses — the
  call fails first.

---

## TokenUsage

::: llm_minder.TokenUsage

## CachedTokenRates

::: llm_minder.CachedTokenRates

## Type aliases

`SimpleBatchProcessor[T]`, `SimpleWorkItem[T]` and `SimpleResult[T]` shorten the
generic types for string prompts with no context. `PostProcessorFunc` and
`ProgressCallbackFunc` are the callable types accepted by `post_processor` and
`progress_callback`.

::: llm_minder.SimpleBatchProcessor

::: llm_minder.SimpleWorkItem

::: llm_minder.SimpleResult

::: llm_minder.PostProcessorFunc

::: llm_minder.ProgressCallbackFunc

## Removed names

`BatchProcessor`, `ProcessingStats`, `LLMGateway` and `grounding_metadata_extractor`
were removed in 1.0; see [Removed in 1.0](../stability.md#removed-in-10) for
replacements.
