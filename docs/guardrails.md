# Deadlines, Budgets and Fail-Fast Guardrails

Guardrails are opt-in. Their defaults preserve normal completion-order,
retry, streaming, token-accounting, and cancellation behavior.

```python
from llm_minder import AbortMode, ErrorCategory, GuardrailConfig, ProcessorConfig

config = ProcessorConfig(
    max_workers=20,
    attempt_timeout=30,
    guardrails=GuardrailConfig(
        total_timeout_per_item=180,
        batch_timeout=3600,
        abort_on_error_categories=frozenset({
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.INSUFFICIENT_BALANCE,
        }),
        abort_mode=AbortMode.DRAIN_ACTIVE,
    ),
)
```

Timeouts must be finite numbers (not booleans) greater than zero, or `None`.
All deadline calculations use a
monotonic clock.

## Per-attempt timeout versus total item deadline

`ProcessorConfig.attempt_timeout` retains its existing meaning: it limits one
`strategy.execute()` attempt after provider-capacity admission. Retries can each
receive another attempt timeout.

`GuardrailConfig.total_timeout_per_item` is one end-to-end logical-item budget.
It starts when execution of an accepted item begins and includes:

- coordinated cooldown and startup-ramp waits;
- pre-execution middleware and strategy error callbacks;
- proactive rate-limiter waits;
- provider-capacity admission;
- every provider attempt;
- retry cooldowns and backoff.

Every wait is bounded by the remaining budget. A provider call receives the
lower of `attempt_timeout` and the remaining total budget, and no retry or new
provider attempt starts after expiry. Completed-attempt timing and token usage
remain on the terminal result.

Postprocessing and artifact persistence occur after the logical provider
execution pipeline and are not included in the total item deadline. They have
their own timeout/durability semantics. The queue wait before a worker picks up
an item is also outside the item deadline; use `batch_timeout` to bound the
whole run. In `LLMCallPool`, the pool semaphore wait is outside the executor's
item deadline; use `submit_timeout` when the caller needs one budget that also
includes pool admission.

Total expiry is terminal and non-retryable:

- exception: `ItemDeadlineExceeded`;
- `error_category`: `framework_total_item_timeout`.

A per-attempt `attempt_timeout` expiry is distinct: it is retryable, with
`error_category="framework_timeout"` and
`timing.timeout_category="framework_execution_timeout"`. Put `framework_timeout`,
not `framework_execution_timeout`, in `abort_on_error_categories`.

`timing.timeout_category` records where the time ran out (see
[`TimeoutCategory`](api/core.md#timeoutcategory) for the values). A deadline reached
while the item waited for provider capacity is `admission_timeout`, for both item
and batch deadlines. The elapsed capacity wait still counts toward
`admission_wait_seconds`, and it's kept when an abort ends the wait too.

## Batch deadlines

`batch_timeout` starts when `process_all()` or streaming execution starts, not
when a processor object is constructed. At expiry, source consumption stops,
no new provider attempt starts, and every already accepted item receives one
terminal result. Work not yet pulled from an async source was never accepted
and is not materialized.

Queued or interrupted collateral items use `BatchDeadlineExceeded` and
`error_category="batch_deadline_exceeded"`. `process_all()` and
`process_prompts()` return completed and collateral results with
`result.termination.kind == "batch_timeout"`. `process_stream()` yields all
terminal results for accepted work and then ends normally. Low-level streaming
callers can inspect `processor.termination` after completion.

The abort mode controls provider calls already in flight:

- `AbortMode.DRAIN_ACTIVE`: let a provider call already registered as active
  finish, but do not start another retry or provider call afterward.
- `AbortMode.CANCEL_ACTIVE`: cancel active provider calls and convert unfinished
  accepted work into batch-deadline results.

External caller cancellation is different: it still propagates after worker,
producer, queue, and strategy cleanup and is not mislabeled as a deadline.

## Configurable fail-fast

`abort_on_error_categories` is empty by default. Use `ErrorCategory` members for the
built-in categories (see [the full list](api/core.md#errorcategory)); plain strings,
including categories returned by a custom classifier, work too. A configured category trips
the shared abort controller only after an item reaches terminal failure; an
intermediate retryable attempt does not abort a batch that could still recover.
The first concurrent trigger wins and records its category and item ID.

Already completed results are preserved. Queued accepted items do not call the
provider and receive `BatchAbortedError` with
`error_category="batch_aborted"`. The active-call behavior follows the same
`abort_mode` described above. The returned batch reports
`termination.kind == "fail_fast"`.

`LLMCallPool` honours the same setting. The first request that finishes failed in
a configured category stops the pool: later submits, including those waiting for a
slot, return `batch_aborted` without calling the provider, and `pool.termination`
reports why. The pool stays stopped until you call `pool.resume()`, so your code
decides: fix the cause (re-authenticate, top up credit) and resume, or leave it
stopped and shut down. Calls that started before a resume can't stop the resumed
pool; a new failure in a configured category stops it again. `call()` and
`call_result()` run one request, so the setting has nothing to stop there.

```python
from llm_minder import ErrorCategory, GuardrailConfig, LLMCallPool, ProcessorConfig

config = ProcessorConfig(
    max_workers=5,
    guardrails=GuardrailConfig(
        abort_on_error_categories=frozenset({ErrorCategory.AUTHENTICATION}),
    ),
)
async with LLMCallPool(strategy, config=config) as pool:
    results = await asyncio.gather(*(pool.submit_result(p) for p in prompts))
    if pool.termination is not None:
        print(pool.termination.reason)  # names the item, category and hint
        if credentials_refreshed():  # your own check
            pool.resume()
```

Choose categories that indicate a batch-wide condition. Good candidates when
your provider classifier supplies reliable status information include:

- `ErrorCategory.AUTHENTICATION` (HTTP 401, or expired or missing Google
  credentials);
- `ErrorCategory.PERMISSION_DENIED` (HTTP 403), when permission is
  account/model-wide; and
- `ErrorCategory.INSUFFICIENT_BALANCE`.

Depending on your setup, these are also batch-wide:

- `ErrorCategory.QUOTA_EXHAUSTED` (Gemini daily quota);
- `ErrorCategory.TOKEN_ESTIMATOR_REQUIRED` and `ErrorCategory.QUOTA_SCOPE_ERROR`
  (configuration failures that affect every item).

Do not use `client_error` as a blanket default: malformed input or validation
can be item-specific. The provider-neutral classifier does not invent auth or
permission categories when reliable status data is unavailable.

## Token and cost budgets

`max_total_tokens` and `max_total_cost` cap what one processor run spends. Both
are opt-in soft caps.

Provisional: this API is new in v0.27 and may change in a minor release; see
[API stability](stability.md#provisional-in-10).

```python
from llm_minder import AttemptUsage, GuardrailConfig, ProcessorConfig

def price(usage: AttemptUsage) -> float:
    # Your prices, in your currency. llm-minder ships no price table.
    return (
        usage.usage.get("input_tokens", 0) * 0.15
        + usage.usage.get("output_tokens", 0) * 0.60
    ) / 1_000_000

config = ProcessorConfig(
    max_workers=20,
    guardrails=GuardrailConfig(
        max_total_tokens=5_000_000,
        max_total_cost=25.0,
        cost_function=price,
    ),
)
```

**What counts.** Every physical provider attempt whose usage the provider
reported, success or failure, including retries and rate-limited tries. Usage is
counted as each attempt finishes, from the same observation that tokens-per-minute
admission reconciles. These are not counted:

- replayed results;
- dry-run calls;
- attempts that never started a provider call;
- attempts whose usage the provider didn't report (they appear in
  `budget_unknown_usage_attempts`);
- usage a strategy adds only later, from `on_error`.

So the budget can be lower than a result's `token_usage`.

**Reaching a cap.** When the total reaches a cap (equality counts), the run
stops like a batch deadline, with `termination.kind == "budget_exceeded"`:

- no new provider call or retry starts;
- `abort_mode` decides whether calls already running drain or are cancelled;
- accepted items that didn't finish receive `BatchBudgetExceeded` with
  `error_category="batch_budget_exceeded"`.

The attempt that reached the cap keeps its outcome if it succeeded. If it failed,
its result becomes `batch_budget_exceeded`, with the provider error kept as the
exception's `__context__`.

**Overshoot.** Final usage can exceed the cap by the usage of the attempt that
reached it plus other attempts that had already started, which is at most
`max_workers` attempts in total. There is no numeric bound unless each attempt's
usage is bounded, for example with `max_tokens`. With `max_workers=1` and a cap of
10 tokens, a single call that reports 100 overshoots by 90. Providers may still
bill calls that `AbortMode.CANCEL_ACTIVE` cancels.

**Cost function.** `max_total_cost` requires `cost_function`, which is called once
per attempt with known usage. It receives an `AttemptUsage`: the item ID, attempt
and try numbers, the strategy, a read-only copy of the token usage, and whether the
call succeeded. Pass the `strategy` to price model escalation per attempt.

- It must be synchronous and fast: it runs on the event loop, and a blocking call
  can't be interrupted.
- It must return a finite, non-negative number.
- If it raises or returns anything else (including an awaitable or a bool), the
  run stops with `budget_exceeded`, tokens are still counted, and
  `budget_cost_complete` becomes `False`. Its exception message is not logged.
- A `cost_function` without `max_total_cost` only tracks cost.

**Reporting.** `await processor.get_stats()` adds `budget_tokens_used`,
`budget_cost_used`, `budget_cost_complete`, and `budget_unknown_usage_attempts`
when a budget or cost function is configured. The termination reason names the
cap and the amount used.

**Scope.** Budgets apply to processor runs (`process_all()`, streaming,
`process_prompts()`, `process_stream()`). `call()`, `call_result()`, and
`LLMCallPool` reject them with `ValueError` rather than ignoring a safety cap.
Every run starts with a fresh budget, including a resumed run.

**Artifacts.** `batch_budget_exceeded` results are audit records like other
batch-abort results: written best-effort and never replayed. A `BatchResult`
serialized with `termination.kind == "budget_exceeded"` can't be read by v0.26
or earlier.

## Provider call cap

`max_total_tokens` and `max_total_cost` are soft caps, because usage is only known
after a call finishes. When a run must not make more than a set number of calls,
for example a paid run approved for "at most 500 calls", use `max_provider_calls`:

```python
config = ProcessorConfig(guardrails=GuardrailConfig(max_provider_calls=500))
```

- Every physical provider attempt counts: first tries, retries, and rate-limit
  retries. Replayed results don't.
- The count is checked and taken before each call starts, in one step, so
  concurrent workers can't overshoot it. An attempt that is admitted but stopped
  before reaching the provider, for example by its item deadline, gives its call back.
- A run whose items all finish within the cap completes normally. Once the cap is
  used up, the next attempt that needs a call is refused, and the run stops exactly
  like a token or cost budget: `termination.kind == "budget_exceeded"`, a reason of
  `Provider call cap reached: 500 of 500 calls started`, and `batch_budget_exceeded`
  for the refused item and for unstarted items. `abort_mode` decides what happens
  to calls already running.
- `await processor.get_stats()` reports `provider_calls_started`.
- Like budgets, the cap applies to processor runs only; `call()`, `call_result()`
  and `LLMCallPool` reject it. It can be combined with `max_total_tokens` and
  `max_total_cost`; whichever trips first stops the run.

The cap counts calls the framework makes. An SDK that retries HTTP requests
internally can bill more than one request per call. The built-in OpenAI-compatible
models pass `max_retries=0` to the SDK by default. If you build the client yourself,
or use Gemini's own retry options, turn those retries off for a strict cap.

## Attempt admission hook

Provisional: new in 1.2; it may change in a minor release.

`max_total_cost` is a soft cap: usage is known only after a call finishes, so calls
already running can take a run past it. When a run must never pass a limit, for
example "this run may spend at most $20", reserve each call's worst case before it
starts and settle to its actual cost afterwards. `GuardrailConfig(admission=...)`
takes an object with two synchronous methods, `admit(request)` and
`settle(ticket, usage, success)` (the `AttemptAdmission` protocol):

- **`admit(request)`** runs immediately before each physical provider attempt
  (first tries, retries and rate-limit retries), after every other gate: provider
  capacity, cooldowns, RPM/TPM admission and the `max_provider_calls` check. A
  ticket is never held while an attempt waits. It returns a ticket, any object, or
  raises `AdmissionRefused` to refuse.
- **`settle(ticket, usage, success)`** is called exactly once for every ticket
  `admit` returned: after a success, a provider failure, a timeout, a cancellation
  or an abort, and also when the attempt was admitted but never reached the
  provider (for example its item deadline expired, or a cooldown began). `usage` is
  the attempt's `TokenUsage`, including usage attached to a failed attempt's
  exception. An attempt that never reached the provider settles with zero usage
  (`input_tokens`, `output_tokens` and `total_tokens` all 0), since nothing was
  sent. `None` means a call started but no usage is known, for example it timed out
  or was cancelled; a provider may still bill it.

The processor calls both methods on the event loop, outside its own locks, with no
await between `admit` and the start of the call. So a hook that checks and reserves
against its own in-memory state is atomic without a lock. That's why both must be
synchronous and must not block: `GuardrailConfig` rejects `async def` methods, and a
hook that needs I/O should reserve in memory and reconcile elsewhere.

**The request.** `AttemptRequest` carries `item_id`, `attempt` (the logical
attempt, as passed to `execute()`), `try_number` (the physical try; a rate-limit
retry keeps `attempt` and gets a new `try_number`), the `strategy`, and what the
call will send:

- `prompt` for a string item, or `messages` for a
  [message-list item](structured-prompts.md). Exactly one is set (`prompt` is `""`
  for a message-list item), so an estimator must read whichever one is.
- `generation_config`: for a `ModelStrategy`, the effective config for this attempt,
  with `attempt_generation_config()` merged over `generation_config`, so the output
  bound (`max_tokens`, `max_output_tokens`) is visible. `None` when the strategy has
  none, and for other strategies. Building it calls `attempt_generation_config()` a
  second time per attempt, so keep that method free of side effects.

The request doesn't include a system instruction or tools set on the model, so
leave a margin for them.

**Refusal.** `AdmissionRefused`, or any other exception from `admit`, stops the run
like a budget: no new provider call or retry starts, `abort_mode` decides what
happens to calls already running, and `termination.kind == "admission_refused"`
with the exception's message as the reason (another exception type is also logged,
and its reason names the type). The refused item and every accepted item that
hadn't finished end with `AdmissionRefused` and
`error_category="batch_admission_refused"`. A refused attempt isn't retried, isn't
classified as a provider error, and isn't counted in `provider_calls_started`.

**A failing `settle`.** An exception from `settle` is logged and stops the run the
same way, as `admission_refused`: a spend cap that can't record spend must not keep
admitting. The attempt that was being settled keeps its own outcome.

**Scope and artifacts.** Like budgets, the hook applies to processor runs only;
`call()`, `call_result()` and `LLMCallPool` reject `GuardrailConfig(admission=...)`
with `ValueError`. `batch_admission_refused` results are audit records like other
batch-abort results, in both artifact stores: written best-effort and never
replayed, so a resumed run executes those items again. A `BatchResult` serialized
with `termination.kind == "admission_refused"` can't be read by 1.1 or earlier.

**SDK retries.** The hook sees the attempts the framework makes. An SDK that
retries HTTP requests inside one attempt can bill several requests for one ticket.
The built-in OpenAI-compatible and Anthropic models build their clients with
`max_retries=0`. With `admission` set, the processor logs a warning, once per model,
when a built-in model's client retries on its own: a `max_retries` override, a
client you passed in, or Gemini `retry_options`.

### Example: a hard spend cap

The cap below reserves `input_estimate * input_price + max_output * output_price`
at `admit`, refuses when committed plus reserved spend would pass the limit, and
settles each ticket to the attempt's actual cost. The input estimate counts UTF-8
bytes, which can't be fewer than the tokens a byte-level tokenizer produces, so it
is high (about four times, for English) but safe. For Anthropic prompt caching the
worst case prices every input token as a cache write (`cache_write_rate` 1.25 for
the 5-minute TTL, 2.0 for the 1-hour TTL), and the settled cost charges cache reads
at the discounted rate. The built-in models report `input_tokens` including cache
reads and writes, which is what `actual_cost` assumes.

```python
import json
import math
from dataclasses import dataclass

from llm_minder import (
    AdmissionRefused,
    AttemptRequest,
    CachedTokenRates,
    GuardrailConfig,
    ProcessorConfig,
    TokenUsage,
)


@dataclass(frozen=True)
class Prices:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_rate: float = 1.0  # fraction of the input price for a cache read
    cache_write_rate: float = 1.0  # multiple of the input price for a cache write


class HardSpendCap:
    """Committed plus reserved spend never passes ``limit``."""

    def __init__(
        self,
        limit: float,
        prices: Prices,
        *,
        default_max_output: int,
        overhead_tokens: int = 2_000,
    ) -> None:
        self.limit = limit
        self.prices = prices
        self.default_max_output = default_max_output  # when the config sets no bound
        self.overhead_tokens = overhead_tokens  # system instruction, tools, formatting
        self.spent = 0.0  # settled cost
        self.reserved = 0.0  # worst case of admitted attempts not yet settled

    def input_estimate(self, request: AttemptRequest) -> int:
        if request.messages is None:
            text = request.prompt
        else:
            text = json.dumps([dict(message) for message in request.messages], default=str)
        return len(text.encode("utf-8")) + self.overhead_tokens

    def max_output(self, request: AttemptRequest) -> int:
        # Providers differ in which alias wins when several are set, so take the
        # largest. A bound that isn't a positive int can't be trusted to apply,
        # so it brings back the default rather than shrinking the reservation.
        config = request.generation_config or {}
        keys = ("max_tokens", "max_output_tokens", "max_completion_tokens")
        bounds: list[int] = []
        for key in (k for k in keys if k in config):
            value = config[key]
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
            bounds.append(value if valid else self.default_max_output)
        return max(bounds, default=self.default_max_output)

    def admit(self, request: AttemptRequest) -> float:
        p = self.prices
        input_price = p.input_per_mtok * max(1.0, p.cache_write_rate)
        worst = (
            self.input_estimate(request) * input_price
            + self.max_output(request) * p.output_per_mtok
        ) / 1_000_000
        if self.spent + self.reserved + worst > self.limit:
            raise AdmissionRefused(
                f"Spend cap ${self.limit:.2f}: ${self.spent:.4f} spent and "
                f"${self.reserved:.4f} reserved; the next attempt could cost ${worst:.4f}"
            )
        self.reserved += worst
        return worst  # the ticket

    def settle(self, ticket: object, usage: TokenUsage | None, success: bool) -> None:
        assert isinstance(ticket, float)
        self.reserved -= ticket
        # Unknown usage may still have been billed (a timeout or cancellation),
        # so a hard cap keeps the worst case.
        self.spent += ticket if usage is None else self.actual_cost(usage)

    def actual_cost(self, usage: TokenUsage) -> float:
        p = self.prices
        read = usage.get("cached_input_tokens", 0)
        written = usage.get("cache_write_input_tokens", 0)
        plain = max(0, usage.get("input_tokens", 0) - read - written)
        input_tokens = plain + read * p.cache_read_rate + written * p.cache_write_rate
        output_cost = usage.get("output_tokens", 0) * p.output_per_mtok
        return (input_tokens * p.input_per_mtok + output_cost) / 1_000_000


cap = HardSpendCap(
    20.0,
    # Placeholder prices per million tokens: use your model's price list.
    Prices(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_read_rate=CachedTokenRates.ANTHROPIC_READ,
        cache_write_rate=CachedTokenRates.ANTHROPIC_WRITE,  # ANTHROPIC_WRITE_1H for 1-hour
    ),
    default_max_output=16_000,  # AnthropicModel's default max_tokens
)
config = ProcessorConfig(guardrails=GuardrailConfig(admission=cap))
```

After the run, `cap.spent` is the settled spend. Keeping the worst case for
`usage=None` makes the cap strict, at the price of charging started calls that
failed before billing (a refused connection, a 429 or 5xx response with no usage) at
their worst case. Attempts that never started settle with zero usage, so they
release their reservation exactly. Release the reservation for `None` instead if you'd rather
under-count those than over-count them. Without caching, leave both cache rates
at 1.0.

## Cache hit-rate floor

A prompt cache that stops hitting can multiply a run's input cost without any call
failing. Every processor run counts cache reads and writes per *cache group*, and
`min_cache_hit_rate` stops the run when a group's recent hit rate falls below a floor.
This section is the reference; [Prompt caching](prompt-caching.md) is the how-to,
including which providers support what and how to choose the values.

Provisional: new in 0.30; it may change in a minor release.

```python
from llm_minder import GuardrailConfig, ProcessorConfig

config = ProcessorConfig(
    max_workers=20,
    guardrails=GuardrailConfig(
        min_cache_hit_rate=0.8,  # None (the default) turns the floor off
        cache_hit_window=50,     # measured calls per group in the sliding window
    ),
)
```

**Groups.** An attempt's group is the `cache_group` key in its metadata:

- Every built-in model takes a `cache_group="..."` label, and a call can override
  it with `config={"cache_group": "..."}`. The label is never sent to the provider,
  and it wins over the defaults below.
- OpenAI-compatible models (OpenAI, OpenRouter, DeepSeek) use the request's
  `prompt_cache_key` if one is set (including one from the model's
  `prompt_cache_key` option, see
  [OpenAI prompt caching](OPENAI_INTEGRATION.md#getting-cache-hits)), else `si:`
  plus a 12-character hash of the system instruction.
- `GeminiModel` uses the system-instruction hash. `GeminiCachedModel` uses
  `gemini-cache:` plus a hash of the cached content, which stays the same when the
  cache is renewed.
- A `cache_group` key from a `metadata_extractors` entry wins over all of these.
- `PydanticAIStrategy` sets no group. `CallableStrategy` and custom strategies can
  set `metadata["cache_group"]` themselves.
- Attempts without a group are counted together as the `None` group.

A group longer than 128 characters is replaced by its SHA-256. A run tracks at most
64 groups; attempts for further groups are counted together in
`cache_groups_overflow`, with one warning.

**What counts.** Provider attempts in this run that returned a result. Failed
attempts, replayed results and dry-run calls aren't counted. An attempt is
*measured* when its usage is known and reports positive input tokens; others are
counted as `unmeasured_calls` and never move the rate.

**The floor.** Each group keeps a window of its last `cache_hit_window` measured
attempts. Once a group's window is full, a window hit rate (cached input tokens
over input tokens) below `min_cache_hit_rate` stops the run like
[configurable fail-fast](#configurable-fail-fast): `termination.kind ==
"fail_fast"`, `termination.error_category == "cache_hit_rate_below_floor"`,
`abort_mode` applies, and items that hadn't started end `batch_aborted`. The window
is also the warm-up, and each group has its own, so a group's first cache writes
aren't held against it. With a floor configured, a 65th group also stops the run,
since the floor can't be enforced for groups it doesn't track. If the same attempt
reaches a token or cost budget, the run ends `budget_exceeded` instead.

**Miss reasons.** A strategy can put a provider's explanation of a miss in
`metadata["cache_miss_reason"]` (a string, kept to 64 characters). Each group
reports its latest one as `last_miss_reason`, and the floor's stop reason ends with
it. OpenAI's Responses API models set it from `prompt_cache_diagnostics` when the
response carries a cache-miss diagnosis.

**Reporting.** `await processor.get_stats()` and the `BATCH_COMPLETED` event carry
`cache_groups`, a list sorted by input tokens, and `cache_groups_overflow` (`None`
unless there was overflow). Each entry has `group`, `measured_calls`,
`unmeasured_calls`, `input_tokens`, `cached_input_tokens`,
`cache_write_input_tokens`, `hit_rate` (cumulative), `window_hit_rate` (over the last
`cache_hit_window` measured calls, `None` until the window is full) and
`last_miss_reason`.
`BatchResult` has no per-group view: results include replays and add up retries,
which isn't what the floor measures.

**Scope.** Processor runs only. `call()`, `call_result()` and `LLMCallPool` reject
`min_cache_hit_rate` with `ValueError`. A resumed run starts with fresh counts.

**Call pools.** `LLMCallPool` keeps the same per-group statistics without a floor.
`pool.cache_stats()` returns `cache_groups` and `cache_groups_overflow` in the same
shape, counted since the pool was created or last reset, and every group includes
`window_hit_rate` over its last `cache_hit_window` measured calls. A long-lived
pool's cumulative `hit_rate` moves slowly, so the window shows a recent drop.
`pool.reset_cache_stats()` clears everything, including the 64-group cap; a call
still running at the reset is counted when it returns. Both methods are synchronous,
so call them from the pool's event loop. `call()` and `call_result()` keep no
statistics.

**Your own tracker.** `CacheGroupTracker` (provisional) is the accounting behind all
of this, with the same rule for what counts and the same window check as the floor.
`pool.cache_tracker` is the pool's live one. Use it to carry a pool's statistics
across a restart, or to apply a floor where the processor can't:

```python
import json

from llm_minder import CacheGroupTracker, LLMCallPool

# On startup: continue the saved statistics, if any.
tracker = CacheGroupTracker.from_dict(json.loads(saved)) if saved else None
pool = LLMCallPool(strategy, config=config, cache_tracker=tracker)

# Between calls: apply a floor yourself.
for breach in pool.cache_tracker.below_floor(0.8):
    logger.warning(breach.message)

# At a checkpoint or on shutdown: save the exact state.
saved = json.dumps(pool.cache_tracker.to_dict())
```

- `record(group, usage, miss_reason=None)` counts one call; `record_result(result)`
  counts a `WorkItemResult` from its metadata and `token_usage`, skipping failures.
- `stats()` returns `cache_groups` and `cache_groups_overflow` in the shape above.
- `below_floor(floor)` returns a `CacheFloorBreach` for each group whose full window
  is below `floor` (lowest first) and one with `overflow=True` if more than 64 groups
  were seen. Each breach's `message` is the stop reason a processor run would report.
- `to_dict()` returns JSON-serializable state, every group's window included, and
  `CacheGroupTracker.from_dict()` restores it exactly (it raises `ValueError` on bad
  data). A pool given a restored tracker uses the tracker's `window`, not
  `cache_hit_window`.

Without saved state, you can rebuild a tracker by feeding it saved results with
`record_result`. It reads lower than the live tracker did when retries were billed:
a result's `token_usage` adds up every attempt of its item, failed ones included,
whereas live tracking counts only the attempt that returned. A processor run always
starts with a fresh tracker; it doesn't take a saved one.

## End-to-end checkpointed run

```python
from pathlib import Path
from llm_minder import (
    AbortMode,
    ArtifactIdentity,
    ErrorCategory,
    GuardrailConfig,
    JsonlArtifactStore,
    ProcessorConfig,
    ResumePolicy,
    process_prompts,
)

store = JsonlArtifactStore(
    "run.jsonl",
    identity=ArtifactIdentity(
        provider="openai",
        model="example-model",
        prompt_version="invoice-v4",
        parser_version="invoice-schema-v2",
        application_version="billing-pipeline-v7",
    ),
)

config = ProcessorConfig(
    max_workers=20,
    attempt_timeout=30,
    guardrails=GuardrailConfig(
        total_timeout_per_item=180,
        batch_timeout=3600,
        abort_on_error_categories=frozenset({
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.INSUFFICIENT_BALANCE,
        }),

        abort_mode=AbortMode.DRAIN_ACTIVE,
    ),
)

result = await process_prompts(
    strategy,
    prompts,
    config=config,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
    preserve_order=True,
)

Path("summary.json").write_text(result.to_json(), encoding="utf-8")
```

Serialization, artifact I/O, and programming failures remain exceptions. They
are not disguised as controlled guardrail termination.

## Retry delay cannot fit

If the actual jittered retry delay is at least the remaining item deadline,
llm-minder fails immediately with `ItemDeadlineExceeded` instead of spending that time
waiting. Its message and cause retain the last provider error; the attempt timing
keeps that error's category and reports zero retry-backoff wait. The final result
uses `framework_total_item_timeout`, so `abort_on_error_categories` matches that
final category rather than the provider category. The existing guardrail audit
and non-replayable-record behavior remains in effect.
