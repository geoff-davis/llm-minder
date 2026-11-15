# Choosing Your Limits

One page, one pass through every limit that matters, in the order you should
decide them. Each section says what the knob bounds, how to pick a value, and
when to leave it alone. The deep dives — the
[production checklist](production-checklist.md),
[high-throughput guide](openai-high-throughput.md),
[guardrails](guardrails.md), and
[bounded work](bounded-work.md) — expand on each step; you should be able to
size a real run without leaving this page.

## The short version

```python
from llm_minder import JsonlArtifactStore, ResumePolicy, llm, process_prompts

strategy = llm("deepseek:deepseek-v4-flash", thinking=False)

batch = await process_prompts(
    strategy,
    prompts,                       # 10,000 items
    concurrency=64,                # step 1 — sizes everything aligned below
    progress=True,
    artifact_store=JsonlArtifactStore("runs/tagging.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

`concurrency=N` (v0.20) coherently sizes the worker pool, provider-capacity
admission, and — for built-in OpenAI-compatible models (OpenAI, OpenRouter,
DeepSeek, `openai-compatible`) created via `llm()` or `from_api_key()` without
an explicit `max_connections` — the httpx connection pool. If you set
nothing else, the rest of this page is the explanation of what you just got
and when to override it.

## Check what you got

`describe_limits()` lists every limit with its value and where the value came
from (explicit, derived from `concurrency`, a default, or advertised by the
strategy), plus the provider capacity and the number of attempts that can run at
once for each strategy. The processor logs the same table at INFO when it first
configures a strategy, and `LLMCallPool` logs it when created.

```python
from llm_minder import ProcessorConfig, describe_limits

config = ProcessorConfig(concurrency=64, max_requests_per_minute=3_000)
print(describe_limits(config, strategy))   # works for call() configs too
# processor.describe_limits() and pool.describe_limits() fill in their own config
```

After a run, `batch.gate_waits()` adds up where attempts waited: provider
capacity, quota (RPM and TPM together), rate-limit cooldown, the startup ramp,
and retry backoff, largest first. For results from `call_result()` or
`LLMCallPool`, pass them to `summarize_gate_waits()`. Time an item spends queued
before a worker picks it up isn't recorded, so the worker pool is not a row.

```python
batch = await process_prompts(strategy, prompts, config=config)
print(batch.gate_waits())
```

Both are provisional (new in 1.1).

## When you don't know your limits

Everything below assumes you know your provider's quota. Often you don't: a new
Gemini project, a tier that changed, or a key shared with other applications.
Setting concurrency very low is safe but leaves throughput unused. This section
shows how to find the limit with one short probing run.

**Look the numbers up first if you can.** It's faster than probing.

- Gemini lists RPM, TPM and requests-per-day for each model and tier in
  [AI Studio](https://aistudio.google.com/) (the project's rate-limit page) and in
  the published [rate-limit table](https://ai.google.dev/gemini-api/docs/rate-limits).
- OpenAI shows your organization's limits in the platform settings, and responses
  can include them in the `x-ratelimit-limit-requests` and
  `x-ratelimit-limit-tokens` headers.

Probe when the console isn't available to you, or when the published numbers
don't match what you see.

### Step 1: configure a probing run

Set `concurrency` to the ceiling you hope for, and let a startup ramp climb to it
slowly. The load then rises gradually instead of arriving in one burst, so the
requests in the minute before the first rate limit are close to what the provider
allows. Shorten the cooldown so touching the limit costs seconds rather than the
default five minutes.

```python
from llm_minder import (
    ProcessorConfig,
    RateLimitConfig,
    StartupRampConfig,
    describe_limits,
)

probe_config = ProcessorConfig(
    concurrency=32,  # the ceiling you hope for
    startup_ramp=StartupRampConfig(
        initial_concurrency=2,
        concurrency_step=2,
        ramp_interval_seconds=15,
    ),
    rate_limit=RateLimitConfig(cooldown_seconds=30, max_cooldown_seconds=120),
)
print(describe_limits(probe_config, strategy))
```

With this ramp, `t` seconds after the strategy's first attempt is admitted, at most
`min(32, 2 + 2 × floor(t / 15))` attempts run at once. It reaches 32 after 225
seconds, so give the probe enough prompts to keep it busy for longer than that:
at about two seconds per call, a few thousand short prompts. Use prompts like the
real ones, since token counts and latency both change where the limit is
reached. The ramp clock keeps running through cooldowns; after a cooldown, the
slow-start ramp from `RateLimitConfig` paces the restart instead.

Check the `describe_limits()` table before spending quota. It should show
`concurrency` 32 (explicit), `startup_ramp.max_concurrency` 32 (from
`max_workers`), no RPM or TPM limit, and a provider capacity of 32 for the
strategy. A strategy that advertises a smaller `max_concurrency` shows that as
its provider capacity instead, and the probe can't go wider than it. If the table
was printed before the batch started, a header may say it is not yet in effect:
a built-in OpenAI-compatible model's connection pool is resized to `concurrency`
when the batch starts.

**The cooldown trade-off.** A short cooldown makes each rate limit cheap, which
is what a probe needs: traffic resumes quickly and the run keeps measuring.
Against a provider that counts per minute, a short cooldown in production instead
sends traffic back into a window that is still full, so it trips again. Repeated
cooldowns grow by `backoff_multiplier` (1.5) up to `max_cooldown_seconds`. A
retry delay the server sends (Gemini's `retryDelay`, an OpenAI-compatible
`Retry-After`) can only lengthen a cooldown, never shorten it, and is itself
capped at `max_cooldown_seconds`.

### Step 2: run it and record the first rate limit

`get_stats()` counts rate limits but doesn't say when the first one happened, or
how many requests went out in the minute before it. A small observer records
both: `ITEM_ADMITTED` fires for every attempt just before it calls the provider.

```python
import time

from llm_minder import (
    BaseObserver,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessingEvent,
)


class FirstRateLimit(BaseObserver):
    """Record when each attempt started and when the first rate limit hit."""

    def __init__(self) -> None:
        self.attempt_starts: list[float] = []
        self.first_rate_limit: float | None = None

    async def on_event(self, event, data):
        now = time.monotonic()
        if event is ProcessingEvent.ITEM_ADMITTED:
            self.attempt_starts.append(now)
        elif event is ProcessingEvent.RATE_LIMIT_HIT and self.first_rate_limit is None:
            self.first_rate_limit = now


probe = FirstRateLimit()
async with ParallelBatchProcessor(config=probe_config, observers=[probe]) as processor:
    for i, prompt in enumerate(sample_prompts):
        await processor.add_work(LLMWorkItem(item_id=f"probe-{i}", strategy=strategy, prompt=prompt))
    batch = await processor.process_all()
    stats = await processor.get_stats()
```

### Step 3: read the result

Start with where attempts waited:

```python
print(batch.gate_waits())
print("rate limits:", stats["rate_limit_count"])
```

- **A large rate-limit cooldown share** means the provider pushed back. Read on
  to find where.
- **A large startup-ramp share and no cooldown** means the run never reached the
  limit. Either the limit is above your ceiling, or the probe ended before the
  ramp finished. Repeat with a higher `concurrency` or more prompts.
- **Items failing with `quota_exhausted`** mean a per-day quota ran out. That
  isn't a per-minute limit and waiting won't fix it; see the caveats below.

The waits of items running at the same time add up, so the totals can exceed the
run's wall time; compare the shares, not the totals with wall time. Time an item
spends queued for a worker isn't counted. The ramp's wait happens at the
provider-capacity gate, so in a probe those two rows are often nearly the same
number. With no rate limit, the cooldown row can still show a fraction of a
second: that's the time spent checking for a cooldown under load, not a cooldown.

Then count requests per minute. If the provider pushed back, count the minute
before the first rate limit. If it never did, the busiest minute is the highest
throughput you observed. Provided those requests succeeded, it's a rate the
provider accepted, so the limit is at least that high:

```python
import statistics
from bisect import bisect_left, bisect_right

starts = probe.attempt_starts  # in time order


def requests_in_minute_ending(t: float) -> int:
    return bisect_right(starts, t) - bisect_left(starts, t - 60)


if probe.first_rate_limit is not None:
    # Roughly the rate at which the provider said no.
    requests_per_minute = requests_in_minute_ending(probe.first_rate_limit)
else:
    # No rate limit: the limit is above this. Repeat wider to find it.
    requests_per_minute = max(map(requests_in_minute_ending, starts), default=0)

latencies = [
    attempt.execution_seconds
    for result in batch.results
    for attempt in result.timing.attempts
    if attempt.success
]
if latencies:
    latency = statistics.median(latencies)
    tokens_per_request = (batch.total_input_tokens + batch.total_output_tokens) / batch.total_items
    tokens_per_minute = requests_per_minute * tokens_per_request
else:
    print("No attempt succeeded; read the failures before probing again:")
    print(batch.summary())
```

Count the requests rather than working the rate out from the ramp's width at the
time of the first 429. A provider that counts per minute trips on the requests
of the whole past minute. While the ramp climbs, the rate at the moment it trips
is higher than that minute's average, so an estimate from the width overshoots.

The count is an estimate too, and it tends to come out a little high. The
observer timestamps attempts when this process admits them and the rate limit
when its response arrives, not when the provider decided. The count includes the
rejected attempt itself, and any attempts admitted while its response was on the
way. With high concurrency and slow responses, that can be many requests the
provider never accepted. Retries made inside a provider SDK are requests the
observer never sees. In a simulated provider with a sliding 700-request minute,
the count came to 714 and the estimate from the width to about 890.

Check which tokens your provider counts towards its TPM limit (input only, or
input and output) and use those in `tokens_per_request`. For per-attempt
detail, each `AttemptTiming` has `cooldown_wait_seconds`, `quota_wait_seconds`
and `startup_ramp_wait_seconds`.

### Step 4: pin per-minute limits below what you saw

Set `max_requests_per_minute`, and `max_tokens_per_minute` if tokens were the
binding limit, about 20% below the counted rate, with `quota_burst_seconds=5`
so the first minute doesn't admit a double helping (see step 5 of the decision
tree below). Restore a normal cooldown, and size `concurrency` with the formula
from step 1 of the decision tree.

The margin is a starting point, not a guarantee: the count is approximate, other
applications may share the quota, and providers change limits. Watch
`get_stats()["rate_limit_count"]` in the first real runs and lower the limits if
cooldowns appear. If the probe never hit a limit and its requests succeeded,
pinning below the busiest minute keeps you under a rate the provider accepted,
but may leave throughput unused. A provider SDK that retries 429s internally can
hide throttling from the probe. The built-in OpenAI-compatible models already
pass `max_retries=0` when they build the client; for a client you build yourself,
turn SDK retries off for the probe.

Pin the per-minute limits rather than the concurrency at which the probe tripped.
The provider counts requests and tokens per minute, not connections, so when
latency changes, the width that reaches the limit changes with it, while the
per-minute numbers stay put.

### Worked example

Measured on 2026-11-15 with `gemini-3.5-flash-lite` through an AI Studio API
key. The 30,000 prompts were one sentence each ("In one short sentence, name a
fact about the number 17."). The probe used a wider version of the step 1
config, `concurrency=128` and `StartupRampConfig(initial_concurrency=8,
concurrency_step=8, ramp_interval_seconds=15)`, which reaches 128 after 225
seconds.

```text
Gate waits across 30000 item(s), 36126.04s of item time
  provider capacity             14341.82s   39.7%  20269 attempt(s)
  startup ramp and slow start   14302.46s   39.6%  20162 attempt(s)
  rate-limit cooldown              16.24s    0.0%  1623 attempt(s)
  quota (RPM/TPM)                   0.00s    0.0%  0 attempt(s)
  retry backoff                     0.00s    0.0%  0 attempt(s)
```

All 30,000 succeeded in 283 seconds with no rate limit. The busiest minute held
10,340 requests, the median call took 0.68 seconds, and requests averaged 50
tokens. The ramp and capacity rows are the same wait, and the cooldown row is
the cost of checking for a cooldown, not a cooldown. So the run never reached the
limit. For this key and model the limit is above about 10,000 requests per minute,
and finding it would take a wider probe.

A wider probe the same day (`concurrency=512`, ramp 32/32/15) peaked at 18,179
requests in a minute, still without a rate limit, before a 900-second batch
timeout ended it. At that width the probing process became the bottleneck: the
median call rose to 0.99 seconds and only about 300 requests were in flight on
average. One process can show that a limit is at least this high, not where it
is.

A rate the provider accepted is still a usable setting:

```python
from llm_minder import ProcessorConfig, RateLimitConfig

config = ProcessorConfig(
    concurrency=90,                  # 8,000 / 60 × 0.68 ≈ 90
    max_requests_per_minute=8_000,   # about 20% below the 10,340 accepted
    quota_burst_seconds=5,
    rate_limit=RateLimitConfig(cooldown_seconds=60.0),
)
```

The probe's prompts were tiny, so it says nothing about the TPM limit. Real
prompts of 1,000 tokens at 8,000 requests per minute come to 8 million tokens per
minute; check that against the published table, and set `max_tokens_per_minute`
with a token estimator if it's the tighter limit.

### Caveats

- **Per-day quotas are terminal.** Gemini reports an exhausted daily quota as a
  429 too, but the classifier marks it `quota_exhausted` and doesn't retry it.
  No per-minute setting helps; the run has to wait for the quota to reset.
- **Limits are per model and per project.** A probe against Flash-Lite says
  nothing about Pro, and two keys in one project share a budget. Probe each model
  you use, and leave room for other applications on the same project.
- **The probe spends real quota.** Keep the sample small and run it at a quiet
  time, or it competes with production traffic and measures less than you have.

## The decision tree

```text
1. concurrency      how many requests in flight at once?
2. connection pool  can the HTTP client actually carry that many?
3. provider concurrency  how many calls may hold provider capacity?
4. quota scope      which strategies spend the same upstream budget?
5. RPM              how many physical attempts may start per minute?
6. TPM + estimator  how much estimated token load may start per minute?
7. startup ramp     should full concurrency arrive gradually?
8. cooldown         what happens when the provider says 429?
9. retries          how often, and how far apart, are failed attempts retried?
10. timeouts        how long may an attempt, item, and batch take?
11. spend caps      how many tokens or how much cost may the run use?
```

### 1. Concurrency — `concurrency=N`

Pick the number of simultaneous provider requests. This is a property of the
**provider**, not your CPU — LLM calls are I/O-bound, so never use
`cpu_count()`.

- Rate-limited endpoints (OpenAI/Gemini tiers): start at **5–10**.
- High-concurrency providers (DeepSeek, self-hosted vLLM): **50–200**.
- If the provider publishes an RPM limit, a serviceable estimate is
  `concurrency ≈ RPM / 60 × typical_latency_seconds`, rounded down.

Prefer the single knob over setting `max_workers`,
`max_provider_concurrency`, and `max_connections` separately — misalignment
between those three is the most common performance bug in this package.
Explicit values for any individual knob still win when you set them.

### 2. Connection pool — `max_connections`

The openai SDK's default httpx pool allows **1000 connections but keeps only
100 alive** between requests. Above 100 concurrent requests, the extra
connections are opened and torn down for each request, so you pay a fresh
TCP/TLS handshake per call. Above 1000, extra workers wait inside httpx for a
connection; that wait counts against `attempt_timeout` but isn't reported as a
capacity wait, so it caps throughput without showing up in admission timing.
High-concurrency
providers such as DeepSeek accept hundreds of concurrent requests, so they are
where an unsized pool shows up first.

- With `concurrency=N` and a factory-built OpenAI-compatible model
  (`llm("...")` or `from_api_key()` without `max_connections`), the pool is
  resized to `N` connections, all kept alive, before the first request —
  nothing to do. Gemini models have no pool resize.
- If you build your own `AsyncOpenAI`/httpx client, size its
  `httpx.Limits(max_connections=..., max_keepalive_connections=...)` to at
  least your concurrency yourself; the framework cannot introspect or resize
  a caller-supplied client.
- An explicit `max_connections` smaller than `concurrency` is treated as a
  real contradiction and warns.

Details: [high-throughput guide](openai-high-throughput.md).

### 3. Provider concurrency — `max_provider_concurrency`

Admission caps how many attempts may hold a provider "slot" at once, and the
wait happens **before** `attempt_timeout` starts — so a queued attempt can't
burn its execution timeout waiting for capacity. With `concurrency=N` this is
already `N`; set it explicitly only to run the framework wider than the
provider (e.g. many cheap local workers feeding a narrow paid API), or when a
strategy advertises its own `max_concurrency` (the lower limit applies).

### 4. Quota scope — `strategy.quota_scope`

Identify which strategies spend the same provider/account RPM and TPM budget.
The default follows `concurrency_scope`. Share one stable object when multiple
models use one account quota; use distinct objects only for genuinely
independent budgets. Quota scope also owns coordinated cooldown, while
concurrency scope owns provider/client capacity. See
[Token-Aware Admission](token-aware-admission.md#quota-scopes).

### 5. Requests per minute — `max_requests_per_minute`

Set the provider/account's physical-attempt budget, with safety margin for
other applications using the same account. Admission is FIFO within each quota
scope. Every retry is another physical request and gets a fresh reservation. A
429 still triggers reactive coordinated cooldown; RPM smoothing cannot
reproduce every provider window exactly.

The default `quota_burst_seconds=None` starts with a full minute of quota and
refills continuously: a busy scope can admit about **twice the configured limit**
in its first minute. Set `quota_burst_seconds=5` to start with five seconds of
quota. For RPM limit `L`, capacity is `C=max(1, L*5/60)`, and a rolling minute
admits at most `C+L` requests. When the one-request floor does not apply this is
`L*65/60`. It is a token bucket, not a strict provider rolling-window limiter.

Limits apply independently to each quota scope within each processor/pool.
Separate strategy instances can create separate budgets. The processor warns
once when distinct scopes exceed `max_workers` with RPM or TPM enabled; reuse a
strategy/model or return one stable shared `quota_scope` to share limits.

### 6. Tokens per minute — `max_tokens_per_minute`

TPM requires an estimator. Reserve estimated input plus expected output, not
only prompt tokens:

```python
from llm_minder import CharacterTokenEstimator, ProcessorConfig

config = ProcessorConfig(
    concurrency=16,
    max_requests_per_minute=500,
    max_tokens_per_minute=200_000,
    token_estimator=CharacterTokenEstimator(expected_output_tokens=300),
)
```

Prefer a provider tokenizer over the character heuristic. Review actual
refunds, underestimation debt, and unknown-usage attempts, then tune the output
allowance. One estimate must fit inside the configured per-minute token limit. A request
larger than the burst bucket may borrow future refill when that bucket is full. Increasing
workers cannot overcome RPM/TPM and can increase task and queue pressure while
work waits.

### 7. Startup ramp — `startup_ramp`

Opening at full concurrency against a cold endpoint can trip instant 429s.
`StartupRampConfig(initial_concurrency=4, concurrency_step=4,
ramp_interval_seconds=2.0)` walks up to full width instead. Skip it for small
runs or generous providers; reach for it when the first minute of a large run
keeps tripping cooldowns.

### 8. Cooldown — `RateLimitConfig`

What happens on 429: one worker triggers a shared cooldown, everyone pauses,
then traffic resumes through a slow-start ramp. Defaults are conservative
(300s cooldown). For providers with short quota windows, start with
`RateLimitConfig(cooldown_seconds=30.0, max_cooldown_seconds=300.0)` — the
classifier honors a server-sent `Retry-After` as a floor regardless. If you
see repeated consecutive cooldowns, your `concurrency` (step 1) is too high;
fix the cause, not the cooldown. Cooldown precedes estimation and reservation,
so paused work does not consume live quota state early.

### 9. Retries and backoff — `RetryConfig`

`RetryConfig` governs failures that are worth retrying but aren't rate limits:
timeouts, connection errors, 5xx responses, and invalid output. The defaults are
`max_attempts=3`, `initial_wait=1.0`, `max_wait=60.0`, `exponential_base=2.0`,
and `jitter=True`. Before attempt *n + 1* the item waits

```text
min(initial_wait × exponential_base^(n − 1), max_wait)
```

so 1s, 2s, 4s, and so on, capped at 60s. With `jitter=True` each wait is scaled
to a random 50–100% of that value, which spreads simultaneous retries apart
("equal jitter"; `jitter="equal"` is the same). `jitter="full"` scales it to a
random 0–100% instead. That spreads retries further, which helps against a
flaky gateway that many items hit at once, but some retries go out almost
immediately. `jitter=False` waits the exact value. Validation failures (`validation_error`,
`structured_output_validation_error`) retry immediately, with no wait.

Rate limits work differently. They don't use this backoff: one rate-limited
call starts the shared cooldown from step 8, and the item retries when it ends.
They also don't consume `max_attempts`; they have their own budget,
`max_rate_limit_retries` (default 20). Errors the classifier marks
non-retryable, such as authentication failures, fail on the first attempt.

Retries and backoff waits count against `GuardrailConfig.total_timeout_per_item`.
If the next wait wouldn't fit in the time left, the item fails immediately with
`ItemDeadlineExceeded`, whose message and cause keep the last provider error,
instead of sleeping past its deadline (see
[guardrails](guardrails.md#retry-delay-cannot-fit)).

```python
from llm_minder import ProcessorConfig, RetryConfig

config = ProcessorConfig(
    concurrency=32,
    retry=RetryConfig(max_attempts=4, initial_wait=0.5, max_wait=20.0),
)
```

### 10. Attempt, item, and batch timeouts

`attempt_timeout` bounds one `execute()` call (default 120s), not its quota or
provider-capacity wait. Size it for one slow response: p99 provider latency
plus margin. Long-output or reasoning models may need more than the common 30s
starting point.

`GuardrailConfig.total_timeout_per_item` bounds one logical item end to end:
cooldown, estimation, RPM/TPM wait, provider admission, every call, and retry
backoff. `GuardrailConfig.batch_timeout` bounds the whole run. On expiry,
accepted items receive terminal results and checkpoints clean up according to
the configured abort mode. Size the batch deadline below the job scheduler's
hard kill and pair it with fail-fast categories such as authentication and
insufficient balance. Details: [guardrails](guardrails.md).

### 11. Spend caps — `max_total_tokens` / `max_total_cost`

`GuardrailConfig.max_total_tokens` and `max_total_cost` (with a
`cost_function`) are opt-in soft caps on provider usage for one processor run.
Reaching a cap stops new provider calls; calls already running can overshoot
it, by at most the usage of up to `max_workers` in-flight attempts. Pair a cap
with a per-call `max_tokens` so the overshoot has a numeric bound. The
budget API is provisional and may change in a minor release (see
[API stability](stability.md#provisional-in-10)). Details:
[token and cost budgets](guardrails.md#token-and-cost-budgets).

## Worked example: 10k items against a rate-limited provider

Target: 10,000 classification prompts against an OpenAI-tier endpoint
(~500 RPM), ~2s per call, overnight job, must survive restarts.

- **Step 1:** `500 / 60 × 2 ≈ 16` → `concurrency=16`.
- **Steps 2–3:** covered by the knob (factory pool and admission both become 16).
- **Step 4:** this strategy owns one account quota, so its default scope is sufficient.
- **Step 5:** reserve 450 RPM with `quota_burst_seconds=5`. This permits at most
  487 requests per rolling minute, leaving headroom below 500. The default full
  bucket could instead permit about 900 in the first minute.
- **Step 6:** reserve measured prompt tokens plus 300 expected output tokens.
- **Step 7:** skip the ramp at this width.
- **Step 8:** `cooldown_seconds=60.0` for reactive 429 recovery.
- **Step 10:** one-call p99 ≈ 10s gives `attempt_timeout=30.0`; allow the
  complete retry/cooldown chain 180s and stop the batch at 1h50m.

```python
from llm_minder import (
    CharacterTokenEstimator,
    ErrorCategory,
    GuardrailConfig,
    JsonlArtifactStore,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    llm,
    process_prompts,
)

config = ProcessorConfig(
    concurrency=16,
    attempt_timeout=30.0,
    max_requests_per_minute=450,
    quota_burst_seconds=5,
    max_tokens_per_minute=200_000,
    token_estimator=CharacterTokenEstimator(expected_output_tokens=300),
    rate_limit=RateLimitConfig(cooldown_seconds=60.0),
    guardrails=GuardrailConfig(
        total_timeout_per_item=180.0,
        batch_timeout=6600.0,
        abort_on_error_categories=frozenset(
            {ErrorCategory.AUTHENTICATION, ErrorCategory.INSUFFICIENT_BALANCE}
        ),
    ),
)

batch = await process_prompts(
    llm("openai:gpt-6-luna"),
    prompts,
    config=config,
    progress=True,
    artifact_store=JsonlArtifactStore("runs/overnight.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
print(batch.summary())
```

Rerunning the same command after a crash replays completed successes from the
artifact and executes only the remainder.

## If memory is the constraint

For very large or unbounded inputs, add `max_queue_size` (bounded input
buffering) and switch to `process_stream()` so results don't accumulate.
That's a memory decision, not a throughput one — see
[bounded work and backpressure](bounded-work.md).

## Numeric configuration

Retry, rate-limit, startup-ramp, guardrail and processor numeric fields require finite

numbers. Counts and concurrency limits require Python integers, excluding booleans
and floats (including integral floats). Delays and timeouts reject NaN, infinity
and booleans. Use the documented `None` options to disable optional limits.
Configuration is validated at construction; use a fresh config or
`dataclasses.replace` when changing values, rather than mutating it afterward.
