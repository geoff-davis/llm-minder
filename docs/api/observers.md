# Observers API Reference

## ProcessorObserver

::: llm_minder.observers.ProcessorObserver

## BaseObserver

::: llm_minder.observers.BaseObserver

## MetricsObserver

::: llm_minder.observers.MetricsObserver

## Prometheus export

`await metrics.export_prometheus()` returns what a `MetricsObserver` has collected
as a string in the Prometheus text exposition format. It is a point-in-time
snapshot, not a registered `prometheus_client` collector: call it whenever you want
fresh values and publish the string yourself. Values accumulate across every batch
the observer sees until `await metrics.reset()`.

### Serving it

A batch job usually exits before a scraper would reach it, so write the text for the
node-exporter [textfile collector](https://github.com/prometheus/node_exporter#textfile-collector)
when the job finishes. Write to a temporary file and rename it, so the collector
never reads a half-written file:

```python
import os
from pathlib import Path

from llm_minder import MetricsObserver, llm, process_prompts


async def run_and_export(prompts: list[str], textfile_dir: Path) -> None:
    metrics = MetricsObserver()
    await process_prompts(llm("openai:gpt-6-luna"), prompts, observers=[metrics])
    target = textfile_dir / "llm_minder.prom"
    partial = textfile_dir / "llm_minder.prom.tmp"
    partial.write_text(await metrics.export_prometheus(), encoding="utf-8")
    os.replace(partial, target)
```

A long-running service can serve it over HTTP instead. This uses only the standard
library; any web framework's handler works the same way:

```python
import asyncio

from llm_minder import MetricsObserver


async def serve_metrics(metrics: MetricsObserver, port: int = 9108) -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")  # ignore the request; every path is /metrics
        body = (await metrics.export_prometheus()).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain; version=0.0.4\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    async with server:
        await server.serve_forever()
```

### What it exports

Every metric is prefixed `llm_minder_` and unlabelled, except the two labelled
counters at the end. Names don't follow the `_total` suffix convention for
counters; dashboards should use them as listed.

<!-- The test_prometheus_docs test keeps this table in step with export_prometheus(). -->

| Metric | Type | Meaning |
| --- | --- | --- |
| `llm_minder_items_processed` | counter | Total items processed |
| `llm_minder_items_succeeded` | counter | Total items succeeded |
| `llm_minder_items_failed` | counter | Total items failed |
| `llm_minder_rate_limits_hit` | counter | Total rate limits encountered |
| `llm_minder_structured_output_recoveries` | counter | Total structured outputs recovered |
| `llm_minder_items_replayed` | counter | Total terminal items replayed from artifacts |
| `llm_minder_items_aborted` | counter | Total collateral items aborted by guardrails |
| `llm_minder_batches_aborted` | counter | Total batches stopped by guardrails |
| `llm_minder_quota_admitted_attempts` | counter | Total live quota reservations admitted |
| `llm_minder_reserved_tokens` | counter | Total tokens reserved for TPM admission |
| `llm_minder_reported_reconciliation_tokens` | counter | Total known tokens used for reconciliation |
| `llm_minder_refunded_tokens` | counter | Total unused reserved tokens refunded |
| `llm_minder_underestimated_tokens` | counter | Total token underestimation charged as debt |
| `llm_minder_unknown_usage_attempts` | counter | Started quota attempts with unknown usage |
| `llm_minder_known_zero_usage_attempts` | counter | Started quota attempts reporting zero usage |
| `llm_minder_token_estimation_failures` | counter | Total framework token-estimation failures |
| `llm_minder_structured_output_retries_avoided` | counter | Total validation retries avoided by structured-output recovery |
| `llm_minder_avg_processing_time` | gauge | Average processing time in seconds |
| `llm_minder_success_rate` | gauge | Success rate (0.0 to 1.0) |
| `llm_minder_total_cooldown_time` | gauge | Total time spent in rate limit cooldown (seconds) |
| `llm_minder_processing_times_count` | gauge | Number of recorded processing time samples |
| `llm_minder_admission_wait_count` | gauge | Number of provider-capacity admissions |
| `llm_minder_admission_wait_seconds_sum` | gauge | Total provider-capacity wait time in seconds |
| `llm_minder_admission_wait_seconds_max` | gauge | Maximum provider-capacity wait time in seconds |
| `llm_minder_avg_admission_wait_seconds` | gauge | Average provider-capacity wait time in seconds |
| `llm_minder_quota_wait_seconds_sum` | gauge | Total RPM/TPM quota wait time in seconds |
| `llm_minder_quota_wait_seconds_max` | gauge | Maximum RPM/TPM quota wait time in seconds |
| `llm_minder_avg_quota_wait_seconds` | gauge | Average RPM/TPM quota wait time in seconds |
| `llm_minder_estimated_input_tokens` | gauge | Total estimated input tokens |
| `llm_minder_estimated_output_tokens` | gauge | Total estimated output tokens |
| `llm_minder_quota_scope_count` | gauge | Number of run-local quota scopes observed |
| `llm_minder_errors_total{error_type}` | counter | Failed items, live or replayed, by exception class name |
| `llm_minder_structured_output_recoveries_by_reason{reason}` | counter | Structured-output recoveries by reason |

The two labelled counters appear only once they have a value. Every distinct label
value becomes its own time series, so keep both sets small and under your control:

- `error_type` is the exception class name of each failed item, so it grows with
  the number of distinct exception classes your strategies raise.
- `reason` is the item's `structured_output_recovery_reason`. The built-in parsing
  helpers set only `trailing_markdown_fence`, but a custom strategy (including a
  `CallableStrategy`) can return any string in its metadata, and that string is
  used as the label value as it is. Return a short, fixed set of reasons, never
  text taken from the response or the input.

Label values are escaped as the exposition format requires (backslash, double
quote and line feed), so any string reads back unchanged.

There are no histograms or summaries, and the timing metrics cover different
populations:

- `avg_processing_time` and `processing_times_count` are the mean and count of the
  `duration` of successful, live (not replayed) items, each measured over the whole
  item including its retries. There is no processing-time sum or maximum.
- Provider-capacity admission wait (`admission_wait_count`, `_seconds_sum`,
  `_seconds_max`, `avg_admission_wait_seconds`) and RPM/TPM quota wait
  (`quota_admitted_attempts`, `quota_wait_seconds_sum`, `_max`,
  `avg_quota_wait_seconds`) are counted per attempt.
- Percentiles are only in `await processor.get_stats()` and the `BATCH_COMPLETED`
  payload: `admission_wait_p50/p95/p99_seconds`, `quota_wait_p50/p95/p99_seconds`
  and `execution_p50/p95/p99_seconds`. They are computed over the latest 10,000
  individual attempts, failed attempts and retries included, so
  `execution_p50_seconds` is not the median of the items behind
  `avg_processing_time`.

Metric names changed in 1.0 from `async_batch_llm_*` to `llm_minder_*`; dashboards
and alerts built on 0.x output need the new names.

## ProcessingEvent

::: llm_minder.ProcessingEvent

## Event contracts

Interface for observers that monitor processing events.

```python
class ProcessorObserver(ABC):
    @abstractmethod
    async def on_event(
        self, event: ProcessingEvent, data: dict[str, Any]
    ) -> None: ...
```

**Events:**

- `BATCH_STARTED`: `{total, max_workers, start_time}`
- `BATCH_COMPLETED`: `{processed, succeeded, failed, total, total_tokens,
  cached_input_tokens, total_admission_wait_seconds,
  max_admission_wait_seconds, total_quota_wait_seconds,
  max_quota_wait_seconds, quota_wait_p50/p95/p99_seconds,
  estimated_input_tokens, estimated_output_tokens, reserved_tokens,
  reported_reconciliation_tokens, refunded_tokens, underestimated_tokens,
  unknown_usage_attempts, known_zero_usage_attempts,
  token_estimation_failures, quota_scope_count,
  admission_wait_p50/p95/p99_seconds,
  execution_p50/p95/p99_seconds, structured_output_recoveries,
  structured_output_retries_avoided, structured_output_recovery_reasons,
  input_queue_high_water_mark, result_queue_high_water_mark,
  cache_groups, cache_groups_overflow, duration}`
- `BATCH_ABORTED`: `{kind, reason, error_category, triggering_item_id, abort_mode}`
- `WORKER_STARTED` / `WORKER_STOPPED`: `{worker_id}`
- `ITEM_STARTED`: `{item_id, worker_id}`
- `ITEM_ADMITTED`: `{item_id, worker_id, attempt, wait_seconds, capacity,
  startup_ramp_wait_seconds}`
- `QUOTA_ADMITTED`: `{item_id, worker_id, attempt, try_number,
  quota_scope_id, wait_seconds, request_reserved, estimated_input_tokens,
  estimated_output_tokens, estimated_total_tokens, reserved_tokens,
  rpm_configured, tpm_configured, limited_by}`
- `QUOTA_RECONCILED`: `{item_id, worker_id, attempt, try_number,
  quota_scope_id, reserved_tokens, reported_tokens, known_usage,
  delta_tokens, disposition}`
- `ITEM_COMPLETED`: `{item_id, duration, tokens, admission_wait_seconds,
  structured_output_recovered, structured_output_recovery_reason,
  structured_output_retries_avoided}`
- `ITEM_FAILED`: `{item_id, submission_index, error_type, error_category}`
- `ITEM_REPLAYED`: `{item_id, submission_index, success, error_type, error_category}`
- `ITEM_DEADLINE_EXCEEDED`: `{item_id}`
- `RATE_LIMIT_HIT`: `{item_id, worker_id, quota_scope_id, strategy_type}`
- `COOLDOWN_STARTED`: `{worker_id, duration, consecutive, quota_scope_id?, strategy_type?}`
- `COOLDOWN_ENDED`: `{duration, error?}`

A key marked `?` may be absent. The keys listed here are the documented payload that
[API stability](../stability.md) covers; an event may carry more keys, which are
diagnostic and can change in a minor release.

`add_work()` raises public `BatchAdmissionClosedError` (a `RuntimeError`) when
finish, shutdown, or batch abort has stopped admission. Rejected items receive
no submission index. Use a new processor for additional work.

Observers may be duck-typed objects with a callable `on_event(event, data)`;
subclassing `ProcessorObserver` is optional. The callback must return an awaitable.
Invalid callbacks are rejected at processor construction. Observer exceptions and
callback timeouts are logged without failing the item.

Each finalized item emits exactly one of `ITEM_COMPLETED`, `ITEM_FAILED`, or
`ITEM_REPLAYED`, after its checkpoint and before any fail-fast `BATCH_ABORTED` event.
A budget stop is different: the provider attempt that reaches the budget emits
`BATCH_ABORTED` (`kind="budget_exceeded"`) before that item's own terminal event. So
does a stop by the [attempt admission hook](../guardrails.md#attempt-admission-hook)
(`kind="admission_refused"`).
Middleware-filtered items count as failed; middleware recovery and `after_process`
changes determine the final event. Fatal checkpoint failures and cancellation before
finalization produce no terminal item event or processed-stat increment. Early
shutdown may abandon accepted items, so processed counts can be lower than total.
Replay does not emit an additional success/failure event or consume live tokens.
For live provider successes, `duration` and `tokens` retain final-attempt values;
new recovery-only success events use the final result's timing and token totals.

**Cleanup note:**

- Preferred: wrap `ParallelBatchProcessor` in `async with` so the ordered close runs automatically.
- If you do not use a context manager, call `await processor.shutdown()` after `process_all()`. It
  stops workers and callbacks, releases admission state, runs strategy cleanups, and closes the
  artifact store. Repeated calls are safe and retry only what did not complete.

---
