# Results, Audit Artifacts, and Resume

Production runs often need two different outputs: a convenient batch summary and
an append-only checkpoint that survives interruption. `BatchResult` serialization
serves the first use case; an artifact store (`JsonlArtifactStore` or
`SqliteArtifactStore`) serves the second.

## Completion order and input order

Batch processing and `process_stream()` publish results in **completion order** by
default. This keeps fast items visible while a slow or retrying item is still in
flight and preserves existing behavior.

Use `preserve_order=True` when collecting a batch in input order:

```python
result = await process_prompts(
    strategy,
    [("slow", "slow prompt"), ("fast", "fast prompt")],
    preserve_order=True,
)
```

The processor assigns an internal `submission_index` when each item is accepted.
It does not derive order from `item_id`, so duplicate IDs are safe (they're accepted,
with a one-time `UserWarning`). To reorder an
existing batch without mutating it, call `ordered = result.in_input_order()`.
This raises `ValueError` if any result predates submission indexes; it never
guesses.

Each `LLMWorkItem` object is a single submission. Duplicate IDs on distinct
objects are supported, but submitting the identical mutable object twice raises
`ValueError`; create a new work item for each queue entry.

`process_stream()` intentionally remains completion ordered. Ordered streaming
would block all later results behind one slow early item and require a reorder
buffer that can grow without a safe bound.

## JSON result serialization

Result mappings use the named `async-batch-llm-result` schema and an integer
schema version. The schema keeps the package's pre-1.0 name, so results written by
older releases still load. They include terminal status, submission index, error category,
replay state, token usage, complete attempt/item timing, and batch termination
metadata.

```python
from pathlib import Path
from llm_minder import BatchResult

value = result.to_json()
Path("summary.json").write_text(value, encoding="utf-8")

restored = BatchResult.from_json(
    Path("summary.json").read_text(encoding="utf-8")
)

result.to_jsonl("summary.jsonl")
restored_lines = BatchResult.from_jsonl("summary.jsonl")
```

JSONL result files contain one complete versioned `WorkItemResult` record per
line. A zero-result batch uses one versioned batch-metadata record so termination
state still round-trips. These are summary exports, not resumable checkpoints;
use an artifact store for checkpoint/resume.

The safe encoder accepts normal JSON primitives plus dataclasses, Pydantic
models, enums, `datetime`/`date`/`time`, UUIDs, filesystem paths, tuples, and
sets. Tuples and sets normalize to JSON lists; dataclasses and Pydantic models
normalize to mappings; dates, UUIDs, and paths normalize to strings. Sets are
ordered deterministically. Without explicit decoders, those normalized values
remain JSON-native mappings, lists, and strings after deserialization.

Application-specific values need an explicit encoder:

```python
payload = result.to_json(
    encoder=lambda value: {"widget_id": value.id}
)
```

`WorkItemResult.from_dict()` and the `BatchResult.from_*()` methods accept
`output_decoder` and `context_decoder` hooks for trusted type reconstruction.
Output decoding runs only for successful results whose output was included and
is not `None`; failed, omitted, and null outputs bypass it. Context decoding runs
only for non-null context. Artifact inspection and replay use the same rules.
Deserialization never imports a class named by untrusted data. Exceptions are
stored only as module name, class name, and redacted message; the restored
runtime `exception` is `None`. Tracebacks and raw exception objects are never
persisted. Framework-controlled error text and values under structured
authorization/API-key/token keys are redacted. Other user-controlled strings
round-trip without text rewriting, so applications must keep credentials out
of prompts, outputs, context, and metadata.

Unsupported values, malformed input, and future schema versions raise
`ResultSerializationError`; values are never silently replaced with `repr()`.
A reader on the same schema version can also reject a value that a newer release
added; see [Compatibility across releases](#compatibility-across-releases).

## Middleware and replay

For each accepted item, the processor establishes its total deadline and runs
`before_process` once before preparing artifact identity or looking up an old
result. Retries use this effective request without repeating preprocessing.
Current filtering therefore always wins over historical success.

A replacement may change prompt, context, and strategy, but cannot change the
accepted item ID. Submission indexes are preserved. Fingerprints, inferred
provider/model identity, strategy class, lookup, and audit fields describe the
effective item; append uses the exact fingerprint prepared before execution.
Explicit `ArtifactIdentity` values still override inferred identity. Automatic
stores still require one consistent inferred identity per run, regardless of
worker preparation order. Identity and input serialization errors produce a
failed `artifact_preparation_error` result for that item during processing, after
middleware. Other accepted items continue; store I/O, format, and checkpoint-write
failures still propagate for ordinary execution checkpoints. To stop immediately
on identity or input errors, include `artifact_preparation_error` in
`GuardrailConfig.abort_on_error_categories`; it remains opt-in so isolated bad
inputs do not discard valid neighboring results.

Replay retains historical output, usage, and timing, binds current effective
context and the current submission index, and neither calls strategy `prepare()`
nor appends a duplicate record. Capacity configuration and admission-scope binding
still run once per effective strategy. A completed lookup is used even if a
deadline or abort arrives concurrently. `after_process` runs once for a newly executed success,
before append, and does not rerun on replay. Changes to `after_process` that leave
effective inputs unchanged require an `application_version` or `parser_version`
bump in `ArtifactIdentity`. Middleware functions are not automatically hashed.

Filtered items produce a current-run `middleware_filtered` result. Preprocessing
filtering and invalid middleware requests do not open the store or append records.
Aborts, [budget stops](guardrails.md#token-and-cost-budgets),
[admission refusals](guardrails.md#attempt-admission-hook), and total-item/batch
deadlines are checkpointed for audit when an artifact
key can be prepared, with `replay_eligible=False`. Interrupted artifact preparation
is not restarted solely for audit, and unrepresentable inputs cannot be checkpointed.
Artifact errors during batch-abort, batch-budget, admission-refusal, or batch-deadline audit
preparation or append
are logged, preserving the controlled stop and its results even when that audit
record cannot be persisted. A per-item deadline is an ordinary failed result in
an otherwise running batch: its checkpoint errors still propagate, although its
record remains ineligible for replay.

An abort can append one audit row per accepted queued item before the run returns.
These writes use the store's normal locking and persistence path, so large queues
can add substantial abort-drain time. Audit preparation and append are not bounded
by the expired guardrails; a store that blocks on open or write can delay draining
indefinitely.

Both backends exclude legacy abort/deadline records during replay as well, allowing
an older compatible success to be reused. A filter record explicitly appended via
the store API has `replay_eligible=False`, including under `REUSE_ALL`. Both JSONL
and SQLite also reject legacy filter records bearing the exact historical
`Skipped by middleware` error, even when those records were marked replayable.
Configuration failures (`token_estimator_required`, `token_estimation_error`,
`token_estimate_exceeds_limit` and `quota_scope_error`) are likewise checkpointed with
`replay_eligible=False`, and legacy records carrying those categories are excluded, so
fixing the configuration and resuming re-executes the item even under `REUSE_ALL`.
Unlike abort and deadline audit records, their checkpoint errors propagate.
`quota_scope_error` fails before an artifact key is prepared, so it is not
checkpointed at all.
Both stores fall back to an older compatible result when a newer record is excluded.
Both retain the existing version-1 schema. Newly executed
results are still checkpointed before publication or post-processing callbacks.

## Resumable JSONL artifacts

An artifact begins with a version-1 manifest followed by versioned item records.
The manifest records UTC creation time, package version, canonical identity,
its SHA-256 fingerprint, and optional user metadata. Each terminal item record
contains input fingerprints, current submission index, strategy class,
identity/provenance, safe result data, timing, token use, error category, replay
eligibility, and optional caller-calculated cost.

The minimal form needs only a path (v0.20.0):

```python
from llm_minder import JsonlArtifactStore, ResumePolicy, process_prompts

result = await process_prompts(
    strategy,
    prompts,
    artifact_store=JsonlArtifactStore("runs/customer-tagging.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

The run closes the store when it finishes: `process_prompts()`, `process_stream()`
and processor context exit all close the store passed in, and a closed store raises
`ArtifactIOError` if used for another run. Construct a new store instance, on the
same path, for each run. Path-based inspection (`read_results`) still works
afterward.

When no `ArtifactIdentity` is given, `provider` and `model` are inferred from
the strategy at run start and the remaining identity fields default to
`"unversioned"`. Prompt — and, by default, context — still participate in the
per-item compatibility fingerprint, so a changed prompt or a changed model never
silently replays a stale result. Inference reads the strategy's `model`
attribute:

| `strategy.model` | Inferred `provider` | Inferred `model` |
| --- | --- | --- |
| A built-in model (`GeminiModel`, `OpenAIModel`, ...) | `gemini`, `openai`, `openrouter`, `deepseek` | the model name |
| A plain string, such as `self.model = "model-a"` | the strategy class name | the string |
| Another object with a non-empty string `_model`, `model_name`, `model_id` or `model` attribute (checked in that order) | the object's class name | that attribute |
| Another object without one, or an empty string | — | raises `ArtifactIdentityError` |
| No `model` attribute (for example `PydanticAIStrategy`) | the strategy class name | `agent.model` or its `model_name`, else `"unknown"` |

The error case fails each item with `artifact_preparation_error` before any
provider call, because the store cannot tell when such a model changes. Pass
`identity=ArtifactIdentity(provider=..., model=...)` to the store, or give the
strategy an `artifact_identity` property that returns one.

Before 1.0.0rc2, a custom strategy whose `model` was a plain string, or an object
without `_model`, inferred `model="unknown"` (with `provider="str"` for a string),
so switching models replayed the old model's results. Checkpoints written that way
no longer match: on resume those items run again and append new records, and the
old records stay in the file as audit history. Nothing is replayed from them and
no error is raised. To keep reusing them, pass an explicit
`ArtifactIdentity(provider="str", model="unknown", prompt_version="unversioned",
parser_version="unversioned", application_version="unversioned")` (or the custom
model's class name as `provider`), but only if you are sure the model never
changed.

Automatic identity covers the provider and the model. `OpenAIModel` adds its API
surface when it is the Responses API, and `DeepSeekStrategy` adds its API surface and
strict response schema. Sampling and request parameters
(temperature, reasoning effort, `extra_body`, and other request kwargs) are not part
of identity or the item fingerprint, so changing them does not invalidate replay.
For example, v0.27 changed the default temperature from `0.0` to `None` (the
provider default); a v0.26 checkpoint resumed with `REUSE_SUCCESSES` replays
successes produced at `temperature=0.0` without warning, except for `OpenAIModel`
on the new Responses default, whose identity changed. When a request parameter
must invalidate replay, pass an explicit `ArtifactIdentity` that records it (for
example `extra={"temperature": 0.0}`) or bump `prompt_version`.

Automatic identity is homogeneous within one live store instance/run. The first
prepared item pins its inferred identity; every later item is inferred and must
match before artifact lookup, append, or provider work begins. Use separate
stores for heterogeneous strategies, or pass one deliberate explicit
`ArtifactIdentity` that describes the mixed/routed run. A later store instance
may pin a different automatic identity and append to the same physical
artifact—older identities remain audit history and only compatible records
participate in replay.

`CallableStrategy` is intentionally stricter: an arbitrary function cannot
safely reveal its provider, model, route, parser, or application version. Pass
`identity=ArtifactIdentity(...)` to `CallableStrategy` or directly to
`JsonlArtifactStore`. Omitting both fails before the invocation callback runs;
llm-minder never derives identity from a lambda, closure, object ID, memory address, or
unstable `repr()`. An explicit store identity takes precedence over the
strategy's identity.

**When to use the full identity:** versioned production pipelines. An
explicit `ArtifactIdentity` lets a prompt-template change, parser change, or
application release invalidate replay even when the literal prompt text is
unchanged (e.g. context assembled outside the prompt, or a new parser reading
the same outputs):

```python
from llm_minder import (
    ArtifactIdentity,
    JsonlArtifactStore,
    ResumePolicy,
    process_prompts,
)

store = JsonlArtifactStore(
    "runs/customer-tagging.jsonl",
    identity=ArtifactIdentity(
        provider="openai",
        model="example-model",
        prompt_version="v3",
        parser_version="v2",
        application_version="2026.07",
    ),
)

result = await process_prompts(
    strategy,
    prompts,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

An explicit identity is the caller-owned compatibility boundary. It may
intentionally describe a mixed-strategy run, but the caller must change it
whenever routing, provider, model, parser, prompt policy, or application
semantics should invalidate replay.

A terminal record is flushed before its result is returned or yielded. Set
`fsync=True` for an operating-system durability barrier after every record;
flush-only is the default. Readers ignore every final line without a newline,
even if it contains complete JSON, and never modify the file during inspection.
Writable preparation validates retained records, truncates the incomplete tail,
and builds replay state only from retained records. A dropped valid item is
executed again; its zero-based stored sequence may be reused, remaining unique
among retained records. Warnings report dropped bytes, a SHA-256 and the item ID
when available, without logging raw payloads. Repair is also fsynced when enabled.

On macOS, `fsync()` flushes host buffers to the drive but does not force the
drive to write its cache to stable storage (that needs `F_FULLFSYNC`, which the
store does not issue). `fsync=True` there does not guarantee against data loss
from an OS crash or power loss.

A file containing only a recognizable canonical manifest prefix can be repaired
and initialized again. Unrecognized files, unsupported versions, and malformed
complete or middle records fail without truncation. After any write failure,
the JSONL instance rejects later appends; close it before opening a new instance
for recovery. JSONL still supports only one writer per file. Cross-process writer
locking remains outside this release; do not resume a file another process is
currently writing.

### Compatibility matching

Replay requires all of the following to match:

- item ID;
- SHA-256 prompt fingerprint;
- context fingerprint when `context_in_identity=True` (the default);
- combined input fingerprint; and
- complete `ArtifactIdentity` fingerprint and supported artifact schema.

Matching only an item ID is never sufficient. Changing provider, model,
prompt/parser/application version, identity `extra`, prompt, or participating
context invalidates the old record. When several records match, the newest
complete record wins.

Context is canonically JSON-encoded before a provider call. Supply `encoder=`
or `context_fingerprinter=` when an application context is not supported; the
store raises `ArtifactSerializationError` rather than silently excluding that
identity component.

Sensitive structured values are redacted from persisted identity, context,
output, and metadata mappings. Their original values still feed the one-way
context/identity fingerprint, so a credential change invalidates replay without
writing the credential itself to the artifact.

### Compatibility across releases

The test suite keeps artifacts and serialized results written by published releases
readable. `tests/test_legacy_fixtures.py` checks that files written by v0.18, v0.21,
v0.24.1 and v0.26 (JSONL and, from v0.21, SQLite artifacts, plus `BatchResult` JSON
and JSONL) still load, replay, and accept new records.

Two caveats apply:

- Replay exclusions apply retroactively. A newer release that stops replaying a
  category (for example the configuration failures above) also ignores older records
  carrying it, and re-executes those items.
- Forward compatibility is not guaranteed. An older release on the same schema
  version can reject a value that a newer release added. For example, a `BatchResult`
  serialized with `termination.kind == "budget_exceeded"` (v0.27) can't be read by
  v0.26, although `RESULT_SCHEMA_VERSION` is still 1.

### Resume policies

- `ResumePolicy.NONE` never reuses old results but still checkpoints new ones.
- `ResumePolicy.REUSE_SUCCESSES` reuses the newest compatible success and reruns
  failures, missing items, and stale items.
- `ResumePolicy.REUSE_ALL` also reuses the newest compatible terminal failure.

Replayed items do not call the provider and are not appended a second time.
They retain historical output, timing, error, and token use, receive the current
run's `submission_index` and current work-item context, and set
`replayed_from_artifact=True`. Persisted historical context is never decoded or
substituted during replay. Historical
tokens remain visible on the item for audit but are excluded from newly consumed
provider-token statistics returned by `processor.get_stats()`. In contrast,
`BatchResult` aggregate token fields are computed from all returned results and
therefore include replayed historical usage. This makes live processor stats a
"spent this run" view and the collected batch an auditable result-history view.

For a callable, compatible replay likewise bypasses the invocation callback.
Changing its explicit identity invalidates replay and executes it again.

When a billed response fails parsing or validation, raise
`TokenTrackingError(token_usage=...)` from the callback. That attempt's usage is
retained and added to later successful or terminal usage. llm-minder cannot account
for transport retries hidden inside an upstream gateway unless the gateway
reports them.

Setting `include_output=False` makes a success record audit-only and therefore
ineligible for replay. An ordinary provider failure remains reusable under `REUSE_ALL` because it
does not need a successful output. Guardrail audit, filtered, and artifact
serialization failures are never replayed, and neither are configuration failures
(see [Middleware and replay](#middleware-and-replay)).

A JSONL iterator started before the same store's first writable preparation can
overlap tail truncation and subsequent appends. In that narrow case, it may see
new records within its original byte range. Finish inspection before starting
writable use if a strict snapshot is required.

### Item-local artifact serialization failures

Built-in JSONL and SQLite stores validate and UTF-8 encode records before writing.
An invalid output, metadata, persisted context, or cost-calculator result makes
that item fail with `artifact_serialization_error` while neighboring items finish.
The returned result retains its output for salvage and its token usage and timing.
The checkpoint is a minimal failure record: output, metadata, raw prompt, raw
context, and calculated cost are omitted. It cannot replay even under `REUSE_ALL`,
so fixing an encoder or persistence configuration allows the item to run again.

Error messages identify the failing JSON path where applicable. Lone surrogates
are rejected in application data. Stored diagnostic error and exception text
escapes them as `\ud800`-style sequences; in-memory error text is preserved.
Canonical encoding and fingerprints for valid Unicode do not change.

Serialization failures do not invoke middleware `on_error` or retry the provider
within the same run. `abort_on_error_categories={"artifact_serialization_error"}`
can stop a run after that item's failure checkpoint. Existing guardrail audit
results keep their original category if their payload needs the same minimal
projection. Only batch-abort, batch-budget, and batch-deadline audit writes remain best-effort;
a failed total-item-deadline checkpoint remains fatal. I/O errors, invalid artifact
format, unprepared store keys, custom-store failures, and failure to serialize or
write even the minimal checkpoint still propagate under those same rules.

Subclassing the built-in stores is not a supported extension interface. In
particular, overriding `append` does not intercept their private minimal-failure
write path. Implement the public `ArtifactStore` protocol for custom write or
redaction policies.

A provider success event may already have reached lifecycle observers before a
checkpoint serialization failure is discovered. Until terminal-event handling is
unified, observer success counts can differ from final result and processor counts
on this path. No second terminal event is emitted.

Inspection decode-error messages identify one-based record positions on both
backends. JSONL's stored `record_sequence` still starts at zero; SQLite's starts
at one. This changes diagnostics only, not stored keys or replay ordering.

### Privacy controls

Prompt and context hashes are always stored for matching, but raw prompts and
raw context are excluded by default. Raw provider responses are never stored.
The independent options are:

- `include_output=True` (default);
- `include_metadata=True` (default);
- `include_prompt=False` (default); and
- `include_context=False` (default).

Outputs and metadata can themselves contain sensitive application data. Review
their shape before enabling artifacts, or disable either field. Raw prompt and
context persistence must be explicitly enabled.

### Generate, review, apply

Artifacts support a provider-free review/apply phase:

```python
from llm_minder import JsonlArtifactStore

review = JsonlArtifactStore.read_results(
    "runs/customer-tagging.jsonl",
    successes_only=True,
)
for item in review.results:
    await apply_approved_output(item.item_id, item.output)
```

Inspection restores raw context only when the writer explicitly used
`include_context=True`. Without a `context_decoder`, the restored value remains
JSON-native; a supplied decoder is applied only when stored context exists.
Context omitted by the default privacy policy, including an original `None`,
inspects as `None`.

Replay never substitutes that historical audit context for the live input and
does not invoke the stored-context decoder. It always attaches the current work
item's context and submission index to the replayed result.

For asynchronous inspection through an open store, use
`async for item in store.iter_results(successes_only=True)`. On both stores,
`iter_results()` raises `ArtifactIOError` once the store is closed: at the start
if it was already closed, otherwise at the iterator's next page read, so results
from a page already read may still come out first. To read a closed artifact,
use `read_results(path)`.

Costs are optional and caller-supplied because this package does not maintain a
provider price database:

```python
store = JsonlArtifactStore(
    "run.jsonl",
    identity=identity,
    cost_calculator=lambda item: calculate_cost_from_current_prices(item),
)
```

The artifact stores `null` when there is no calculator.

## Indexed SQLite artifacts for large restartable runs

`SqliteArtifactStore` (v0.21) stores the same logical version-1 records in a
standard-library SQLite database with indexed replay lookup and bounded
iteration. It exists for 100k–1M-item restartable runs where scanning or
retaining a JSONL history on open becomes the bottleneck; JSONL remains the
recommended portable, human-inspectable audit format.

```python
from llm_minder import ResumePolicy, SqliteArtifactStore, process_stream

store = SqliteArtifactStore("runs/customer-tagging.sqlite")
async for item in process_stream(
    strategy,
    prompts,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
):
    await sink(item)
```

Everything documented above for JSONL applies unchanged: identity resolution
and explicit `ArtifactIdentity`, the complete compatibility fingerprint, the
three resume policies, replay accounting, privacy controls
(`include_output` / `include_metadata` / `include_prompt` /
`include_context`), custom encoders/decoders and `context_fingerprinter`, and
optional `cost_calculator`. The two backends share one codec, so a record
that would replay from JSONL replays from SQLite under the same conditions.
Context-free items match through a NULL-safe indexed predicate. There is no
automatic JSONL→SQLite import; opt in by creating a new store, and existing
JSONL files remain fully supported.

| Backend | Best fit | Reopen behavior | Write behavior | Human inspection |
| --- | --- | --- | --- | --- |
| JSONL | Small/medium runs, portable audit log | Resume scans/indexes complete history; `iter_results()` streams line pages | One flushed line per record | Excellent |
| SQLite | 100k–1M restartable runs | Indexed replay and bounded keyset iteration — no history decode on open | Batched transactions with WAL checkpoints | Use SQLite tools/API |

“100k+” is a recommendation, not a cliff — the crossover depends on record
size and environment.

### Batched commits and durability

Appends are grouped into transactions (`commit_batch_size`, default 100,
within `commit_interval_seconds`, default 0.01). An `append()` — and therefore
the processor's publication of that item's result — completes only after its
transaction commits, preserving checkpoint-before-result-publication. A
transaction failure rolls back without exposing partial rows.

`SqliteDurability` selects the synchronization policy:

- `BALANCED` (default) — WAL with `synchronous=NORMAL`. A normal process
  crash retains a consistent database and committed history. A sudden OS or
  power failure may lose transactions committed since the last WAL sync;
  the loss window can span more than one transaction.
- `FULL` — WAL with `synchronous=FULL`; stronger power-loss durability at
  lower throughput. Exact guarantees always depend on SQLite, the OS,
  filesystem, and storage hardware. On macOS, SQLite syncs with plain `fsync()`
  unless its `fullfsync` pragma is enabled (off by default, and the store does
  not set it), so neither policy guarantees power-loss durability there.

WAL auto-checkpointing keeps the log at a bounded plateau during healthy
writes; `close()` drains accepted appends, attempts a truncating checkpoint,
and terminates the store's worker thread. An external long-lived reader can
make that final checkpoint report “busy” — recorded, not an error and not
data loss. SQLite may leave an `-shm` sidecar after a healthy close, and the
database directory will transiently contain `-wal`/`-shm` files during a run.
llm-minder does not encrypt SQLite artifacts; use filesystem permissions and volume
encryption as your deployment requires.

Cleanup finishes before `close()` reports an error. A previously unobserved
writer or cancelled-caller operation error is delivered first; a distinct
checkpoint or connection-close error is retained for the next `close()`.
Each store-level error is delivered once, and later closes return silently.
Observing an error never restores a failed store to usability.

### Inspection

Inspecting an existing SQLite store before its first run does not pin its model
identity. Its first prepared item registers the resolved identity before appending,
even if the connection was already opened by inspection. Automatic stores still
reject mixed effective identities within one run.

Live `store.iter_results()` first settles detached errors, prepares the
writable store, and flushes its accepted writes. It then streams a finite
committed high-water snapshot in keyset-paged batches (`read_batch_size`),
decoding one row at a time. No cursor or read transaction is held across an
async yield, so a slow consumer never blocks the writer or WAL checkpointing.

`SqliteArtifactStore.read_results(path)` is a separate path-based,
materializing convenience. It opens an internal SQLite `mode=ro` connection,
never creates a schema or identity, never starts a writer, never changes WAL or
synchronization policy, and never checkpoints an active writer. Normal reads
participate in SQLite locking and change detection, allowing a writer to start
before or during inspection; SQLite may create empty `-wal`/`-shm`
coordination sidecars in a writable directory. A clean database in a
non-writable directory uses SQLite's immutable fallback, under the filesystem
guarantee that no writer can start while the read is active. The method
captures a committed high-water sequence and excludes rows committed later; a
subsequent read sees them. Its connection and owned reader thread close after
success, error, or ordinary cancellation. Repeated cancellation remains prompt
and lets the already-running reader thread clean itself up without
synchronously blocking the event loop. Unlike the synchronous JSONL
counterpart, it is an **async** classmethod because SQLite I/O stays off the
event loop:

```python
from llm_minder import SqliteArtifactStore

review = await SqliteArtifactStore.read_results(
    "runs/customer-tagging.sqlite",
    successes_only=True,
)
```

Opening a writable store validates schema and version markers, records the
current run's distinct identity when needed (a sequential identity history for
provenance), and never decodes stored history before a lookup needs it. The
read-only path validates the same application ID, versions, tables, columns,
required index names, and manifest without recording an identity. Each item row
consumed by either path is checked against the supported logical artifact
schema before its result or persisted context JSON is decoded; unsupported or
malformed row versions fail loudly instead of falling back to an older
compatible row.

## Inspecting a store from the command line

The `llm-minder` command reads a JSONL or SQLite artifact store without running a batch, so you can answer "how
did that run go?" and "what should I rerun?" from a shell. It is also available as `python -m llm_minder`. It uses
only the standard library, and it is **provisional** (see [API stability](stability.md)): commands, flags, exit codes
and JSON fields may still change in a minor release.

```text
llm-minder inspect  STORE [--input-price P --output-price P --cached-token-rate R [--cache-write-rate R]] [--json]
llm-minder failures STORE [--category C]... [--limit N] [--ids-only | --json] [--full]
llm-minder show     STORE ITEM_ID [--output] [--prompt] [--metadata] [--full] [prices] [--json]
llm-minder --version
```

- **`inspect`** summarizes the store:
  - the manifest;
  - the identities seen, with record counts;
  - the record and item counts;
  - an inventory of items **by their newest record**: succeeded, failed by category, and stopped by a guardrail by
    category;
  - token totals and recorded cost, summed over **every** record, because earlier attempts were billed too;
  - an optional cost estimate;
  - the recorded guardrail stops.

  The inventory says nothing about whether a given identity or prompt would replay.
- **`failures`** lists items whose newest record is a failure, in store order. Each line has the item id, the
  category and the error text.
  - `--category` repeats and filters.
  - `--limit` caps the list; the total is still reported.
  - `--ids-only` prints one id per line, for feeding a rerun.
  - Items stopped by a guardrail (`batch_aborted`, `batch_deadline_exceeded`, `batch_budget_exceeded`,
    `batch_admission_refused`, `framework_total_item_timeout`) are left out unless you name their category, so a
    rerun list isn't padded with stopped work. Stopped items can include attempts that ran and were billed (timed
    out or aborted), not only work that never started.
- **`show`** prints every record for one item, oldest first. For each record you get:
  - the error text;
  - token usage, the recorded cost and an optional estimate;
  - per-attempt timing.

  The output, prompt and metadata appear only with `--output`, `--prompt` and `--metadata`.

**Cost.** The recorded cost is the sum of each record's `calculated_cost`, with a count of how many records have
one. An **estimate** needs `--input-price` and `--output-price`, per million tokens in any currency, and
`--cached-token-rate`.

| Flag | Accepts |
| --- | --- |
| `--input-price`, `--output-price` | a finite number, 0 or more |
| `--cached-token-rate` | a finite number in [0, 1], or `gemini`, `openai`, `anthropic-read` or `deepseek` |
| `--cache-write-rate` (optional) | a finite number, 0 or more, or `openai-write`, `anthropic-write` or `anthropic-write-1h` |

- Names resolve to the [`CachedTokenRates`](api/core.md) constants.
- `inspect` applies the `BatchResult.estimated_cost` arithmetic once, to the token totals. `show` applies it to each
  record. A sum of `show`'s estimates can differ from `inspect`'s figure in the last digits, because
  `effective_input_tokens` rounds up.
- An invalid price or rate is a usage error, reported before the store is opened. So is an estimate too large to
  represent, which finite prices can still produce for large token counts. That error is reported before anything
  is printed.

**Privacy.**

- Output and context are never decoded, and no output, context or custom decoder ever runs.
- Output, prompts and item metadata are hidden unless their flag is given.
- The manifest's user metadata and its identity's `extra` mapping appear only as their keys.
- **Error text is shown by default**, because it is the point of `failures`. It is truncated to 200 characters
  unless `--full` is given, and that is not redaction: error and stop-reason text can contain fragments of a prompt
  or an output. The library already redacted framework-controlled text and credential-like keys when it wrote the
  record.

**Item ids.**

- Ids are printed exactly. Human-readable output escapes backslashes and control characters (`\n`, `\r`, `\x1b`, NUL
  and so on), so a stored string can't forge lines or terminal escapes.
- `--ids-only` refuses to print an id containing a line break, NUL or another control character. It prints nothing,
  names how many ids are affected, and exits 5; `failures --json` carries every id exactly.

**Exit codes.**

| Code | Meaning |
| --- | --- |
| 0 | Success, including "no failures" |
| 2 | Usage error, including an invalid price or rate, or an estimate that can't be represented |
| 3 | The store is missing, unreadable, foreign (text or SQLite), from a newer release, or corrupt |
| 4 | `show`: the item isn't in the store |
| 5 | `failures --ids-only`: an id can't be printed one per line |

On every non-zero exit, stdout is empty and the error is on stderr. The one exception is a store rewritten in place
during the command; see the next paragraph.

**Read-only, one snapshot, two passes.**

- The CLI never creates, repairs, migrates or truncates a store.
- **JSONL:** the store is read as bytes and nothing is created beside it. Each command captures the file size when
  it starts and reads only up to that point. A partial last line is treated like a truncated tail.
- **SQLite:**
  - A store is opened with SQLite's native read-only mode inside one read transaction. That mode takes part in WAL
    locking, so reading a store while a run is writing to it is safe.
  - Like `SqliteArtifactStore.read_results`, the open may create the standard `-wal` (empty) and `-shm`
    coordination files beside any SQLite-format file, including one it then rejects. The CLI never deletes them,
    because a live writer may own them.
  - A store that was closed cleanly in a directory you can't write to may fail with exit 3 and a message to copy it
    somewhere writable.
- **Passes:** each command reads its snapshot twice.
  - Pass 1 validates every record and builds only counts. Corruption anywhere is found there, before any output.
  - Pass 2 re-reads the same snapshot and streams the output record by record.
  - Records appended during the command aren't reported.
  - The store only appends, so don't edit a JSONL store in place. If its bytes are rewritten during a command, pass
    2 can still hit a decode error. The command then exits 3 with a message, and any lines already printed stay
    printed.

**Memory.**

- Peak memory is the grouping state plus the records in flight.
- **Grouping state:** one entry per distinct item id (the id plus a sequence number, a success flag and a
  category), plus a counter per identity and per category.
- **Records in flight:** one JSONL line, or one page of 100 SQLite rows. A single huge output, prompt or error costs
  roughly its own size while it is parsed.
- Nothing else is held across records. Output text is streamed in pass 2, so memory doesn't grow with the size of
  the history or the number of records printed.

`benchmarks/scale_soak/cli_memory.py` measures both axes:

| Store | Axis | Command | Python heap peak | RSS over baseline |
| --- | --- | --- | --- | --- |
| JSONL | 100,000 items | `inspect`, `failures` | 18.8 MiB | 51 MiB |
| JSONL | 100,000 items | `show` (one item) | 0.2 MiB | 0.3 MiB |
| JSONL | one 8 MiB output, 2 MiB error | every command | 26.2 MiB | 26 to 36 MiB |
| SQLite | 100,000 items | `inspect`, `failures` | 18.8 MiB | 51 to 53 MiB |
| SQLite | 100,000 items | `show` (one item) | 0.2 MiB | 2.4 MiB |
| SQLite | one 8 MiB output, 2 MiB error | `inspect`, `failures --full` | 20.1 MiB | 19 to 20 MiB |
| SQLite | one 8 MiB output, 2 MiB error | `show --output --full` | 28.1 MiB | 27 MiB |

These numbers are from Python 3.15 on Linux (2026-11-18).

- The grouping state costs about 190 bytes of Python heap per distinct item id.
- A single large payload costs about three times its size while it is parsed: the raw line or row, the decoded
  string, and the output being written.
- The RSS figures were measured with `tracemalloc` running, which adds its own overhead, and exclude the
  interpreter and package import baseline.
- Run `uv run python -m benchmarks.scale_soak.cli_memory` to measure your own stores' shape.

### JSON output

`--json` prints one JSON document followed by a newline.

- **Envelope.** Every document starts with `"schema_version": 1`, then `"command"` (`"inspect"`, `"failures"` or
  `"show"`), then `"store": {"path": ..., "kind": "jsonl" | "sqlite"}`. The path is printed as given.
- **Compatibility within version 1:** keys may be added, and consumers must ignore keys they don't know. Renaming
  or removing a key, or changing its type, bumps `schema_version`.
- **Field names** follow the stored names: `item_id`, `error_category`, `recorded_at`, `calculated_cost`, the
  `TokenUsage` keys and the `AttemptTiming` keys.
- **Nullability.** A field a store may lack (for example an old record's `recorded_at`) is present as `null`. A
  field controlled by a flag is **absent** unless the flag is given.
- **Units:**
  - token counts are integers;
  - seconds are floats;
  - costs are floats in the prices' currency;
  - prices are per million tokens.
- **Numbers are always finite**; no document contains `NaN` or `Infinity`.
- **Error text** (`error`, `first_reason`, `exception.message`) is truncated to 200 characters unless `--full` is
  given. A matching `*_truncated` boolean says whether it was cut. `inspect` has no `--full` flag, so its
  `first_reason` is always truncated. JSON escaping protects control characters, so the text is otherwise as
  stored.
- **Streaming.** `failures` and `show` stream their arrays, so the document is never built in memory.

#### `inspect --json`

```json
{
  "schema_version": 1,
  "command": "inspect",
  "store": {"path": "run.sqlite", "kind": "sqlite"},
  "manifest": {
    "artifact_schema_version": 1,
    "sqlite_schema_version": 1,
    "created_at": "2026-11-17T10:00:00Z",
    "package_version": "1.2.0",
    "identity_fingerprint": "ab12...",
    "identity": {
      "provider": "openai",
      "model": "gpt-5",
      "prompt_version": "v2",
      "parser_version": null,
      "application_version": null,
      "extra_keys": ["temperature"]
    },
    "user_metadata_keys": ["experiment", "owner"]
  },
  "record_count": 1250,
  "item_count": 1000,
  "identities": [
    {"provider": "openai", "model": "gpt-5", "prompt_version": "v2", "records": 1250}
  ],
  "items_by_newest_record": {
    "succeeded": 950,
    "failed": [{"error_category": "rate_limit", "items": 30}, {"error_category": null, "items": 2}],
    "stopped_by_guardrail": [{"error_category": "batch_deadline_exceeded", "items": 18}]
  },
  "token_totals": {
    "input_tokens": 1000000,
    "output_tokens": 200000,
    "cached_input_tokens": 400000,
    "cache_write_input_tokens": 0,
    "total_tokens": 1200000
  },
  "recorded_cost": {"amount": 12.5, "records_with_cost": 1200, "records": 1250},
  "estimated_cost": {
    "amount": 3.0,
    "input_price": 1.25,
    "output_price": 10.0,
    "cached_token_rate": 0.5,
    "cache_write_rate": null
  },
  "guardrail_stops": [
    {"error_category": "batch_deadline_exceeded", "count": 18, "first_reason": "...", "first_reason_truncated": false}
  ]
}
```

- **`manifest`:**
  - `sqlite_schema_version` is `null` for JSONL.
  - `identity` has the five named `ArtifactIdentity` fields, each a string or `null`, plus `extra_keys`, the sorted
    keys of `identity.extra`. The `extra` values are never printed.
  - `user_metadata_keys` lists the keys of the store's user metadata.
- **`identities`** has one entry per distinct provider, model and prompt version over all records. It is sorted by
  `records` (most first), then by the three fields, with `null` first.
- **`items_by_newest_record`** counts each item once, by its newest record.
  - `failed` and `stopped_by_guardrail` are lists, because a category can be `null`. Both are sorted by `items`
    (most first), then by category.
  - A guardrail-stopped item appears only under `stopped_by_guardrail`.
- **`token_totals`** are sums over all records. All five keys are always present. `total_tokens` is the sum of the
  stored totals, not a recomputed value.
- **`recorded_cost.amount`** is `null` when no record has a cost. When only some records do, it is their partial
  sum.
- **`estimated_cost`** is `null` unless prices are given. The rates are shown resolved, as numbers.
- **`guardrail_stops`** lists only categories with at least one recorded stop, in the order above. It reports what
  was recorded, not how the run ended: a run can trip a limit without a stopped-item record, and a resumed run can
  finish cleanly after an earlier stop.

#### `failures --json`

```json
{
  "schema_version": 1,
  "command": "failures",
  "store": {"path": "run.jsonl", "kind": "jsonl"},
  "categories": null,
  "total": 32,
  "shown": 32,
  "excluded_guardrail_stops": 18,
  "failures": [
    {
      "item_id": "doc-17",
      "sequence": 412,
      "recorded_at": "2026-11-17T10:04:11Z",
      "error_category": "rate_limit",
      "error": "429 Too Many Requests ...",
      "error_truncated": false,
      "provider": "openai",
      "model": "gpt-5",
      "prompt_version": "v2"
    }
  ]
}
```

- **`categories`** lists the `--category` values in the order given, or is `null` without a filter.
- **`total`** is the number of matching items before `--limit`; `shown` is the smaller of `total` and the limit.
- **`excluded_guardrail_stops`** counts items whose newest record is a guardrail stop that was left out because no
  `--category` named its category.
- **`failures`** is ordered by `sequence`, the store's own ordering: the SQLite `record_sequence`, or the record's
  position in the JSONL file. It is stable for an unchanged store, but can't be compared across stores.

#### `show --json`

```json
{
  "schema_version": 1,
  "command": "show",
  "store": {"path": "run.sqlite", "kind": "sqlite"},
  "item_id": "doc-17",
  "record_count": 2,
  "records": [
    {
      "sequence": 101,
      "is_newest": false,
      "recorded_at": "2026-11-17T10:01:00Z",
      "success": false,
      "error_category": "rate_limit",
      "error": "429 ...",
      "error_truncated": false,
      "exception": {"module": "openai", "class_name": "RateLimitError", "message": "...", "message_truncated": false},
      "identity": {"provider": "openai", "model": "gpt-5", "prompt_version": "v2", "identity_fingerprint": "ab12..."},
      "token_usage": {"input_tokens": 1200, "output_tokens": 0},
      "calculated_cost": 0.0015,
      "estimated_cost": null,
      "submission_index": 17,
      "replayed_from_artifact": false,
      "timing": {
        "total_seconds": 4.2,
        "timeout_category": null,
        "attempts": [{"attempt": 1, "try_number": 1, "total_seconds": 4.2, "...": "every AttemptTiming key"}]
      }
    }
  ]
}
```

- **`records`** lists the item's records oldest first. `is_newest` marks the record `inspect` and `failures`
  classify by.
- **`token_usage`** is as stored: only the keys the record has.
- **`estimated_cost`** is `null` unless prices are given.
- **`exception`** is the stored descriptor (module, class name and redacted message), or `null`.
- **`timing.attempts`:** each attempt carries every current `AttemptTiming` key. A key an older release didn't
  record is `null`.
- **Flag-gated keys:**
  - `--output` adds `output_included` and `output`: the stored JSON value, never decoded, or `null` when it wasn't
    stored.
  - `--prompt` adds `prompt_stored` and `prompt`: the stored string or message list, or `null` when the store was
    written without `include_prompt`.
  - `--metadata` adds `metadata_included` and `metadata`.

## Process-safety boundary

Both stores are single-process, single-writer designs. `JsonlArtifactStore`
serializes writes with an `asyncio.Lock`, guaranteeing complete,
non-interleaved JSON records for concurrent workers sharing one store
instance in one process. `SqliteArtifactStore` owns one connection, one
worker thread, and one writer task per instance; sequential reopen is
supported and a concurrently open external reader obeys the configured busy
timeout. Neither implements distributed work claiming, leasing, or
cross-process exactly-once execution — use one writer per artifact.

On open, the JSONL store still builds an in-memory replay index over complete
item records for constant-time resume lookup (its `iter_results()` streams
bounded line pages instead of materializing history). Very large
audit histories are exactly what `SqliteArtifactStore` is for.
