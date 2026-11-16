# Review context: v1.2 development

v1.1.0 is released (2026-11-15). Paste this into background review prompts so they
don't re-report accepted trade-offs. Report new evidence rather than reopening the
settled decisions below. Prune at each release.

## Settled decisions (shipped in 1.0/1.1)

- The public API is frozen under `docs/stability.md`. New names start provisional;
  `scripts/api_snapshot.py` must not change unless a PR adds API on purpose.
- `GuardrailConfig.max_provider_calls` counts attempts the library makes, including
  its own retries. SDK and gateway retries are not counted; the built-in
  OpenAI-compatible models pass `max_retries=0`, and
  `docs/guardrails.md#provider-call-cap` tells users who build their own client (or
  use Gemini's retry options) to turn retries off. Processor-only: `call()`/`LLMCallPool` reject it.
- A closed JSONL or SQLite store raises `ArtifactIOError` ("closed") from every
  method, including an `iter_results()` iterator already running (raised at the next
  page, not mid-page).
- `describe_limits()` gaps are by design: no worker-pool wait row, RPM and TPM share
  one quota row, and the startup-ramp row overlaps capacity.
- `AttemptTiming.cooldown_wait_seconds` counts only time spent waiting on an active
  cooldown.
- Optional SDK handles live in `llm_minder.models._sdk`; tests patch them there.
- `dev` and `docs` are dependency groups, not published extras.
- `GuardrailConfig.admission` (#331, agreed API on the issue): `admit`/`settle` are synchronous and run
  outside the controller lock; a refusal, or any exception from `admit` or `settle`, stops the run as
  `admission_refused` with items `batch_admission_refused` (fail closed). `AdmissionRefused` subclasses
  `BatchAbortedError`. `max_provider_calls` is unchanged and checked before `admit`. An attempt that never
  reached the provider settles with zero usage; `None` means a started call with unknown usage.
  Processor-only. `attempt_generation_config()` runs twice per attempt when a hook is set (documented).

## Out of scope unless the PR says otherwise

- The v1.2 milestone issues not being worked on in the PR under review.
- Unscheduled: #338 (JSONL store memory), #340 (batch vs caching cost estimator).
- Multi-process coordination, persistent queues and provider batch APIs
  (`AGENTS.md`, "Out of scope by design").

## Review method

Follow `AGENTS.md`: reproduce against an isolated base, preserve the review target,
and exercise both artifact backends when record eligibility, replay or store
behaviour changes. A passing suite alone does not establish coverage of a newly
identified path.
