# Repository guide

**llm-minder** runs batches of LLM requests in parallel on one asyncio event loop: worker pool, retries with
backoff, scoped RPM/TPM admission and coordinated cooldowns, resumable JSONL/SQLite checkpoints, deadlines and
budgets. The core is provider-agnostic (a strategy pattern); Gemini, OpenAI, OpenRouter, DeepSeek, Anthropic and
PydanticAI are built in. Until v0.32.0 the package was named `async-batch-llm`.

`CLAUDE.md` imports this file; edit here only. This file holds what an agent would get wrong without being told.
For everything else read the source: `README.md` and `docs/` for the API and guides, `examples/README.md` for the
examples, `CHANGELOG.md` for what shipped, `pyproject.toml` for the current version, `docs/stability.md` for the
1.0 compatibility promise.

## Setup

```bash
uv sync
npm ci                 # Node 20+; installs the markdownlint-cli2 version pinned in package-lock.json
uv run prek install    # once per clone; installs the pre-commit and pre-push hooks
```

## Before starting work

This repo is developed from several machines, so a locally green checkout can trail `origin/main`. On 2026-07-02 a
review and a 22-commit fix batch were built on v0.10-era code while `origin/main` was at v0.15.0; the PR (#59)
conflicted and duplicated fixes main already had. Start every session with:

```bash
git fetch origin
git log --oneline main..origin/main   # any output means local main is stale
git checkout main && git merge --ff-only origin/main
git checkout -b my-feature
```

Then check for anything waiting on the maintainer, and report it before starting the session's task (an outside
feature request once sat unanswered for 25 days):

```bash
python3 scripts/needs_response.py     # open issues/PRs whose latest human activity isn't a maintainer's
gh api repos/geoff-davis/llm-minder/dependabot/alerts \
  --jq '.[] | select(.state=="open") | [.number, .security_advisory.severity, .dependency.package.name, .security_advisory.summary] | @tsv'
```

The `check-branch-fresh` pre-push hook (`scripts/check_branch_fresh.sh`) covers main moving mid-session: it fetches
`origin/main` (failing open when offline) and refuses a push that is missing commits from it, printing the rebase
fix. For an intentionally behind push, skip once with `SKIP=check-branch-fresh git push`.

## Commands

Use the `make` targets; they pin the right paths. Run `make help` for the full list.

| Task | Command |
| --- | --- |
| Full local pipeline (definition of done) | `make ci` |
| Lint, and lint with fixes | `make lint`, `make lint-fix` |
| Format `src/` and `tests/` | `make format` |
| Type check (mypy, ty) | `make typecheck`, `make typecheck-ty` |
| Tests, default selection | `uv run pytest` (or `make test`) |
| Markdown lint with fixes | `make markdown-lint-fix` |
| Build and check the sdist/wheel | `make package-check` |
| One test | `uv run pytest tests/test_retry_logic.py -k partial_name` |
| Preview the docs site | `uv sync --group docs && uv run mkdocs serve` |

Without make, the equivalents are `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/` and
`uv run mypy src/llm_minder/ --ignore-missing-imports`. Do not add `examples/` to the ruff paths.

Do not use `make test-fast` as the default run. Its `-m 'not slow'` replaces the configured marker expression, so
it selects the `integration` tests, which make paid API calls when keys are set.

## Commits and pull requests

- Hooks run on every commit (ruff, mypy and ty at the versions in `uv.lock`; `uv lock --check`; whitespace,
  YAML/TOML; markdownlint at the version in `package-lock.json`, skipped with a message when `node_modules` is
  absent; no commits to `main`) and `check-branch-fresh` on every push. Ruff and mypy are local hooks, so there is
  no hook `rev` to keep in step with the lockfile. Run them on everything with `uv run prek run --all-files`. Do
  not use `--no-verify`.
- `make ci` must pass before requesting review.
- Commit subjects are short and imperative (`Add on_error retry callback`), with a focused scope. Record breaking
  behavior in `CHANGELOG.md`.
- Pull requests link their issues, outline behavioral changes and list new commands or flags.
- Workflow-file changes need a push credential with the `workflow` scope (SSH works; the gh HTTPS token may not).
- Markdown: 120-character lines, a language on every code fence (`text` for plain output), blank lines around
  lists and fences (`.markdownlint.json`).

## Code style

- Python 3.11+, Ruff with a 100-character limit. Type-hinted, dataclass-friendly APIs; keep async flows explicit.
- Concrete strategy classes end in `Strategy`, classifiers in `Classifier`. Observers and middleware are named for
  their side effect (`LoggingObserver`).
- `examples/` is excluded from ruff on purpose: examples check environment variables before importing optional
  dependencies, which fails E402. Do not "fix" them.

## Tests

- `slow`, `integration` and `benchmark` tests are deselected by default. Real API calls carry `integration`.
- Any test that triggers a retry uses the shared `fast_retry` / `fast_rate_limit` fixtures in `tests/conftest.py`;
  otherwise each retry costs a second or more against the library defaults.
- The default test timeout is 60 s (`pytest-timeout`), so a deadlock fails instead of hanging CI. A test may
  override it; the scale-soak and integration tests do. `faulthandler_timeout = 50` dumps every stack first.
- `tests/test_doc_examples.py` parses every fenced Python block in `README.md`, this file, `examples/README.md`
  and `docs/`, resolves `llm_minder` imports and compares overridden hook signatures with the live base classes.
  It does not run the snippets. Opt a block out with `<!-- doc-snippet: skip -->` above the fence.
- `tests/test_legacy_fixtures.py` checks that artifacts written by earlier releases still read and replay. Each
  release adds one fixture with `scripts/write_legacy_fixtures.py` (see `/release-prep`).
- `tests/test_stability_page.py` requires every `__all__` name to be classified on `docs/stability.md`;
  `tests/test_categories.py` requires every produced error or timeout category to be in `ErrorCategory` /
  `TimeoutCategory`; `tests/test_release_docs.py` pins the release version and is updated by each release PR.
- `MockAgent` (`src/llm_minder/testing/mocks.py`) and `FakeStrategy` simulate rate limits, errors and latency
  without API calls. Reuse the helpers in `src/llm_minder/testing/` instead of writing new ones.

## Changing behavior

- When an API changes, search for the old pattern and update `README.md`, `docs/`, `examples/`, this file and the
  tests together.
- Every example handles a missing API key gracefully and uses the current built-in API, not a custom strategy that
  duplicates built-in functionality.
- Before changing a category list, a swallow-or-re-raise rule or a replay predicate, enumerate every member and
  every caller (see "Review Fixes & Handoffs").

## Design invariants

Read the named file before changing any of these.

- **Locks are `asyncio.Lock`s.** They serialize tasks on one event loop and give no OS-thread safety. Most guard
  one piece of state and are released before another is taken. `_stats_lock`, `_results_lock` and
  `_submission_lock` live on the `BatchProcessor` base (`base.py`); `_rate_limit_lock` is a
  `ParallelBatchProcessor` property returning `RateLimitCoordinator._lock`; collaborators and models own their own.
- **Known lock nesting.** `StrategyLifecycle` holds its lock while running `strategy.prepare()`, so a lock a model
  takes inside `prepare()` (for example `GeminiCachedModel`'s cache lock) is acquired under it. Keep that order
  (lifecycle, then model) and do not call back into the lifecycle from model code. When adding a lock, check
  whether it can be taken while another is held.
- **The attempt admission hook runs outside every lock.** `GuardrailConfig.admission` is called right after
  `AbortController.begin_provider_call` returns, with no await before the call starts, and each ticket is settled
  synchronously and once (`AdmissionSlot`): on the attempt's `finally` before any await, or in
  `await_with_guardrails` on a pre-start failure. Adding an await before either breaks exactly-once settle under
  cancellation (`_internal/guardrails.py`, `_internal/item_executor.py`).
- **Rate-limit state lives on `RateLimitCoordinator`** (`_internal/rate_limit_coordinator.py`).
  `ParallelBatchProcessor._in_cooldown` is a read-only property that delegates to it; do not assign it. Only one
  worker triggers the cooldown for a generation (atomic check-and-set under the coordinator's lock); callers whose
  observed generation is stale no-op. `ErrorInfo.suggested_wait` is a floor on the cooldown and is set only from
  genuine server signals such as `Retry-After`.
- **Tokens from failed attempts are still counted.** Strategies attach `_failed_token_usage` to the exception when
  the provider has billed but parsing failed; `TokenExtractor` (`token_extractor.py`) recovers it, and usage is
  aggregated across attempts into `WorkItemResult.token_usage`.
- **Cached-token billing is provider-specific.** Pass a `CachedTokenRates` constant to
  `effective_input_tokens()` / `estimated_cost()`. The billable estimate rounds up.
- **Provider models.** `OpenAIModel` uses the Responses API by default (`store=False`);
  `api_surface="chat_completions"` opts out. `DeepSeekModel` defaults to Chat Completions and needs
  `api_surface="responses"` for strict `response_schema`. OpenAI, OpenRouter and DeepSeek models subclass
  `OpenAICompatibleModel`, and their strategies are thin subclasses of `ModelStrategy`; a new OpenAI-compatible
  provider overrides `_default_base_url` and, if needed, `_extract_tokens`. Tests patch the optional SDK handles in
  `models/_sdk.py`.
- **The two artifact stores must agree.** `JsonlArtifactStore` and `SqliteArtifactStore` have drifted apart more
  than once; exercise both whenever replay, record eligibility or lookup changes.
- **Post-processors must not mutate `result.output`.** Checkpointing happens before post-processing, so a
  mutation changes the returned result while the persisted and replayed result keeps the original, on both stores.
  Treat the result as read-only or build new objects.
- **API details that are easy to get wrong.** `ParallelBatchProcessor.get_stats()` and
  `MetricsObserver.get_metrics()` are async. `LLMWorkItem.strategy` takes a strategy instance, not a raw agent or
  model. `LLMWorkItem.prompt` is a string; message lists go in `messages=`, and a custom strategy accepts them only
  by overriding `execute_messages()` (`docs/structured-prompts.md`).

## CI and releases

- `test.yml` runs on every push and PR: `test` (Python 3.11 to 3.15; all five legs are required checks),
  `test-macos`, `quality` (ruff, mypy, ty, `make package-check`), `sdk-compat` (each SDK's floor and the latest
  release of every supported major), `scale-smoke`, `docs-build` and `security` (pip-audit on every locked
  dependency: runtime, provider extras, and the dev and docs groups; plus npm audit). `quality` also runs
  `make markdown-lint` after `npm ci`.
- Every `--cov` run, including the 3.12 coverage leg, enforces `fail_under` in `[tool.coverage.report]`. Raise it
  when coverage grows; never lower it to pass.
- Every checkout has a 2-minute step timeout, a low-speed abort and one retry step, and each job's timeout leaves
  4 minutes for those two attempts on top of its own work. Keep that pattern in new jobs.
- `docs.yml` deploys the MkDocs site to GitHub Pages on push to `main`.
- `publish.yml` is tag-triggered: version check, `make package-check`, PyPI upload. It does not create the GitHub
  release; `/release-tag` does, after PyPI confirms.
- GitHub Actions are pinned to exact tags.
- The release process is `/release-prep`, merge, `/release-tag`; see "Release Process" in `CONTRIBUTING.md`.

## Direct Herdr Reviews

For this repository, Claude implements and Codex reviews. The user authorizes direct review requests and replies
through Herdr; no manual relay is needed. That covers message delivery only: a review verdict does not authorize a
commit, push, merge, tag or publication.

- Verify `HERDR_ENV=1` and check `herdr agent list` for agent kind and repository before every send. The live
  names are `abl-implementer` (Claude) and `abl-reviewer` (Codex). Names are cleared when an agent exits; after a
  restart, restore them with `herdr agent rename <pane> <name>`. If identity is ambiguous, stop and ask.
- Codex's sandbox reaches Herdr only through `prefix_rule(pattern=["herdr", "agent"], decision="allow")` in
  `~/.codex/rules/default.rules`. A `herdr agent prompt` reply is typed into the recipient's pane, so it arrives
  looking like pasted text.

### When review is required

Review is required before pushing unless every changed file (`git diff --name-only <base>..<head>`) is on the
exempt list below. These paths are always reviewed and take precedence over it: `AGENTS.md` and `CLAUDE.md` at any
depth, everything under `.claude/` and `.github/`, `.pre-commit-config.yaml`, `pyproject.toml`, `Makefile` and
`scripts/`. Exempt:

- `docs/**` (including `docs/internal/`), `CHANGELOG.md` and other `*.md` files;
- `uv.lock`, and Dependabot pull requests with green CI;
- a whole-tree format commit that contains nothing else.

Decide by the paths, not by whether the change feels behavioral. Public-API, shared-policy (category lists,
swallow-or-re-raise rules, replay predicates), artifact-schema and security changes also get the API or design
agreed first. A release gets a review of the range since the last release before `/release-prep`.

Every pull request description has a `Review:` line: either
`docs/internal/reviews/<branch>_review.md, approved at <head SHA>` or `exempt: <which rule>`. The person merging
checks it. The handoff file is named for the branch, with `/` replaced by `-`
(`feat-331-attempt-admission_review.md`), because the first review happens before the push and so before a PR
number exists.

### Tiers

The implementer declares the tier in the handoff; the reviewer may raise it with a reason.

| Tier | Applies to | Evidence | Rounds |
| --- | --- | --- | --- |
| Docs | A reviewed path whose change is prose only: agent instructions, `.claude/` commands and review context | Lint and the doc tests; the handoff states what the text now requires. No regression tests or probes | One, plus one to verify blocking fixes |
| Small | A local fix that adds, removes or loosens no guard | Regression test failing on the pre-fix code; focused tests; lint and types | One |
| Standard | New or changed logic, contracts, or the scope of a guard | As Small, plus a probe of each new or changed condition | One, plus one to verify blocking fixes |
| Critical | Replay and record eligibility, store schemas, checkpoint and data-loss guards, cancellation and lock ordering, public API, security | Probes of every condition in the changed acceptance or refusal logic | Up to three |

A condition probe mutates one conjunct, threshold or budget term in an isolated copy and counts as caught only when
the intended assertion fails. An import error, a harness failure or removing a whole guard does not count. Exempt
paths need no review at all; "Review Fixes & Handoffs" below applies to the code tiers.

### When implementing

- Commit the slice locally with normal hooks; that commit is the review target. Do not amend or rebase it during
  the review; answer findings with new commits.
- Write the handoff to `docs/internal/reviews/<branch>_review.md` before prompting (that directory is gitignored and
  local to the main checkout, which Codex also reads). Include base and head SHAs, tier, acceptance criteria,
  evidence as the tier requires, the policy sweep where one applies,
  settled decisions from `.claude/review-context.md`, known limitations and out-of-scope items.
- Prompt only an idle or done reviewer, without `--wait`, with the handoff path, the target and the round:
  `herdr agent prompt abl-reviewer "<request>"`. Keep the prompt short; the file holds the detail.
- After sending, poll the handoff file in the background about once a minute until it holds a
  `## Reviewer verdict, round <n>` heading whose verdict names the exact target, or until 60 minutes pass. An older
  verdict does not count, and an `idle` or `done` state may predate the request. If the reviewer is blocked or settles
  without a matching verdict, inspect it with `herdr agent read abl-reviewer`; do not resend. At the time limit, stop
  and tell the user.
- Fix blocking findings. Record each non-blocking one with its home: a follow-up commit, an issue number, or
  deliberately unchanged with the reason.

### When reviewing

- Read-only. Review the exact target in an isolated snapshot under `/tmp/abl-review/<PR>-r<round>/`; never stash,
  reset or edit the implementer's tree. Leave implementation edits to Claude.
- Run the same probe on base and target before calling something a regression. Reuse the implementer's evidence
  and run focused checks; repeat full suites only when evidence is missing, stale or invalidated.
- Give each finding a stable ID (`R1`, `R2`, ...), severity, blocking or not, location, trigger and evidence. A
  finding blocks only if it defeats the acceptance criteria or introduces a regression. Separate confirmed defects
  from hypotheses.
- Append the round under a `## Reviewer verdict, round <n>` heading, with a verdict naming the exact target, and
  update the finding ledger. Then send one prompt to
  `abl-implementer` with the verdict.
- Delete the round directory once the verdict is sent. Probe code and output that can't be regenerated go into the
  verdict or a committed regression test first.

### Limits

Three review rounds per pull request, counting the first, unless the user extends them. No acknowledgment messages
and no duplicate prompts; on uncertain delivery, inspect the peer before retrying. A blocked approval dialog goes
to the user; one agent never answers another's permission prompt. Stop and tell the user on approval, after the
third round, at the time limit, on a stalled exchange, or on any decision that needs new authority.

## Review Fixes & Handoffs

For bug fixes and shared-policy changes, provide evidence that lets another session verify the change without
rebuilding the investigation. Scale the handoff to the change; documentation-only edits do not need regression tests.

- Identify the base revision and the exact review target: a commit or an immutable patch/snapshot. If a regression
  arose in uncommitted work, also identify the pre-fix snapshot; a test against `HEAD` alone cannot establish that
  regression. Commit only when authorized. Use isolated checkouts or snapshots for comparisons; never stash, reset,
  or otherwise alter another session's working tree to perform a review.
- Map each finding to the changed behavior, implementation location, and exact regression test or probe command.
  Show the same behavioral check failing before the fix and passing afterward, with revisions/snapshots and relevant
  environment details. Import errors, missing dependencies, and missing new APIs are not fail-first proof. If a
  before-run is unavailable or the finding is structural, state that limitation instead of claiming reproduction.
- Before changing a category list, exception-swallowing rule, or replay predicate, enumerate its members and callers.
  Check the resulting behavior in every affected path, including success, ordinary failure, per-item timeout, batch
  abort/deadline, and persistence failure where applicable. Distinct policies may need distinct named sets; do not
  equate replay eligibility with permission to swallow checkpoint errors. Include the policy sweep in the handoff.
- Exercise both JSONL and SQLite for replay, lookup, or record-eligibility changes. Reuse reviewer-supplied probes
  and existing fixtures, folding useful cases into the nearest regression suite rather than duplicating them.
- State validation commands, results, interpreter versions, deliberate omissions, and accepted tradeoffs precisely.
  A passing full suite does not replace finding-specific evidence. Reviewers should reuse that evidence and run
  focused checks; repeat full suites when new changes or unresolved concerns warrant it. Verify Python 3.11 and a
  current supported interpreter when cancellation or asyncio semantics may differ.
- If review is delegated, include the exact target, settled decisions, known limitations, and out-of-scope items
  so subsequent rounds do not rediscover accepted tradeoffs. Verify candidate findings before presenting them as
  confirmed regressions.

The reason for this protocol: the v0.24.0 lifecycle work took ten review rounds, and most repeat findings traced to
missing evidence rather than missing skill. Most round-N fixes introduced a round-N+1 defect in a sibling path:
dropping the abort-time append lost the audit trail; adding caller attribution to capacity warnings broke warning
deduplication; folding the per-item timeout into the abort audit categories silently swallowed store write errors.

`/code-review` finds defects a foreground pass misses, but it has no memory across rounds: paste the settled
decisions and out-of-scope list from `.claude/review-context.md` into its prompt, or it re-reports accepted
trade-offs. Let it do discovery and verify its candidates yourself. Prune `.claude/review-context.md` at each
release.

## Out of scope by design

Do not "fix" these without an issue that says so.

- Single process only: no multi-process coordination or distributed locks.
- Parallel individual calls, not provider batch APIs.
- The work queue is in memory. Queued work is lost on a crash; results already checkpointed to a JSONL or SQLite
  artifact store replay on resume.
- DeepSeek reuses `OpenAIErrorClassifier` because it is OpenAI-compatible.

## Context window

A full context window is compacted, and the session continues: the conversation is replaced by a summary and work
carries on. Running low is not a reason to stop, rush, or hand work to a subagent that would not otherwise get it.
When the token usage reminder shows the window is nearly full (auto-compaction fires at about 13k to 15k tokens
remaining), reach a clean state first: notes or the plan file updated, the review handoff current, and the working
tree either committed or described.

## Where context lives

- `docs/internal/` (gitignored, local to the main checkout): plans, design documents, handoffs and Codex review
  notes (`*-codex-review.md`). Read these to pick up an in-progress design.
- GitHub issues: `gh issue list -R geoff-davis/llm-minder`. `docs/roadmap.md` lists releases.
- `CHANGELOG.md`: what shipped, release by release.
- `docs/migration/`: migration guides; `index.md` says which to read from a given release.
- `.claude/review-context.md`: settled decisions and out-of-scope items for reviews.
