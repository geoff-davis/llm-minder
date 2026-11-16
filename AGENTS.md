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
| Tests, default selection | `make test-fast` |
| Markdown lint with fixes | `make markdown-lint-fix` |
| Build and check the sdist/wheel | `make package-check` |
| One test | `uv run pytest tests/test_retry_logic.py -k partial_name` |
| Preview the docs site | `uv sync --group docs && uv run mkdocs serve` |

Without make, the equivalents are `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/` and
`uv run mypy src/llm_minder/ --ignore-missing-imports`. Do not add `examples/` to the ruff paths.

## Commits and pull requests

- Hooks run on every commit (ruff, mypy, whitespace, YAML/TOML, markdownlint, no commits to `main`) and
  `check-branch-fresh` on every push. Run them on everything with `uv run prek run --all-files`. Do not use
  `--no-verify`.
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
- `pytest-timeout` caps every test at 60 s so a deadlock fails instead of hanging CI.
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
- **API details that are easy to get wrong.** `ParallelBatchProcessor.get_stats()` and
  `MetricsObserver.get_metrics()` are async. `LLMWorkItem.strategy` takes a strategy instance, not a raw agent or
  model. `LLMWorkItem.prompt` is a string; message lists go in `messages=`, and a custom strategy accepts them only
  by overriding `execute_messages()` (`docs/structured-prompts.md`).

## CI and releases

- `test.yml` runs on every push and PR: `test` (Python 3.11 to 3.15; all five legs are required checks),
  `test-macos`, `quality` (ruff, mypy, ty, `make package-check`), `sdk-compat` (each SDK's floor and the latest
  release of every supported major), `scale-smoke`, `docs-build` and `security` (pip-audit on the locked runtime
  dependencies, npm audit). Markdownlint runs only in the hook and `make ci`, not in CI.
- `docs.yml` deploys the MkDocs site to GitHub Pages on push to `main`.
- `publish.yml` is tag-triggered: version check, `make package-check`, PyPI upload. It does not create the GitHub
  release; `/release-tag` does, after PyPI confirms.
- GitHub Actions are pinned to exact tags.
- The release process is `/release-prep`, merge, `/release-tag`; see "Release Process" in `CONTRIBUTING.md`.

## Direct Herdr Reviews

For this repository, Claude implements and Codex reviews. The user authorizes direct
review requests and replies through Herdr; no manual relay is needed.

- Verify `HERDR_ENV=1` and use `herdr agent list` to check agent identity and repository
  before sending. The current live names are `abl-implementer` (Claude) and
  `abl-reviewer` (Codex). Names belong to live occupants; rediscover them after restarts.
- Claude requests review with `herdr agent prompt abl-reviewer "<review request>"`.
  Include the base, immutable commit or patch/hash, evidence location, scope, settled
  decisions, and known limitations. Keep the review target unchanged during review.
- Codex replies with `herdr agent prompt abl-implementer "<review verdict and findings>"`.
  Review read-only in an isolated snapshot; leave implementation edits to Claude.
  Follow the evidence protocol below and identify the exact target in every verdict.
- Put each review round's files under one directory, `/tmp/abl-review/<PR>-r<round>/`,
  and delete the whole directory once the verdict is sent. Snapshots, tarballs,
  environments, logs and test temp dirs can be regenerated, so none of them is kept.
  Probe scripts and their output can't be: before deleting, put the full probe code (or
  exact commands to recreate it) and the output that matters in the verdict, or point to
  a committed regression test that covers it.
- Send without `--wait` so neither agent blocks waiting for the other's turn. Do not
  send acknowledgment loops or duplicate prompts. If delivery is uncertain, inspect
  the target before retrying; never answer an approval dialog on the user's behalf.
- Review approval does not itself authorize a commit, push, merge, tag, or publication;
  follow the user's authorization for those actions.

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

## Where context lives

- `docs/internal/` (gitignored, local to the main checkout): plans, design documents, handoffs and Codex review
  notes (`*-codex-review.md`). Read these to pick up an in-progress design.
- GitHub issues: `gh issue list -R geoff-davis/llm-minder`. `docs/roadmap.md` lists releases.
- `CHANGELOG.md`: what shipped, release by release.
- `docs/migration/`: migration guides; `index.md` says which to read from a given release.
- `.claude/review-context.md`: settled decisions and out-of-scope items for reviews.
