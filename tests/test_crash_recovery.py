"""Crash-injection harness for artifact-store durability (#328).

A child process runs a writer and SIGKILLs itself at a named point (``tests/_crash_points.py``). The
parent then checks, against the reopened store, the invariants agreed in the #328 design note:

1. the child really died at its point (``-SIGKILL``, no traceback); exit 0 means the point was missed;
2. read-only inspection before recovery has exactly the point's allowed outcome and leaves the file
   byte-identical;
3. a writable reopen succeeds (crash states our own writer produced are always recoverable), after
   which read-only inspection returns;
4. nothing published before the kill is lost;
5. nothing partial is retained;
6. resuming finishes every item exactly once, leaving one success record per item;
7. after the resume, the SQLite WAL is checkpointed away.

SIGKILL tests process death only: the page cache survives it, so nothing here says anything about
power loss or OS crashes.

The default selection runs the six points of the design note's default set. The ``slow`` sweep runs
every other point, the non-streaming publication branch, random record indexes and byte cuts drawn
from a seed (``CRASH_SEED`` to reproduce one), and a timed external kill that samples states no hook
can reach.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from llm_minder import (
    ArtifactError,
    ArtifactIdentity,
    JsonlArtifactStore,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    process_prompts,
)
from llm_minder.testing import FakeStrategy
from tests._crash_points import (
    DEFAULT_POINTS,
    ID_BY_PROMPT,
    ITEMS,
    KILL_AFTER_PUBLISH,
    POINTS,
    PROMPT_BY_ID,
    SQLITE_STORE_KWARGS,
    TIMED_ITEMS,
    CrashPoint,
    read_side_log,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the harness kills the child with SIGKILL"
)

REPO = Path(__file__).resolve().parents[1]
IDENTITY = ArtifactIdentity(provider="p", model="m")

#: The sweep's seed: ``CRASH_SEED`` reproduces a run; otherwise a fresh one, shown in every failure.
SEED = int(os.environ.get("CRASH_SEED") or random.SystemRandom().randrange(2**32))
#: Random cases per point that has something to vary.
RANDOM_CASES = 2
#: Timed external-kill cases per store.
TIMED_CASES = 3
#: Fresh delays tried when the timed kill lands after the work finished (design note section 5).
TIMED_RETRIES = 3


def _make_store(kind: str, path: Path) -> Any:
    if kind == "jsonl":
        return JsonlArtifactStore(path, identity=IDENTITY)
    return SqliteArtifactStore(path, identity=IDENTITY, **SQLITE_STORE_KWARGS)


async def _read(kind: str, path: Path) -> Any:
    if kind == "jsonl":
        return JsonlArtifactStore.read_results(path)
    return await SqliteArtifactStore.read_results(path)


def _child_env() -> dict[str, str]:
    # The child imports the same checkout: an inherited PYTHONPATH (a probe's src/ copy) comes first,
    # then this checkout's src/ and root (for ``tests._crash_points``).
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (env.get("PYTHONPATH"), os.fspath(REPO / "src"), os.fspath(REPO)) if part
    )
    return env


def _run_child(
    point: CrashPoint, path: Path, workdir: Path, params: dict[str, Any] | None = None
) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, "-m", "tests._crash_child", point.id, os.fspath(path)]
    args.append(os.fspath(workdir))
    if params:
        args.append(json.dumps(params))
    return subprocess.run(
        args,
        cwd=workdir,
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _store_path(tmp_path: Path, kind: str) -> Path:
    return tmp_path / ("store.jsonl" if kind == "jsonl" else "store.sqlite")


@pytest.mark.asyncio
@pytest.mark.parametrize("point_id", DEFAULT_POINTS)
async def test_crash_point_recovers(
    point_id: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    await _check_point(POINTS[point_id], {}, tmp_path, caplog, f"point {point_id}")


def _sweep_cases() -> list[Any]:
    """Every case of the slow sweep, as ``(point_id, case)`` ids; parameters are drawn per test."""
    cases = []
    for point_id, point in sorted(POINTS.items()):
        if point_id not in DEFAULT_POINTS:
            cases.append(pytest.param(point_id, "fixed", id=f"{point_id}-fixed"))
        if point.layer == "processor":
            cases.append(pytest.param(point_id, "list", id=f"{point_id}-list"))
        if point.random_params is not None:
            for index in range(RANDOM_CASES):
                case = f"random{index}"
                cases.append(pytest.param(point_id, case, id=f"{point_id}-{case}"))
    # The torn-manifest cut lengths named in the design note, beyond the default 20.
    for cut in (1, 41, 42, -1):
        case = f"cut{cut}"
        cases.append(pytest.param("jsonl-create-torn-manifest", case, id=f"manifest-{case}"))
    return cases


def _case_params(point: CrashPoint, case: str) -> dict[str, Any]:
    if case == "fixed":
        return {}
    if case == "list":
        return {"branch": "list"}
    if case.startswith("cut"):
        return {"cut": int(case[3:])}
    assert point.random_params is not None
    rng = random.Random(f"{SEED}:{point.id}:{case}")
    params = point.random_params(rng)
    if point.layer == "processor":
        params["branch"] = rng.choice(["stream", "list"])
    return params


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(("point_id", "case"), _sweep_cases())
async def test_crash_sweep_recovers(
    point_id: str, case: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    point = POINTS[point_id]
    params = _case_params(point, case)
    label = f"point {point_id}, case {case}, CRASH_SEED={SEED}, params {params}"
    # Captured output is shown with any failure, including the shared invariants' bare assertions.
    print(f"reproduce: {label}")
    await _check_point(point, params, tmp_path, caplog, label)


async def _check_point(
    point: CrashPoint,
    params: dict[str, Any],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    label: str,
) -> None:
    path = _store_path(tmp_path, point.store)
    workdir = tmp_path / "work"
    workdir.mkdir()

    # 1. The child died at its point.
    child = _run_child(point, path, workdir, params)
    context = f"{label}; child stderr:\n{child.stderr}"
    assert child.returncode == -signal.SIGKILL, f"invariant 1: {context}"
    assert "Traceback" not in child.stderr, context

    published = read_side_log(workdir / "published.log")
    killed_batch = read_side_log(workdir / "batch.log")
    _check_landing(point, params, path, child.stderr, published, context)

    # 2. Read-only inspection before recovery: exactly the allowed outcome, file unchanged.
    if point.hot_journal:
        assert Path(f"{path}-journal").exists(), context
    before = path.read_bytes() if path.exists() else None
    expected = point.inspection
    if expected.error is None:
        await _read(point.store, path)
    else:
        with pytest.raises(ArtifactError) as raised:
            await _read(point.store, path)
        assert type(raised.value).__name__ == expected.error, repr(raised.value)
        assert expected.match in str(raised.value), repr(raised.value)
    after = path.read_bytes() if path.exists() else None
    assert after == before, "read-only inspection changed the store file"
    if point.identity_missing:
        assert _identity_rows(path) == 0, context

    # 3-7.
    killed = set(killed_batch) if point.killed_batch else set()
    if point.killed_batch:
        # At commit the batch holds more than one row; a random insert index may cut its first.
        assert len(killed_batch) > (0 if "insert" in params else 1), killed_batch
    stored = await _recover_and_resume(
        point.store,
        path,
        caplog,
        published,
        items=ITEMS,
        torn_tails={1} if point.torn_tail else {0},
        killed=killed,
    )
    retained_successes = {r.item_id for r in stored.results if r.success}
    if point.retained_unpublished:
        assert retained_successes - set(published), "the cut was not before publication"
    if point.all_retained:
        assert retained_successes == {item_id for item_id, _ in ITEMS}, sorted(retained_successes)


def _check_landing(
    point: CrashPoint,
    params: dict[str, Any],
    path: Path,
    stderr: str,
    published: list[str],
    context: str,
) -> None:
    """Point-specific evidence, before any reopen, that the kill landed where the table says."""
    if point.published_at_kill:
        assert len(published) == params.get("k", KILL_AFTER_PUBLISH), (published, context)
    if point.complete_json_tail:
        raw = path.read_bytes()
        tail = raw[raw.rfind(b"\n") + 1 :]
        assert json.loads(tail)["record_type"] == "item", context
    if point.repaired_at_kill:
        assert "Dropped" in stderr, context
        raw = path.read_bytes()
        assert raw == b"" or raw.endswith(b"\n"), (raw[-80:], context)
    if point.empty_file:
        assert path.exists() and path.stat().st_size == 0, context
    if point.rollback_journal:
        # Header bytes 18-19 are 1 for a rollback-journal database and 2 for WAL.
        assert path.read_bytes()[18:20] == b"\x01\x01", context
        assert not Path(f"{path}-wal").exists(), context
    if point.wal_truncated_at_kill:
        wal = Path(f"{path}-wal")
        assert wal.exists() and wal.stat().st_size == 0, context
        assert Path(f"{path}-shm").exists(), context


def _identity_rows(path: Path) -> int:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM identities").fetchone()[0])
    finally:
        connection.close()


async def _recover_and_resume(
    kind: str,
    path: Path,
    caplog: pytest.LogCaptureFixture,
    published: list[str],
    *,
    items: list[tuple[str, str]],
    torn_tails: set[int],
    killed: set[str],
) -> Any:
    """Invariants 3-7 against the crashed store; returns the store's contents before the resume."""
    # 3. Writable reopen succeeds; afterwards inspection returns.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        store = _make_store(kind, path)
        try:
            try:
                await store._prepare()
            finally:
                await store.close()
        except ArtifactError as exc:
            pytest.fail(f"invariant 3: writable reopen failed: {exc!r}")
    dropped = [r for r in caplog.records if "Dropped" in r.getMessage()]
    stored = await _read(kind, path)

    # 4. Nothing published is lost.
    retained_successes = {r.item_id for r in stored.results if r.success}
    assert set(published) <= retained_successes, (
        f"invariant 4: published but not retained: {sorted(set(published) - retained_successes)}"
    )

    # 5. Nothing partial is retained.
    for result in stored.results:
        assert result.success, result
        assert result.output == f"out:{PROMPT_BY_ID[result.item_id]}", result
    if kind == "jsonl":
        # Complete lines only, as the store reads them: a final segment without a newline is a torn
        # tail the store ignores (and may not even be valid UTF-8).
        complete = path.read_bytes().split(b"\n")[:-1]
        sequences = [json.loads(line)["record_sequence"] for line in complete[1:]]
        assert len(sequences) == len(set(sequences)), sequences
    retained_ids = {r.item_id for r in stored.results}
    assert not retained_ids & killed, "part of the killed batch was retained"
    assert len(dropped) in torn_tails, f"invariant 5: {[r.getMessage() for r in dropped]}"

    # 6. The resume finishes exactly once.
    executed: list[str] = []

    def respond(prompt: str) -> str:
        executed.append(ID_BY_PROMPT[prompt])
        return f"out:{prompt}"

    strategy: FakeStrategy[str] = FakeStrategy(response=respond)
    resumed = await process_prompts(
        strategy,
        items,
        config=ProcessorConfig(max_workers=3),
        artifact_store=_make_store(kind, path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert resumed.succeeded == len(items)
    replayed = {r.item_id for r in resumed.results if r.replayed_from_artifact}
    assert replayed == retained_successes
    assert Counter(executed) == Counter(
        item_id for item_id, _ in items if item_id not in retained_successes
    )
    final = await _read(kind, path)
    per_item = Counter(r.item_id for r in final.results if r.success)
    assert per_item == Counter(item_id for item_id, _ in items), f"invariant 6: {per_item}"

    # 7. A clean close after the resume checkpoints the WAL away.
    if kind == "sqlite":
        wal = Path(f"{path}-wal")
        assert not wal.exists() or wal.stat().st_size == 0
    return stored


# ── Timed external kill ────────────────────────────────────────────────────


#: Upper bound on waiting for one protocol line (READY or DONE) or for the child to exit.
PROTOCOL_TIMEOUT = 30.0


class _TimedChild:
    """A timed child whose pipes are drained by threads, so every wait has a deadline.

    Use it as a context manager: on any exit, normal or not (a failed assertion, a timeout, an
    interrupt), a child that is still alive is killed and reaped.
    """

    def __init__(self, popen: subprocess.Popen[str]) -> None:
        self.popen = popen
        self.stdout_lines: list[str] = []
        self.stderr = ""
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_parts: list[str] = []
        self._readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    def _read_stdout(self) -> None:
        assert self.popen.stdout is not None
        for line in self.popen.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read_stderr(self) -> None:
        assert self.popen.stderr is not None
        self._stderr_parts.append(self.popen.stderr.read())

    def next_line(self, timeout: float) -> str | None:
        """The next stdout line, or None at end of output; fails after ``timeout`` seconds."""
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(f"timed child printed no protocol line within {timeout}s")
        if line is not None:
            self.stdout_lines.append(line)
        return line

    def finish(self) -> int:
        """Wait (bounded) for exit and drain both pipes; returns the exit code."""
        returncode = self.popen.wait(timeout=PROTOCOL_TIMEOUT)
        for reader in self._readers:
            reader.join(timeout=PROTOCOL_TIMEOUT)
        while self.next_line(PROTOCOL_TIMEOUT) is not None:
            pass
        self.stderr = "".join(self._stderr_parts)
        return returncode

    def __enter__(self) -> _TimedChild:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.popen.poll() is None:
            self.popen.kill()
        self.popen.wait(timeout=PROTOCOL_TIMEOUT)
        for reader in self._readers:
            reader.join(timeout=PROTOCOL_TIMEOUT)


def _start_timed_child(kind: str, path: Path, workdir: Path) -> _TimedChild:
    workdir.mkdir()
    args = [sys.executable, "-m", "tests._crash_child", "timed", kind, os.fspath(path)]
    args.append(os.fspath(workdir))
    popen = subprocess.Popen(
        args,
        cwd=workdir,
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return _TimedChild(popen)


def _await_ready(child: _TimedChild) -> None:
    """Wait (bounded) until the child prints READY; fail with its stderr if it exits first."""
    line = child.next_line(PROTOCOL_TIMEOUT)
    if line != "READY\n":
        returncode = child.finish()
        pytest.fail(f"timed child exited before READY ({returncode}): {child.stderr}")


def _classify_attempt(returncode: int, stdout_lines: list[str]) -> str:
    """``crash`` for a SIGKILL that landed before DONE, ``retry`` for a run that reached DONE.

    A child that printed DONE has finished its work and closed the store; killing it during
    interpreter teardown still returns -SIGKILL, but that is not a crash state, so it is retried.
    """
    if "DONE\n" in stdout_lines:
        assert returncode in (0, -signal.SIGKILL), (returncode, stdout_lines)
        return "retry"
    assert returncode == -signal.SIGKILL, (returncode, stdout_lines)
    return "crash"


def _calibrate(kind: str, root: Path) -> float:
    """The READY-to-DONE time of one unkilled run of the timed workload: the delay range D."""
    with _start_timed_child(kind, _store_path(root, kind), root / "work") as child:
        _await_ready(child)
        ready = time.monotonic()
        done = child.next_line(PROTOCOL_TIMEOUT)  # timed at DONE: interpreter teardown is no work
        elapsed = time.monotonic() - ready
        returncode = child.finish()
        assert returncode == 0 and done == "DONE\n", (returncode, done, child.stderr)
    return elapsed


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("case", range(TIMED_CASES))
@pytest.mark.parametrize("kind", ["jsonl", "sqlite"])
async def test_timed_kill_recovers(
    kind: str, case: int, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rng = random.Random(f"{SEED}:timed:{kind}:{case}")
    print(f"reproduce: timed {kind} case {case}, CRASH_SEED={SEED}")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    span = _calibrate(kind, calibration)

    # Kill after a random delay measured from READY. A run that reached DONE is never a crash case,
    # whatever its exit code; retry it with a fresh delay (and store) from the first half of the range.
    for attempt in range(1 + TIMED_RETRIES):
        delay = rng.uniform(0, span if attempt == 0 else span / 2)
        root = tmp_path / f"attempt-{attempt}"
        root.mkdir()
        path = _store_path(root, kind)
        with _start_timed_child(kind, path, root / "work") as child:
            _await_ready(child)
            time.sleep(delay)
            child.popen.send_signal(signal.SIGKILL)
            returncode = child.finish()
        stderr = child.stderr
        if _classify_attempt(returncode, child.stdout_lines) == "crash":
            break
    else:
        pytest.fail(f"timed kill never landed before DONE (D={span:.3f}s, CRASH_SEED={SEED})")

    label = f"timed {kind} case {case}, CRASH_SEED={SEED}, D={span:.3f}s, delay={delay:.3f}s"
    print(f"reproduce: {label}")
    assert "Traceback" not in stderr, f"{label}: {stderr}"
    workdir = root / "work"
    published = read_side_log(workdir / "published.log")

    # 2. After READY, every fixed point's inspection returns, so this one must too.
    before = path.read_bytes()
    await _read(kind, path)
    assert path.read_bytes() == before, "read-only inspection changed the store file"

    # 3-7. A JSONL kill may or may not tear the final record.
    await _recover_and_resume(
        kind,
        path,
        caplog,
        published,
        items=TIMED_ITEMS,
        torn_tails={0, 1} if kind == "jsonl" else {0},
        killed=set(),
    )


# ── Timed-kill driver ──────────────────────────────────────────────────────


def _script_child(code: str) -> _TimedChild:
    popen = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return _TimedChild(popen)


_READY_THEN_IDLE = 'import time; print("READY", flush=True); time.sleep(60)'
_READY_DONE_THEN_IDLE = 'import time; print("READY"); print("DONE", flush=True); time.sleep(60)'


def test_attempt_killed_before_done_is_a_crash() -> None:
    with _script_child(_READY_THEN_IDLE) as child:
        _await_ready(child)
        child.popen.send_signal(signal.SIGKILL)
        returncode = child.finish()
    assert _classify_attempt(returncode, child.stdout_lines) == "crash"


def test_attempt_killed_after_done_is_retried() -> None:
    # The child is still alive (as in interpreter teardown) when the kill lands, so it exits -9.
    with _script_child(_READY_DONE_THEN_IDLE) as child:
        _await_ready(child)
        assert child.next_line(PROTOCOL_TIMEOUT) == "DONE\n"
        child.popen.send_signal(signal.SIGKILL)
        returncode = child.finish()
    assert returncode == -signal.SIGKILL
    assert _classify_attempt(returncode, child.stdout_lines) == "retry"
    assert _classify_attempt(0, ["READY\n", "DONE\n"]) == "retry"


def test_attempt_with_other_exit_fails() -> None:
    with pytest.raises(AssertionError):
        _classify_attempt(1, ["READY\n"])


@pytest.mark.parametrize(
    "code", ["import time; time.sleep(60)", _READY_THEN_IDLE], ids=["no-ready", "no-done"]
)
def test_silent_child_fails_promptly_and_is_reaped(
    code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[_TimedChild] = []

    def start(kind: str, path: Path, workdir: Path) -> _TimedChild:
        started.append(_script_child(code))
        return started[-1]

    monkeypatch.setattr(sys.modules[__name__], "_start_timed_child", start)
    monkeypatch.setattr(sys.modules[__name__], "PROTOCOL_TIMEOUT", 0.5)
    began = time.monotonic()
    with pytest.raises(pytest.fail.Exception, match="no protocol line"):
        _calibrate("jsonl", tmp_path)
    assert time.monotonic() - began < 10
    assert started[0].popen.poll() is not None


def test_interrupted_attempt_reaps_the_child() -> None:
    with pytest.raises(KeyboardInterrupt), _script_child(_READY_THEN_IDLE) as child:
        _await_ready(child)
        raise KeyboardInterrupt
    assert child.popen.poll() is not None
