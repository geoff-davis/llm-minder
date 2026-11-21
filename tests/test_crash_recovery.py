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
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
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
    ID_BY_PROMPT,
    ITEMS,
    POINTS,
    PROMPT_BY_ID,
    SQLITE_STORE_KWARGS,
    CrashPoint,
    read_side_log,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the harness kills the child with SIGKILL"
)

REPO = Path(__file__).resolve().parents[1]
IDENTITY = ArtifactIdentity(provider="p", model="m")


def _make_store(point: CrashPoint, path: Path) -> Any:
    if point.store == "jsonl":
        return JsonlArtifactStore(path, identity=IDENTITY)
    return SqliteArtifactStore(path, identity=IDENTITY, **SQLITE_STORE_KWARGS)


async def _read(point: CrashPoint, path: Path) -> Any:
    if point.store == "jsonl":
        return JsonlArtifactStore.read_results(path)
    return await SqliteArtifactStore.read_results(path)


def _run_child(point: CrashPoint, path: Path, workdir: Path) -> subprocess.CompletedProcess[str]:
    # The child imports the same checkout: an inherited PYTHONPATH (a probe's src/ copy) comes first,
    # then this checkout's src/ and root (for ``tests._crash_points``).
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (env.get("PYTHONPATH"), os.fspath(REPO / "src"), os.fspath(REPO)) if part
    )
    return subprocess.run(
        [sys.executable, "-m", "tests._crash_child", point.id, os.fspath(path), os.fspath(workdir)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("point_id", sorted(POINTS))
async def test_crash_point_recovers(
    point_id: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    point = POINTS[point_id]
    path = tmp_path / ("store.jsonl" if point.store == "jsonl" else "store.sqlite")
    workdir = tmp_path / "work"
    workdir.mkdir()

    # 1. The child died at its point.
    child = _run_child(point, path, workdir)
    context = f"point {point_id}; child stderr:\n{child.stderr}"
    assert child.returncode == -signal.SIGKILL, f"invariant 1: {context}"
    assert "Traceback" not in child.stderr, context

    published = read_side_log(workdir / "published.log")
    killed_batch = read_side_log(workdir / "batch.log")

    # 2. Read-only inspection before recovery: exactly the allowed outcome, file unchanged.
    if point.hot_journal:
        assert Path(f"{path}-journal").exists(), context
    before = path.read_bytes() if path.exists() else None
    expected = point.inspection
    if expected.error is None:
        await _read(point, path)
    else:
        with pytest.raises(ArtifactError) as raised:
            await _read(point, path)
        assert type(raised.value).__name__ == expected.error, repr(raised.value)
        assert expected.match in str(raised.value), repr(raised.value)
    after = path.read_bytes() if path.exists() else None
    assert after == before, "read-only inspection changed the store file"

    # 3. Writable reopen succeeds; afterwards inspection returns.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        store = _make_store(point, path)
        try:
            try:
                await store._prepare()
            finally:
                await store.close()
        except ArtifactError as exc:
            pytest.fail(f"invariant 3: writable reopen failed: {exc!r}")
    dropped = [r for r in caplog.records if "Dropped" in r.getMessage()]
    stored = await _read(point, path)

    # 4. Nothing published is lost.
    retained_successes = {r.item_id for r in stored.results if r.success}
    assert set(published) <= retained_successes, (
        f"invariant 4: published but not retained: {sorted(set(published) - retained_successes)}"
    )

    # 5. Nothing partial is retained.
    for result in stored.results:
        assert result.success, result
        assert result.output == f"out:{PROMPT_BY_ID[result.item_id]}", result
    if point.store == "jsonl":
        # Complete lines only, as the store reads them: a final segment without a newline is a torn
        # tail the store ignores (and may not even be valid UTF-8).
        complete = path.read_bytes().split(b"\n")[:-1]
        sequences = [json.loads(line)["record_sequence"] for line in complete[1:]]
        assert len(sequences) == len(set(sequences)), sequences
    if point.killed_batch:
        assert len(killed_batch) > 1, killed_batch
        retained_ids = {r.item_id for r in stored.results}
        assert not retained_ids & set(killed_batch), "part of the killed batch was retained"
    if point.retained_unpublished:
        assert retained_successes - set(published), "the cut was not before publication"
    if point.torn_tail:
        assert len(dropped) == 1, f"invariant 5: {[r.getMessage() for r in dropped]}"
    else:
        assert dropped == [], [r.getMessage() for r in dropped]

    # 6. The resume finishes exactly once.
    executed: list[str] = []

    def respond(prompt: str) -> str:
        executed.append(ID_BY_PROMPT[prompt])
        return f"out:{prompt}"

    strategy: FakeStrategy[str] = FakeStrategy(response=respond)
    resumed = await process_prompts(
        strategy,
        ITEMS,
        config=ProcessorConfig(max_workers=3),
        artifact_store=_make_store(point, path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert resumed.succeeded == len(ITEMS)
    replayed = {r.item_id for r in resumed.results if r.replayed_from_artifact}
    assert replayed == retained_successes
    assert Counter(executed) == Counter(
        item_id for item_id, _ in ITEMS if item_id not in retained_successes
    )
    final = await _read(point, path)
    per_item = Counter(r.item_id for r in final.results if r.success)
    assert per_item == Counter(item_id for item_id, _ in ITEMS), f"invariant 6: {per_item}"

    # 7. A clean close after the resume checkpoints the WAL away.
    if point.store == "sqlite":
        wal = Path(f"{path}-wal")
        assert not wal.exists() or wal.stat().st_size == 0
