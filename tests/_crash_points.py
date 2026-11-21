"""Named fault points inside the artifact stores, for the crash-injection harness (#328).

Each point is reached by test-side monkeypatching; there is no production hook. An installer patches one
point and runs an *action* when the point is reached. The crash harness passes :func:`kill_self`; the
in-process close-failure tests planned for #351 (G13/G16) can pass :func:`raise_once` or a cancellation
action instead, reusing the same knowledge of which private method each store goes through.

The installers run inside the child process (``tests/_crash_child.py``). The parent
(``tests/test_crash_recovery.py``) only reads the table: which store, which workload layer, and the
outcome read-only inspection is allowed to have before recovery (invariant 2 of the design note).
"""

from __future__ import annotations

import os
import pathlib
import random
import signal
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Action = Callable[[], None]

#: The workload: 12 items, prompts with a two-byte UTF-8 character so a byte cut can land inside it.
ITEMS: list[tuple[str, str]] = [(f"item-{i:02d}", f"prompt-ü-{i}") for i in range(12)]
#: The timed external-kill workload (design note section 5): longer, so a random delay lands mid-run.
TIMED_ITEMS: list[tuple[str, str]] = [(f"timed-{i:03d}", f"timed-ü-{i}") for i in range(200)]
TIMED_LATENCY_SECONDS = 0.001
PROMPT_BY_ID = dict(ITEMS + TIMED_ITEMS)
ID_BY_PROMPT = {prompt: item_id for item_id, prompt in ITEMS + TIMED_ITEMS}

#: SQLite batching that makes a batch really hold more than one row.
SQLITE_STORE_KWARGS: dict[str, Any] = {"commit_batch_size": 4, "commit_interval_seconds": 0.05}

#: Bound on how long the mid-batch point waits for a premature publication before killing.
BATCH_EVENT_WAIT_SECONDS = 0.5

#: The JSONL record index (counting the manifest as write 1) that the mid-record point tears.
JSONL_TORN_WRITE = 6
#: The append call after which ``jsonl-after-append`` kills.
JSONL_KILL_AFTER_APPEND = 5
#: The publication after whose log entry the ``*-after-publish`` points kill.
KILL_AFTER_PUBLISH = 5
#: The committed record batch after which ``sqlite-after-commit`` kills.
SQLITE_KILL_AFTER_COMMIT = 2
#: How many items the repair points' setup run completes before the file is torn.
REPAIR_SETUP_ITEMS = 6


def output_for(prompt: str) -> str:
    return f"out:{prompt}"


# ── Actions ────────────────────────────────────────────────────────────────


def kill_self() -> None:
    """SIGKILL this process: no ``finally``, no ``atexit``, no buffered flush."""
    os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(60)  # pragma: no cover - the signal is delivered before this returns


def raise_once(exc: BaseException) -> Action:
    """An action that raises ``exc`` the first time and does nothing afterwards (#351 G13)."""
    fired = False

    def action() -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise exc

    return action


# ── Side channels written by the child ─────────────────────────────────────


class SideLog:
    """An append-only side file written with raw ``os.write``.

    No ``fsync``: data that ``write(2)`` accepted survives SIGKILL (the page cache survives the
    process), which is all the harness claims.
    """

    def __init__(self, path: Path) -> None:
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def write(self, line: str) -> None:
        os.write(self._fd, f"{line}\n".encode())


def read_side_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


# ── Mid-batch coordination ─────────────────────────────────────────────────


class BatchWatch:
    """Tracks the item ids of the SQLite batch in flight and whether any was published early.

    The connection hook fills ``ids`` from the batch's inserts (writer thread); the publication
    wrapper calls :meth:`published` (event-loop thread). On correct code nothing in a batch can be
    published before its commit, so the hook's bounded wait times out.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ids: set[str] = set()
        self.event = threading.Event()

    def begin(self) -> None:
        with self._lock:
            self.ids = set()
            self.event.clear()

    def add(self, item_id: str) -> None:
        with self._lock:
            self.ids.add(item_id)

    def published(self, item_id: str) -> None:
        with self._lock:
            if item_id in self.ids:
                self.event.set()


@dataclass
class ChildContext:
    """What an installer may need from the child."""

    path: Path
    workdir: Path
    published_log: SideLog
    batch_log: SideLog
    batch_watch: BatchWatch
    store: Any = None  # set before installers that patch the store instance run
    fired: bool = False
    #: Case parameters from the parent (the slow sweep's random k and byte cuts, the branch).
    params: dict[str, Any] = field(default_factory=dict)
    #: Called with the publication count right after each publication's log entry.
    after_publish: Callable[[int], None] | None = None
    published_count: int = 0


def record_publication(ctx: ChildContext, item_id: str) -> None:
    """The publication boundary: log ``item_id`` synchronously, before the result is published.

    Both publication branches call this: the streaming ``_publish_stream_result`` wrapper and the
    non-streaming ``_results.append`` wrapper (``tests/_crash_child.py``).
    """
    ctx.published_log.write(item_id)
    ctx.batch_watch.published(item_id)
    ctx.published_count += 1
    if ctx.after_publish is not None:
        ctx.after_publish(ctx.published_count)


# ── Installers ─────────────────────────────────────────────────────────────


class _TornWriteProxy:
    """Wraps the JSONL append handle; on write number ``index`` writes ``cut(data)`` bytes, then acts."""

    def __init__(
        self,
        handle: Any,
        index: int,
        cut: Callable[[bytes], int],
        action: Action,
        ctx: ChildContext,
    ) -> None:
        self._handle = handle
        self._index = index
        self._cut = cut
        self._action = action
        self._ctx = ctx
        self._writes = 0

    def write(self, text: str) -> int:
        self._writes += 1
        if self._writes != self._index:
            return int(self._handle.write(text))
        self._handle.flush()
        data = text.encode("utf-8")
        cut = self._cut(data)
        assert 0 < cut < len(data), (cut, len(data))
        self._handle.buffer.write(data[:cut])
        self._handle.buffer.flush()
        self._ctx.fired = True
        self._action()
        return len(text)  # pragma: no cover - only reached by a non-killing action

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def _install_jsonl_tear(
    ctx: ChildContext, action: Action, index: int, cut: Callable[[bytes], int]
) -> None:
    real_open = pathlib.Path.open
    target = ctx.path.resolve()

    def patched_open(self: pathlib.Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, mode, *args, **kwargs)
        if mode == "a" and self.resolve() == target:
            return _TornWriteProxy(handle, index, cut, action, ctx)
        return handle

    pathlib.Path.open = patched_open  # type: ignore[method-assign]


def _cut_inside_multibyte(data: bytes) -> int:
    # Cut between the two bytes of the first "ü" (0xC3 0xBC) in the record's output.
    return data.index(b"\xc3\xbc") + 1


def _cut_from_params(ctx: ChildContext, default: Callable[[bytes], int]) -> Callable[[bytes], int]:
    """The byte cut for a tear: ``cut`` (negative counts from the end), ``cut_frac``, or ``default``.

    A ``cut_frac`` in (0, 1) scales with the line, which the parent cannot know in advance; the
    result is clamped so at least one byte is written and at least one is withheld.
    """
    if "cut" in ctx.params:
        cut = int(ctx.params["cut"])
        return lambda data: cut if cut > 0 else len(data) + cut
    if "cut_frac" in ctx.params:
        frac = float(ctx.params["cut_frac"])
        return lambda data: min(max(1, int(frac * len(data))), len(data) - 1)
    return default


def install_jsonl_create_torn_manifest(ctx: ChildContext, action: Action) -> None:
    _install_jsonl_tear(ctx, action, index=1, cut=_cut_from_params(ctx, lambda data: 20))


def install_jsonl_mid_record(ctx: ChildContext, action: Action) -> None:
    index = int(ctx.params.get("write", JSONL_TORN_WRITE))
    _install_jsonl_tear(ctx, action, index=index, cut=_cut_from_params(ctx, _cut_inside_multibyte))


def install_jsonl_record_no_newline(ctx: ChildContext, action: Action) -> None:
    # The whole line except its "\n": complete JSON that the reader must still drop.
    index = int(ctx.params.get("write", JSONL_TORN_WRITE))
    _install_jsonl_tear(ctx, action, index=index, cut=lambda data: len(data) - 1)


def install_jsonl_after_append(ctx: ChildContext, action: Action) -> None:
    store = ctx.store
    original = store.append
    kill_at = int(ctx.params.get("k", JSONL_KILL_AFTER_APPEND))
    calls = 0

    async def append(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        value = await original(*args, **kwargs)
        calls += 1
        if calls == kill_at:
            ctx.fired = True
            action()
        return value

    store.append = append


def install_after_publish(ctx: ChildContext, action: Action) -> None:
    """Act right after the k-th publication's log entry, before the result is published."""
    kill_at = int(ctx.params.get("k", KILL_AFTER_PUBLISH))

    def hook(count: int) -> None:
        if count == kill_at:
            ctx.fired = True
            action()

    ctx.after_publish = hook


def install_jsonl_repair_then_die(ctx: ChildContext, action: Action) -> None:
    """Act after writable prepare truncated a torn tail, before the manifest or next record."""
    import llm_minder.artifacts as module

    original = module._truncate_artifact_tail

    def truncate(path: Path, snapshot: Any, *, fsync: bool) -> None:
        original(path, snapshot, fsync=fsync)
        if snapshot.tail:
            ctx.fired = True
            action()

    setattr(module, "_truncate_artifact_tail", truncate)  # noqa: B010 - test-side patch


def install_jsonl_during_close(ctx: ChildContext, action: Action) -> None:
    store = ctx.store
    original = store._close_sync

    def close_sync(handle: Any) -> None:
        ctx.fired = True
        action()
        original(handle)  # pragma: no cover - only reached by a non-killing action

    store._close_sync = close_sync


def _install_sqlite_factory(factory: type[sqlite3.Connection]) -> None:
    import llm_minder.sqlite_artifacts as module

    real_connect = sqlite3.connect

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return real_connect(*args, factory=factory, **kwargs)

    setattr(module.sqlite3, "connect", connect)  # noqa: B010 - test-side patch of a typed overload


def install_sqlite_init_mid_txn(ctx: ChildContext, action: Action) -> None:
    class Connection(sqlite3.Connection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # One cached page spills initialization pages into the file before commit, leaving a
            # non-empty file and a hot journal, as a real crash mid-transaction can.
            super().execute("PRAGMA cache_size=1")

        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            if sql.startswith("PRAGMA user_version="):
                ctx.fired = True
                action()
            return super().execute(sql, *args)

    _install_sqlite_factory(Connection)


def install_sqlite_mid_batch(ctx: ChildContext, action: Action) -> None:
    """Act at the ``commit()`` of the first batch holding more than one row, after its inserts.

    Before acting, wait up to ``BATCH_EVENT_WAIT_SECONDS`` for any item of this batch to be
    published. Correct code resolves append futures only after the commit, so the wait times out
    and the kill finds nothing of the batch published. Code that resolves them before ``commit()``
    publishes during the wait and dies with a published-but-uncommitted item.

    With the ``insert`` parameter (the slow sweep's random k), act instead before the k-th
    ``INSERT INTO item_records`` of the run, wherever it falls in its batch; the batch ids logged
    then include that item, and none of them may be retained.
    """
    watch = ctx.batch_watch
    at_insert = ctx.params.get("insert")
    inserts = 0

    class Connection(sqlite3.Connection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            super().execute("PRAGMA cache_size=1")

        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            nonlocal inserts
            stripped = sql.lstrip()
            if stripped.startswith("BEGIN IMMEDIATE"):
                watch.begin()
            elif stripped.startswith("INSERT INTO item_records"):
                watch.add(args[0][2])  # the item_id parameter
                inserts += 1
                if at_insert is not None and inserts == int(at_insert) and not ctx.fired:
                    for item_id in sorted(watch.ids):
                        ctx.batch_log.write(item_id)
                    ctx.fired = True
                    action()
            return super().execute(sql, *args)

        def commit(self) -> None:
            if at_insert is None and not ctx.fired and len(watch.ids) > 1:
                watch.event.wait(BATCH_EVENT_WAIT_SECONDS)
                for item_id in sorted(watch.ids):
                    ctx.batch_log.write(item_id)
                ctx.fired = True
                action()
            super().commit()

    _install_sqlite_factory(Connection)


def _act_on_statement(ctx: ChildContext, action: Action, prefix: str) -> None:
    """Act on entry to the first statement starting with ``prefix``."""

    class Connection(sqlite3.Connection):
        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            if not ctx.fired and sql.lstrip().startswith(prefix):
                ctx.fired = True
                action()
            return super().execute(sql, *args)

    _install_sqlite_factory(Connection)


def install_sqlite_create_empty(ctx: ChildContext, action: Action) -> None:
    # The first statement after connect() is the busy_timeout pragma; the file exists, empty.
    _act_on_statement(ctx, action, "")


def install_sqlite_init_after_commit(ctx: ChildContext, action: Action) -> None:
    _act_on_statement(ctx, action, "PRAGMA journal_mode=WAL")


def install_sqlite_before_identity(ctx: ChildContext, action: Action) -> None:
    _act_on_statement(ctx, action, "INSERT INTO identities")


def install_sqlite_after_commit(ctx: ChildContext, action: Action) -> None:
    """Act when the k-th record batch's ``commit()`` returns, before its futures resolve."""
    kill_at = int(ctx.params.get("k", SQLITE_KILL_AFTER_COMMIT))
    inserts = 0
    batches = 0

    class Connection(sqlite3.Connection):
        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            nonlocal inserts
            stripped = sql.lstrip()
            if stripped.startswith("BEGIN IMMEDIATE"):
                inserts = 0
            elif stripped.startswith("INSERT INTO item_records"):
                inserts += 1
            return super().execute(sql, *args)

        def commit(self) -> None:
            nonlocal batches
            super().commit()
            if inserts:  # a record batch, not the schema initialization
                batches += 1
                if batches == kill_at:
                    ctx.fired = True
                    action()

    _install_sqlite_factory(Connection)


def install_sqlite_close_after_checkpoint(ctx: ChildContext, action: Action) -> None:
    """Act on entry to ``connection.close()`` once the close checkpoint has run."""
    checkpointed = False

    class Connection(sqlite3.Connection):
        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            nonlocal checkpointed
            cursor = super().execute(sql, *args)
            # execute() steps the pragma once, which runs the whole checkpoint.
            if sql.lstrip().startswith("PRAGMA wal_checkpoint"):
                checkpointed = True
            return cursor

        def close(self) -> None:
            if checkpointed:
                ctx.fired = True
                action()
            super().close()

    _install_sqlite_factory(Connection)


def install_sqlite_close_before_checkpoint(ctx: ChildContext, action: Action) -> None:
    class Connection(sqlite3.Connection):
        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            if sql.lstrip().startswith("PRAGMA wal_checkpoint"):
                ctx.fired = True
                action()
            return super().execute(sql, *args)

    _install_sqlite_factory(Connection)


# ── The point table ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Inspection:
    """The outcome read-only inspection may have before recovery (invariant 2)."""

    error: str | None = None  # exception class name, or None when inspection must return
    match: str = ""


Params = dict[str, Any]


def _random_write(rng: random.Random) -> Params:
    # Writes 2..13 are the 12 records (write 1 is the manifest).
    return {"write": rng.randint(2, len(ITEMS) + 1)}


def _random_tear(rng: random.Random) -> Params:
    return {**_random_write(rng), "cut_frac": rng.random()}


def _random_k(rng: random.Random) -> Params:
    return {"k": rng.randint(1, len(ITEMS))}


def _random_manifest_cut(rng: random.Random) -> Params:
    return {"cut_frac": rng.random()}


def _random_insert(rng: random.Random) -> Params:
    return {"insert": rng.randint(1, len(ITEMS))}


def _random_commit(rng: random.Random) -> Params:
    # 12 rows in batches of at most 4 make at least 3 record batches.
    return {"k": rng.randint(1, 3)}


def _random_repair_cut(rng: random.Random) -> Params:
    return {"cut_frac": rng.random()}


def _random_manifest_prefix(rng: random.Random) -> Params:
    # Every leading slice of the canonical 42-byte manifest prefix is a torn manifest.
    return {"cut": rng.randint(1, 42)}


@dataclass(frozen=True)
class CrashPoint:
    id: str
    store: str  # "jsonl" or "sqlite"
    # "processor" (real batch run), "store" (prepare/close only), or a repair setup:
    # "repair-record" (a completed partial run, then a torn record) or "repair-manifest"
    # (a torn manifest prefix), each followed by a writable prepare.
    layer: str
    installer: Callable[[ChildContext, Action], None]
    inspection: Inspection
    description: str
    torn_tail: bool = False  # writable reopen must log exactly one "Dropped ... bytes" warning
    hot_journal: bool = False
    patches_store: bool = (
        False  # the installer patches the store instance, so it runs after construction
    )
    # Point-specific evidence that the cut landed where the description says:
    retained_unpublished: bool = (
        False  # some retained success was never published (cut before publication)
    )
    killed_batch: bool = (
        False  # the child logged a batch of more than one item that the kill rolled back
    )
    published_at_kill: bool = False  # the k-th publication was logged; the kill followed it
    complete_json_tail: bool = False  # the torn tail is complete JSON missing only its newline
    repaired_at_kill: bool = False  # the child logged the repair and the file has no torn tail
    all_retained: bool = False  # every item was checkpointed before the kill
    empty_file: bool = False  # the kill left a 0-byte store file
    rollback_journal: bool = False  # the database is initialized but not yet in WAL mode
    identity_missing: bool = False  # the database is initialized but holds no identity row
    wal_truncated_at_kill: bool = False  # the close checkpoint truncated the WAL before the kill
    #: Random case parameters for the slow sweep, or None when the point has nothing to vary.
    random_params: Callable[[random.Random], Params] | None = None


POINTS: dict[str, CrashPoint] = {
    point.id: point
    for point in (
        CrashPoint(
            "jsonl-create-torn-manifest",
            "jsonl",
            "store",
            install_jsonl_create_torn_manifest,
            Inspection("ArtifactFormatError", "manifest"),
            "first manifest write cut after 20 bytes",
            torn_tail=True,
            random_params=_random_manifest_cut,
        ),
        CrashPoint(
            "jsonl-mid-record",
            "jsonl",
            "processor",
            install_jsonl_mid_record,
            Inspection(),
            "record write cut inside a two-byte UTF-8 character",
            torn_tail=True,
            random_params=_random_tear,
        ),
        CrashPoint(
            "jsonl-record-no-newline",
            "jsonl",
            "processor",
            install_jsonl_record_no_newline,
            Inspection(),
            "record write cut just before its newline",
            torn_tail=True,
            complete_json_tail=True,
            random_params=_random_write,
        ),
        CrashPoint(
            "jsonl-after-append",
            "jsonl",
            "processor",
            install_jsonl_after_append,
            Inspection(),
            "after append() returns, before publication",
            patches_store=True,
            retained_unpublished=True,
            random_params=_random_k,
        ),
        CrashPoint(
            "jsonl-after-publish",
            "jsonl",
            "processor",
            install_after_publish,
            Inspection(),
            "after the k-th publication's log entry, before the result is published",
            published_at_kill=True,
            random_params=_random_k,
        ),
        CrashPoint(
            "jsonl-repair-then-die",
            "jsonl",
            "repair-record",
            install_jsonl_repair_then_die,
            Inspection(),
            "reopen of a torn record: after the tail is truncated, before the next write",
            repaired_at_kill=True,
            random_params=_random_repair_cut,
        ),
        CrashPoint(
            "jsonl-repair-then-die-manifest",
            "jsonl",
            "repair-manifest",
            install_jsonl_repair_then_die,
            # Truncating a torn manifest leaves a 0-byte file (hypothesis H1).
            Inspection("ArtifactFormatError", "empty"),
            "reopen of a torn manifest: after the tail is truncated, before the manifest write",
            repaired_at_kill=True,
            empty_file=True,
            random_params=_random_manifest_prefix,
        ),
        CrashPoint(
            "jsonl-during-close",
            "jsonl",
            "processor",
            install_jsonl_during_close,
            Inspection(),
            "on entry to the store's _close_sync",
            patches_store=True,
            all_retained=True,
        ),
        CrashPoint(
            "sqlite-create-empty",
            "sqlite",
            "store",
            install_sqlite_create_empty,
            Inspection("ArtifactFormatError", ""),
            "after connect(), on the first statement",
            empty_file=True,
        ),
        CrashPoint(
            "sqlite-init-mid-txn",
            "sqlite",
            "store",
            install_sqlite_init_mid_txn,
            Inspection("ArtifactIOError", "attempt to write a readonly database"),
            "inside the initialization transaction, before PRAGMA user_version",
            hot_journal=True,
        ),
        CrashPoint(
            "sqlite-init-after-commit",
            "sqlite",
            "store",
            install_sqlite_init_after_commit,
            Inspection(),
            "after the initialization commit, at PRAGMA journal_mode=WAL",
            rollback_journal=True,
        ),
        CrashPoint(
            "sqlite-before-identity",
            "sqlite",
            "store",
            install_sqlite_before_identity,
            Inspection(),
            "after WAL and the pragmas, at the identity insert",
            identity_missing=True,
        ),
        CrashPoint(
            "sqlite-mid-batch",
            "sqlite",
            "processor",
            install_sqlite_mid_batch,
            Inspection(),
            "at commit() of a multi-row batch, after its inserts",
            killed_batch=True,
            random_params=_random_insert,
        ),
        CrashPoint(
            "sqlite-after-commit",
            "sqlite",
            "processor",
            install_sqlite_after_commit,
            Inspection(),
            "after the k-th record batch's commit() returns, before its futures resolve",
            retained_unpublished=True,
            random_params=_random_commit,
        ),
        CrashPoint(
            "sqlite-after-publish",
            "sqlite",
            "processor",
            install_after_publish,
            Inspection(),
            "after the k-th publication's log entry, before the result is published",
            published_at_kill=True,
            random_params=_random_k,
        ),
        CrashPoint(
            "sqlite-close-before-checkpoint",
            "sqlite",
            "processor",
            install_sqlite_close_before_checkpoint,
            Inspection(),
            "close, at PRAGMA wal_checkpoint(TRUNCATE)",
        ),
        CrashPoint(
            "sqlite-close-after-checkpoint",
            "sqlite",
            "processor",
            install_sqlite_close_after_checkpoint,
            Inspection(),
            "close, after the checkpoint, on entry to connection.close()",
            wal_truncated_at_kill=True,
        ),
    )
}

#: The default (not slow) set agreed in the design note: one point per phase.
DEFAULT_POINTS: tuple[str, ...] = (
    "jsonl-create-torn-manifest",
    "jsonl-mid-record",
    "jsonl-after-append",
    "sqlite-init-mid-txn",
    "sqlite-mid-batch",
    "sqlite-close-before-checkpoint",
)
