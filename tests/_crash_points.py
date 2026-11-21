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
import signal
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Action = Callable[[], None]

#: The workload: 12 items, prompts with a two-byte UTF-8 character so a byte cut can land inside it.
ITEMS: list[tuple[str, str]] = [(f"item-{i:02d}", f"prompt-ü-{i}") for i in range(12)]
PROMPT_BY_ID = dict(ITEMS)
ID_BY_PROMPT = {prompt: item_id for item_id, prompt in ITEMS}

#: SQLite batching that makes a batch really hold more than one row.
SQLITE_STORE_KWARGS: dict[str, Any] = {"commit_batch_size": 4, "commit_interval_seconds": 0.05}

#: Bound on how long the mid-batch point waits for a premature publication before killing.
BATCH_EVENT_WAIT_SECONDS = 0.5

#: The JSONL record index (counting the manifest as write 1) that the mid-record point tears.
JSONL_TORN_WRITE = 6
#: The append call after which ``jsonl-after-append`` kills.
JSONL_KILL_AFTER_APPEND = 5


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


def install_jsonl_create_torn_manifest(ctx: ChildContext, action: Action) -> None:
    _install_jsonl_tear(ctx, action, index=1, cut=lambda data: 20)


def install_jsonl_mid_record(ctx: ChildContext, action: Action) -> None:
    _install_jsonl_tear(ctx, action, index=JSONL_TORN_WRITE, cut=_cut_inside_multibyte)


def install_jsonl_after_append(ctx: ChildContext, action: Action) -> None:
    store = ctx.store
    original = store.append
    calls = 0

    async def append(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        value = await original(*args, **kwargs)
        calls += 1
        if calls == JSONL_KILL_AFTER_APPEND:
            ctx.fired = True
            action()
        return value

    store.append = append


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
    """
    watch = ctx.batch_watch

    class Connection(sqlite3.Connection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            super().execute("PRAGMA cache_size=1")

        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            stripped = sql.lstrip()
            if stripped.startswith("BEGIN IMMEDIATE"):
                watch.begin()
            elif stripped.startswith("INSERT INTO item_records"):
                watch.add(args[0][2])  # the item_id parameter
            return super().execute(sql, *args)

        def commit(self) -> None:
            if not ctx.fired and len(watch.ids) > 1:
                watch.event.wait(BATCH_EVENT_WAIT_SECONDS)
                for item_id in sorted(watch.ids):
                    ctx.batch_log.write(item_id)
                ctx.fired = True
                action()
            super().commit()

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


@dataclass(frozen=True)
class CrashPoint:
    id: str
    store: str  # "jsonl" or "sqlite"
    layer: str  # "processor" (real batch run) or "store" (prepare/close only)
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
        ),
        CrashPoint(
            "jsonl-mid-record",
            "jsonl",
            "processor",
            install_jsonl_mid_record,
            Inspection(),
            "record write cut inside a two-byte UTF-8 character",
            torn_tail=True,
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
            "sqlite-mid-batch",
            "sqlite",
            "processor",
            install_sqlite_mid_batch,
            Inspection(),
            "at commit() of a multi-row batch, after its inserts",
            killed_batch=True,
        ),
        CrashPoint(
            "sqlite-close-before-checkpoint",
            "sqlite",
            "processor",
            install_sqlite_close_before_checkpoint,
            Inspection(),
            "close, at PRAGMA wal_checkpoint(TRUNCATE)",
        ),
    )
}
