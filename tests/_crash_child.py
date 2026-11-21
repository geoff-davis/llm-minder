"""Child process for the crash-injection harness (#328).

Usage:

- ``python -m tests._crash_child <point-id> <store-path> <workdir> [<params-json>]``: install one
  kill point from ``_crash_points.POINTS``, run the workload, and exit 0 if the point never fired;
  the parent treats exit 0 as "kill point not reached" and fails. ``params`` carries the slow
  sweep's case parameters (record index, byte cut, publication branch).
- ``python -m tests._crash_child timed <store-kind> <store-path> <workdir>``: no kill point; print
  ``READY`` once the store is prepared and ``DONE`` once the run and close finish, for the parent's
  timed external kill.

Published results are logged to ``<workdir>/published.log`` synchronously at the publication
boundary, on whichever branch the run publishes through: the streaming ``_publish_stream_result``
(the default) or, with ``{"branch": "list"}``, the non-streaming ``_results.append``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from llm_minder import (
    ArtifactIdentity,
    JsonlArtifactStore,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
)
from llm_minder.testing import FakeStrategy
from tests._crash_points import (
    ITEMS,
    POINTS,
    REPAIR_SETUP_ITEMS,
    SQLITE_STORE_KWARGS,
    TIMED_ITEMS,
    TIMED_LATENCY_SECONDS,
    BatchWatch,
    ChildContext,
    SideLog,
    kill_self,
    output_for,
    record_publication,
)

IDENTITY = ArtifactIdentity(provider="p", model="m")


def make_store(kind: str, path: Path) -> Any:
    if kind == "jsonl":
        return JsonlArtifactStore(path, identity=IDENTITY)
    return SqliteArtifactStore(path, identity=IDENTITY, **SQLITE_STORE_KWARGS)


class _PublishingList(list[Any]):
    """The non-streaming result list; logs each result on entry to ``append``, before adding it."""

    def __init__(self, ctx: ChildContext, items: Any = ()) -> None:
        super().__init__(items)
        self._ctx = ctx

    def append(self, result: Any) -> None:
        record_publication(self._ctx, result.item_id)
        super().append(result)


def _list_publishing_processor(ctx: ChildContext) -> Any:
    """A processor whose ``_results`` list, reset at each run start, logs publications."""

    class Processor(ParallelBatchProcessor[Any, str, None]):
        @property  # type: ignore[override]
        def _results(self) -> list[Any]:
            return self.__dict__["_crash_results"]  # type: ignore[no-any-return]

        @_results.setter
        def _results(self, value: list[Any]) -> None:
            self.__dict__["_crash_results"] = _PublishingList(ctx, value)

    return Processor


async def run_processor(
    ctx: ChildContext,
    items: list[tuple[str, str]] = ITEMS,
    *,
    latency: float = 0.0,
    store: Any = None,
) -> None:
    store = ctx.store if store is None else store
    strategy: FakeStrategy[str] = FakeStrategy(response=output_for, latency=latency)
    branch = ctx.params.get("branch", "stream")
    work = [LLMWorkItem(item_id=i, strategy=strategy, prompt=p) for i, p in items]

    if branch == "list":
        processor = _list_publishing_processor(ctx)(
            config=ProcessorConfig(max_workers=3),
            artifact_store=store,
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        async with processor:
            for item in work:
                await processor.add_work(item)
            await processor.process_all()
        return

    assert branch == "stream", branch
    processor = ParallelBatchProcessor[Any, str, None](
        config=ProcessorConfig(max_workers=3),
        artifact_store=store,
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    publish = processor._publish_stream_result

    async def logged_publish(result: Any) -> None:
        # The publication boundary: log synchronously on entry, before publishing.
        record_publication(ctx, result.item_id)
        await publish(result)

    processor._publish_stream_result = logged_publish  # type: ignore[method-assign]

    async def feed() -> None:
        for item in work:
            await processor.add_work(item)
        await processor.finish()

    async with processor:
        processor.start()
        producer = asyncio.create_task(feed())
        async for _ in processor.results():
            pass
        await producer


async def run_store(ctx: ChildContext) -> None:
    await ctx.store._prepare()
    await ctx.store.close()


async def run_repair_record(ctx: ChildContext) -> None:
    """Complete part of a run, tear a record onto the end, then prepare the store again."""
    await run_processor(ctx, ITEMS[:REPAIR_SETUP_ITEMS])
    raw = ctx.path.read_bytes()
    last = raw[raw.rstrip(b"\n").rfind(b"\n") + 1 :]  # the final complete line, with its newline
    frac = float(ctx.params.get("cut_frac", 0.5))
    cut = min(max(1, int(frac * len(last))), len(last) - 1)
    with ctx.path.open("ab") as handle:
        handle.write(last[:cut])
    await run_store_fresh(ctx)


async def run_repair_manifest(ctx: ChildContext) -> None:
    """Leave a torn manifest prefix, then prepare the store again."""
    prefix = b'{"artifact_schema_version":1,"created_at":'
    ctx.path.write_bytes(prefix[: int(ctx.params.get("cut", 20))])
    await run_store_fresh(ctx)


async def run_store_fresh(ctx: ChildContext) -> None:
    # The repair points are JSONL-only: a fresh store instance reopens the torn file.
    store = make_store("jsonl", ctx.path)
    await store._prepare()
    await store.close()


WORKLOADS = {
    "processor": run_processor,
    "store": run_store,
    "repair-record": run_repair_record,
    "repair-manifest": run_repair_manifest,
}


async def run_timed(ctx: ChildContext) -> None:
    await ctx.store._prepare()
    print("READY", flush=True)
    await run_processor(ctx, TIMED_ITEMS, latency=TIMED_LATENCY_SECONDS)
    print("DONE", flush=True)


def _context(raw_path: str, raw_workdir: str, params: dict[str, Any]) -> ChildContext:
    workdir = Path(raw_workdir)
    return ChildContext(
        path=Path(raw_path),
        workdir=workdir,
        published_log=SideLog(workdir / "published.log"),
        batch_log=SideLog(workdir / "batch.log"),
        batch_watch=BatchWatch(),
        params=params,
    )


def main() -> int:
    if sys.argv[1] == "timed":
        kind, raw_path, raw_workdir = sys.argv[2:5]
        ctx = _context(raw_path, raw_workdir, {})
        ctx.store = make_store(kind, ctx.path)
        asyncio.run(run_timed(ctx))
        return 0

    point_id, raw_path, raw_workdir = sys.argv[1:4]
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    point = POINTS[point_id]
    ctx = _context(raw_path, raw_workdir, params)
    # Patch module-level entry points (open, sqlite3.connect) before the store exists, and the
    # store instance after.
    if point.patches_store:
        ctx.store = make_store(point.store, ctx.path)
        point.installer(ctx, kill_self)
    else:
        point.installer(ctx, kill_self)
        ctx.store = make_store(point.store, ctx.path)
    asyncio.run(WORKLOADS[point.layer](ctx))
    print(f"kill point {point_id} not reached (fired={ctx.fired})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
