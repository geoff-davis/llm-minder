"""Child process for the crash-injection harness (#328).

Usage: ``python -m tests._crash_child <point-id> <store-path> <workdir>``

Installs one kill point from ``_crash_points.POINTS``, runs the workload, and exits 0 if the point
never fired; the parent treats exit 0 as "kill point not reached" and fails. Published results are
logged to ``<workdir>/published.log`` synchronously at the publication boundary.
"""

from __future__ import annotations

import asyncio
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
    SQLITE_STORE_KWARGS,
    BatchWatch,
    ChildContext,
    SideLog,
    kill_self,
    output_for,
)

IDENTITY = ArtifactIdentity(provider="p", model="m")


def make_store(kind: str, path: Path) -> Any:
    if kind == "jsonl":
        return JsonlArtifactStore(path, identity=IDENTITY)
    return SqliteArtifactStore(path, identity=IDENTITY, **SQLITE_STORE_KWARGS)


async def run_processor(ctx: ChildContext) -> None:
    strategy: FakeStrategy[str] = FakeStrategy(response=output_for)
    processor = ParallelBatchProcessor[Any, str, None](
        config=ProcessorConfig(max_workers=3),
        artifact_store=ctx.store,
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    publish = processor._publish_stream_result

    async def logged_publish(result: Any) -> None:
        # The publication boundary: log synchronously on entry, before publishing.
        ctx.published_log.write(result.item_id)
        ctx.batch_watch.published(result.item_id)
        await publish(result)

    processor._publish_stream_result = logged_publish  # type: ignore[method-assign]

    async def feed() -> None:
        for item_id, prompt in ITEMS:
            await processor.add_work(LLMWorkItem(item_id=item_id, strategy=strategy, prompt=prompt))
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


def main() -> int:
    point_id, raw_path, raw_workdir = sys.argv[1:4]
    point = POINTS[point_id]
    workdir = Path(raw_workdir)
    ctx = ChildContext(
        path=Path(raw_path),
        workdir=workdir,
        published_log=SideLog(workdir / "published.log"),
        batch_log=SideLog(workdir / "batch.log"),
        batch_watch=BatchWatch(),
    )
    # Patch module-level entry points (open, sqlite3.connect) before the store exists, and the
    # store instance after.
    if point.patches_store:
        ctx.store = make_store(point.store, ctx.path)
        point.installer(ctx, kill_self)
    else:
        point.installer(ctx, kill_self)
        ctx.store = make_store(point.store, ctx.path)
    workload = run_processor if point.layer == "processor" else run_store
    asyncio.run(workload(ctx))
    print(f"kill point {point_id} not reached (fired={ctx.fired})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
