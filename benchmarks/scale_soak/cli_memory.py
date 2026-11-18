"""Peak memory of the ``llm-minder`` inspection CLI on its two axes (#318).

    uv run python -m benchmarks.scale_soak.cli_memory [--items 100000] [--output-mib 8]

- **Cardinality:** a store with ``--items`` distinct item ids (10% failures with a
  short error). The grouping state grows with the number of ids.
- **Single large payload:** a store with one item whose output is ``--output-mib``
  MiB and whose error is a quarter of that, plus a few small items. One record is
  parsed at a time (one SQLite page of rows), so its size shows up once.

Each command runs in a fresh interpreter that reports the Python heap peak
(``tracemalloc``) and its peak RSS (``VmHWM``; Linux only). A baseline run that only imports the CLI is
subtracted from the RSS. No provider calls are made. Writes JSON to ``--report``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CHILD = r"""
import io, json, sys, tracemalloc
from llm_minder import cli

class Discard(io.TextIOBase):
    def write(self, text):
        return len(text)

argv = json.loads(sys.argv[1])
tracemalloc.start()
code = cli.main(argv, stdout=Discard()) if argv else 0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
# VmHWM (peak RSS of this address space) resets on exec, unlike ru_maxrss,
# which a forked child inherits from its parent.
with open('/proc/self/status') as status:
    rss_kib = next(int(line.split()[1]) for line in status if line.startswith('VmHWM:'))
print(json.dumps({"code": code, "heap_peak": peak, "rss_kib": rss_kib}))
"""


async def _write(path: Path, kind: str, results: Any) -> None:
    from llm_minder import ArtifactIdentity, JsonlArtifactStore, LLMWorkItem, SqliteArtifactStore
    from llm_minder.testing import FakeStrategy

    identity = ArtifactIdentity(provider="bench", model="bench-model", prompt_version="1")
    if kind == "sqlite":
        store: Any = SqliteArtifactStore(path, identity=identity, commit_interval_seconds=0.0)
    else:
        store = JsonlArtifactStore(path, identity=identity)
    try:
        for result in results:
            item = LLMWorkItem(item_id=result.item_id, strategy=FakeStrategy("o"), prompt="p")
            prepared = await store.prepare_item(item)
            await store.append(item, prepared, result)
    finally:
        await store.close()


def _cardinality(items: int) -> Any:
    from llm_minder import WorkItemResult

    for n in range(items):
        failed = n % 10 == 0
        yield WorkItemResult(
            item_id=f"item-{n:07d}",
            success=not failed,
            output=None if failed else f"output {n}",
            error="rate limited by provider" if failed else None,
            error_category="rate_limit" if failed else None,
            token_usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )


def _large(output_mib: int) -> Any:
    from llm_minder import WorkItemResult

    size = output_mib * 1024 * 1024
    yield WorkItemResult(item_id="small-1", success=True, output="ok")
    yield WorkItemResult(
        item_id="big-fail",
        success=False,
        error="e" * (size // 4),
        error_category="validation_error",
    )
    yield WorkItemResult(item_id="big", success=True, output="x" * size)
    yield WorkItemResult(item_id="small-2", success=True, output="ok")


def _measure(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(argv)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run(items: int, output_mib: int, workdir: Path) -> dict[str, Any]:
    baseline = _measure([])
    report: dict[str, Any] = {
        "items": items,
        "output_mib": output_mib,
        "python": sys.version.split()[0],
        "baseline_rss_kib": baseline["rss_kib"],
        "runs": [],
    }
    for kind in ("jsonl", "sqlite"):
        for axis, results, commands in (
            (
                "cardinality",
                _cardinality(items),
                [
                    ["inspect", "--json"],
                    ["failures", "--json"],
                    ["show", "item-0000000", "--json"],
                ],
            ),
            (
                "large_payload",
                _large(output_mib),
                [
                    ["inspect", "--json"],
                    ["failures", "--json", "--full"],
                    ["show", "big", "--json", "--output", "--full"],
                ],
            ),
        ):
            path = workdir / f"{axis}.{kind}"
            started = time.perf_counter()
            asyncio.run(_write(path, kind, results))
            write_seconds = time.perf_counter() - started
            for command in commands:
                started = time.perf_counter()
                result = _measure([command[0], str(path), *command[1:]])
                assert result["code"] == 0, (kind, axis, command, result)
                report["runs"].append(
                    {
                        "store": kind,
                        "axis": axis,
                        "command": " ".join(command),
                        "store_bytes": path.stat().st_size,
                        "write_seconds": round(write_seconds, 2),
                        "seconds": round(time.perf_counter() - started, 2),
                        "heap_peak_mib": round(result["heap_peak"] / 2**20, 2),
                        "rss_over_baseline_mib": round(
                            (result["rss_kib"] - baseline["rss_kib"]) / 1024, 2
                        ),
                    }
                )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--output-mib", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("benchmark-results/cli-memory.json"))
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        report = run(args.items, args.output_mib, Path(tmp))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'store':7} {'axis':14} {'command':40} {'heap MiB':>9} {'RSS+ MiB':>9} {'s':>6}")
    for row in report["runs"]:
        print(
            f"{row['store']:7} {row['axis']:14} {row['command']:40} "
            f"{row['heap_peak_mib']:>9} {row['rss_over_baseline_mib']:>9} {row['seconds']:>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
