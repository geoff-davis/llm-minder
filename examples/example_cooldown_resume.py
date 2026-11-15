"""Credential-free demo: a shared rate-limit cooldown, then a crash and a resume.

This is the scenario behind the README animation. It needs no extras and no API
key: a fake async client stands in for a provider.

Run 1 sends 20 prompts through 5 workers with a JSONL checkpoint. The client
returns one 429, so every worker pauses for the cooldown and then ramps back up.
After 12 items the run is cancelled, as if the process were killed. Run 2 is the
same command with ``ResumePolicy.REUSE_SUCCESSES``: it replays the 12 finished
items from the checkpoint and calls the client only for the other 8.

    uv run python examples/example_cooldown_resume.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from llm_minder import (
    ArtifactIdentity,
    BaseObserver,
    CallableStrategy,
    CallOutcome,
    JsonlArtifactStore,
    ProcessingEvent,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    process_prompts,
)

ITEMS = 20
CRASH_AFTER = 12
IDENTITY = ArtifactIdentity(provider="fake", model="demo", prompt_version="v1")


class FakeClient:
    """Answers after a short delay; the sixth call gets a 429."""

    def __init__(self, *, rate_limit_on_call: int | None = None) -> None:
        self.calls = 0
        self.rate_limit_on_call = rate_limit_on_call

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        call = self.calls
        await asyncio.sleep(0.15)
        if call == self.rate_limit_on_call:
            raise RuntimeError("429 Too Many Requests")
        return prompt.upper()


def make_strategy(client: FakeClient) -> CallableStrategy[str]:
    async def invoke(prompt: str, *, attempt: int, timeout: float, state: Any) -> CallOutcome[str]:
        text = await client.generate(prompt)
        return CallOutcome(text, token_usage={"input_tokens": 3, "output_tokens": 2})

    return CallableStrategy(invoke, identity=IDENTITY)


class Narrator(BaseObserver):
    """Prints the events the animation shows."""

    def __init__(self, workers: int, crash: asyncio.Event | None = None) -> None:
        self.workers = workers
        self.crash = crash
        self.completed = 0

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        if event is ProcessingEvent.COOLDOWN_STARTED:
            print(
                f"429 from the provider: all {self.workers} workers pause {data['duration']:.1f}s"
            )
        elif event is ProcessingEvent.COOLDOWN_ENDED:
            print("cooldown over: workers ramp back up, the item retries")
        elif event is ProcessingEvent.ITEM_COMPLETED:
            self.completed += 1
            if self.crash is not None and self.completed == CRASH_AFTER:
                self.crash.set()


async def run(
    path: Path,
    prompts: list[tuple[str, str]],
    client: FakeClient,
    *,
    crash: bool,
) -> Any:
    config = ProcessorConfig(
        concurrency=5,
        rate_limit=RateLimitConfig(cooldown_seconds=1.0, slow_start_items=5),
    )
    crash_event = asyncio.Event() if crash else None
    narrator = Narrator(workers=5, crash=crash_event)
    store = JsonlArtifactStore(path, identity=IDENTITY)
    try:
        task = asyncio.create_task(
            process_prompts(
                make_strategy(client),
                prompts,
                config=config,
                artifact_store=store,
                resume=ResumePolicy.REUSE_SUCCESSES,
                observers=[narrator],
            )
        )
        if crash_event is None:
            return await task
        await crash_event.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return None
    finally:
        await store.close()


async def main() -> None:
    prompts = [(f"doc-{i:02d}", f"summarize document {i}") for i in range(ITEMS)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.jsonl"

        print(f"run 1: {ITEMS} items, 5 workers, JSONL checkpoint")
        first = FakeClient(rate_limit_on_call=6)
        await run(path, prompts, first, crash=True)
        stored = len(JsonlArtifactStore.read_results(path).results)
        print(f"process killed (simulated crash): {stored} of {ITEMS} items checkpointed")

        print()
        print("run 2: same command, ResumePolicy.REUSE_SUCCESSES")
        second = FakeClient()
        batch = await run(path, prompts, second, crash=False)
        replayed = ITEMS - second.calls
        print(f"replayed {replayed} from the checkpoint, called the client {second.calls} times")
        print(
            f"Items:     {batch.total_items} total — {batch.succeeded} succeeded, {batch.failed} failed"
        )


if __name__ == "__main__":
    asyncio.run(main())
