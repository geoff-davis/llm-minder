"""Stateful property tests for ``RateLimitCoordinator`` (#327).

A Hypothesis state machine drives one coordinator on its own event loop with an
injected manual clock and sleep. Rules: report a rate limit with the current
generation or a stale one (waiting or not), report a success, finish the running
cooldown, cancel a waiting reporter, and cancel the coordinator-owned cooldown
task from outside (the third-party interruption that ``_cooldown_done`` and
``_run_cooldown`` recover from). Cooldowns are 0 or 5 s, so a zero cooldown that
resumes without sleeping is covered too.

Invariants, after the loop is drained:

- the generation counter equals the reports a model accepts: a report with the
  current generation while not cooling down starts exactly one new generation,
  and any other report (stale, or during a cooldown) changes nothing;
- one COOLDOWN_STARTED at most per generation, and while no cooldown is running,
  exactly one COOLDOWN_ENDED per generation (none lost, none twice);
- the pause flag matches the cooldown state, and a waiting reporter exists only
  during a cooldown: once it ends, every waiter has resumed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from llm_minder import BaseObserver, ProcessingEvent
from llm_minder._internal.cleanup import CleanupInterruptedError
from llm_minder._internal.event_dispatcher import EventDispatcher
from llm_minder._internal.rate_limit_coordinator import RateLimitCoordinator
from llm_minder.strategies.rate_limit import FixedDelayStrategy


@dataclass
class _ManualTime:
    now: float = 0.0
    sleepers: list[asyncio.Future[None]] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.sleepers.append(future)
        try:
            await future
        finally:
            if not future.done():
                future.cancel()
        self.now += delay

    def live(self) -> list[asyncio.Future[None]]:
        self.sleepers = [future for future in self.sleepers if not future.done()]
        return self.sleepers


class _Events(BaseObserver):
    _abl_fast_observer = True

    def __init__(self) -> None:
        self.started = 0
        self.ended = 0

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        if event is ProcessingEvent.COOLDOWN_STARTED:
            self.started += 1
        elif event is ProcessingEvent.COOLDOWN_ENDED:
            self.ended += 1


async def _drain() -> None:
    """Enough loop turns for a report, its owned cooldown task, the strategy call,
    the events and a finalization to reach their next await."""
    for _ in range(16):
        await asyncio.sleep(0)


class RateLimitCoordinatorMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.time = _ManualTime()
        self.events = _Events()
        self.coord: RateLimitCoordinator | None = None
        self.reporters: dict[int, asyncio.Task[None]] = {}
        self.next_id = 0
        self.model_generation = 0
        self.model_in_cooldown = False
        self.interrupted = False

    @initialize(cooldown=st.sampled_from([0.0, 5.0]))
    def setup(self, cooldown: float) -> None:
        self.cooldown = cooldown
        self.coord = RateLimitCoordinator(
            FixedDelayStrategy(cooldown=cooldown),
            EventDispatcher(observers=[self.events], middlewares=[]),
            clock=self.time,
            sleep=self.time.sleep,
        )

    # -- helpers -------------------------------------------------------------

    def _settle(self) -> None:
        self.loop.run_until_complete(_drain())
        for reporter_id, task in tuple(self.reporters.items()):
            if not task.done():
                continue
            if not task.cancelled():
                task.result()  # any other exception fails the test
            del self.reporters[reporter_id]

    def _owned(self) -> list[asyncio.Task[None]]:
        assert self.coord is not None
        return [task for task in self.coord._owned_cooldowns if not task.done()]

    def teardown(self) -> None:
        try:
            if self.coord is not None:
                for future in self.time.live():
                    future.set_result(None)
                self.loop.run_until_complete(_drain())
                try:
                    self.loop.run_until_complete(self.coord.shutdown())
                except CleanupInterruptedError:
                    # A third-party cancellation is reported once, by shutdown;
                    # the next call finalizes the paused generation.
                    assert self.interrupted
                    self.loop.run_until_complete(self.coord.shutdown())
                self.loop.run_until_complete(_drain())
                assert not self.coord.is_paused
                assert self.events.ended == self.coord.current_generation
                for task in self.reporters.values():
                    assert task.done(), "a reporter is still waiting after shutdown"
        finally:
            leftover = asyncio.all_tasks(self.loop)
            for task in leftover:
                task.cancel()
            if leftover:
                self.loop.run_until_complete(asyncio.gather(*leftover, return_exceptions=True))
            self.loop.close()

    # -- rules ---------------------------------------------------------------

    @rule(stale=st.booleans(), wait=st.booleans())
    def report(self, stale: bool, wait: bool) -> None:
        assert self.coord is not None
        observed = self.coord.current_generation
        if stale:
            if observed == 0:
                return
            observed -= 1
        accepted = not stale and not self.model_in_cooldown
        if accepted:
            self.model_generation += 1
            # A zero cooldown finalizes without sleeping, within this step.
            self.model_in_cooldown = self.cooldown > 0
        reporter_id = self.next_id
        self.next_id += 1
        self.reporters[reporter_id] = self.loop.create_task(
            self.coord.handle_rate_limit(reporter_id, observed, wait=wait)
        )
        self._settle()

    @rule()
    def success(self) -> None:
        assert self.coord is not None
        self.loop.run_until_complete(self.coord.on_item_success())
        self._settle()

    @precondition(lambda self: bool(self.time.live()))
    @rule()
    def finish_cooldown(self) -> None:
        for future in self.time.live():
            future.set_result(None)
        self.model_in_cooldown = False
        self._settle()

    @precondition(lambda self: bool(self.reporters))
    @rule(data=st.data())
    def cancel_reporter(self, data: st.DataObject) -> None:
        reporter_id = data.draw(st.sampled_from(sorted(self.reporters)))
        self.reporters[reporter_id].cancel()
        self._settle()

    @precondition(lambda self: bool(self._owned()))
    @rule(data=st.data())
    def interrupt_cooldown(self, data: st.DataObject) -> None:
        """A third party cancels the owned cooldown task; workers must still resume."""
        task = data.draw(st.sampled_from(self._owned()))
        task.cancel()
        self.interrupted = True
        self.model_in_cooldown = False
        self._settle()

    # -- invariants ----------------------------------------------------------

    @invariant()
    def one_generation_per_accepted_report(self) -> None:
        if self.coord is None:
            return
        assert self.coord.current_generation == self.model_generation
        assert self.coord._in_cooldown == self.model_in_cooldown

    @invariant()
    def one_cooldown_per_generation(self) -> None:
        if self.coord is None:
            return
        generation = self.coord.current_generation
        assert self.events.started <= generation
        assert self.events.ended <= generation
        if not self.coord._in_cooldown:
            assert self.events.ended == generation

    @invariant()
    def waiters_resume_when_the_cooldown_ends(self) -> None:
        if self.coord is None:
            return
        assert self.coord.is_paused == self.coord._in_cooldown
        if self.reporters:
            assert self.coord._in_cooldown, "a reporter waits with no cooldown running"


RateLimitCoordinatorMachine.TestCase.settings = settings()
TestRateLimitCoordinatorStateful = RateLimitCoordinatorMachine.TestCase


@pytest.mark.slow
class TestRateLimitCoordinatorStatefulThorough(RateLimitCoordinatorMachine.TestCase):  # type: ignore[misc,valid-type]
    settings = settings.get_profile("thorough")
