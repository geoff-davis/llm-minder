"""Stateful property tests for ``CapacityLimiter`` (#327).

A Hypothesis state machine drives one limiter on its own event loop, over two
independent capacity scopes, with an injected frozen clock and an injected jitter
sleep. Rules: admit a holder, release a holder, cancel a waiting or holding task,
and a burst of releases and cancellations a few loop turns apart with no drain
between them (a cancellation can then land in the gate's ``release()`` while a
woken waiter holds the condition lock, the shape of #294's leaked slot), and a
release cancelled while the test holds the scope's condition lock, which stands
in for that woken waiter. Some runs add a
startup ramp: the frozen clock keeps the limit at the ramp's initial concurrency,
and every acquire first awaits the injected jitter sleep, so a cancellation can
also land before the gate is reached. The ramp's boundary wake-up uses a real
timer and is not exercised here; ``tests/test_startup_ramp.py`` covers it.

Invariants, after the loop is drained:

- each scope's active count equals the tasks holding a slot in it, so a slot
  leaked or released twice changes the count and fails;
- the active count never exceeds the scope's limit and never goes negative;
- no waiter is stranded: while a task waits for a scope, that scope is full.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

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

from llm_minder import StartupRampConfig
from llm_minder._internal.capacity import CapacityLimiter, _CapacityGate


@dataclass
class _Scope:
    max_concurrency: int


@dataclass
class _Holder:
    scope: int
    task: asyncio.Task[None]
    release: asyncio.Future[None]
    holding: bool = False


@dataclass
class _JitterLog:
    delays: list[float] = field(default_factory=list)

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)
        await asyncio.sleep(0)


async def _drain() -> None:
    """Enough loop turns for a release's notify, the woken waiter's child task and
    the condition re-acquire to reach their next await."""
    for _ in range(12):
        await asyncio.sleep(0)


async def _finish(tasks: list[asyncio.Task[None]]) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)
    await _drain()


def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and join whatever the gates left running (notifier and timer tasks)
    so closing the loop destroys no pending task."""
    try:
        leftover = asyncio.all_tasks(loop)
        for task in leftover:
            task.cancel()
        if leftover:
            loop.run_until_complete(asyncio.gather(*leftover, return_exceptions=True))
    finally:
        loop.close()


class CapacityLimiterMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.jitter = _JitterLog()
        self.limiter: CapacityLimiter | None = None
        self.scopes: list[_Scope] = []
        self.limits: list[int] = []
        self.holders: dict[int, _Holder] = {}
        self.next_id = 0

    @initialize(
        capacities=st.lists(st.integers(1, 3), min_size=2, max_size=2),
        ramp_initial=st.one_of(st.none(), st.integers(1, 2)),
    )
    def setup(self, capacities: list[int], ramp_initial: int | None) -> None:
        ramp = (
            StartupRampConfig(
                initial_concurrency=ramp_initial,
                ramp_interval_seconds=3_600.0,
                max_concurrency=3,
                jitter_seconds=0.5,
            )
            if ramp_initial is not None
            else None
        )
        self.limiter = CapacityLimiter(
            None,
            max_workers=3,
            startup_ramp=ramp,
            clock=lambda: 0.0,
            sleep=self.jitter.sleep,
        )
        self.scopes = [_Scope(capacity) for capacity in capacities]
        self.limits = [
            min(capacity, ramp_initial) if ramp_initial is not None else capacity
            for capacity in capacities
        ]

    # -- helpers -------------------------------------------------------------

    async def _hold(self, holder_id: int, scope: int, release: asyncio.Future[None]) -> None:
        assert self.limiter is not None
        async with self.limiter.admit(self.scopes[scope]):
            self.holders[holder_id].holding = True
            try:
                await release
            finally:
                self.holders[holder_id].holding = False

    def _settle(self) -> None:
        self.loop.run_until_complete(_drain())
        for holder_id, holder in tuple(self.holders.items()):
            if not holder.task.done():
                continue
            if not holder.task.cancelled():
                holder.task.result()  # any other exception fails the test
            del self.holders[holder_id]

    def _gate(self, scope: int) -> _CapacityGate | None:
        assert self.limiter is not None
        entry = self.limiter._entries.get(id(self.scopes[scope]))
        return entry[2] if entry is not None else None

    def _live(self, *, holding: bool) -> list[int]:
        return sorted(
            holder_id for holder_id, holder in self.holders.items() if holder.holding is holding
        )

    def teardown(self) -> None:
        try:
            tasks = [holder.task for holder in self.holders.values()]
            for task in tasks:
                task.cancel()
            self.loop.run_until_complete(_finish(tasks))
            for scope in range(len(self.scopes)):
                gate = self._gate(scope)
                if gate is not None:
                    assert gate._active == 0, "a slot leaked after every holder ended"
        finally:
            _close_loop(self.loop)

    # -- rules ---------------------------------------------------------------

    @rule(scope=st.integers(0, 1))
    def admit(self, scope: int) -> None:
        holder_id = self.next_id
        self.next_id += 1
        release: asyncio.Future[None] = self.loop.create_future()
        task = self.loop.create_task(self._hold(holder_id, scope, release))
        self.holders[holder_id] = _Holder(scope, task, release)
        self._settle()

    @precondition(lambda self: bool(self._live(holding=True)))
    @rule(data=st.data())
    def release(self, data: st.DataObject) -> None:
        holder_id = data.draw(st.sampled_from(self._live(holding=True)))
        self.holders[holder_id].release.set_result(None)
        self._settle()

    @precondition(lambda self: bool(self.holders))
    @rule(data=st.data())
    def cancel(self, data: st.DataObject) -> None:
        holder_id = data.draw(st.sampled_from(sorted(self.holders)))
        self.holders[holder_id].task.cancel()
        self._settle()

    @precondition(lambda self: bool(self._live(holding=True)))
    @rule(data=st.data(), steps=st.integers(2, 4))
    def burst(self, data: st.DataObject, steps: int) -> None:
        """Releases and cancellations a few loop turns apart, with no drain between
        them, so a cancellation can land while ``release()`` waits for the
        condition lock or while a woken waiter holds it (#294's shape)."""

        async def run() -> None:
            for _ in range(steps):
                holding = [
                    holder_id
                    for holder_id, holder in self.holders.items()
                    if holder.holding and not holder.release.done()
                ]
                live = [
                    holder_id
                    for holder_id, holder in self.holders.items()
                    if not holder.task.done()
                ]
                kinds = (["release"] if holding else []) + (["cancel"] if live else [])
                if not kinds:
                    break
                kind = data.draw(st.sampled_from(kinds))
                if kind == "release":
                    holder_id = data.draw(st.sampled_from(sorted(holding)))
                    self.holders[holder_id].release.set_result(None)
                else:
                    holder_id = data.draw(st.sampled_from(sorted(live)))
                    self.holders[holder_id].task.cancel()
                for _ in range(data.draw(st.integers(0, 3))):
                    await asyncio.sleep(0)

        self.loop.run_until_complete(run())
        self._settle()

    @precondition(lambda self: bool(self._live(holding=True)))
    @rule(data=st.data(), turns=st.integers(0, 3))
    def release_cancelled_while_queued(self, data: st.DataObject, turns: int) -> None:
        """Release a holder while its scope's condition lock is held, then cancel
        it: the cancellation lands while ``release()`` waits for the lock, the case
        it hands its wake-up to a detached task for. Holding the lock stands in for
        a woken waiter between re-acquiring it and re-checking; reaching that
        window through real waiters takes a long exact interleaving."""
        holder_id = data.draw(st.sampled_from(self._live(holding=True)))
        holder = self.holders[holder_id]
        gate = self._gate(holder.scope)
        assert gate is not None

        async def run() -> None:
            await gate._condition.acquire()
            try:
                holder.release.set_result(None)
                for _ in range(turns + 1):
                    await asyncio.sleep(0)
                holder.task.cancel()
                for _ in range(turns):
                    await asyncio.sleep(0)
            finally:
                gate._condition.release()

        self.loop.run_until_complete(run())
        self._settle()

    # -- invariants ----------------------------------------------------------

    @invariant()
    def active_counts_match_the_holders(self) -> None:
        for scope in range(len(self.scopes)):
            gate = self._gate(scope)
            holding = sum(
                1 for holder in self.holders.values() if holder.scope == scope and holder.holding
            )
            active = gate._active if gate is not None else 0
            assert active == holding
            assert 0 <= active <= self.limits[scope]

    @invariant()
    def no_waiter_is_stranded(self) -> None:
        for scope in range(len(self.scopes)):
            waiting = [
                holder
                for holder in self.holders.values()
                if holder.scope == scope and not holder.holding
            ]
            if waiting:
                gate = self._gate(scope)
                assert gate is not None
                assert gate._active == self.limits[scope], "a waiter is stranded"


CapacityLimiterMachine.TestCase.settings = settings()
TestCapacityLimiterStateful = CapacityLimiterMachine.TestCase


@pytest.mark.slow
class TestCapacityLimiterStatefulThorough(CapacityLimiterMachine.TestCase):  # type: ignore[misc,valid-type]
    settings = settings.get_profile("thorough")
