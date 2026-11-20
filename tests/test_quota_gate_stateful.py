"""Stateful property tests for ``QuotaGate`` (#327).

A Hypothesis state machine drives one gate on its own event loop with a manual
clock: reserve, start, refund before start, reconcile, finalize unknown, repeat a
finalization, cancel a waiter (alone, or in the same step as a refund that may
grant it), and advance time. After every rule the loop is drained before the
invariants run, so a wake-up that is merely scheduled is not mistaken for a
stranded waiter.

Invariants:

- the gate's available request and token quota equals an independent model of
  refill, consumption, refunds and reconciliation, so a reservation that settles
  twice or not at all changes the books and fails;
- a repeated finalization of any kind returns nothing and changes nothing;
- request quota never goes negative;
- FIFO: no reservation is granted while an older one is still waiting;
- no waiter is stranded: while anything waits, the head is not admissible now,
  a wake is scheduled, and the waiter count matches the live waiting tasks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

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

from llm_minder import TokenEstimate
from llm_minder._internal.admission import QuotaGate, QuotaReservation
from tests._loop_quiescence import run_until_quiescent

_TOLERANCE = 1e-6
T = TypeVar("T")


@dataclass
class _Sleeper:
    deadline: float
    future: asyncio.Future[None]


@dataclass
class _ManualClock:
    now: float = 0.0
    sleepers: list[_Sleeper] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.sleepers.append(_Sleeper(self.now + delay, future))
        try:
            await future
        finally:
            if not future.done():
                future.cancel()

    def release_due(self) -> None:
        for sleeper in tuple(self.sleepers):
            if sleeper.future.done():
                self.sleepers.remove(sleeper)
            elif sleeper.deadline <= self.now + 1e-9:
                sleeper.future.set_result(None)
                self.sleepers.remove(sleeper)

    def next_deadline(self) -> float | None:
        live = [s.deadline for s in self.sleepers if not s.future.done()]
        return min(live) if live else None


async def _drain() -> None:
    """Run the loop until nothing on it can make progress, so an invariant
    checked next sees settled state rather than a cascade still in flight."""
    await run_until_quiescent()


class QuotaGateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.clock = _ManualClock()
        self.gate: QuotaGate | None = None
        self.next_id = 0
        self.waiting: dict[int, asyncio.Task[QuotaReservation]] = {}
        self.unstarted: dict[int, QuotaReservation] = {}
        self.started: dict[int, QuotaReservation] = {}
        self.finalized: list[QuotaReservation] = []
        self.granted_ids: set[int] = set()

    # -- setup and helpers --------------------------------------------------

    @initialize(rpm=st.integers(1, 6), tpm=st.one_of(st.none(), st.integers(10, 200)))
    def setup(self, rpm: int, tpm: int | None) -> None:
        self.gate = QuotaGate(rpm, tpm, clock=self.clock, sleep=self.clock.sleep)
        self.rpm = rpm
        self.tpm = tpm
        self.request_capacity = max(1.0, float(rpm))
        self.request_model = self.request_capacity
        self.token_capacity = max(1.0, float(tpm)) if tpm is not None else None
        self.token_model = self.token_capacity

    def _settle(self) -> None:
        self.loop.run_until_complete(_drain())
        for item_id, task in tuple(self.waiting.items()):
            if not task.done():
                continue
            del self.waiting[item_id]
            if task.cancelled():
                continue
            reservation = task.result()  # any other exception fails the test
            self.granted_ids.add(item_id)
            self.unstarted[item_id] = reservation
            self.request_model -= 1.0
            if self.token_model is not None:
                self.token_model -= reservation.reserved_tokens

    def _refund_model(self, reservation: QuotaReservation) -> None:
        self.request_model = min(self.request_capacity, self.request_model + 1.0)
        if self.token_model is not None and self.token_capacity is not None:
            self.token_model = min(
                self.token_capacity, self.token_model + reservation.reserved_tokens
            )

    def _in_loop(self, call: Callable[[], T]) -> T:
        """Run a synchronous gate call inside the loop, as production code does:
        a refund can schedule the gate's wake task."""

        async def run() -> T:
            return call()

        return self.loop.run_until_complete(run())

    def _availability(self) -> tuple[float | None, float | None]:
        assert self.gate is not None
        return self.gate.request_available, self.gate.token_available

    def teardown(self) -> None:
        try:
            for task in self.waiting.values():
                task.cancel()
            if self.gate is not None:
                self.loop.run_until_complete(_drain())
                self.loop.run_until_complete(self.gate.shutdown())
            self.loop.run_until_complete(_drain())
        finally:
            self.loop.close()

    # -- rules ---------------------------------------------------------------

    @rule(tokens=st.integers(1, 200))
    def reserve(self, tokens: int) -> None:
        assert self.gate is not None
        estimate = TokenEstimate(min(tokens, self.tpm)) if self.tpm is not None else None
        item_id = self.next_id
        self.next_id += 1
        self.waiting[item_id] = self.loop.create_task(self.gate.reserve(estimate))
        self._settle()

    @rule(seconds=st.sampled_from([0.25, 1.0, 7.5, 20.0, 61.0]))
    def advance(self, seconds: float) -> None:
        self.clock.now += seconds
        self.request_model = min(
            self.request_capacity, self.request_model + seconds * self.rpm / 60.0
        )
        if self.token_model is not None and self.token_capacity is not None:
            assert self.tpm is not None
            self.token_model = min(
                self.token_capacity, self.token_model + seconds * self.tpm / 60.0
            )
        self.clock.release_due()
        self._settle()

    @precondition(lambda self: self.clock.next_deadline() is not None)
    @rule()
    def advance_to_next_wake(self) -> None:
        deadline = self.clock.next_deadline()
        assert deadline is not None
        self.advance(max(0.0, deadline - self.clock.now))

    @precondition(lambda self: bool(self.unstarted))
    @rule(data=st.data())
    def start(self, data: st.DataObject) -> None:
        item_id = data.draw(st.sampled_from(sorted(self.unstarted)))
        reservation = self.unstarted.pop(item_id)
        reservation.mark_provider_started()
        self.started[item_id] = reservation

    @precondition(lambda self: bool(self.unstarted))
    @rule(data=st.data())
    def refund_before_start(self, data: st.DataObject) -> None:
        item_id = data.draw(st.sampled_from(sorted(self.unstarted)))
        reservation = self.unstarted.pop(item_id)
        assert self._in_loop(reservation.finalize_before_start) is not None
        self._refund_model(reservation)
        self.finalized.append(reservation)
        self._settle()

    @precondition(lambda self: bool(self.unstarted) and bool(self.waiting))
    @rule(data=st.data())
    def refund_then_cancel_waiter(self, data: st.DataObject) -> None:
        """The refund may grant the waiter that is cancelled before it resumes."""
        item_id = data.draw(st.sampled_from(sorted(self.unstarted)))
        waiter_id = data.draw(st.sampled_from(sorted(self.waiting)))
        reservation = self.unstarted.pop(item_id)
        waiter = self.waiting[waiter_id]

        def refund_and_cancel() -> object:
            # No await between the two: a waiter granted by the refund is
            # cancelled before it can resume.
            finalization = reservation.finalize_before_start()
            waiter.cancel()
            return finalization

        assert self._in_loop(refund_and_cancel) is not None
        self._refund_model(reservation)
        self.finalized.append(reservation)
        self._settle()

    @precondition(lambda self: bool(self.started))
    # Small reports often refund into a bucket that refilled meanwhile, where the
    # capacity clamp matters; large ones create underestimation debt.
    @rule(data=st.data(), reported=st.one_of(st.integers(0, 3), st.integers(0, 300)))
    def reconcile(self, data: st.DataObject, reported: int) -> None:
        item_id = data.draw(st.sampled_from(sorted(self.started)))
        reservation = self.started.pop(item_id)
        finalization = self._in_loop(lambda: reservation.reconcile(reported))
        assert finalization is not None
        delta = reported - reservation.reserved_tokens
        assert finalization.delta_tokens == delta
        if self.token_model is not None and self.token_capacity is not None and delta:
            self.token_model = min(self.token_capacity, self.token_model - delta)
        self.finalized.append(reservation)
        self._settle()

    @precondition(lambda self: bool(self.started))
    @rule(data=st.data())
    def finalize_unknown(self, data: st.DataObject) -> None:
        item_id = data.draw(st.sampled_from(sorted(self.started)))
        reservation = self.started.pop(item_id)
        assert self._in_loop(reservation.finalize_unknown) is not None
        self.finalized.append(reservation)
        self._settle()

    @precondition(lambda self: bool(self.finalized))
    @rule(data=st.data())
    def finalize_again(self, data: st.DataObject) -> None:
        reservation = data.draw(st.sampled_from(self.finalized))
        before = self._availability()
        first = reservation.finalization
        assert self._in_loop(reservation.finalize_before_start) is None
        assert self._in_loop(lambda: reservation.reconcile(5)) is None
        assert self._in_loop(reservation.finalize_unknown) is None
        assert self._in_loop(reservation.finalize_request_only) is None
        assert self._in_loop(reservation.finalize) is False
        assert reservation.finalization is first
        self._settle()
        assert self._availability() == before

    @precondition(lambda self: bool(self.waiting))
    @rule(data=st.data())
    def cancel_waiter(self, data: st.DataObject) -> None:
        waiter_id = data.draw(st.sampled_from(sorted(self.waiting)))
        self.waiting[waiter_id].cancel()
        self._settle()

    # -- invariants ----------------------------------------------------------

    @invariant()
    def books_match_the_model(self) -> None:
        if self.gate is None:
            return
        request, token = self._availability()
        assert request == pytest.approx(self.request_model, abs=_TOLERANCE)
        assert request is not None and request >= -_TOLERANCE
        if self.token_model is not None:
            assert token == pytest.approx(self.token_model, abs=_TOLERANCE)

    @invariant()
    def fifo_within_the_scope(self) -> None:
        if self.waiting and self.granted_ids:
            assert max(self.granted_ids) < min(self.waiting)

    @invariant()
    def no_waiter_is_stranded(self) -> None:
        if self.gate is None:
            return
        assert self.gate.waiter_count == len(self.waiting)
        if not self.waiting:
            return
        self.gate._refill()
        head = self.gate._waiters[0]
        assert not self.gate._ready(head.estimate), "an admissible head is still waiting"
        assert self.gate.has_wake_task
        assert self.clock.next_deadline() is not None


QuotaGateMachine.TestCase.settings = settings()
TestQuotaGateStateful = QuotaGateMachine.TestCase


@pytest.mark.slow
class TestQuotaGateStatefulThorough(QuotaGateMachine.TestCase):  # type: ignore[misc,valid-type]
    settings = settings.get_profile("thorough")
