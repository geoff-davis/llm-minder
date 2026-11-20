"""Stateful property tests for ``AbortController`` and ``await_with_guardrails`` (#327).

A Hypothesis state machine drives one controller, with an optional provider-call
cap and an ``AttemptAdmission`` hook, on its own event loop. Each attempt runs
``await_with_guardrails(active_provider=True, admission=slot)`` the way the item
executor does, and settles its ticket in its own ``finally``. Rules: start an
attempt (normal, with an already-expired item deadline, or with an ``on_start``
that raises, the two pre-start failures that refund the call), finish or fail a
running provider call, cancel an attempt, finish a call and cancel its attempt a
few loop turns later (landing in the guardrail join, #379), and stop the batch:
refuse the next admission, make the next settle fail, or abort. The abort mode is
drawn per run: ``CANCEL_ACTIVE`` cancels running calls, ``DRAIN_ACTIVE`` lets them
finish.

Invariants, after the loop is drained:

- ``provider_calls_started`` equals the attempts whose ``on_start`` ran, so a
  refunded pre-start attempt frees exactly one count, and it never passes
  ``max_provider_calls``;
- ``_active_provider_calls`` equals the attempts still awaiting their provider,
  so it returns to 0 once none are (#379);
- no ``on_start`` runs after the batch was aborted;
- every admitted ticket is settled at most once, and exactly once when its
  attempt has finished; an attempt that never started settles with zero usage
  and ``success=False``; a started one settles with its reported usage or
  ``None`` (#331).

A second cancellation of a caller already joining its provider task is not a
defect (#384's correction): awaiting the task forwards the cancel and waits for
it to settle. No rule asserts otherwise.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, cast

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

from llm_minder import AbortMode, AdmissionRefused, AttemptRequest, TokenUsage
from llm_minder._internal.guardrails import (
    AbortCause,
    AbortController,
    AdmissionSlot,
    await_with_guardrails,
)
from tests._loop_quiescence import run_until_quiescent

_ZERO: TokenUsage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


@dataclass
class _Hook:
    refuse_next: bool = False
    fail_next_settle: bool = False
    admitted: list[int] = field(default_factory=list)
    settles: dict[int, list[tuple[TokenUsage | None, bool]]] = field(default_factory=dict)

    def admit(self, request: AttemptRequest) -> object:
        if self.refuse_next:
            self.refuse_next = False
            raise AdmissionRefused("refused by the test")
        ticket = int(request.item_id)
        self.admitted.append(ticket)
        return ticket

    def settle(self, ticket: object, usage: TokenUsage | None, success: bool) -> None:
        assert isinstance(ticket, int)
        self.settles.setdefault(ticket, []).append((usage, success))
        if self.fail_next_settle:
            self.fail_next_settle = False
            raise RuntimeError("settle failed in the test")


@dataclass
class _Attempt:
    task: asyncio.Task[None]
    provider: asyncio.Future[TokenUsage] | None = None
    on_start_ran: bool = False
    reported: TokenUsage | None = None


class _OnStartFailed(Exception):
    pass


async def _drain() -> None:
    """Run the loop until nothing on it can make progress, so an invariant
    checked next sees settled state rather than a cascade still in flight."""
    await run_until_quiescent()


class GuardrailsMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.hook = _Hook()
        self.controller: AbortController | None = None
        self.attempts: dict[int, _Attempt] = {}
        self.finished: dict[int, _Attempt] = {}
        self.next_id = 0
        self.started_after_abort = False

    @initialize(
        cap=st.one_of(st.none(), st.integers(0, 4)),
        mode=st.sampled_from(list(AbortMode)),
    )
    def setup(self, cap: int | None, mode: AbortMode) -> None:
        self.controller = AbortController(mode, max_provider_calls=cap, admission=self.hook)

    # -- the attempt, as the item executor runs it ---------------------------

    async def _attempt(self, attempt_id: int, kind: str) -> None:
        controller = self.controller
        assert controller is not None
        record = self.attempts[attempt_id]
        slot = AdmissionSlot(
            str(attempt_id),
            lambda: AttemptRequest(
                item_id=str(attempt_id),
                attempt=1,
                try_number=1,
                strategy=cast(Any, None),
                prompt="",
                messages=None,
                generation_config=None,
            ),
        )

        def on_start() -> None:
            if controller.aborted:
                self.started_after_abort = True
            if kind == "on_start_fails":
                raise _OnStartFailed
            record.on_start_ran = True

        async def provider() -> TokenUsage:
            record.provider = asyncio.get_running_loop().create_future()
            return await record.provider

        usage: TokenUsage | None = None
        success = False
        try:
            usage = await await_with_guardrails(
                provider(),
                item_deadline=time.perf_counter() - 1.0 if kind == "expired" else None,
                item_id=str(attempt_id),
                abort_controller=controller,
                active_provider=True,
                on_start=on_start,
                admission=slot,
            )
            record.reported = usage
            success = True
        except Exception:
            pass
        finally:
            controller.settle_attempt(slot, usage, success)

    # -- helpers -------------------------------------------------------------

    def _settle(self) -> None:
        self.loop.run_until_complete(_drain())
        for attempt_id, attempt in tuple(self.attempts.items()):
            if attempt.task.done():
                if not attempt.task.cancelled():
                    attempt.task.result()  # the attempt swallows Exception; others fail
                self.finished[attempt_id] = self.attempts.pop(attempt_id)

    def _running(self) -> list[int]:
        return sorted(
            attempt_id
            for attempt_id, attempt in self.attempts.items()
            if attempt.provider is not None and not attempt.provider.done()
        )

    def teardown(self) -> None:
        try:
            tasks = [attempt.task for attempt in self.attempts.values()]
            for task in tasks:
                task.cancel()
            if tasks:
                self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            self.loop.run_until_complete(_drain())
            if self.controller is not None:
                assert self.controller._active_provider_calls == 0
                self.finished.update(self.attempts)
                self.attempts.clear()
                self.tickets_settle_exactly_once()
        finally:
            leftover = asyncio.all_tasks(self.loop)
            for task in leftover:
                task.cancel()
            if leftover:
                self.loop.run_until_complete(asyncio.gather(*leftover, return_exceptions=True))
            self.loop.close()

    # -- rules ---------------------------------------------------------------

    @rule(kind=st.sampled_from(["normal", "normal", "expired", "on_start_fails"]))
    def start_attempt(self, kind: str) -> None:
        attempt_id = self.next_id
        self.next_id += 1
        record = _Attempt(task=None)  # type: ignore[arg-type]
        self.attempts[attempt_id] = record
        record.task = self.loop.create_task(self._attempt(attempt_id, kind))
        self._settle()

    @precondition(lambda self: bool(self._running()))
    @rule(data=st.data(), tokens=st.integers(0, 50), fail=st.booleans())
    def finish_provider(self, data: st.DataObject, tokens: int, fail: bool) -> None:
        attempt = self.attempts[data.draw(st.sampled_from(self._running()))]
        assert attempt.provider is not None
        if fail:
            attempt.provider.set_exception(RuntimeError("provider failed"))
        else:
            attempt.provider.set_result(
                {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}
            )
        self._settle()

    @precondition(lambda self: bool(self.attempts))
    @rule(data=st.data())
    def cancel_attempt(self, data: st.DataObject) -> None:
        attempt_id = data.draw(st.sampled_from(sorted(self.attempts)))
        self.attempts[attempt_id].task.cancel()
        self._settle()

    @precondition(lambda self: bool(self._running()))
    @rule(data=st.data(), turns=st.integers(0, 4))
    def finish_then_cancel(self, data: st.DataObject, turns: int) -> None:
        """Cancel the attempt a few turns after its call returns, so the cancel
        can land while ``await_with_guardrails`` joins its control tasks."""
        attempt = self.attempts[data.draw(st.sampled_from(self._running()))]
        assert attempt.provider is not None
        attempt.provider.set_result({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

        async def turns_then_cancel() -> None:
            for _ in range(turns):
                await asyncio.sleep(0)
            attempt.task.cancel()

        self.loop.run_until_complete(turns_then_cancel())
        self._settle()

    # Each of these ends the run's admissions, so they share one rule (and one
    # share of the rule draws) and are skipped once the batch has stopped.
    @precondition(lambda self: self.controller is not None and not self.controller.aborted)
    @rule(kind=st.sampled_from(["refuse_next_admission", "fail_next_settle", "abort"]))
    def stop(self, kind: str) -> None:
        assert self.controller is not None
        if kind == "refuse_next_admission":
            self.hook.refuse_next = True
        elif kind == "fail_next_settle":
            self.hook.fail_next_settle = True
        else:
            self.controller.trip_now(AbortCause(kind="fail_fast", reason="aborted by the test"))
        self._settle()

    # -- invariants ----------------------------------------------------------

    def _all(self) -> list[_Attempt]:
        return [*self.attempts.values(), *self.finished.values()]

    @invariant()
    def started_count_matches_the_started_attempts(self) -> None:
        if self.controller is None:
            return
        started = sum(1 for attempt in self._all() if attempt.on_start_ran)
        assert self.controller.provider_calls_started == started
        cap = self.controller.max_provider_calls
        if cap is not None:
            assert self.controller.provider_calls_started <= cap

    @invariant()
    def active_count_matches_the_running_calls(self) -> None:
        if self.controller is None:
            return
        running = sum(1 for attempt in self.attempts.values() if attempt.on_start_ran)
        assert self.controller._active_provider_calls == running

    @invariant()
    def nothing_starts_after_an_abort(self) -> None:
        assert not self.started_after_abort

    @invariant()
    def tickets_settle_exactly_once(self) -> None:
        for ticket, settles in self.hook.settles.items():
            assert len(settles) == 1, f"ticket {ticket} settled {len(settles)} times"
        for attempt_id in self.hook.admitted:
            attempt = self.attempts.get(attempt_id) or self.finished[attempt_id]
            settles = self.hook.settles.get(attempt_id, [])
            if attempt_id in self.attempts:
                # Still running: admitted and started, not yet settled.
                assert attempt.on_start_ran and not settles
                continue
            assert len(settles) == 1, f"finished ticket {attempt_id} was never settled"
            usage, success = settles[0]
            if not attempt.on_start_ran:
                assert usage == _ZERO and success is False
            else:
                assert usage is None or usage == attempt.reported


GuardrailsMachine.TestCase.settings = settings()
TestGuardrailsStateful = GuardrailsMachine.TestCase


@pytest.mark.slow
class TestGuardrailsStatefulThorough(GuardrailsMachine.TestCase):  # type: ignore[misc,valid-type]
    settings = settings.get_profile("thorough")
