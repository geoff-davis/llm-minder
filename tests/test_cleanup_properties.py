"""Property tests for ``run_cleanup_steps`` ordering and precedence (#327).

For any sequence of teardown steps, each succeeding, failing, raising
``CancelledError`` itself or raising a process-control exception, optionally
marked as a barrier or as preserving a completed result, optionally grouped
behind a lazily built phase, and for zero, one or two cancellations delivered to
the closing caller while a chosen step runs (with or without a pre-existing body
exception or cancellation), the outcome must match the contract in
``docs/cleanup-lifecycle-contract.md``:

- steps run in order, each at most once; every step runs unless an interrupted
  barrier or a forced abort stops the sequence, and then no later step starts;
- a phase is built exactly once, and only if it is reached;
- one delivery is deferred: teardown finishes and the cancellation is re-raised;
  a pre-existing cancellation counts as the first delivery, and the next
  delivery forces an abort that propagates at once;
- ordinary failures never escape ``run_cleanup_steps``; they are reported in
  order, and ``CancelledError`` raised inside a step becomes
  ``CleanupInterruptedError``;
- the first process-control exception is raised after the remaining steps and
  takes precedence over a deferred cancellation;
- ``CleanupReport.raise_first`` raises the first ordinary failure, skipping
  result-preserving steps when asked to.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from llm_minder._internal.cleanup import (
    CleanupAction,
    CleanupInterruptedError,
    CleanupPhase,
    CleanupReport,
    CleanupStep,
    run_cleanup_steps,
)

_LOGGER = logging.getLogger("tests.cleanup_properties")
_OUTCOMES = ("ok", "fail", "cancel_inside", "fatal")


@dataclass(frozen=True)
class StepSpec:
    outcome: str
    barrier: bool
    preserves: bool


@dataclass(frozen=True)
class Scenario:
    steps: tuple[StepSpec, ...]
    phase_start: int | None  # steps from this index on are built by one phase
    primary: str | None  # None, "body" or "cancel"
    deliveries: int  # cancellations delivered to the caller while ``at_step`` runs
    at_step: int


class _Fatal(KeyboardInterrupt):
    """A process-control exception the harness can tell apart from a real ^C."""


@dataclass
class _Run:
    runs: list[int]
    errors: list[BaseException | None]
    phase_builds: int = 0
    order: list[int] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    started: dict[int, asyncio.Event] = field(default_factory=dict)


def _step_specs() -> st.SearchStrategy[StepSpec]:
    return st.builds(
        StepSpec,
        outcome=st.sampled_from(_OUTCOMES),
        barrier=st.booleans(),
        preserves=st.booleans(),
    )


@st.composite
def scenarios(draw: st.DrawFn) -> Scenario:
    steps = tuple(draw(st.lists(_step_specs(), min_size=1, max_size=6)))
    phase_start = draw(st.one_of(st.none(), st.integers(0, len(steps) - 1)))
    primary = draw(st.sampled_from([None, "body", "cancel"]))
    deliveries = draw(st.integers(0, 2))
    at_step = draw(st.integers(0, len(steps) - 1))
    return Scenario(steps, phase_start, primary, deliveries, at_step)


def _execute(scenario: Scenario) -> tuple[_Run, CleanupReport | None, BaseException | None, bool]:
    """Run the scenario on a fresh loop.

    Returns the record, the report (if returned), the exception the caller
    raised (if any), and whether the cancellations were actually delivered.
    """
    record = _Run(runs=[0] * len(scenario.steps), errors=[None] * len(scenario.steps))
    delivered = False
    callers: list[asyncio.Task[CleanupReport]] = []

    def make_step(index: int, spec: StepSpec, release: asyncio.Future[None]) -> CleanupStep:
        async def run() -> None:
            record.runs[index] += 1
            record.order.append(index)
            record.started[index].set()
            if index == scenario.at_step and scenario.deliveries:
                await release
            record.completed.append(index)
            error: BaseException | None = None
            if spec.outcome == "fail":
                error = ValueError(f"step {index} failed")
            elif spec.outcome == "cancel_inside":
                error = asyncio.CancelledError(f"step {index} cancelled itself")
            elif spec.outcome == "fatal":
                error = _Fatal(f"step {index} fatal")
            record.errors[index] = error
            if error is not None:
                raise error

        return CleanupStep(
            name=f"step{index}",
            run=run,
            preserves_completed_result=spec.preserves,
            barrier=spec.barrier,
        )

    async def scenario_main() -> tuple[CleanupReport | None, BaseException | None]:
        nonlocal delivered
        release: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        record.started = {i: asyncio.Event() for i in range(len(scenario.steps))}
        built = [make_step(i, spec, release) for i, spec in enumerate(scenario.steps)]
        actions: list[CleanupAction] = list(built)
        if scenario.phase_start is not None:
            tail = built[scenario.phase_start :]

            def build() -> list[CleanupAction]:
                record.phase_builds += 1
                return list(tail)

            actions = [*built[: scenario.phase_start], CleanupPhase("phase", build)]
        primary: BaseException | None = None
        if scenario.primary == "body":
            primary = RuntimeError("body failed")
        elif scenario.primary == "cancel":
            primary = asyncio.CancelledError()
        caller = asyncio.create_task(
            run_cleanup_steps(actions, name="prop", logger=_LOGGER, primary_exception=primary)
        )
        callers.append(caller)
        if scenario.deliveries:
            reached = asyncio.create_task(record.started[scenario.at_step].wait())
            await asyncio.wait({reached, caller}, return_when=asyncio.FIRST_COMPLETED)
            reached.cancel()
            if record.started[scenario.at_step].is_set() and not caller.done():
                delivered = True
                for _ in range(scenario.deliveries):
                    caller.cancel()
                    for _ in range(5):
                        await asyncio.sleep(0)
            if not release.done():
                release.set_result(None)
        await asyncio.wait({caller})
        for _ in range(5):
            await asyncio.sleep(0)
        if caller.cancelled():
            return None, asyncio.CancelledError()
        error = caller.exception()
        return (None, error) if error is not None else (caller.result(), None)

    try:
        report, raised = asyncio.run(scenario_main())
    except _Fatal as fatal:
        # A process-control exception escapes the event loop, as it does in
        # production; asyncio.run re-raises it here.
        report, raised = None, fatal
        for caller in callers:
            if caller.done() and not caller.cancelled():
                caller.exception()  # retrieved: no "never retrieved" warning
    return record, report, raised, delivered


def _expected_stop(scenario: Scenario, aborting: bool) -> tuple[int, bool]:
    """Index of the last step that starts, and whether a barrier stopped the run."""
    for index, spec in enumerate(scenario.steps):
        if aborting and index == scenario.at_step:
            return index, False
        if spec.barrier and spec.outcome == "cancel_inside":
            return index, True
    return len(scenario.steps) - 1, False


def _check(scenario: Scenario) -> None:
    record, report, raised, delivered = _execute(scenario)
    deliveries = scenario.deliveries if delivered else 0
    aborting = deliveries + (1 if scenario.primary == "cancel" and deliveries else 0) >= 2
    last, barrier_stop = _expected_stop(scenario, aborting)

    # Order and exactly-once execution.
    assert record.order == list(range(last + 1))
    assert record.runs == [1] * (last + 1) + [0] * (len(scenario.steps) - last - 1)
    if scenario.phase_start is not None:
        assert record.phase_builds == (1 if scenario.phase_start <= last else 0)

    if aborting:
        assert isinstance(raised, asyncio.CancelledError)
        assert scenario.at_step not in record.completed, "a forced abort abandons the step"
        assert report is None
        return

    ran = scenario.steps[: last + 1]
    fatal = next((record.errors[i] for i, spec in enumerate(ran) if spec.outcome == "fatal"), None)
    if fatal is not None:
        assert raised is fatal, "the first process-control exception takes precedence"
        return
    assert record.completed == list(range(last + 1)), "a deferred delivery lets steps finish"
    if deliveries:
        assert isinstance(raised, asyncio.CancelledError), "a single delivery is deferred"
        return
    assert raised is None, "ordinary cleanup failures never escape run_cleanup_steps"
    assert report is not None
    assert report.aborted is barrier_stop
    assert report.fatal is None
    expected = [
        (i, spec) for i, spec in enumerate(ran) if spec.outcome in ("fail", "cancel_inside")
    ]
    assert [issue.step_name for issue in report.issues] == [f"step{i}" for i, _ in expected]
    for issue, (i, spec) in zip(report.issues, expected, strict=True):
        assert issue.preserves_completed_result is spec.preserves
        if spec.outcome == "fail":
            assert issue.error is record.errors[i]
        else:
            assert isinstance(issue.error, CleanupInterruptedError)
            assert issue.error.__cause__ is record.errors[i]
    for preserve in (False, True):
        candidates = [
            issue for issue in report.issues if not (preserve and issue.preserves_completed_result)
        ]
        if candidates:
            with pytest.raises(type(candidates[0].error)) as caught:
                report.raise_first(preserve_completed_result=preserve)
            assert caught.value is candidates[0].error
        else:
            report.raise_first(preserve_completed_result=preserve)


_OK = StepSpec("ok", barrier=False, preserves=False)


@given(scenarios())
# A pre-existing cancellation plus one delivery forces an abort.
@example(Scenario((_OK, _OK), phase_start=None, primary="cancel", deliveries=1, at_step=0))
# Two deliveries during a phase step abandon it and skip the rest of the phase.
@example(Scenario((_OK, _OK, _OK), phase_start=1, primary=None, deliveries=2, at_step=1))
def test_cleanup_ordering_and_precedence(scenario: Scenario) -> None:
    _check(scenario)


@pytest.mark.slow
@settings(settings.get_profile("thorough"))
@given(scenarios())
def test_cleanup_ordering_and_precedence_thorough(scenario: Scenario) -> None:
    _check(scenario)
