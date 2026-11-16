"""Shared monotonic deadline waits and first-cause batch abort control."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from ..base import BatchTermination, LLMWorkItem, TokenUsage, WorkItemResult
from ..core import AbortMode
from ..strategies.errors import (
    AdmissionRefused,
    BatchAbortedError,
    BatchAdmissionClosedError,
    BatchBudgetExceeded,
    BatchDeadlineExceeded,
    ItemDeadlineExceeded,
)
from .cleanup import cancel_and_join

if TYPE_CHECKING:
    from ..attempt_admission import AttemptAdmission, AttemptRequest

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class _OperationTimerExpired(TimeoutError):
    """Only the guardrail operation timer may raise this internal signal."""


class BatchAdmissionStopped(BatchAdmissionClosedError):
    """Internal signal that a controlled abort stopped source admission."""


@dataclass(frozen=True)
class AbortCause:
    kind: Literal["batch_timeout", "fail_fast", "budget_exceeded", "admission_refused"]
    reason: str
    error_category: str | None = None
    triggering_item_id: str | None = None


def _unstarted_usage() -> TokenUsage:
    """Known-zero usage for an admitted attempt that never reached the provider."""
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class AdmissionSlot:
    """One physical attempt's ``AttemptAdmission`` ticket (#331).

    Created per attempt by the executor and handed to ``await_with_guardrails``.
    ``admitted`` and ``settled`` make ``AbortController.settle_attempt`` idempotent,
    so every path that might end the attempt can call it and the hook still sees
    exactly one ``settle`` per admitted ticket.
    """

    __slots__ = ("admitted", "build_request", "item_id", "settled", "ticket")

    def __init__(self, item_id: str, build_request: Callable[[], AttemptRequest]) -> None:
        self.item_id = item_id
        self.build_request = build_request
        self.ticket: object = None
        self.admitted = False
        self.settled = False


def _close_awaitable(value: object) -> None:
    if inspect.iscoroutine(value):
        value.close()


def _admission_refusal_reason(exc: Exception) -> str:
    if isinstance(exc, AdmissionRefused):
        return str(exc) or "Attempt admission refused"
    return f"Attempt admission hook raised {type(exc).__name__}: {exc}"


class AbortController:
    """Concurrency-safe, idempotent first-cause controller for one batch run."""

    def __init__(
        self,
        mode: AbortMode,
        *,
        max_provider_calls: int | None = None,
        admission: AttemptAdmission | None = None,
        trip_hook: Callable[[AbortCause], bool] | None = None,
        announce: Callable[[AbortCause], Awaitable[None]] | None = None,
    ) -> None:
        self.mode = AbortMode(mode)
        self.event = asyncio.Event()
        self._lock = asyncio.Lock()
        self.cause: AbortCause | None = None
        self._active_provider_calls = 0
        # The provider-call cap (GuardrailConfig.max_provider_calls). ``trip_hook``
        # lets the owner close admission in the same synchronous step as the trip;
        # ``announce`` emits the abort event later, outside the lock.
        self.max_provider_calls = max_provider_calls
        self.provider_calls_started = 0
        # The user's AttemptAdmission hook (GuardrailConfig.admission, #331). It is
        # called after begin_provider_call releases the lock, never under it.
        self.admission = admission
        # Models already checked for SDK-level retries behind the hook.
        self.sdk_retry_warned: set[int] = set()
        self._trip_hook = trip_hook
        self._announce = announce
        # A synchronous stop (call cap, admission refusal or settle failure) that
        # still has to be announced, outside the lock and once.
        self._unannounced: AbortCause | None = None

    @property
    def aborted(self) -> bool:
        return self.event.is_set()

    async def trip(self, cause: AbortCause) -> bool:
        """Record the first cause and wake waiters; return whether this call won."""
        async with self._lock:
            return self.trip_now(cause)

    def trip_now(self, cause: AbortCause) -> bool:
        """Synchronously record the first cause; return whether this call won.

        Needs no lock: without an await, no other coroutine on the loop can
        interleave, and ``begin_provider_call`` reads ``cause`` under its lock
        without awaiting in between either. The budget guardrail uses this so its
        threshold decision and the stop are one indivisible step.
        """
        if self.cause is not None:
            return False
        self.cause = cause
        self.event.set()
        return True

    async def begin_provider_call(self, item_id: str | None) -> None:
        """Atomically register a provider attempt before a possible abort.

        With ``max_provider_calls`` set, an attempt beyond the cap is refused here
        and stops the run, so the count can never pass the cap.
        """
        async with self._lock:
            if self.cause is not None:
                raise self.exception_for(item_id)
            cap = self.max_provider_calls
            if cap is not None and self.provider_calls_started >= cap:
                cause = AbortCause(
                    kind="budget_exceeded",
                    reason=f"Provider call cap reached: {cap:,} of {cap:,} calls started",
                    error_category="batch_budget_exceeded",
                    triggering_item_id=item_id,
                )
                self._trip_sync(cause)
                raise self.exception_for(item_id)
            self.provider_calls_started += 1
            self._active_provider_calls += 1

    def _trip_sync(self, cause: AbortCause) -> bool:
        """Stop the run in one synchronous step; remember the cause for announcing."""
        won = self._trip_hook(cause) if self._trip_hook is not None else self.trip_now(cause)
        if won:
            self._unannounced = cause
        return won

    def admit_attempt(self, slot: AdmissionSlot) -> None:
        """Ask the admission hook for a ticket; on refusal, stop the run and raise.

        Called right after ``begin_provider_call`` returns, outside the lock and
        with no await in between, so no abort can land between the two. The
        caller undoes the call registration if this raises. Any ``Exception``
        from building the request or from ``admit`` is a refusal (a spend cap
        that can't decide must not admit); other ``BaseException`` propagate.
        """
        hook = self.admission
        if hook is None:
            raise RuntimeError("AbortController has no admission hook")
        if self.cause is not None:
            raise self.exception_for(slot.item_id)
        try:
            ticket = hook.admit(slot.build_request())
            if inspect.isawaitable(ticket):
                _close_awaitable(ticket)
                raise TypeError(
                    "AttemptAdmission.admit() must be synchronous; it returned an awaitable"
                )
        except Exception as exc:
            if not isinstance(exc, AdmissionRefused):
                logger.error(
                    "AttemptAdmission.admit() raised for %s; stopping the run",
                    slot.item_id,
                    exc_info=exc,
                )
            self._trip_sync(
                AbortCause(
                    kind="admission_refused",
                    reason=_admission_refusal_reason(exc),
                    error_category="batch_admission_refused",
                    triggering_item_id=slot.item_id,
                )
            )
            raise self.exception_for(slot.item_id) from exc
        slot.ticket = ticket
        slot.admitted = True

    def settle_attempt(self, slot: AdmissionSlot, usage: TokenUsage | None, success: bool) -> None:
        """Settle an admitted ticket exactly once; never awaits, never raises ``Exception``.

        A failing ``settle`` fails closed: it is logged and stops the run as
        ``admission_refused``, because a spend cap that can't record spend must
        not keep admitting.
        """
        if not slot.admitted or slot.settled:
            return
        slot.settled = True
        hook = self.admission
        if hook is None:
            return
        try:
            # Typed as returning None; an async settle() would hand back a coroutine.
            settle: Callable[..., object] = hook.settle
            result = settle(
                slot.ticket, None if usage is None else cast(TokenUsage, dict(usage)), success
            )
            if inspect.isawaitable(result):
                _close_awaitable(result)
                raise TypeError(
                    "AttemptAdmission.settle() must be synchronous; it returned an awaitable"
                )
        except Exception as exc:
            logger.error(
                "AttemptAdmission.settle() raised for %s; stopping the run",
                slot.item_id,
                exc_info=exc,
            )
            self._trip_sync(
                AbortCause(
                    kind="admission_refused",
                    reason=f"Attempt admission settle raised {type(exc).__name__}: {exc}",
                    error_category="batch_admission_refused",
                    triggering_item_id=slot.item_id,
                )
            )

    def refund_unstarted_provider_call(self) -> None:
        """Undo the count for a registered attempt that never reached the provider."""
        self.provider_calls_started = max(0, self.provider_calls_started - 1)

    async def announce_pending(self) -> None:
        """Emit the abort event for a synchronous stop, once, after it happened."""
        cause = self._unannounced
        self._unannounced = None
        if cause is not None and self._announce is not None:
            await self._announce(cause)

    def end_provider_call_now(self) -> None:
        # No lock: a synchronous decrement cannot interleave with another
        # coroutine, and cannot be lost to a cancellation while queued for one.
        self._active_provider_calls = max(0, self._active_provider_calls - 1)

    def exception_for(self, item_id: str | None) -> Exception:
        cause = self.cause
        if cause is None:
            raise RuntimeError("AbortController has no cause")
        if cause.kind == "batch_timeout":
            return BatchDeadlineExceeded(cause.reason, item_id=item_id)
        if cause.kind == "budget_exceeded":
            return BatchBudgetExceeded(cause.reason, item_id=item_id)
        if cause.kind == "admission_refused":
            return AdmissionRefused(cause.reason, item_id=item_id)
        return BatchAbortedError(cause.reason, item_id=item_id)

    def raise_if_aborted(self, item_id: str | None) -> None:
        if self.aborted:
            raise self.exception_for(item_id)

    def result_for(self, work_item: LLMWorkItem[Any, Any, Any]) -> WorkItemResult[Any, Any]:
        exception = self.exception_for(work_item.item_id)
        if isinstance(exception, BatchDeadlineExceeded):
            category = "batch_deadline_exceeded"
        elif isinstance(exception, BatchBudgetExceeded):
            category = "batch_budget_exceeded"
        elif isinstance(exception, AdmissionRefused):
            category = "batch_admission_refused"
        else:
            category = "batch_aborted"
        return WorkItemResult(
            item_id=work_item.item_id,
            success=False,
            error=f"{type(exception).__name__}: {exception}",
            context=work_item.context,
            exception=exception,
            submission_index=work_item.submission_index,
            error_category=category,
        )

    def termination(self) -> BatchTermination:
        cause = self.cause
        if cause is None:
            return BatchTermination()
        return BatchTermination(
            kind=cause.kind,
            reason=cause.reason,
            error_category=cause.error_category,
            triggering_item_id=cause.triggering_item_id,
        )


def remaining_seconds(deadline: float | None, *, item_id: str | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise ItemDeadlineExceeded(
            "End-to-end item deadline exceeded",
            item_id=item_id,
        )
    return remaining


async def await_with_guardrails(
    awaitable: Awaitable[_T],
    *,
    item_deadline: float | None,
    item_id: str | None,
    abort_controller: AbortController | None = None,
    operation_timeout: float | None = None,
    active_provider: bool = False,
    on_start: Callable[[], None] | None = None,
    admission: AdmissionSlot | None = None,
) -> _T:
    """Await work bounded by item deadline, operation timeout, and batch abort.

    ``active_provider`` honors ``drain_active`` by ignoring a batch abort after
    the call starts. The controller is still checked immediately before start,
    so no new provider call begins after an abort.

    ``admission`` (with ``active_provider``) asks the controller's admission hook
    for a ticket right after the call is registered. A pre-start failure settles
    the ticket here with zero usage and ``success=False`` (nothing reached the
    provider, so nothing was billed); once the call has started,
    the caller settles it with the attempt's usage.
    """

    def close_unstarted() -> None:
        if inspect.iscoroutine(awaitable):
            awaitable.close()

    def release_unstarted() -> None:
        # Never awaits: the refund, the settle and the decrement can't be split
        # by a cancellation.
        if provider_registered and abort_controller is not None:
            abort_controller.refund_unstarted_provider_call()
            if admission is not None:
                abort_controller.settle_attempt(admission, _unstarted_usage(), False)
            abort_controller.end_provider_call_now()

    provider_registered = False
    try:
        if abort_controller is not None and active_provider:
            await abort_controller.begin_provider_call(item_id)
            if admission is not None and abort_controller.admission is not None:
                try:
                    abort_controller.admit_attempt(admission)
                except BaseException:
                    # Refused or interrupted: not a call, and nothing to settle.
                    abort_controller.refund_unstarted_provider_call()
                    abort_controller.end_provider_call_now()
                    raise
            provider_registered = True
        elif abort_controller is not None:
            abort_controller.raise_if_aborted(item_id)
    except BaseException:
        close_unstarted()
        raise
    try:
        remaining = remaining_seconds(item_deadline, item_id=item_id)
    except BaseException:
        close_unstarted()
        release_unstarted()
        raise

    timeout = operation_timeout
    deadline_is_limit = False
    if remaining is not None and (timeout is None or remaining <= timeout):
        timeout = remaining
        deadline_is_limit = True

    try:
        if on_start is not None:
            on_start()
    except BaseException:
        close_unstarted()
        release_unstarted()
        raise

    task = asyncio.ensure_future(awaitable)
    timer = asyncio.create_task(asyncio.sleep(timeout)) if timeout is not None else None
    watch_abort = (
        asyncio.create_task(abort_controller.event.wait())
        if abort_controller is not None
        and not (active_provider and abort_controller.mode is AbortMode.DRAIN_ACTIVE)
        else None
    )
    controls = {control for control in (timer, watch_abort) if control is not None}
    try:
        if not controls:
            return await task
        done, _ = await asyncio.wait({task, *controls}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        # Consume only the task's own outcome: a cancellation of this caller
        # during the join must propagate, not become a timeout or abort error.
        await cancel_and_join(task)
        if watch_abort is not None and watch_abort in done:
            assert abort_controller is not None
            raise abort_controller.exception_for(item_id)
        if deadline_is_limit:
            raise ItemDeadlineExceeded("End-to-end item deadline exceeded", item_id=item_id)
        raise _OperationTimerExpired
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        raise
    finally:
        # Release the call before the first await: a cancellation while joining
        # the controls must not leave it counted as active (#379).
        if provider_registered and abort_controller is not None:
            abort_controller.end_provider_call_now()
        for control in controls:
            if not control.done():
                control.cancel()
        if controls:
            await asyncio.gather(*controls, return_exceptions=True)


__all__ = [
    "AbortCause",
    "AbortController",
    "AdmissionSlot",
    "BatchAdmissionStopped",
    "await_with_guardrails",
    "remaining_seconds",
]
