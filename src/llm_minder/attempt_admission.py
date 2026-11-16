"""A user hook that admits or refuses each provider attempt (provisional, new in 1.2).

Set ``GuardrailConfig(admission=...)`` to an object with synchronous ``admit`` and
``settle`` methods. The processor calls ``admit`` immediately before each physical
provider attempt, after every other gate (capacity, cooldown, RPM/TPM quota and
``max_provider_calls``), and ``settle`` exactly once for every attempt ``admit``
accepted. See the guardrails guide for a worked hard spend cap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .base import TokenUsage
    from .llm_strategies import LLMCallStrategy


@dataclass(frozen=True)
class AttemptRequest:
    """What one provider attempt is about to send, passed to ``AttemptAdmission.admit()``.

    Attributes:
        item_id: The work item's ID.
        attempt: The logical attempt number, as passed to ``execute()``.
        try_number: The physical try for this item, counting from 1. A rate-limit
            retry reuses ``attempt`` but gets a new ``try_number``.
        strategy: The strategy that will make the call (after any middleware
            replaced it).
        prompt: The item's prompt; ``""`` for a message-list item.
        messages: The item's message list for a message-list item, otherwise
            ``None``. Exactly one of ``prompt`` and ``messages`` is set, so an
            estimator must read whichever one is.
        generation_config: For a ``ModelStrategy``, the effective
            ``generation_config`` for this attempt, after
            ``attempt_generation_config()`` is merged over it; ``None`` when the
            strategy has none, and for other strategies.
    """

    item_id: str
    attempt: int
    try_number: int
    strategy: LLMCallStrategy[Any]
    prompt: str
    messages: Sequence[Mapping[str, Any]] | None
    generation_config: Mapping[str, Any] | None


class AttemptAdmission(Protocol):
    """Admit or refuse each provider attempt, and settle each admitted one.

    Both methods are synchronous and must not block: the processor calls them on
    the event loop with no await in between, so ``admit`` can check and reserve
    against in-memory state atomically, without a lock.

    ``admit`` returns a ticket (any object, ``None`` included) or raises
    :class:`~llm_minder.AdmissionRefused` to refuse. A refusal, or any other
    exception from ``admit``, stops the run as ``admission_refused``; the refused
    attempt is not made and isn't counted toward ``max_provider_calls``.

    ``settle`` is called exactly once for each ticket: after a success, a provider
    failure, a timeout, a cancellation or an abort, and also when the attempt is
    admitted but never reaches the provider (with zero usage and
    ``success=False``). ``usage`` is the token usage the framework recorded for the
    attempt, including usage attached to a failed attempt's exception, or ``None``
    when a call started but no usage is known (it may still have been billed). An exception from ``settle`` is logged and stops the run
    as ``admission_refused``.
    """

    def admit(self, request: AttemptRequest) -> object:
        """Return a ticket for this attempt, or raise ``AdmissionRefused``."""
        ...

    def settle(self, ticket: object, usage: TokenUsage | None, success: bool) -> None:
        """Release or reconcile ``ticket`` once its attempt has finished."""
        ...


__all__ = ["AttemptAdmission", "AttemptRequest"]
