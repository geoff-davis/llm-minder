"""Executor-side helpers for the ``AttemptAdmission`` hook (#331)."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..attempt_admission import AttemptRequest
from .guardrails import AdmissionSlot
from .messages import messages_for_attempt

if TYPE_CHECKING:
    from ..base import LLMWorkItem, RetryState
    from ..llm_strategies import LLMCallStrategy

logger = logging.getLogger(__name__)


def admission_slot(
    work_item: LLMWorkItem[Any, Any, Any],
    strategy: LLMCallStrategy[Any],
    *,
    attempt: int,
    try_number: int,
    retry_state: RetryState | None,
) -> AdmissionSlot:
    """A slot whose request is built only when the hook is actually asked."""

    def build() -> AttemptRequest:
        from ..llm_strategies import ModelStrategy

        generation_config: Any = None
        if isinstance(strategy, ModelStrategy):
            config = strategy._effective_generation_config(attempt, retry_state)
            if config is not None:
                generation_config = MappingProxyType(dict(config))
        messages = (
            None
            if work_item.messages is None
            else tuple(MappingProxyType(message) for message in messages_for_attempt(work_item))
        )
        return AttemptRequest(
            item_id=work_item.item_id,
            attempt=attempt,
            try_number=try_number,
            strategy=strategy,
            prompt="" if messages is not None else work_item.prompt,
            messages=messages,
            generation_config=generation_config,
        )

    return AdmissionSlot(work_item.item_id, build)


def _sdk_retry_setting(model: object) -> str | None:
    """Describe a built-in model client's own retry setting when it retries, else None."""
    if not type(model).__module__.startswith("llm_minder.models"):
        return None
    client = getattr(model, "_client", None)
    if client is None:
        return None
    # OpenAI-compatible and Anthropic SDK clients.
    max_retries = getattr(client, "max_retries", None)
    if isinstance(max_retries, int) and not isinstance(max_retries, bool) and max_retries > 0:
        return f"max_retries={max_retries}"
    # google-genai: Client(http_options=HttpOptions(retry_options=HttpRetryOptions(...))).
    http_options = getattr(getattr(client, "_api_client", None), "_http_options", None)
    retry_options = getattr(http_options, "retry_options", None)
    if retry_options is not None:
        attempts = getattr(retry_options, "attempts", None)
        # The SDK treats attempts=None as its default (several attempts); 0 or 1 is none.
        if attempts is None or (isinstance(attempts, int) and attempts > 1):
            return f"retry_options.attempts={attempts}"
    return None


def warn_if_sdk_retries(strategy: LLMCallStrategy[Any], warned: set[int]) -> None:
    """Warn once per model when its SDK client retries behind the admission hook."""
    model = getattr(strategy, "model", None)
    if model is None or id(model) in warned:
        return
    warned.add(id(model))
    try:
        setting = _sdk_retry_setting(model)
    except Exception:  # a diagnostic must never fail an attempt
        return
    if setting is not None:
        logger.warning(
            "GuardrailConfig.admission is set, but %s's SDK client retries on its own "
            "(%s). The hook admits and settles each framework attempt; SDK retries "
            "inside one attempt are invisible to it and can spend past a cap. Build "
            "the client with retries off (max_retries=0, or no Gemini retry_options).",
            type(model).__name__,
            setting,
        )


__all__ = ["admission_slot", "warn_if_sdk_retries"]
