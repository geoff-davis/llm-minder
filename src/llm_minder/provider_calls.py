"""Public helpers for strategies that make their own provider call.

A strategy that overrides ``execute()`` uses these to report what the built-in
strategies report: how long the provider call took, and the tokens a billed
response used when parsing it fails afterwards.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING

from ._internal.execution_state import record_provider_seconds

if TYPE_CHECKING:
    from .base import RetryState, TokenUsage


class provider_call:  # noqa: N801 - used like a function: ``with provider_call(state):``
    """Time a provider call and record it as this attempt's provider time.

    Use it around the call itself, inside ``execute()``, with the ``state``
    passed to ``execute()``::

        async def execute(self, prompt, attempt, timeout, state=None):
            async with provider_call(state):
                response = await self.model.generate(prompt, config=config)
            ...

    The time is recorded even when the call raises. It fills
    ``AttemptTiming.provider_seconds`` and the provider side of the timing
    statistics. Plain ``with`` works too. ``state=None`` records nothing.
    Leave response parsing outside the block, so it isn't counted as provider
    time. If the block runs more than once in an attempt, the last one wins.
    """

    def __init__(self, state: RetryState | None) -> None:
        self._state = state
        self._started = 0.0

    def __enter__(self) -> provider_call:
        self._started = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        record_provider_seconds(self._state, time.perf_counter() - self._started)

    async def __aenter__(self) -> provider_call:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, traceback)


def attach_token_usage(exception: BaseException, tokens: TokenUsage | Mapping[str, int]) -> None:
    """Attach billed token usage to an exception before re-raising it.

    Use it when a response the provider already billed fails afterwards, for
    example in parsing, so the failed attempt's tokens still count::

        try:
            output = parse(response)
        except ValueError as e:
            attach_token_usage(e, response.token_usage)
            raise

    The exception keeps its type, so classification and ``on_error()`` see the
    original error. The usage is copied onto the exception's
    ``_failed_token_usage`` attribute, which the framework reads first; setting
    that attribute yourself is also supported.
    """
    exception.__dict__["_failed_token_usage"] = dict(tokens)


__all__ = ["attach_token_usage", "provider_call"]
