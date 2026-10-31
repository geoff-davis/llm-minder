"""Former module path of :class:`~async_batch_llm.call_pool.LLMCallPool`.

``LLMCallPool`` moved to :mod:`async_batch_llm.call_pool` in 0.28. This module stays
importable through 1.x; import ``LLMCallPool`` from ``async_batch_llm``. The old
``LLMGateway`` alias was removed in 1.0.
"""

from __future__ import annotations

from typing import Any

from .call_pool import LLMCallPool


def __getattr__(name: str) -> Any:
    if name == "LLMGateway":
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. LLMGateway was removed "
            "in 1.0; use LLMCallPool, the same class."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LLMCallPool"]
