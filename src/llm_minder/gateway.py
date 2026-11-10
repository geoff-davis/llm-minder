"""Former module path of :class:`~llm_minder.call_pool.LLMCallPool`.

``LLMCallPool`` moved to :mod:`llm_minder.call_pool` in 0.28. This module stays
importable through 1.x; import ``LLMCallPool`` from ``llm_minder``. The old
``LLMGateway`` alias was removed in 1.0.
"""

from __future__ import annotations

from typing import Any

from ._internal.removed_names import removed_name_error
from .call_pool import LLMCallPool


def __getattr__(name: str) -> Any:
    if name == "LLMGateway":
        raise removed_name_error(
            __name__,
            __file__,
            name,
            "LLMGateway was removed in 1.0; use LLMCallPool, the same class.",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LLMCallPool"]
