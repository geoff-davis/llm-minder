"""Overflow-safe bounded exponential delays and retry jitter."""

import math
import random


def capped_backoff(initial: float, base: float, exponent: int, cap: float) -> float:
    """Compute min(initial * base**exponent, cap) without an overflowing power."""
    if initial <= 0 or cap <= 0:
        return 0.0
    if initial >= cap:
        return cap
    if base == 1 or exponent <= 0:
        return initial
    if base > 1 and exponent >= (math.log(cap) - math.log(initial)) / math.log(base):
        return cap
    # Even when the scaled result is finite, base**exponent may overflow.
    try:
        return min(initial * base**exponent, cap)
    except OverflowError:
        return min(math.exp(math.log(initial) + exponent * math.log(base)), cap)


def apply_jitter(wait: float, mode: object) -> float:
    """Scale ``wait`` by ``RetryConfig.jitter``.

    ``"full"`` picks a random 0-100% of the wait. ``"equal"`` and any other truthy
    value pick 50-100%. A falsy value returns the wait unchanged.
    """
    if mode == "full":
        return wait * random.random()
    if mode:
        return wait * (0.5 + random.random() * 0.5)
    return wait
