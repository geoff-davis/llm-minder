"""Early checks for public convenience entry points."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from difflib import get_close_matches
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


def validate_strategy(strategy: object) -> None:
    # Keep this leaf usable from base.py without a strategy/base import cycle.
    from ..llm_strategies import LLMCallStrategy

    if not isinstance(strategy, LLMCallStrategy):
        raise TypeError(
            f"strategy must be an LLMCallStrategy, got {type(strategy).__name__}. "
            "Wrap your client with OpenAIStrategy(model), PydanticAIStrategy(agent=...), "
            "or CallableStrategy(fn)."
        )


# Keywords removed in 1.0, with what replaces them.
_REMOVED_KEYWORDS = {
    "timeout_per_item": "It was removed in 1.0; use attempt_timeout (same per-attempt meaning).",
    "enable_detailed_logging": (
        "It was removed in 1.0; set the level of the 'llm_minder' logger instead."
    ),
}

# ParallelBatchProcessor parameters removed in 1.0; they live on its config.
REMOVED_PROCESSOR_KEYWORDS = {
    "max_workers": (
        "It was removed from ParallelBatchProcessor in 1.0; "
        "use config=ProcessorConfig(max_workers=...)."
    ),
    "timeout_per_item": ("It was removed in 1.0; use config=ProcessorConfig(attempt_timeout=...)."),
    "rate_limit_cooldown": (
        "It was removed in 1.0; use "
        "config=ProcessorConfig(rate_limit=RateLimitConfig(cooldown_seconds=...))."
    ),
}


def processor_keyword_hints(
    config_fields: Iterable[str], *, concurrency_shorthand: bool = False
) -> dict[str, str]:
    """Hints for keywords passed to a processor-building entry point.

    ``ProcessorConfig`` settings point at ``config=ProcessorConfig(...)``, and
    the keywords 1.0 removed point at their replacement *on the config*: the
    plain ``_REMOVED_KEYWORDS`` hints describe ``ProcessorConfig``'s own
    keywords, which these entry points don't accept. ``concurrency_shorthand``
    mentions the ``concurrency=`` shorthand the streaming helpers accept.
    """
    hints = {
        name: f"It is a ProcessorConfig setting; use config=ProcessorConfig({name}=...)."
        for name in config_fields
    }
    hints["enable_detailed_logging"] = _REMOVED_KEYWORDS["enable_detailed_logging"]
    hints.update(REMOVED_PROCESSOR_KEYWORDS)
    if concurrency_shorthand:
        hints["max_workers"] = (
            "It is a ProcessorConfig setting; use concurrency=... or "
            "config=ProcessorConfig(max_workers=...)."
        )
    return hints


def validate_keywords(
    kwargs: Mapping[str, object],
    allowed: Iterable[str],
    removed: Mapping[str, str] = _REMOVED_KEYWORDS,
) -> None:
    choices = set(allowed)
    unknown = set(kwargs) - choices
    if not unknown:
        return
    messages = []
    for name in sorted(unknown):
        if name == "timeout":
            hint = "Use config=ProcessorConfig(attempt_timeout=...) for per-attempt timeouts."
        elif name in removed:
            hint = removed[name]
        else:
            matches = get_close_matches(name, sorted(choices), n=1)
            hint = f"Did you mean {matches[0]!r}?" if matches else "Check the processor arguments."
        messages.append(f"Unexpected processor keyword {name!r}. {hint}")
    raise TypeError(" ".join(messages))


def suggest_keyword_errors(
    function: Callable[_P, _R], removed: Mapping[str, str] = _REMOVED_KEYWORDS
) -> Callable[_P, _R]:
    """Keep a constructor's signature while validating its keywords."""
    import inspect
    from functools import wraps

    names = frozenset(inspect.signature(function).parameters)

    @wraps(function)
    def checked(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        validate_keywords(kwargs, names, removed)
        return function(*args, **kwargs)

    return checked
