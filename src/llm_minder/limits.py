"""Explain the limits a configuration resolves to, and which ones a run waited on (#319).

:func:`describe_limits` reports each throughput limit with its value and where the
value came from. :func:`summarize_gate_waits` (also ``BatchResult.gate_waits()``)
adds up the admission timings already recorded on each result, so a slow run shows
which gate its attempts actually waited on.

Both are read-only: nothing here changes admission.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._internal.capacity import strategy_max_concurrency

if TYPE_CHECKING:
    from .base import WorkItemResult
    from .core.config import ProcessorConfig

__all__ = [
    "EffectiveLimits",
    "GateWait",
    "GateWaitReport",
    "LimitRow",
    "describe_limits",
    "summarize_gate_waits",
]

_ALL_STRATEGIES = "all strategies"


@dataclass(frozen=True)
class LimitRow:
    """One resolved limit: its value, where the value came from, and what it applies to.

    ``value`` is ``None`` when the limit is not set (no limit). ``scope`` is
    ``"all strategies"`` for configuration-wide limits, or a strategy's label
    (its class name) for limits resolved per strategy.
    """

    name: str
    value: int | float | None
    source: str
    scope: str = _ALL_STRATEGIES


@dataclass(frozen=True)
class EffectiveLimits:
    """The limits a ``ProcessorConfig`` resolves to, one row per limit and scope.

    ``in_effect`` is False when a row may still change at batch start: a built-in
    OpenAI-compatible model created without ``max_connections`` resizes its
    connection pool to ``concurrency`` when a batch run first configures it. ``str()``
    gives a readable table.
    """

    rows: tuple[LimitRow, ...]
    in_effect: bool = True

    def __str__(self) -> str:
        header = "Effective limits" + (
            "" if self.in_effect else " (before batch start; see notes on strategy rows)"
        )
        lines = [header]
        current_scope: str | None = None
        for row in self.rows:
            if row.scope != current_scope:
                current_scope = row.scope
                lines.append(f"  [{row.scope}]")
            lines.append(f"    {row.name:<28} {_format_value(row.value):<10} {row.source}")
        return "\n".join(lines)


@dataclass(frozen=True)
class GateWait:
    """Time attempts spent waiting at one admission gate.

    ``share`` is ``total_seconds`` divided by the summed end-to-end time of the
    items reported (0.0 when that is zero). Waits of concurrent items add up, so
    compare shares between gates rather than with wall time. ``attempts_waited``
    counts attempts that waited at least a millisecond at this gate.
    """

    gate: str
    total_seconds: float
    attempts_waited: int
    share: float


@dataclass(frozen=True)
class GateWaitReport:
    """Where a run's attempts waited, largest total first. ``str()`` gives a table.

    The worker pool is not a row: time an item spends queued before a worker picks
    it up is not recorded per item.
    """

    gates: tuple[GateWait, ...]
    items: int
    item_seconds: float

    def __str__(self) -> str:
        lines = [f"Gate waits across {self.items} item(s), {self.item_seconds:.2f}s of item time"]
        if not any(gate.total_seconds > 0 for gate in self.gates):
            lines.append("  no attempt waited at an admission gate")
            return "\n".join(lines)
        for gate in self.gates:
            lines.append(
                f"  {gate.gate:<28} {gate.total_seconds:>9.2f}s  {gate.share:>6.1%}  "
                f"{gate.attempts_waited} attempt(s)"
            )
        return "\n".join(lines)


def _format_value(value: int | float | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _config_source(config: ProcessorConfig, name: str) -> str:
    """Return how ``config.<name>`` got its value, using the record from __post_init__."""
    value = getattr(config, name)
    recorded = getattr(config, "_derived_limits", {}).get(name)
    if recorded is not None and recorded[0] == value:
        return str(recorded[1])
    return "not set" if value is None else "explicit"


def _strategy_labels(strategies: list[Any]) -> list[str]:
    labels: list[str] = []
    seen: dict[str, int] = {}
    for strategy in strategies:
        name = type(strategy).__name__
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name} #{seen[name]}")
    return labels


def _resolve(
    config: ProcessorConfig,
    strategies: Iterable[Any],
    *,
    may_resize: Callable[[Any], bool],
) -> EffectiveLimits:
    """Resolve the limit table; shared by every surface so they cannot drift."""
    max_workers = int(config.max_workers or 0)
    ramp = config.startup_ramp
    rows = [
        LimitRow("concurrency", config.concurrency, _config_source(config, "concurrency")),
        LimitRow("max_workers", max_workers, _config_source(config, "max_workers")),
        LimitRow(
            "max_provider_concurrency",
            config.max_provider_concurrency,
            _config_source(config, "max_provider_concurrency"),
        ),
    ]
    ramp_cap: tuple[int, str] | None = None
    if ramp is not None:
        if ramp.max_concurrency is not None:
            ramp_cap = (ramp.max_concurrency, "startup_ramp.max_concurrency")
            rows.append(LimitRow("startup_ramp.max_concurrency", ramp.max_concurrency, "explicit"))
        else:
            ramp_cap = (max_workers, "startup ramp cap")
            rows.append(LimitRow("startup_ramp.max_concurrency", max_workers, "from max_workers"))
    for name in ("max_requests_per_minute", "max_tokens_per_minute", "quota_burst_seconds"):
        value = getattr(config, name)
        source = _config_source(config, name)
        if value is not None:
            source += ", per quota scope"
        rows.append(LimitRow(name, value, source))

    in_effect = True
    strategy_list = list(strategies)
    for strategy, label in zip(strategy_list, _strategy_labels(strategy_list), strict=True):
        advertised = strategy_max_concurrency(strategy)
        resizes = may_resize(strategy)
        if advertised is None:
            source = "not advertised"
        else:
            source = "advertised by the strategy (e.g. its connection pool)"
        if resizes:
            in_effect = False
            source += (
                f"; a batch run may resize it to concurrency={config.concurrency} at start "
                "(built-in models created without max_connections)"
            )
        rows.append(LimitRow("strategy max_concurrency", advertised, source, label))

        bounds = [
            bound
            for bound in (
                (advertised, "strategy max_concurrency") if advertised is not None else None,
                (config.max_provider_concurrency, "max_provider_concurrency")
                if config.max_provider_concurrency is not None
                else None,
                ramp_cap,
            )
            if bound is not None
        ]
        capacity = min(value for value, _ in bounds) if bounds else None
        if capacity is None:
            capacity_source = "no capacity gate"
        else:
            capacity_source = (
                "lowest of " + ", ".join(f"{name}={value}" for value, name in bounds)
                if len(bounds) > 1
                else f"from {bounds[0][1]}"
            )
        rows.append(LimitRow("provider capacity", capacity, capacity_source, label))

        if capacity is None or max_workers < capacity:
            in_flight, in_flight_source = max_workers, "bound by max_workers"
        elif capacity < max_workers:
            in_flight, in_flight_source = capacity, "bound by provider capacity"
        else:
            in_flight, in_flight_source = max_workers, "max_workers and provider capacity agree"
        rows.append(LimitRow("concurrent attempts", in_flight, in_flight_source, label))
    return EffectiveLimits(rows=tuple(rows), in_effect=in_effect)


def _batch_may_resize(config: ProcessorConfig) -> Callable[[Any], bool]:
    def may_resize(strategy: Any) -> bool:
        return config.concurrency is not None and callable(
            getattr(strategy, "request_concurrency", None)
        )

    return may_resize


def describe_limits(config: ProcessorConfig, *strategies: Any) -> EffectiveLimits:
    """Return the limits ``config`` resolves to, with one block of rows per strategy.

    Each row gives the value and where it came from: ``explicit``, ``derived from
    concurrency``, ``default``, advertised by the strategy, or ``not set``. Strategy
    rows give the provider capacity (the lowest of the strategy's advertised
    ``max_concurrency``, ``max_provider_concurrency`` and the startup-ramp cap) and
    the number of attempts that can run at once.

    Use it before a run of any kind, including ``call()``. The batch processor and
    ``LLMCallPool`` have a ``describe_limits()`` method that fills in their own
    configuration and strategies.

    Provisional (new in 1.1).
    """
    return _resolve(config, strategies, may_resize=_batch_may_resize(config))


# Waits shorter than this are scheduling noise, not waiting at a gate.
_WAITED_SECONDS = 0.001

_GATES: tuple[tuple[str, Callable[[Any], float]], ...] = (
    ("provider capacity", lambda attempt: attempt.admission_wait_seconds),
    ("quota (RPM/TPM)", lambda attempt: attempt.quota_wait_seconds),
    ("rate-limit cooldown", lambda attempt: attempt.cooldown_wait_seconds),
    ("startup ramp and slow start", lambda attempt: attempt.startup_ramp_wait_seconds),
    ("retry backoff", lambda attempt: attempt.retry_backoff_seconds),
)


def summarize_gate_waits(results: Iterable[WorkItemResult[Any, Any]]) -> GateWaitReport:
    """Add up the recorded waits of ``results`` per admission gate, largest first.

    Results replayed from an artifact are skipped: their timings belong to an
    earlier run. The startup-ramp row includes the capacity gate's ramp wait, which
    is also part of the provider-capacity row, and the slow-start delay after a
    rate-limit cooldown, which is not.

    Provisional (new in 1.1).
    """
    current = [result for result in results if not result.replayed_from_artifact]
    item_seconds = sum(result.timing.total_seconds for result in current)
    attempts = [attempt for result in current for attempt in result.timing.attempts]
    gates = []
    for gate, read in _GATES:
        waits = [read(attempt) for attempt in attempts]
        total = sum(waits)
        gates.append(
            GateWait(
                gate=gate,
                total_seconds=total,
                attempts_waited=sum(1 for wait in waits if wait >= _WAITED_SECONDS),
                share=total / item_seconds if item_seconds > 0 else 0.0,
            )
        )
    gates.sort(key=lambda gate: gate.total_seconds, reverse=True)
    return GateWaitReport(gates=tuple(gates), items=len(current), item_seconds=item_seconds)
