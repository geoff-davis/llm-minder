"""``llm-minder``: read-only inspection of JSONL and SQLite artifact stores (#318).

Three commands, all read-only (see "Inspecting a store from the command line" in
``docs/results-and-artifacts.md``)::

    llm-minder inspect  STORE [prices] [--json]
    llm-minder failures STORE [--category C]... [--limit N] [--ids-only | --json] [--full]
    llm-minder show     STORE ITEM_ID [--output] [--prompt] [--metadata] [--full] [prices] [--json]

Every command reads one snapshot of the store (``_internal/store_reader.py``) in
two passes. Pass 1 validates every record and builds only grouping state; any
corruption, or an estimate that cannot be represented, is found there, before
anything is written to stdout. Pass 2 re-reads the same snapshot and streams
output record by record, keeping nothing.

The output formats are provisional (``docs/stability.md``). The JSON documents
carry ``"schema_version": 1``; the human-readable text is not a contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, TextIO, cast

from ._internal.artifact_codec import GUARDRAIL_AUDIT_CATEGORIES
from ._internal.store_reader import (
    StoredRecord,
    StoreSnapshot,
    StoreSummary,
    open_snapshot,
    summarize,
)
from .artifacts import ArtifactFormatError, ArtifactIOError
from .base import BatchResult, CachedTokenRates, TokenUsage, WorkItemResult

JSON_SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_STORE = 3
EXIT_NOT_FOUND = 4
EXIT_UNREPRESENTABLE_ID = 5

TRUNCATE_AT = 200

TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "total_tokens",
)

IDENTITY_FIELDS = (
    "provider",
    "model",
    "prompt_version",
    "parser_version",
    "application_version",
)

# The current AttemptTiming key set (serialization._attempt_to_dict). Every
# attempt in ``show --json`` carries all of them; keys an older record lacks
# are null. tests/test_cli.py checks this list against the serializer.
ATTEMPT_KEYS = (
    "attempt",
    "try_number",
    "total_seconds",
    "admission_wait_seconds",
    "startup_ramp_wait_seconds",
    "execution_seconds",
    "provider_seconds",
    "cooldown_wait_seconds",
    "quota_wait_seconds",
    "estimated_input_tokens",
    "estimated_output_tokens",
    "reserved_tokens",
    "reported_tokens",
    "reconciliation_delta_tokens",
    "quota_scope_id",
    "retry_backoff_seconds",
    "success",
    "error_type",
    "error_category",
    "timeout_category",
)

READ_RATE_NAMES = {
    "gemini": CachedTokenRates.GEMINI,
    "openai": CachedTokenRates.OPENAI,
    "anthropic-read": CachedTokenRates.ANTHROPIC_READ,
    "deepseek": CachedTokenRates.DEEPSEEK,
}
WRITE_RATE_NAMES = {
    "openai-write": CachedTokenRates.OPENAI_WRITE,
    "anthropic-write": CachedTokenRates.ANTHROPIC_WRITE,
    "anthropic-write-1h": CachedTokenRates.ANTHROPIC_WRITE_1H,
}

_GUARDRAIL = frozenset(GUARDRAIL_AUDIT_CATEGORIES)
# Characters that cannot appear in one line of ``--ids-only`` output.
_UNREPRESENTABLE_CATEGORIES = frozenset({"Cc", "Cs", "Zl", "Zp"})


class _CommandError(Exception):
    """A failure with a fixed exit code, reported on stderr."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


# --- Text helpers --------------------------------------------------------------


def escape(text: str) -> str:
    """Escape control, format, surrogate and separator characters, and backslashes.

    Human-readable output passes every stored string through this, so a stored
    value can't forge extra lines or terminal escape sequences.
    """
    out: list[str] = []
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif unicodedata.category(char)[0] in "CZ" and char != " ":
            code = ord(char)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
        else:
            out.append(char)
    return "".join(out)


def truncate(text: str | None, full: bool) -> tuple[str | None, bool]:
    """Truncate stored error text to :data:`TRUNCATE_AT` characters unless ``full``."""
    if text is None or full or len(text) <= TRUNCATE_AT:
        return text, False
    return text[:TRUNCATE_AT], True


def _human_text(text: str | None, full: bool) -> str:
    value, truncated = truncate(text, full)
    if value is None:
        return "-"
    return escape(value) + (" [truncated; --full shows all]" if truncated else "")


def _is_representable_id(item_id: str) -> bool:
    return not any(unicodedata.category(char) in _UNREPRESENTABLE_CATEGORIES for char in item_id)


def _dumps(value: Any) -> str:
    # allow_nan=False: no document may contain NaN or Infinity. ASCII output keeps
    # lone surrogates and control characters escaped whatever the terminal encoding.
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def _none_first(value: str | None) -> tuple[int, str]:
    return (0, "") if value is None else (1, value)


# --- Prices ----------------------------------------------------------------------


@dataclass(frozen=True)
class Prices:
    input_price: float
    output_price: float
    cached_token_rate: float
    cache_write_rate: float | None


def _finite_number(text: str, what: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what} must be a number; got {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{what} must be finite; got {text!r}")
    return value


def _price(text: str) -> float:
    value = _finite_number(text, "price")
    if value < 0:
        raise argparse.ArgumentTypeError(f"price must be 0 or more; got {text!r}")
    return value


def _read_rate(text: str) -> float:
    if text in READ_RATE_NAMES:
        return READ_RATE_NAMES[text]
    if text in WRITE_RATE_NAMES:
        raise argparse.ArgumentTypeError(
            f"{text!r} is a cache-write rate; --cached-token-rate takes one of "
            f"{', '.join(READ_RATE_NAMES)} or a number in [0, 1]"
        )
    if text and (text[0].isalpha()) and text.lower() not in {"inf", "infinity", "nan"}:
        raise argparse.ArgumentTypeError(
            f"unknown rate name {text!r}; use one of {', '.join(READ_RATE_NAMES)} or a number"
        )
    value = _finite_number(text, "--cached-token-rate")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"--cached-token-rate must be in [0, 1] (the fraction of the input price paid for "
            f"cached tokens); got {text!r}"
        )
    return value


def _write_rate(text: str) -> float:
    if text in WRITE_RATE_NAMES:
        return WRITE_RATE_NAMES[text]
    if text in READ_RATE_NAMES:
        raise argparse.ArgumentTypeError(
            f"{text!r} is a cache-read rate; --cache-write-rate takes one of "
            f"{', '.join(WRITE_RATE_NAMES)} or a number"
        )
    if text and (text[0].isalpha()) and text.lower() not in {"inf", "infinity", "nan"}:
        raise argparse.ArgumentTypeError(
            f"unknown rate name {text!r}; use one of {', '.join(WRITE_RATE_NAMES)} or a number"
        )
    value = _finite_number(text, "--cache-write-rate")
    if value < 0:
        raise argparse.ArgumentTypeError(f"--cache-write-rate must be 0 or more; got {text!r}")
    return value


def _prices(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Prices | None:
    given = [args.input_price is not None, args.output_price is not None]
    rates = [args.cached_token_rate is not None, args.cache_write_rate is not None]
    if not any(given) and not any(rates):
        return None
    if not all(given):
        parser.error("--input-price and --output-price must be given together")
    if args.cached_token_rate is None:
        parser.error("prices need --cached-token-rate (a number in [0, 1] or a rate name)")
    return Prices(
        args.input_price, args.output_price, args.cached_token_rate, args.cache_write_rate
    )


def estimate(tokens: Mapping[str, int], prices: Prices, where: str) -> float:
    """``BatchResult.estimated_cost`` arithmetic on one token mapping, always finite.

    Raises a usage error (exit 2) when the result is not representable: finite
    prices can still overflow for large token counts.
    """
    usage = cast(TokenUsage, {key: tokens[key] for key in TOKEN_KEYS if key in tokens})
    result: WorkItemResult[Any, Any] = WorkItemResult(
        item_id="estimate",
        success=True,
        token_usage=usage,
    )
    try:
        value = BatchResult([result]).estimated_cost(
            prices.input_price,
            prices.output_price,
            prices.cached_token_rate,
            cache_write_rate=prices.cache_write_rate,
        )
    except OverflowError:
        value = math.inf
    if not math.isfinite(value):
        raise _CommandError(
            EXIT_USAGE,
            f"the estimated cost {where} is not representable: the prices are too large for "
            "the recorded token counts",
        )
    return value


# --- inspect ---------------------------------------------------------------------


def _identity_summary(identity: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in IDENTITY_FIELDS:
        value = identity.get(name)
        if value is not None and not isinstance(value, str):
            raise ArtifactFormatError(f"Malformed store manifest: identity.{name} must be a string")
        fields[name] = value
    extra = identity.get("extra", {})
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise ArtifactFormatError("Malformed store manifest: identity.extra must be an object")
    fields["extra_keys"] = sorted(str(key) for key in extra)
    return fields


def _category_counts(counts: Mapping[str | None, int]) -> list[dict[str, Any]]:
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], _none_first(kv[0])))
    return [{"error_category": category, "items": count} for category, count in ordered]


def _inspect_document(
    store: str, snapshot: StoreSnapshot, summary: StoreSummary, prices: Prices | None
) -> dict[str, Any]:
    manifest = snapshot.manifest
    succeeded, failed, stopped = summary.inventory()
    totals = {key: int(summary.token_totals.get(key, 0)) for key in TOKEN_KEYS}
    identities = sorted(
        summary.identities.items(),
        key=lambda kv: (-kv[1], *(_none_first(part) for part in kv[0])),
    )
    estimated: dict[str, Any] | None = None
    if prices is not None:
        estimated = {
            "amount": estimate(totals, prices, "for the store's token totals"),
            "input_price": prices.input_price,
            "output_price": prices.output_price,
            "cached_token_rate": prices.cached_token_rate,
            "cache_write_rate": prices.cache_write_rate,
        }
    stops = []
    for category in GUARDRAIL_AUDIT_CATEGORIES:
        recorded = summary.guardrail_stops.get(category)
        if recorded is None or recorded.count == 0:
            continue
        reason, truncated = truncate(recorded.first_reason, False)
        stops.append(
            {
                "error_category": category,
                "count": recorded.count,
                "first_reason": reason,
                "first_reason_truncated": truncated,
            }
        )
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": "inspect",
        "store": {"path": store, "kind": manifest.kind},
        "manifest": {
            "artifact_schema_version": manifest.artifact_schema_version,
            "sqlite_schema_version": manifest.sqlite_schema_version,
            "created_at": manifest.created_at,
            "package_version": manifest.package_version,
            "identity_fingerprint": manifest.identity_fingerprint,
            "identity": _identity_summary(manifest.identity),
            "user_metadata_keys": sorted(str(key) for key in manifest.user_metadata),
        },
        "record_count": summary.record_count,
        "item_count": summary.item_count,
        "identities": [
            {"provider": p, "model": m, "prompt_version": v, "records": count}
            for (p, m, v), count in identities
        ],
        "items_by_newest_record": {
            "succeeded": succeeded,
            "failed": _category_counts(failed),
            "stopped_by_guardrail": _category_counts(stopped),
        },
        "token_totals": totals,
        "recorded_cost": {
            "amount": summary.recorded_cost if summary.recorded_cost_records else None,
            "records_with_cost": summary.recorded_cost_records,
            "records": summary.record_count,
        },
        "estimated_cost": estimated,
        "guardrail_stops": stops,
    }


def _inspect_text(doc: dict[str, Any]) -> str:
    def show(value: Any) -> str:
        return "-" if value is None else escape(str(value))

    manifest = doc["manifest"]
    identity = manifest["identity"]
    lines = [
        f"Store: {escape(doc['store']['path'])} ({doc['store']['kind']})",
        f"  artifact schema {manifest['artifact_schema_version']}"
        + (
            f", SQLite schema {manifest['sqlite_schema_version']}"
            if manifest["sqlite_schema_version"] is not None
            else ""
        ),
        f"  created {show(manifest['created_at'])} by llm-minder {show(manifest['package_version'])}",
        "  identity: "
        + ", ".join(f"{name}={show(identity[name])}" for name in IDENTITY_FIELDS)
        + (
            f"; extra keys: {', '.join(escape(k) for k in identity['extra_keys'])}"
            if identity["extra_keys"]
            else ""
        ),
        "  user metadata keys: "
        + (", ".join(escape(k) for k in manifest["user_metadata_keys"]) or "-"),
        "",
        f"Records: {doc['record_count']}   Items: {doc['item_count']}",
        "Identities seen (provider / model / prompt version: records):",
    ]
    for entry in doc["identities"]:
        lines.append(
            f"  {show(entry['provider'])} / {show(entry['model'])} / "
            f"{show(entry['prompt_version'])}: {entry['records']}"
        )
    items = doc["items_by_newest_record"]
    lines += ["", "Items by newest record per item id:", f"  succeeded: {items['succeeded']}"]
    for label, key in (("failed", "failed"), ("stopped by guardrail", "stopped_by_guardrail")):
        total = sum(entry["items"] for entry in items[key])
        lines.append(f"  {label}: {total}")
        for entry in items[key]:
            lines.append(f"    {show(entry['error_category'])}: {entry['items']}")
    lines += ["", "Tokens (summed over all records, earlier attempts included):"]
    lines += [f"  {key}: {value}" for key, value in doc["token_totals"].items()]
    cost = doc["recorded_cost"]
    lines.append("")
    if cost["amount"] is None:
        lines.append("Recorded cost: none recorded")
    else:
        lines.append(
            f"Recorded cost: {cost['amount']:.6g} "
            f"({cost['records_with_cost']} of {cost['records']} records have a recorded cost)"
        )
    if doc["estimated_cost"] is not None:
        est = doc["estimated_cost"]
        lines.append(
            f"Estimated cost: {est['amount']:.6g} (estimate from the given prices per million tokens)"
        )
    lines += ["", "Recorded guardrail stops (what was recorded, not how the run ended):"]
    if not doc["guardrail_stops"]:
        lines.append("  none recorded")
    for stop in doc["guardrail_stops"]:
        reason = escape(stop["first_reason"]) if stop["first_reason"] is not None else "-"
        if stop["first_reason_truncated"]:
            reason += " [truncated]"
        lines.append(f"  {stop['error_category']}: {stop['count']} (first reason: {reason})")
    return "\n".join(lines) + "\n"


def _run_inspect(args: argparse.Namespace, prices: Prices | None, out: TextIO) -> int:
    with open_snapshot(args.store) as snapshot:
        summary = summarize(snapshot)
        doc = _inspect_document(args.store, snapshot, summary, prices)
    text = _dumps(doc) + "\n" if args.json else _inspect_text(doc)
    out.write(text)
    return EXIT_OK


# --- failures --------------------------------------------------------------------


def _failure_entry(record: StoredRecord, full: bool) -> dict[str, Any]:
    error, truncated = truncate(record.error, full)
    return {
        "item_id": record.item_id,
        "sequence": record.sequence,
        "recorded_at": record.recorded_at,
        "error_category": record.error_category,
        "error": error,
        "error_truncated": truncated,
        "provider": record.provider,
        "model": record.model,
        "prompt_version": record.prompt_version,
    }


def _run_failures(args: argparse.Namespace, out: TextIO) -> int:
    named = set(args.category or ())

    def selected(success: bool, category: str | None) -> bool:
        if named:
            return category in named
        return not success and category not in _GUARDRAIL

    with open_snapshot(args.store) as snapshot:
        summary = summarize(snapshot)  # pass 1: validate everything, group only
        total = 0
        excluded = 0
        for _sequence, success, category in summary.newest.values():
            if selected(success, category):
                total += 1
            elif category in _GUARDRAIL:  # a stop no --category named
                excluded += 1
        limit = args.limit if args.limit is not None else total
        shown = min(total, limit)

        if args.ids_only:
            chosen = sorted(
                (sequence, item_id)
                for item_id, (sequence, success, category) in summary.newest.items()
                if selected(success, category)
            )[:shown]
            bad = sum(1 for _sequence, item_id in chosen if not _is_representable_id(item_id))
            if bad:
                raise _CommandError(
                    EXIT_UNREPRESENTABLE_ID,
                    f"{bad} item id(s) contain a line break, NUL or other control character and "
                    "can't be printed one per line; use `failures --json`",
                )
            for _sequence, item_id in chosen:
                out.write(item_id + "\n")
            return EXIT_OK

        if args.json:
            header = {
                "schema_version": JSON_SCHEMA_VERSION,
                "command": "failures",
                "store": {"path": args.store, "kind": snapshot.manifest.kind},
                "categories": list(args.category) if args.category else None,
                "total": total,
                "shown": shown,
                "excluded_guardrail_stops": excluded,
            }
            out.write(_dumps(header)[:-1] + ', "failures": [')
        else:
            what = "matching items" if named else "failed items"
            out.write(f"{total} {what} (by newest record per item id)")
            if shown < total:
                out.write(f"; showing the first {shown}")
            out.write("\n")

        printed = 0
        for record in _stream(snapshot):  # pass 2: stream, keeping nothing
            if printed >= shown:
                break
            newest = summary.newest[record.item_id]
            if newest[0] != record.sequence or not selected(record.success, record.error_category):
                continue
            if args.json:
                out.write((", " if printed else "") + _dumps(_failure_entry(record, args.full)))
            else:
                out.write(
                    f"{escape(record.item_id)}\t{escape(record.error_category or '-')}\t"
                    f"{_human_text(record.error, args.full)}\n"
                )
            printed += 1
        if args.json:
            out.write("]}\n")
        elif excluded:
            out.write(
                f"({excluded} item(s) stopped by a guardrail are not listed; name their category "
                "with --category to include them. Stopped items can include attempts that ran "
                "and were billed.)\n"
            )
    return EXIT_OK


def _stream(snapshot: StoreSnapshot) -> Iterator[StoredRecord]:
    """Pass 2. A decode error here means the snapshot changed under the command."""
    try:
        yield from snapshot.records()
    except (ArtifactFormatError, ArtifactIOError) as exc:
        raise _CommandError(
            EXIT_STORE,
            f"{exc} (found while printing: the store changed during the command; output above "
            "is incomplete)",
        ) from exc


# --- show ------------------------------------------------------------------------


def _optional(value: Any, kind: type | tuple[type, ...]) -> Any:
    if isinstance(value, bool) and kind is not bool:
        return None
    return value if isinstance(value, kind) else None


def _timing(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = result.get("timing")
    if not isinstance(raw, Mapping):
        raw = {}
    attempts = raw.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    return {
        "total_seconds": _optional(raw.get("total_seconds"), (int, float)),
        "timeout_category": _optional(raw.get("timeout_category"), str),
        "attempts": [
            {
                key: (attempt.get(key) if isinstance(attempt, Mapping) else None)
                for key in ATTEMPT_KEYS
            }
            for attempt in attempts
        ],
    }


def _exception(result: Mapping[str, Any], full: bool) -> dict[str, Any] | None:
    raw = result.get("exception")
    if not isinstance(raw, Mapping):
        return None
    message, truncated = truncate(_optional(raw.get("message"), str), full)
    return {
        "module": _optional(raw.get("module"), str),
        "class_name": _optional(raw.get("class_name"), str),
        "message": message,
        "message_truncated": truncated,
    }


def _included(result: Mapping[str, Any], flag: str, key: str) -> bool:
    value = result.get(flag)
    if isinstance(value, bool):
        return value
    return key in result


def _show_record(
    record: StoredRecord, newest_sequence: int, args: argparse.Namespace, prices: Prices | None
) -> dict[str, Any]:
    result = record.result
    error, error_truncated = truncate(record.error, args.full)
    entry: dict[str, Any] = {
        "sequence": record.sequence,
        "is_newest": record.sequence == newest_sequence,
        "recorded_at": record.recorded_at,
        "success": record.success,
        "error_category": record.error_category,
        "error": error,
        "error_truncated": error_truncated,
        "exception": _exception(result, args.full),
        "identity": {
            "provider": record.provider,
            "model": record.model,
            "prompt_version": record.prompt_version,
            "identity_fingerprint": record.identity_fingerprint,
        },
        "token_usage": dict(record.token_usage),
        "calculated_cost": record.calculated_cost,
        "estimated_cost": (
            None
            if prices is None
            else estimate(record.token_usage, prices, f"for record {record.sequence}")
        ),
        "submission_index": _optional(result.get("submission_index"), int),
        "replayed_from_artifact": _optional(result.get("replayed_from_artifact"), bool),
        "timing": _timing(result),
    }
    if args.output:
        included = _included(result, "output_included", "output")
        entry["output_included"] = included
        entry["output"] = result.get("output") if included else None
    if args.prompt:
        entry["prompt_stored"] = record.prompt is not None
        entry["prompt"] = record.prompt
    if args.metadata:
        included = _included(result, "metadata_included", "metadata")
        entry["metadata_included"] = included
        entry["metadata"] = result.get("metadata") if included else None
    return entry


def _show_text(entry: dict[str, Any], number: int) -> str:
    def show(value: Any) -> str:
        return "-" if value is None else escape(str(value))

    identity = entry["identity"]
    lines = [
        f"Record {number} (sequence {entry['sequence']})"
        + (", newest" if entry["is_newest"] else ""),
        f"  recorded at: {show(entry['recorded_at'])}",
        f"  success: {entry['success']}   category: {show(entry['error_category'])}",
        f"  identity: {show(identity['provider'])} / {show(identity['model'])} / "
        f"{show(identity['prompt_version'])}",
    ]
    if entry["error"] is not None:
        error = escape(entry["error"]) + (
            " [truncated; --full shows all]" if entry["error_truncated"] else ""
        )
        lines.append(f"  error: {error}")
    exception = entry["exception"]
    if exception is not None:
        lines.append(f"  exception: {show(exception['module'])}.{show(exception['class_name'])}")
    tokens = ", ".join(f"{escape(k)}={v}" for k, v in entry["token_usage"].items()) or "-"
    lines.append(f"  tokens: {tokens}")
    cost = "-" if entry["calculated_cost"] is None else f"{entry['calculated_cost']:.6g}"
    line = f"  recorded cost: {cost}"
    if entry["estimated_cost"] is not None:
        line += f"   estimate: {entry['estimated_cost']:.6g}"
    lines.append(line)
    timing = entry["timing"]
    lines.append(
        f"  timing: total {show(timing['total_seconds'])}s, timeout category "
        f"{show(timing['timeout_category'])}"
    )
    for attempt in timing["attempts"]:
        waits = ", ".join(
            f"{name} {show(attempt[key])}"
            for name, key in (
                ("admission", "admission_wait_seconds"),
                ("quota", "quota_wait_seconds"),
                ("cooldown", "cooldown_wait_seconds"),
                ("backoff", "retry_backoff_seconds"),
            )
        )
        lines.append(
            f"    attempt {show(attempt['attempt'])} try {show(attempt['try_number'])}: "
            f"total {show(attempt['total_seconds'])}s, provider {show(attempt['provider_seconds'])}s, "
            f"execution {show(attempt['execution_seconds'])}s; waits: {waits}; "
            f"error {show(attempt['error_type'])}, timeout {show(attempt['timeout_category'])}"
        )
    for flag, key, label in (
        ("output_included", "output", "output"),
        ("prompt_stored", "prompt", "prompt"),
        ("metadata_included", "metadata", "metadata"),
    ):
        if flag in entry:
            if not entry[flag]:
                lines.append(f"  {label}: not stored")
            else:
                lines.append(f"  {label}: {escape(_dumps(entry[key]))}")
    return "\n".join(lines) + "\n"


def _render_show(entry: dict[str, Any], number: int, as_json: bool) -> str:
    try:
        return _dumps(entry) if as_json else _show_text(entry, number)
    except ValueError as exc:  # a non-finite stored number (NaN/Infinity tokens in the JSON)
        raise ArtifactFormatError(
            f"Malformed artifact record at sequence {entry['sequence']}: {exc}"
        ) from exc


def _run_show(args: argparse.Namespace, prices: Prices | None, out: TextIO) -> int:
    item_id = args.item_id
    with open_snapshot(args.store) as snapshot:
        # Pass 1: validate every record, and render (then drop) each of this item's
        # records, so a bad estimate or value fails before stdout starts.
        count = 0
        newest = -1
        for record in snapshot.records():
            if record.item_id == item_id:
                count += 1
                newest = record.sequence
                _render_show(_show_record(record, newest, args, prices), count, args.json)
        if count == 0:
            raise _CommandError(EXIT_NOT_FOUND, f"item {escape(item_id)} not found in the store")

        # Pass 2: stream.
        if args.json:
            header = {
                "schema_version": JSON_SCHEMA_VERSION,
                "command": "show",
                "store": {"path": args.store, "kind": snapshot.manifest.kind},
                "item_id": item_id,
                "record_count": count,
            }
            out.write(_dumps(header)[:-1] + ', "records": [')
        else:
            out.write(f"Item {escape(item_id)}: {count} record(s), oldest first\n")
        number = 0
        for record in _stream(snapshot):
            if record.item_id != item_id:
                continue
            number += 1
            text = _render_show(_show_record(record, newest, args, prices), number, args.json)
            if args.json:
                out.write((", " if number > 1 else "") + text)
            else:
                out.write(text)
        if args.json:
            out.write("]}\n")
    return EXIT_OK


# --- Entry point -------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {escape(message)}\n")


def _add_prices(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "cost estimate",
        "Prices are per million tokens, in any currency. --cached-token-rate is required with "
        f"prices: a number in [0, 1] or one of {', '.join(READ_RATE_NAMES)}. --cache-write-rate "
        f"is optional: a number >= 0 or one of {', '.join(WRITE_RATE_NAMES)}.",
    )
    group.add_argument("--input-price", type=_price, metavar="P")
    group.add_argument("--output-price", type=_price, metavar="P")
    group.add_argument("--cached-token-rate", type=_read_rate, metavar="R")
    group.add_argument("--cache-write-rate", type=_write_rate, metavar="R")


def _version() -> str:
    from . import __version__

    return f"llm-minder {__version__}"


def _limit(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--limit must be an integer; got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"--limit must be 1 or more; got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="llm-minder",
        description="Inspect a JSONL or SQLite artifact store, read-only.",
        epilog="Exit codes: 0 success; 2 usage error; 3 store missing, unreadable, foreign, "
        "too new or corrupt; 4 item not found; 5 --ids-only can't print an id.",
    )
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    inspect = commands.add_parser(
        "inspect",
        help="summarize a store: manifest, identities, item inventory, tokens, cost",
        description="Summarize a store. Items are classified by their newest record; tokens and "
        "cost are summed over every record.",
    )
    inspect.add_argument("store")
    _add_prices(inspect)
    inspect.add_argument("--json", action="store_true", help="print one JSON document")

    failures = commands.add_parser(
        "failures",
        help="list items whose newest record is a failure",
        description="List items whose newest record is a failure, in store order. Items stopped "
        "by a guardrail are left out unless their category is named with --category; they can "
        "include attempts that ran and were billed (timed out or aborted), not only unstarted work.",
    )
    failures.add_argument("store")
    failures.add_argument(
        "--category", action="append", metavar="C", help="only this category (repeatable)"
    )
    failures.add_argument("--limit", type=_limit, metavar="N", help="list at most N items")
    output = failures.add_mutually_exclusive_group()
    output.add_argument("--ids-only", action="store_true", help="print one item id per line")
    output.add_argument("--json", action="store_true", help="print one JSON document")
    failures.add_argument("--full", action="store_true", help="don't truncate error text")

    show = commands.add_parser(
        "show",
        help="print every record for one item, oldest first",
        description="Print every record for one item, oldest first. Output, prompt and metadata "
        "appear only with their flags. Error text is shown and may contain prompt fragments.",
    )
    show.add_argument("store")
    show.add_argument("item_id")
    show.add_argument("--output", action="store_true", help="include the stored output (JSON)")
    show.add_argument("--prompt", action="store_true", help="include the stored prompt, if any")
    show.add_argument("--metadata", action="store_true", help="include the stored metadata")
    show.add_argument("--full", action="store_true", help="don't truncate error text")
    _add_prices(show)
    show.add_argument("--json", action="store_true", help="print one JSON document")
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Run the CLI and return its exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    out = stdout if stdout is not None else sys.stdout
    try:
        if args.command == "failures":
            return _run_failures(args, out)
        try:
            prices = _prices(args, parser)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else EXIT_USAGE
        if args.command == "inspect":
            return _run_inspect(args, prices, out)
        return _run_show(args, prices, out)
    except _CommandError as exc:
        _error(str(exc))
        return exc.code
    except (ArtifactIOError, ArtifactFormatError) as exc:
        _error(escape(str(exc)))
        return EXIT_STORE
    except BrokenPipeError:
        # Reader went away (e.g. `| head`); stop quietly without a traceback at exit.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return EXIT_OK


def _error(message: str) -> None:
    sys.stderr.write(f"llm-minder: error: {message}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
