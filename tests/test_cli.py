"""Tests for the ``llm-minder`` store-inspection CLI (#318).

Every command runs against both backends (AGENTS.md, "The two artifact stores
must agree") and against the legacy fixtures. JSON output is checked against the
shapes documented in docs/results-and-artifacts.md, "Inspecting a store from the
command line".
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from llm_minder import (
    ArtifactIdentity,
    AttemptTiming,
    BatchResult,
    CachedTokenRates,
    WorkItemResult,
    WorkItemTiming,
    cli,
)
from llm_minder._internal import store_reader
from llm_minder.serialization import _attempt_to_dict
from tests.test_store_reader import (
    CONTROL_IDS,
    FIXTURES,
    HISTORY,
    KINDS,
    LEGACY,
    _blank_sqlite,
    _foreign_wal_sqlite,
    _future_jsonl,
    _future_sqlite,
    _result,
    write,
)

PRICES = ["--input-price", "1.25", "--output-price", "10", "--cached-token-rate", "0.5"]


class Sink(io.TextIOBase):
    """A stdout that keeps nothing (for memory tests) or everything (default)."""

    def __init__(self, keep: bool = True) -> None:
        self.keep = keep
        self.parts: list[str] = []
        self.written = 0

    def write(self, text: str) -> int:  # type: ignore[override]
        self.written += len(text)
        if self.keep:
            self.parts.append(text)
        return len(text)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def run(*argv: str | Path, keep: bool = True) -> tuple[int, str, Sink]:
    sink = Sink(keep)
    code = cli.main([str(a) for a in argv], stdout=sink)
    return code, sink.text, sink


def run_json(*argv: str | Path) -> dict[str, Any]:
    code, out, _ = run(*argv, "--json")
    assert code == 0, out
    assert out.endswith("\n") and out.count("\n") == 1
    return json.loads(out)


@pytest.fixture(params=KINDS)
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def store(tmp_path: Path, kind: str) -> Path:
    return write(tmp_path / f"store.{kind}", kind, HISTORY, include_prompt=True)


# --- inspect -----------------------------------------------------------------------


def test_inspect_json_shape(store: Path, kind: str) -> None:
    doc = run_json("inspect", store)
    assert list(doc) == [
        "schema_version",
        "command",
        "store",
        "manifest",
        "record_count",
        "item_count",
        "identities",
        "items_by_newest_record",
        "token_totals",
        "recorded_cost",
        "estimated_cost",
        "guardrail_stops",
    ]
    assert doc["schema_version"] == 1 and doc["command"] == "inspect"
    assert doc["store"] == {"path": str(store), "kind": kind}
    manifest = doc["manifest"]
    assert set(manifest) == {
        "artifact_schema_version",
        "sqlite_schema_version",
        "created_at",
        "package_version",
        "identity_fingerprint",
        "identity",
        "user_metadata_keys",
    }
    assert (manifest["sqlite_schema_version"] is None) == (kind == "jsonl")
    assert manifest["identity"] == {
        "provider": "fixture",
        "model": "fixture-model",
        "prompt_version": "1",
        "parser_version": None,
        "application_version": None,
        "extra_keys": [],
    }
    assert doc["record_count"] == len(HISTORY)
    assert doc["item_count"] == 6
    assert doc["identities"] == [
        {"provider": "fixture", "model": "fixture-model", "prompt_version": "1", "records": 7}
    ]
    # "a" failed first, then succeeded: classified by its newest record.
    assert doc["items_by_newest_record"] == {
        "succeeded": 2,
        "failed": [{"error_category": "validation_error", "items": 1}],
        "stopped_by_guardrail": [
            {"error_category": "batch_deadline_exceeded", "items": 2},
            {"error_category": "framework_total_item_timeout", "items": 1},
        ],
    }
    # Tokens are summed over all records, including a's failed first attempt (5).
    assert doc["token_totals"]["input_tokens"] == 5 + 1 + 7 + 1 + 1 + 1 + 1
    assert set(doc["token_totals"]) == set(cli.TOKEN_KEYS)
    assert doc["token_totals"]["cache_write_input_tokens"] == 0
    assert doc["recorded_cost"] == {"amount": None, "records_with_cost": 0, "records": 7}
    assert doc["estimated_cost"] is None
    assert doc["guardrail_stops"] == [
        {
            "error_category": "batch_deadline_exceeded",
            "count": 2,
            "first_reason": "Deadline hit",
            "first_reason_truncated": False,
        },
        {
            "error_category": "framework_total_item_timeout",
            "count": 1,
            "first_reason": "Item timed out",
            "first_reason_truncated": False,
        },
    ]


def test_backends_agree(tmp_path: Path) -> None:
    docs = {}
    for kind in KINDS:
        path = write(tmp_path / f"s.{kind}", kind, HISTORY)
        doc = run_json("inspect", path)
        for key in ("store", "manifest"):
            doc.pop(key)
        failures = run_json("failures", path)
        failures.pop("store")
        for entry in failures["failures"]:
            entry.pop("sequence")
            entry.pop("recorded_at")
        docs[kind] = (doc, failures)
    assert docs["jsonl"] == docs["sqlite"]


def test_inspect_estimate_matches_batch_result(tmp_path: Path, kind: str) -> None:
    tokens = {"input_tokens": 1_000_000, "cached_input_tokens": 400_000, "output_tokens": 200_000}
    path = write(tmp_path / f"s.{kind}", kind, [_result("x", tokens=tokens)])
    doc = run_json("inspect", path, *PRICES)
    expected = BatchResult(
        [WorkItemResult(item_id="x", success=True, token_usage=tokens)]  # type: ignore[arg-type]
    ).estimated_cost(1.25, 10.0, 0.5)
    assert doc["estimated_cost"] == {
        "amount": expected,
        "input_price": 1.25,
        "output_price": 10.0,
        "cached_token_rate": 0.5,
        "cache_write_rate": None,
    }
    assert expected == 3.0


def test_rate_names_resolve(store: Path) -> None:
    doc = run_json(
        "inspect",
        store,
        "--input-price",
        "1",
        "--output-price",
        "1",
        "--cached-token-rate",
        "anthropic-read",
        "--cache-write-rate",
        "anthropic-write-1h",
    )
    assert doc["estimated_cost"]["cached_token_rate"] == CachedTokenRates.ANTHROPIC_READ
    assert doc["estimated_cost"]["cache_write_rate"] == CachedTokenRates.ANTHROPIC_WRITE_1H


def test_recorded_cost_partial(tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path / f"s.{kind}", kind, HISTORY[:3])
    real = store_reader._validated_record

    def with_cost(**fields: Any) -> Any:
        if fields["item_id"] == "b":
            fields["calculated_cost"] = 0.25
        return real(**fields)

    monkeypatch.setattr(store_reader, "_validated_record", with_cost)
    doc = run_json("inspect", path)
    assert doc["recorded_cost"] == {"amount": 0.25, "records_with_cost": 1, "records": 3}


def test_recorded_cost_of_zero_is_not_null(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path / f"s.{kind}", kind, HISTORY[:2])
    real = store_reader._validated_record

    def zero_cost(**fields: Any) -> Any:
        fields["calculated_cost"] = 0.0 if fields["item_id"] == "b" else None
        return real(**fields)

    monkeypatch.setattr(store_reader, "_validated_record", zero_cost)
    doc = run_json("inspect", path)
    assert doc["recorded_cost"] == {"amount": 0.0, "records_with_cost": 1, "records": 2}


def test_identity_extra_values_never_printed(tmp_path: Path, kind: str) -> None:
    identity = ArtifactIdentity(
        provider="p", model="m", extra={"system_prompt": "PRIVATE PROMPT", "temperature": 0.2}
    )
    path = _write_with_identity(tmp_path / f"s.{kind}", kind, identity)
    doc = run_json("inspect", path)
    assert doc["manifest"]["identity"]["extra_keys"] == ["system_prompt", "temperature"]
    for argv in (("inspect", path), ("inspect", path, "--json"), ("show", path, "b", "--json")):
        code, out, _ = run(*argv)
        assert code == 0
        assert "PRIVATE PROMPT" not in out


def _write_with_identity(path: Path, kind: str, identity: ArtifactIdentity) -> Path:
    import asyncio

    from llm_minder import LLMWorkItem
    from llm_minder.testing import FakeStrategy
    from tests.test_store_reader import STORE_CLASSES

    async def go() -> None:
        store = STORE_CLASSES[kind](path, identity=identity)
        try:
            for result in HISTORY[:2]:
                item = LLMWorkItem(item_id=result.item_id, strategy=FakeStrategy("o"), prompt="p")
                prepared = await store.prepare_item(item)
                await store.append(item, prepared, result)
        finally:
            await store.close()

    asyncio.run(go())
    return path


def test_inspect_human_output(store: Path) -> None:
    code, out, _ = run("inspect", store)
    assert code == 0
    assert "Items by newest record per item id" in out
    assert "summed over all records" in out
    assert "Recorded guardrail stops" in out


# --- failures ----------------------------------------------------------------------


def test_failures_json_shape_and_guardrail_exclusion(store: Path, kind: str) -> None:
    doc = run_json("failures", store)
    assert list(doc) == [
        "schema_version",
        "command",
        "store",
        "categories",
        "total",
        "shown",
        "excluded_guardrail_stops",
        "failures",
    ]
    assert doc["command"] == "failures" and doc["categories"] is None
    assert (doc["total"], doc["shown"], doc["excluded_guardrail_stops"]) == (1, 1, 3)
    (entry,) = doc["failures"]
    assert list(entry) == [
        "item_id",
        "sequence",
        "recorded_at",
        "error_category",
        "error",
        "error_truncated",
        "provider",
        "model",
        "prompt_version",
    ]
    assert entry["item_id"] == "c" and entry["error"] == "bad json"
    assert entry["error_truncated"] is False


def test_failures_category_names_guardrail_items(store: Path) -> None:
    doc = run_json("failures", store, "--category", "batch_deadline_exceeded")
    assert doc["categories"] == ["batch_deadline_exceeded"]
    assert [e["item_id"] for e in doc["failures"]] == ["d", "e"]
    assert doc["excluded_guardrail_stops"] == 1  # f's category was not named
    doc = run_json(
        "failures",
        store,
        "--category",
        "framework_total_item_timeout",
        "--category",
        "batch_deadline_exceeded",
    )
    assert doc["categories"] == ["framework_total_item_timeout", "batch_deadline_exceeded"]
    assert [e["item_id"] for e in doc["failures"]] == ["d", "e", "f"]
    assert doc["excluded_guardrail_stops"] == 0


def test_failures_newest_record_classifies(store: Path) -> None:
    # "a" failed (server_error) and then succeeded: not a failure.
    doc = run_json("failures", store, "--category", "server_error")
    assert doc["total"] == 0 and doc["failures"] == []


def test_failures_limit_and_order(tmp_path: Path, kind: str) -> None:
    entries = [
        _result(f"i{n}", success=False, category="server_error", error="x") for n in range(5)
    ]
    path = write(tmp_path / f"s.{kind}", kind, entries)
    doc = run_json("failures", path, "--limit", "2")
    assert (doc["total"], doc["shown"]) == (5, 2)
    assert [e["item_id"] for e in doc["failures"]] == ["i0", "i1"]
    sequences = [e["sequence"] for e in run_json("failures", path)["failures"]]
    assert sequences == sorted(sequences)
    code, out, _ = run("failures", path, "--ids-only", "--limit", "3")
    assert (code, out) == (0, "i0\ni1\ni2\n")
    code, out, _ = run("failures", path, "--limit", "1")
    assert code == 0 and "5 failed items" in out and "showing the first 1" in out


def test_failures_truncation(tmp_path: Path, kind: str) -> None:
    long = "e" * 250
    path = write(
        tmp_path / f"s.{kind}", kind, [_result("x", success=False, category="c", error=long)]
    )
    (entry,) = run_json("failures", path)["failures"]
    assert entry["error"] == "e" * 200 and entry["error_truncated"] is True
    (entry,) = run_json("failures", path, "--full")["failures"]
    assert entry["error"] == long and entry["error_truncated"] is False
    code, out, _ = run("failures", path)
    assert "e" * 201 not in out and "truncated" in out


def test_failures_lists_only_the_newest_record(tmp_path: Path, kind: str) -> None:
    entries = [
        _result("x", success=False, category="server_error", error="first"),
        _result("x", success=False, category="server_error", error="second"),
    ]
    path = write(tmp_path / f"s.{kind}", kind, entries)
    doc = run_json("failures", path)
    assert [e["error"] for e in doc["failures"]] == ["second"]
    assert run("failures", path, "--ids-only")[:2] == (0, "x\n")


def test_truncation_boundary() -> None:
    assert cli.truncate("e" * 200, False) == ("e" * 200, False)
    assert cli.truncate("e" * 201, False) == ("e" * 200, True)
    assert cli.truncate("e" * 201, True) == ("e" * 201, False)
    assert cli.truncate(None, False) == (None, False)


def test_failures_human_output(store: Path) -> None:
    code, out, _ = run("failures", store)
    assert code == 0
    assert out.splitlines()[1] == "c\tvalidation_error\tbad json"
    assert "3 item(s) stopped by a guardrail are not listed" in out


# --- Item ids ----------------------------------------------------------------------


@pytest.mark.parametrize("item_id", CONTROL_IDS)
def test_control_character_ids(tmp_path: Path, kind: str, item_id: str) -> None:
    path = write(
        tmp_path / f"s.{kind}",
        kind,
        [_result(item_id, success=False, category="c", error=f"err {item_id}")],
    )
    code, out, _ = run("failures", path, "--ids-only")
    assert code == cli.EXIT_UNREPRESENTABLE_ID and out == ""
    (entry,) = run_json("failures", path)["failures"]
    assert entry["item_id"] == item_id
    assert run_json("show", path, item_id)["item_id"] == item_id
    for argv in (("failures", path), ("show", path, item_id), ("inspect", path)):
        code, out, _ = run(*argv)
        assert code == 0
        assert not any(c in out.replace("\n", "") for c in "\r\x1b\x00")
        assert item_id not in out


def test_ids_only_checks_only_printed_ids(tmp_path: Path, kind: str) -> None:
    entries = [
        _result("fine", success=False, category="c", error="x"),
        _result("bad\nid", success=False, category="c", error="x"),
    ]
    path = write(tmp_path / f"s.{kind}", kind, entries)
    assert run("failures", path, "--ids-only", "--limit", "1")[:2] == (0, "fine\n")
    code, out, _ = run("failures", path, "--ids-only")
    assert code == 5 and out == ""


def test_escape() -> None:
    assert cli.escape("a\nb\r\x1b[31m\x00\\  ") == "a\\nb\\r\\x1b[31m\\x00\\\\ \\u2028"


# --- show --------------------------------------------------------------------------


def test_show_json_shape(store: Path, kind: str) -> None:
    doc = run_json("show", store, "a")
    assert list(doc) == ["schema_version", "command", "store", "item_id", "record_count", "records"]
    assert doc["record_count"] == 2
    first, second = doc["records"]
    assert first["sequence"] < second["sequence"]
    assert (first["is_newest"], second["is_newest"]) == (False, True)
    assert list(first) == [
        "sequence",
        "is_newest",
        "recorded_at",
        "success",
        "error_category",
        "error",
        "error_truncated",
        "exception",
        "identity",
        "token_usage",
        "calculated_cost",
        "estimated_cost",
        "submission_index",
        "replayed_from_artifact",
        "timing",
    ]
    assert first["token_usage"] == {"input_tokens": 5}
    assert first["error"] == "boom" and first["success"] is False
    assert first["identity"]["provider"] == "fixture"
    assert set(first["timing"]) == {"total_seconds", "timeout_category", "attempts"}
    assert first["estimated_cost"] is None


def test_show_flag_gated_keys(store: Path) -> None:
    plain = run_json("show", store, "b")["records"][0]
    for key in (
        "output",
        "output_included",
        "prompt",
        "prompt_stored",
        "metadata",
        "metadata_included",
    ):
        assert key not in plain
    full = run_json("show", store, "b", "--output", "--prompt", "--metadata")["records"][0]
    assert full["output_included"] is True and full["output"] == "out"
    assert full["prompt_stored"] is True and full["prompt"] == "prompt for b"
    assert "metadata_included" in full and "metadata" in full


def test_show_prompt_not_stored(tmp_path: Path, kind: str) -> None:
    path = write(tmp_path / f"s.{kind}", kind, HISTORY[:2])
    record = run_json("show", path, "b", "--prompt")["records"][0]
    assert record["prompt_stored"] is False and record["prompt"] is None
    code, out, _ = run("show", path, "b", "--prompt")
    assert "prompt: not stored" in out


def test_show_attempts_have_every_key(tmp_path: Path, kind: str) -> None:
    result = WorkItemResult(
        item_id="t",
        success=True,
        output="o",
        timing=WorkItemTiming(
            total_seconds=1.5, attempts=[AttemptTiming(attempt=1, try_number=1, total_seconds=1.5)]
        ),
    )
    path = write(tmp_path / f"s.{kind}", kind, [result])
    (attempt,) = run_json("show", path, "t")["records"][0]["timing"]["attempts"]
    assert tuple(attempt) == cli.ATTEMPT_KEYS
    assert attempt["total_seconds"] == 1.5


def test_attempt_keys_match_serializer() -> None:
    assert tuple(_attempt_to_dict(AttemptTiming(attempt=1, try_number=1))) == cli.ATTEMPT_KEYS


def test_show_missing_attempt_keys_are_null(tmp_path: Path) -> None:
    path = write(tmp_path / "s.jsonl", "jsonl", [_result("t")])
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["result"]["timing"]["attempts"] = [{"attempt": 1, "try_number": 1}]
    path.write_text("\n".join([lines[0], json.dumps(record)]) + "\n")
    (attempt,) = run_json("show", path, "t")["records"][0]["timing"]["attempts"]
    assert attempt["attempt"] == 1 and attempt["quota_scope_id"] is None
    assert tuple(attempt) == cli.ATTEMPT_KEYS


def test_show_estimate_per_record(store: Path) -> None:
    records = run_json("show", store, "a", *PRICES)["records"]
    assert records[0]["estimated_cost"] == pytest.approx(5 / 1e6 * 1.25)
    assert records[1]["estimated_cost"] == pytest.approx(7 / 1e6 * 1.25 + 3 / 1e6 * 10)


def test_show_not_found_exit_and_stdout(store: Path) -> None:
    code, out, _ = run("show", store, "nope", "--json")
    assert (code, out) == (cli.EXIT_NOT_FOUND, "")


def test_show_human_output(store: Path) -> None:
    code, out, _ = run("show", store, "a", "--output", *PRICES)
    assert code == 0
    assert "Record 1 (sequence" in out and ", newest" in out
    assert "error: boom" in out and "estimate:" in out


def test_show_truncation(tmp_path: Path, kind: str) -> None:
    path = write(
        tmp_path / f"s.{kind}", kind, [_result("x", success=False, category="c", error="z" * 300)]
    )
    record = run_json("show", path, "x")["records"][0]
    assert record["error"] == "z" * 200 and record["error_truncated"] is True
    record = run_json("show", path, "x", "--full")["records"][0]
    assert record["error"] == "z" * 300 and record["error_truncated"] is False


# --- Estimate validity and usage errors --------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ["--input-price", "1", "--output-price", "1", "--cached-token-rate", "1.5"],
        ["--input-price", "1", "--output-price", "1", "--cached-token-rate", "-0.1"],
        ["--input-price", "1", "--output-price", "1", "--cached-token-rate", "nan"],
        ["--input-price", "1", "--output-price", "1", "--cached-token-rate", "openai-write"],
        ["--input-price", "1", "--output-price", "1", "--cached-token-rate", "bogus"],
        ["--input-price", "-1", "--output-price", "1", "--cached-token-rate", "0.5"],
        ["--input-price", "inf", "--output-price", "1", "--cached-token-rate", "0.5"],
        ["--input-price", "1", "--output-price", "nan", "--cached-token-rate", "0.5"],
        [
            "--input-price",
            "1",
            "--output-price",
            "1",
            "--cached-token-rate",
            "0.5",
            "--cache-write-rate",
            "-1",
        ],
        [
            "--input-price",
            "1",
            "--output-price",
            "1",
            "--cached-token-rate",
            "0.5",
            "--cache-write-rate",
            "inf",
        ],
        [
            "--input-price",
            "1",
            "--output-price",
            "1",
            "--cached-token-rate",
            "0.5",
            "--cache-write-rate",
            "openai",
        ],
        ["--input-price", "1", "--output-price", "1"],
        ["--input-price", "1", "--cached-token-rate", "0.5"],
        ["--cached-token-rate", "0.5"],
        ["--cache-write-rate", "1.25"],
    ],
)
@pytest.mark.parametrize("command", ["inspect", "show"])
def test_invalid_prices_are_usage_errors_before_opening(
    tmp_path: Path, flags: list[str], command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist.jsonl"
    argv = [command, missing] + (["x"] if command == "show" else []) + flags
    code, out, _ = run(*argv)
    assert (code, out) == (cli.EXIT_USAGE, "")  # not 3: the store was never opened
    assert "error" in capsys.readouterr().err


def test_boundary_rates_accepted(store: Path) -> None:
    for rate in ("0", "1"):
        doc = run_json(
            "inspect",
            store,
            "--input-price",
            "0",
            "--output-price",
            "0",
            "--cached-token-rate",
            rate,
        )
        assert doc["estimated_cost"]["amount"] == 0.0


def test_unrepresentable_estimate_is_exit_2_before_stdout(tmp_path: Path, kind: str) -> None:
    path = write(
        tmp_path / f"s.{kind}",
        kind,
        [
            _result("ok", tokens={"input_tokens": 1}),
            _result("big", tokens={"input_tokens": 2_000_000}),
        ],
    )
    huge = ["--input-price", "1e308", "--output-price", "0", "--cached-token-rate", "0.5"]
    for argv in (
        ("inspect", path),
        ("inspect", path, "--json"),
        ("show", path, "big"),
        ("show", path, "big", "--json"),
    ):
        code, out, _ = run(*argv, *huge)
        assert (code, out) == (cli.EXIT_USAGE, ""), argv


def test_show_estimate_failure_on_later_record_prints_nothing(tmp_path: Path, kind: str) -> None:
    entries = [
        _result("x", tokens={"input_tokens": 1}),
        _result("x", tokens={"input_tokens": 2_000_000}),
    ]
    path = write(tmp_path / f"s.{kind}", kind, entries)
    huge = ["--input-price", "1e308", "--output-price", "0", "--cached-token-rate", "0.5"]
    assert run("show", path, "x", "--json", *huge)[:2] == (cli.EXIT_USAGE, "")


def test_estimate_overflow_error_is_mapped() -> None:
    prices = cli.Prices(1.0, 1.0, 0.5, None)
    with pytest.raises(cli._CommandError) as info:
        cli.estimate({"cached_input_tokens": 10**400, "input_tokens": 10**400}, prices, "here")
    assert info.value.code == cli.EXIT_USAGE


@pytest.mark.parametrize(
    "argv",
    [
        ["failures", "S", "--ids-only", "--json"],
        ["failures", "S", "--limit", "0"],
        ["failures", "S", "--limit", "x"],
        ["inspect"],
        [],
        ["bogus", "S"],
    ],
)
def test_usage_errors(argv: list[str], tmp_path: Path) -> None:
    argv = [str(tmp_path / "s.jsonl") if a == "S" else a for a in argv]
    assert run(*argv)[:2] == (cli.EXIT_USAGE, "")


def test_non_finite_stored_number_is_a_store_error(tmp_path: Path) -> None:
    path = write(tmp_path / "s.jsonl", "jsonl", [_result("t")])
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace('"total_seconds":0.0', '"total_seconds":NaN', 1)
    assert "NaN" in lines[1]
    path.write_text("\n".join(lines) + "\n")
    assert run("show", path, "t", "--json")[:2] == (cli.EXIT_STORE, "")


# --- Store handling ----------------------------------------------------------------


def _missing(path: Path) -> None:
    pass


def _directory(path: Path) -> None:
    path.mkdir()


def _empty(path: Path) -> None:
    path.write_bytes(b"")


def _foreign_text(path: Path) -> None:
    path.write_text("hello\nworld\n")


@pytest.mark.parametrize(
    "make",
    [
        _missing,
        _directory,
        _empty,
        _foreign_text,
        _foreign_wal_sqlite,
        _future_jsonl,
        _future_sqlite,
        _blank_sqlite,
    ],
)
@pytest.mark.parametrize("command", [["inspect"], ["failures"], ["show", "a"]])
def test_rejected_stores_exit_3(tmp_path: Path, make: Any, command: list[str]) -> None:
    path = tmp_path / "input"
    make(path)
    argv = [command[0], path, *command[1:]]
    code, out, _ = run(*argv, "--json")
    assert (code, out) == (cli.EXIT_STORE, "")


def test_rejected_text_input_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    _foreign_text(path)
    run("inspect", path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.jsonl"]


def test_mid_store_corruption_leaves_stdout_empty(tmp_path: Path) -> None:
    entries = [_result(f"i{n}", success=False, category="c", error="x") for n in range(4)]
    path = write(tmp_path / "s.jsonl", "jsonl", entries)
    lines = path.read_bytes().splitlines(keepends=True)
    lines[3] = b"{not json\n"
    path.write_bytes(b"".join(lines))
    for argv in (
        ("failures", path),
        ("failures", path, "--json"),
        ("show", path, "i0"),
        ("inspect", path),
    ):
        assert run(*argv)[:2] == (cli.EXIT_STORE, ""), argv


def test_late_corruption_reports_store_error(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = cli._stream

    def corrupt(snapshot: Any) -> Any:
        yield next(iter(real(snapshot)))
        raise cli._CommandError(cli.EXIT_STORE, "changed")

    monkeypatch.setattr(cli, "_stream", corrupt)
    code, _out, _ = run("show", store, "a")
    assert code == cli.EXIT_STORE and "changed" in capsys.readouterr().err


def test_stream_maps_decode_errors(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def records(self) -> Any:
            raise store_reader.ArtifactFormatError("bad bytes")
            yield

    with pytest.raises(cli._CommandError, match="store changed during the command"):
        list(cli._stream(Broken()))  # type: ignore[arg-type]


def _tree(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


ALL_COMMANDS = [
    ["inspect"],
    ["inspect", "--json"],
    ["failures"],
    ["failures", "--json"],
    ["failures", "--ids-only"],
    ["show", "a", "--output", "--prompt", "--metadata"],
]


def test_jsonl_is_read_only(tmp_path: Path) -> None:
    path = write(tmp_path / "s.jsonl", "jsonl", HISTORY)
    before = _tree(tmp_path)
    for command in ALL_COMMANDS:
        assert run(command[0], path, *command[1:])[0] == 0
    assert _tree(tmp_path) == before


def test_sqlite_is_read_only(tmp_path: Path) -> None:
    path = write(tmp_path / "s.sqlite", "sqlite", HISTORY)
    main_before = (path.read_bytes(), path.stat().st_mtime_ns)
    for command in ALL_COMMANDS:
        assert run(command[0], path, *command[1:])[0] == 0
    assert (path.read_bytes(), path.stat().st_mtime_ns) == main_before
    extra = {p.name for p in tmp_path.iterdir()} - {"s.sqlite"}
    assert extra <= {"s.sqlite-wal", "s.sqlite-shm"}
    wal = tmp_path / "s.sqlite-wal"
    assert not wal.exists() or wal.stat().st_size == 0
    assert not (tmp_path / "s.sqlite-journal").exists()


def test_concurrent_append_reports_starting_snapshot(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path / f"s.{kind}", kind, HISTORY)
    real = cli.summarize

    def append_then_summarize(snapshot: Any) -> Any:
        write(path, kind, [_result("late", success=False, category="server_error", error="late")])
        return real(snapshot)

    monkeypatch.setattr(cli, "summarize", append_then_summarize)
    failures = run_json("failures", path, "--category", "server_error")
    assert failures["total"] == 0 and failures["failures"] == []
    inspect = run_json("inspect", path)
    assert inspect["record_count"] == len(HISTORY) + 1  # the first append is now in the snapshot
    monkeypatch.setattr(cli, "summarize", real)
    assert run_json("inspect", path)["record_count"] == len(HISTORY) + 2


# --- Memory --------------------------------------------------------------------------


def _peak(argv: list[Any]) -> tuple[int, int]:
    tracemalloc.start()
    try:
        code, _out, sink = run(*argv, keep=False)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert code == 0
    return peak, sink.written


@pytest.mark.parametrize("kind_", KINDS)
@pytest.mark.parametrize("command", ["failures", "show"])
def test_two_pass_output_memory_is_bounded(tmp_path: Path, kind_: str, command: str) -> None:
    """Peak memory doesn't grow with the number of printed records.

    Pass 2 streams. Growing the printed output fourfold may grow the peak only by
    the grouping state (one small entry per extra item id, for ``failures``),
    far less than the extra output. A SQLite page of records is the floor.
    """
    error = "q" * 10_000

    def measure(count: int) -> tuple[int, int]:
        if command == "failures":
            entries = [
                _result(f"i{n}", success=False, category="c", error=error) for n in range(count)
            ]
            argv: list[Any] = ["failures", None, "--json", "--full"]
        else:
            entries = [
                _result("same", success=False, category="c", error=error) for _ in range(count)
            ]
            argv = ["show", None, "same", "--json", "--full"]
        fast = {"commit_interval_seconds": 0.0} if kind_ == "sqlite" else {}
        argv[1] = write(tmp_path / f"{count}.{kind_}", kind_, entries, **fast)
        return _peak(argv)

    small_peak, small_written = measure(200)
    large_peak, large_written = measure(800)
    assert large_written > 800 * 10_000
    assert large_peak < large_written / 4, (large_peak, large_written)
    assert large_peak - small_peak < (large_written - small_written) / 10, (small_peak, large_peak)


# --- Legacy fixtures and entry points ------------------------------------------------


@pytest.mark.parametrize(("release", "name"), LEGACY)
def test_legacy_fixtures(tmp_path: Path, release: str, name: str) -> None:
    path = tmp_path / name
    shutil.copy(FIXTURES / release / name, path)
    doc = run_json("inspect", path)
    assert doc["manifest"]["identity"]["provider"] == "fixture"
    failures = run_json("failures", path)
    assert "bad" in {e["item_id"] for e in failures["failures"]}
    show = run_json("show", path, "ok1", "--output", "--prompt", "--metadata", *PRICES)
    for record in show["records"]:
        assert (
            tuple(record["timing"]["attempts"][0]) == cli.ATTEMPT_KEYS
            if record["timing"]["attempts"]
            else True
        )
    for command in (["inspect"], ["failures", "--ids-only"], ["show", "bad"]):
        assert run(command[0], path, *command[1:])[0] == 0


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_minder import __version__

    assert run("--version")[0] == 0
    assert capsys.readouterr().out == f"llm-minder {__version__}\n"


def test_python_dash_m(store: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "llm_minder", "failures", str(store), "--ids-only"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    assert (proc.returncode, proc.stdout) == (0, "c\n")


def test_json_never_contains_nan() -> None:
    with pytest.raises(ValueError):
        cli._dumps({"x": float("inf")})


def test_failures_ids_only_sqlite_uses_no_journal(tmp_path: Path) -> None:
    path = write(tmp_path / "s.sqlite", "sqlite", HISTORY)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    connection.close()
    assert run("failures", path, "--ids-only")[:2] == (0, "c\n")
