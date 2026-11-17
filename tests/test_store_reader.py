"""Tests for the private read-only store reader behind the inspection CLI (#318).

Every behavior is checked on both backends: the JSONL and SQLite stores must
agree (AGENTS.md, "The two artifact stores must agree").
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import stat
from pathlib import Path
from typing import Any

import pytest

from llm_minder import (
    ArtifactFormatError,
    ArtifactIdentity,
    ArtifactIOError,
    JsonlArtifactStore,
    LLMWorkItem,
    SqliteArtifactStore,
    WorkItemResult,
)
from llm_minder._internal.artifact_codec import GUARDRAIL_AUDIT_CATEGORIES
from llm_minder._internal.store_reader import (
    SQLITE_PAGE_SIZE,
    StoredRecord,
    detect_kind,
    open_snapshot,
    summarize,
)
from llm_minder.testing import FakeStrategy

IDENTITY = ArtifactIdentity(provider="fixture", model="fixture-model", prompt_version="1")
FIXTURES = Path(__file__).parent / "fixtures"
KINDS = ["jsonl", "sqlite"]
STORE_CLASSES: dict[str, Any] = {"jsonl": JsonlArtifactStore, "sqlite": SqliteArtifactStore}
CONTROL_IDS = ["line\nbreak", "carriage\rreturn", "escape\x1b[31m", "nul\x00byte"]


def _result(
    item_id: str,
    *,
    success: bool = True,
    category: str | None = None,
    error: str | None = None,
    tokens: dict[str, int] | None = None,
    output: Any = "out",
) -> WorkItemResult[Any, Any]:
    return WorkItemResult(
        item_id=item_id,
        success=success,
        output=output if success else None,
        error=error if not success else None,
        error_category=category,
        token_usage=tokens or {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},  # type: ignore[typeddict-item]
    )


def _store_path(tmp_path: Path, kind: str, name: str = "store") -> Path:
    return tmp_path / f"{name}.{kind}"


async def _write(
    path: Path,
    kind: str,
    entries: list[WorkItemResult[Any, Any]],
    **store_kwargs: Any,
) -> None:
    store = STORE_CLASSES[kind](path, identity=IDENTITY, **store_kwargs)
    try:
        for result in entries:
            item = LLMWorkItem(
                item_id=result.item_id,
                strategy=FakeStrategy("out"),
                prompt=f"prompt for {result.item_id}",
            )
            prepared = await store.prepare_item(item)
            await store.append(item, prepared, result)
    finally:
        await store.close()


def write(path: Path, kind: str, entries: list[WorkItemResult[Any, Any]], **kw: Any) -> Path:
    asyncio.run(_write(path, kind, entries, **kw))
    return path


def records(path: Path) -> list[StoredRecord]:
    with open_snapshot(path) as snapshot:
        return list(snapshot.records())


def _comparable(record: StoredRecord) -> dict[str, Any]:
    """Backend-neutral fields (sequences and timestamps differ by design)."""
    return {
        "item_id": record.item_id,
        "success": record.success,
        "error_category": record.error_category,
        "identity": (record.provider, record.model, record.prompt_version),
        "identity_fingerprint": record.identity_fingerprint,
        "calculated_cost": record.calculated_cost,
        "token_usage": dict(record.token_usage),
        "error": record.error,
        "output": record.result.get("output"),
        "prompt": record.prompt,
    }


HISTORY = [
    _result("a", success=False, category="server_error", error="boom", tokens={"input_tokens": 5}),
    _result("b"),
    _result("a", tokens={"input_tokens": 7, "output_tokens": 3}),
    _result("c", success=False, category="validation_error", error="bad json"),
    _result("d", success=False, category="batch_deadline_exceeded", error="Deadline hit"),
    _result("e", success=False, category="batch_deadline_exceeded", error="Deadline again"),
    _result("f", success=False, category="framework_total_item_timeout", error="Item timed out"),
]


# --- Records and manifest ----------------------------------------------------------


def test_both_backends_yield_the_same_records(tmp_path: Path) -> None:
    jsonl = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY, include_prompt=True)
    sqlite = write(_store_path(tmp_path, "sqlite"), "sqlite", HISTORY, include_prompt=True)

    jsonl_records = records(jsonl)
    sqlite_records = records(sqlite)

    assert [_comparable(r) for r in jsonl_records] == [_comparable(r) for r in sqlite_records]
    assert [r.item_id for r in jsonl_records] == [r.item_id for r in HISTORY]
    assert jsonl_records[0].prompt == "prompt for a"
    for found in (jsonl_records, sqlite_records):
        sequences = [r.sequence for r in found]
        assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)


@pytest.mark.parametrize("kind", KINDS)
def test_manifest(tmp_path: Path, kind: str) -> None:
    path = write(_store_path(tmp_path, kind), kind, HISTORY[:1], user_metadata={"run": "r1"})
    with open_snapshot(path) as snapshot:
        manifest = snapshot.manifest
    assert manifest.kind == kind
    assert manifest.artifact_schema_version == 1
    assert manifest.sqlite_schema_version == (1 if kind == "sqlite" else None)
    assert manifest.identity["provider"] == "fixture"
    assert manifest.identity_fingerprint
    assert manifest.user_metadata == {"run": "r1"}
    assert manifest.package_version and manifest.created_at
    assert detect_kind(path) == kind


@pytest.mark.parametrize("kind", KINDS)
def test_prompt_absent_unless_stored(tmp_path: Path, kind: str) -> None:
    path = write(_store_path(tmp_path, kind), kind, HISTORY[:1])
    assert records(path)[0].prompt is None


@pytest.mark.parametrize("kind", KINDS)
def test_custom_decoders_never_run(tmp_path: Path, kind: str) -> None:
    class Payload:
        def __init__(self, value: int) -> None:
            self.value = value

    path = write(
        _store_path(tmp_path, kind),
        kind,
        [_result("x", output=Payload(3))],
        encoder=lambda value: {"payload": value.value},
    )
    assert records(path)[0].result["output"] == {"payload": 3}


@pytest.mark.parametrize("kind", KINDS)
def test_control_character_ids_round_trip(tmp_path: Path, kind: str) -> None:
    path = write(_store_path(tmp_path, kind), kind, [_result(i) for i in CONTROL_IDS])
    assert [r.item_id for r in records(path)] == CONTROL_IDS
    with open_snapshot(path) as snapshot:
        assert set(summarize(snapshot).newest) == set(CONTROL_IDS)


# --- Grouping state ----------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_summary_classifies_by_newest_record_and_sums_all(tmp_path: Path, kind: str) -> None:
    path = write(_store_path(tmp_path, kind), kind, HISTORY)
    with open_snapshot(path) as snapshot:
        summary = summarize(snapshot)

    assert summary.record_count == len(HISTORY)
    assert summary.item_count == 6
    succeeded, failed, stopped = summary.inventory()
    # "a" failed first, then succeeded: its newest record counts.
    assert succeeded == 2
    assert dict(failed) == {"validation_error": 1}
    assert dict(stopped) == {"batch_deadline_exceeded": 2, "framework_total_item_timeout": 1}
    # Tokens sum every record, including a's billed failure.
    assert summary.token_totals["input_tokens"] == 5 + 1 + 7 + 1 + 1 + 1 + 1
    assert summary.token_totals["output_tokens"] == 3 + 5 * 1
    assert summary.identities == {("fixture", "fixture-model", "1"): len(HISTORY)}
    stops = summary.guardrail_stops
    assert set(stops) == {"batch_deadline_exceeded", "framework_total_item_timeout"}
    assert (
        stops["batch_deadline_exceeded"].count,
        stops["batch_deadline_exceeded"].first_reason,
    ) == (
        2,
        "Deadline hit",
    )
    assert stops["framework_total_item_timeout"].count == 1


@pytest.mark.parametrize("kind", KINDS)
def test_is_newest_marks_one_record_per_item(tmp_path: Path, kind: str) -> None:
    path = write(_store_path(tmp_path, kind), kind, HISTORY)
    with open_snapshot(path) as snapshot:
        summary = summarize(snapshot)
        newest = [r for r in snapshot.records() if summary.is_newest(r)]
    assert [r.item_id for r in newest] == ["b", "a", "c", "d", "e", "f"]
    assert newest[1].success


@pytest.mark.parametrize("kind", KINDS)
def test_recorded_cost_counts_records_that_have_one(tmp_path: Path, kind: str) -> None:
    def cost(result: WorkItemResult[Any, Any]) -> float | None:
        return 0.25 if result.success else None

    path = write(_store_path(tmp_path, kind), kind, HISTORY[:3], cost_calculator=cost)
    with open_snapshot(path) as snapshot:
        summary = summarize(snapshot)
    assert summary.recorded_cost == pytest.approx(0.5)
    assert summary.recorded_cost_records == 2


def test_guardrail_categories_are_the_audit_set() -> None:
    from llm_minder._internal import store_reader

    assert store_reader._GUARDRAIL_CATEGORIES == frozenset(GUARDRAIL_AUDIT_CATEGORIES)


# --- Snapshots ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_snapshot_ignores_concurrent_appends(tmp_path: Path, kind: str) -> None:
    path = write(_store_path(tmp_path, kind), kind, HISTORY[:2])
    with open_snapshot(path) as snapshot:
        first = [r.item_id for r in snapshot.records()]
        write(path, kind, [_result("late1"), _result("late2")])
        second = [r.item_id for r in snapshot.records()]
        summary = summarize(snapshot)
    assert first == second == ["a", "b"]
    assert summary.record_count == 2
    assert [r.item_id for r in records(path)][-2:] == ["late1", "late2"]


@pytest.mark.parametrize("kind", KINDS)
def test_records_span_many_sqlite_pages(tmp_path: Path, kind: str) -> None:
    count = SQLITE_PAGE_SIZE * 2 + 7
    path = write(_store_path(tmp_path, kind), kind, [_result(f"i{n}") for n in range(count)])
    found = records(path)
    assert [r.item_id for r in found] == [f"i{n}" for n in range(count)]


def test_jsonl_truncated_tail_is_ignored(tmp_path: Path) -> None:
    path = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY[:2])
    with path.open("ab") as handle:
        handle.write(b'{"artifact_schema_version":1,"record_type":"item","item_id":"torn"')
    assert [r.item_id for r in records(path)] == ["a", "b"]


def test_jsonl_complete_final_line_without_newline_is_a_tail(tmp_path: Path) -> None:
    """A line without its newline is unfinished, even if it parses."""
    path = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY[:2])
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + lines[-1].rstrip(b"\n"))
    assert [r.item_id for r in records(path)] == ["a"]


def test_jsonl_line_straddling_the_snapshot_boundary_is_excluded(tmp_path: Path) -> None:
    """A record half-written when the snapshot opened stays out, even once finished."""
    path = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY[:2])
    lines = path.read_bytes().splitlines(keepends=True)
    head, last = b"".join(lines[:-1]), lines[-1]
    path.write_bytes(head + last[:10])
    with open_snapshot(path) as snapshot:
        with path.open("ab") as handle:
            handle.write(last[10:])
        assert [r.item_id for r in snapshot.records()] == ["a"]
    assert [r.item_id for r in records(path)] == ["a", "b"]


def test_jsonl_malformed_middle_line_is_an_error(tmp_path: Path) -> None:
    path = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY[:3])
    lines = path.read_bytes().splitlines(keepends=True)
    lines[2] = b"{not json\n"
    path.write_bytes(b"".join(lines))
    with open_snapshot(path) as snapshot, pytest.raises(ArtifactFormatError, match="line 3"):
        list(snapshot.records())


def test_jsonl_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY[:2])
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b"\n  \n" + b"".join(lines[1:]))
    assert [r.item_id for r in records(path)] == ["a", "b"]


# --- Record validation -------------------------------------------------------------

BAD_JSONL_FIELDS = [
    ("item_id", 5, "item_id"),
    ("success", "yes", "success"),
    ("result", [], "result"),
    ("calculated_cost", "free", "calculated_cost"),
    ("calculated_cost", True, "calculated_cost"),
    ("error_category", 3, "error_category"),
    ("recorded_at", 3, "recorded_at"),
    ("provider", 3, "provider"),
]


@pytest.mark.parametrize(("key", "value", "message"), BAD_JSONL_FIELDS)
def test_jsonl_malformed_record_fields(tmp_path: Path, key: str, value: Any, message: str) -> None:
    path = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY[:2])
    lines = path.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[2])
    record[key] = value
    lines[2] = json.dumps(record).encode() + b"\n"
    path.write_bytes(b"".join(lines))
    with open_snapshot(path) as snapshot, pytest.raises(ArtifactFormatError, match=message):
        list(snapshot.records())


BAD_RESULT_FIELDS = [
    ({"token_usage": []}, "token_usage"),
    ({"token_usage": {"input_tokens": "1"}}, "input_tokens"),
    ({"token_usage": {"input_tokens": True}}, "input_tokens"),
    ({"error": 3}, "result.error"),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("patch", "message"), BAD_RESULT_FIELDS)
def test_malformed_result_fields(
    tmp_path: Path, kind: str, patch: dict[str, Any], message: str
) -> None:
    path = write(_store_path(tmp_path, kind), kind, HISTORY[:2])
    if kind == "jsonl":
        lines = path.read_bytes().splitlines(keepends=True)
        record = json.loads(lines[2])
        record["result"].update(patch)
        lines[2] = json.dumps(record).encode() + b"\n"
        path.write_bytes(b"".join(lines))
    else:
        with sqlite3.connect(path) as connection:
            (text,) = connection.execute(
                "SELECT result_json FROM item_records ORDER BY record_sequence DESC LIMIT 1"
            ).fetchone()
            result = json.loads(text)
            result.update(patch)
            connection.execute(
                "UPDATE item_records SET result_json = ? WHERE record_sequence = "
                "(SELECT MAX(record_sequence) FROM item_records)",
                (json.dumps(result),),
            )
    with open_snapshot(path) as snapshot, pytest.raises(ArtifactFormatError, match=message):
        list(snapshot.records())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("result_json", "{not json", "result_json"),
        ("raw_prompt_json", "{not json", "raw_prompt_json"),
        ("success", 2, "success"),
        ("calculated_cost", "free", "calculated_cost"),
    ],
)
def test_sqlite_malformed_columns(tmp_path: Path, column: str, value: Any, message: str) -> None:
    path = write(_store_path(tmp_path, "sqlite"), "sqlite", HISTORY[:2])
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(f"UPDATE item_records SET {column} = ? WHERE item_id = 'b'", (value,))
    with open_snapshot(path) as snapshot, pytest.raises(ArtifactFormatError, match=message):
        list(snapshot.records())


# --- Rejected inputs ---------------------------------------------------------------


def _foreign_wal_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE t (x)")
    connection.close()


def _future_sqlite(path: Path) -> None:
    write(path, "sqlite", HISTORY[:1])
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    connection.close()


def _future_jsonl(path: Path) -> None:
    write(path, "jsonl", HISTORY[:1])
    lines = path.read_bytes().splitlines(keepends=True)
    manifest = json.loads(lines[0])
    manifest["artifact_schema_version"] = 2
    path.write_bytes(json.dumps(manifest).encode() + b"\n" + b"".join(lines[1:]))


def _blank_sqlite(path: Path) -> None:
    """A SQLite file whose llm-minder initialization never ran."""
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (x)")
    connection.execute("DROP TABLE t")
    connection.commit()
    connection.close()


def _manifest_patch(**changes: Any) -> Any:
    def make(path: Path) -> None:
        write(path, "jsonl", HISTORY[:1])
        lines = path.read_bytes().splitlines(keepends=True)
        manifest = json.loads(lines[0])
        manifest.update(changes)
        path.write_bytes(json.dumps(manifest).encode() + b"\n" + b"".join(lines[1:]))

    return make


REJECTED: list[tuple[str, Any, type[Exception], str]] = [
    ("missing", None, ArtifactIOError, "does not exist"),
    ("dir_input", lambda p: p.mkdir(), ArtifactIOError, "Artifact path is a directory"),
    ("empty", lambda p: p.write_bytes(b""), ArtifactFormatError, "empty"),
    ("foreign_text", lambda p: p.write_text("hello\nworld\n"), ArtifactFormatError, "JSON"),
    ("foreign_json", lambda p: p.write_text('{"a": 1}\n'), ArtifactFormatError, "schema"),
    (
        "torn_manifest",
        lambda p: p.write_text('{"artifact_schema_version":1'),
        ArtifactFormatError,
        "manifest",
    ),
    ("future_jsonl", _future_jsonl, ArtifactFormatError, "future"),
    ("manifest_identity", _manifest_patch(identity="x"), ArtifactFormatError, "manifest"),
    ("manifest_metadata", _manifest_patch(user_metadata=[]), ArtifactFormatError, "manifest"),
    ("manifest_created_at", _manifest_patch(created_at=5), ArtifactFormatError, "manifest"),
    ("foreign_wal_sqlite", _foreign_wal_sqlite, ArtifactFormatError, "not an llm-minder"),
    ("future_sqlite", _future_sqlite, ArtifactFormatError, "future"),
    ("blank_sqlite", _blank_sqlite, ArtifactFormatError, "blank"),
    (
        "corrupt_sqlite",
        lambda p: p.write_bytes(b"SQLite format 3\x00" + b"\xff" * 200),
        ArtifactFormatError,
        "Invalid SQLite",
    ),
]


@pytest.mark.parametrize(
    ("name", "make", "error", "message"), REJECTED, ids=[r[0] for r in REJECTED]
)
def test_rejected_inputs(
    tmp_path: Path, name: str, make: Any, error: type[Exception], message: str
) -> None:
    path = tmp_path / name
    if make is not None:
        make(path)
    with pytest.raises(error, match=message):
        with open_snapshot(path) as snapshot:
            list(snapshot.records())


@pytest.mark.parametrize(
    "make",
    [lambda p: p.write_text("hello\nworld\n"), lambda p: p.write_text('{"a": 1}\n'), _future_jsonl],
    ids=["foreign_text", "foreign_json", "future_jsonl"],
)
def test_rejected_non_sqlite_input_creates_nothing(tmp_path: Path, make: Any) -> None:
    path = tmp_path / "input"
    make(path)
    before = sorted(tmp_path.iterdir())
    with pytest.raises(ArtifactFormatError):
        open_snapshot(path)
    assert sorted(tmp_path.iterdir()) == before


# --- Read-only ---------------------------------------------------------------------


def _fingerprint(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


@pytest.mark.parametrize("kind", KINDS)
def test_reading_never_writes_the_store(tmp_path: Path, kind: str) -> None:
    path = write(_store_path(tmp_path, kind), kind, HISTORY)
    if kind == "jsonl":
        with path.open("ab") as handle:  # a torn tail must not be repaired
            handle.write(b'{"torn":')
    before = _fingerprint(path)
    siblings = sorted(p.name for p in tmp_path.iterdir())
    with open_snapshot(path) as snapshot:
        summarize(snapshot)
        list(snapshot.records())
    assert _fingerprint(path) == before
    after = sorted(p.name for p in tmp_path.iterdir())
    if kind == "jsonl":
        assert after == siblings
    else:
        created = set(after) - set(siblings)
        assert created <= {f"{path.name}-wal", f"{path.name}-shm"}
        wal = Path(f"{path}-wal")
        if f"{path.name}-wal" in created:
            assert wal.stat().st_size == 0
        assert not Path(f"{path}-journal").exists()


def test_uncheckpointed_wal_is_read_and_left_untouched(tmp_path: Path) -> None:
    """A copy of a live store, taken with frames still in its WAL, reads read-only.

    A read-write open would checkpoint the WAL into the main file when it closed.
    """
    live = write(tmp_path / "live.sqlite", "sqlite", HISTORY[:2])
    writer = sqlite3.connect(live)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO item_records (logical_schema_version, recorded_at, item_id, "
            "submission_index, prompt_fingerprint, context_fingerprint, input_fingerprint, "
            "identity_fingerprint, strategy_class, success, error_category, replay_eligible, "
            "calculated_cost, raw_prompt_json, raw_context_json, result_json) "
            "SELECT logical_schema_version, recorded_at, 'from_wal', submission_index, "
            "prompt_fingerprint, context_fingerprint, input_fingerprint, identity_fingerprint, "
            "strategy_class, success, error_category, replay_eligible, calculated_cost, "
            "raw_prompt_json, raw_context_json, result_json FROM item_records "
            "WHERE item_id = 'b'"
        )
        writer.commit()
        copy_dir = tmp_path / "copy"
        copy_dir.mkdir()
        shutil.copy(live, copy_dir / "store.sqlite")
        shutil.copy(f"{live}-wal", copy_dir / "store.sqlite-wal")
    finally:
        writer.close()
    path = copy_dir / "store.sqlite"
    assert Path(f"{path}-wal").stat().st_size > 0
    before = _fingerprint(path)
    assert [r.item_id for r in records(path)] == ["a", "b", "from_wal"]
    assert _fingerprint(path) == before
    assert Path(f"{path}-wal").stat().st_size > 0


@pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)
def test_closed_wal_store_in_read_only_directory_is_a_clear_error(tmp_path: Path) -> None:
    directory = tmp_path / "archive"
    directory.mkdir()
    path = write(directory / "store.sqlite", "sqlite", HISTORY[:2])
    for sidecar in ("-wal", "-shm"):
        Path(f"{path}{sidecar}").unlink(missing_ok=True)
    directory.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(ArtifactIOError, match="directory is not writable"):
            open_snapshot(path)
    finally:
        directory.chmod(stat.S_IRWXU)


# --- Legacy fixtures ---------------------------------------------------------------

LEGACY = [
    (release.name, name)
    for release in sorted(FIXTURES.iterdir())
    if release.is_dir()
    for name in ("artifacts.jsonl", "artifacts.sqlite")
    if (release / name).exists()
]


@pytest.mark.parametrize(("release", "name"), LEGACY)
def test_legacy_fixtures_read(tmp_path: Path, release: str, name: str) -> None:
    path = tmp_path / name
    shutil.copy(FIXTURES / release / name, path)
    with open_snapshot(path) as snapshot:
        summary = summarize(snapshot)
        assert snapshot.manifest.identity["provider"] == "fixture"
    succeeded, failed, stopped = summary.inventory()
    assert {"ok1", "ok2", "bad"} <= set(summary.newest)
    assert succeeded >= 2 and sum(failed.values()) >= 1
    assert not stopped


@pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)
@pytest.mark.parametrize("kind", KINDS)
def test_read_only_store_file_opens(tmp_path: Path, kind: str) -> None:
    """The open asks for read access only, so a read-only file is readable."""
    path = write(_store_path(tmp_path, kind), kind, HISTORY[:2])
    path.chmod(stat.S_IRUSR)
    try:
        assert [r.item_id for r in records(path)] == ["a", "b"]
    finally:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


# --- Round-1 review regressions (R1-R3) -------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("version", "message"),
    [
        (2, "Unsupported future artifact schema version 2"),
        ("x", "Unsupported artifact schema"),
        (0, "Unsupported artifact schema version 0"),
    ],
)
def test_newest_record_with_unsupported_schema_is_rejected(
    tmp_path: Path, kind: str, version: Any, message: str
) -> None:
    """R1: each record's own version is checked, not only the manifest's."""
    path = write(_store_path(tmp_path, kind), kind, HISTORY[:3])
    if kind == "jsonl":
        lines = path.read_bytes().splitlines(keepends=True)
        record = json.loads(lines[-1])
        record["artifact_schema_version"] = version
        lines[-1] = json.dumps(record).encode() + b"\n"
        path.write_bytes(b"".join(lines))
    else:
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE item_records SET logical_schema_version = ? WHERE record_sequence = "
                "(SELECT MAX(record_sequence) FROM item_records)",
                (version,),
            )
        connection.close()
    with open_snapshot(path) as snapshot, pytest.raises(ArtifactFormatError, match=message):
        list(snapshot.records())


def test_jsonl_huge_integer_cost_is_malformed(tmp_path: Path) -> None:
    """R2: a JSON integer too large for a float is a format error, not OverflowError."""
    path = write(_store_path(tmp_path, "jsonl"), "jsonl", HISTORY[:2])
    lines = path.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[-1])
    record["calculated_cost"] = 10**400
    lines[-1] = json.dumps(record).encode() + b"\n"
    path.write_bytes(b"".join(lines))
    with (
        open_snapshot(path) as snapshot,
        pytest.raises(ArtifactFormatError, match="calculated_cost"),
    ):
        list(snapshot.records())


@pytest.mark.parametrize("column", ["logical_schema_version", "sqlite_schema_version"])
def test_sqlite_non_integer_manifest_version_is_malformed(tmp_path: Path, column: str) -> None:
    """R2: the shared validator's int() conversion failure maps to a format error."""
    path = write(_store_path(tmp_path, "sqlite"), "sqlite", HISTORY[:1])
    with sqlite3.connect(path) as connection:
        connection.execute(f"UPDATE manifest SET {column} = 'bad'")
    connection.close()
    with pytest.raises(ArtifactFormatError, match="Malformed SQLite artifact"):
        open_snapshot(path)


@pytest.mark.parametrize("kind", KINDS)
def test_recorded_cost_sum_overflow_is_an_error(tmp_path: Path, kind: str) -> None:
    """R3: individually finite costs whose sum overflows are refused, not reported as inf."""
    path = write(
        _store_path(tmp_path, kind), kind, HISTORY[:3], cost_calculator=lambda result: 1e308
    )
    with open_snapshot(path) as snapshot:
        assert all(r.calculated_cost == 1e308 for r in snapshot.records())
        with pytest.raises(ArtifactFormatError, match="floating-point range"):
            summarize(snapshot)


# --- Round-2 review regressions (R2: decoder representation limits) ----------------

PLACEHOLDER = '"__PAYLOAD__"'
DECODER_LIMITS = {
    # Longer than sys.get_int_max_str_digits() (4300 by default): ValueError.
    "digits": "9" * 5000,
    # Deep enough to raise RecursionError on every supported interpreter.
    "nesting": "[" * 100_000 + "]" * 100_000,
}


def _with_payload(value: dict[str, Any], path: tuple[str, ...], payload: str) -> str:
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "__PAYLOAD__"
    return json.dumps(value).replace(PLACEHOLDER, payload)


@pytest.mark.parametrize("limit", sorted(DECODER_LIMITS))
@pytest.mark.parametrize(
    "site",
    ["jsonl_manifest", "jsonl_record", "sqlite_manifest", "sqlite_result", "sqlite_prompt"],
)
def test_decoder_limits_are_format_errors(tmp_path: Path, limit: str, site: str) -> None:
    """R2: the decoder's digit and nesting limits stay on and map to ArtifactFormatError."""
    payload = DECODER_LIMITS[limit]
    kind = site.split("_")[0]
    path = write(_store_path(tmp_path, kind), kind, HISTORY[:2], include_prompt=True)
    if kind == "jsonl":
        lines = path.read_bytes().splitlines(keepends=True)
        index, key_path = (
            (0, ("user_metadata", "x"))
            if site == "jsonl_manifest"
            else (
                2,
                ("result", "token_usage", "input_tokens"),
            )
        )
        lines[index] = _with_payload(json.loads(lines[index]), key_path, payload).encode() + b"\n"
        path.write_bytes(b"".join(lines))
    else:
        with sqlite3.connect(path) as connection:
            if site == "sqlite_manifest":
                connection.execute(
                    "UPDATE manifest SET user_metadata_json = ?",
                    (_with_payload({}, ("x",), payload),),
                )
            elif site == "sqlite_prompt":
                connection.execute(
                    "UPDATE item_records SET raw_prompt_json = ? WHERE item_id = 'b'", (payload,)
                )
            else:
                (text,) = connection.execute(
                    "SELECT result_json FROM item_records WHERE item_id = 'b'"
                ).fetchone()
                connection.execute(
                    "UPDATE item_records SET result_json = ? WHERE item_id = 'b'",
                    (_with_payload(json.loads(text), ("output",), payload),),
                )
        connection.close()
    with pytest.raises(ArtifactFormatError):
        with open_snapshot(path) as snapshot:
            list(snapshot.records())
