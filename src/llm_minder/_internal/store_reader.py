"""Synchronous read-only snapshots of JSONL and SQLite artifact stores (#318).

Private reader for the store-inspection CLI. It is deliberately separate from
the stores' own read paths:

- **Read-only.** It never creates, repairs, migrates or truncates a store. A
  JSONL store is read as bytes. A SQLite store is opened only with SQLite's
  native ``mode=ro`` plus ``PRAGMA query_only``, never ``immutable=1``, so it
  takes part in WAL locking and change detection while a writer is live. That
  open may create the standard ``-wal``/``-shm`` coordination files beside a
  SQLite-format file (valid or rejected); a file without the SQLite header is
  never opened through SQLite.
- **One snapshot per open.** A SQLite snapshot holds one read transaction; a
  JSONL snapshot captures the file size once and every pass stops at that byte
  boundary. :meth:`StoreSnapshot.records` can be iterated more than once and
  each pass sees the same records.
- **Streaming.** Records are yielded one at a time: one JSONL line, or one page
  of :data:`SQLITE_PAGE_SIZE` rows, is held while it is parsed.
- **Undecoded.** Stored results are returned as the stored JSON objects. No
  output, context or custom decoder ever runs.

Errors are the stores' own: :class:`~llm_minder.ArtifactIOError` for a path that
cannot be read and :class:`~llm_minder.ArtifactFormatError` for a file that is
not a supported, well-formed llm-minder store.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from collections import Counter
from collections.abc import Generator, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal

from ..artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactFormatError,
    ArtifactIOError,
    _validate_jsonl_record,
)
from ..sqlite_artifacts import SqliteArtifactStore
from .artifact_codec import GUARDRAIL_AUDIT_CATEGORIES

StoreKind = Literal["jsonl", "sqlite"]

SQLITE_MAGIC = b"SQLite format 3\x00"
SQLITE_PAGE_SIZE = 100

_GUARDRAIL_CATEGORIES = frozenset(GUARDRAIL_AUDIT_CATEGORIES)


@dataclass(frozen=True)
class StoreManifest:
    """The creation manifest of a store, as recorded."""

    kind: StoreKind
    artifact_schema_version: int
    sqlite_schema_version: int | None
    created_at: str | None
    package_version: str | None
    identity: Mapping[str, Any]
    identity_fingerprint: str | None
    user_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class StoredRecord:
    """One terminal item record, undecoded.

    ``sequence`` orders records within one store: the SQLite ``record_sequence``,
    or the record's 0-based position after the manifest in a JSONL file. Both
    follow append order. ``result`` is the stored work-item result object;
    ``prompt`` is the stored prompt string or message list, or ``None`` when the
    store was written without ``include_prompt``.
    """

    sequence: int
    item_id: str
    recorded_at: str | None
    success: bool
    error_category: str | None
    identity_fingerprint: str | None
    provider: str | None
    model: str | None
    prompt_version: str | None
    calculated_cost: float | None
    token_usage: Mapping[str, int]
    result: Mapping[str, Any]
    prompt: Any = None

    @property
    def error(self) -> str | None:
        """The stored error text (not redacted; it may contain prompt fragments)."""
        error = self.result.get("error")
        return error if isinstance(error, str) else None

    @property
    def is_guardrail_stop(self) -> bool:
        return self.error_category in _GUARDRAIL_CATEGORIES


def _format_error(where: str, message: str) -> ArtifactFormatError:
    return ArtifactFormatError(f"Malformed artifact record at {where}: {message}")


def _loads(data: str | bytes, context: str) -> Any:
    """Parse stored JSON, mapping every decoder failure to ``ArtifactFormatError``.

    Besides syntax errors this covers the decoder's representation limits, which
    stay enabled: an integer literal longer than ``sys.get_int_max_str_digits()``
    raises ``ValueError`` and deep nesting raises ``RecursionError``. Every JSON
    value the reader parses (JSONL lines, SQLite manifest, prompt and result
    columns) goes through here.
    """
    try:
        return json.loads(data)
    except RecursionError as exc:
        raise ArtifactFormatError(f"{context}: JSON nesting is too deep to decode") from exc
    except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError, integer-digit limit
        raise ArtifactFormatError(f"{context}: {exc}") from exc


def _finite_cost(value: Any, where: str) -> float | None:
    """Return a stored cost as a finite float; a huge JSON integer is malformed too."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _format_error(where, "calculated_cost must be a finite number or null")
    try:
        cost = float(value)
    except OverflowError:
        cost = math.inf
    if not math.isfinite(cost):
        raise _format_error(where, "calculated_cost must be a finite number or null")
    return cost


def _check_schema_version(version: Any, where: str) -> None:
    """Apply the stores' per-record logical schema version policy."""
    if isinstance(version, bool) or not isinstance(version, int):
        raise ArtifactFormatError(f"Unsupported artifact schema version {version!r} at {where}")
    if version > ARTIFACT_SCHEMA_VERSION:
        raise ArtifactFormatError(
            f"Unsupported future artifact schema version {version} at {where}"
        )
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactFormatError(f"Unsupported artifact schema version {version!r} at {where}")


def _validated_record(
    *,
    sequence: int,
    where: str,
    item_id: Any,
    recorded_at: Any,
    success: Any,
    error_category: Any,
    identity_fingerprint: Any,
    provider: Any,
    model: Any,
    prompt_version: Any,
    calculated_cost: Any,
    result: Any,
    prompt: Any,
) -> StoredRecord:
    """Check the fields the inspection commands rely on and build a record."""
    if not isinstance(item_id, str):
        raise _format_error(where, "item_id must be a string")
    if not isinstance(success, bool):
        raise _format_error(where, "success must be a boolean")
    if not isinstance(result, dict):
        raise _format_error(where, "result must be a JSON object")
    raw_usage = result.get("token_usage", {})
    if raw_usage is None:
        raw_usage = {}
    if not isinstance(raw_usage, dict):
        raise _format_error(where, "result.token_usage must be an object")
    for key, value in raw_usage.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise _format_error(where, f"result.token_usage[{key!r}] must be an integer")
    error = result.get("error")
    if error is not None and not isinstance(error, str):
        raise _format_error(where, "result.error must be a string or null")
    cost = _finite_cost(calculated_cost, where)
    fields = {
        "recorded_at": recorded_at,
        "error_category": error_category,
        "identity_fingerprint": identity_fingerprint,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
    }
    for key, value in fields.items():
        if value is not None and not isinstance(value, str):
            raise _format_error(where, f"{key} must be a string or null")
    return StoredRecord(
        sequence=sequence,
        item_id=item_id,
        recorded_at=recorded_at,
        success=success,
        error_category=error_category,
        identity_fingerprint=identity_fingerprint,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        calculated_cost=cost,
        token_usage=raw_usage,
        result=result,
        prompt=prompt,
    )


class StoreSnapshot:
    """One read-only snapshot of a store. Use :func:`open_snapshot`."""

    manifest: StoreManifest

    def records(self) -> Iterator[StoredRecord]:
        """Yield every item record in the snapshot, in sequence order."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the snapshot; repeated calls are safe."""

    def __enter__(self) -> StoreSnapshot:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


# --- JSONL ---------------------------------------------------------------------


class _JsonlSnapshot(StoreSnapshot):
    def __init__(self, path: Path, size: int) -> None:
        self.path = path
        self.size = size
        self.manifest = self._read_manifest()

    def _open(self) -> BinaryIO:
        try:
            return self.path.open("rb")
        except OSError as exc:
            raise ArtifactIOError(f"Could not read artifact {self.path}: {exc}") from exc

    def _lines(self) -> Iterator[tuple[int, bytes]]:
        """Yield (line number, complete line) up to the captured byte boundary.

        A final line without a newline at the boundary is a truncated tail (a
        torn write, or an append in progress when the snapshot was taken) and is
        not yielded, matching the store's own reader.
        """
        offset = 0
        line_number = 0
        with self._open() as handle:
            try:
                while offset < self.size:
                    line = handle.readline(self.size - offset)
                    if not line:
                        return
                    offset += len(line)
                    line_number += 1
                    if not line.endswith(b"\n"):
                        return
                    if line.strip():
                        yield line_number, line
            except OSError as exc:
                raise ArtifactIOError(f"Could not read artifact {self.path}: {exc}") from exc

    def _parsed(self) -> Generator[tuple[int, Any], None, None]:
        for line_number, line in self._lines():
            yield (
                line_number,
                _loads(line, f"Malformed artifact JSON at non-final line {line_number}"),
            )

    def _read_manifest(self) -> StoreManifest:
        with closing(self._parsed()) as lines:
            return self._manifest_from(lines)

    def _manifest_from(self, lines: Iterator[tuple[int, Any]]) -> StoreManifest:
        for line_number, value in lines:
            _validate_jsonl_record(value, line_number=line_number, manifest_seen=False)
            identity = value.get("identity")
            user_metadata = value.get("user_metadata")
            created_at = value.get("created_at")
            package_version = value.get("package_version")
            fingerprint = value.get("identity_fingerprint")
            if not isinstance(identity, dict) or not isinstance(user_metadata, dict):
                raise ArtifactFormatError("Malformed artifact manifest at line 1")
            for item in (created_at, package_version, fingerprint):
                if item is not None and not isinstance(item, str):
                    raise ArtifactFormatError("Malformed artifact manifest at line 1")
            return StoreManifest(
                kind="jsonl",
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                sqlite_schema_version=None,
                created_at=created_at,
                package_version=package_version,
                identity=identity,
                identity_fingerprint=fingerprint,
                user_metadata=user_metadata,
            )
        raise ArtifactFormatError("Artifact has no complete manifest record")

    def records(self) -> Iterator[StoredRecord]:
        manifest_seen = False
        sequence = 0
        for line_number, value in self._parsed():
            record = _validate_jsonl_record(
                value, line_number=line_number, manifest_seen=manifest_seen
            )
            if record is None:
                manifest_seen = True
                continue
            prompt = record.get("raw_prompt")
            if prompt is None:
                prompt = record.get("raw_messages")
            yield _validated_record(
                sequence=sequence,
                where=f"line {line_number}",
                item_id=record.get("item_id"),
                recorded_at=record.get("recorded_at"),
                success=record.get("success"),
                error_category=record.get("error_category"),
                identity_fingerprint=record.get("identity_fingerprint"),
                provider=record.get("provider"),
                model=record.get("model"),
                prompt_version=record.get("prompt_version"),
                calculated_cost=record.get("calculated_cost"),
                result=record.get("result"),
                prompt=prompt,
            )
            sequence += 1


# --- SQLite --------------------------------------------------------------------

_SQLITE_RECORDS = f"""
    SELECT r.record_sequence, r.logical_schema_version, r.item_id, r.recorded_at, r.success, r.error_category,
           r.identity_fingerprint, i.provider, i.model, i.prompt_version,
           r.calculated_cost, r.raw_prompt_json, r.result_json
      FROM item_records AS r
      LEFT JOIN identities AS i ON i.identity_fingerprint = r.identity_fingerprint
     WHERE r.record_sequence > ?
     ORDER BY r.record_sequence
     LIMIT {SQLITE_PAGE_SIZE}
"""


def _read_only_directory_message(path: Path, exc: Exception) -> str:
    return (
        f"Could not open SQLite artifact {path} read-only: {exc}. SQLite needs to create "
        "its WAL coordination files (-wal, -shm) beside a store, and the directory is "
        "not writable. Copy the store (with any -wal file) to a writable directory and "
        "inspect the copy."
    )


class _SqliteSnapshot(StoreSnapshot):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None
        try:
            self._connection = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None
            )
            self._connection.execute("PRAGMA query_only=ON")
            # One read transaction holds one snapshot for every pass. Under WAL a
            # concurrent writer's later commits are invisible to it.
            self._connection.execute("BEGIN")
            SqliteArtifactStore._validate_schema_connection_sync(self._connection, path)
            self.manifest = self._read_manifest()
        except Exception as exc:
            self.close()
            mapped = self._map_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        except BaseException:
            self.close()
            raise

    def _map_error(self, exc: Exception) -> Exception:
        if isinstance(exc, (ArtifactFormatError, ArtifactIOError)):
            return exc
        if isinstance(exc, sqlite3.OperationalError):
            if not os.access(self.path.parent, os.W_OK):
                return ArtifactIOError(_read_only_directory_message(self.path, exc))
            return ArtifactIOError(f"Could not read SQLite artifact {self.path}: {exc}")
        if isinstance(exc, sqlite3.DatabaseError):
            return ArtifactFormatError(f"Invalid SQLite artifact {self.path}: {exc}")
        if isinstance(exc, OSError):
            return ArtifactIOError(f"Could not read SQLite artifact {self.path}: {exc}")
        if isinstance(exc, (ValueError, TypeError, OverflowError)):
            # The schema checks convert stored header and manifest values with int().
            return ArtifactFormatError(f"Malformed SQLite artifact metadata in {self.path}: {exc}")
        return exc

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ArtifactIOError(f"SQLite artifact snapshot is closed: {self.path}")
        return self._connection

    def _json(self, text: Any, *, column: str, where: str) -> Any:
        if not isinstance(text, str):
            raise _format_error(where, f"{column} must be JSON text")
        return _loads(text, f"Malformed artifact record at {where}: {column} is not valid JSON")

    def _read_manifest(self) -> StoreManifest:
        row = (
            self._require_connection()
            .execute(
                """
                SELECT logical_schema_version, sqlite_schema_version, created_at,
                       package_version, initial_identity_json,
                       initial_identity_fingerprint, user_metadata_json
                  FROM manifest WHERE id = 1
                """
            )
            .fetchone()
        )
        where = "the SQLite manifest"
        identity = self._json(row[4], column="initial_identity_json", where=where)
        user_metadata = self._json(row[6], column="user_metadata_json", where=where)
        if not isinstance(identity, dict) or not isinstance(user_metadata, dict):
            raise _format_error(where, "identity and user metadata must be JSON objects")
        for value in (row[2], row[3], row[5]):
            if value is not None and not isinstance(value, str):
                raise _format_error(where, "text columns must hold text")
        return StoreManifest(
            kind="sqlite",
            artifact_schema_version=int(row[0]),
            sqlite_schema_version=int(row[1]),
            created_at=row[2],
            package_version=row[3],
            identity=identity,
            identity_fingerprint=row[5],
            user_metadata=user_metadata,
        )

    def _page(self, after: int) -> list[tuple[Any, ...]]:
        try:
            return self._require_connection().execute(_SQLITE_RECORDS, (after,)).fetchall()
        except sqlite3.Error as exc:
            raise self._map_error(exc) from exc  # always a new ArtifactError

    def records(self) -> Iterator[StoredRecord]:
        after = -1
        while True:
            page = self._page(after)
            if not page:
                return
            for row in page:
                (
                    sequence,
                    logical_schema_version,
                    item_id,
                    recorded_at,
                    success,
                    error_category,
                    identity_fingerprint,
                    provider,
                    model,
                    prompt_version,
                    calculated_cost,
                    raw_prompt_json,
                    result_json,
                ) = row
                after = int(sequence)
                where = f"SQLite sequence {after}"
                _check_schema_version(logical_schema_version, f"{where} for item {item_id!r}")
                if success not in (0, 1) or isinstance(success, float):
                    raise _format_error(where, "success must be 0 or 1")
                prompt = (
                    None
                    if raw_prompt_json is None
                    else self._json(raw_prompt_json, column="raw_prompt_json", where=where)
                )
                yield _validated_record(
                    sequence=after,
                    where=where,
                    item_id=item_id,
                    recorded_at=recorded_at,
                    success=bool(success),
                    error_category=error_category,
                    identity_fingerprint=identity_fingerprint,
                    provider=provider,
                    model=model,
                    prompt_version=prompt_version,
                    calculated_cost=calculated_cost,
                    result=self._json(result_json, column="result_json", where=where),
                    prompt=prompt,
                )
            # Drop the page before fetching the next one.
            del page

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.close()
        except sqlite3.Error as exc:
            raise ArtifactIOError(
                f"Could not close SQLite artifact reader {self.path}: {exc}"
            ) from exc


# --- Opening -------------------------------------------------------------------


def detect_kind(path: Path) -> StoreKind:
    """Classify a store file by its header, without opening it through SQLite."""
    try:
        if not path.exists():
            raise ArtifactIOError(f"Artifact does not exist: {path}")
        if path.is_dir():
            raise ArtifactIOError(f"Artifact path is a directory: {path}")
        with path.open("rb") as handle:
            header = handle.read(len(SQLITE_MAGIC))
    except ArtifactIOError:
        raise
    except OSError as exc:
        raise ArtifactIOError(f"Could not read artifact {path}: {exc}") from exc
    if not header:
        raise ArtifactFormatError(f"Artifact is empty: {path}")
    return "sqlite" if header == SQLITE_MAGIC else "jsonl"


def open_snapshot(path: str | Path) -> StoreSnapshot:
    """Open a read-only snapshot of the JSONL or SQLite store at ``path``."""
    store_path = Path(path)
    if detect_kind(store_path) == "sqlite":
        return _SqliteSnapshot(store_path)
    try:
        size = store_path.stat().st_size
    except OSError as exc:
        raise ArtifactIOError(f"Could not inspect artifact {store_path}: {exc}") from exc
    return _JsonlSnapshot(store_path, size)


# --- Grouping state --------------------------------------------------------------


@dataclass
class GuardrailStops:
    """Recorded guardrail stops for one category: a count and the first reason."""

    count: int = 0
    first_reason: str | None = None


@dataclass
class StoreSummary:
    """Grouping state built in one pass over a snapshot.

    Memory is one entry per distinct item id (its newest record's sequence,
    success flag and interned category) plus counters per identity, category
    and token key. No record text other than one first reason per guardrail
    category is kept.
    """

    record_count: int = 0
    newest: dict[str, tuple[int, bool, str | None]] = field(default_factory=dict)
    identities: Counter[tuple[str | None, str | None, str | None]] = field(default_factory=Counter)
    token_totals: Counter[str] = field(default_factory=Counter)
    recorded_cost: float = 0.0
    recorded_cost_records: int = 0
    guardrail_stops: dict[str, GuardrailStops] = field(default_factory=dict)

    def add(self, record: StoredRecord) -> None:
        self.record_count += 1
        category = None if record.error_category is None else sys.intern(record.error_category)
        previous = self.newest.get(record.item_id)
        if previous is None or record.sequence > previous[0]:
            self.newest[record.item_id] = (record.sequence, record.success, category)
        self.identities[(record.provider, record.model, record.prompt_version)] += 1
        self.token_totals.update(record.token_usage)
        if record.calculated_cost is not None:
            total = self.recorded_cost + record.calculated_cost
            if not math.isfinite(total):
                raise ArtifactFormatError(
                    "Recorded costs sum beyond the floating-point range at record "
                    f"sequence {record.sequence} (item {record.item_id!r})"
                )
            self.recorded_cost = total
            self.recorded_cost_records += 1
        if category is not None and category in _GUARDRAIL_CATEGORIES:
            stops = self.guardrail_stops.setdefault(category, GuardrailStops())
            stops.count += 1
            if stops.first_reason is None:
                stops.first_reason = record.error

    @property
    def item_count(self) -> int:
        return len(self.newest)

    def is_newest(self, record: StoredRecord) -> bool:
        newest = self.newest.get(record.item_id)
        return newest is not None and newest[0] == record.sequence

    def inventory(self) -> tuple[int, Counter[str | None], Counter[str | None]]:
        """Items by newest record: (succeeded, failed by category, stopped by category)."""
        succeeded = 0
        failed: Counter[str | None] = Counter()
        stopped: Counter[str | None] = Counter()
        for _sequence, success, category in self.newest.values():
            if category is not None and category in _GUARDRAIL_CATEGORIES:
                stopped[category] += 1
            elif success:
                succeeded += 1
            else:
                failed[category] += 1
        return succeeded, failed, stopped


def summarize(snapshot: StoreSnapshot) -> StoreSummary:
    """Validate every record in the snapshot and build its grouping state."""
    summary = StoreSummary()
    for record in snapshot.records():
        summary.add(record)
    return summary


__all__ = [
    "SQLITE_MAGIC",
    "SQLITE_PAGE_SIZE",
    "GuardrailStops",
    "StoreKind",
    "StoreManifest",
    "StoreSnapshot",
    "StoredRecord",
    "StoreSummary",
    "detect_kind",
    "open_snapshot",
    "summarize",
]
