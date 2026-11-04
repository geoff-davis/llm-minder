"""The public API matches the checked-in snapshot (tests/api_snapshot.txt).

A failure here means a stable name, signature, field, enum value or documented event
payload changed. If the change is intended, regenerate the snapshot and commit it with
the change:

    uv run python scripts/api_snapshot.py --update
"""

from __future__ import annotations

import difflib
import importlib.util
from pathlib import Path

from llm_minder import ProcessingEvent

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "api_snapshot.py"
_spec = importlib.util.spec_from_file_location("api_snapshot", SCRIPT)
assert _spec is not None and _spec.loader is not None
api_snapshot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api_snapshot)


def test_public_api_matches_snapshot():
    expected = api_snapshot.SNAPSHOT_PATH.read_text()
    actual = api_snapshot.render()
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                "tests/api_snapshot.txt",
                "current API",
            )
        )
        raise AssertionError(
            "The public API changed. If that's intended, run "
            "`uv run python scripts/api_snapshot.py --update` and commit the result.\n" + diff
        )


def test_every_event_has_documented_payload_keys():
    documented = {
        line.split(":")[0].removeprefix("event ") for line in api_snapshot._documented_payloads()
    }
    assert documented == {event.name for event in ProcessingEvent}
