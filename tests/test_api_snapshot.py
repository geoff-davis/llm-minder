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


def _render(cls: type) -> str:
    return "\n".join(api_snapshot._render_class(cls.__name__, cls))


def test_snapshot_sees_typeddict_keys_and_requiredness():
    from typing import NotRequired, TypedDict

    class Before(TypedDict):
        a: int
        b: NotRequired[str]

    class Dropped(TypedDict):
        a: int

    class NowRequired(TypedDict):
        a: int
        b: str

    rendered = {_render(cls).split("\n", 1)[1] for cls in (Before, Dropped, NowRequired)}
    assert len(rendered) == 3


def test_snapshot_sees_changed_default_factories():
    import dataclasses

    from llm_minder import RetryConfig

    @dataclasses.dataclass
    class Config:
        retry: RetryConfig = dataclasses.field(default_factory=RetryConfig)

    @dataclasses.dataclass
    class Changed:
        retry: RetryConfig = dataclasses.field(default_factory=lambda: RetryConfig(max_attempts=99))

    before = _render(Config).split("\n", 1)[1]
    after = _render(Changed).split("\n", 1)[1]
    assert before != after
    assert "max_attempts=99" in after
