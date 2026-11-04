"""docs/stability.md classifies every public export exactly once."""

from __future__ import annotations

import re
from pathlib import Path

import llm_minder

PAGE = Path(__file__).parents[1] / "docs" / "stability.md"


def _rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("| `")]


def _first_cell_names(rows: list[str]) -> list[str]:
    names: list[str] = []
    for row in rows:
        names.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", row.split("|")[1]))
    return names


def test_every_export_is_classified_once():
    # The removed-in-1.0 table lists names that are no longer exported.
    classified, _, _ = PAGE.read_text().partition("### Removed in 1.0")
    rows = _rows(classified)
    names = _first_cell_names(rows)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == []
    assert sorted(set(names)) == sorted(llm_minder.__all__)


def test_removed_names_are_not_exported():
    _, _, removed = PAGE.read_text().partition("### Removed in 1.0")
    names = set(_first_cell_names(_rows(removed.split("\n## ")[0])))
    assert {"LLMGateway", "BatchProcessor", "ProcessingStats"} <= names
    assert names.isdisjoint(llm_minder.__all__)
