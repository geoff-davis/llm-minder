"""Release-document integrity checks and v0.20 onboarding regressions."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "notebooks" / "llm_minder_quickstart.ipynb"
TERMINAL_ASSET = ROOT / "docs" / "assets" / "v0.20-quickstart.gif"


def test_notebook_is_valid_safe_json_with_parseable_python() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "sk-" not in source
    assert "AIza" not in source
    assert not re.search(r"(?i)(?:api_key|token|secret)\s*=\s*['\"][^.'\"]{8,}", source)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        python_source = "\n".join(
            line
            for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        compile(
            python_source,
            str(NOTEBOOK),
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )


def test_terminal_asset_is_portable_and_small() -> None:
    assert TERMINAL_ASSET.read_bytes().startswith((b"GIF87a", b"GIF89a"))
    assert TERMINAL_ASSET.stat().st_size < 2 * 1024 * 1024

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "https://raw.githubusercontent.com/geoff-davis/llm-minder/main/"
        "docs/assets/v0.20-quickstart.gif"
    ) in readme
    assert (
        "https://colab.research.google.com/github/geoff-davis/llm-minder/blob/main/"
        "notebooks/llm_minder_quickstart.ipynb"
    ) in readme
    assert not re.search(r"!\[[^]]*]\((?!https://)", readme)


def test_primary_onboarding_uses_high_level_api() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    getting_started = (ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8")
    for content in (readme, docs_home, getting_started):
        assert 'llm("openai:gpt-6-luna")' in content
        assert "process_prompts" in content
        assert "concurrency=" in content
        assert "summary()" in content
    assert "progress=True" in readme
    assert "ParallelBatchProcessor(" not in docs_home
    assert "PydanticAIStrategy(" not in docs_home


def test_comparison_and_troubleshooting_are_linked_and_navigable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for path in ("comparison.md", "troubleshooting.md"):
        assert (ROOT / "docs" / path).is_file()
        assert path in navigation
        assert path in docs_home
        assert path.removesuffix(".md") in readme


def test_release_history_and_migration_are_coherent() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    migration = (ROOT / "docs" / "migration" / "v0.20.md").read_text(encoding="utf-8")
    migration_v022 = (ROOT / "docs" / "migration" / "v0.22.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "migration" / "index.md").read_text(encoding="utf-8")

    assert "v0.19.0 was not published" in changelog
    assert "## [0.19" not in changelog
    assert "## [0.21.0] - 2026-07-26" in changelog
    assert "## [0.22.0] - 2026-08-17" in changelog
    assert "## [0.20.0] - 2026-07-20" in changelog
    assert "v0.18.x to v0.20.0" in migration
    assert "v0.19.0 was not published" in migration
    assert "v0.19 was never published" in index
    # The unpublished v0.19's old page now redirects to the v0.20 guide (#321).
    assert "MIGRATION_V0_19.md: migration/v0.20.md" in navigation
    assert "v0.20 Migration: migration/v0.20.md" in navigation
    assert "v0.22 Migration: migration/v0.22.md" in navigation
    assert "v0.21.x to v0.22" in migration_v022
    assert "No checkpoint migration or invalidation is required" in migration_v022


def test_release_version_and_tag_workflow_agree() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert re.search(r'^version = "1\.0\.1"$', pyproject, re.MULTILINE)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    # Unreleased notes may accumulate between releases; validate heading order
    # without requiring that section to stay empty.
    headings = re.findall(r"^## \[.*$", changelog, re.MULTILINE)
    assert headings[:2] == ["## [Unreleased]", "## [1.0.1] - 2026-11-13"]
    assert '"v${PKG_VERSION}" != "${GITHUB_REF_NAME}"' in workflow


def test_status_wording_matches_the_stable_classifier() -> None:
    # #343: the landing pages kept calling the project beta after 1.0 shipped.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if "Development Status :: 5 - Production/Stable" not in pyproject:
        return
    pages = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/index.md": (ROOT / "docs" / "index.md").read_text(encoding="utf-8"),
        "mkdocs.yml": (ROOT / "mkdocs.yml").read_text(encoding="utf-8"),
    }
    stale = re.compile(r"\bbeta\b|pre-1\.0|ahead of 1\.0|API Stability \(draft", re.IGNORECASE)
    found = {name: stale.findall(text) for name, text in pages.items() if stale.search(text)}
    assert not found


def test_source_does_not_claim_features_shipped_in_v019() -> None:
    stale = []
    for path in (ROOT / "src").rglob("*.py"):
        if "v0.19" in path.read_text(encoding="utf-8"):
            stale.append(path.relative_to(ROOT))
    assert not stale


def _redirect_maps() -> dict[str, str]:
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    block = navigation.split("redirect_maps:\n", 1)[1]
    maps: dict[str, str] = {}
    # The map is the run of eight-space-indented lines (mkdocs.yml has no PyYAML
    # in the dev group, so parse the block by hand).
    for line in block.splitlines():
        if not line.startswith("        "):
            break
        entry = line.strip()
        if entry and not entry.startswith("#"):
            old, new = (part.strip() for part in entry.split(":", 1))
            maps[old] = new
    return maps


def test_moved_pages_redirect_to_existing_pages() -> None:
    maps = _redirect_maps()
    for old in (
        "MIGRATION_V0_10.md",
        "MIGRATION_V0_19.md",
        "MIGRATION_V0_20.md",
        "MIGRATION_V0_22.md",
        "release-notes-v0.20.md",
        "sqlite-maintenance-evidence.md",
        "v0.22-scale-evidence.md",
    ):
        assert old in maps, old
    for old, new in maps.items():
        assert not (ROOT / "docs" / old).exists(), f"{old} redirects but still exists"
        assert (ROOT / "docs" / new).is_file(), f"{old} redirects to missing {new}"


def test_every_migration_guide_is_in_the_index_and_nav() -> None:
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "migration" / "index.md").read_text(encoding="utf-8")
    guides = sorted(p.name for p in (ROOT / "docs" / "migration").glob("*.md"))
    assert "index.md" in guides
    for name in guides:
        assert f"migration/{name}" in navigation, name
        if name != "index.md":
            assert f"]({name})" in index, name
    # No migration guide is left at the top of docs/.
    assert not list((ROOT / "docs").glob("MIGRATION*.md"))
