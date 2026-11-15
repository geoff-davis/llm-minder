"""PEP 639 license metadata for llm-minder and the async-batch-llm shim (#337)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = {"llm-minder": ROOT, "async-batch-llm": ROOT / "shim"}


def _pyproject(directory: Path) -> dict:
    with (directory / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(params=sorted(PROJECTS), ids=sorted(PROJECTS))
def project_dir(request: pytest.FixtureRequest) -> Path:
    return PROJECTS[request.param]


def test_license_is_an_spdx_expression_with_its_file(project_dir: Path) -> None:
    project = _pyproject(project_dir)["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert (project_dir / "LICENSE").is_file()


def test_no_license_classifier(project_dir: Path) -> None:
    # PEP 639 forbids a License :: classifier alongside a license expression.
    classifiers = _pyproject(project_dir)["project"]["classifiers"]
    assert not [c for c in classifiers if c.startswith("License ::")]


def test_hatchling_floor_writes_license_expression(project_dir: Path) -> None:
    # hatchling < 1.27 silently writes the legacy License field instead.
    requires = _pyproject(project_dir)["build-system"]["requires"]
    (hatchling,) = [r for r in requires if r.startswith("hatchling")]
    match = re.fullmatch(r"hatchling>=(\d+)\.(\d+)(?:\.\d+)?", hatchling)
    assert match, hatchling
    assert (int(match[1]), int(match[2])) >= (1, 27)


def test_sdist_ships_the_license(project_dir: Path) -> None:
    include = _pyproject(project_dir)["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "/LICENSE" in include


def test_shim_license_matches_the_main_license() -> None:
    assert (ROOT / "shim" / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()


# PEP 735 dependency groups (#337, part 2).

DEV_COMMAND_FILES = [
    *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
    ROOT / "Makefile",
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CLAUDE.md",
    ROOT / "docs" / "contributing.md",
]


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower()


def test_dev_and_docs_are_dependency_groups_not_extras() -> None:
    pyproject = _pyproject(ROOT)
    extras = pyproject["project"]["optional-dependencies"]
    assert not {"dev", "docs"} & set(extras)
    assert {"dev", "docs"} <= set(pyproject["dependency-groups"])


def test_dev_group_installs_every_provider_extra() -> None:
    # A group can't include the project's own extras, so dev repeats them; keep
    # the copy complete so the test suite has every SDK it imports.
    pyproject = _pyproject(ROOT)
    all_extra = {_requirement_name(r) for r in pyproject["project"]["optional-dependencies"]["all"]}
    dev_group = {_requirement_name(r) for r in pyproject["dependency-groups"]["dev"]}
    assert all_extra <= dev_group, sorted(all_extra - dev_group)


@pytest.mark.parametrize("path", DEV_COMMAND_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_command_installs_the_removed_extras(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"--extra[ =](dev|docs)\b|llm-minder\[(dev|docs)\]", text)
