"""Tests for package version."""

import re
from importlib.metadata import version


def test_version_matches_package():
    """Verify __version__ matches package metadata."""
    from llm_minder import __version__

    # Should match pyproject.toml version
    package_version = version("llm-minder")
    assert __version__ == package_version, (
        f"__version__ ({__version__}) doesn't match package metadata ({package_version})"
    )


def test_version_format():
    """Verify the version is X.Y.Z, optionally a PEP 440 pre-release (1.0.0rc1) or +dev."""
    from llm_minder import __version__

    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\+dev)?", __version__), (
        f"Version should be X.Y.Z[aN|bN|rcN][+dev], got: {__version__}"
    )
