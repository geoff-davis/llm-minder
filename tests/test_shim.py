"""The async-batch-llm compatibility package in shim/ (published from shim-v* tags).

Each check runs in a fresh interpreter with shim/src on the path, so the import
hooks the shim installs never leak into the rest of the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "shim"


def _run(tmp_path: Path, code: str, *python_flags: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(code))
    env = {**os.environ, "PYTHONPATH": str(SHIM / "src")}
    return subprocess.run(
        [sys.executable, *python_flags, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
    )


def test_old_imports_are_the_new_modules(tmp_path):
    result = _run(
        tmp_path,
        """
        import warnings
        warnings.simplefilter("ignore", DeprecationWarning)

        import async_batch_llm
        import async_batch_llm.testing
        import async_batch_llm.strategies.errors as old_errors
        from async_batch_llm import LLMMinderError, ParallelBatchProcessor
        from async_batch_llm.models import OpenAIModel

        import llm_minder
        import llm_minder.models
        import llm_minder.strategies.errors
        import llm_minder.testing

        assert async_batch_llm is llm_minder
        assert async_batch_llm.testing is llm_minder.testing
        assert old_errors is llm_minder.strategies.errors
        assert ParallelBatchProcessor is llm_minder.ParallelBatchProcessor
        assert OpenAIModel is llm_minder.models.OpenAIModel
        assert LLMMinderError is llm_minder.AsyncBatchLLMError
        try:
            import async_batch_llm.no_such_module
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("missing submodule imported")
        print("ok")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_import_warns_once_at_the_importing_line(tmp_path):
    result = _run(
        tmp_path,
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import async_batch_llm
            import async_batch_llm.testing
            from async_batch_llm import llm

        deprecations = [w for w in caught if "renamed to llm-minder" in str(w.message)]
        assert len(deprecations) == 1, [str(w.message) for w in deprecations]
        warning = deprecations[0]
        assert warning.category is DeprecationWarning
        assert warning.filename.endswith("app.py"), warning.filename
        assert warning.lineno == 6, warning.lineno
        print("ok")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_default_filters_show_the_warning_in_main(tmp_path):
    result = _run(tmp_path, "import async_batch_llm\n")
    assert result.returncode == 0, result.stderr
    assert "DeprecationWarning: async-batch-llm has been renamed" in result.stderr


@pytest.fixture(scope="module")
def projects() -> tuple[dict, dict]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        main = tomllib.load(f)["project"]
    with (SHIM / "pyproject.toml").open("rb") as f:
        shim = tomllib.load(f)["project"]
    return main, shim


def test_shim_forwards_every_user_extra(projects):
    main, shim = projects
    user_extras = set(main["optional-dependencies"]) - {"dev", "docs"}
    assert set(shim["optional-dependencies"]) == user_extras
    for extra, requirements in shim["optional-dependencies"].items():
        assert requirements == [
            shim["dependencies"][0].replace("llm-minder", f"llm-minder[{extra}]", 1)
        ]


def test_shim_requires_llm_minder_and_matching_python(projects):
    main, shim = projects
    assert shim["name"] == "async-batch-llm"
    assert shim["requires-python"] == main["requires-python"]
    assert len(shim["dependencies"]) == 1
    assert shim["dependencies"][0].startswith("llm-minder")
