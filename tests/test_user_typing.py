"""Run the offline parts of the strict-typing sample app (tests/user_typing/sample_app.py).

``make typecheck-user`` type-checks the sample; this keeps it working code, not just
code that type-checks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parent / "user_typing" / "sample_app.py"
_spec = importlib.util.spec_from_file_location("sample_app", SAMPLE)
assert _spec is not None and _spec.loader is not None
sample_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sample_app)


@pytest.mark.asyncio
async def test_sample_processor_runs(tmp_path):
    result = await sample_app.run_processor(tmp_path)
    assert result.succeeded == 2
    assert sorted(r.output for r in result.results) == ["HELLO ADA", "HELLO GRACE"]
    assert sample_app.handle(result.results[0]) == "ok"


@pytest.mark.asyncio
async def test_sample_high_level_api_runs():
    outputs = await sample_app.run_high_level()
    assert sorted(o["length"] for o in outputs) == [1, 2, 3, 4, 5]
