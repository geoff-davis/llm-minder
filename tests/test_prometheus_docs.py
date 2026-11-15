"""The Prometheus metric table in docs/api/observers.md matches export_prometheus() (#344)."""

from __future__ import annotations

import re
from pathlib import Path

from llm_minder import MetricsObserver, ProcessingEvent

DOC = Path(__file__).parent.parent / "docs" / "api" / "observers.md"


def _documented() -> dict[str, str]:
    section = DOC.read_text(encoding="utf-8").split("## Prometheus export", 1)[1]
    section = section.split("\n## ", 1)[0]
    rows = re.findall(r"^\| `(llm_minder_\w+)(?:\{(\w+)\})?` \| (counter|gauge) \|", section, re.M)
    return {name + (f"{{{label}}}" if label else ""): kind for name, label, kind in rows}


async def _exported() -> dict[str, str]:
    metrics = MetricsObserver()
    # Populate both labelled counters, which are omitted while empty.
    await metrics.on_event(ProcessingEvent.ITEM_FAILED, {"error_type": "ValueError"})
    await metrics.on_event(
        ProcessingEvent.ITEM_COMPLETED,
        {
            "duration": 1.0,
            "structured_output_recovered": True,
            "structured_output_recovery_reason": "trailing_fence",
        },
    )
    text = await metrics.export_prometheus()
    types = dict(re.findall(r"^# TYPE (\w+) (\w+)$", text, re.M))
    labels = dict(re.findall(r"^(\w+)\{(\w+)=", text, re.M))
    return {
        name + (f"{{{labels[name]}}}" if name in labels else ""): kind
        for name, kind in types.items()
    }


async def test_documented_metrics_match_the_exporter() -> None:
    assert _documented() == await _exported()


async def test_every_exported_line_is_valid_exposition_format() -> None:
    metrics = MetricsObserver()
    await metrics.on_event(ProcessingEvent.ITEM_FAILED, {"error_type": 'Odd"Name'})
    sample = re.compile(r'^llm_minder_\w+(\{\w+="(?:[^"\\]|\\.)*"\})? -?[0-9.e+-]+$')
    for line in (await metrics.export_prometheus()).splitlines():
        assert (
            not line
            or line.startswith("# HELP ")
            or line.startswith("# TYPE ")
            or sample.match(line)
        ), line


# Label values the exposition format needs escaped (#362): a backslash, a double
# quote, a line feed, a trailing backslash, and a literal backslash followed by n.
AWKWARD = ["back\\slash", 'dou"ble', "line\nfeed", "trailing\\", "literal\\n"]
_ESCAPES = {"\\": "\\", '"': '"', "n": "\n"}


def _unescape(text: str) -> str:
    """Decode a label value as a Prometheus text-format parser does."""
    out, chars = [], iter(text)
    for char in chars:
        if char == "\\":
            escaped = next(chars)
            assert escaped in _ESCAPES, f"invalid escape \\{escaped}"
            out.append(_ESCAPES[escaped])
        else:
            out.append(char)
    return "".join(out)


async def _export_awkward() -> str:
    metrics = MetricsObserver()
    for value in AWKWARD:
        await metrics.on_event(ProcessingEvent.ITEM_FAILED, {"error_type": value})
        await metrics.on_event(
            ProcessingEvent.ITEM_COMPLETED,
            {
                "duration": 1.0,
                "structured_output_recovered": True,
                "structured_output_recovery_reason": value,
            },
        )
    return await metrics.export_prometheus()


async def test_awkward_label_values_round_trip() -> None:
    text = await _export_awkward()
    sample = re.compile(r'^(llm_minder_\w+)\{(\w+)="((?:[^"\\\n]|\\.)*)"\} (\S+)$')
    decoded: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("llm_minder_") and "{" in line:
            match = sample.match(line)
            assert match, f"malformed sample line: {line!r}"
            decoded.setdefault(match.group(2), []).append(_unescape(match.group(3)))
    assert decoded == {"error_type": AWKWARD, "reason": AWKWARD}


async def test_awkward_label_values_are_escaped_exactly() -> None:
    text = await _export_awkward()
    assert 'llm_minder_errors_total{error_type="back\\\\slash"} 1' in text
    assert 'llm_minder_errors_total{error_type="dou\\"ble"} 1' in text
    assert 'llm_minder_errors_total{error_type="line\\nfeed"} 1' in text
    assert 'llm_minder_errors_total{error_type="trailing\\\\"} 1' in text
    assert 'llm_minder_errors_total{error_type="literal\\\\n"} 1' in text
    assert 'llm_minder_structured_output_recoveries_by_reason{reason="literal\\\\n"} 1' in text
