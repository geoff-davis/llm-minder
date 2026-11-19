"""Property tests for result serialization and message fingerprints (#327).

- A ``WorkItemResult`` built from generated JSON-native values round-trips
  through ``work_item_result_to_dict`` / ``work_item_result_from_dict`` and
  through ``BatchResult.to_json`` / ``from_json``: the encoded mapping is a
  fixed point, survives a strict JSON round trip, and decodes to an equal result
  whenever the library's redaction left the values unchanged.
- A message-list prompt (#315, #377) fingerprints in its own ``messages:v1:``
  domain: deterministic, independent of key order, unchanged by normalization,
  and never equal to the fingerprint of any string prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_minder import BatchResult, WorkItemResult
from llm_minder._internal.artifact_codec import MESSAGES_FINGERPRINT_TAG, fingerprint_messages
from llm_minder._internal.messages import normalize_messages
from llm_minder.base import AttemptTiming, WorkItemTiming
from llm_minder.serialization import work_item_result_from_dict, work_item_result_to_dict

# Keys from a small alphabet that can't spell a redacted key such as "token" or "api_key".
_KEYS = st.text(alphabet="abcdefgh_", min_size=1, max_size=6)
_TEXT = st.text(alphabet=st.characters(exclude_categories=("Cs",)), max_size=20)
_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(2**63), 2**63),
    st.floats(allow_nan=False, allow_infinity=False),
    _TEXT,
)
_JSON = st.recursive(
    _SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3), st.dictionaries(_KEYS, children, max_size=3)
    ),
    max_leaves=8,
)
_SECONDS = st.floats(0.0, 1e6, allow_nan=False, allow_infinity=False)
_CATEGORY = st.one_of(st.none(), st.sampled_from(["rate_limit", "timeout", "batch_aborted"]))
_TOKENS = st.dictionaries(
    st.sampled_from(
        [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
        ]
    ),
    st.integers(0, 10**12),
)
_OPTIONAL_INT = st.one_of(st.none(), st.integers(0, 10**9))


def _attempts() -> st.SearchStrategy[AttemptTiming]:
    return st.builds(
        AttemptTiming,
        attempt=st.integers(1, 20),
        try_number=st.integers(1, 20),
        total_seconds=_SECONDS,
        admission_wait_seconds=_SECONDS,
        startup_ramp_wait_seconds=_SECONDS,
        execution_seconds=_SECONDS,
        provider_seconds=st.one_of(st.none(), _SECONDS),
        cooldown_wait_seconds=_SECONDS,
        retry_backoff_seconds=_SECONDS,
        success=st.booleans(),
        error_type=st.one_of(st.none(), st.sampled_from(["ValueError", "TimeoutError"])),
        error_category=_CATEGORY,
        timeout_category=st.one_of(st.none(), st.sampled_from(["framework_attempt_timeout"])),
        quota_wait_seconds=_SECONDS,
        estimated_input_tokens=_OPTIONAL_INT,
        estimated_output_tokens=_OPTIONAL_INT,
        reserved_tokens=st.integers(0, 10**9),
        reported_tokens=_OPTIONAL_INT,
        reconciliation_delta_tokens=st.one_of(st.none(), st.integers(-(10**9), 10**9)),
        quota_scope_id=_OPTIONAL_INT,
    )


@st.composite
def results(draw: st.DrawFn) -> WorkItemResult[Any, Any]:
    success = draw(st.booleans())
    return WorkItemResult(
        item_id=draw(_TEXT.filter(lambda s: s.strip() != "")),
        success=success,
        output=draw(_JSON) if success else None,
        error=None if success else draw(_TEXT),
        context=draw(_JSON),
        token_usage=draw(_TOKENS),
        metadata=draw(st.one_of(st.none(), st.dictionaries(_KEYS, _JSON, max_size=3))),
        admission_wait_seconds=draw(_SECONDS),
        timing=WorkItemTiming(
            total_seconds=draw(_SECONDS),
            attempts=draw(st.lists(_attempts(), max_size=3)),
            timeout_category=draw(
                st.one_of(st.none(), st.sampled_from(["framework_total_item_timeout"]))
            ),
        ),
        submission_index=draw(st.one_of(st.none(), st.integers(0, 10**6))),
        error_category=None if success else draw(_CATEGORY),
        replayed_from_artifact=draw(st.booleans()),
    )


def _check_result_round_trip(result: WorkItemResult[Any, Any]) -> None:
    encoded = work_item_result_to_dict(result)
    assert json.loads(json.dumps(encoded, allow_nan=False)) == encoded
    decoded = work_item_result_from_dict(encoded)
    assert work_item_result_to_dict(decoded) == encoded
    assert decoded.submission_index == result.submission_index
    if encoded["error"] == result.error:
        assert decoded == result


def _check_batch_round_trip(items: list[WorkItemResult[Any, Any]]) -> None:
    batch = BatchResult(items)
    payload = batch.to_json()
    restored = BatchResult.from_json(payload)
    assert restored.to_json() == payload
    assert [r.item_id for r in restored.results] == [r.item_id for r in items]


_MESSAGE_EXTRAS = st.dictionaries(
    st.sampled_from(["name", "cache_control", "x_extra"]), _JSON, max_size=2
)


@st.composite
def message_lists(draw: st.DrawFn) -> list[dict[str, Any]]:
    messages = []
    for _ in range(draw(st.integers(1, 4))):
        content: Any
        if draw(st.booleans()):
            content = draw(_TEXT)
        else:
            content = draw(
                st.lists(
                    st.fixed_dictionaries(
                        {"type": st.just("text"), "text": _TEXT}, optional={"x": _JSON}
                    ),
                    max_size=3,
                )
            )
        message = {
            "role": draw(st.sampled_from(["system", "user", "assistant"])),
            "content": content,
        }
        message.update(draw(_MESSAGE_EXTRAS))
        messages.append(message)
    return messages


def _reversed_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reversed_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed_keys(item) for item in value]
    return value


def _check_message_fingerprint(messages: list[dict[str, Any]], prompt: str) -> None:
    fingerprint = fingerprint_messages(messages)
    # The literal, not the constant: the domain must not silently change.
    assert fingerprint.startswith("messages:v1:")
    assert MESSAGES_FINGERPRINT_TAG == "messages:v1:"
    assert re.fullmatch("[0-9a-f]{64}", fingerprint.removeprefix("messages:v1:"))
    assert fingerprint == fingerprint_messages(messages)
    assert fingerprint == fingerprint_messages(_reversed_keys(messages))
    normalized = normalize_messages(messages)
    assert normalize_messages(normalized) == normalized
    assert fingerprint_messages(normalized) == fingerprint
    # A string prompt's fingerprint is a bare SHA-256 hex digest, so the domains never meet.
    assert not re.fullmatch("[0-9a-f]{64}", fingerprint)
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() != fingerprint
    assert hashlib.sha256(fingerprint.encode("utf-8")).hexdigest() != fingerprint


@given(results())
def test_result_round_trip(result: WorkItemResult[Any, Any]) -> None:
    _check_result_round_trip(result)


@given(st.lists(results(), max_size=3))
def test_batch_result_round_trip(items: list[WorkItemResult[Any, Any]]) -> None:
    _check_batch_round_trip(items)


@given(message_lists(), _TEXT)
def test_message_fingerprint_domain(messages: list[dict[str, Any]], prompt: str) -> None:
    _check_message_fingerprint(messages, prompt)


@pytest.mark.slow
@settings(settings.get_profile("thorough"))
@given(results())
def test_result_round_trip_thorough(result: WorkItemResult[Any, Any]) -> None:
    _check_result_round_trip(result)


@pytest.mark.slow
@settings(settings.get_profile("thorough"))
@given(st.lists(results(), max_size=3))
def test_batch_result_round_trip_thorough(items: list[WorkItemResult[Any, Any]]) -> None:
    _check_batch_round_trip(items)


@pytest.mark.slow
@settings(settings.get_profile("thorough"))
@given(message_lists(), _TEXT)
def test_message_fingerprint_domain_thorough(messages: list[dict[str, Any]], prompt: str) -> None:
    _check_message_fingerprint(messages, prompt)
