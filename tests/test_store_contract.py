"""One contract suite for both artifact stores (#326).

Each test runs on JsonlArtifactStore and SqliteArtifactStore (ids ``jsonl`` and
``sqlite``) and uses only the surface they share. Backend-specific behaviour (JSONL
line numbers and truncated-tail recovery, SQLite indexes and maintenance) stays in
test_artifacts.py and test_sqlite_artifacts.py.
"""

from __future__ import annotations

import inspect
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from llm_minder import (
    ArtifactError,
    ArtifactFormatError,
    ArtifactIdentity,
    ArtifactIOError,
    ArtifactSerializationError,
    JsonlArtifactStore,
    LLMWorkItem,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    WorkItemResult,
    process_prompts,
    process_stream,
)
from tests.test_artifacts import _CountingStrategy, _identity, _records


@pytest.fixture(
    params=[
        pytest.param(JsonlArtifactStore, id="jsonl"),
        pytest.param(SqliteArtifactStore, id="sqlite"),
    ]
)
def store_type(request: pytest.FixtureRequest) -> Any:
    """The store class under test."""
    return request.param


@pytest.fixture
def suffix(store_type: Any) -> str:
    """The file suffix conventional for ``store_type``."""
    return ".jsonl" if store_type is JsonlArtifactStore else ".sqlite"


@pytest.mark.asyncio
async def test_inspection_restores_explicitly_persisted_context(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"stored-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", {"tenant": "current", "values": [1, 2]})],
        artifact_store=store_type(path, identity=_identity(), include_context=True),
    )

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path)
    else:
        materialized = await store_type.read_results(path)
    assert materialized.results[0].context == {
        "tenant": "current",
        "values": [1, 2],
    }

    reader = store_type(path)
    try:
        streamed = [result async for result in reader.iter_results()]
    finally:
        await reader.close()
    assert streamed[0].context == {"tenant": "current", "values": [1, 2]}


@pytest.mark.asyncio
async def test_replay_uses_current_context_without_decoding_historical_context(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"replay-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [
            ("dummy", "dummy", {"run": "historical"}),
            ("item", "prompt", {"run": "historical"}),
        ],
        artifact_store=store_type(
            path,
            identity=_identity(),
            include_context=True,
            context_in_identity=False,
        ),
    )

    decoder_calls: list[Any] = []

    def context_decoder(value: Any) -> Any:
        decoder_calls.append(value)
        raise AssertionError("historical context must not be decoded during replay")

    current_context = {"run": "current"}
    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("item", "prompt", current_context)],
        artifact_store=store_type(
            path,
            identity=_identity(),
            context_in_identity=False,
            context_decoder=context_decoder,
        ),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == []
    assert decoder_calls == []
    assert result.results[0].context is current_context
    assert result.results[0].submission_index == 0


@pytest.mark.asyncio
async def test_inspection_applies_custom_context_decoder(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    class ContextPayload:
        def __init__(self, value: str) -> None:
            self.value = value

    def encoder(value: Any) -> Any:
        if isinstance(value, ContextPayload):
            return {"payload": value.value}
        return value

    decoded: list[Any] = []

    def decoder(value: Any) -> ContextPayload:
        decoded.append(value)
        return ContextPayload(value["payload"])

    path = tmp_path / f"custom-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", ContextPayload("persisted"))],
        artifact_store=store_type(
            path,
            identity=_identity(),
            include_context=True,
            encoder=encoder,
        ),
    )

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path, context_decoder=decoder)
    else:
        materialized = await store_type.read_results(path, context_decoder=decoder)
    restored = materialized.results[0].context
    assert isinstance(restored, ContextPayload)
    assert restored.value == "persisted"

    reader = store_type(path, context_decoder=decoder)
    try:
        streamed = [result async for result in reader.iter_results()]
    finally:
        await reader.close()
    streamed_context = streamed[0].context
    assert isinstance(streamed_context, ContextPayload)
    assert streamed_context.value == "persisted"
    assert decoded == [{"payload": "persisted"}, {"payload": "persisted"}]


@pytest.mark.asyncio
async def test_inspection_does_not_decode_context_that_was_not_persisted(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"omitted-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [
            ("omitted", "one", {"private": True}),
            ("none", "two", None),
        ],
        artifact_store=store_type(path, identity=_identity(), include_context=False),
    )

    decoder_calls: list[Any] = []

    def decoder(value: Any) -> Any:
        decoder_calls.append(value)
        return value

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path, context_decoder=decoder)
    else:
        materialized = await store_type.read_results(path, context_decoder=decoder)
    assert [result.context for result in materialized.results] == [None, None]
    assert decoder_calls == []


@pytest.mark.asyncio
async def test_original_none_context_does_not_invoke_decoder(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"none-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("none", "prompt", None)],
        artifact_store=store_type(path, identity=_identity(), include_context=True),
    )
    decoder_calls: list[Any] = []

    def decoder(value: Any) -> Any:
        decoder_calls.append(value)
        return value

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path, context_decoder=decoder)
    else:
        materialized = await store_type.read_results(path, context_decoder=decoder)
    assert materialized.results[0].context is None
    assert decoder_calls == []


@pytest.mark.asyncio
async def test_context_decoder_failure_uses_artifact_format_error(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"context-decoder-error{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", {"stored": True})],
        artifact_store=store_type(path, identity=_identity(), include_context=True),
    )

    def fail_decoder(value: Any) -> Any:
        raise RuntimeError(f"cannot decode {value!r}")

    with pytest.raises(ArtifactFormatError, match="Stored context decoder failed"):
        if store_type is JsonlArtifactStore:
            store_type.read_results(path, context_decoder=fail_decoder)
        else:
            await store_type.read_results(path, context_decoder=fail_decoder)


@pytest.mark.asyncio
async def test_output_decoder_failure_uses_specific_artifact_format_error(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"output-decoder-error{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt")],
        artifact_store=store_type(path, identity=_identity()),
    )

    def fail_decoder(value: Any) -> Any:
        raise RuntimeError(f"cannot decode {value!r}")

    with pytest.raises(ArtifactFormatError, match="Stored output decoder failed"):
        if store_type is JsonlArtifactStore:
            store_type.read_results(path, output_decoder=fail_decoder)
        else:
            await store_type.read_results(path, output_decoder=fail_decoder)


@pytest.mark.parametrize("explicit", [False, True])
async def test_art2_inspect_then_run_registers_new_identity(tmp_path, store_type, explicit):
    class Named(_CountingStrategy):
        def __init__(self, name):
            super().__init__()
            self.model = type("Model", (), {"_model": name})()

    path = tmp_path / "identity.artifact"
    await process_prompts(Named("old"), ["a"], artifact_store=store_type(path))
    kwargs = {"identity": _identity(model="new")} if explicit else {}
    store = store_type(path, **kwargs)
    assert len([r async for r in store.iter_results()]) == 1
    strategy = Named("new")
    result = await process_prompts(
        strategy,
        ["a", "b"],
        artifact_store=store,
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert result.succeeded == 2
    # Both items run concurrently; only the set of executed prompts is stable.
    assert sorted(strategy.calls) == ["a", "b"]
    reader = store_type(path)
    try:
        assert len([r async for r in reader.iter_results()]) == 3
    finally:
        await reader.close()


@pytest.mark.parametrize("bad", [float("nan"), {1: "value"}, "bad\ud800text"])
@pytest.mark.parametrize("surface", ["batch", "stream"])
async def test_art4_invalid_output_is_local_and_cannot_replay(tmp_path, store_type, bad, surface):
    class Output(_CountingStrategy):
        async def execute(self, prompt, attempt, timeout, state=None):
            self.calls.append(prompt)
            return bad if prompt == "20" else prompt, {"total_tokens": 7}, None

    path = tmp_path / "invalid-output.artifact"
    strategy = Output()
    prompts = [(str(i), str(i)) for i in range(1, 22)]
    kwargs = {
        "artifact_store": store_type(path, identity=_identity()),
        "config": ProcessorConfig(max_workers=4),
    }
    if surface == "batch":
        results = (await process_prompts(strategy, prompts, **kwargs)).results
    else:
        results = [r async for r in process_stream(strategy, prompts, **kwargs)]
    assert sum(r.success for r in results) == 20
    failed = next(r for r in results if r.item_id == "20")
    assert not failed.success
    assert failed.output is bad
    assert failed.error_category == "artifact_serialization_error"
    assert "$." in failed.error
    assert failed.token_usage["total_tokens"] == 7
    reader = store_type(path)
    try:
        stored = [r async for r in reader.iter_results()]
    finally:
        await reader.close()
    assert len(stored) == 21
    stored_bad = next(r for r in stored if r.item_id == "20")
    assert not stored_bad.success
    assert stored_bad.output is None
    strategy.calls.clear()
    replay = await process_prompts(
        strategy,
        prompts,
        artifact_store=store_type(path, identity=_identity()),
        resume=ResumePolicy.REUSE_ALL,
    )
    assert replay.succeeded == 20
    assert strategy.calls == ["20"]


async def test_art3_surrogate_prompt_reports_prompt_path(tmp_path, store_type):
    strategy = _CountingStrategy()
    result = await process_prompts(
        strategy,
        ["\ud800"],
        artifact_store=store_type(tmp_path / "prompt", identity=_identity()),
    )
    failed = result.results[0]
    assert failed.error_category == "artifact_preparation_error"
    assert "$.prompt" in failed.error
    assert "Context fingerprinter failed" not in failed.error
    assert strategy.calls == []


async def test_art6_decoders_only_receive_present_successful_outputs(tmp_path, store_type):
    from pydantic import BaseModel

    class Payload(BaseModel):
        value: int

    path = tmp_path / "decode.artifact"
    strategy = _CountingStrategy()
    store = store_type(path, identity=_identity())
    for name, success, output in [
        ("ok", True, {"value": 1}),
        ("failed", False, None),
        ("empty", True, None),
        ("failed-output", False, {"bad": 1}),
    ]:
        item = LLMWorkItem(name, strategy, name)
        key = await store.prepare_item(item)
        await store.append(item, key, WorkItemResult(name, success, output=output))
    await store.close()
    decoded = []

    def decode(value):
        decoded.append(value)
        return Payload.model_validate(value)

    kwargs = {"output_decoder": decode}
    if store_type is JsonlArtifactStore:
        read = store_type.read_results(path, **kwargs)
    else:
        read = await store_type.read_results(path, **kwargs)
    assert len(read.results) == 4
    reader = store_type(path, **kwargs)
    try:
        assert len([r async for r in reader.iter_results()]) == 4
    finally:
        await reader.close()
    result = await process_prompts(
        strategy,
        [(name, name) for name in ["ok", "failed", "empty", "failed-output"]],
        artifact_store=store_type(path, identity=_identity(), **kwargs),
        resume=ResumePolicy.REUSE_ALL,
    )
    assert all(r.replayed_from_artifact for r in result.results)
    assert decoded == [{"value": 1}] * 3
    assert strategy.calls == []


@pytest.mark.parametrize("mode", ["error", "metadata", "cost"])
async def test_art3_surrogate_diagnostics_and_bad_metadata_are_isolated(tmp_path, store_type, mode):
    from llm_minder.core import RetryConfig

    class Output(_CountingStrategy):
        async def execute(self, prompt, attempt, timeout, state=None):
            if prompt == "bad" and mode == "error":
                raise RuntimeError("provider said \ud800")
            return (
                prompt,
                {"total_tokens": 7},
                {"note": "\udc80"} if prompt == "bad" and mode == "metadata" else None,
            )

    def cost(result):
        if result.item_id == "bad":
            raise ValueError("cost hook failed")
        return 1.0

    path = tmp_path / "surrogates"
    result = await process_prompts(
        Output(),
        [("bad", "bad"), ("good", "good")],
        config=ProcessorConfig(max_workers=2, retry=RetryConfig(max_attempts=1)),
        artifact_store=store_type(
            path, identity=_identity(), cost_calculator=cost if mode == "cost" else None
        ),
    )
    assert result.succeeded == 1
    bad = next(r for r in result.results if r.item_id == "bad")
    if mode == "error":
        assert bad.error_category != "artifact_serialization_error"
        assert "\ud800" in bad.error
    else:
        assert bad.error_category == "artifact_serialization_error"
        assert bad.output == "bad"
    reader = store_type(path)
    try:
        records = [r async for r in reader.iter_results()]
    finally:
        await reader.close()
    assert len(records) == 2
    if mode == "error":
        stored_bad = next(r for r in records if r.item_id == "bad")
        assert "\\ud800" in stored_bad.error
        stored_bad.error.encode("utf-8")


@pytest.mark.parametrize(
    "category",
    [
        "batch_aborted",
        "batch_deadline_exceeded",
        "batch_budget_exceeded",
        "framework_total_item_timeout",
    ],
)
async def test_art4_guardrail_fallback_preserves_category(tmp_path, store_type, category):
    from llm_minder import ParallelBatchProcessor

    path = tmp_path / "audit"
    async with ParallelBatchProcessor(
        artifact_store=store_type(path, identity=_identity())
    ) as processor:

        async def terminal(prepared, worker_id):
            return WorkItemResult(
                prepared.effective_item.item_id,
                False,
                error="stopped",
                error_category=category,
                metadata={"bad": float("nan")},
            )

        processor._executor.execute_prepared = terminal
        await processor.add_work(LLMWorkItem("x", _CountingStrategy(), "x"))
        result = await processor.process_all()
    assert result.results[0].error_category == category
    reader = store_type(path)
    try:
        records = [r async for r in reader.iter_results()]
    finally:
        await reader.close()
    assert records[0].error_category == category
    assert records[0].metadata is None
    strategy = _CountingStrategy()
    await process_prompts(
        strategy,
        [("x", "x")],
        artifact_store=store_type(path, identity=_identity()),
        resume=ResumePolicy.REUSE_ALL,
    )
    assert strategy.calls == ["x"]


async def test_art4_unprepared_key_is_not_a_recoverable_record_error(tmp_path, store_type):
    from llm_minder import ArtifactSerializationError
    from llm_minder.artifacts import _RecordSerializationError

    store = store_type(tmp_path / "unprepared", identity=_identity())
    try:
        with pytest.raises(ArtifactSerializationError) as caught:
            await store.append(
                LLMWorkItem("x", _CountingStrategy(), "x"),
                object(),
                WorkItemResult("x", True, output="x"),
            )
        assert not isinstance(caught.value, _RecordSerializationError)
    finally:
        await store.close()


async def test_art4_custom_store_without_private_fallback_stays_fatal(tmp_path, store_type):
    from llm_minder.artifacts import _RecordSerializationError

    backing = store_type(tmp_path / "custom", identity=_identity())

    class Custom:
        async def prepare_item(self, item):
            return await backing.prepare_item(item)

        async def lookup(self, item, key, policy):
            return None

        async def append(self, item, key, result):
            raise _RecordSerializationError("custom failure")

        async def close(self):
            await backing.close()

    with pytest.raises(_RecordSerializationError, match="custom failure"):
        await process_prompts(_CountingStrategy(), ["x"], artifact_store=Custom())


@pytest.mark.parametrize(
    "category",
    [
        None,
        "batch_aborted",
        "batch_deadline_exceeded",
        "batch_budget_exceeded",
        "framework_total_item_timeout",
    ],
)
async def test_art4_fallback_io_keeps_existing_fatal_vs_best_effort_policy(
    tmp_path, store_type, category
):
    from llm_minder import ParallelBatchProcessor

    store = store_type(tmp_path / "fallback-io", identity=_identity())

    async def fail_fallback(*args):
        raise ArtifactIOError("fallback disk failure")

    store._append_serialization_failure = fail_fallback
    processor = ParallelBatchProcessor(artifact_store=store)

    async def terminal(prepared, worker_id):
        return WorkItemResult("x", category is None, output=float("nan"), error_category=category)

    processor._executor.execute_prepared = terminal
    await processor.add_work(LLMWorkItem("x", _CountingStrategy(), "x"))
    try:
        if category in {"batch_aborted", "batch_deadline_exceeded", "batch_budget_exceeded"}:
            result = await processor.process_all()
            assert result.results[0].error_category == category
        else:
            with pytest.raises(ArtifactIOError, match="fallback disk failure"):
                await processor.process_all()
    finally:
        await processor.cleanup()


async def test_art4_fail_fast_runs_after_failure_checkpoint(tmp_path, store_type):
    from llm_minder import GuardrailConfig

    class Output(_CountingStrategy):
        async def execute(self, prompt, attempt, timeout, state=None):
            self.calls.append(prompt)
            return float("nan"), {}, None

    path = tmp_path / "abort"
    strategy = Output()
    result = await process_prompts(
        strategy,
        [("a", "a"), ("b", "b")],
        artifact_store=store_type(path, identity=_identity()),
        config=ProcessorConfig(
            max_workers=1,
            guardrails=GuardrailConfig(abort_on_error_categories={"artifact_serialization_error"}),
        ),
    )
    assert result.failed == 2
    assert strategy.calls == ["a"]
    assert result.termination.triggering_item_id == "a"
    reader = store_type(path)
    try:
        records = [r async for r in reader.iter_results()]
    finally:
        await reader.close()
    assert (
        next(r for r in records if r.item_id == "a").error_category
        == "artifact_serialization_error"
    )


@pytest.mark.parametrize("channel", ["context", "item_id"])
async def test_art3_surrogate_input_has_accurate_path(tmp_path, store_type, channel):
    prompt = ("x", "x", "\ud800") if channel == "context" else ("\ud800", "x")
    result = await process_prompts(
        _CountingStrategy(),
        [prompt],
        artifact_store=store_type(tmp_path / "input", identity=_identity()),
    )
    assert result.failed == 1
    assert f"$.{channel}" in result.results[0].error
    assert "Context fingerprinter failed" not in result.results[0].error


@pytest.mark.parametrize("consumer", ["read", "iter"])
async def test_art8_decode_error_sequence_messages_are_one_based(tmp_path, store_type, consumer):
    path = tmp_path / "sequence"
    await process_prompts(
        _CountingStrategy(), ["x"], artifact_store=store_type(path, identity=_identity())
    )

    def decoder(value):
        raise ValueError("cannot decode")

    with pytest.raises(ArtifactFormatError, match="sequence 1 .*item"):
        if consumer == "read":
            if store_type is JsonlArtifactStore:
                store_type.read_results(path, output_decoder=decoder)
            else:
                await store_type.read_results(path, output_decoder=decoder)
        else:
            reader = store_type(path, output_decoder=decoder)
            try:
                _ = [r async for r in reader.iter_results()]
            finally:
                await reader.close()
    if store_type is JsonlArtifactStore:
        assert _records(path)[1]["record_sequence"] == 0


async def test_art3_manifest_metadata_is_validated_before_open(tmp_path, store_type):
    from llm_minder import ArtifactSerializationError

    path = tmp_path / "manifest"
    with pytest.raises(ArtifactSerializationError, match="user_metadata"):
        store_type(path, identity=_identity(), user_metadata={"bad": "\ud800"})
    assert not path.exists()


@pytest.mark.asyncio
async def test_append_without_resolved_identity_raises_artifact_error(tmp_path, store_type):
    # Reopening an existing artifact without an identity and appending an item
    # this store never prepared reports the missing identity on both stores. The
    # JSONL store used to wrap it as a "Cost calculator failed" serialization error.
    from llm_minder._internal.artifact_codec import fingerprint_work_item

    path = tmp_path / ("a.jsonl" if store_type is JsonlArtifactStore else "a.sqlite")
    item = LLMWorkItem(item_id="x", strategy=_CountingStrategy(), prompt="p")
    await process_prompts(
        _CountingStrategy(), ["old"], artifact_store=store_type(path, identity=_identity())
    )
    store = store_type(path)
    prepared = fingerprint_work_item(
        item, context_in_identity=True, encoder=None, context_fingerprinter=None
    )
    try:
        with pytest.raises(ArtifactError, match="Artifact identity is not resolved yet") as info:
            await store.append(item, prepared, WorkItemResult(item_id="x", success=True))
        assert not isinstance(info.value, ArtifactSerializationError)
    finally:
        await store.close()


# --- Gaps found by the #326 audit: behaviour asserted on only one backend before. ---


async def _read(store_type: Any, path: Path, **kwargs: Any) -> Any:
    """``read_results`` is sync on JSONL and async on SQLite."""
    result = store_type.read_results(path, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _corrupt_newest_schema_version(store_type: Any, path: Path, value: int) -> None:
    """Give the newest item record an unsupported schema version (backend-specific)."""
    if store_type is JsonlArtifactStore:
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[-1])
        record["artifact_schema_version"] = value
        lines[-1] = json.dumps(record)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE item_records SET logical_schema_version = ? "
            "WHERE record_sequence = (SELECT MAX(record_sequence) FROM item_records)",
            (value,),
        )


@pytest.mark.parametrize(
    "identity_change",
    [
        {"provider": "other-provider"},
        {"prompt_version": "prompt-v2"},
        {"parser_version": "parser-v2"},
        {"application_version": "app-v2"},
    ],
)
async def test_changed_identity_field_is_not_reused(
    tmp_path: Path, store_type: Any, suffix: str, identity_change: dict[str, str]
) -> None:
    path = tmp_path / f"identity{suffix}"
    await process_prompts(
        _CountingStrategy(), [("x", "one")], artifact_store=store_type(path, identity=_identity())
    )
    strategy = _CountingStrategy()
    await process_prompts(
        strategy,
        [("x", "one")],
        artifact_store=store_type(path, identity=_identity(**identity_change)),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert strategy.calls == ["one"]


async def test_sensitive_values_change_the_hash_but_are_not_persisted(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"secret{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", {"api_key": "context-old"})],
        artifact_store=store_type(path, identity=_identity()),
    )
    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("item", "prompt", {"api_key": "context-new"})],
        artifact_store=store_type(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == ["prompt"]
    assert not result.results[0].replayed_from_artifact

    identity_path = tmp_path / f"identity-secret{suffix}"
    for secret in ("old-secret", "new-secret"):
        identity = ArtifactIdentity(provider="test", model="model", extra={"api_key": secret})
        strategy = _CountingStrategy()
        await process_prompts(
            strategy,
            [("item", "prompt")],
            artifact_store=store_type(identity_path, identity=identity),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert strategy.calls == ["prompt"]

    for artifact in (path, identity_path):
        stored = b"".join(p.read_bytes() for p in artifact.parent.glob(f"{artifact.name}*"))
        for secret in (b"context-old", b"context-new", b"old-secret", b"new-secret"):
            assert secret not in stored


async def test_concurrent_workers_persist_each_item_exactly_once(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"concurrent{suffix}"
    result = await process_prompts(
        _CountingStrategy(delay=0.001),
        [(str(index), f"p{index}") for index in range(40)],
        config=ProcessorConfig(max_workers=10),
        artifact_store=store_type(path, identity=_identity()),
    )
    assert result.total_items == 40
    stored = await _read(store_type, path)
    assert sorted(r.item_id for r in stored.results) == sorted(str(i) for i in range(40))


async def test_failures_are_persisted_without_a_traceback(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"failures{suffix}"
    await process_prompts(
        _CountingStrategy(failures={"bad"}),
        [("ok", "good"), ("failed", "bad")],
        artifact_store=store_type(path, identity=_identity()),
    )
    successes = await _read(store_type, path, successes_only=True)
    assert [r.item_id for r in successes.results] == ["ok"]
    failed = [r for r in (await _read(store_type, path)).results if r.item_id == "failed"]
    assert failed[0].error and "traceback" not in failed[0].error.lower()
    assert b"traceback" not in path.read_bytes().lower()


async def test_interrupted_stream_resumes_only_unfinished_items(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"interrupted{suffix}"
    items = [(str(index), f"p{index}") for index in range(6)]
    stream = process_stream(
        _CountingStrategy(delay=0.005),
        items,
        config=ProcessorConfig(max_workers=1, max_queue_size=1),
        artifact_store=store_type(path, identity=_identity()),
    )
    async for _ in stream:
        break
    await stream.aclose()

    completed = len((await _read(store_type, path)).results)
    assert 1 <= completed < 6
    second = _CountingStrategy()
    result = await process_prompts(
        second,
        items,
        artifact_store=store_type(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert result.total_items == 6
    assert len(second.calls) == 6 - completed


async def test_items_with_and_without_context_both_replay(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"null-context{suffix}"
    entries: list[Any] = [("none", "a"), ("context", "b", {"v": 1})]
    await process_prompts(
        _CountingStrategy(), entries, artifact_store=store_type(path, identity=_identity())
    )
    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        entries,
        artifact_store=store_type(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == []
    assert all(r.replayed_from_artifact for r in result.results)


@pytest.mark.parametrize("policy", [ResumePolicy.REUSE_SUCCESSES, ResumePolicy.REUSE_ALL])
async def test_newest_unsupported_record_is_not_hidden_by_an_older_one(
    tmp_path: Path, store_type: Any, suffix: str, policy: ResumePolicy
) -> None:
    path = tmp_path / f"newest-unsupported{suffix}"

    class _Fixed(_CountingStrategy):
        def __init__(self, output: str) -> None:
            super().__init__()
            self.output = output

        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            _, usage, metadata = await super().execute(*args, **kwargs)
            return self.output, usage, metadata

    # Same item and prompt twice, so the older record is a compatible fallback.
    for output in ("older", "newer"):
        await process_prompts(
            _Fixed(output),
            [("id", "p")],
            artifact_store=store_type(path, identity=_identity()),
        )
    item = LLMWorkItem(item_id="id", strategy=_CountingStrategy(), prompt="p")
    store = store_type(path, identity=_identity())
    try:
        replayed = await store.lookup(item, await store.prepare_item(item), policy)
        assert replayed is not None and replayed.output == "newer"
    finally:
        await store.close()

    _corrupt_newest_schema_version(store_type, path, 2)
    store = store_type(path, identity=_identity())
    try:
        # JSONL validates the whole file in prepare_item; SQLite reads the row in lookup.
        with pytest.raises(ArtifactFormatError, match="future artifact schema version 2"):
            key = await store.prepare_item(item)
            await store.lookup(item, key, policy)
    finally:
        await store.close()


async def _store_with_one_item(store_type: Any, path: Path) -> tuple[Any, Any, Any]:
    await process_prompts(
        _CountingStrategy(), [("id", "p")], artifact_store=store_type(path, identity=_identity())
    )
    store = store_type(path, identity=_identity())
    item = LLMWorkItem(item_id="id", strategy=_CountingStrategy(), prompt="p")
    key = await store.prepare_item(item)
    return store, item, key


async def test_close_is_idempotent(tmp_path: Path, store_type: Any, suffix: str) -> None:
    store, item, key = await _store_with_one_item(store_type, tmp_path / f"close{suffix}")
    assert await store.lookup(item, key, ResumePolicy.REUSE_ALL) is not None
    await store.close()
    await store.close()


async def test_operations_after_close_raise_closed(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    store, item, key = await _store_with_one_item(store_type, tmp_path / f"closed{suffix}")
    result = WorkItemResult(item_id="id", success=True, output="P")
    await store.close()
    with pytest.raises(ArtifactIOError, match="closed"):
        await store.prepare_item(item)
    with pytest.raises(ArtifactIOError, match="closed"):
        await store.lookup(item, key, ResumePolicy.REUSE_ALL)
    with pytest.raises(ArtifactIOError, match="closed"):
        await store.append(item, key, result)


async def test_iter_results_after_close_raises_closed(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    store, _item, _key = await _store_with_one_item(store_type, tmp_path / f"iter-closed{suffix}")
    await store.close()
    with pytest.raises(ArtifactIOError, match="closed"):
        [r async for r in store.iter_results()]


async def test_iter_results_successes_only_filters_failures(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"iter-filter{suffix}"
    await process_prompts(
        _CountingStrategy(failures={"bad"}),
        [("ok", "good"), ("failed", "bad")],
        artifact_store=store_type(path, identity=_identity()),
    )
    store = store_type(path, identity=_identity())
    try:
        assert [r.item_id async for r in store.iter_results(successes_only=True)] == ["ok"]
        assert sorted([r.item_id async for r in store.iter_results()]) == ["failed", "ok"]
    finally:
        await store.close()


async def test_directory_path_is_rejected(tmp_path: Path, store_type: Any) -> None:
    directory = tmp_path / "dir"
    directory.mkdir()
    store = store_type(directory, identity=_identity())
    item = LLMWorkItem(item_id="id", strategy=_CountingStrategy(), prompt="p")
    try:
        # The wording differs (JSONL: "Is a directory"; SQLite: "is a directory").
        with pytest.raises(ArtifactIOError, match="(?i)directory"):
            await store.prepare_item(item)
    finally:
        await store.close()


async def test_zero_length_file_starts_a_new_artifact(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"empty{suffix}"
    path.touch()
    result = await process_prompts(
        _CountingStrategy(), [("id", "p")], artifact_store=store_type(path, identity=_identity())
    )
    assert result.succeeded == 1
    assert [r.item_id for r in (await _read(store_type, path)).results] == ["id"]


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs enforced POSIX modes")
async def test_read_results_needs_no_write_permission(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    directory = tmp_path / "readonly"
    directory.mkdir()
    path = directory / f"artifact{suffix}"
    await process_prompts(
        _CountingStrategy(), [("id", "p")], artifact_store=store_type(path, identity=_identity())
    )
    path.chmod(0o444)
    directory.chmod(0o555)
    try:
        assert [r.item_id for r in (await _read(store_type, path)).results] == ["id"]
    finally:
        directory.chmod(0o755)
        path.chmod(0o644)
