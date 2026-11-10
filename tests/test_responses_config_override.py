"""Per-call config overrides model defaults under any alias on Responses surfaces.

Regression for rc1: a default ``max_output_tokens`` plus a per-call ``max_tokens``
(or a ``json_mode`` default plus a per-call ``text.format``) raised a
non-retryable ``ValueError`` on ``OpenAIModel``, and ``DeepSeekModel`` silently
kept the default bound.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_minder import (
    DeepSeekModel,
    DeepSeekStrategy,
    ErrorInfo,
    LLMWorkItem,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    RetryState,
    TruncatedResponseError,
)
from llm_minder._internal.responses_translation import (
    merge_request_configs,
    translate_request_config,
)
from llm_minder.classifiers.openai import OpenAIErrorClassifier
from tests._openai_http import openai_http

_SCHEMA_FORMAT = {"type": "json_schema", "name": "x", "schema": {"type": "object"}}

# ── merge + translate (pure) ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("defaults", "override", "expected"),
    [
        # Output-token bound: every spelling on either side.
        ({"max_output_tokens": 1000}, {"max_tokens": 4000}, {"max_output_tokens": 4000}),
        ({"max_tokens": 1000}, {"max_output_tokens": 4000}, {"max_output_tokens": 4000}),
        ({"max_completion_tokens": 1000}, {"max_tokens": 4000}, {"max_output_tokens": 4000}),
        ({"max_tokens": 1000}, {"max_completion_tokens": 4000}, {"max_output_tokens": 4000}),
        ({"max_tokens": 1000}, {"max_tokens": 4000}, {"max_output_tokens": 4000}),
        # Two spellings in the defaults are replaced together by one per-call value.
        (
            {"max_tokens": 1000, "max_output_tokens": 2000},
            {"max_tokens": 4000},
            {"max_output_tokens": 4000},
        ),
        # Output format: response_format (json_mode) vs text.format.
        (
            {"response_format": {"type": "json_object"}},
            {"text": {"format": _SCHEMA_FORMAT}},
            {"text": {"format": _SCHEMA_FORMAT}},
        ),
        (
            {"text": {"format": {"type": "json_object"}, "verbosity": "low"}},
            {"response_format": {"type": "text"}},
            {"text": {"format": {"type": "text"}, "verbosity": "low"}},
        ),
        (
            {"text": {"format": {"type": "json_object"}, "verbosity": "low"}},
            {"text": {"format": _SCHEMA_FORMAT}},
            {"text": {"format": _SCHEMA_FORMAT, "verbosity": "low"}},
        ),
        # A per-call text without a format keeps the default format.
        (
            {"response_format": {"type": "json_object"}},
            {"text": {"verbosity": "high"}},
            {"text": {"verbosity": "high", "format": {"type": "json_object"}}},
        ),
        # Reasoning effort: reasoning_effort vs reasoning.effort.
        (
            {"reasoning": {"effort": "high", "summary": "auto"}},
            {"reasoning_effort": "low"},
            {"reasoning": {"summary": "auto", "effort": "low"}},
        ),
        (
            {"reasoning_effort": "high"},
            {"reasoning": {"effort": "low"}},
            {"reasoning": {"effort": "low"}},
        ),
        (
            {"reasoning": {"summary": "auto"}},
            {"reasoning": {"effort": "low"}},
            {"reasoning": {"summary": "auto", "effort": "low"}},
        ),
        # Unaliased keys: plain per-call replacement, as before.
        ({"top_p": 0.1, "user": "a"}, {"top_p": 0.9}, {"top_p": 0.9, "user": "a"}),
        ({"logprobs": True}, {"logprobs": False}, {}),
        (
            {"include": ["reasoning.encrypted_content"]},
            {"logprobs": True},
            {"include": ["reasoning.encrypted_content", "message.output_text.logprobs"]},
        ),
    ],
)
def test_per_call_alias_replaces_default(
    defaults: dict[str, Any], override: dict[str, Any], expected: dict[str, Any]
) -> None:
    snapshot = (json.dumps(defaults), json.dumps(override))
    assert translate_request_config(merge_request_configs(defaults, override)) == expected
    assert (json.dumps(defaults), json.dumps(override)) == snapshot  # never mutated


@pytest.mark.parametrize(
    ("defaults", "override"),
    [
        ({}, {"max_tokens": 5, "max_output_tokens": 6}),
        ({"max_output_tokens": 1}, {"max_tokens": 5, "max_completion_tokens": 6}),
        ({"max_tokens": 5, "max_output_tokens": 6}, {"top_p": 0.5}),
        ({}, {"response_format": {"type": "json_object"}, "text": {"format": {"type": "text"}}}),
        ({}, {"reasoning_effort": "low", "reasoning": {"effort": "high"}}),
    ],
)
def test_conflict_within_one_source_still_raises(
    defaults: dict[str, Any], override: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="same config; pass only one") as caught:
        translate_request_config(merge_request_configs(defaults, override))
    assert "chat_completions" not in str(caught.value)


# ── Models (mocked clients) ──────────────────────────────────────────────────


def _stop_client() -> MagicMock:
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=RuntimeError("stop-before-network"))
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop-before-network"))
    return client


async def _sent(model: Any, config: dict[str, Any], *, chat: bool = False) -> dict[str, Any]:
    with pytest.raises(RuntimeError, match="stop-before-network"):
        await model.generate("hi", config=config)
    create = model._client.chat.completions.create if chat else model._client.responses.create
    return dict(create.call_args.kwargs)


async def test_openai_per_call_max_tokens_beats_default_max_output_tokens() -> None:
    model = OpenAIModel("gpt", _stop_client(), extra_body={"max_output_tokens": 1000})
    assert (await _sent(model, {"max_tokens": 4000}))["max_output_tokens"] == 4000


async def test_openai_per_call_text_format_beats_json_mode_default() -> None:
    model = OpenAIModel(
        "gpt",
        _stop_client(),
        # What from_api_key(json_mode=True) puts in extra_body.
        extra_body={"response_format": {"type": "json_object"}, "text": {"verbosity": "low"}},
    )
    sent = await _sent(model, {"text": {"format": _SCHEMA_FORMAT}})
    assert sent["text"] == {"format": _SCHEMA_FORMAT, "verbosity": "low"}


@pytest.mark.parametrize("alias", ["max_tokens", "max_completion_tokens"])
async def test_deepseek_per_call_token_alias_beats_default_max_output_tokens(alias: str) -> None:
    model = DeepSeekModel(
        "deepseek-v4-flash",
        _stop_client(),
        api_surface="responses",
        extra_body={"max_output_tokens": 1000},
    )
    sent = await _sent(model, {alias: 4000})
    assert sent["max_output_tokens"] == 4000
    assert alias not in sent.get("extra_body", {})


class _DeepSeekRaiseBound(DeepSeekStrategy):
    def __init__(self, model: Any, alias: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.alias = alias

    def attempt_generation_config(
        self, attempt: int, state: RetryState | None
    ) -> dict[str, Any] | None:
        return {self.alias: 4000} if attempt > 1 else None


@pytest.mark.parametrize("alias", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("where", ["model-default", "strategy-config"])
async def test_deepseek_attempt_token_alias_beats_max_output_tokens(alias: str, where: str) -> None:
    bound = {"max_output_tokens": 1000}
    model = DeepSeekModel(
        "deepseek-v4-flash",
        _stop_client(),
        api_surface="responses",
        extra_body=bound if where == "model-default" else None,
    )
    strategy = _DeepSeekRaiseBound(
        model, alias, generation_config=bound if where == "strategy-config" else None
    )
    with pytest.raises(RuntimeError, match="stop-before-network"):
        await strategy.execute("hi", attempt=2, timeout=10.0)
    sent = model._client.responses.create.call_args.kwargs
    assert sent["max_output_tokens"] == 4000
    assert alias not in sent.get("extra_body", {})


async def test_deepseek_per_call_text_keeps_the_schema_format() -> None:
    model = DeepSeekModel(
        "deepseek-v4-flash",
        _stop_client(),
        api_surface="responses",
        response_schema={"type": "object", "properties": {}},
    )
    sent = await _sent(model, {"text": {"verbosity": "low"}})
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["verbosity"] == "low"


async def test_chat_completions_surface_is_unchanged() -> None:
    model = OpenAIModel(
        "gpt", _stop_client(), api_surface="chat_completions", extra_body={"max_tokens": 1000}
    )
    sent = await _sent(model, {"max_completion_tokens": 4000}, chat=True)
    assert sent["extra_body"] == {"max_tokens": 1000, "max_completion_tokens": 4000}
    # The strategy-level merge stays shallow on Chat Completions too.
    merged = model._merge_call_config({"max_tokens": 1}, {"max_completion_tokens": 2})
    assert merged == {"max_tokens": 1, "max_completion_tokens": 2}


# ── End to end: the documented truncation remedy ─────────────────────────────


def _response_body(*, truncated: bool) -> dict[str, Any]:
    output = (
        []
        if truncated
        else [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "full answer", "annotations": []}],
            }
        ]
    )
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test",
        "status": "incomplete" if truncated else "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    if truncated:
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


class _RaiseBoundOnRetry(OpenAIStrategy):
    def attempt_generation_config(
        self, attempt: int, state: RetryState | None
    ) -> dict[str, Any] | None:
        return {"max_tokens": 4000} if attempt > 1 else None


class _RetryTruncations(OpenAIErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, TruncatedResponseError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="response_truncated",
            )
        return super().classify(exception)


@pytest.mark.parametrize(
    "generation_config",
    [None, {"max_output_tokens": 1000}],
    ids=["model-default", "strategy-config"],
)
async def test_truncation_retry_raises_the_bound_over_a_max_output_tokens_default(
    fast_retry: RetryConfig, generation_config: dict[str, Any] | None
) -> None:
    from openai import AsyncOpenAI

    httpx = openai_http()
    sent: list[dict[str, Any]] = []

    def handle(request: Any) -> Any:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_response_body(truncated=len(sent) == 1))

    client = AsyncOpenAI(
        api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    model = OpenAIModel(
        "gpt-test",
        client,
        extra_body=None if generation_config else {"max_output_tokens": 1000},
    )
    strategy = _RaiseBoundOnRetry(model, generation_config=generation_config)
    config = ProcessorConfig(max_workers=1, retry=fast_retry)
    async with ParallelBatchProcessor[None, str, None](
        config=config, error_classifier=_RetryTruncations()
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="go"))
        result = await processor.process_all()

    item = result.results[0]
    assert item.success, item.error
    assert item.output == "full answer"
    assert [body.get("max_output_tokens") for body in sent] == [1000, 4000]
    await client.close()
