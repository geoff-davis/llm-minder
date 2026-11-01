"""Normalized EmptyResponseError.reason across the built-in models (issue #273)."""

from __future__ import annotations

import copy
import pickle
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from async_batch_llm import (
    DeepSeekModel,
    DefaultErrorClassifier,
    EmptyResponseError,
    ErrorCategory,
    ErrorInfo,
    GeminiCachedModel,
    GeminiErrorClassifier,
    GeminiModel,
    LLMResponse,
    LLMWorkItem,
    ModelStrategy,
    OpenAIErrorClassifier,
    OpenAIModel,
    OpenRouterErrorClassifier,
    OpenRouterModel,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIErrorClassifier,
    RetryConfig,
    RetryState,
    TruncatedResponseError,
)

_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


# ── The exception types ──────────────────────────────────────────────────────


def test_defaults_keep_the_old_constructor_working():
    exc = EmptyResponseError("empty")
    assert isinstance(exc, ValueError)
    assert exc.reason == "unknown"
    assert exc.provider_reason is None
    assert str(exc) == "empty"


def test_truncated_response_is_an_empty_response_with_reason_max_tokens():
    exc = TruncatedResponseError("cut off", token_usage=_USAGE, provider_reason="length")
    assert isinstance(exc, EmptyResponseError)
    assert isinstance(exc, ValueError)
    assert exc.reason == "max_tokens"
    assert exc.provider_reason == "length"
    assert exc._failed_token_usage == _USAGE


@pytest.mark.parametrize(
    "exc",
    [
        EmptyResponseError(
            "blocked", token_usage=_USAGE, reason="content_filter", provider_reason="SAFETY"
        ),
        TruncatedResponseError("cut off", token_usage=_USAGE, provider_reason="MAX_TOKENS"),
    ],
    ids=["empty", "truncated"],
)
def test_reason_survives_copy_and_pickle(exc):
    for clone in (copy.copy(exc), copy.deepcopy(exc), pickle.loads(pickle.dumps(exc))):
        assert type(clone) is type(exc)
        assert str(clone) == str(exc)
        assert (clone.reason, clone.provider_reason) == (exc.reason, exc.provider_reason)
        assert clone._failed_token_usage == _USAGE


@pytest.mark.parametrize(
    "classifier",
    [
        DefaultErrorClassifier(),
        GeminiErrorClassifier(),
        OpenAIErrorClassifier(),
        OpenRouterErrorClassifier(),
        PydanticAIErrorClassifier(),
    ],
    ids=lambda c: type(c).__name__,
)
def test_truncation_has_its_own_non_retryable_category(classifier):
    info = classifier.classify(TruncatedResponseError("cut off"))
    assert info.error_category == ErrorCategory.RESPONSE_TRUNCATED == "response_truncated"
    assert not info.is_retryable
    assert not info.is_rate_limit
    other = classifier.classify(EmptyResponseError("blocked", reason="content_filter"))
    assert other.error_category == "empty_response"
    assert not other.is_retryable


# ── Gemini ───────────────────────────────────────────────────────────────────


class _AsyncIterList:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> _AsyncIterList:
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _gemini_client(response: Any) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    client.aio.caches.list = AsyncMock(return_value=_AsyncIterList([]))
    cache = MagicMock()
    cache.name = "test-cache"
    cache.expire_time = None
    cache.create_time = None
    client.aio.caches.create = AsyncMock(return_value=cache)
    return client


@pytest.fixture(params=["plain", "cached"])
def gemini_generate(request):
    """Run one generate() on GeminiModel or GeminiCachedModel against a canned response."""

    async def run(response: Any) -> LLMResponse:
        client = _gemini_client(response)
        if request.param == "plain":
            return await GeminiModel("gemini-test", client).generate("q")
        model = GeminiCachedModel(model="gemini-test", client=client, cached_content=[])
        await model.prepare()
        return await model.generate("q")

    return run


def _gemini_usage(types: Any) -> Any:
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10, candidates_token_count=0, total_token_count=10
    )


@pytest.mark.parametrize(
    ("finish_reason", "reason"),
    [
        ("MAX_TOKENS", "max_tokens"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "tool_call"),
        ("STOP", "no_content"),
        ("OTHER", "unknown"),
    ],
)
async def test_gemini_finish_reasons(gemini_generate, finish_reason, reason):
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(finish_reason=finish_reason)],
        usage_metadata=_gemini_usage(types),
    )
    with pytest.raises(EmptyResponseError, match=finish_reason) as caught:
        await gemini_generate(response)
    assert caught.value.reason == reason
    assert caught.value.provider_reason == finish_reason
    assert isinstance(caught.value, TruncatedResponseError) == (reason == "max_tokens")


async def test_gemini_blocked_prompt_uses_the_block_reason(gemini_generate):
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"),
        usage_metadata=_gemini_usage(types),
    )
    with pytest.raises(EmptyResponseError, match="finish_reason=unknown") as caught:
        await gemini_generate(response)
    assert caught.value.reason == "content_filter"
    assert caught.value.provider_reason == "SAFETY"


async def test_gemini_without_any_reason_is_unknown(gemini_generate):
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(usage_metadata=_gemini_usage(types))
    with pytest.raises(EmptyResponseError) as caught:
        await gemini_generate(response)
    assert caught.value.reason == "unknown"
    assert caught.value.provider_reason is None


def _function_call_candidate(types: Any, finish_reason: str) -> Any:
    part = types.Part(function_call=types.FunctionCall(name="lookup", args={}))
    return types.Candidate(
        finish_reason=finish_reason, content=types.Content(role="model", parts=[part])
    )


async def test_gemini_function_call_reply_is_tool_call(gemini_generate):
    # Gemini ends a function call with a plain STOP; the parts tell it apart.
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        candidates=[_function_call_candidate(types, "STOP")],
        usage_metadata=_gemini_usage(types),
    )
    with pytest.raises(EmptyResponseError, match="finish_reason=STOP") as caught:
        await gemini_generate(response)
    assert caught.value.reason == "tool_call"
    assert caught.value.provider_reason == "STOP"


@pytest.mark.parametrize(
    ("finish_reason", "reason"), [("MAX_TOKENS", "max_tokens"), ("SAFETY", "content_filter")]
)
async def test_gemini_truncation_and_blocks_win_over_a_function_call_part(
    gemini_generate, finish_reason, reason
):
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        candidates=[_function_call_candidate(types, finish_reason)],
        usage_metadata=_gemini_usage(types),
    )
    with pytest.raises(EmptyResponseError) as caught:
        await gemini_generate(response)
    assert caught.value.reason == reason
    assert caught.value.provider_reason == finish_reason


def _empty_text_candidate(types: Any, finish_reason: str) -> Any:
    return types.Candidate(
        finish_reason=finish_reason,
        content=types.Content(role="model", parts=[types.Part(text="")]),
    )


@pytest.mark.parametrize(
    ("finish_reason", "reason"), [("MAX_TOKENS", "max_tokens"), ("SAFETY", "content_filter")]
)
async def test_gemini_empty_string_fails_when_cut_off_or_blocked(
    gemini_generate, finish_reason, reason
):
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        candidates=[_empty_text_candidate(types, finish_reason)],
        usage_metadata=_gemini_usage(types),
    )
    assert response.text == ""
    with pytest.raises(EmptyResponseError) as caught:
        await gemini_generate(response)
    assert caught.value.reason == reason


async def test_gemini_empty_string_after_a_normal_stop_is_a_valid_answer(gemini_generate):
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        candidates=[_empty_text_candidate(types, "STOP")],
        usage_metadata=_gemini_usage(types),
    )
    assert (await gemini_generate(response)).text == ""


# ── Chat Completions (OpenAI, OpenRouter, DeepSeek) ──────────────────────────


def _chat_completion(*, finish_reason: str, message: dict[str, Any] | None) -> Any:
    ChatCompletion = pytest.importorskip("openai.types.chat").ChatCompletion
    choices = (
        []
        if message is None
        else [{"index": 0, "finish_reason": finish_reason, "message": message}]
    )
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": choices,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def _chat_client(response: Any) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


_CHAT_MODELS = [
    pytest.param(
        lambda c: OpenAIModel("gpt-test", c, api_surface="chat_completions"), id="openai-chat"
    ),
    pytest.param(lambda c: OpenRouterModel("vendor/model", c), id="openrouter"),
    pytest.param(lambda c: DeepSeekModel("deepseek-test", c), id="deepseek-chat"),
]
_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "lookup", "arguments": "{}"},
}


@pytest.mark.parametrize("model_factory", _CHAT_MODELS)
@pytest.mark.parametrize(
    ("finish_reason", "message", "reason"),
    [
        ("length", {"role": "assistant", "content": None}, "max_tokens"),
        ("content_filter", {"role": "assistant", "content": None}, "content_filter"),
        (
            "tool_calls",
            {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL]},
            "tool_call",
        ),
        ("stop", {"role": "assistant", "content": None, "refusal": "I can't."}, "refusal"),
        ("stop", {"role": "assistant", "content": None}, "no_content"),
    ],
    ids=["length", "content_filter", "tool_calls", "refusal", "stop"],
)
async def test_chat_finish_reasons(model_factory, finish_reason, message, reason):
    model = model_factory(
        _chat_client(_chat_completion(finish_reason=finish_reason, message=message))
    )
    with pytest.raises(EmptyResponseError, match=f"finish_reason='{finish_reason}'") as caught:
        await model.generate("hello")
    assert caught.value.reason == reason
    assert caught.value.provider_reason == finish_reason
    assert isinstance(caught.value, TruncatedResponseError) == (reason == "max_tokens")
    assert caught.value._failed_token_usage["output_tokens"] == 5


@pytest.mark.parametrize("model_factory", _CHAT_MODELS)
@pytest.mark.parametrize(
    ("finish_reason", "message", "reason"),
    [
        ("length", {"role": "assistant", "content": ""}, "max_tokens"),
        ("content_filter", {"role": "assistant", "content": ""}, "content_filter"),
        ("stop", {"role": "assistant", "content": "", "refusal": "I can't."}, "refusal"),
    ],
    ids=["length", "content_filter", "refusal"],
)
async def test_chat_empty_string_fails_when_cut_off_blocked_or_refused(
    model_factory, finish_reason, message, reason
):
    model = model_factory(
        _chat_client(_chat_completion(finish_reason=finish_reason, message=message))
    )
    with pytest.raises(EmptyResponseError) as caught:
        await model.generate("hello")
    assert caught.value.reason == reason
    assert caught.value.provider_reason == finish_reason
    assert caught.value._failed_token_usage["output_tokens"] == 5


@pytest.mark.parametrize("model_factory", _CHAT_MODELS)
@pytest.mark.parametrize(
    ("finish_reason", "message"),
    [
        ("stop", {"role": "assistant", "content": ""}),
        ("tool_calls", {"role": "assistant", "content": "", "tool_calls": [_TOOL_CALL]}),
    ],
    ids=["stop", "tool_calls"],
)
async def test_chat_empty_string_is_still_a_valid_answer_otherwise(
    model_factory, finish_reason, message
):
    model = model_factory(
        _chat_client(_chat_completion(finish_reason=finish_reason, message=message))
    )
    response = await model.generate("hello")
    assert response.text == ""


@pytest.mark.parametrize("model_factory", _CHAT_MODELS)
async def test_chat_without_choices_is_unknown(model_factory):
    model = model_factory(_chat_client(_chat_completion(finish_reason="stop", message=None)))
    with pytest.raises(EmptyResponseError, match="No choices") as caught:
        await model.generate("hello")
    assert caught.value.reason == "unknown"
    assert caught.value.provider_reason is None


# ── Responses API (OpenAI, DeepSeek) ─────────────────────────────────────────


def _responses_response(
    *, status: str, output: list[dict[str, Any]], incomplete: str | None = None
) -> Any:
    Response = pytest.importorskip("openai.types.responses").Response
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "test",
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    if incomplete is not None:
        body["incomplete_details"] = {"reason": incomplete}
    return Response.model_validate(body)


def _responses_client(response: Any) -> MagicMock:
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=response)
    return client


_RESPONSES_MODELS = [
    pytest.param(lambda c: OpenAIModel("gpt-test", c), id="openai-responses"),
    pytest.param(
        lambda c: DeepSeekModel("deepseek-test", c, api_surface="responses"),
        id="deepseek-responses",
    ),
]
_REFUSAL = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "refusal", "refusal": "I can't help with that."}],
}


@pytest.mark.parametrize("model_factory", _RESPONSES_MODELS)
@pytest.mark.parametrize(
    ("incomplete", "reason"),
    [("max_output_tokens", "max_tokens"), ("content_filter", "content_filter")],
)
async def test_responses_incomplete_reasons(model_factory, incomplete, reason):
    response = _responses_response(status="incomplete", output=[], incomplete=incomplete)
    model = model_factory(_responses_client(response))
    with pytest.raises(EmptyResponseError, match=f"reason={incomplete}") as caught:
        await model.generate("hello")
    assert caught.value.reason == reason
    assert caught.value.provider_reason == incomplete
    assert isinstance(caught.value, TruncatedResponseError) == (reason == "max_tokens")


@pytest.mark.parametrize("model_factory", _RESPONSES_MODELS)
async def test_responses_completed_without_text_is_no_content(model_factory):
    model = model_factory(_responses_client(_responses_response(status="completed", output=[])))
    with pytest.raises(EmptyResponseError, match="No text returned") as caught:
        await model.generate("hello")
    assert caught.value.reason == "no_content"
    assert caught.value.provider_reason == "completed"


async def test_openai_responses_refusal():
    response = _responses_response(status="completed", output=[_REFUSAL])
    model = OpenAIModel("gpt-test", _responses_client(response))
    with pytest.raises(EmptyResponseError, match="refused") as caught:
        await model.generate("hello")
    assert caught.value.reason == "refusal"
    assert caught.value.provider_reason == "completed"


# ── What the reason enables: a growing output bound without execute() ──────


class _RetryTruncations(DefaultErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, TruncatedResponseError):
            return ErrorInfo(True, False, False, "response_truncated")
        return super().classify(exception)


class _GrowingBound(ModelStrategy[str]):
    def attempt_generation_config(
        self, attempt: int, state: RetryState | None
    ) -> dict[str, Any] | None:
        return {"max_tokens": 1024 * 2 ** (attempt - 1)}


async def test_a_classifier_and_attempt_config_retry_truncation_with_a_larger_bound():
    async def generate(prompt: str, **kwargs: Any) -> LLMResponse:
        if kwargs["config"]["max_tokens"] < 4096:
            raise TruncatedResponseError("cut off", token_usage=_USAGE, provider_reason="length")
        return LLMResponse(text="done", input_tokens=10, output_tokens=5, total_tokens=15)

    model = AsyncMock()
    model.generate.side_effect = generate
    config = ProcessorConfig(
        max_workers=1,
        retry=RetryConfig(max_attempts=4, initial_wait=0.01, jitter=False),
    )
    async with ParallelBatchProcessor[str, Any, None](
        config=config, error_classifier=_RetryTruncations()
    ) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="x", strategy=_GrowingBound(model), prompt="p")
        )
        item = (await processor.process_all()).results[0]

    assert item.success and item.output == "done"
    bounds = [call.kwargs["config"]["max_tokens"] for call in model.generate.await_args_list]
    assert bounds == [1024, 2048, 4096]
    assert item.token_usage["output_tokens"] == 15
