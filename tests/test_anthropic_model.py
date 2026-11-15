"""Tests for AnthropicModel, AnthropicStrategy and llm("anthropic:...") (#314).

No live API calls: the model gets a mocked client, or a real ``AsyncAnthropic`` whose
HTTP transport is an ``httpx2.MockTransport``.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from llm_minder import (  # noqa: E402
    AnthropicErrorClassifier,
    AnthropicModel,
    AnthropicStrategy,
    CachedTokenRates,
    EmptyResponseError,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    TruncatedResponseError,
    call_result,
    llm,
)
from llm_minder.models import _sdk  # noqa: E402

MODEL = "claude-opus-5-5"


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _usage(input_tokens=10, output_tokens=5, cache_read=None, cache_write=None) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def _message(content=None, *, stop_reason="end_turn", usage=None, **extra) -> SimpleNamespace:
    message = SimpleNamespace(
        content=[_text("hello")] if content is None else content,
        stop_reason=stop_reason,
        model=MODEL,
        usage=_usage() if usage is None else usage,
        stop_details=None,
    )
    for name, value in extra.items():
        setattr(message, name, value)
    return message


def _client(response=None) -> SimpleNamespace:
    create = AsyncMock(return_value=_message() if response is None else response)
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def _sent(client) -> dict:
    return client.messages.create.await_args.kwargs


# ── Request mapping ──────────────────────────────────────────────────────────


class TestRequest:
    async def test_string_prompt_is_one_user_message_with_default_max_tokens(self):
        client = _client()
        await AnthropicModel(MODEL, client).generate("Hi")
        assert _sent(client) == {
            "model": MODEL,
            "max_tokens": 16_000,
            "messages": [{"role": "user", "content": "Hi"}],
        }

    async def test_list_prompt_raises_type_error_pointing_at_315(self):
        client = _client()
        with pytest.raises(TypeError, match="315"):
            await AnthropicModel(MODEL, client).generate([{"role": "user", "content": "x"}])
        client.messages.create.assert_not_awaited()

    async def test_system_instruction_default_and_per_call_override(self):
        client = _client()
        model = AnthropicModel(MODEL, client, system_instruction="default")
        await model.generate("x")
        assert _sent(client)["system"] == "default"
        await model.generate("x", system_instruction="override")
        assert _sent(client)["system"] == "override"

    async def test_temperature_sent_only_when_set(self):
        client = _client()
        model = AnthropicModel(MODEL, client)
        await model.generate("x")
        assert "temperature" not in _sent(client)
        await model.generate("x", temperature=0.3)
        assert _sent(client)["temperature"] == 0.3

    async def test_cache_control_is_top_level(self):
        client = _client()
        await AnthropicModel(MODEL, client, cache_control={"type": "ephemeral"}).generate("x")
        assert _sent(client)["cache_control"] == {"type": "ephemeral"}

    async def test_extra_body_and_config_merge_into_request(self):
        client = _client()
        model = AnthropicModel(
            MODEL,
            client,
            max_tokens=100,
            extra_body={"output_config": {"effort": "low"}, "thinking": {"type": "disabled"}},
        )
        await model.generate("x", config={"output_config": {"effort": "high"}, "max_tokens": 500})
        sent = _sent(client)
        assert sent["output_config"] == {"effort": "high"}
        assert sent["thinking"] == {"type": "disabled"}
        assert sent["max_tokens"] == 500

    async def test_extra_headers_forwarded(self):
        client = _client()
        await AnthropicModel(MODEL, client, extra_headers={"anthropic-beta": "b"}).generate("x")
        assert _sent(client)["extra_headers"] == {"anthropic-beta": "b"}

    async def test_unnamed_fields_go_to_sdk_extra_body(self):
        # A client whose create() doesn't name a field (an older SDK, a platform
        # client) gets it through the SDK's extra_body instead of a TypeError.
        seen = []

        async def create(*, model, max_tokens, messages, extra_body=None):
            seen.append({"max_tokens": max_tokens, "extra_body": extra_body})
            return _message()

        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        model = AnthropicModel(MODEL, client, extra_body={"future_field": 1})
        await model.generate("x", system_instruction="s", config={"extra_body": {"other": 2}})
        assert seen == [
            {"max_tokens": 16_000, "extra_body": {"other": 2, "system": "s", "future_field": 1}}
        ]

    async def test_cache_group_label_is_not_sent(self):
        client = _client()
        model = AnthropicModel(MODEL, client, cache_group="model-label")
        response = await model.generate("x", config={"cache_group": "call-label"})
        assert "cache_group" not in _sent(client)
        assert response.metadata["cache_group"] == "call-label"
        response = await model.generate("x")
        assert response.metadata["cache_group"] == "model-label"

    async def test_cache_group_defaults_to_system_hash(self):
        response = await AnthropicModel(MODEL, _client(), system_instruction="sys").generate("x")
        assert response.metadata["cache_group"].startswith("si:")

    @pytest.mark.parametrize("value", [0, -1, True, 1.5])
    def test_max_tokens_must_be_positive_int(self, value):
        with pytest.raises(ValueError, match="max_tokens"):
            AnthropicModel(MODEL, _client(), max_tokens=value)

    def test_cache_control_must_be_a_dict(self):
        with pytest.raises(TypeError, match="cache_control"):
            AnthropicModel(MODEL, _client(), cache_control="ephemeral")  # type: ignore[arg-type]


# ── Response mapping ─────────────────────────────────────────────────────────


class TestResponse:
    async def test_text_blocks_are_concatenated(self):
        client = _client(_message([_text("a"), SimpleNamespace(type="thinking"), _text("b")]))
        response = await AnthropicModel(MODEL, client).generate("x")
        assert response.text == "ab"

    async def test_metadata(self):
        client = _client(_message(_request_id="req_1"))
        response = await AnthropicModel(MODEL, client).generate("x")
        assert response.metadata == {
            "provider": "anthropic",
            "finish_reason": "end_turn",
            "model": MODEL,
            "request_id": "req_1",
        }

    async def test_metadata_is_json_serializable_with_mock_attributes(self):
        message = MagicMock()
        message.content = [_text("ok")]
        message.usage = _usage()
        response = await AnthropicModel(MODEL, _client(message)).generate("x")
        json.dumps(response.metadata)
        assert response.metadata == {"provider": "anthropic"}

    async def test_reasoning_and_tool_calls_views(self):
        content = [
            SimpleNamespace(type="thinking", thinking="step 1", signature="s"),
            SimpleNamespace(type="thinking", thinking="", signature="s"),
            SimpleNamespace(type="redacted_thinking", data="x"),
            _text("Calling a tool."),
            SimpleNamespace(type="tool_use", id="tu_1", name="lookup", input={"q": "x"}),
            SimpleNamespace(type="tool_use", id="tu_2", name="", input={}),
        ]
        response = await AnthropicModel(MODEL, _client(_message(content))).generate("x")
        assert response.reasoning == "step 1"
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert (call.id, call.name, json.loads(call.arguments)) == ("tu_1", "lookup", {"q": "x"})
        json.dumps(response.metadata)

    async def test_text_with_max_tokens_succeeds(self):
        client = _client(_message([_text("partial")], stop_reason="max_tokens"))
        response = await AnthropicModel(MODEL, client).generate("x")
        assert response.text == "partial"
        assert response.metadata["finish_reason"] == "max_tokens"

    async def test_refusal_with_text_keeps_stop_details(self):
        details = SimpleNamespace(
            model_dump=lambda **_: {"type": "refusal", "category": "cyber", "explanation": None}
        )
        client = _client(_message([_text("I can't")], stop_reason="refusal", stop_details=details))
        response = await AnthropicModel(MODEL, client).generate("x")
        assert response.metadata["stop_details"] == {
            "type": "refusal",
            "category": "cyber",
            "explanation": None,
        }

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
    async def test_no_text_at_output_bound_is_truncated(self, stop_reason):
        client = _client(_message([], stop_reason=stop_reason, usage=_usage(7, 100)))
        with pytest.raises(TruncatedResponseError) as raised:
            await AnthropicModel(MODEL, client).generate("x")
        assert raised.value.reason == "max_tokens"
        assert raised.value.provider_reason == stop_reason
        assert raised.value._failed_token_usage["output_tokens"] == 100
        info = AnthropicErrorClassifier().classify(raised.value)
        assert (info.error_category, info.is_retryable) == ("response_truncated", False)

    async def test_empty_text_at_output_bound_is_truncated(self):
        client = _client(_message([_text("")], stop_reason="max_tokens"))
        with pytest.raises(TruncatedResponseError):
            await AnthropicModel(MODEL, client).generate("x")

    async def test_refusal_without_text_is_content_filter(self):
        details = SimpleNamespace(model_dump=lambda **_: {"type": "refusal", "category": "bio"})
        client = _client(_message([], stop_reason="refusal", stop_details=details))
        with pytest.raises(EmptyResponseError) as raised:
            await AnthropicModel(MODEL, client).generate("x")
        assert not isinstance(raised.value, TruncatedResponseError)
        assert raised.value.reason == "content_filter"
        assert raised.value.provider_reason == "refusal"
        assert "bio" in str(raised.value)
        info = AnthropicErrorClassifier().classify(raised.value)
        assert (info.error_category, info.is_retryable) == ("empty_response", False)

    @pytest.mark.parametrize(
        ("stop_reason", "reason"),
        [
            ("tool_use", "tool_call"),
            ("end_turn", "no_content"),
            ("pause_turn", "no_content"),
            ("something_new", "unknown"),
            (None, "unknown"),
        ],
    )
    async def test_other_textless_responses(self, stop_reason, reason):
        client = _client(_message([], stop_reason=stop_reason, usage=_usage(3, 4)))
        with pytest.raises(EmptyResponseError) as raised:
            await AnthropicModel(MODEL, client).generate("x")
        assert raised.value.reason == reason
        assert raised.value._failed_token_usage["input_tokens"] == 3

    async def test_empty_text_with_end_turn_is_a_valid_answer(self):
        response = await AnthropicModel(MODEL, _client(_message([_text("")]))).generate("x")
        assert response.text == ""


# ── Tokens ───────────────────────────────────────────────────────────────────


class TestTokens:
    async def test_cache_reads_and_writes_fold_into_input(self):
        usage = _usage(input_tokens=10, output_tokens=5, cache_read=100, cache_write=20)
        response = await AnthropicModel(MODEL, _client(_message(usage=usage))).generate("x")
        assert response.input_tokens == 130
        assert response.cached_input_tokens == 100
        assert response.cache_write_input_tokens == 20
        assert response.output_tokens == 5
        assert response.total_tokens == 135
        assert response.token_usage["cache_write_input_tokens"] == 20

    async def test_none_cache_fields_count_as_zero(self):
        response = await AnthropicModel(MODEL, _client()).generate("x")
        assert (response.input_tokens, response.cached_input_tokens) == (10, 0)
        assert response.cache_write_input_tokens == 0
        assert response.total_tokens == 15

    async def test_missing_usage_is_zero(self):
        message = SimpleNamespace(content=[_text("x")], stop_reason="end_turn")
        response = await AnthropicModel(MODEL, _client(message)).generate("x")
        assert (response.input_tokens, response.output_tokens, response.total_tokens) == (0, 0, 0)

    async def test_textless_failure_carries_cache_tokens(self):
        usage = _usage(input_tokens=2, output_tokens=0, cache_read=8, cache_write=4)
        client = _client(_message([], stop_reason="end_turn", usage=usage))
        with pytest.raises(EmptyResponseError) as raised:
            await AnthropicModel(MODEL, client).generate("x")
        assert raised.value._failed_token_usage == {
            "input_tokens": 14,
            "output_tokens": 0,
            "total_tokens": 14,
            "cached_input_tokens": 8,
            "cache_write_input_tokens": 4,
        }

    def test_anthropic_read_rate_constant(self):
        assert CachedTokenRates.ANTHROPIC_READ == 0.10


# ── Strategy ─────────────────────────────────────────────────────────────────


class TestStrategy:
    def test_recommends_anthropic_classifier(self):
        strategy = AnthropicStrategy(AnthropicModel(MODEL, _client()))
        assert isinstance(strategy.recommended_error_classifier(), AnthropicErrorClassifier)

    async def test_execute_returns_text_tokens_and_metadata(self):
        strategy = AnthropicStrategy(AnthropicModel(MODEL, _client()))
        output, tokens, metadata = await strategy.execute("x", attempt=1, timeout=10)
        assert output == "hello"
        assert tokens["input_tokens"] == 10
        assert metadata["provider"] == "anthropic"

    async def test_attempt_generation_config_reaches_request(self):
        class Escalating(AnthropicStrategy[str]):
            def attempt_generation_config(self, attempt, state):
                return {"output_config": {"effort": "high"}} if attempt > 1 else None

        client = _client()
        strategy = Escalating(
            AnthropicModel(MODEL, client), generation_config={"output_config": {"effort": "low"}}
        )
        await strategy.execute("x", attempt=1, timeout=10)
        assert _sent(client)["output_config"] == {"effort": "low"}
        await strategy.execute("x", attempt=2, timeout=10)
        assert _sent(client)["output_config"] == {"effort": "high"}

    async def test_parser_failure_attaches_billed_tokens(self):
        def parser(response):
            raise ValueError("bad output")

        usage = _usage(input_tokens=4, output_tokens=6, cache_read=10)
        strategy = AnthropicStrategy(
            AnthropicModel(MODEL, _client(_message(usage=usage))), response_parser=parser
        )
        with pytest.raises(ValueError) as raised:
            await strategy.execute("x", attempt=1, timeout=10)
        assert raised.value._failed_token_usage["input_tokens"] == 14
        assert raised.value._failed_token_usage["cached_input_tokens"] == 10

    async def test_failed_attempt_tokens_reach_the_result(self):
        client = _client(_message([], stop_reason="max_tokens", usage=_usage(5, 50)))
        result = await call_result(AnthropicStrategy(AnthropicModel(MODEL, client)), "x")
        assert not result.success
        assert result.error_category == "response_truncated"
        assert result.token_usage["output_tokens"] == 50

    async def test_529_triggers_shared_cooldown_then_succeeds(self, fast_retry, fast_rate_limit):
        overloaded = _status_error(anthropic.OverloadedError, 529, "overloaded_error")
        create = AsyncMock(side_effect=[overloaded, _message()])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        strategy = AnthropicStrategy(AnthropicModel(MODEL, client))
        config = ProcessorConfig(max_workers=1, retry=fast_retry, rate_limit=fast_rate_limit)
        async with ParallelBatchProcessor[None, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="x"))
            result = await processor.process_all()
            stats = await processor.get_stats()
        assert result.succeeded == 1
        assert stats["rate_limit_count"] == 1
        assert create.await_count == 2


# ── Client ownership ─────────────────────────────────────────────────────────


class TestFromApiKey:
    def test_builds_first_party_client_with_no_sdk_retries(self):
        model = AnthropicModel.from_api_key(MODEL, api_key="sk-ant-test")
        assert isinstance(model._client, anthropic.AsyncAnthropic)
        assert model._client.max_retries == 0
        assert model._owns_client is True
        assert model.max_concurrency is None

    def test_max_retries_can_be_overridden(self):
        model = AnthropicModel.from_api_key(MODEL, api_key="k", max_retries=3)
        assert model._client.max_retries == 3

    def test_api_key_none_lets_sdk_resolve_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        model = AnthropicModel.from_api_key(MODEL)
        assert model._client.api_key == "sk-ant-env"

    def test_base_url_and_model_options_forwarded(self):
        model = AnthropicModel.from_api_key(
            MODEL,
            api_key="k",
            base_url="https://proxy.example.com",
            max_tokens=123,
            cache_control={"type": "ephemeral"},
            system_instruction="s",
        )
        assert model._client.base_url.host == "proxy.example.com"
        assert (model._max_tokens, model._cache_control) == (123, {"type": "ephemeral"})

    def test_direct_constructor_does_not_own_client(self):
        model = AnthropicModel(MODEL, anthropic.AsyncAnthropic(api_key="k"))
        assert model._owns_client is False

    def test_max_connections_sizes_pool(self):
        model = AnthropicModel.from_api_key(MODEL, api_key="k", max_connections=7)
        assert model.max_concurrency == 7
        assert AnthropicStrategy(model).max_concurrency == 7

    def test_max_connections_and_http_client_conflict(self):
        with pytest.raises(ValueError, match="max_connections or http_client"):
            AnthropicModel.from_api_key(
                MODEL,
                api_key="k",
                max_connections=2,
                http_client=anthropic.DefaultAsyncHttpxClient(),
            )

    def test_max_connections_must_be_positive(self):
        with pytest.raises(ValueError, match="max_connections"):
            AnthropicModel.from_api_key(MODEL, api_key="k", max_connections=0)

    def test_http_client_uses_sdk_transport_limits(self):
        client = _sdk._build_anthropic_http_client(5)
        assert isinstance(client, httpx2.AsyncClient)

    async def test_cleanup_closes_owned_client_and_prepare_reopens(self, monkeypatch):
        built = []

        def construct(**kwargs):
            client = MagicMock()
            client.close = AsyncMock()
            client.messages.create = AsyncMock(return_value=_message())
            client.kwargs = kwargs
            built.append(client)
            return client

        monkeypatch.setattr(_sdk, "AsyncAnthropic", construct)
        model = AnthropicModel.from_api_key(MODEL, api_key="k")
        await model.generate("x")
        await model.cleanup()
        built[0].close.assert_awaited_once()
        await model.cleanup()  # idempotent
        built[0].close.assert_awaited_once()
        await model.generate("x")
        assert len(built) == 2
        assert built[1].kwargs == {"api_key": "k", "max_retries": 0}

    async def test_cleanup_skips_caller_client(self):
        client = MagicMock()
        client.close = AsyncMock()
        await AnthropicModel(MODEL, client).cleanup()
        client.close.assert_not_awaited()

    async def test_cleanup_keeps_caller_http_client_open(self):
        transport = anthropic.DefaultAsyncHttpxClient()
        try:
            model = AnthropicModel.from_api_key(MODEL, api_key="k", http_client=transport)
            await model.cleanup()
            assert not transport.is_closed
        finally:
            await transport.aclose()

    async def test_request_concurrency_resizes_default_pool(self, monkeypatch):
        built = []

        def construct(**kwargs):
            client = MagicMock()
            client.close = AsyncMock()
            client.kwargs = kwargs
            built.append(client)
            return client

        monkeypatch.setattr(_sdk, "AsyncAnthropic", construct)
        model = AnthropicModel.from_api_key(MODEL, api_key="k")
        assert await model.request_concurrency(12) is True
        assert model.max_concurrency == 12
        assert "http_client" in built[1].kwargs
        built[0].close.assert_awaited_once()
        sized = AnthropicModel.from_api_key(MODEL, api_key="k", max_connections=3)
        assert await sized.request_concurrency(12) is False
        assert await AnthropicModel(MODEL, MagicMock()).request_concurrency(12) is False

    def test_failed_construction_closes_new_client(self, monkeypatch):
        closed = []
        client = SimpleNamespace(close=lambda: closed.append(True))
        monkeypatch.setattr(_sdk, "AsyncAnthropic", lambda **_: client)
        with pytest.raises(ValueError, match="max_tokens"):
            AnthropicModel.from_api_key(MODEL, api_key="k", max_tokens=0)
        assert closed == [True]


# ── Factory ──────────────────────────────────────────────────────────────────


class TestFactory:
    def test_anthropic_prefix(self):
        strategy = llm(f"anthropic:{MODEL}", api_key="k", max_tokens=200, max_connections=4)
        assert isinstance(strategy, AnthropicStrategy)
        assert isinstance(strategy.model, AnthropicModel)
        assert strategy.model._model == MODEL
        assert strategy.model._max_tokens == 200
        assert strategy.model.max_concurrency == 4
        assert strategy.model._client.max_retries == 0

    def test_sdk_kwargs_accepted_and_typos_rejected(self):
        llm(f"anthropic:{MODEL}", api_key="k", timeout=30.0)
        with pytest.raises(TypeError, match="Unknown anthropic model kwargs: typo"):
            llm(f"anthropic:{MODEL}", api_key="k", typo=1)

    def test_missing_sdk_names_extra(self, monkeypatch):
        monkeypatch.setattr(_sdk, "AsyncAnthropic", None)
        with pytest.raises(ImportError, match=r"llm-minder\[anthropic\]"):
            llm(f"anthropic:{MODEL}", api_key="k")


# ── Real SDK client over a mock transport ────────────────────────────────────


def _message_json(**overrides) -> dict:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {"type": "thinking", "thinking": "consider", "signature": "sig"},
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {"q": 1}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 4,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 6,
        },
    }
    body.update(overrides)
    return body


def _sdk_model(handler, **model_kwargs) -> AnthropicModel:
    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    return AnthropicModel.from_api_key(MODEL, api_key="k", http_client=http_client, **model_kwargs)


class TestRealSdk:
    async def test_request_body_and_response_parsing(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json=_message_json(), headers={"request-id": "req_9"})

        model = _sdk_model(
            handler,
            system_instruction="sys",
            cache_control={"type": "ephemeral"},
            extra_body={"output_config": {"effort": "low"}, "future_field": True},
        )
        response = await model.generate("Hi")
        body = seen[0]
        assert body["model"] == MODEL
        assert body["max_tokens"] == 16_000
        assert body["messages"] == [{"role": "user", "content": "Hi"}]
        assert body["system"] == "sys"
        assert body["cache_control"] == {"type": "ephemeral"}
        assert body["output_config"] == {"effort": "low"}
        assert body["future_field"] is True
        assert "temperature" not in body
        assert response.text == "Hello"
        assert response.reasoning == "consider"
        assert response.tool_calls[0].arguments == json.dumps({"q": 1})
        assert (response.input_tokens, response.cached_input_tokens) == (59, 50)
        assert response.cache_write_input_tokens == 6
        assert response.metadata["request_id"] == "req_9"
        assert response.metadata["finish_reason"] == "tool_use"
        json.dumps(response.metadata)

    @pytest.mark.parametrize("default_system", ["default", None])
    async def test_cache_group_follows_the_system_prompt_actually_sent(self, default_system):
        # The SDK merges extra_body over named parameters, so a nested
        # extra_body system is what goes on the wire; the group must hash it (#375 R1).
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json=_message_json())

        model = _sdk_model(handler, system_instruction=default_system)
        response = await model.generate("Hi", config={"extra_body": {"system": "actual"}})
        assert seen[0]["system"] == "actual"
        expected = "si:" + hashlib.sha256(b"actual").hexdigest()[:12]
        assert response.metadata["cache_group"] == expected

    async def test_explicit_cache_group_still_wins_over_the_sent_system(self):
        def handler(request):
            return httpx2.Response(200, json=_message_json())

        model = _sdk_model(handler, system_instruction="default", cache_group="mine")
        response = await model.generate("Hi", config={"extra_body": {"system": "actual"}})
        assert response.metadata["cache_group"] == "mine"

    async def test_per_call_headers_override_model_defaults(self):
        # Per-call extra_headers win per key; untouched defaults survive (#375 R2).
        seen = []

        def handler(request):
            seen.append(request.headers)
            return httpx2.Response(200, json=_message_json())

        model = _sdk_model(handler, extra_headers={"x-review": "default", "x-keep": "kept"})
        await model.generate("Hi", config={"extra_headers": {"x-review": "call"}})
        assert seen[0]["x-review"] == "call"
        assert seen[0]["x-keep"] == "kept"

        await model.generate("Hi")
        assert seen[1]["x-review"] == "default"

    async def test_overloaded_is_not_retried_by_sdk_and_classifies_as_rate_limit(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx2.Response(
                529,
                json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
                headers={"retry-after": "4"},
            )

        with pytest.raises(anthropic.APIStatusError) as raised:
            await _sdk_model(handler).generate("x")
        assert len(calls) == 1  # max_retries=0
        info = AnthropicErrorClassifier().classify(raised.value)
        assert (info.is_rate_limit, info.is_retryable, info.error_category) == (
            True,
            True,
            "server_overload",
        )
        assert info.suggested_wait == 4.0

    async def test_rate_limit_retry_after(self):
        def handler(request):
            return httpx2.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
                headers={"retry-after": "7"},
            )

        with pytest.raises(anthropic.RateLimitError) as raised:
            await _sdk_model(handler).generate("x")
        info = AnthropicErrorClassifier().classify(raised.value)
        assert (info.is_rate_limit, info.error_category, info.suggested_wait) == (
            True,
            "rate_limit",
            7.0,
        )


def _status_error(cls, status: int, error_type: str, headers=None, details=None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error: dict = {"type": error_type, "message": error_type}
    if details is not None:
        error["details"] = details
    body = {"type": "error", "error": error}
    response = httpx2.Response(status, request=request, json=body, headers=headers or {})
    return cls(error_type, response=response, body=body)


@pytest.mark.parametrize("store_name", ["jsonl", "sqlite"])
async def test_artifact_identity_names_the_anthropic_provider(tmp_path, store_name):
    # Checkpoints record provider="anthropic", like the other built-ins (#375 review).
    from llm_minder import JsonlArtifactStore, SqliteArtifactStore
    from llm_minder.artifacts import infer_artifact_identity

    def handler(request):
        return httpx2.Response(200, json=_message_json())

    strategy = AnthropicStrategy(_sdk_model(handler))
    identity = infer_artifact_identity(strategy)
    assert identity.provider == "anthropic"
    assert identity.model == MODEL

    store_type = JsonlArtifactStore if store_name == "jsonl" else SqliteArtifactStore
    from llm_minder import ProcessorConfig, process_prompts

    path = tmp_path / f"store.{store_name}"
    result = await process_prompts(
        strategy,
        [("x", "Hi")],
        config=ProcessorConfig(max_workers=1),
        artifact_store=store_type(path),
    )
    assert result.succeeded == 1
    reopened = store_type(path)
    try:
        records = [r async for r in reopened.iter_results()]
    finally:
        await reopened.close()
    assert len(records) == 1
    await strategy.cleanup()
