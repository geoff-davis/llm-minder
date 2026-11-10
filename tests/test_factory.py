"""Tests for the string-based strategy factory (issue #95).

No live API calls — every test constructs strategies with an explicit
dummy api_key (or asserts on the error path).
"""

import pytest

import llm_minder.models as models_module
from llm_minder import (
    DeepSeekModel,
    DeepSeekStrategy,
    GeminiModel,
    GeminiStrategy,
    OpenAIModel,
    OpenAIStrategy,
    OpenRouterModel,
    OpenRouterStrategy,
    llm,
)


class TestProviderPrefixes:
    def test_openai(self):
        strategy = llm("openai:gpt-4o-mini", api_key="sk-test")
        assert isinstance(strategy, OpenAIStrategy)
        assert isinstance(strategy.model, OpenAIModel)
        assert strategy.model._model == "gpt-4o-mini"

    def test_openrouter(self):
        strategy = llm("openrouter:anthropic/claude-haiku-4-5", api_key="sk-or-test")
        assert isinstance(strategy, OpenRouterStrategy)
        assert isinstance(strategy.model, OpenRouterModel)
        assert strategy.model._model == "anthropic/claude-haiku-4-5"

    def test_deepseek(self):
        strategy = llm("deepseek:deepseek-v4-flash", api_key="sk-test")
        assert isinstance(strategy, DeepSeekStrategy)
        assert isinstance(strategy.model, DeepSeekModel)
        assert strategy.model._model == "deepseek-v4-flash"

    def test_gemini(self):
        strategy = llm("gemini:gemini-2.5-flash", api_key="test-key")
        assert isinstance(strategy, GeminiStrategy)
        assert isinstance(strategy.model, GeminiModel)
        assert strategy.model._model == "gemini-2.5-flash"

    def test_model_id_may_contain_colons(self):
        strategy = llm("openrouter:meta-llama/llama-3.1-8b-instruct:free", api_key="sk-or-test")
        assert strategy.model._model == "meta-llama/llama-3.1-8b-instruct:free"

    def test_provider_prefix_is_case_insensitive(self):
        strategy = llm("OpenAI:gpt-4o-mini", api_key="sk-test")
        assert isinstance(strategy, OpenAIStrategy)

    def test_openai_family_models_own_their_client(self):
        strategy = llm("openai:gpt-4o-mini", api_key="sk-test")
        assert strategy.model._owns_client is True


class TestKwargForwarding:
    def test_max_connections_reaches_capacity_advertisement(self):
        strategy = llm("deepseek:deepseek-v4-flash", api_key="sk-test", max_connections=7)
        assert strategy.model.max_concurrency == 7
        assert strategy.max_concurrency == 7

    def test_deepseek_thinking_toggle(self):
        strategy = llm("deepseek:deepseek-v4-flash", api_key="sk-test", thinking=False)
        assert strategy.model._default_extra_body == {"thinking": {"type": "disabled"}}

    def test_system_instruction_forwarded(self):
        strategy = llm(
            "gemini:gemini-2.5-flash", api_key="test-key", system_instruction="Be terse."
        )
        assert strategy.model._default_system_instruction == "Be terse."

    def test_strategy_kwargs_forwarded(self):
        def parser(response):
            return len(response.text)

        strategy = llm(
            "openai:gpt-4o-mini",
            api_key="sk-test",
            response_parser=parser,
            temperature=None,
            generation_config={"max_tokens": 5},
        )
        assert strategy.response_parser is parser
        assert strategy.temperature is None
        assert strategy.generation_config == {"max_tokens": 5}

    def test_default_parser_returns_text(self):
        from llm_minder import LLMResponse

        strategy = llm("openai:gpt-4o-mini", api_key="sk-test")
        response = LLMResponse(text="hello", input_tokens=1, output_tokens=1, total_tokens=2)
        assert strategy.response_parser(response) == "hello"


class TestErrorMessages:
    def test_unknown_prefix_lists_valid_prefixes(self):
        with pytest.raises(ValueError) as exc_info:
            llm("anthropic:claude-sonnet-4-5")
        message = str(exc_info.value)
        assert "anthropic" in message
        for prefix in ("gemini", "openai", "openrouter", "deepseek"):
            assert f"'{prefix}:'" in message
        assert "llm-minder[openai]" in message

    def test_missing_colon(self):
        with pytest.raises(ValueError, match="provider:model"):
            llm("gpt-4o-mini")

    def test_empty_model_id(self):
        with pytest.raises(ValueError, match="provider:model"):
            llm("openai:")

    def test_missing_openai_dependency_names_extra(self, monkeypatch):
        monkeypatch.setattr(models_module, "AsyncOpenAI", None)
        with pytest.raises(ImportError, match=r"llm-minder\[openai\]"):
            llm("openai:gpt-4o-mini", api_key="sk-test")

    def test_missing_gemini_dependency_names_extra(self, monkeypatch):
        monkeypatch.setattr(models_module, "genai", None)
        with pytest.raises(ImportError, match=r"llm-minder\[gemini\]"):
            llm("gemini:gemini-2.5-flash", api_key="test-key")

    def test_gemini_no_api_key_names_env_vars(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            llm("gemini:gemini-2.5-flash")

    def test_openrouter_no_api_key_names_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            llm("openrouter:anthropic/claude-haiku-4-5")


class TestEnvKeyResolution:
    def test_gemini_reads_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        strategy = llm("gemini:gemini-2.5-flash")
        assert isinstance(strategy, GeminiStrategy)

    def test_gemini_falls_back_to_gemini_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        strategy = llm("gemini:gemini-2.5-flash")
        assert isinstance(strategy, GeminiStrategy)

    def test_openrouter_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        strategy = llm("openrouter:anthropic/claude-haiku-4-5")
        assert isinstance(strategy, OpenRouterStrategy)


class TestErrorClassifierSelection:
    """The factory returns provider-named strategies, so the streaming API's
    classifier auto-selection keeps working."""

    @pytest.mark.parametrize(
        ("spec", "classifier_name"),
        [
            ("gemini:gemini-2.5-flash", "GeminiErrorClassifier"),
            ("openai:gpt-4o-mini", "OpenAIErrorClassifier"),
            ("openrouter:anthropic/claude-haiku-4-5", "OpenRouterErrorClassifier"),
            ("deepseek:deepseek-v4-flash", "OpenAIErrorClassifier"),
        ],
    )
    def test_recommended_classifier(self, spec, classifier_name):
        strategy = llm(spec, api_key="test-key")
        assert type(strategy.recommended_error_classifier()).__name__ == classifier_name


@pytest.mark.asyncio
async def test_prov9_factory_gemini_owned_client_reopens_and_shares(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from llm_minder import call

    clients = []

    def construct(**kwargs):
        client = MagicMock()
        client.aio.aclose = AsyncMock()
        client.aio.models.generate_content = AsyncMock(
            return_value=SimpleNamespace(
                text="ok",
                candidates=[],
                usage_metadata=None,
            )
        )
        clients.append(client)
        return client

    monkeypatch.setattr(models_module.genai, "Client", construct)
    strategy = llm("gemini:test", api_key="test")
    peer = GeminiStrategy(strategy.model)
    await strategy.prepare()
    await peer.prepare()
    await strategy.cleanup()
    clients[0].aio.aclose.assert_not_awaited()
    await peer.cleanup()
    clients[0].aio.aclose.assert_awaited_once()
    clients[0].close.assert_called_once()
    assert await call(strategy, "q") == "ok"
    assert len(clients) == 2
    clients[1].aio.aclose.assert_awaited_once()
    clients[1].close.assert_called_once()


def test_prov9_factory_missing_openai_key_is_value_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        llm("openai:test")


@pytest.mark.parametrize("provider", ["openai", "openrouter", "deepseek", "gemini"])
def test_prov9_factory_lists_valid_kwargs(provider):
    with pytest.raises(
        TypeError, match="Valid kwargs:.*api_key.*system_instruction|Valid kwargs:.*api_key"
    ) as raised:
        llm(f"{provider}:test", api_key="test", typo=True)
    assert "typo" in str(raised.value)
    assert "temperature" not in str(raised.value)


def _record_openai_clients(monkeypatch):
    """Record every AsyncOpenAI client the models build."""
    built = []
    real = models_module.AsyncOpenAI

    def construct(**kwargs):
        client = real(**kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(models_module, "AsyncOpenAI", construct)
    return built


@pytest.mark.parametrize(
    ("spec", "kwargs", "unknown"),
    [
        # DeepSeekModel.from_api_key drops the base class's prompt_cache_key.
        ("deepseek:deepseek-v4-flash", {"prompt_cache_key": "k"}, "prompt_cache_key"),
        (
            "openai-compatible:m",
            {"base_url": "http://x/v1", "api_surface": "responses"},
            "api_surface",
        ),
        ("openrouter:m", {"thinking": False}, "thinking"),
        ("openai:m", {"response_schema": dict}, "response_schema"),
    ],
)
def test_factory_rejects_kwargs_the_provider_does_not_take(monkeypatch, spec, kwargs, unknown):
    built = _record_openai_clients(monkeypatch)
    provider = spec.partition(":")[0]
    with pytest.raises(TypeError, match=f"Unknown {provider} model kwargs: {unknown}\\."):
        llm(spec, api_key="sk-test", **kwargs)
    assert built == []


@pytest.mark.parametrize(
    ("spec", "kwargs"),
    [
        ("openai:m", {"prompt_cache_key": "k", "api_surface": "chat_completions"}),
        ("openrouter:m", {"prompt_cache_key": "k", "referer": "r", "title": "t"}),
        ("openai-compatible:m", {"base_url": "http://x/v1", "prompt_cache_key": "k"}),
        ("deepseek:m", {"thinking": False, "api_surface": "responses", "cache_group": "g"}),
    ],
)
def test_factory_accepts_each_providers_own_kwargs(spec, kwargs):
    assert llm(spec, api_key="sk-test", **kwargs).model is not None


def test_failed_model_construction_closes_the_built_client(monkeypatch):
    built = _record_openai_clients(monkeypatch)
    with pytest.raises(ValueError, match="api_surface"):
        DeepSeekModel.from_api_key("m", api_key="sk-test", api_surface="bogus")
    assert len(built) == 1
    assert built[0].is_closed()


@pytest.mark.asyncio
async def test_failed_model_construction_closes_the_client_inside_a_loop(monkeypatch):
    import asyncio

    built = _record_openai_clients(monkeypatch)
    with pytest.raises(ValueError, match="api_surface"):
        OpenAIModel.from_api_key("m", api_key="sk-test", api_surface="bogus")
    for _ in range(3):
        await asyncio.sleep(0)
    assert len(built) == 1
    assert built[0].is_closed()


def test_failed_client_construction_closes_the_sized_http_client(monkeypatch):
    http_clients = []
    real = models_module._build_openai_http_client

    def build(max_connections):
        http_clients.append(real(max_connections))
        return http_clients[-1]

    monkeypatch.setattr(models_module, "_build_openai_http_client", build)
    with pytest.raises(TypeError, match="not_an_sdk_option"):
        OpenAIModel.from_api_key("m", api_key="sk-test", max_connections=2, not_an_sdk_option=1)
    assert len(http_clients) == 1
    assert http_clients[0].is_closed


def test_missing_key_builds_no_sized_http_client(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        models_module,
        "_build_openai_http_client",
        lambda n: pytest.fail("built an http client before resolving the key"),
    )
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekModel.from_api_key("m", max_connections=2)


def test_failed_model_construction_leaves_a_caller_http_client_open(monkeypatch):
    from openai import DefaultAsyncHttpxClient

    mine = DefaultAsyncHttpxClient()
    with pytest.raises(ValueError, match="api_surface"):
        DeepSeekModel.from_api_key("m", api_key="sk-test", api_surface="bogus", http_client=mine)
    assert not mine.is_closed


def test_factory_gemini_construction_failure_closes_the_client(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    clients = []

    def construct(**kwargs):
        client = MagicMock()
        client.aio.aclose = AsyncMock()
        clients.append(client)
        return client

    monkeypatch.setattr(models_module.genai, "Client", construct)
    with pytest.raises(ValueError, match="cache_group"):
        llm("gemini:test", api_key="test", cache_group="")
    clients[0].aio.aclose.assert_awaited_once()
    clients[0].close.assert_called_once()
