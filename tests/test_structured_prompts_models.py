"""Per-model mapping of structured message prompts (#315), with mocked SDK clients.

One canonical Chat Completions shape; each built-in model maps it to its provider.
No live API calls.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_minder import (
    DeepSeekModel,
    DeepSeekStrategy,
    GeminiStrategy,
    OpenAIModel,
    OpenAIStrategy,
    OpenRouterModel,
    OpenRouterStrategy,
    ProcessorConfig,
    call_result,
)
from llm_minder._internal.messages import is_chat_message_list, to_anthropic, to_gemini
from llm_minder.models import GeminiCachedModel, GeminiModel

CACHE = {"type": "ephemeral"}
MESSAGES = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": [{"type": "text", "text": "Doc", "cache_control": CACHE}]},
    {"role": "assistant", "content": "Earlier answer"},
    {"role": "user", "content": "Question"},
]


def _si_group(text: str) -> str:
    return "si:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ── OpenAI-compatible Chat Completions: pass through ─────────────────────────


def _chat_response() -> MagicMock:
    response = MagicMock()
    response.model = "m"
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "hello"
    response.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens = 10, 5, 15
    usage.prompt_tokens_details = None
    response.usage = usage
    response.error = None
    return response


class TestChatCompletions:
    async def test_messages_pass_through_with_cache_control(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_chat_response())
        model = OpenRouterModel("anthropic/claude-haiku-4-5", client, system_instruction="dflt")
        await OpenRouterStrategy(model).execute_messages(MESSAGES, 1, 10.0)
        sent = client.chat.completions.create.await_args.kwargs["messages"]
        # The list's system message replaces the model default.
        assert sent == MESSAGES

    async def test_default_system_instruction_is_added_without_a_system_message(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_chat_response())
        model = DeepSeekModel("deepseek-v4-flash", client, system_instruction="dflt")
        await DeepSeekStrategy(model).execute_messages([{"role": "user", "content": "q"}], 1, 10.0)
        sent = client.chat.completions.create.await_args.kwargs["messages"]
        assert sent == [{"role": "system", "content": "dflt"}, {"role": "user", "content": "q"}]


# ── Responses: the existing translation ──────────────────────────────────────


def _responses_response() -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=20,
        output_tokens=8,
        total_tokens=28,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return SimpleNamespace(
        id="resp_1",
        model="m",
        status="completed",
        error=None,
        incomplete_details=None,
        output_text="hello",
        output=[],
        usage=usage,
    )


EXPECTED_RESPONSES_INPUT = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": [{"type": "input_text", "text": "Doc"}]},
    {"role": "assistant", "content": "Earlier answer"},
    {"role": "user", "content": "Question"},
]


class TestResponses:
    async def test_openai_translates_messages(self):
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_responses_response())
        model = OpenAIModel("gpt-6-luna", client, system_instruction="dflt")
        await OpenAIStrategy(model).execute_messages(MESSAGES, 1, 10.0)
        kwargs = client.responses.create.await_args.kwargs
        assert kwargs["input"] == EXPECTED_RESPONSES_INPUT
        assert "instructions" not in kwargs  # the list's system message wins

    async def test_deepseek_responses_translates_messages(self):
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_responses_response())
        model = DeepSeekModel(
            "deepseek-v4-flash", client, api_surface="responses", system_instruction="dflt"
        )
        await DeepSeekStrategy(model).execute_messages(MESSAGES, 1, 10.0)
        kwargs = client.responses.create.await_args.kwargs
        assert kwargs["input"] == EXPECTED_RESPONSES_INPUT
        assert "instructions" not in kwargs

    async def test_deepseek_responses_keeps_native_items_and_strings(self):
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_responses_response())
        model = DeepSeekModel("deepseek-v4-flash", client, api_surface="responses")
        native = [
            {"role": "user", "content": [{"type": "input_text", "text": "n"}]},
            {"type": "message", "role": "user", "content": "typed"},
        ]
        await model.generate(native)
        assert client.responses.create.await_args.kwargs["input"] == native
        await model.generate("plain")
        assert client.responses.create.await_args.kwargs["input"] == "plain"


# ── Anthropic ────────────────────────────────────────────────────────────────

anthropic = pytest.importorskip("anthropic")


def _anthropic_client() -> SimpleNamespace:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        stop_reason="end_turn",
        model="claude-opus-5-5",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
        stop_details=None,
    )
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=message)))


class TestAnthropic:
    def _model(self, client, **kwargs):
        from llm_minder import AnthropicModel

        return AnthropicModel("claude-opus-5-5", client, **kwargs)

    async def test_system_becomes_text_blocks_when_a_part_has_extras(self):
        client = _anthropic_client()
        messages = [
            {"role": "system", "content": "Rules."},
            {
                "role": "system",
                "content": [{"type": "text", "text": "Doc", "cache_control": CACHE}],
            },
            {"role": "user", "content": "Q"},
        ]
        response = await self._model(client, system_instruction="dflt").generate(messages)
        sent = client.messages.create.await_args.kwargs
        assert sent["system"] == [
            {"type": "text", "text": "Rules."},
            {"type": "text", "text": "Doc", "cache_control": CACHE},
        ]
        assert sent["messages"] == [{"role": "user", "content": "Q"}]
        # The cache group is labelled by the system actually sent.
        assert response.metadata["cache_group"] == _si_group("Rules.\n\nDoc")

    async def test_plain_system_stays_a_string_and_messages_pass_through(self):
        client = _anthropic_client()
        from llm_minder import AnthropicStrategy

        strategy = AnthropicStrategy(self._model(client, system_instruction="dflt"))
        _, _, metadata = await strategy.execute_messages(MESSAGES, 1, 10.0)
        sent = client.messages.create.await_args.kwargs
        assert sent["system"] == "Be terse."
        assert sent["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Doc", "cache_control": CACHE}]},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Question"},
        ]
        assert metadata["cache_group"] == _si_group("Be terse.")

    async def test_system_precedence(self):
        client = _anthropic_client()
        model = self._model(client, system_instruction="dflt")
        user_only = [{"role": "user", "content": "Q"}]
        await model.generate(user_only)
        assert client.messages.create.await_args.kwargs["system"] == "dflt"
        await model.generate(user_only, system_instruction="per-call")
        assert client.messages.create.await_args.kwargs["system"] == "per-call"
        await model.generate(MESSAGES, system_instruction="per-call")
        assert client.messages.create.await_args.kwargs["system"] == "Be terse."

    @pytest.mark.parametrize(
        ("messages", "match"),
        [
            (
                [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}],
                "system message after a non-system one",
            ),
            ([{"role": "tool", "content": "x"}], r"role is 'tool'"),
            (
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}],
                "only 'text' parts are supported",
            ),
            ([{"role": "user", "content": "x", "name": "bob"}], r"keys \['name'\]"),
            ([{"role": "system", "content": "only system"}], "at least one user or assistant"),
        ],
    )
    async def test_unsupported_shapes_raise_before_the_call(self, messages, match):
        client = _anthropic_client()
        with pytest.raises(ValueError, match=match):
            await self._model(client).generate(messages)
        client.messages.create.assert_not_awaited()

    async def test_unsupported_shape_is_a_non_retryable_logic_error(self, fast_retry):
        from llm_minder import AnthropicStrategy

        client = _anthropic_client()
        strategy = AnthropicStrategy(self._model(client))
        result = await call_result(
            strategy,
            [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}],
            config=ProcessorConfig(retry=fast_retry),
        )
        assert not result.success
        assert result.error_category == "logic_error"
        assert len(result.timing.attempts) == 1
        client.messages.create.assert_not_awaited()

    def test_to_anthropic_without_system(self):
        messages, system = to_anthropic([{"role": "user", "content": "x"}])
        assert (messages, system) == ([{"role": "user", "content": "x"}], None)


# ── Gemini ───────────────────────────────────────────────────────────────────


def _gemini_response() -> MagicMock:
    response = MagicMock()
    response.text = "hello"
    usage = response.usage_metadata
    usage.prompt_token_count, usage.candidates_token_count = 10, 5
    usage.tool_use_prompt_token_count = usage.thoughts_token_count = None
    usage.total_token_count, usage.cached_content_token_count = 15, 0
    response.candidates = []
    return response


def _gemini_client() -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=_gemini_response())
    return client


EXPECTED_GEMINI_CONTENTS = [
    {"role": "user", "parts": [{"text": "Doc"}]},
    {"role": "model", "parts": [{"text": "Earlier answer"}]},
    {"role": "user", "parts": [{"text": "Question"}]},
]


class TestGemini:
    async def test_messages_map_to_contents_and_system_instruction(self):
        client = _gemini_client()
        model = GeminiModel("gemini-3.5-flash", client, system_instruction="dflt")
        _, _, metadata = await GeminiStrategy(model).execute_messages(MESSAGES, 1, 10.0)
        kwargs = client.aio.models.generate_content.await_args.kwargs
        assert kwargs["contents"] == EXPECTED_GEMINI_CONTENTS
        assert kwargs["config"]["system_instruction"] == "Be terse."
        assert metadata["cache_group"] == _si_group("Be terse.")

    async def test_system_precedence(self):
        client = _gemini_client()
        model = GeminiModel("gemini-3.5-flash", client, system_instruction="dflt")
        user_only = [{"role": "user", "content": "Q"}]
        await model.generate(user_only)
        config = client.aio.models.generate_content.await_args.kwargs["config"]
        assert config["system_instruction"] == "dflt"
        await model.generate(user_only, system_instruction="per-call")
        config = client.aio.models.generate_content.await_args.kwargs["config"]
        assert config["system_instruction"] == "per-call"
        await model.generate(MESSAGES, system_instruction="per-call")
        config = client.aio.models.generate_content.await_args.kwargs["config"]
        assert config["system_instruction"] == "Be terse."

    @pytest.mark.parametrize(
        "raw",
        [
            ["a string part", "another"],
            [{"text": "a part dict"}],
            [{"role": "user", "parts": [{"text": "a content dict"}]}],
        ],
    )
    async def test_raw_contents_lists_are_unchanged(self, raw):
        assert not is_chat_message_list(raw)
        client = _gemini_client()
        await GeminiModel("gemini-3.5-flash", client).generate(raw)
        assert client.aio.models.generate_content.await_args.kwargs["contents"] is raw

    @pytest.mark.parametrize(
        ("messages", "match"),
        [
            ([{"role": "developer", "content": "x"}], r"role is 'developer'"),
            (
                [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}],
                "only 'text' parts are supported",
            ),
            (
                [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}],
                "system message after a non-system one",
            ),
        ],
    )
    async def test_unsupported_shapes_raise_before_the_call(self, messages, match):
        client = _gemini_client()
        with pytest.raises(ValueError, match=match):
            await GeminiModel("gemini-3.5-flash", client).generate(messages)
        client.aio.models.generate_content.assert_not_awaited()

    def test_text_part_extras_are_dropped(self):
        contents, system = to_gemini(MESSAGES)
        assert contents == EXPECTED_GEMINI_CONTENTS and system == "Be terse."

    def _cached_model(self) -> tuple[GeminiCachedModel, MagicMock]:
        client = _gemini_client()
        cache = MagicMock()
        cache.name = "cache-1"
        model = GeminiCachedModel("gemini-3.5-flash", client, cached_content=[])
        model._cache = cache
        model._cache_created_at = 10**12  # never expired
        return model, client

    async def test_cached_model_sends_messages_after_the_cached_content(self):
        model, client = self._cached_model()
        await model.generate(MESSAGES[1:])
        kwargs = client.aio.models.generate_content.await_args.kwargs
        assert kwargs["contents"] == EXPECTED_GEMINI_CONTENTS
        assert kwargs["config"]["cached_content"] == "cache-1"
        assert "system_instruction" not in kwargs["config"]

    async def test_cached_model_rejects_a_system_message(self):
        model, client = self._cached_model()
        with pytest.raises(ValueError, match="system message cannot be sent with cached content"):
            await model.generate(MESSAGES)
        client.aio.models.generate_content.assert_not_awaited()
