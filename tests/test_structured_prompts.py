"""Structured message prompts (#315): LLMWorkItem.messages and execute_messages().

Covers validation, the opt-in strategy hook and its configuration error, the
executor routing (execute, dry run, token estimation, middleware), the call-side
entry points, FakeStrategy, and fingerprints. Provider mappings are in
test_structured_prompts_models.py; checkpoint round trips on both stores are in
test_store_contract.py.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_minder import (
    BaseMiddleware,
    CallableStrategy,
    CallOutcome,
    LLMCallPool,
    LLMCallStrategy,
    LLMResponse,
    LLMWorkItem,
    ModelStrategy,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryState,
    TokenEstimate,
    TokenUsage,
    call,
    call_result,
    process_prompts,
    process_stream,
)
from llm_minder._internal.artifact_codec import fingerprint_messages, fingerprint_work_item
from llm_minder._internal.messages import render_messages, supports_messages
from llm_minder.testing import FakeStrategy

MESSAGES = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": [{"type": "text", "text": "Hi", "cache_control": {"x": 1}}]},
]
TOKENS: TokenUsage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class StrOnly(LLMCallStrategy[str]):
    """A strategy written for string prompts only."""

    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt, TOKENS, None


class Recorder(LLMCallStrategy[str]):
    """Supports both forms and records what it was handed."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.messages: list[Any] = []

    async def execute(self, prompt, attempt, timeout, state=None):
        self.prompts.append(prompt)
        return f"str:{prompt}", TOKENS, None

    async def execute_messages(self, messages, attempt, timeout, state=None):
        self.messages.append(messages)
        return f"msgs:{messages[-1]['content']}", TOKENS, None


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    def test_messages_are_deep_copied_into_json_values(self):
        source = [{"role": "user", "content": ("a",), "extra": {"k": (1, 2)}}]
        source[0]["content"] = [{"type": "text", "text": "a"}]  # type: ignore[assignment]
        item = LLMWorkItem(item_id="i", strategy=Recorder(), messages=source)
        source[0]["content"][0]["text"] = "changed"  # type: ignore[index]
        assert item.messages == [
            {"role": "user", "content": [{"type": "text", "text": "a"}], "extra": {"k": [1, 2]}}
        ]
        assert item.prompt == ""

    def test_messages_is_keyword_only_so_positions_are_unchanged(self):
        parameters = inspect.signature(LLMWorkItem).parameters
        assert parameters["messages"].kind is inspect.Parameter.KEYWORD_ONLY
        positional = [p for p in parameters.values() if p.kind is not p.KEYWORD_ONLY]
        assert [p.name for p in positional][:5] == [
            "item_id",
            "strategy",
            "prompt",
            "context",
            "submission_index",
        ]

    @pytest.mark.parametrize(
        ("messages", "error", "match"),
        [
            ("hello", TypeError, r"\$\.messages must be a list"),
            ({"role": "user", "content": "x"}, TypeError, r"\$\.messages must be a list"),
            ([], ValueError, "at least one message"),
            (["hi"], TypeError, r"\$\.messages\[0\] must be a mapping"),
            ([{"content": "x"}], TypeError, r"\$\.messages\[0\]\.role must be a string"),
            ([{"role": "user"}], ValueError, r"\$\.messages\[0\]\.content is required"),
            (
                [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
                + [{"role": "user", "content": {"text": "c"}}],
                TypeError,
                r"\$\.messages\[2\]\.content must be a string or a list",
            ),
            (
                [{"role": "user", "content": ["text"]}],
                TypeError,
                r"\$\.messages\[0\]\.content\[0\] must be a mapping",
            ),
            (
                [{"role": "user", "content": "x", "obj": object()}],
                ValueError,
                r"\$\.messages\[0\]\.obj",
            ),
            (
                [{"role": "user", "content": "bad \ud800"}],
                ValueError,
                r"surrogate at \$\.messages\[0\]\.content",
            ),
        ],
    )
    def test_shape_errors_name_a_path(self, messages, error, match):
        with pytest.raises(error, match=match):
            LLMWorkItem(item_id="i", strategy=Recorder(), messages=messages)

    def test_exactly_one_of_prompt_and_messages(self):
        with pytest.raises(ValueError, match="exactly one of prompt or messages"):
            LLMWorkItem(item_id="i", strategy=Recorder(), prompt="x", messages=MESSAGES)

    def test_string_items_are_unchanged(self):
        item = LLMWorkItem(item_id="i", strategy=StrOnly(), prompt="x")
        assert item.messages is None
        # An empty prompt with no messages stays valid, as before.
        LLMWorkItem(item_id="i", strategy=StrOnly())
        with pytest.raises(TypeError, match="prompt must be a string"):
            LLMWorkItem(item_id="i", strategy=StrOnly(), prompt=[{"role": "user"}])  # type: ignore[arg-type]

    def test_render_messages_keeps_text_parts_only(self):
        rendered = render_messages(
            [
                {"role": "system", "content": "S"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": "u"}},
                        {"type": "text", "text": "b"},
                    ],
                },
            ]
        )
        assert rendered == "system: S\nuser: a\nb"


# ── The opt-in hook and the configuration error ──────────────────────────────


class TestUnsupportedStrategy:
    def test_support_means_overriding_execute_messages(self):
        assert supports_messages(Recorder())
        assert supports_messages(FakeStrategy("x"))
        assert supports_messages(ModelStrategy(MagicMock()))
        assert not supports_messages(StrOnly())
        assert not supports_messages(CallableStrategy(_invoke))

    def test_subclass_overriding_only_execute_is_not_silently_bypassed(self):
        """A built-in subclass that customizes execute() must not inherit the messages hook."""

        class CustomExecute(OpenAIStrategy[str]):
            async def execute(self, prompt, attempt, timeout, state=None):  # type: ignore[override]
                return await super().execute(prompt + " (custom)", attempt, timeout, state)

        class CustomBoth(CustomExecute):
            async def execute_messages(self, messages, attempt, timeout, state=None):  # type: ignore[override]
                return await super().execute_messages(messages, attempt, timeout, state)

        assert not supports_messages(CustomExecute(MagicMock()))
        assert supports_messages(CustomBoth(MagicMock()))
        with pytest.raises(TypeError, match="overrides execute\\(\\) but not execute_messages"):
            LLMWorkItem(item_id="i", strategy=CustomExecute(MagicMock()), messages=MESSAGES)

    def test_item_construction_raises_naming_the_strategy_and_fix(self):
        with pytest.raises(TypeError) as excinfo:
            LLMWorkItem(item_id="i", strategy=StrOnly(), messages=MESSAGES)
        message = str(excinfo.value)
        assert "StrOnly does not support structured messages" in message
        assert "override execute_messages()" in message
        assert "string prompt" in message

    async def test_base_hook_raises_type_error(self):
        with pytest.raises(TypeError, match="StrOnly does not support"):
            await StrOnly().execute_messages(MESSAGES, 1, 1.0)

    async def test_add_work_rechecks_a_swapped_strategy(self):
        item = LLMWorkItem(item_id="i", strategy=Recorder(), messages=MESSAGES)
        item.strategy = StrOnly()
        async with ParallelBatchProcessor[str, str, None]() as processor:
            with pytest.raises(TypeError, match="StrOnly does not support"):
                await processor.add_work(item)

    async def test_call_and_call_result_raise_before_any_call(self):
        strategy = StrOnly()
        strategy.execute = AsyncMock(side_effect=AssertionError("no call"))  # type: ignore[method-assign]
        with pytest.raises(TypeError, match="StrOnly does not support"):
            await call_result(strategy, MESSAGES)
        with pytest.raises(TypeError, match="StrOnly does not support"):
            await call(strategy, MESSAGES)

    async def test_pool_submit_raises(self):
        async with LLMCallPool(StrOnly()) as pool:
            with pytest.raises(TypeError, match="StrOnly does not support"):
                await pool.submit(MESSAGES)
            with pytest.raises(TypeError, match="StrOnly does not support"):
                await pool.submit_result(MESSAGES)
            assert await pool.submit("still works") == "still works"

    async def test_process_prompts_raises(self):
        with pytest.raises(TypeError, match="StrOnly does not support"):
            await process_prompts(StrOnly(), [MESSAGES])

    def test_callable_strategy_is_a_follow_up(self):
        with pytest.raises(TypeError, match="CallableStrategy does not support"):
            LLMWorkItem(item_id="i", strategy=CallableStrategy(_invoke), messages=MESSAGES)


async def _invoke(prompt: str, *, attempt: int, timeout: float, state: RetryState | None):
    return CallOutcome(prompt, token_usage={"total_tokens": 1})


# ── ModelStrategy.execute_messages ───────────────────────────────────────────


def _mock_model(text: str = "out") -> MagicMock:
    model = MagicMock(spec=["generate"])
    model.generate = AsyncMock(
        return_value=LLMResponse(
            text=text, input_tokens=3, output_tokens=2, total_tokens=5, metadata={"m": 1}
        )
    )
    return model


class TestModelStrategy:
    async def test_mirrors_execute_with_the_message_list(self):
        class Escalating(ModelStrategy[str]):
            def attempt_generation_config(self, attempt, state):
                return {"max_tokens": 99} if attempt > 1 else None

        model = _mock_model()
        strategy = Escalating(model, temperature=0.2, generation_config={"top_p": 0.5})
        output, tokens, metadata = await strategy.execute_messages(MESSAGES, 2, 10.0)
        assert (output, tokens["total_tokens"], metadata) == ("out", 5, {"m": 1})
        args, kwargs = model.generate.await_args
        assert args == (MESSAGES,)
        assert kwargs == {"temperature": 0.2, "config": {"top_p": 0.5, "max_tokens": 99}}

    async def test_parse_failure_attaches_token_usage(self):
        def parser(response: LLMResponse) -> str:
            raise ValueError("bad output")

        strategy = ModelStrategy(_mock_model(), parser)
        with pytest.raises(ValueError, match="bad output") as excinfo:
            await strategy.execute_messages(MESSAGES, 1, 10.0)
        assert excinfo.value.__dict__["_failed_token_usage"]["total_tokens"] == 5

    async def test_strategy_token_estimator_receives_the_rendering(self):
        seen: list[str] = []

        def estimator(prompt, *, strategy, attempt, state):
            seen.append(prompt)
            return TokenEstimate(input_tokens=len(prompt))

        strategy = ModelStrategy(_mock_model(), token_estimator=estimator)
        result = await call_result(
            strategy, MESSAGES, config=ProcessorConfig(max_tokens_per_minute=10_000)
        )
        assert result.success
        assert seen == ["system: Be terse.\nuser: Hi"]


# ── FakeStrategy ─────────────────────────────────────────────────────────────


class TestFakeStrategy:
    async def test_fixed_response(self):
        assert await call(FakeStrategy("fixed"), MESSAGES) == "fixed"

    async def test_callable_response_receives_the_rendering(self):
        fake = FakeStrategy(lambda prompt: prompt.upper())
        assert await call(fake, MESSAGES) == "SYSTEM: BE TERSE.\nUSER: HI"
        assert await call(fake, "plain") == "PLAIN"

    async def test_messages_handler_receives_the_list(self):
        seen: list[Any] = []

        def handler(messages):
            seen.append(messages)
            return len(messages)

        fake = FakeStrategy(lambda prompt: -1, messages_handler=handler)
        assert await call(fake, MESSAGES) == 2
        assert seen == [MESSAGES]
        assert await call(fake, "plain") == -1

    async def test_schedule_and_counter_are_shared(self, fast_retry):
        fake = FakeStrategy(
            "ok",
            failure_schedule=[ConnectionError("boom"), None],
            token_usage={"input_tokens": 4, "output_tokens": 1},
        )
        result = await call_result(fake, MESSAGES, config=ProcessorConfig(retry=fast_retry))
        assert result.success and result.output == "ok"
        assert fake.calls == 2
        assert result.token_usage["input_tokens"] == 4

    def test_messages_handler_must_be_callable(self):
        with pytest.raises(TypeError, match="messages_handler"):
            FakeStrategy("x", messages_handler="nope")  # type: ignore[arg-type]


# ── Executor routing ─────────────────────────────────────────────────────────


class TestExecutor:
    async def test_process_prompts_accepts_bare_and_tuple_message_entries(self):
        recorder = Recorder()
        batch = await process_prompts(
            recorder,
            [
                "plain",
                [{"role": "user", "content": "bare"}],
                ("t2", [{"role": "user", "content": "pair"}]),
                ("t3", [{"role": "user", "content": "triple"}], {"ctx": 1}),
            ],
            preserve_order=True,
        )
        assert [r.output for r in batch.results] == [
            "str:plain",
            "msgs:bare",
            "msgs:pair",
            "msgs:triple",
        ]
        assert [r.item_id for r in batch.results] == ["item_0", "item_1", "t2", "t3"]
        assert batch.results[3].context == {"ctx": 1}
        assert recorder.prompts == ["plain"]

    async def test_process_stream_accepts_messages(self):
        results = [r async for r in process_stream(Recorder(), [MESSAGES])]
        assert (
            results[0].output == "msgs:[{'type': 'text', 'text': 'Hi', 'cache_control': {'x': 1}}]"
        )

    async def test_top_level_str_bytes_and_bare_mapping_entries_are_rejected(self):
        for prompts in ("abc", b"abc"):
            with pytest.raises(TypeError, match="iterable of prompt entries"):
                await process_prompts(Recorder(), prompts)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=r"wrap the message list: \[messages\]"):
            await process_prompts(Recorder(), MESSAGES)  # type: ignore[arg-type]

    async def test_each_attempt_gets_a_private_copy(self, fast_retry):
        class Mutating(LLMCallStrategy[int]):
            def __init__(self) -> None:
                self.lengths: list[int] = []

            async def execute(self, prompt, attempt, timeout, state=None):
                raise AssertionError("not used")

            async def execute_messages(self, messages, attempt, timeout, state=None):
                self.lengths.append(len(messages))
                messages.append({"role": "user", "content": "retry feedback"})
                if attempt == 1:
                    raise ConnectionError("transient")
                return len(messages), TOKENS, None

        strategy = Mutating()
        item = LLMWorkItem(item_id="i", strategy=strategy, messages=MESSAGES)
        async with ParallelBatchProcessor[str, int, None](
            config=ProcessorConfig(retry=fast_retry)
        ) as processor:
            await processor.add_work(item)
            batch = await processor.process_all()
        assert batch.results[0].output == 3
        assert strategy.lengths == [2, 2]
        assert item.messages == MESSAGES

    async def test_dry_run_uses_the_base_placeholder(self):
        class WithDryRun(Recorder):
            async def dry_run(self, prompt: str) -> tuple[str, TokenUsage]:
                raise AssertionError("dry_run(prompt) is for string prompts")

        result = await call_result(WithDryRun(), MESSAGES, config=ProcessorConfig(dry_run=True))
        assert result.success
        assert result.output == "[DRY-RUN] Mock output for prompt: system: Be terse.\nuser: Hi..."

    async def test_processor_token_estimator_receives_the_rendering(self):
        seen: list[str] = []

        def estimator(prompt, *, strategy, attempt, state):
            seen.append(prompt)
            return TokenEstimate(input_tokens=5)

        config = ProcessorConfig(max_tokens_per_minute=10_000, token_estimator=estimator)
        await process_prompts(Recorder(), ["plain", MESSAGES], config=config, preserve_order=True)
        assert sorted(seen) == sorted(["plain", "system: Be terse.\nuser: Hi"])

    async def test_middleware_may_replace_messages_and_is_revalidated(self):
        class Rewrite(BaseMiddleware[str, str, None]):
            async def before_process(self, work_item):
                if work_item.item_id == "replace":
                    return LLMWorkItem(
                        item_id=work_item.item_id,
                        strategy=work_item.strategy,
                        messages=[*work_item.messages, {"role": "user", "content": "added"}],
                    )
                if work_item.item_id == "mutate-bad":
                    work_item.messages.append({"role": 3, "content": "x"})
                    return work_item
                if work_item.item_id == "both":
                    work_item.prompt = "also a prompt"
                    return work_item
                if work_item.item_id == "prompt-only":
                    # Middleware that only rewrites prompt leaves messages alone.
                    work_item.prompt = work_item.prompt.upper()
                    return work_item
                return work_item

        recorder = Recorder()
        async with ParallelBatchProcessor[str, str, None](middlewares=[Rewrite()]) as processor:
            for item_id in ("replace", "mutate-bad", "both", "prompt-only"):
                await processor.add_work(
                    LLMWorkItem(item_id=item_id, strategy=recorder, messages=MESSAGES)
                )
            batch = await processor.process_all()
        by_id = batch.by_id()
        assert by_id["replace"].output == "msgs:added"
        assert by_id["prompt-only"].output == (
            "msgs:[{'type': 'text', 'text': 'Hi', 'cache_control': {'x': 1}}]"
        )
        for item_id, match in (
            ("mutate-bad", r"\$\.messages\[2\]\.role must be a string"),
            ("both", "exactly one of prompt or messages"),
        ):
            assert not by_id[item_id].success
            assert "Invalid before_process work item" in (by_id[item_id].error or "")
            assert re.search(match, by_id[item_id].error or "")
        assert len(recorder.messages) == 2

    async def test_middleware_swapping_to_a_str_only_strategy_fails_the_item(self):
        class Swap(BaseMiddleware[str, str, None]):
            async def before_process(self, work_item):
                work_item.strategy = StrOnly()
                return work_item

        async with ParallelBatchProcessor[str, str, None](middlewares=[Swap()]) as processor:
            await processor.add_work(
                LLMWorkItem(item_id="i", strategy=Recorder(), messages=MESSAGES)
            )
            batch = await processor.process_all()
        result = batch.results[0]
        assert not result.success
        assert "StrOnly does not support structured messages" in (result.error or "")


# ── Call-side entry points ───────────────────────────────────────────────────


class TestCallSide:
    async def test_call_and_call_result_route_a_list_to_messages(self):
        recorder = Recorder()
        assert await call(recorder, [{"role": "user", "content": "one"}]) == "msgs:one"
        result = await call_result(recorder, ({"role": "user", "content": "two"},))
        assert result.output == "msgs:two"
        assert await call(recorder, "three") == "str:three"

    async def test_pool_submit_routes_a_list_to_messages(self):
        async with LLMCallPool(Recorder()) as pool:
            assert await pool.submit([{"role": "user", "content": "x"}]) == "msgs:x"
            result = await pool.submit_result([{"role": "user", "content": "y"}])
            assert result.output == "msgs:y"
            assert await pool.submit("z") == "str:z"

    async def test_invalid_messages_raise_from_call(self):
        with pytest.raises(TypeError, match=r"\$\.messages\[0\]\.role"):
            await call(Recorder(), [{"content": "x"}])


# ── Fingerprints ─────────────────────────────────────────────────────────────


class TestFingerprint:
    def _prepared(self, item: LLMWorkItem[Any, Any, Any]):
        return fingerprint_work_item(
            item, context_in_identity=True, encoder=None, context_fingerprinter=None
        )

    def test_string_fingerprint_is_unchanged(self):
        prepared = self._prepared(LLMWorkItem(item_id="i", strategy=Recorder(), prompt="x"))
        assert prepared.prompt_fingerprint == hashlib.sha256(b"x").hexdigest()

    def test_messages_fingerprint_is_prefixed_canonical_json(self):
        item = LLMWorkItem(item_id="i", strategy=Recorder(), messages=MESSAGES)
        canonical = json.dumps(
            MESSAGES, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        expected = hashlib.sha256(b"llm-minder:messages:v1\0" + canonical).hexdigest()
        assert self._prepared(item).prompt_fingerprint == expected == fingerprint_messages(MESSAGES)

    def test_key_order_does_not_matter_but_content_does(self):
        reordered = [{"content": m["content"], "role": m["role"]} for m in copy.deepcopy(MESSAGES)]
        assert fingerprint_messages(reordered) == fingerprint_messages(MESSAGES)
        changed = copy.deepcopy(MESSAGES)
        changed[0]["content"] = "Be verbose."
        assert fingerprint_messages(changed) != fingerprint_messages(MESSAGES)

    def test_a_string_and_its_message_never_share_a_fingerprint(self):
        as_string = self._prepared(LLMWorkItem(item_id="i", strategy=Recorder(), prompt="x"))
        as_message = self._prepared(
            LLMWorkItem(
                item_id="i", strategy=Recorder(), messages=[{"role": "user", "content": "x"}]
            )
        )
        as_json = self._prepared(
            LLMWorkItem(
                item_id="i",
                strategy=Recorder(),
                prompt=json.dumps([{"role": "user", "content": "x"}], separators=(",", ":")),
            )
        )
        fingerprints = {p.prompt_fingerprint for p in (as_string, as_message, as_json)}
        assert len(fingerprints) == 3
        assert as_string.input_fingerprint != as_message.input_fingerprint
