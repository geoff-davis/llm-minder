"""GuardrailConfig.admission: the AttemptAdmission hook (#331)."""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_minder import (
    AbortMode,
    AdmissionRefused,
    ArtifactIdentity,
    AttemptRequest,
    BatchAbortedError,
    ErrorCategory,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallPool,
    LLMResponse,
    LLMWorkItem,
    ModelStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    RetryState,
    SqliteArtifactStore,
    TokenUsage,
    attach_token_usage,
    call_result,
    process_prompts,
)
from llm_minder._internal.attempt_admission import _sdk_retry_setting, warn_if_sdk_retries
from llm_minder._internal.guardrails import AbortController, AdmissionSlot, await_with_guardrails
from llm_minder.observers import BaseObserver, ProcessingEvent
from llm_minder.serialization import batch_result_from_json, batch_result_to_json
from llm_minder.testing import FakeRateLimitError, FakeStrategy

_USAGE = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
_ZERO = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class Recorder:
    """Issues unique tickets and records every admit and settle call."""

    def __init__(
        self,
        *,
        refuse_at: int | None = None,
        admit_error: Exception | None = None,
        settle_error_at: int | None = None,
    ) -> None:
        self.requests: list[AttemptRequest] = []
        self.settles: list[tuple[object, TokenUsage | None, bool]] = []
        self._tickets = itertools.count()
        self.refuse_at = refuse_at
        self.admit_error = admit_error
        self.settle_error_at = settle_error_at

    def admit(self, request: AttemptRequest) -> object:
        self.requests.append(request)
        if self.refuse_at is not None and len(self.requests) == self.refuse_at:
            if self.admit_error is not None:
                raise self.admit_error
            raise AdmissionRefused(f"refused {request.item_id}")
        return ("ticket", next(self._tickets))

    def settle(self, ticket: object, usage: TokenUsage | None, success: bool) -> None:
        self.settles.append((ticket, usage, success))
        if self.settle_error_at is not None and len(self.settles) == self.settle_error_at:
            raise RuntimeError("ledger offline")

    @property
    def admitted(self) -> int:
        refused = 1 if self.refuse_at is not None and len(self.requests) >= self.refuse_at else 0
        return len(self.requests) - refused

    def assert_settled_exactly_once(self) -> None:
        tickets = [ticket for ticket, _, _ in self.settles]
        assert len(tickets) == len(set(tickets)), f"a ticket was settled twice: {tickets}"
        assert len(tickets) == self.admitted, (len(tickets), self.admitted)


class _Events(BaseObserver):
    def __init__(self) -> None:
        self.events: list[tuple[ProcessingEvent, dict[str, Any]]] = []

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.events.append((event, dict(data)))

    def aborted(self) -> list[dict[str, Any]]:
        return [data for event, data in self.events if event is ProcessingEvent.BATCH_ABORTED]


def _config(hook: Any, **kwargs: Any) -> ProcessorConfig:
    guardrail_kwargs = kwargs.pop("guardrails", {})
    return ProcessorConfig(guardrails=GuardrailConfig(admission=hook, **guardrail_kwargs), **kwargs)


def _subset(usage: TokenUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {key: usage[key] for key in _USAGE}  # type: ignore[literal-required]


# --- Configuration -----------------------------------------------------------


def test_none_is_the_default():
    assert GuardrailConfig().admission is None


@pytest.mark.parametrize("hook", [object(), 3, "hook"])
def test_rejects_objects_without_admit_and_settle(hook):
    with pytest.raises(ValueError, match="admission must be an AttemptAdmission"):
        GuardrailConfig(admission=hook)


def test_rejects_async_methods():
    class AsyncAdmit:
        async def admit(self, request):
            return None

        def settle(self, ticket, usage, success):
            pass

    class AsyncSettle:
        def admit(self, request):
            return None

        async def settle(self, ticket, usage, success):
            pass

    with pytest.raises(ValueError, match=r"admission.admit\(\) must be synchronous"):
        GuardrailConfig(admission=AsyncAdmit())
    with pytest.raises(ValueError, match=r"admission.settle\(\) must be synchronous"):
        GuardrailConfig(admission=AsyncSettle())


@pytest.mark.asyncio
async def test_single_call_and_pool_reject_the_hook():
    config = _config(Recorder())
    with pytest.raises(ValueError, match="GuardrailConfig.admission is supported only"):
        await call_result(FakeStrategy("ok"), "p", config=config)
    with pytest.raises(ValueError, match="GuardrailConfig.admission is supported only"):
        LLMCallPool(FakeStrategy("ok"), config=config)


def test_category_and_termination_vocabulary():
    assert ErrorCategory.BATCH_ADMISSION_REFUSED == "batch_admission_refused"
    assert issubclass(AdmissionRefused, BatchAbortedError)


# --- The request -------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_admits_once_and_settles_with_usage():
    hook = Recorder()
    strategy = FakeStrategy("ok", token_usage=_USAGE)
    batch = await process_prompts(strategy, ["a", "b"], config=_config(hook))

    assert batch.succeeded == 2 and batch.termination.kind == "completed"
    hook.assert_settled_exactly_once()
    assert len(hook.requests) == 2
    request = next(r for r in hook.requests if r.prompt == "a")
    assert (request.attempt, request.try_number, request.messages) == (1, 1, None)
    assert request.strategy is strategy
    assert request.generation_config is None  # not a ModelStrategy
    assert all(success for _, _, success in hook.settles)
    assert all(_subset(usage) == _USAGE for _, usage, _ in hook.settles)


@pytest.mark.asyncio
async def test_messages_item_sets_messages_not_prompt():
    hook = Recorder()
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    strategy = FakeStrategy("ok", messages_handler=lambda m: "ok", token_usage=_USAGE)
    async with ParallelBatchProcessor[None, str, None](config=_config(hook)) as processor:
        await processor.add_work(LLMWorkItem(item_id="m", strategy=strategy, messages=messages))
        batch = await processor.process_all()

    assert batch.succeeded == 1
    (request,) = hook.requests
    assert request.prompt == ""
    assert request.messages is not None
    assert [dict(m) for m in request.messages] == messages
    with pytest.raises(TypeError):
        request.messages[0]["content"] = "changed"  # type: ignore[index]
    hook.assert_settled_exactly_once()
    assert _subset(hook.settles[0][1]) == _USAGE


class _Growing(ModelStrategy[str]):
    def attempt_generation_config(
        self, attempt: int, state: RetryState | None
    ) -> dict[str, Any] | None:
        return {"max_tokens": 100 * attempt}


@pytest.mark.asyncio
async def test_model_strategy_request_carries_the_effective_generation_config(fast_retry):
    model = AsyncMock()
    model.generate.side_effect = [
        ConnectionError("flaky"),
        LLMResponse(text="ok", input_tokens=1, output_tokens=1, total_tokens=2),
    ]
    hook = Recorder()
    strategy = _Growing(model, generation_config={"max_tokens": 1, "top_p": 0.5})
    batch = await process_prompts(strategy, ["p"], config=_config(hook, retry=fast_retry))

    assert batch.succeeded == 1
    configs = [dict(r.generation_config or {}) for r in hook.requests]
    assert configs == [{"max_tokens": 100, "top_p": 0.5}, {"max_tokens": 200, "top_p": 0.5}]
    # The request's config is what was actually sent.
    sent = [call.kwargs["config"] for call in model.generate.await_args_list]
    assert sent == configs
    assert [(r.attempt, r.try_number) for r in hook.requests] == [(1, 1), (2, 2)]
    hook.assert_settled_exactly_once()
    assert [success for _, _, success in hook.settles] == [False, True]


# --- Exactly-once settle -----------------------------------------------------


@pytest.mark.asyncio
async def test_provider_failure_settles_with_attached_usage(fast_retry):
    failure = ConnectionError("billed then dropped")
    attach_token_usage(failure, _USAGE)
    hook = Recorder()
    strategy = FakeStrategy("ok", failure_schedule=[failure], token_usage=_USAGE)
    batch = await process_prompts(strategy, ["a"], config=_config(hook, retry=fast_retry))

    assert batch.succeeded == 1 and strategy.calls == 2
    hook.assert_settled_exactly_once()
    assert [(_subset(u), s) for _, u, s in hook.settles] == [(_USAGE, False), (_USAGE, True)]


@pytest.mark.asyncio
async def test_failure_without_usage_settles_with_none(fast_retry):
    hook = Recorder()
    strategy = FakeStrategy("ok", failure_schedule=[ConnectionError("x")] * 3)
    batch = await process_prompts(strategy, ["a"], config=_config(hook, retry=fast_retry))

    assert not batch.results[0].success
    hook.assert_settled_exactly_once()
    assert hook.settles and all(u is None and not s for _, u, s in hook.settles)
    assert len(hook.settles) == 3


@pytest.mark.asyncio
async def test_rate_limit_retry_is_a_new_try_with_the_same_attempt(fast_retry, fast_rate_limit):
    hook = Recorder()
    strategy = FakeStrategy("ok", failure_schedule=[FakeRateLimitError()])
    batch = await process_prompts(
        strategy, ["a"], config=_config(hook, retry=fast_retry, rate_limit=fast_rate_limit)
    )
    assert batch.succeeded == 1
    assert [(r.attempt, r.try_number) for r in hook.requests] == [(1, 1), (1, 2)]
    hook.assert_settled_exactly_once()


@pytest.mark.asyncio
async def test_framework_timeout_settles_each_try():
    hook = Recorder()
    retry = RetryConfig(max_attempts=2, initial_wait=0.001, max_wait=0.001, jitter=False)
    strategy = FakeStrategy("ok", latency=5)
    batch = await process_prompts(
        strategy, ["a"], config=_config(hook, retry=retry, attempt_timeout=0.05)
    )
    assert batch.results[0].error_category == "framework_timeout"
    hook.assert_settled_exactly_once()
    assert len(hook.settles) == 2
    assert all(u is None and not s for _, u, s in hook.settles)


@pytest.mark.asyncio
async def test_cancelling_the_run_settles_calls_in_flight():
    hook = Recorder()
    strategy = FakeStrategy("ok", latency=30)
    task = asyncio.create_task(
        process_prompts(strategy, ["a", "b", "c"], config=_config(hook, max_workers=3))
    )
    for _ in range(200):
        if len(hook.requests) == 3:
            break
        await asyncio.sleep(0.005)
    assert len(hook.requests) == 3
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    hook.assert_settled_exactly_once()
    assert all(u is None and not s for _, u, s in hook.settles)


@pytest.mark.asyncio
async def test_cancel_active_abort_settles_calls_in_flight():
    hook = Recorder()
    strategy = FakeStrategy("ok", latency=30)
    config = _config(
        hook,
        max_workers=3,
        guardrails={"batch_timeout": 0.1, "abort_mode": AbortMode.CANCEL_ACTIVE},
    )
    batch = await process_prompts(strategy, ["a", "b", "c", "d"], config=config)

    assert batch.termination.kind == "batch_timeout"
    assert len(hook.requests) == 3
    hook.assert_settled_exactly_once()
    assert all(u is None and not s for _, u, s in hook.settles)


@pytest.mark.asyncio
async def test_drain_active_refusal_lets_running_calls_settle_with_usage():
    hook = Recorder(refuse_at=4)
    strategy = FakeStrategy("ok", latency=0.05, token_usage=_USAGE)
    events = _Events()
    async with ParallelBatchProcessor[None, str, None](
        config=_config(hook, max_workers=4), observers=[events]
    ) as processor:
        for i in range(6):
            await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="p"))
        batch = await processor.process_all()

    assert batch.termination.kind == "admission_refused"
    assert batch.succeeded == 3 and strategy.calls == 3
    hook.assert_settled_exactly_once()
    assert all(_subset(u) == _USAGE and s for _, u, s in hook.settles)
    assert len(events.aborted()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["deadline", "callback", "cancel"])
async def test_pre_start_failure_settles_with_zero_usage_and_refunds_the_call(failure):
    hook = Recorder()
    controller = AbortController(AbortMode.DRAIN_ACTIVE, max_provider_calls=1, admission=hook)
    ran: list[bool] = []

    async def body() -> str:
        ran.append(True)
        return "ok"

    def start() -> None:
        if failure == "callback":
            raise RuntimeError("start failure")
        if failure == "cancel":
            raise asyncio.CancelledError("start cancel")

    request = AttemptRequest(
        item_id="x",
        attempt=1,
        try_number=1,
        strategy=FakeStrategy("ok"),
        prompt="p",
        messages=None,
        generation_config=None,
    )
    slot = AdmissionSlot("x", lambda: request)
    deadline = time.perf_counter() - 1 if failure == "deadline" else None
    with pytest.raises(BaseException):  # noqa: B017 - each case raises a different type
        await await_with_guardrails(
            body(),
            item_deadline=deadline,
            item_id="x",
            abort_controller=controller,
            active_provider=True,
            on_start=start,
            admission=slot,
        )
    assert not ran
    assert hook.requests == [request]
    assert hook.settles == [(("ticket", 0), _ZERO, False)]
    assert controller.provider_calls_started == controller._active_provider_calls == 0
    # A second settle from the caller's finally is a no-op.
    controller.settle_attempt(slot, None, False)
    assert len(hook.settles) == 1


@pytest.mark.asyncio
async def test_cooldown_pause_at_start_settles_and_asks_again():
    """The executor's pause path: a cooldown begins between admit and the call's start."""
    strategy = FakeStrategy("ok", token_usage=_USAGE)

    class PauseOnFirstAdmit(Recorder):
        processor: Any = None

        def admit(self, request: AttemptRequest) -> object:
            ticket = super().admit(request)
            if len(self.requests) == 1:
                cooldown = self.processor._admission_registry.resolve(strategy).cooldown
                cooldown._rate_limit_event.clear()  # as if another worker hit a 429
                asyncio.get_running_loop().call_later(0.02, cooldown._rate_limit_event.set)
            return ticket

    hook = PauseOnFirstAdmit()
    async with ParallelBatchProcessor[None, str, None](config=_config(hook)) as processor:
        hook.processor = processor
        await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="p"))
        batch = await processor.process_all()
        stats = await processor.get_stats()

    assert batch.succeeded == 1 and strategy.calls == 1
    assert stats["provider_calls_started"] == 1  # the paused start gave its call back
    # Same physical try asked twice; the unstarted ticket settled with zero usage first.
    assert [(r.attempt, r.try_number) for r in hook.requests] == [(1, 1), (1, 1)]
    assert [(t, _subset(u), ok) for t, u, ok in hook.settles] == [
        (("ticket", 0), _subset(_ZERO), False),
        (("ticket", 1), _USAGE, True),
    ]
    hook.assert_settled_exactly_once()


# --- Refusal -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_refusal_stops_the_run_with_admission_refused():
    hook = Recorder(refuse_at=3)
    strategy = FakeStrategy("ok", token_usage=_USAGE)
    events = _Events()
    config = _config(hook, max_workers=1, guardrails={"max_provider_calls": 10})
    async with ParallelBatchProcessor[None, str, None](
        config=config, observers=[events]
    ) as processor:
        for i in range(5):
            await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="p"))
        batch = await processor.process_all()
        stats = await processor.get_stats()

    assert strategy.calls == 2
    assert stats["provider_calls_started"] == 2  # the refused attempt isn't a call
    termination = batch.termination
    assert termination.kind == "admission_refused"
    assert termination.reason == "refused i2"
    assert termination.error_category == "batch_admission_refused"
    assert termination.triggering_item_id == "i2"
    by_id = {r.item_id: r for r in batch.results}
    assert by_id["i0"].success and by_id["i1"].success
    refused = by_id["i2"]
    assert refused.error_category == "batch_admission_refused"
    assert isinstance(refused.exception, AdmissionRefused)
    assert refused.timing is not None and len(refused.timing.attempts) == 1  # not retried
    for result in batch.results:
        if not result.success:
            assert result.error_category == "batch_admission_refused"
    assert len(hook.requests) == 3  # no attempt after the refusal
    hook.assert_settled_exactly_once()
    aborted = events.aborted()
    assert len(aborted) == 1
    assert aborted[0]["kind"] == "admission_refused"
    assert aborted[0]["triggering_item_id"] == "i2"
    assert stats["aborted"] == batch.failed


@pytest.mark.asyncio
async def test_admit_raising_another_exception_fails_closed(caplog):
    hook = Recorder(refuse_at=1, admit_error=KeyError("price table"))
    strategy = FakeStrategy("ok")
    with caplog.at_level(logging.ERROR, logger="llm_minder"):
        batch = await process_prompts(strategy, ["a", "b"], config=_config(hook, max_workers=1))

    assert strategy.calls == 0
    assert batch.termination.kind == "admission_refused"
    assert batch.termination.reason == "Attempt admission hook raised KeyError: 'price table'"
    assert {r.error_category for r in batch.results} == {"batch_admission_refused"}
    refused = next(r for r in batch.results if r.item_id == "item_0")
    assert isinstance(refused.exception, AdmissionRefused)
    assert "AttemptAdmission.admit() raised for item_0" in caplog.text


@pytest.mark.asyncio
async def test_admit_returning_an_awaitable_fails_closed():
    class Sneaky:
        def admit(self, request):
            async def ticket() -> object:
                return object()

            return ticket()

        def settle(self, ticket, usage, success):
            raise AssertionError("never admitted")

    strategy = FakeStrategy("ok")
    batch = await process_prompts(strategy, ["a"], config=_config(Sneaky()))
    assert strategy.calls == 0
    assert batch.termination.kind == "admission_refused"
    assert "must be synchronous" in (batch.termination.reason or "")


@pytest.mark.asyncio
async def test_empty_refusal_message_gets_a_default_reason():
    class Refuse:
        def admit(self, request):
            raise AdmissionRefused("")

        def settle(self, ticket, usage, success):
            pass

    batch = await process_prompts(FakeStrategy("ok"), ["a"], config=_config(Refuse()))
    assert batch.termination.reason == "Attempt admission refused"


@pytest.mark.asyncio
async def test_settle_raising_fails_closed(caplog):
    hook = Recorder(settle_error_at=1)
    strategy = FakeStrategy("ok", token_usage=_USAGE)
    events = _Events()
    with caplog.at_level(logging.ERROR, logger="llm_minder"):
        async with ParallelBatchProcessor[None, str, None](
            config=_config(hook, max_workers=1), observers=[events]
        ) as processor:
            for i in range(3):
                await processor.add_work(
                    LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="p")
                )
            batch = await processor.process_all()

    assert strategy.calls == 1
    by_id = {r.item_id: r for r in batch.results}
    assert by_id["i0"].success  # the settled attempt keeps its outcome
    assert batch.termination.kind == "admission_refused"
    assert (
        batch.termination.reason == "Attempt admission settle raised RuntimeError: ledger offline"
    )
    assert batch.termination.triggering_item_id == "i0"
    assert all(
        r.error_category == "batch_admission_refused" for r in batch.results if not r.success
    )
    assert len(hook.requests) == 1
    hook.assert_settled_exactly_once()
    assert "AttemptAdmission.settle() raised for i0" in caplog.text
    assert len(events.aborted()) == 1


@pytest.mark.asyncio
async def test_call_cap_behaviour_is_unchanged_with_a_hook():
    hook = Recorder()
    strategy = FakeStrategy("ok", failure_schedule=[ConnectionError("x"), ConnectionError("y")])
    retry = RetryConfig(max_attempts=5, initial_wait=0.001, max_wait=0.001, jitter=False)
    config = _config(hook, retry=retry, guardrails={"max_provider_calls": 2})
    batch = await process_prompts(strategy, ["a"], config=config)

    assert strategy.calls == 2
    assert batch.termination.kind == "budget_exceeded"
    assert batch.termination.reason == "Provider call cap reached: 2 of 2 calls started"
    assert batch.results[0].error_category == "batch_budget_exceeded"
    # The cap refuses before admit: the hook only saw the two counted calls.
    assert len(hook.requests) == 2
    hook.assert_settled_exactly_once()


# --- Artifacts and serialization ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
@pytest.mark.parametrize("policy", [ResumePolicy.REUSE_SUCCESSES, ResumePolicy.REUSE_ALL])
async def test_refused_records_are_audited_and_never_replayed(tmp_path: Path, store_type, policy):
    path = tmp_path / "run.store"
    identity = ArtifactIdentity(provider="fake", model="fake")
    prompts = ["a", "b", "c", "d"]
    first = await process_prompts(
        FakeStrategy("ok"),
        prompts,
        config=_config(Recorder(refuse_at=3), max_workers=1),
        artifact_store=store_type(path, identity=identity),
    )
    assert first.termination.kind == "admission_refused"
    refused_ids = {r.item_id for r in first.results if not r.success}
    assert refused_ids and all(
        r.error_category == "batch_admission_refused" for r in first.results if not r.success
    )

    reader = store_type(path, identity=identity)
    try:
        recorded = [r async for r in reader.iter_results()]
    finally:
        await reader.close()
    assert {r.item_id for r in recorded if not r.success} == refused_ids
    assert all(r.error_category == "batch_admission_refused" for r in recorded if not r.success)

    strategy = FakeStrategy("ok")
    second = await process_prompts(
        strategy,
        prompts,
        config=ProcessorConfig(max_workers=1),
        artifact_store=store_type(path, identity=identity),
        resume=policy,
    )
    assert second.succeeded == 4 and second.termination.kind == "completed"
    replayed = {r.item_id for r in second.results if r.replayed_from_artifact}
    assert replayed == {"item_0", "item_1"}
    assert strategy.calls == 4 - len(replayed)


@pytest.mark.asyncio
async def test_admission_refused_termination_round_trips():
    batch = await process_prompts(
        FakeStrategy("ok"), ["a", "b"], config=_config(Recorder(refuse_at=1), max_workers=1)
    )
    restored = batch_result_from_json(batch_result_to_json(batch))
    assert restored.termination == batch.termination
    assert restored.termination.kind == "admission_refused"
    assert {r.error_category for r in restored.results} == {"batch_admission_refused"}


# --- SDK retry warning -------------------------------------------------------


def test_sdk_retry_warning_reads_the_clients_own_setting(caplog):
    openai = pytest.importorskip("openai")
    from llm_minder import OpenAIModel, OpenAIStrategy

    retrying = OpenAIModel("gpt-x", openai.AsyncOpenAI(api_key="test", max_retries=2))
    quiet = OpenAIModel("gpt-x", openai.AsyncOpenAI(api_key="test", max_retries=0))
    assert _sdk_retry_setting(retrying) == "max_retries=2"
    assert _sdk_retry_setting(quiet) is None

    warned: set[int] = set()
    with caplog.at_level(logging.WARNING, logger="llm_minder"):
        for _ in range(3):
            warn_if_sdk_retries(OpenAIStrategy(retrying), warned)
            warn_if_sdk_retries(OpenAIStrategy(quiet), warned)
    messages = [r.getMessage() for r in caplog.records if "retries on its own" in r.getMessage()]
    assert len(messages) == 1 and "max_retries=2" in messages[0]


def test_sdk_retry_setting_from_api_key_defaults_are_quiet():
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    from llm_minder import AnthropicModel, OpenAIModel

    assert _sdk_retry_setting(OpenAIModel.from_api_key("gpt-x", api_key="test")) is None
    assert _sdk_retry_setting(AnthropicModel.from_api_key("claude-x", api_key="test")) is None
    overridden = AnthropicModel.from_api_key("claude-x", api_key="test", max_retries=3)
    assert _sdk_retry_setting(overridden) == "max_retries=3"


def test_sdk_retry_setting_for_gemini_retry_options():
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    from llm_minder import GeminiModel

    plain = GeminiModel("gemini-x", genai.Client(api_key="test"))
    assert _sdk_retry_setting(plain) is None
    retrying = GeminiModel(
        "gemini-x",
        genai.Client(
            api_key="test",
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3)),
        ),
    )
    assert _sdk_retry_setting(retrying) == "retry_options.attempts=3"
    once = GeminiModel(
        "gemini-x",
        genai.Client(
            api_key="test",
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        ),
    )
    assert _sdk_retry_setting(once) is None


def test_sdk_retry_setting_ignores_custom_models():
    class Custom:
        def __init__(self) -> None:
            self._client = type("C", (), {"max_retries": 5})()

    assert _sdk_retry_setting(Custom()) is None


# --- The documented example ---------------------------------------------------


def _doc_example_namespace() -> dict[str, Any]:
    text = (Path(__file__).parents[1] / "docs" / "guardrails.md").read_text()
    section = text.split("### Example: a hard spend cap", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S)
    assert code is not None
    namespace: dict[str, Any] = {}
    exec(compile(code.group(1), "guardrails.md", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.asyncio
async def test_documented_hard_spend_cap_never_passes_its_limit():
    namespace = _doc_example_namespace()
    cap_cls, prices_cls = namespace["HardSpendCap"], namespace["Prices"]
    usage = {"input_tokens": 1_000, "output_tokens": 100, "total_tokens": 1_100}
    cap = cap_cls(
        0.02, prices_cls(input_per_mtok=3.0, output_per_mtok=15.0), default_max_output=200
    )
    strategy = FakeStrategy("ok", token_usage=usage)
    batch = await process_prompts(
        strategy,
        ["x" * 100] * 50,
        config=ProcessorConfig(max_workers=5, guardrails=GuardrailConfig(admission=cap)),
    )
    assert batch.termination.kind == "admission_refused"
    assert "Spend cap $0.02" in (batch.termination.reason or "")
    assert 0 < strategy.calls < 50
    assert cap.reserved == pytest.approx(0.0, abs=1e-12)
    assert cap.spent <= cap.limit
    per_call = (1_000 * 3.0 + 100 * 15.0) / 1_000_000
    assert cap.spent == pytest.approx(strategy.calls * per_call)


def test_documented_actual_cost_prices_anthropic_cache_reads_and_writes():
    namespace = _doc_example_namespace()
    cap = namespace["cap"]  # the documented Anthropic-priced instance
    usage = {
        "input_tokens": 10_000,
        "cached_input_tokens": 6_000,
        "cache_write_input_tokens": 3_000,
        "output_tokens": 500,
        "total_tokens": 10_500,
    }
    expected = ((1_000 + 6_000 * 0.10 + 3_000 * 1.25) * 3.0 + 500 * 15.0) / 1_000_000
    assert cap.actual_cost(usage) == pytest.approx(expected)
    # The reservation assumes every input token is a cache write.
    request = AttemptRequest(
        item_id="x",
        attempt=1,
        try_number=1,
        strategy=FakeStrategy("ok"),
        prompt="é" * 10,
        messages=None,
        generation_config={"max_tokens": 1_000},
    )
    worst = cap.admit(request)
    assert worst == pytest.approx(((20 + 2_000) * 3.0 * 1.25 + 1_000 * 15.0) / 1_000_000)
    cap.settle(worst, None, False)  # unknown usage keeps the worst case
    assert cap.spent == pytest.approx(worst) and cap.reserved == pytest.approx(0.0)
    # An attempt that never started settles with zero usage and releases exactly.
    again = cap.admit(request)
    cap.settle(again, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, False)
    assert cap.spent == pytest.approx(worst) and cap.reserved == pytest.approx(0.0)
