"""OpenAIModel."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._internal.responses_translation import (
    merge_request_configs,
    translate_input,
    translate_request_config,
)
from ..base import LLMResponse
from ._openai_compatible import OpenAICompatibleModel
from ._responses import _ResponsesSurface
from ._shared import (
    MetadataExtractor,
    _has_system_message,
    _raise_empty_response,
    _reason_text,
    _run_extractors,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


class OpenAIModel(_ResponsesSurface, OpenAICompatibleModel):
    """LLM model backed by OpenAI, on the Responses API by default.

    Uses the OpenAI SDK's default base URL (``https://api.openai.com/v1``).
    OpenAI's automatic prompt cache surfaces in ``cached_input_tokens`` for
    prompts longer than ~1024 tokens.

    **API surface.** Since v0.27 requests use the Responses API
    (``api_surface="responses"``) with ``store=False`` unless the request sets
    ``store``. Chat-style request fields are translated: ``max_tokens`` →
    ``max_output_tokens``, ``response_format`` → ``text.format``,
    ``reasoning_effort`` → ``reasoning.effort``, ``logprobs`` → ``include``,
    Chat ``tools``/``tool_choice`` → Responses function tools, and Chat message
    lists (including assistant ``tool_calls`` and ``tool`` replies) → Responses
    input items. Fields with no Responses equivalent (``n``, ``stop``,
    ``seed``, penalties, ``logit_bias``, legacy ``functions``) and ``stream`` /
    ``background`` raise ``ValueError`` before any request. Pass
    ``api_surface="chat_completions"`` for the previous Chat Completions
    behavior. For other OpenAI-compatible servers (vLLM, proxies, Azure-style
    endpoints) use :class:`OpenAICompatibleModel` instead.

    A completed Responses call that returns only function calls succeeds with
    ``text=""``, ``finish_reason="tool_calls"`` and the calls in
    ``metadata["tool_calls"]``. An incomplete call with text succeeds with
    ``finish_reason="length"`` (token limit) or ``"content_filter"``; without
    text it raises :class:`EmptyResponseError`, as does a refusal.

    Example:
        >>> model = OpenAIModel.from_api_key("gpt-6-luna", api_key="sk-...")
        >>> response = await model.generate("Hello!")
        >>> print(response.text, response.cached_input_tokens)

    Added in v0.9.0.
    """

    _default_base_url: str | None = None
    _install_extras: str = "openai"
    _responses_label = "OpenAI"
    _responses_chat_aliases = True

    def __init__(
        self,
        model: str,
        client: "AsyncOpenAI",
        *,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        api_surface: str = "responses",
        prompt_cache_key: str | None = None,
        cache_group: str | None = None,
    ):
        """See :class:`OpenAICompatibleModel`; adds ``api_surface``
        (``"responses"``, the default, or ``"chat_completions"``)."""
        if api_surface not in {"chat_completions", "responses"}:
            raise ValueError(
                f"api_surface must be 'responses' or 'chat_completions' (got {api_surface!r})."
            )
        if api_surface == "responses":
            try:
                create_response = getattr(getattr(client, "responses", None), "create", None)
            except Exception:
                create_response = None
            if not callable(create_response):
                raise ValueError(
                    "OpenAIModel uses the Responses API by default, which needs a client with "
                    "callable responses.create (openai>=1.66.2). Upgrade the SDK, or pass "
                    'api_surface="chat_completions".'
                ) from None
        self.api_surface = api_surface
        super().__init__(
            model,
            client,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_body=extra_body,
            metadata_extractors=metadata_extractors,
            prompt_cache_key=prompt_cache_key,
            cache_group=cache_group,
        )

    @classmethod
    def from_api_key(  # type: ignore[override]
        cls,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        json_mode: bool = False,
        max_connections: int | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        api_surface: str = "responses",
        prompt_cache_key: str | None = None,
        cache_group: str | None = None,
        **client_kwargs: Any,
    ) -> "OpenAIModel":
        """Build an OpenAIModel with an owned client; adds ``api_surface``.

        ``json_mode`` adds ``response_format={"type": "json_object"}``, which the
        Responses surface translates to ``text.format``.
        """
        return super().from_api_key(
            model,
            api_key,
            base_url=base_url,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_body=extra_body,
            json_mode=json_mode,
            max_connections=max_connections,
            metadata_extractors=metadata_extractors,
            prompt_cache_key=prompt_cache_key,
            cache_group=cache_group,
            _instance_kwargs={"api_surface": api_surface},
            **client_kwargs,
        )

    @property
    def artifact_identity_extra(self) -> dict[str, Any]:
        """Replay identity: empty on Chat Completions, so v0.26 checkpoints match."""
        return {} if self.api_surface == "chat_completions" else {"api_surface": self.api_surface}

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = None,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call the selected OpenAI API surface and normalize the response."""
        if self.api_surface == "chat_completions":
            return await super().generate(
                prompt,
                temperature=temperature,
                system_instruction=system_instruction,
                config=config,
            )
        return await self._generate_responses(
            prompt,
            temperature=temperature,
            system_instruction=system_instruction,
            config=config,
        )

    def _extract_metadata(self, response: Any) -> dict[str, Any] | None:
        """Chat metadata plus ``reasoning_tokens`` (the key Gemini also emits)."""
        metadata = super()._extract_metadata(response)
        details = getattr(getattr(response, "usage", None), "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            metadata = dict(metadata or {})
            metadata["reasoning_tokens"] = reasoning_tokens
        return metadata

    def _responses_request(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None,
        system_instruction: str | None,
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Translate Chat-style fields to Responses; ``store=False`` by default."""
        # Per-call fields beat defaults under any alias (max_tokens vs
        # max_output_tokens, response_format vs text.format, ...).
        merged = merge_request_configs(
            self._default_extra_body, config, chat_aliases=self._responses_chat_aliases
        )
        request_config = translate_request_config(merged)
        request_config.setdefault("store", False)
        input_value = translate_input(prompt)
        si = system_instruction or self._default_system_instruction
        has_system = isinstance(input_value, list) and _has_system_message(input_value)

        call_kwargs: dict[str, Any] = {"model": self._model, "input": input_value}
        if si is not None and not has_system:
            call_kwargs["instructions"] = si
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        # Keywords the 1.66 SDK accepts; newer fields travel in extra_body.
        response_parameters = {
            "include",
            "max_output_tokens",
            "parallel_tool_calls",
            "reasoning",
            "store",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "user",
        }
        for key in response_parameters:
            if key in request_config:
                call_kwargs[key] = request_config.pop(key)
        if request_config:
            call_kwargs["extra_body"] = request_config
        if self._default_extra_headers:
            call_kwargs["extra_headers"] = self._default_extra_headers
        return call_kwargs

    def _responses_result(self, response: Any, tokens: tuple[int, int, int, int]) -> LLMResponse:
        """OpenAI policy: keep partial text; allow completed function-call-only output.

        Success is decided from the provider's own output, before metadata
        extractors run, so an extractor can't turn a failure into a success or
        the reverse.
        """
        text = self._responses_text(response)
        output = _openai_output(response)
        status = getattr(response, "status", None)
        cache_writes = self._extract_cache_write_tokens(response)
        if not text:
            if output.refusals:
                _raise_empty_response(
                    "OpenAI refused the request.",
                    tokens,
                    cache_writes,
                    provider_reason=_reason_text(status),
                    reason="refusal",
                )
            if status == "incomplete":
                details = getattr(response, "incomplete_details", None)
                reason = getattr(details, "reason", None)
                _raise_empty_response(
                    f"Incomplete OpenAI Responses output (reason={reason or 'unknown'}).",
                    tokens,
                    cache_writes,
                    provider_reason=_reason_text(reason),
                )
            if not (status == "completed" and output.tool_calls):
                _raise_empty_response(
                    "No text returned from OpenAI Responses API.",
                    tokens,
                    cache_writes,
                    provider_reason=_reason_text(status),
                )
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            metadata=self._responses_metadata(response, output),
            raw=response,
            cache_write_input_tokens=cache_writes,
        )

    def _responses_metadata(
        self, response: Any, output: "_OpenAIOutput | None" = None
    ) -> dict[str, Any] | None:
        """Chat-compatible ``finish_reason`` plus the reserved provider-output keys."""
        if output is None:
            output = _openai_output(response)
        metadata: dict[str, Any] = {"api_surface": "responses"}
        request_id = getattr(response, "id", None)
        if isinstance(request_id, str) and request_id:
            metadata["provider_request_id"] = request_id
        model = getattr(response, "model", None)
        if isinstance(model, str) and model:
            metadata["model"] = model
        if output.reasoning or output.summaries:
            metadata["reasoning"] = (
                "".join(output.reasoning) if output.reasoning else "\n\n".join(output.summaries)
            )
        if output.tool_calls:
            metadata["tool_calls"] = [dict(call) for call in output.tool_calls]
        if output.refusals:
            metadata["refusal"] = "".join(output.refusals)
        if output.logprobs:
            metadata["logprobs"] = list(output.logprobs)

        status = getattr(response, "status", None)
        if isinstance(status, str) and status:
            metadata["response_status"] = status
            metadata["finish_reason"] = _openai_finish_reason(
                response, status, bool(output.tool_calls)
            )
        usage = getattr(response, "usage", None)
        details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = (
            details.get("reasoning_tokens")
            if isinstance(details, Mapping)
            else getattr(details, "reasoning_tokens", None)
        )
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            metadata["reasoning_tokens"] = reasoning_tokens
        return _run_extractors(response, self._metadata_extractors, metadata)


@dataclass(frozen=True)
class _OpenAIOutput:
    """Provider facts read from a Responses ``output`` list."""

    reasoning: tuple[str, ...]
    summaries: tuple[str, ...]
    tool_calls: tuple[dict[str, Any], ...]
    refusals: tuple[str, ...]
    logprobs: tuple[dict[str, Any], ...]


def _openai_output(response: Any) -> _OpenAIOutput:
    reasoning: list[str] = []
    summaries: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    refusals: list[str] = []
    logprobs: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or ():
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            for content in getattr(item, "content", None) or ():
                value = getattr(content, "text", None)
                if isinstance(value, str) and value:
                    reasoning.append(value)
            for summary in getattr(item, "summary", None) or ():
                value = getattr(summary, "text", None)
                if isinstance(value, str) and value:
                    summaries.append(value)
        elif item_type == "function_call":
            name = getattr(item, "name", None)
            if isinstance(name, str) and name:
                call_id = getattr(item, "call_id", None)
                arguments = getattr(item, "arguments", None)
                tool_calls.append(
                    {
                        "id": call_id if isinstance(call_id, str) else None,
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else "",
                    }
                )
        elif item_type == "message":
            for content in getattr(item, "content", None) or ():
                content_type = getattr(content, "type", None)
                if content_type == "refusal":
                    value = getattr(content, "refusal", None)
                    if isinstance(value, str) and value:
                        refusals.append(value)
                elif content_type == "output_text":
                    for entry in getattr(content, "logprobs", None) or ():
                        dumped = entry.model_dump() if hasattr(entry, "model_dump") else entry
                        if isinstance(dumped, dict):
                            logprobs.append(dumped)
    return _OpenAIOutput(
        tuple(reasoning), tuple(summaries), tuple(tool_calls), tuple(refusals), tuple(logprobs)
    )


def _openai_finish_reason(response: Any, status: str, has_tool_calls: bool) -> str:
    """Map a Responses status to the Chat Completions ``finish_reason`` vocabulary."""
    if status == "completed":
        return "tool_calls" if has_tool_calls else "stop"
    if status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        if reason == "max_output_tokens":
            return "length"
        if reason == "content_filter":
            return "content_filter"
    return status
