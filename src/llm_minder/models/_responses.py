"""The Responses API surface shared by OpenAIModel and DeepSeekModel."""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from .._internal.responses_translation import (
    merge_request_configs,
)
from ..base import LLMResponse
from ..strategies.errors import (
    ProviderResponseError,
    StructuredOutputSchemaError,
)
from ._shared import (
    MetadataExtractor,
    _cache_group,
    _diagnostics_miss_reason,
    _has_system_message,
    _openai_cache_write_tokens,
    _raise_empty_response,
    _reason_text,
    _run_extractors,
    _system_text,
    _usage_dict,
    _with_cache_group,
)

if TYPE_CHECKING:
    pass

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


class _ResponsesSurface:
    """Shared Responses API transport and extraction for OpenAI-compatible models.

    Used by :class:`DeepSeekModel` (``api_surface="responses"``) and
    :class:`OpenAIModel` (its default). The call, usage and text extraction are
    shared; request building (:meth:`_responses_request`), result policy
    (:meth:`_responses_result`) and metadata (:meth:`_responses_metadata`) are
    per-provider hooks, and the defaults here are DeepSeek's behavior.
    """

    _responses_label = "DeepSeek"
    response_schema: dict[str, Any] | None = None
    response_schema_name: str | None = None
    response_schema_identity: str | None = None
    response_schema_hash: str | None = None
    # Provided by OpenAICompatibleModel.
    _client: Any
    _model: str
    _default_system_instruction: str | None
    _default_extra_body: dict[str, Any] | None
    _default_extra_headers: dict[str, str] | None
    _metadata_extractors: list[MetadataExtractor] | None
    _client_used: bool
    api_surface: str
    # Whether the request translator treats response_format/reasoning_effort as
    # aliases of text.format/reasoning.effort (OpenAI yes, DeepSeek no).
    _responses_chat_aliases = False

    def _merge_call_config(
        self, base: Mapping[str, Any] | None, override: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Merge a strategy's ``generation_config`` with one attempt's override.

        Used by ``ModelStrategy.execute()``. On the Responses surface a field
        replaces every alias of itself (see ``merge_request_configs``); on Chat
        Completions it stays a shallow top-level merge.
        """
        if self.api_surface != "responses":
            return {**(base or {}), **override}
        return merge_request_configs(base, override, chat_aliases=self._responses_chat_aliases)

    async def _generate_responses(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None,
        system_instruction: str | None,
        config: dict[str, Any] | None,
    ) -> LLMResponse:
        # prepare() comes from OpenAICompatibleModel, the other base.
        await cast(Any, self).prepare()
        self._client_used = True
        call_kwargs = self._responses_request(
            prompt,
            temperature=temperature,
            system_instruction=system_instruction,
            config=config,
        )
        extra_body = dict(call_kwargs.get("extra_body") or {})
        # extra_body is merged into the request last, so its instructions win on the
        # wire. A null there clears only the top-level field, not a system message
        # in the input, so fall back to that.
        instructions = (
            extra_body["instructions"]
            if "instructions" in extra_body
            else call_kwargs.get("instructions")
        ) or _system_text(call_kwargs.get("input"))
        label = cast(Any, self)._apply_cache_options(extra_body, instructions)
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        else:
            call_kwargs.pop("extra_body", None)
        try:
            response = await self._client.responses.create(**call_kwargs)
        except Exception as exc:
            if self.response_schema is not None and _looks_like_schema_rejection(exc):
                raise StructuredOutputSchemaError(
                    f"{self._responses_label} rejected response schema {self.response_schema_name!r} "
                    f"({self.response_schema_hash}): {exc}"
                ) from exc
            raise

        tokens = self._extract_responses_tokens(response)
        status = getattr(response, "status", None)
        provider_error = getattr(response, "error", None)
        if provider_error is not None or status == "failed":
            code, message = _response_error_text(provider_error)
            error_text = f"{code or ''} {message}".lower()
            if self.response_schema is not None and "schema" in error_text:
                schema_error = StructuredOutputSchemaError(
                    f"{self._responses_label} rejected response schema "
                    f"{self.response_schema_name!r}: {message}",
                    token_usage=self._tokens_dict(tokens, _openai_cache_write_tokens(response)),
                )
                raise schema_error
            response_error = ProviderResponseError(
                f"{self._responses_label} Responses API failed ({code or 'unknown'}): {message}",
                provider_error=provider_error,
                token_usage=self._tokens_dict(tokens, _openai_cache_write_tokens(response)),
            )
            raise response_error
        return _with_cache_group(
            self._responses_result(response, tokens),
            label or _cache_group(extra_body.get("prompt_cache_key"), instructions),
            _diagnostics_miss_reason(response),
        )

    def _responses_request(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None,
        system_instruction: str | None,
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build ``responses.create`` kwargs (DeepSeek's request policy)."""
        input_value: str | list[Any] = prompt if isinstance(prompt, str) else list(prompt)
        si = system_instruction or self._default_system_instruction
        has_system = isinstance(input_value, list) and _has_system_message(input_value)

        # Per-call fields beat defaults under any alias, so a per-call
        # max_tokens replaces a default max_output_tokens. Every token alias the
        # merge drops must then map to the native field, or the bound is lost.
        request_config = merge_request_configs(self._default_extra_body, config, chat_aliases=False)
        for alias in ("max_completion_tokens", "max_tokens"):
            if alias in request_config:
                request_config.setdefault("max_output_tokens", request_config[alias])
                request_config.pop(alias)

        call_kwargs: dict[str, Any] = {"model": self._model, "input": input_value}
        if si is not None and not has_system:
            call_kwargs["instructions"] = si
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        # Keep SDK keywords compatible with OpenAI 1.66.0; newer/provider
        # fields (including max_tool_calls and top_logprobs) use extra_body.
        response_parameters = {
            "max_output_tokens",
            "parallel_tool_calls",
            "reasoning",
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
        """Turn a non-failed response into a result (DeepSeek's policy: any
        incomplete or textless response raises)."""
        cache_writes = _openai_cache_write_tokens(response)
        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None)
            _raise_empty_response(
                f"Incomplete {self._responses_label} Responses output (reason={reason or 'unknown'}).",
                tokens,
                cache_writes,
                provider_reason=_reason_text(reason),
            )

        text = self._responses_text(response)
        if not text:
            _raise_empty_response(
                f"No text returned from {self._responses_label} Responses API.",
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
            metadata=self._responses_metadata(response),
            raw=response,
            cache_write_input_tokens=cache_writes,
        )

    @staticmethod
    def _responses_text(response: Any) -> str:
        """``output_text``, else the concatenated ``output_text`` message parts."""
        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text:
            parts: list[str] = []
            for item in getattr(response, "output", None) or ():
                if getattr(item, "type", None) != "message":
                    continue
                for content in getattr(item, "content", None) or ():
                    value = getattr(content, "text", None)
                    if getattr(content, "type", None) == "output_text" and isinstance(value, str):
                        parts.append(value)
            text = "".join(parts)
        return text

    @staticmethod
    def _tokens_dict(
        tokens: tuple[int, int, int, int], cache_write_tokens: int = 0
    ) -> dict[str, int]:
        return _usage_dict(tokens, cache_write_tokens)

    def _extract_responses_tokens(self, response: Any) -> tuple[int, int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0, 0
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        # Early Responses SDKs retain this newer field as an untyped mapping.
        cached = (
            details.get("cached_tokens", 0)
            if isinstance(details, Mapping)
            else getattr(details, "cached_tokens", 0)
        )
        cached_tokens = int(cached or 0)
        return input_tokens, output_tokens, total_tokens, cached_tokens

    def _responses_metadata(self, response: Any) -> dict[str, Any] | None:
        metadata: dict[str, Any] = {"api_surface": "responses"}
        request_id = getattr(response, "id", None)
        if isinstance(request_id, str) and request_id:
            metadata["provider_request_id"] = request_id
        model = getattr(response, "model", None)
        if isinstance(model, str) and model:
            metadata["model"] = model
        status = getattr(response, "status", None)
        if isinstance(status, str) and status:
            metadata["finish_reason"] = status
        if self.response_schema_hash is not None:
            metadata["response_schema"] = {
                "name": self.response_schema_name,
                "identity": self.response_schema_identity,
                "sha256": self.response_schema_hash,
            }

        reasoning: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in getattr(response, "output", None) or ():
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                for content in getattr(item, "content", None) or ():
                    value = getattr(content, "text", None)
                    if isinstance(value, str) and value:
                        reasoning.append(value)
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
        if reasoning:
            metadata["reasoning"] = "".join(reasoning)
        if tool_calls:
            metadata["tool_calls"] = tool_calls
        return _run_extractors(response, self._metadata_extractors, metadata)


def _response_error_text(error: Any) -> tuple[str | None, str]:
    """Extract a provider response error code and human-readable message."""
    if isinstance(error, Mapping):
        raw_code = error.get("code")
        raw_message = error.get("message")
    else:
        raw_code = getattr(error, "code", None)
        raw_message = getattr(error, "message", None)
    code = str(raw_code) if raw_code is not None else None
    message = str(raw_message) if raw_message is not None else str(error)
    return code, message


def _looks_like_schema_rejection(exception: Exception) -> bool:
    """Recognize a provider's deterministic JSON Schema request rejection."""
    status = getattr(exception, "status_code", None)
    if status not in (400, 422):
        return False
    body = getattr(exception, "body", None)
    text = f"{exception} {body}".lower()
    return any(marker in text for marker in ("json_schema", "json schema", "text.format", "schema"))
