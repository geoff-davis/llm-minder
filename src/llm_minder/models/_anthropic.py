"""AnthropicModel: the native Anthropic Messages API (#314)."""

import asyncio
import inspect
import json
import logging
from collections.abc import Mapping
from typing import Any, TypeVar

from ..base import LLMResponse
from ..strategies.errors import EmptyResponseReason
from . import _sdk

# Bound for annotations only, so typing.get_type_hints() resolves them. Code reads
# _sdk.<name> at call time; tests/test_models_sdk_wiring.py enforces that.
from ._sdk import AsyncAnthropic
from ._shared import (
    MetadataExtractor,
    _cache_group,
    _cache_label,
    _pop_cache_label,
    _raise_empty_response,
    _reason_text,
    _run_extractors,
    _with_cache_group,
)

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")

# Bound to AnthropicModel so from_api_key() returns the calling subclass type.
TA = TypeVar("TA", bound="AnthropicModel")

# Messages API stop reasons -> EmptyResponseError.reason for a reply with no text.
# A refusal is the API's safety classifier stopping the reply, so it is reported
# as a content filter, like a Gemini safety block or an OpenAI content_filter.
_ANTHROPIC_EMPTY_REASONS: dict[str, EmptyResponseReason] = {
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "refusal": "content_filter",
    "tool_use": "tool_call",
    "end_turn": "no_content",
    "stop_sequence": "no_content",
    "pause_turn": "no_content",
}

# An empty-string reply is still a valid answer, except when it was cut off or refused.
_EMPTY_TEXT_FAILURES = frozenset({"max_tokens", "content_filter"})


def _usage_count(usage: Any, name: str) -> int:
    """A non-negative integer usage field, or 0 when absent, None or not an int."""
    value = getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _json_safe(value: Any) -> Any:
    """``value`` as plain JSON data, or None when it can't be made JSON-serializable."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            value = dump(mode="json", exclude_none=True)
        except Exception:
            return None
    elif isinstance(value, Mapping):
        value = dict(value)
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return None


def _block_type(block: Any) -> str | None:
    kind = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
    return kind if isinstance(kind, str) else None


def _block_field(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, Mapping) else getattr(block, name, None)


def _content_blocks(response: Any) -> list[Any]:
    content = getattr(response, "content", None)
    return list(content) if isinstance(content, (list, tuple)) else []


def _accepted_parameters(method: Any) -> frozenset[str] | None:
    """Keyword names ``method`` accepts, or None when it takes ``**kwargs`` (or can't be read)."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return None
    return frozenset(
        p.name
        for p in parameters
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


class AnthropicModel:
    """LLM model backed by the native Anthropic Messages API.

    Wraps any async Anthropic client exposing ``messages.create``: the
    first-party ``anthropic.AsyncAnthropic`` and the platform clients
    (``AsyncAnthropicVertex``, ``AsyncAnthropicBedrockMantle``,
    ``AsyncAnthropicFoundry``). Calls are non-streaming.

    A string prompt becomes one user message and ``system_instruction`` the
    top-level ``system`` field. ``temperature`` is sent only when set; current
    Claude models reject sampling parameters, and the API's 400 fails fast as a
    non-retryable ``client_error``. Thinking and effort have no defaults here:
    pass them through ``extra_body`` or per-call ``config`` (for example
    ``extra_body={"output_config": {"effort": "low"}}``).

    ``input_tokens`` counts every prompt token, including cache reads and cache
    writes (Anthropic's own ``input_tokens`` excludes both);
    ``cached_input_tokens`` is the cache reads and ``cache_write_input_tokens``
    the cache writes. Price them with ``CachedTokenRates.ANTHROPIC_READ`` (0.10)
    and a ``cache_write_rate`` of ``CachedTokenRates.ANTHROPIC_WRITE`` (1.25,
    5-minute TTL) or ``CachedTokenRates.ANTHROPIC_WRITE_1H`` (2.0, 1-hour TTL).

    Implements ``ManagedLLMModel``: :meth:`cleanup` closes the client only when
    this model built it (:meth:`from_api_key`). Caller-supplied clients are left
    alone.

    Provisional (new in 1.2, #314).
    """

    _install_extras: str = "anthropic"

    def __init__(
        self,
        model: str,
        client: "AsyncAnthropic",
        *,
        max_tokens: int = 16_000,
        system_instruction: str | None = None,
        cache_control: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        cache_group: str | None = None,
    ):
        """
        Args:
            model: Claude model id, e.g. ``"claude-opus-5-5"``.
            client: An async Anthropic client with ``messages.create``. The
                model does NOT take ownership of it; use :meth:`from_api_key`
                for a model that manages its client's lifecycle.
            max_tokens: The required ``max_tokens`` bound sent on every call.
                A per-call ``config={"max_tokens": ...}`` overrides it. Above
                about 21,000 the SDK refuses a non-streaming call unless the
                client has an explicit ``timeout``.
            system_instruction: Default ``system`` prompt. A per-call
                ``system_instruction`` takes precedence.
            cache_control: Sent as the top-level ``cache_control`` field on
                every call, which turns on automatic prompt caching, e.g.
                ``{"type": "ephemeral"}`` or
                ``{"type": "ephemeral", "ttl": "1h"}``.
            extra_headers: Headers forwarded on every call (e.g. an
                ``anthropic-beta`` header).
            extra_body: Request fields merged into every call, e.g.
                ``thinking``, ``output_config`` or ``tools``. Fields the
                client's ``messages.create`` doesn't name go in the SDK's
                ``extra_body``. Per-call ``config`` wins over these.
            metadata_extractors: Optional hooks contributing extra keys to
                ``LLMResponse.metadata``, merged over the built-in keys.
            cache_group: Label for this model's calls in the per-group cache
                statistics, ahead of the system-instruction hash. A per-call
                ``config={"cache_group": ...}`` overrides it and is never sent.
        """
        if _sdk.AsyncAnthropic is None:
            raise ImportError(
                f"anthropic is required for {type(self).__name__}. "
                f"Install with: pip install 'llm-minder[{self._install_extras}]'"
            )
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer (got {max_tokens!r})")
        if cache_control is not None and not isinstance(cache_control, Mapping):
            raise TypeError(
                "cache_control must be a dict such as {'type': 'ephemeral'} or None "
                f"(got {type(cache_control).__name__})"
            )

        self._abl_owned_lifecycle = True
        self._model = model
        self._client = client
        self._max_tokens = max_tokens
        self._default_system_instruction = system_instruction
        self._cache_control = dict(cache_control) if cache_control is not None else None
        self._default_extra_headers = extra_headers
        self._default_extra_body = extra_body
        self._metadata_extractors = metadata_extractors
        self._cache_group_label = _cache_label(cache_group, "cache_group")
        # Known only when from_api_key(max_connections=...) sized the pool.
        self.max_concurrency: int | None = None
        # Ownership state, set by from_api_key(); mirrors OpenAICompatibleModel.
        self._owns_client: bool = False
        self._rebuild_client_kwargs: dict[str, Any] | None = None
        self._reopen_kwargs: dict[str, Any] | None = None
        self._reopen_pool_size: int | None = None
        self._caller_http_client = False
        self._client_closed = False
        self._client_used = False
        self._client_lock: asyncio.Lock | None = None
        self._client_lock_loop: asyncio.AbstractEventLoop | None = None

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = None,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call ``client.messages.create`` and normalize the response.

        Args:
            prompt: A string, sent as one user message. Structured message
                lists are not supported yet (#315).
            temperature: Sampling temperature, sent only when not ``None``.
            system_instruction: Per-call override for the ``system`` prompt.
            config: Per-call request fields merged over ``extra_body``, e.g.
                ``max_tokens`` or ``output_config``. A ``cache_group`` key is a
                statistics label and is never sent.

        Returns:
            Normalized LLMResponse.

        Raises:
            TypeError: ``prompt`` is not a string.
            EmptyResponseError: The response had no text (a
                ``TruncatedResponseError`` when it stopped at ``max_tokens``).
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"{type(self).__name__} takes a string prompt; structured message lists "
                "(and per-block cache_control) are not supported yet. See "
                "https://github.com/geoff-davis/llm-minder/issues/315."
            )
        await self.prepare()
        self._client_used = True

        fields: dict[str, Any] = {}
        if self._default_extra_body:
            fields.update(self._default_extra_body)
        if config:
            fields.update(config)
        label = _pop_cache_label(fields) or self._cache_group_label

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        system = system_instruction or self._default_system_instruction
        if system is not None:
            request["system"] = system
        # Omitted when None: current Claude models reject sampling parameters.
        if temperature is not None:
            request["temperature"] = temperature
        if self._cache_control is not None:
            request["cache_control"] = dict(self._cache_control)
        request.update(fields)

        create = self._client.messages.create
        call_kwargs = self._split_request(create, request)
        if self._default_extra_headers:
            call_kwargs["extra_headers"] = self._default_extra_headers

        response = await create(**call_kwargs)

        # Extract tokens first: even a textless response was billed.
        tokens, cache_writes = self._extract_tokens(response)
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens

        blocks = _content_blocks(response)
        texts = [
            text
            for block in blocks
            if _block_type(block) == "text" and isinstance(text := _block_field(block, "text"), str)
        ]
        stop_reason = _reason_text(getattr(response, "stop_reason", None))
        reason = _ANTHROPIC_EMPTY_REASONS.get(stop_reason or "", "unknown")
        text = "".join(texts)
        if not texts or (text == "" and reason in _EMPTY_TEXT_FAILURES):
            _raise_empty_response(
                f"No text in {type(self).__name__} response "
                f"(stop_reason={stop_reason!r}{self._refusal_note(response)}).",
                tokens,
                cache_writes,
                provider_reason=stop_reason,
                reason=reason,
            )

        return _with_cache_group(
            LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_tokens,
                metadata=_run_extractors(
                    response, self._metadata_extractors, self._extract_metadata(response)
                ),
                raw=response,
                cache_write_input_tokens=cache_writes,
            ),
            label or _cache_group(None, request.get("system")),
        )

    @staticmethod
    def _split_request(create: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Route fields ``create`` doesn't name into the SDK's ``extra_body``.

        Older SDK releases (and platform clients) may not yet name a field the
        API accepts; sending it as a keyword would raise ``TypeError``.
        """
        request = dict(request)
        # An explicit SDK-level extra_body (from extra_body= or config) is kept.
        nested = request.pop("extra_body", None)
        extra: dict[str, Any] = dict(nested) if isinstance(nested, Mapping) else {}
        accepted = _accepted_parameters(create)
        if accepted is None:
            if extra:
                request["extra_body"] = extra
            return request
        call_kwargs: dict[str, Any] = {}
        for key, value in request.items():
            (call_kwargs if key in accepted else extra)[key] = value
        if extra:
            call_kwargs["extra_body"] = extra
        return call_kwargs

    @staticmethod
    def _refusal_note(response: Any) -> str:
        details = _json_safe(getattr(response, "stop_details", None))
        if isinstance(details, dict) and isinstance(details.get("category"), str):
            return f", category={details['category']!r}"
        return ""

    # ── Overridable extraction hooks ────────────────────────────────────

    def _extract_tokens(self, response: Any) -> tuple[tuple[int, int, int, int], int]:
        """Return ``((input, output, total, cached), cache_writes)``.

        Anthropic's ``usage.input_tokens`` excludes cache reads and writes, so
        both are added back: ``input`` counts every prompt token, as it does for
        the other providers.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return (0, 0, 0, 0), 0
        uncached = _usage_count(usage, "input_tokens")
        cache_reads = _usage_count(usage, "cache_read_input_tokens")
        cache_writes = _usage_count(usage, "cache_creation_input_tokens")
        output_tokens = _usage_count(usage, "output_tokens")
        input_tokens = uncached + cache_reads + cache_writes
        return (
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            cache_reads,
        ), cache_writes

    def _extract_metadata(self, response: Any) -> dict[str, Any] | None:
        """Provider, stop reason, model, request id, refusal details, and the
        reserved ``reasoning`` and ``tool_calls`` keys.

        The ``isinstance`` guards keep the dict JSON-serializable against SDK
        drift and test doubles such as ``MagicMock``.
        """
        metadata: dict[str, Any] = {"provider": "anthropic"}
        try:
            stop_reason = _reason_text(getattr(response, "stop_reason", None))
            if stop_reason is not None:
                metadata["finish_reason"] = stop_reason
            model = getattr(response, "model", None)
            if isinstance(model, str) and model:
                metadata["model"] = model
            request_id = getattr(response, "_request_id", None)
            if isinstance(request_id, str) and request_id:
                metadata["request_id"] = request_id
            stop_details = getattr(response, "stop_details", None)
            if stop_details is not None:
                details = _json_safe(stop_details)
                if isinstance(details, dict) and details:
                    metadata["stop_details"] = details

            thoughts: list[str] = []
            calls: list[dict[str, Any]] = []
            for block in _content_blocks(response):
                kind = _block_type(block)
                if kind == "thinking":
                    thought = _block_field(block, "thinking")
                    if isinstance(thought, str) and thought:
                        thoughts.append(thought)
                elif kind == "tool_use":
                    name = _block_field(block, "name")
                    if not isinstance(name, str) or not name:
                        continue
                    call_id = _block_field(block, "id")
                    try:
                        arguments = json.dumps(_block_field(block, "input"))
                    except (TypeError, ValueError):
                        arguments = ""
                    calls.append(
                        {
                            "id": call_id if isinstance(call_id, str) else None,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
            if thoughts:
                metadata["reasoning"] = "\n\n".join(thoughts)
            if calls:
                metadata["tool_calls"] = calls
        except Exception as e:
            logger.warning(f"Failed to extract response metadata: {e}", exc_info=True)
        return metadata

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def prepare(self) -> None:
        """Reopen an owned client after cleanup, preserving its configured pool."""
        if not self._client_closed or self._reopen_kwargs is None:
            return
        async with self._lifecycle_lock():
            if not self._client_closed:
                return
            kwargs = dict(self._reopen_kwargs)
            if self._reopen_pool_size is not None:
                kwargs["http_client"] = _sdk._build_anthropic_http_client(self._reopen_pool_size)
            self._client = _sdk.AsyncAnthropic(**kwargs)
            self._client_closed = False
            self._client_used = False

    def _lifecycle_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._client_lock is None or self._client_lock_loop is not loop:
            self._client_lock = asyncio.Lock()
            self._client_lock_loop = loop
        return self._client_lock

    def _can_resize_pool(self) -> bool:
        """Whether :meth:`request_concurrency` would rebuild the pool (read by describe_limits)."""
        return (
            self._owns_client
            and not self._client_used
            and self.max_concurrency is None
            and self._rebuild_client_kwargs is not None
        )

    async def request_concurrency(self, concurrency: int) -> bool:
        """Size an owned default-pool client for ``concurrency`` parallel requests.

        Same contract as :meth:`OpenAICompatibleModel.request_concurrency`:
        returns False, leaving the model untouched, for caller-supplied clients
        and for models built with an explicit ``max_connections``.
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1; got {concurrency}.")
        rebuild_kwargs = self._rebuild_client_kwargs
        if not self._can_resize_pool() or rebuild_kwargs is None:
            return False
        new_client = _sdk.AsyncAnthropic(
            http_client=_sdk._build_anthropic_http_client(concurrency),
            **rebuild_kwargs,
        )
        old_client = self._client
        self._client = new_client
        self.max_concurrency = concurrency
        self._reopen_pool_size = concurrency
        self._client_closed = False
        close = getattr(old_client, "close", None)
        if close is not None:
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning(
                    f"Failed to close replaced {type(self).__name__} client: {e}",
                    exc_info=True,
                )
        return True

    async def cleanup(self) -> None:
        """Close the client if this model owns it.

        Models built directly with a client don't own it. Models built with
        :meth:`from_api_key` do, unless the caller supplied the HTTP transport
        (closing the client would close it too). A later prepare reopens an
        owned client with its configured pool size.
        """
        if not self._owns_client or self._client is None or self._caller_http_client:
            return
        async with self._lifecycle_lock():
            if self._client_closed:
                return
            close = getattr(self._client, "close", None)
            if close is None:
                return
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning(f"Failed to close {type(self).__name__} client: {e}", exc_info=True)
            finally:
                self._client_closed = True

    # ── Convenience constructor ─────────────────────────────────────────

    @classmethod
    def from_api_key(
        cls: type[TA],
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        max_connections: int | None = None,
        max_tokens: int = 16_000,
        system_instruction: str | None = None,
        cache_control: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        cache_group: str | None = None,
        **client_kwargs: Any,
    ) -> TA:
        """Build the model with a new first-party ``AsyncAnthropic`` client.

        The returned model owns the client and closes it in :meth:`cleanup`.
        The client is built with ``max_retries=0``, since the framework does
        the retrying; pass ``max_retries=`` to override. Other
        ``client_kwargs`` (``timeout``, ``http_client``, ``default_headers``,
        ...) go to the SDK constructor.

        Args:
            model: Claude model id.
            api_key: API key. When ``None`` the SDK resolves its own
                credentials (``ANTHROPIC_API_KEY`` and its other mechanisms).
            base_url: Optional API base URL.
            max_connections: Size of the HTTP connection pool (both
                ``max_connections`` and ``max_keepalive_connections``). Set it
                to at least ``ProcessorConfig.max_workers``. Exposed as
                ``model.max_concurrency``. Mutually exclusive with
                ``http_client``.

        The remaining keyword arguments are those of the constructor.
        """
        if _sdk.AsyncAnthropic is None:
            raise ImportError(
                f"anthropic is required for {cls.__name__}. "
                f"Install with: pip install 'llm-minder[{cls._install_extras}]'"
            )
        caller_http_client = client_kwargs.get("http_client") is not None
        own_http_client: Any = None
        if max_connections is not None:
            if "http_client" in client_kwargs:
                raise ValueError(
                    "Pass either max_connections or http_client, not both. "
                    "max_connections is a convenience for sizing the default "
                    "connection pool; if you build your own http_client (with "
                    "anthropic.DefaultAsyncHttpxClient), set its limits there."
                )
            if max_connections < 1:
                raise ValueError(f"max_connections must be >= 1; got {max_connections}.")
            own_http_client = _sdk._build_anthropic_http_client(max_connections)
            client_kwargs["http_client"] = own_http_client
        if base_url is not None:
            client_kwargs.setdefault("base_url", base_url)
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        client_kwargs.setdefault("max_retries", 0)

        try:
            client = _sdk.AsyncAnthropic(**client_kwargs)
        except BaseException:
            if own_http_client is not None:
                _sdk._close_unused_client(own_http_client, "aclose")
            raise

        try:
            instance = cls(
                model,
                client,
                max_tokens=max_tokens,
                system_instruction=system_instruction,
                cache_control=cache_control,
                extra_headers=extra_headers,
                extra_body=extra_body,
                metadata_extractors=metadata_extractors,
                cache_group=cache_group,
            )
        except BaseException:
            # The model never took the client, so release it here, unless the
            # caller supplied its transport, which closing the client would close.
            if not caller_http_client:
                _sdk._close_unused_client(client, "close")
            raise
        instance.max_concurrency = max_connections
        instance._owns_client = True
        instance._caller_http_client = caller_http_client
        instance._reopen_kwargs = dict(client_kwargs)
        instance._reopen_pool_size = max_connections
        if max_connections is not None:
            instance._reopen_kwargs.pop("http_client", None)
        if max_connections is None and "http_client" not in client_kwargs:
            instance._rebuild_client_kwargs = dict(client_kwargs)
        return instance
