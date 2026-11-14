"""OpenAICompatibleModel, the base of the OpenAI, OpenRouter and DeepSeek models."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from ..base import LLMResponse
from . import _sdk
from ._shared import (
    _EMPTY_TEXT_FAILURES,
    MetadataExtractor,
    _auto_prompt_cache_key,
    _cache_group,
    _cache_label,
    _coerce_to_messages,
    _empty_response_reason,
    _has_system_message,
    _openai_cache_write_tokens,
    _pop_cache_label,
    _raise_empty_response,
    _reason_text,
    _run_extractors,
    _system_text,
    _with_cache_group,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


# Bound to OpenAICompatibleModel so from_api_key() returns the calling
# subclass type — subclass overrides don't need a cast(). See issue #10.
TM = TypeVar("TM", bound="OpenAICompatibleModel")


class OpenAICompatibleModel:
    """Base class for OpenAI chat-completions-compatible providers.

    Wraps an ``AsyncOpenAI`` client pointed at any chat-completions endpoint
    (OpenAI itself, OpenRouter, DeepSeek, HuggingFace Inference Providers,
    Together, Fireworks, local vLLM, etc.). Subclasses customize the default
    base URL, the install-extras hint, the env var read by
    :meth:`from_api_key`, and optionally the token/metadata extractors.

    Implements the ``ManagedLLMModel`` protocol — :meth:`cleanup` closes the
    underlying ``AsyncOpenAI`` client when this model owns it (i.e. it was
    constructed via :meth:`from_api_key`). User-provided clients are left
    alone.

    Models built with :meth:`from_api_key` and ``max_connections=N`` expose
    ``max_concurrency=N`` so execution surfaces can diagnose worker/pool
    mismatches. It remains ``None`` for caller-supplied clients because their
    effective transport capacity cannot be inspected reliably.

    Added in v0.9.0.
    """

    # Subclasses override.
    _default_base_url: str | None = None
    _install_extras: str = "openai"
    # Env var :meth:`from_api_key` reads when ``api_key`` is None.
    # ``None`` means: let the OpenAI SDK pick up its own default
    # (``OPENAI_API_KEY``); subclasses set it to read a different env var
    # ourselves and forward to the SDK explicitly.
    _api_key_env_var: str | None = None

    def __init__(
        self,
        model: str,
        client: "AsyncOpenAI",
        *,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        prompt_cache_key: str | None = None,
        cache_group: str | None = None,
    ):
        """
        Args:
            model: Provider model id (e.g. "gpt-6-luna" or
                "anthropic/claude-haiku-4-5").
            client: Initialized AsyncOpenAI (point ``base_url`` at the desired
                endpoint). The model does NOT take ownership of the client —
                use :meth:`from_api_key` if you want the model to manage the
                client's lifecycle.
            system_instruction: Default system message prepended to each call.
                Per-call ``system_instruction`` argument takes precedence.
            extra_headers: Default headers forwarded on every call (e.g.
                OpenRouter's ``HTTP-Referer``/``X-Title``).
            extra_body: Default extra body fields forwarded on every call
                (e.g. OpenRouter ``provider`` routing config).
            metadata_extractors: Optional hooks that contribute extra keys to
                ``LLMResponse.metadata`` (e.g. ``reasoning_content`` or
                ``logprobs``). Merged on top of the built-in
                ``finish_reason``/``model`` metadata; user keys win.
                Added in v0.15.0.
            prompt_cache_key: Sent as ``prompt_cache_key`` (in ``extra_body``) on
                every call, which routes requests that share it to the same prompt
                cache. ``"auto"`` sends ``abl-si-`` plus a hash of the system
                instruction actually sent on each call, and no key when there is
                none. A ``prompt_cache_key`` in per-call ``config`` wins; one in
                ``extra_body`` as well raises ``ValueError``. Provisional.
            cache_group: Label for this model's calls in the per-group cache
                statistics, ahead of the key and the system-instruction hash. A
                per-call ``config={"cache_group": ...}`` overrides it and is never
                sent. Provisional.
        """
        if _sdk.AsyncOpenAI is None:
            raise ImportError(
                f"openai is required for {type(self).__name__}. "
                f"Install with: pip install 'llm-minder[{self._install_extras}]'"
            )

        self._abl_owned_lifecycle = True
        self._model = model
        self._client = client
        self._default_system_instruction = system_instruction
        self._default_extra_headers = extra_headers
        self._default_extra_body = extra_body
        self._metadata_extractors = metadata_extractors
        self._prompt_cache_key = _cache_label(prompt_cache_key, "prompt_cache_key")
        if self._prompt_cache_key is not None and extra_body and "prompt_cache_key" in extra_body:
            raise ValueError("Pass prompt_cache_key or extra_body['prompt_cache_key'], not both")
        self._cache_group_label = _cache_label(cache_group, "cache_group")
        # Known only when this class constructs and sizes the underlying httpx
        # pool via from_api_key(max_connections=...). User-supplied clients are
        # intentionally not introspected because their transport may be wrapped.
        self.max_concurrency: int | None = None
        # Set to True only by from_api_key(); cleanup() uses this to decide
        # whether to close the underlying httpx connections.
        self._owns_client: bool = False
        # SDK constructor kwargs stashed by from_api_key() when it built the
        # client with the *default* httpx pool — lets request_concurrency()
        # rebuild the client with a right-sized pool before the first request.
        # None when the caller supplied the client or an explicit pool.
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
        """Call ``client.chat.completions.create`` and normalize the response.

        Args:
            prompt: A string (becomes a single user message) or a list of
                OpenAI-shaped message dicts (passed through unchanged — used
                for multimodal content and Anthropic-via-OpenRouter
                ``cache_control`` markers).
            temperature: Sampling temperature. Pass ``None`` to omit the
                parameter so the provider uses its own default — required for
                reasoning models that reject an explicit ``temperature``.
            system_instruction: Per-call override for the system message.
            config: Per-call extra kwargs forwarded to the SDK call (merged
                over the instance's ``extra_body``). Use this to pass
                ``max_tokens``, ``response_format``, etc.

        Returns:
            Normalized LLMResponse.
        """
        await self.prepare()
        self._client_used = True
        messages = _coerce_to_messages(prompt)
        si = system_instruction or self._default_system_instruction
        if si is not None and not _has_system_message(messages):
            messages = [{"role": "system", "content": si}, *messages]

        # Merge extra_body: instance defaults + per-call config overrides.
        extra_body: dict[str, Any] = {}
        if self._default_extra_body:
            extra_body.update(self._default_extra_body)
        if config:
            extra_body.update(config)
        label = self._apply_cache_options(extra_body, _system_text(messages))

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        # Omit temperature entirely when None so providers that reject an
        # explicit value (OpenAI reasoning models) use their own default.
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        if self._default_extra_headers:
            call_kwargs["extra_headers"] = self._default_extra_headers

        response = await self._client.chat.completions.create(**call_kwargs)

        # Some gateways report upstream failures inside an HTTP-200 body;
        # surface those before interpreting the (empty) choices list.
        self._raise_on_response_error(response)

        # Extract tokens up front: even a no-content response was billed for
        # the prompt (and, for finish_reason="length", the full output).
        tokens = self._extract_tokens(response)
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens
        cache_writes = self._extract_cache_write_tokens(response)

        # Validate content present (None typically means a tool call or a
        # finish-reason like "length"/"content_filter").
        if not response.choices:
            _raise_empty_response(
                f"No choices returned from {type(self).__name__}.", tokens, cache_writes
            )
        message = response.choices[0].message
        text = getattr(message, "content", None)
        if text is None or text == "":
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
            provider_reason = _reason_text(finish_reason)
            refused = _reason_text(getattr(message, "refusal", None)) is not None
            reason = "refusal" if refused else _empty_response_reason(provider_reason)
            if text is None or reason in _EMPTY_TEXT_FAILURES:
                _raise_empty_response(
                    f"Empty response content from model "
                    f"(finish_reason={finish_reason!r}). "
                    "This typically indicates a tool call, content filter, "
                    "or token limit was reached.",
                    tokens,
                    cache_writes,
                    provider_reason=provider_reason,
                    reason=reason,
                )

        return _with_cache_group(
            LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_tokens,
                metadata=self._build_metadata(response),
                raw=response,
                cache_write_input_tokens=cache_writes,
            ),
            label or _cache_group(extra_body.get("prompt_cache_key"), _system_text(messages)),
        )

    # ── Overridable extraction hooks ────────────────────────────────────

    def _raise_on_response_error(self, response: Any) -> None:
        """Hook: raise if the provider embedded an error in a 200 response.

        No-op here — OpenAI proper signals errors via HTTP status codes.
        Gateway subclasses that use error-in-body semantics (OpenRouter)
        override this.
        """
        return

    def _build_metadata(self, response: Any) -> dict[str, Any] | None:
        """Built-in ``_extract_metadata`` plus any user-supplied extractors.

        Subclasses customize the built-in payload by overriding
        ``_extract_metadata``; user extractors merge on top of that here.
        """
        return _run_extractors(
            response, self._metadata_extractors, self._extract_metadata(response)
        )

    def _extract_tokens(self, response: Any) -> tuple[int, int, int, int]:
        """Extract (input, output, total, cached) token counts.

        Reads OpenAI's ``usage.prompt_tokens`` / ``completion_tokens`` /
        ``total_tokens`` and the nested ``prompt_tokens_details.cached_tokens``
        when present. Subclasses (e.g. for DeepSeek's
        ``prompt_cache_hit_tokens``) override this.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0, 0

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        cached_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)

        return input_tokens, output_tokens, total_tokens, cached_tokens

    def _extract_cache_write_tokens(self, response: Any) -> int:
        """Extract the input tokens written to the provider's prompt cache.

        Reads ``usage.prompt_tokens_details.cache_write_tokens`` (Chat
        Completions) or ``usage.input_tokens_details.cache_write_tokens``
        (Responses); 0 when absent. Separate from :meth:`_extract_tokens` so
        subclasses that override that hook keep working unchanged.
        """
        return _openai_cache_write_tokens(response)

    def _extract_metadata(self, response: Any) -> dict[str, Any] | None:
        """Extract finish_reason, routed model name, and the reserved
        provider-output keys (``reasoning``, ``tool_calls``, ``logprobs``).

        The ``isinstance`` guards below are deliberate: they keep the
        metadata dict JSON-serializable against SDK drift (and keep test
        doubles like ``MagicMock``, whose auto-attributes are truthy, from
        leaking into it). Emitted shapes are the documented contract read by
        the typed views in ``provider_output.py``.
        """
        metadata: dict[str, Any] = {}
        try:
            if response.choices:
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason is not None:
                    metadata["finish_reason"] = str(finish_reason)

                message = getattr(choice, "message", None)
                if message is not None:
                    # Reasoning trace: DeepSeek reasoning_content, falling
                    # back to OpenRouter's reasoning.
                    reasoning = getattr(message, "reasoning_content", None)
                    if not isinstance(reasoning, str) or not reasoning:
                        reasoning = getattr(message, "reasoning", None)
                    if isinstance(reasoning, str) and reasoning:
                        metadata["reasoning"] = reasoning

                    # Tool calls -> plain dicts; arguments stays the raw JSON
                    # string. Visibility only — the framework never executes.
                    calls: list[dict[str, Any]] = []
                    raw_calls = getattr(message, "tool_calls", None)
                    if raw_calls:
                        try:
                            iterator = iter(raw_calls)
                        except TypeError:
                            iterator = iter(())
                        for tc in iterator:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", None)
                            if not isinstance(name, str) or not name:
                                continue
                            args = getattr(fn, "arguments", None)
                            tc_id = getattr(tc, "id", None)
                            calls.append(
                                {
                                    "id": tc_id if isinstance(tc_id, str) else None,
                                    "name": name,
                                    "arguments": args if isinstance(args, str) else "",
                                }
                            )
                    if calls:
                        metadata["tool_calls"] = calls

                # Logprobs: model_dump() keeps it JSON-serializable.
                logprobs = getattr(choice, "logprobs", None)
                if logprobs is not None:
                    dumped = logprobs.model_dump() if hasattr(logprobs, "model_dump") else logprobs
                    if isinstance(dumped, (dict, list)):
                        metadata["logprobs"] = dumped

            routed_model = getattr(response, "model", None)
            if routed_model:
                metadata["model"] = str(routed_model)
        except Exception as e:
            logger.warning(f"Failed to extract response metadata: {e}", exc_info=True)
        return metadata or None

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
                kwargs["http_client"] = _sdk._build_openai_http_client(self._reopen_pool_size)
            self._client = _sdk.AsyncOpenAI(**kwargs)
            self._client_closed = False
            self._client_used = False

    def _lifecycle_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._client_lock is None or self._client_lock_loop is not loop:
            self._client_lock = asyncio.Lock()
            self._client_lock_loop = loop
        return self._client_lock

    async def request_concurrency(self, concurrency: int) -> bool:
        """Ask the model to support ``concurrency`` parallel requests.

        Called by execution surfaces when ``ProcessorConfig.concurrency`` is
        set (v0.20.0, issue #97), before the first request. When this model
        owns its client (built via :meth:`from_api_key`) and no explicit
        ``max_connections`` was given, the ``AsyncOpenAI`` client is rebuilt
        with an httpx pool sized to ``concurrency`` (the SDK's default pool
        allows 1000 connections but keeps only 100 alive between requests, and
        ABL cannot report an unknown capacity) and ``max_concurrency`` starts
        advertising the new size.

        Returns True when the pool was resized. Returns False — leaving the
        model untouched — for caller-supplied clients and for models built
        with an explicit ``max_connections`` (an explicit value always wins;
        a genuine contradiction is surfaced by the existing capacity warning
        instead).
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1; got {concurrency}.")
        if (
            not self._owns_client
            or self._client_used
            or self.max_concurrency is not None
            or self._rebuild_client_kwargs is None
        ):
            return False
        new_client = _sdk.AsyncOpenAI(
            http_client=_sdk._build_openai_http_client(concurrency),
            **self._rebuild_client_kwargs,
        )
        old_client = self._client
        self._client = new_client
        self.max_concurrency = concurrency
        self._reopen_pool_size = concurrency
        self._client_closed = False
        # The old client made no requests yet (this runs before batch start);
        # close it so its default httpx transport doesn't linger.
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
        """Close the underlying AsyncOpenAI client if this model owns it.

        Models constructed directly with ``OpenAIModel(model, client=...)``
        do NOT own the client — the caller is expected to close it. Models
        constructed via :meth:`from_api_key` own the client and close it here,
        unless an HTTP transport was supplied by the caller. Closing that SDK
        client would also close the caller's transport, so both remain open.
        A later prepare reopens an owned client with its configured pool size.
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
                logger.warning(
                    f"Failed to close {type(self).__name__} client: {e}",
                    exc_info=True,
                )
            finally:
                self._client_closed = True

    # ── Convenience constructor ─────────────────────────────────────────

    def _apply_cache_options(self, extra_body: dict[str, Any], system_text: object) -> str | None:
        """Resolve per-request cache options in the final ``extra_body``.

        Removes a per-call ``cache_group`` (it's a label, not a request field),
        adds the model's ``prompt_cache_key`` unless the call set one, and returns
        the caller's label: the per-call one, else the model's.
        """
        label = _pop_cache_label(extra_body) or self._cache_group_label
        if self._prompt_cache_key is not None and "prompt_cache_key" not in extra_body:
            key = (
                _auto_prompt_cache_key(system_text)
                if self._prompt_cache_key == "auto"
                else self._prompt_cache_key
            )
            if key is not None:
                extra_body["prompt_cache_key"] = key
        return label

    @classmethod
    def from_api_key(
        cls: type[TM],
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
        prompt_cache_key: str | None = None,
        cache_group: str | None = None,
        _instance_kwargs: dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> TM:
        """Build the model with a freshly-constructed AsyncOpenAI client.

        The returned model owns the client — its connections are released
        when the framework calls :meth:`cleanup` (typically when the
        ``ParallelBatchProcessor`` exits).

        Uses ``base_url`` (if provided) or the class's ``_default_base_url``.
        Pass ``client_kwargs`` to forward additional kwargs (timeout,
        max_retries, http_client, etc.) to the SDK constructor.

        Args:
            model: Provider model id.
            api_key: API key. If ``None``:

                - For ``OpenAIModel`` the OpenAI SDK auto-reads
                  ``OPENAI_API_KEY``.
                - For ``OpenRouterModel`` (and other subclasses with
                  ``_api_key_env_var`` set) we read the env var ourselves
                  and forward it to the SDK explicitly.
                - If neither path resolves, raises ``ValueError``.
            json_mode: When ``True``, request JSON output by adding
                ``response_format={"type": "json_object"}`` to ``extra_body``
                (forwarded on every call). A convenience over hand-passing it
                yourself; an explicit ``response_format`` in ``extra_body``
                takes precedence. Most providers still require the word "JSON"
                somewhere in your prompt/system instruction for this to take
                effect. Pair with :func:`llm_minder.pydantic_json_parser`
                on the strategy, since some providers (DeepSeek) still wrap the
                JSON in markdown fences even in JSON mode (issue #26).
            max_connections: Size of the underlying httpx connection pool
                (both ``max_connections`` and ``max_keepalive_connections``).
                **Set this to at least ``ProcessorConfig.max_workers``.**
                Without it, the openai SDK's own default applies (1000
                connections, 100 kept alive; the same in openai 1.66.2 and
                3.x): above 100 concurrent requests, connections beyond
                the keep-alive limit are closed after each response and
                reopened, and above 1000 the excess workers block waiting for
                a connection (see issue #25). ABL also can't see the default
                pool's capacity, so it can't warn or gate on it. Mutually
                exclusive with passing your own
                ``http_client``; raises ``ValueError`` if you pass both.
                The value is also exposed as ``model.max_concurrency`` and
                forwarded by ``ModelStrategy`` for processor/gateway validation.
        """
        if _sdk.AsyncOpenAI is None:
            raise ImportError(
                f"openai is required for {cls.__name__}. "
                f"Install with: pip install 'llm-minder[{cls._install_extras}]'"
            )
        # Resolve the key before building any client, so a missing key leaks nothing.
        resolved_key = api_key
        if resolved_key is None and cls._api_key_env_var is not None:
            import os as _os

            key_env = cls._api_key_env_var
            resolved_key = _os.environ.get(key_env)
            if not resolved_key:
                raise ValueError(
                    f"No API key for {cls.__name__}: pass api_key= or set the "
                    f"{key_env} environment variable."
                )

        caller_http_client = client_kwargs.get("http_client") is not None
        own_http_client: Any = None
        if max_connections is not None:
            if "http_client" in client_kwargs:
                raise ValueError(
                    "Pass either max_connections or http_client, not both. "
                    "max_connections is a convenience for sizing the default "
                    "httpx pool; if you build your own http_client, set its "
                    "limits there instead."
                )
            if max_connections < 1:
                raise ValueError(f"max_connections must be >= 1; got {max_connections}.")
            own_http_client = _sdk._build_openai_http_client(max_connections)
            client_kwargs["http_client"] = own_http_client
        effective_base_url = base_url or cls._default_base_url
        if effective_base_url is not None:
            client_kwargs.setdefault("base_url", effective_base_url)

        # When resolved_key is None and the SDK can self-resolve (OpenAIModel),
        # don't pass api_key at all — letting the SDK raise its own clear
        # error if neither path produces one.
        if resolved_key is not None:
            client_kwargs["api_key"] = resolved_key
        client_kwargs.setdefault("max_retries", 0)
        from openai import OpenAIError

        try:
            client = _sdk.AsyncOpenAI(**client_kwargs)
        except BaseException as exc:
            if own_http_client is not None:
                _sdk._close_unused_client(own_http_client, "aclose")
            # Keep all SDK authentication mechanisms and configuration errors.
            # Only translate its known missing-credentials diagnostics.
            if isinstance(exc, OpenAIError) and str(exc).startswith(
                (
                    "Missing credentials.",
                    "The api_key client option must be set",
                )
            ):
                raise ValueError(
                    f"No credentials for {cls.__name__}: pass SDK credentials or set "
                    "OPENAI_API_KEY (or OPENAI_ADMIN_KEY where supported)."
                ) from exc
            raise

        if json_mode:
            # Inject a JSON response_format, letting any explicit caller-supplied
            # response_format in extra_body win.
            effective_extra_body: dict[str, Any] = {"response_format": {"type": "json_object"}}
            if extra_body:
                effective_extra_body.update(extra_body)
            extra_body = effective_extra_body

        try:
            instance = cls(
                model,
                client,
                system_instruction=system_instruction,
                extra_headers=extra_headers,
                extra_body=extra_body,
                metadata_extractors=metadata_extractors,
                # Only when set, so subclasses written for the older constructor
                # signature (and DeepSeekModel, which takes no prompt_cache_key) keep
                # working with this factory.
                **{
                    name: value
                    for name, value in (
                        ("prompt_cache_key", prompt_cache_key),
                        ("cache_group", cache_group),
                    )
                    if value is not None
                },
                **(_instance_kwargs or {}),
            )
        except BaseException:
            # The model never took the client, so release it here — unless the
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
            # Default-pool client: keep the constructor kwargs so
            # request_concurrency() can rebuild it with a right-sized pool.
            instance._rebuild_client_kwargs = dict(client_kwargs)
        return instance
