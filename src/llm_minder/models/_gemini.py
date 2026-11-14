"""Gemini models: GeminiModel and GeminiCachedModel."""

import asyncio
import copy
import hashlib
import json
import logging
import time
from typing import Any

from ..base import LLMResponse
from . import _sdk

# Bound for annotations only, so typing.get_type_hints() resolves them. Code reads
# _sdk.<name> at call time; tests/test_models_sdk_wiring.py enforces that.
from ._sdk import Content, genai
from ._shared import (
    _EMPTY_TEXT_FAILURES,
    MetadataExtractor,
    _cache_group,
    _cache_label,
    _canonical_cache_content,
    _decode_tags_from_display_name,
    _encode_tags_to_display_name,
    _extract_metadata,
    _extract_tokens,
    _gemini_empty_reason,
    _pop_cache_label,
    _raise_empty_response,
    _run_extractors,
    _with_cache_group,
)

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


class GeminiModel:
    """
    LLM model backed by the Google Gemini API.

    Wraps a genai.Client and model name, handling API calls, token extraction,
    and response normalization. Implements the LLMModel protocol.

    Example:
        >>> client = genai.Client(api_key="...")
        >>> model = GeminiModel("gemini-3.5-flash", client)
        >>> response = await model.generate("Hello!")
        >>> print(response.text, response.input_tokens)

    Added in v0.6.0.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        *,
        safety_settings: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        cache_group: str | None = None,
    ):
        """
        Args:
            model: Model name (e.g., "gemini-3.1-flash-lite-preview").
            client: Initialized genai.Client.
            safety_settings: Default safety settings for all calls.
            system_instruction: Default system instruction (overridable per-call).
            metadata_extractors: Optional hooks that contribute extra keys to
                ``LLMResponse.metadata``.
                Merged on top of the built-in ``safety_ratings``/``finish_reason``;
                user keys win. Added in v0.15.0.
            cache_group: Label for this model's calls in the per-group cache
                statistics, in place of the system-instruction hash. A per-call
                ``config={"cache_group": ...}`` overrides it and is never sent.
                Provisional.
        """
        if _sdk.genai is None:
            raise ImportError(
                "google-genai is required for GeminiModel. "
                "Install with: pip install 'llm-minder[gemini]'"
            )

        self._abl_owned_lifecycle = True
        self._owns_client = False
        self._reopen_kwargs: dict[str, Any] | None = None
        self._client_closed = False
        self._client_lifecycle_lock: asyncio.Lock | None = None
        self._client_lifecycle_loop: asyncio.AbstractEventLoop | None = None
        self._model = model
        self._client = client
        self._safety_settings = safety_settings
        self._default_system_instruction = system_instruction
        self._cache_group_label = _cache_label(cache_group, "cache_group")
        self._metadata_extractors = metadata_extractors

    def _lifecycle_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._client_lifecycle_loop is not loop:
            self._client_lifecycle_loop = loop
            self._client_lifecycle_lock = asyncio.Lock()
        assert self._client_lifecycle_lock is not None
        return self._client_lifecycle_lock

    async def prepare(self) -> None:
        """Reopen a closed factory-owned client; caller clients stay untouched."""
        if self._owns_client and self._client_closed:
            async with self._lifecycle_lock():
                if self._client_closed:
                    assert self._reopen_kwargs is not None
                    self._client = _sdk.genai.Client(**self._reopen_kwargs)
                    self._client_closed = False

    async def cleanup(self) -> None:
        """Close both transports of a factory-owned Gemini client."""
        if not self._owns_client:
            return
        async with self._lifecycle_lock():
            if self._client_closed:
                return
            try:
                await self._client.aio.aclose()
            finally:
                try:
                    self._client.close()
                finally:
                    self._client_closed = True

    def _build_metadata(self, response: Any) -> dict[str, Any] | None:
        """Built-in Gemini metadata plus any user-supplied extractors."""
        return _run_extractors(response, self._metadata_extractors, _extract_metadata(response))

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = None,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response from Gemini.

        Args:
            prompt: Text prompt or list of content parts (multimodal).
            temperature: Sampling temperature. Pass ``None`` to omit it and
                use the model default.
            system_instruction: Override default system instruction.
            config: Additional provider-specific config entries.

        Returns:
            Normalized LLMResponse.
        """
        await self.prepare()

        # Build config dict
        call_config: dict[str, Any] = {}
        if temperature is not None:
            call_config["temperature"] = temperature

        si = system_instruction or self._default_system_instruction
        if si is not None:
            call_config["system_instruction"] = si

        if self._safety_settings:
            call_config["safety_settings"] = self._safety_settings

        if config:
            call_config.update(config)
        label = _pop_cache_label(call_config) or self._cache_group_label

        # Make the API call (config is built as a dict; the SDK accepts this at runtime
        # even though the type stubs say GenerateContentConfig)
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=call_config,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        )

        # Extract tokens
        tokens = _extract_tokens(response)
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens

        # Extract text (may be None if safety-blocked)
        text = response.text
        if not text:
            metadata = _extract_metadata(response)
            provider_reason, reason = _gemini_empty_reason(response, metadata)
            if text is None or reason in _EMPTY_TEXT_FAILURES:
                safety_info = ""
                if metadata and "safety_ratings" in metadata:
                    safety_info = f" Safety ratings: {metadata['safety_ratings']}"
                _raise_empty_response(
                    f"Empty response from model (finish_reason="
                    f"{(metadata or {}).get('finish_reason', 'unknown')}).{safety_info}",
                    tokens,
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
            ),
            # From the merged config: a per-call system_instruction overrides the default.
            label or _cache_group(None, call_config.get("system_instruction")),
        )


class GeminiCachedModel:
    """
    LLM model backed by Google Gemini with context caching.

    Wraps a genai.Client with cache lifecycle management. Implements the
    ManagedLLMModel protocol: call prepare() before first use, cleanup()
    when done.

    IMPORTANT — share one instance across work items.
        Create ONE GeminiCachedModel and reuse it across every LLMWorkItem that
        should share the cached context. Constructing a new instance per item
        defeats caching entirely and can cost 10× more. The framework calls
        prepare() exactly once per unique instance, so sharing is the intended
        lifecycle. See examples/example_llm_strategies.py for the pattern.

    This provides 70-90% cost savings when shared correctly.

    Example:
        >>> model = GeminiCachedModel(
        ...     "gemini-3.5-flash", client,
        ...     cached_content=[system_instruction, context_docs],
        ... )
        >>> await model.prepare()  # finds or creates cache
        >>> response = await model.generate("Process this")
        >>> await model.cleanup()  # preserves cache for reuse

    Added in v0.6.0.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        cached_content: list["Content"],
        *,
        cache_ttl_seconds: int = 3600,
        cache_renewal_buffer_seconds: int = 300,
        auto_renew: bool = True,
        cache_tags: dict[str, str] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        cache_group: str | None = None,
    ):
        """
        Args:
            model: Model name (e.g., "gemini-3.5-flash").
            client: Initialized genai.Client.
            cached_content: Content to cache (system instructions, documents).
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour).
            cache_renewal_buffer_seconds: Renew this many seconds before expiry
                (default: 300 = 5 minutes).
            auto_renew: Auto-renew expired caches in generate() (default: True).
            cache_tags: Tags for precise cache matching. Encoded into the cache's
                ``display_name`` at creation (google-genai ``CreateCachedContentConfig``
                has no ``metadata`` field) and decoded on lookup. Keep tag values
                short — Gemini's ``display_name`` has a 128-character limit.
            safety_settings: Default safety settings for all calls.
            metadata_extractors: Optional hooks that contribute extra keys to
                ``LLMResponse.metadata``.
                Merged on top of the built-in metadata; user keys win.
                Added in v0.15.0.
            cache_group: Label for this model's calls in the per-group cache
                statistics, in place of the cached-content hash. A per-call
                ``config={"cache_group": ...}`` overrides it and is never sent.
                Provisional.
        """
        if _sdk.genai is None:
            raise ImportError(
                "google-genai is required for GeminiCachedModel. "
                "Install with: pip install 'llm-minder[gemini]'"
            )

        if cache_renewal_buffer_seconds >= cache_ttl_seconds:
            raise ValueError(
                f"cache_renewal_buffer_seconds ({cache_renewal_buffer_seconds}) "
                f"must be less than cache_ttl_seconds ({cache_ttl_seconds})."
            )

        if 10 <= cache_ttl_seconds < 60:
            import warnings

            warnings.warn(
                f"cache_ttl_seconds ({cache_ttl_seconds}) is less than 60 seconds. "
                f"Very short TTLs defeat the purpose of caching. "
                f"Recommended minimum: 300 seconds (5 minutes).",
                UserWarning,
                stacklevel=2,
            )

        if cache_renewal_buffer_seconds < 60:
            import warnings

            warnings.warn(
                f"cache_renewal_buffer_seconds ({cache_renewal_buffer_seconds}) is less than "
                f"60 seconds. Small buffers risk renewing on every call if generation takes "
                f"longer than the buffer. Recommended minimum: 60 seconds.",
                UserWarning,
                stacklevel=2,
            )

        self._model = model
        self._client = client
        self._cached_content = copy.deepcopy(cached_content)
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_renewal_buffer_seconds = cache_renewal_buffer_seconds
        self._auto_renew = auto_renew
        if cache_tags and "abl-content" in cache_tags:
            raise ValueError("cache_tags key 'abl-content' is reserved for content identity")
        canonical_content = _canonical_cache_content(self._cached_content)
        canonical = json.dumps(
            {
                "model": model.rsplit("/models/", 1)[-1].removeprefix("models/"),
                "contents": canonical_content,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self._cache_tags = dict(cache_tags or {})
        self._cache_tags["abl-content"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
        display_name = _encode_tags_to_display_name(self._cache_tags)
        if len(display_name) > 128:
            raise ValueError(
                "cache_tags exceed the 128-character display_name budget including "
                "the reserved abl-content fingerprint and JSON escaping"
            )
        self._safety_settings = safety_settings
        self._metadata_extractors = metadata_extractors
        self._cache_group_label = _cache_label(cache_group, "cache_group")

        self._cache: Any = None
        self._cache_created_at: float | None = None
        self._cache_expires_at: float | None = None
        self._cache_lock: Any = None
        self._prepared = False

    def _build_metadata(self, response: Any) -> dict[str, Any] | None:
        """Built-in Gemini metadata plus any user-supplied extractors."""
        return _run_extractors(response, self._metadata_extractors, _extract_metadata(response))

    @property
    def cache_name(self) -> str | None:
        """The name of the active cache, or None."""
        return self._cache.name if self._cache else None

    # ── Lifecycle ───────────────────────────────────────────────

    async def prepare(self) -> None:
        """Find or create the Gemini cache. Idempotent."""
        if self._prepared:
            return

        import asyncio

        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()

        async with self._cache_lock:
            if self._prepared:
                return
            await self._find_or_create_cache()
            self._prepared = True

    async def cleanup(self) -> None:
        """Preserve cache for reuse (does not delete). Idempotent."""
        if self._cache:
            logger.info(
                f"Leaving cache active for reuse: {self._cache.name} "
                f"(TTL: {self._cache_ttl_seconds}s, will expire naturally)"
            )

    async def delete_cache(self) -> None:
        """Explicitly delete the cache.

        Safe to call concurrently: the cache lock serializes delete attempts
        so the provider API fires at most once, and late callers that arrive
        after the cache is cleared return silently.
        """
        if self._cache is None:
            return

        import asyncio as _asyncio

        if getattr(self, "_cache_lock", None) is None:
            self._cache_lock = _asyncio.Lock()

        async with self._cache_lock:
            cache = self._cache
            if cache is None:
                # A concurrent caller already finished the delete.
                return

            # Capture the name up front so log messages don't depend on
            # self._cache still existing after concurrent callers clear it.
            cache_name = cache.name
            # Clear state BEFORE the API call so concurrent tasks that
            # re-enter see an empty cache and no-op.
            self._cache = None
            self._cache_created_at = None
            self._cache_expires_at = None
            self._prepared = False

            try:
                await self._client.aio.caches.delete(name=cache_name)
                logger.info(f"Deleted Gemini cache: {cache_name}")
            except Exception as e:
                # Keep Exception (not BaseException) so KeyboardInterrupt still propagates;
                # cache-delete failures are best-effort — caches expire on their own.
                logger.warning(
                    f"Failed to delete Gemini cache '{cache_name}': {e}. "
                    "Cache may have already expired or been deleted.",
                    exc_info=True,
                )

    # ── Generate ────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = None,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response using the cached context.

        Args:
            prompt: Text prompt or multimodal content parts.
            temperature: Sampling temperature. Pass ``None`` to omit it and
                use the model default.
            system_instruction: Not supported with caching — raises ValueError.
            config: Additional provider-specific config entries.

        Returns:
            Normalized LLMResponse.
        """
        if system_instruction is not None:
            raise ValueError(
                "system_instruction cannot be overridden per-call with cached models. "
                "The system instruction is baked into the cache at creation time."
            )

        # Auto-renew if expired
        if self._auto_renew and self._is_cache_expired():
            import asyncio

            if self._cache_lock is None:
                self._cache_lock = asyncio.Lock()

            async with self._cache_lock:
                if self._is_cache_expired():
                    age_str = (
                        f"{time.time() - self._cache_created_at:.0f}s"
                        if self._cache_created_at is not None
                        else "unknown (cache not yet initialized)"
                    )
                    logger.info(
                        "Cache expired or about to expire, renewing before API call "
                        f"(age: {age_str}, "
                        f"renewal buffer: {self._cache_renewal_buffer_seconds}s)"
                    )
                    await self._renew_cache()

        if self._cache is None:
            raise RuntimeError("Cache not initialized — call prepare() first")

        # Build config with cache reference
        call_config: dict[str, Any] = {
            "cached_content": self._cache.name,
        }
        if temperature is not None:
            call_config["temperature"] = temperature

        if self._safety_settings:
            call_config["safety_settings"] = self._safety_settings

        if config:
            call_config.update(config)
        label = _pop_cache_label(call_config) or self._cache_group_label

        requested_cache = call_config["cached_content"]
        owned_cache_name = self._cache.name
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=call_config,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            )
        except Exception as exc:
            from google.genai.errors import APIError

            missing_cache = (
                isinstance(exc, APIError)
                and exc.code in (403, 404)
                and "cachedcontent" in str(exc.message).lower().replace(" ", "")
                and requested_cache == owned_cache_name
            )
            if not self._auto_renew or not missing_cache:
                raise
            assert self._cache_lock is not None
            async with self._cache_lock:
                if self._cache is not None and self._cache.name == requested_cache:
                    self._cache = None
                    self._cache_created_at = None
                    self._cache_expires_at = None
                    self._prepared = False
                    await self._find_or_create_cache(renewing=True, exclude_name=requested_cache)
                    self._prepared = True
            assert self._cache is not None
            call_config["cached_content"] = self._cache.name
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=call_config,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            )

        tokens = _extract_tokens(response)
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens

        text = response.text
        if not text:
            metadata = _extract_metadata(response)
            provider_reason, reason = _gemini_empty_reason(response, metadata)
            if text is None or reason in _EMPTY_TEXT_FAILURES:
                safety_info = ""
                if metadata and "safety_ratings" in metadata:
                    safety_info = f" Safety ratings: {metadata['safety_ratings']}"
                _raise_empty_response(
                    f"Empty response from model (finish_reason="
                    f"{(metadata or {}).get('finish_reason', 'unknown')}).{safety_info}",
                    tokens,
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
            ),
            label or "gemini-cache:" + self._cache_tags["abl-content"][:12],
        )

    # ── Cache internals ─────────────────────────────────────────

    def _is_cache_expired(self) -> bool:
        if self._cache is None:
            return True
        if self._cache_expires_at is not None:
            return self._cache_expires_at - time.time() <= self._cache_renewal_buffer_seconds
        if self._cache_created_at is None:
            return True
        cache_age = time.time() - self._cache_created_at
        expires_in = self._cache_ttl_seconds - cache_age
        return expires_in <= self._cache_renewal_buffer_seconds

    def _remember_cache_expiry(
        self, cache: Any, *, renewed: bool = False, newly_created: bool = False
    ) -> None:
        expiry = getattr(cache, "expire_time", None)
        created = getattr(cache, "create_time", None)
        if renewed:
            self._cache_created_at = time.time()
        elif created is not None:
            self._cache_created_at = created.timestamp()
        elif newly_created:
            self._cache_created_at = time.time()
        else:
            # Unknown age on an adopted cache must not grant it a fresh TTL.
            self._cache_created_at = time.time() - self._cache_ttl_seconds
        self._cache_expires_at = (
            expiry.timestamp()
            if expiry is not None
            else self._cache_created_at + self._cache_ttl_seconds
        )

    def _cache_within_renewal_buffer(self, cache: Any) -> bool:
        """True if ``cache`` expires within the renewal buffer.

        Reads the provider's ``expire_time`` so renewal lookups can skip a
        candidate that is itself about to expire. Returns False when the
        expiry is unknown (don't skip what we can't evaluate).
        """
        expire_time = getattr(cache, "expire_time", None)
        if expire_time is None:
            return False
        try:
            expires_in = expire_time.timestamp() - time.time()
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(expires_in <= self._cache_renewal_buffer_seconds)

    async def _renew_cache(self) -> None:
        """Extend the current cache's TTL in place, or find/create a fresh one.

        Prefers ``caches.update()`` so the SAME cache's TTL is pushed out. This
        fixes the old renewal bug: ``_find_or_create_cache`` would re-list and
        re-adopt the very same near-expiry cache, extending nothing — so every
        ``generate()`` call inside the renewal-buffer window kept doing a
        ``caches.list()`` round-trip and the cache eventually lapsed anyway.

        Falls back to find-or-create (skipping near-expiry candidates) when
        there is no current cache or the update call fails.
        """
        cache = self._cache
        if cache is not None and getattr(cache, "name", None):
            try:
                from google.genai.types import UpdateCachedContentConfig

                updated = await self._client.aio.caches.update(
                    name=cache.name,
                    config=UpdateCachedContentConfig(ttl=f"{self._cache_ttl_seconds}s"),
                )
                # Some SDK versions return the refreshed cache object; keep our
                # reference current when they do.
                if updated is not None:
                    self._cache = updated
                if updated is not None:
                    self._remember_cache_expiry(updated, renewed=True)
                else:
                    self._cache_created_at = time.time()
                    self._cache_expires_at = time.time() + self._cache_ttl_seconds
                self._prepared = True
                logger.info(
                    f"Renewed Gemini cache TTL in place: {self._cache.name} "
                    f"(TTL: {self._cache_ttl_seconds}s)"
                )
                return
            except Exception as e:
                # Keep Exception (not BaseException) so KeyboardInterrupt still
                # propagates; an in-place renew failure is recoverable via the
                # find-or-create fallback below.
                logger.warning(
                    f"Failed to update TTL for cache '{getattr(cache, 'name', '?')}': {e}. "
                    "Falling back to find-or-create."
                )

        # Drop the stale reference and find/create — but skip caches that are
        # themselves inside the renewal buffer so we don't re-adopt a dud.
        self._cache = None
        self._cache_created_at = None
        self._cache_expires_at = None
        self._prepared = False
        await self._find_or_create_cache(renewing=True)
        self._prepared = True

    async def _find_or_create_cache(
        self, renewing: bool = False, *, exclude_name: str | None = None
    ) -> None:
        try:
            caches = await self._client.aio.caches.list()

            async for cache in caches:
                if cache.name == exclude_name:
                    continue
                if not cache.model or cache.model.rsplit("/models/", 1)[-1].removeprefix(
                    "models/"
                ) != self._model.rsplit("/models/", 1)[-1].removeprefix("models/"):
                    continue

                # During renewal, ignore any candidate that is itself about to
                # expire — adopting it would just put us straight back into the
                # renewal-buffer window.
                if renewing and self._cache_within_renewal_buffer(cache):
                    logger.debug(
                        f"Skipping cache {cache.name} during renewal: "
                        "within renewal buffer of expiry"
                    )
                    continue

                if self._cache_tags:
                    cache_tags = _decode_tags_from_display_name(
                        getattr(cache, "display_name", None)
                    )
                    if cache_tags is None:
                        logger.debug(
                            f"Skipping cache {cache.name}: no abl-tags display_name "
                            f"(want {self._cache_tags})"
                        )
                        continue
                    tags_match = all(cache_tags.get(k) == v for k, v in self._cache_tags.items())
                    if not tags_match:
                        logger.debug(
                            f"Skipping cache {cache.name}: tags don't match "
                            f"(want {self._cache_tags}, has {cache_tags})"
                        )
                        continue

                self._cache = cache
                self._remember_cache_expiry(cache)

                tag_info = f" with tags {self._cache_tags}" if self._cache_tags else ""
                age = time.time() - (self._cache_created_at or time.time())
                logger.info(
                    f"Reusing existing Gemini cache: {self._cache.name}{tag_info} (age: {age:.0f}s)"
                )
                return
        except Exception as e:
            logger.warning(f"Failed to list existing caches: {e}")

        await self._create_new_cache()

    async def _create_new_cache(self) -> None:
        from google.genai.types import CreateCachedContentConfig

        config_kwargs: dict[str, Any] = {
            "contents": self._cached_content,
            "ttl": f"{self._cache_ttl_seconds}s",
        }
        if self._cache_tags:
            # google-genai's CreateCachedContentConfig has no `metadata` field —
            # round-trip tags through `display_name` with a sentinel prefix.
            config_kwargs["display_name"] = _encode_tags_to_display_name(self._cache_tags)

        self._cache = await self._client.aio.caches.create(
            model=self._model,
            config=CreateCachedContentConfig(**config_kwargs),
        )

        self._remember_cache_expiry(self._cache, newly_created=True)
        tag_info = f" with tags {self._cache_tags}" if self._cache_tags else ""
        logger.info(
            f"Created new Gemini cache: {self._cache.name}{tag_info} "
            f"(TTL: {self._cache_ttl_seconds}s)"
        )


# ── OpenAI-compatible models ────────────────────────────────────────────────
