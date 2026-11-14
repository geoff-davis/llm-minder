"""OpenRouterModel."""

import logging
from typing import TYPE_CHECKING, Any

from ..strategies.errors import (
    ProviderResponseError,
)
from ._openai_compatible import OpenAICompatibleModel
from ._shared import MetadataExtractor

if TYPE_CHECKING:
    pass

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


class OpenRouterModel(OpenAICompatibleModel):
    """LLM model backed by OpenRouter (https://openrouter.ai).

    OpenRouter exposes a unified OpenAI-compatible API for many upstream
    providers (Anthropic, OpenAI, Google, Mistral, DeepSeek, etc.). Model
    ids are prefixed with the provider, e.g. ``"anthropic/claude-haiku-4-5"``.

    Caching is provider-dependent:

    - **OpenAI / Gemini (implicit) / DeepSeek** — automatic; ``cached_input_tokens``
      is populated when the upstream cache hits.
    - **Anthropic** — opt-in. Pass ``prompt`` as a list of message dicts with
      ``cache_control: {"type": "ephemeral"}`` markers on the blocks you want
      cached.

    Example:
        >>> model = OpenRouterModel.from_api_key(
        ...     "anthropic/claude-haiku-4-5",
        ...     api_key="sk-or-...",
        ...     referer="https://my-app.example.com",
        ...     title="My App",
        ... )
        >>> response = await model.generate("Hello!")

    Added in v0.9.0.
    """

    _default_base_url: str | None = "https://openrouter.ai/api/v1"
    _install_extras: str = "openrouter"
    _api_key_env_var: str | None = "OPENROUTER_API_KEY"

    def _raise_on_response_error(self, response: Any) -> None:
        """Raise :class:`ProviderResponseError` for error-in-200 bodies.

        OpenRouter reports upstream failures (no provider available,
        upstream errors, upstream rate limits) as HTTP 200 with an
        ``error`` object and empty ``choices``, so the SDK never raises.
        Without this check the empty choices list would surface as a
        non-retryable ``EmptyResponseError`` even though these are
        precisely the transient failures the classifier is built to retry.
        """
        error = getattr(response, "error", None)
        if not error:
            return
        if isinstance(error, dict):
            message = error.get("message") or str(error)
            code = error.get("code")
        else:
            message = str(getattr(error, "message", None) or error)
            code = getattr(error, "code", None)
        raise ProviderResponseError(
            f"OpenRouter returned an error in the response body (code={code}): {message}",
            code=code if isinstance(code, int) else None,
            provider_error=error,
        )

    def _extract_metadata(self, response: Any) -> dict[str, Any] | None:
        """Add OpenRouter's ``provider`` field to the metadata."""
        metadata = super()._extract_metadata(response) or {}
        provider = getattr(response, "provider", None)
        if provider:
            metadata["provider"] = str(provider)
        return metadata or None

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
        referer: str | None = None,
        title: str | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        prompt_cache_key: str | None = None,
        cache_group: str | None = None,
        **client_kwargs: Any,
    ) -> "OpenRouterModel":
        """Build an OpenRouterModel.

        If ``api_key`` is None, reads ``OPENROUTER_API_KEY`` from the
        environment and raises ``ValueError`` if neither is set. (The
        OpenAI SDK doesn't know about ``OPENROUTER_API_KEY``, so we have
        to read it ourselves rather than relying on the SDK's default.)

        ``referer`` and ``title`` map to OpenRouter's optional
        ``HTTP-Referer`` and ``X-Title`` headers (used for app attribution
        on openrouter.ai's leaderboard).
        """
        merged_headers = dict(extra_headers) if extra_headers else {}
        if referer is not None:
            merged_headers.setdefault("HTTP-Referer", referer)
        if title is not None:
            merged_headers.setdefault("X-Title", title)

        # super().from_api_key is generic over cls (returns the calling
        # subclass type), so this returns OpenRouterModel directly — no cast
        # needed. See issue #10.
        return super().from_api_key(
            model,
            api_key,
            base_url=base_url,
            system_instruction=system_instruction,
            extra_headers=merged_headers or None,
            extra_body=extra_body,
            json_mode=json_mode,
            max_connections=max_connections,
            metadata_extractors=metadata_extractors,
            prompt_cache_key=prompt_cache_key,
            cache_group=cache_group,
            **client_kwargs,
        )
