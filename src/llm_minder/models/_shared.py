"""Helpers shared by the provider models: metadata, usage, empty responses, cache groups."""

import base64
import enum
import hashlib
import io
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, NoReturn

from pydantic import BaseModel

from ..base import LLMResponse
from ..strategies.errors import (
    EmptyResponseError,
    EmptyResponseReason,
    TruncatedResponseError,
)

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


# A user-supplied hook that maps a raw provider response onto extra metadata
# keys. Returns None (or an empty dict) to contribute nothing. Extractors run
# best-effort and merge on top of the model's built-in metadata — see
# _run_extractors. Added in v0.15.0 (issue #52).
MetadataExtractor = Callable[[Any], dict[str, Any] | None]

# Sentinel prefix for encoding cache_tags into Gemini's CachedContent.display_name.
# google-genai's CreateCachedContentConfig does not expose a metadata field, so we
# round-trip tags through display_name, marked with this prefix so we can tell
# llm-minder-tagged caches apart from caches with user-chosen display names.
_TAG_DISPLAY_NAME_PREFIX = "abl-tags:"


def _fallback_cache_content(value: Any) -> Any:
    """Conservative typed identity when the SDK normalizer is unavailable."""
    from google.genai import types

    if isinstance(value, types.File):
        return {"type": "File", "uri": value.uri, "mime_type": value.mime_type}
    if isinstance(value, BaseModel):
        return {
            "type": type(value).__name__,
            "value": value.model_dump(mode="json", exclude_none=True),
        }
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, list):
        return {"type": "list", "value": [_fallback_cache_content(item) for item in value]}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "value": {key: _fallback_cache_content(item) for key, item in value.items()},
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"type": type(value).__name__, "value": value}
    try:
        from importlib import import_module

        Image = import_module("PIL.Image").Image
    except ImportError:
        Image = None  # type: ignore[misc,assignment]
    if Image is not None and isinstance(value, Image):
        encoded = io.BytesIO()
        value.save(encoded, format="PNG")
        return {
            "type": "image",
            "mode": value.mode,
            "size": value.size,
            "format": value.format,
            "data": base64.b64encode(encoded.getvalue()).decode("ascii"),
        }
    raise ValueError(f"Unsupported cached content type: {type(value).__name__}")


def _canonical_cache_content(contents: Any) -> Any:
    """Fingerprint the normalized request, not volatile SDK handle metadata."""
    # Preserve construction with empty contents (validation belongs to create).
    if isinstance(contents, list) and not contents:
        return []
    try:
        from google.genai._transformers import t_contents
    except ImportError:
        # Separate namespace: a fallback digest cannot match an SDK-normalized
        # digest accidentally. SDK upgrades may cause a safe one-time cache miss.
        return {"fallback": _fallback_cache_content(contents)}
    return [content.model_dump(mode="json", exclude_none=True) for content in t_contents(contents)]


def _encode_tags_to_display_name(tags: dict[str, str]) -> str:
    """Encode cache_tags as a deterministic string for the CachedContent display_name.

    Uses sorted, compact JSON so equal tag sets always produce the same display_name —
    critical for cache lookup to match. Prefixed with a sentinel so we can tell our
    tag encoding apart from a user-assigned display name.
    """
    encoded = json.dumps(tags, sort_keys=True, separators=(",", ":"))
    return f"{_TAG_DISPLAY_NAME_PREFIX}{encoded}"


def _decode_tags_from_display_name(display_name: str | None) -> dict[str, str] | None:
    """Decode cache_tags from a CachedContent display_name.

    Returns None when the display_name is absent or was not produced by
    _encode_tags_to_display_name. Callers should treat None as "this cache has no
    tag metadata we can match against".
    """
    if not display_name or not display_name.startswith(_TAG_DISPLAY_NAME_PREFIX):
        return None
    try:
        decoded = json.loads(display_name[len(_TAG_DISPLAY_NAME_PREFIX) :])
    except ValueError:
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _enum_name(value: Any) -> str:
    """Return an SDK enum's wire value (``"STOP"``), not ``str()``'s ``"FinishReason.STOP"``."""
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _extract_metadata(response: Any) -> dict[str, Any] | None:
    """Extract safety ratings and finish reason from a Gemini response."""
    metadata: dict[str, Any] = {}

    try:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Safety ratings
            if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                ratings: dict[str, str] = {}
                for rating in candidate.safety_ratings:
                    category = (
                        _enum_name(rating.category) if hasattr(rating, "category") else "UNKNOWN"
                    )
                    probability = (
                        _enum_name(rating.probability)
                        if hasattr(rating, "probability")
                        else "UNKNOWN"
                    )
                    ratings[category] = probability
                metadata["safety_ratings"] = ratings

            # Finish reason
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                metadata["finish_reason"] = _enum_name(candidate.finish_reason)

        thoughts = getattr(getattr(response, "usage_metadata", None), "thoughts_token_count", 0)
        if isinstance(thoughts, int) and thoughts > 0:
            metadata["reasoning_tokens"] = thoughts

        # Grounding (google_search tool) — only present when the caller
        # requested it, so default payloads are unchanged for everyone else.
        grounding = _extract_grounding(response)
        if grounding:
            metadata["grounding"] = grounding
    except Exception as e:
        # Metadata extraction is best-effort; missing/partial attributes on the
        # provider response shouldn't break the call. Keep `except Exception` so
        # KeyboardInterrupt still propagates.
        logger.warning(f"Failed to extract response metadata: {e}", exc_info=True)

    return metadata or None


def _run_extractors(
    response: Any,
    extractors: list[MetadataExtractor] | None,
    base: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge user-supplied metadata extractors on top of the built-in ``base``.

    Each extractor runs best-effort: a failing extractor is logged and skipped
    so it can't break the call. Later extractors win key collisions, and any
    user extractor overrides a built-in key of the same name. Returns ``None``
    when nothing was extracted, preserving the "empty → None" contract that
    ``LLMResponse.metadata`` expects.

    Added in v0.15.0 (issue #52).
    """
    merged: dict[str, Any] = dict(base) if base else {}
    for extractor in extractors or ():
        try:
            extra = extractor(response)
        except Exception as e:
            # Keep `except Exception` (not bare) so KeyboardInterrupt propagates.
            logger.warning(f"Metadata extractor {extractor!r} failed: {e}", exc_info=True)
            continue
        if extra:
            merged.update(extra)
    return merged or None


def _extract_grounding(response: Any) -> dict[str, Any] | None:
    """Extract Gemini grounding into the documented plain-dict shape.

    Returns the inner ``metadata['grounding']`` value — ``{"sources":
    [{"uri", "title"}], "queries": [...], "supports": [...]}`` — or ``None``
    when the response carries no grounding metadata. This dict shape is the
    public compatibility contract that the typed ``Grounding`` view
    (``provider_output.py``) reads.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    gm = getattr(candidates[0], "grounding_metadata", None)
    if gm is None:
        return None

    grounding: dict[str, Any] = {}

    sources: list[dict[str, Any]] = []
    for chunk in getattr(gm, "grounding_chunks", None) or ():
        web = getattr(chunk, "web", None)
        if web is None:
            continue
        sources.append({"uri": getattr(web, "uri", None), "title": getattr(web, "title", None)})
    if sources:
        grounding["sources"] = sources

    queries = [q for q in (getattr(gm, "web_search_queries", None) or ()) if q]
    if queries:
        grounding["queries"] = queries

    supports: list[dict[str, Any]] = []
    for support in getattr(gm, "grounding_supports", None) or ():
        segment = getattr(support, "segment", None)
        supports.append(
            {
                "text": getattr(segment, "text", None),
                "start_index": getattr(segment, "start_index", None),
                "end_index": getattr(segment, "end_index", None),
                "chunk_indices": list(getattr(support, "grounding_chunk_indices", None) or ()),
            }
        )
    if supports:
        grounding["supports"] = supports

    return grounding or None


def grounding_metadata_extractor(response: Any) -> dict[str, Any] | None:
    """Metadata extractor for Gemini grounding (``google_search``).

    Since v0.16.0 the built-in Gemini models emit ``metadata['grounding']``
    **by default** whenever the response carries grounding metadata, so
    passing this extractor to ``GeminiModel``/``GeminiCachedModel`` is
    redundant (harmless — it re-emits the same key with the same value).
    It left the public API in 1.0 and is kept here only as a private helper.

    Returns ``{"grounding": {...}}`` with ``sources`` (``[{"uri",
    "title"}]``), ``queries`` (the ``web_search_queries`` the model issued),
    and ``supports`` (answer-span → source-index links), or ``None`` when the
    response carries no grounding metadata. Read the typed view via
    ``result.grounding`` (see ``provider_output.py``).

    Added in v0.15.0 (issue #52).
    """
    grounding = _extract_grounding(response)
    return {"grounding": grounding} if grounding else None


def _usage_dict(tokens: tuple[int, int, int, int], cache_write_tokens: int = 0) -> dict[str, int]:
    """Token-usage dict from an (input, output, total, cached) tuple plus cache writes."""
    input_tokens, output_tokens, total_tokens, cached_tokens = tokens
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_tokens,
    }
    if cache_write_tokens:
        usage["cache_write_input_tokens"] = cache_write_tokens
    return usage


# Provider stop reasons (lower-cased) -> EmptyResponseError.reason. Gemini
# FinishReason / BlockedReason names, Chat Completions finish_reason values and
# Responses API incomplete_details.reason values share one table.
_EMPTY_RESPONSE_REASONS: dict[str, EmptyResponseReason] = {
    # Output bound reached
    "max_tokens": "max_tokens",
    "length": "max_tokens",
    "max_output_tokens": "max_tokens",
    # Safety and policy blocks
    "content_filter": "content_filter",
    "safety": "content_filter",
    "recitation": "content_filter",
    "blocklist": "content_filter",
    "prohibited_content": "content_filter",
    "spii": "content_filter",
    "image_safety": "content_filter",
    "image_prohibited_content": "content_filter",
    "image_recitation": "content_filter",
    # Tool calls instead of text
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "malformed_function_call": "tool_call",
    "unexpected_tool_call": "tool_call",
    "too_many_tool_calls": "tool_call",
    # Finished normally, but without text
    "stop": "no_content",
    "completed": "no_content",
}


# An empty-string reply ("" rather than no text at all) is still a valid answer,
# except when the provider says it was cut off, blocked or refused.
_EMPTY_TEXT_FAILURES: frozenset[EmptyResponseReason] = frozenset(
    {"max_tokens", "content_filter", "refusal"}
)


def _empty_response_reason(provider_reason: str | None) -> EmptyResponseReason:
    """Normalize a provider's stop reason; unrecognized values are ``"unknown"``."""
    if provider_reason is None:
        return "unknown"
    return _EMPTY_RESPONSE_REASONS.get(provider_reason.lower(), "unknown")


def _reason_text(value: Any) -> str | None:
    """A provider reason as a string, or None when absent or not reason-shaped."""
    if isinstance(value, enum.Enum):
        value = value.value
    return value if isinstance(value, str) and value else None


def _raise_empty_response(
    message: str,
    tokens: tuple[int, int, int, int],
    cache_write_tokens: int = 0,
    *,
    provider_reason: str | None = None,
    reason: EmptyResponseReason | None = None,
) -> NoReturn:
    """Raise EmptyResponseError carrying the tokens the provider already billed.

    The API call succeeded (and was billed) even though it produced no usable
    text — without attaching the counts, the framework's failed-attempt
    accounting would report zero spend for the attempt. ``reason`` defaults to
    the normalized ``provider_reason``; a truncation raises
    :class:`TruncatedResponseError`.
    """
    if reason is None:
        reason = _empty_response_reason(provider_reason)
    token_usage = _usage_dict(tokens, cache_write_tokens)
    if reason == "max_tokens":
        raise TruncatedResponseError(
            message, token_usage=token_usage, provider_reason=provider_reason
        )
    raise EmptyResponseError(
        message, token_usage=token_usage, reason=reason, provider_reason=provider_reason
    )


def _gemini_empty_reason(
    response: Any, metadata: dict[str, Any] | None
) -> tuple[str | None, EmptyResponseReason]:
    """``(provider_reason, reason)`` for a textless Gemini response.

    The provider reason is the candidate's finish reason, or the prompt's block
    reason when no candidate came back. A function call ends with a plain
    ``STOP``, so the candidate's parts decide ``tool_call`` unless the finish
    reason already names a truncation or a block.
    """
    provider_reason = _reason_text((metadata or {}).get("finish_reason"))
    if provider_reason is None:
        feedback = getattr(response, "prompt_feedback", None)
        provider_reason = _reason_text(getattr(feedback, "block_reason", None))
    reason = _empty_response_reason(provider_reason)
    if reason in ("no_content", "unknown") and _gemini_has_function_call(response):
        reason = "tool_call"
    return provider_reason, reason


def _gemini_has_function_call(response: Any) -> bool:
    """Whether the first candidate's content includes a function-call part."""
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return False
    parts = getattr(getattr(candidates[0], "content", None), "parts", None)
    if not isinstance(parts, list):
        return False
    return any(getattr(part, "function_call", None) is not None for part in parts)


def _openai_cache_write_tokens(response: Any) -> int:
    """Prompt-cache write tokens from OpenAI-shaped usage, or 0.

    Chat Completions reports them under ``usage.prompt_tokens_details`` and
    Responses under ``usage.input_tokens_details``, either as a typed object
    or (older SDKs) a plain mapping.
    """
    usage = getattr(response, "usage", None)
    for name in ("prompt_tokens_details", "input_tokens_details"):
        details = getattr(usage, name, None)
        if details is None:
            continue
        value = (
            details.get("cache_write_tokens")
            if isinstance(details, Mapping)
            else getattr(details, "cache_write_tokens", None)
        )
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _extract_tokens(response: Any) -> tuple[int, int, int, int]:
    """Extract token counts from a Gemini response.

    Returns:
        (input_tokens, output_tokens, total_tokens, cached_input_tokens)
    """
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        logger.debug(
            "No usage_metadata on response (%s); token counts will be zero.",
            type(response).__name__,
        )
        return 0, 0, 0, 0

    input_tokens = (getattr(usage_metadata, "prompt_token_count", 0) or 0) + (
        getattr(usage_metadata, "tool_use_prompt_token_count", 0) or 0
    )
    output_tokens = (getattr(usage_metadata, "candidates_token_count", 0) or 0) + (
        getattr(usage_metadata, "thoughts_token_count", 0) or 0
    )
    total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
    cached_tokens = 0
    if hasattr(usage_metadata, "cached_content_token_count"):
        cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0) or 0

    return input_tokens, output_tokens, total_tokens, cached_tokens


def _coerce_to_messages(prompt: str | list[Any]) -> list[Any]:
    """Convert a string prompt to a single user message; pass lists through."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


def _system_text(messages: object) -> str | None:
    """The first system message's text, if ``messages`` is a list that has one."""
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


def _cache_group(cache_key: object, system_text: object) -> str | None:
    """The ``cache_group`` a built-in model stamps (issue #244).

    An explicit ``prompt_cache_key`` names the group; otherwise a short hash of the
    system instruction does, since that's the prefix the provider caches.
    """
    if isinstance(cache_key, str) and cache_key:
        return cache_key
    if isinstance(system_text, str) and system_text:
        return "si:" + hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:12]
    return None


def _with_cache_group(
    response: LLMResponse, group: str | None, miss_reason: str | None = None
) -> LLMResponse:
    """Add ``cache_group`` (and ``cache_miss_reason``) to the response metadata
    unless an extractor already set them."""
    if group is not None or miss_reason is not None:
        metadata = dict(response.metadata or {})
        if group is not None:
            metadata.setdefault("cache_group", group)
        if miss_reason is not None:
            metadata.setdefault("cache_miss_reason", miss_reason)
        response.metadata = metadata
    return response


def _cache_label(value: object, name: str) -> str | None:
    """Validate a caller-set ``cache_group`` label: ``None`` or a non-empty string."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string or None (got {value!r})")
    return value


def _pop_cache_label(request: dict[str, Any]) -> str | None:
    """Remove a per-call ``cache_group`` from request config so it never reaches the SDK."""
    return _cache_label(request.pop("cache_group", None), "config['cache_group']")


def _auto_prompt_cache_key(system_text: object) -> str | None:
    """The key ``prompt_cache_key="auto"`` sends: a hash of the instruction actually sent."""
    if isinstance(system_text, str) and system_text:
        return "abl-si-" + hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:16]
    return None


def _diagnostics_miss_reason(response: Any) -> str | None:
    """The ``reason`` from an OpenAI Responses ``prompt_cache_diagnostics`` cache miss.

    Diagnostics appear only when the request asked for them through
    ``prompt_cache_options``. Older SDKs keep the field as a plain dict.
    """
    diagnostics = getattr(response, "prompt_cache_diagnostics", None)
    if isinstance(diagnostics, Mapping):
        kind, reason = diagnostics.get("type"), diagnostics.get("reason")
    else:
        kind = getattr(diagnostics, "type", None)
        reason = getattr(diagnostics, "reason", None)
    if kind == "cache_miss" and isinstance(reason, str) and reason:
        return reason
    return None


def _has_system_message(messages: list[Any]) -> bool:
    """Return True if ``messages`` already contains a system-role entry."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return True
    return False
