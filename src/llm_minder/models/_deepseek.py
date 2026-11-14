"""DeepSeekModel and its request helpers."""

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from ..base import LLMResponse
from ..strategies.errors import (
    StructuredOutputValidationError,
)
from ._openai_compatible import OpenAICompatibleModel
from ._responses import _ResponsesSurface

# Bound for annotations only, so typing.get_type_hints() resolves them. Code reads
# _sdk.<name> at call time; tests/test_models_sdk_wiring.py enforces that.
from ._sdk import AsyncOpenAI
from ._shared import MetadataExtractor

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


def _merge_thinking(
    extra_body: dict[str, Any] | None, thinking: bool | None
) -> dict[str, Any] | None:
    """Fold a DeepSeek ``thinking`` toggle into an ``extra_body`` dict.

    ``thinking=None`` leaves ``extra_body`` untouched (use the model's default).
    ``True``/``False`` map to DeepSeek's ``{"thinking": {"type": "enabled"}}`` /
    ``{"type": "disabled"}`` request field. An explicit ``thinking`` key already
    present in ``extra_body`` wins.
    """
    if thinking is None:
        return extra_body
    merged = dict(extra_body) if extra_body else {}
    merged.setdefault("thinking", {"type": "enabled" if thinking else "disabled"})
    return merged


def _schema_name(value: str) -> str:
    """Validate the portable Responses API schema-name subset."""
    if not value or len(value) > 64 or any(not (char.isalnum() or char in "_-") for char in value):
        raise ValueError(
            "schema_name must be 1-64 alphanumeric, underscore, or hyphen characters "
            f"(got {value!r})."
        )
    return value


def _prepare_response_schema(
    response_schema: type[BaseModel] | Mapping[str, Any],
    schema_name: str | None,
) -> tuple[dict[str, Any], str, str, str]:
    """Return a JSON-safe schema, request name, identity, and canonical hash."""
    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        schema = response_schema.model_json_schema()
        name = _schema_name(schema_name or response_schema.__name__)
        identity = f"{response_schema.__module__}.{response_schema.__qualname__}"
    elif isinstance(response_schema, Mapping):
        schema = dict(response_schema)
        title = schema.get("title")
        default_name = title if isinstance(title, str) and title else "structured_output"
        name = _schema_name(schema_name or default_name)
        identity = name
    else:
        raise TypeError("response_schema must be a Pydantic BaseModel class or JSON Schema mapping")

    try:
        canonical = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"response_schema must be JSON-serializable: {exc}") from exc
    normalized = json.loads(canonical)
    if not isinstance(normalized, dict):  # Defensive; ``schema`` starts as a mapping.
        raise TypeError("response_schema must serialize to a JSON object")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return normalized, name, identity, digest


def _responses_thinking(
    extra_body: dict[str, Any] | None,
    thinking: bool | None,
) -> dict[str, Any] | None:
    """Map DeepSeek's chat ``thinking`` toggle to Responses ``reasoning``."""
    merged = dict(extra_body) if extra_body else {}
    chat_thinking = merged.pop("thinking", None)
    if "reasoning" not in merged:
        if thinking is not None:
            merged["reasoning"] = {"effort": "high" if thinking else "none"}
        elif isinstance(chat_thinking, Mapping):
            kind = chat_thinking.get("type")
            if kind == "enabled":
                merged["reasoning"] = {"effort": "high"}
            elif kind == "disabled":
                merged["reasoning"] = {"effort": "none"}
    return merged or None


class DeepSeekModel(_ResponsesSurface, OpenAICompatibleModel):
    """LLM model backed by DeepSeek's OpenAI-compatible API.

    Points at ``https://api.deepseek.com`` and reads
    ``DEEPSEEK_API_KEY`` in :meth:`from_api_key`. Model ids are bare DeepSeek
    names, e.g. ``"deepseek-v4-flash"`` or ``"deepseek-v4-pro"``.

    DeepSeek's automatic context cache reports hits at the **top level** of the
    ``usage`` object (``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``)
    rather than under OpenAI's nested ``prompt_tokens_details.cached_tokens`` —
    so this subclass overrides :meth:`_extract_tokens` to surface them in
    ``cached_input_tokens``. Use :attr:`CachedTokenRates.DEEPSEEK` (~2%) when
    computing billable tokens.

    (Calling DeepSeek *through OpenRouter* uses :class:`OpenRouterModel`
    instead; the native cache fields aren't reliably forwarded there, which is
    why direct access via this class gives better cache telemetry.)

    **Thinking mode.** DeepSeek's V4 models (``deepseek-v4-flash`` /
    ``deepseek-v4-pro``) default to *thinking*, which for a batch
    classification job is a surprising, expensive default — thinking can emit
    several times the output tokens (and cost, and latency) of non-thinking.
    Pass ``thinking=False`` to force non-thinking mode explicitly rather than
    relying on the ``deepseek-chat`` (non-thinking) / ``deepseek-reasoner``
    (thinking) aliases, which DeepSeek is deprecating. Under the hood this sends
    ``extra_body={"thinking": {"type": "disabled"}}`` for Chat Completions,
    or ``reasoning={"effort": "none"}`` on the Responses API.

    **Strict structured output.** Set ``api_surface="responses"`` and pass a
    Pydantic model class or JSON Schema mapping as ``response_schema``. The
    schema is sent through ``text.format.type="json_schema"`` and
    :class:`DeepSeekStrategy` parses successful output automatically. DeepSeek
    validates the requested surface and schema locally; model IDs are passed
    through for provider validation without an implicit surface fallback.

    Example:
        >>> model = DeepSeekModel.from_api_key(
        ...     "deepseek-v4-flash", api_key="sk-...", thinking=False
        ... )
        >>> response = await model.generate("Hello!")
        >>> print(response.text, response.cached_input_tokens)

    Added in v0.10.0.
    """

    _default_base_url: str | None = "https://api.deepseek.com"
    _install_extras: str = "deepseek"
    _api_key_env_var: str | None = "DEEPSEEK_API_KEY"

    def __init__(
        self,
        model: str,
        client: "AsyncOpenAI",
        *,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        thinking: bool | None = None,
        api_surface: str = "chat_completions",
        response_schema: type[BaseModel] | Mapping[str, Any] | None = None,
        schema_name: str | None = None,
        json_mode: bool = False,
        metadata_extractors: list[MetadataExtractor] | None = None,
        cache_group: str | None = None,
    ):
        """See :class:`OpenAICompatibleModel`; adds the DeepSeek ``thinking``
        toggle and opt-in Responses API structured output. DeepSeek's prefix
        cache takes no key, so there is no ``prompt_cache_key``; ``cache_group``
        labels calls for the per-group cache statistics."""
        if api_surface not in {"chat_completions", "responses"}:
            raise ValueError(
                f"api_surface must be 'chat_completions' or 'responses' (got {api_surface!r})."
            )
        if response_schema is not None and api_surface != "responses":
            raise ValueError(
                "response_schema requires api_surface='responses'; Chat Completions only "
                "supports json_mode=True without schema enforcement."
            )
        if response_schema is not None and json_mode:
            raise ValueError("Pass response_schema or json_mode=True, not both.")
        if api_surface == "responses":
            try:
                create_response = getattr(getattr(client, "responses", None), "create", None)
            except Exception:
                create_response = None
            if not callable(create_response):
                raise ValueError(
                    "DeepSeek Responses requires a client with callable responses.create. "
                    "Install llm-minder[deepseek] or supply a Responses-capable client."
                ) from None

        self.api_surface = api_surface
        self.response_schema: dict[str, Any] | None = None
        self.response_schema_name: str | None = None
        self.response_schema_identity: str | None = None
        self.response_schema_hash: str | None = None
        self._response_model: type[BaseModel] | None = None
        if response_schema is not None:
            (
                self.response_schema,
                self.response_schema_name,
                self.response_schema_identity,
                self.response_schema_hash,
            ) = _prepare_response_schema(response_schema, schema_name)
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                self._response_model = response_schema

        if api_surface == "responses":
            configured_body = _responses_thinking(extra_body, thinking)
            configured_body = dict(configured_body) if configured_body else {}
            if self.response_schema is not None:
                requested_format = {
                    "type": "json_schema",
                    "name": self.response_schema_name,
                    "schema": self.response_schema,
                }
                text_config = configured_body.get("text")
                if text_config is not None and not isinstance(text_config, Mapping):
                    raise TypeError("extra_body['text'] must be a mapping")
                existing_format = text_config.get("format") if text_config else None
                if existing_format is not None and existing_format != requested_format:
                    raise ValueError("extra_body['text']['format'] conflicts with response_schema")
                configured_body["text"] = {**dict(text_config or {}), "format": requested_format}
            elif json_mode:
                text_config = configured_body.get("text")
                if text_config is not None and not isinstance(text_config, Mapping):
                    raise TypeError("extra_body['text'] must be a mapping")
                configured_body["text"] = {
                    "format": {"type": "json_object"},
                    **dict(text_config or {}),
                }
            effective_extra_body = configured_body or None
        else:
            effective_extra_body = _merge_thinking(extra_body, thinking)
            if json_mode:
                configured_body = {
                    "response_format": {"type": "json_object"},
                    **dict(effective_extra_body or {}),
                }
                effective_extra_body = configured_body

        super().__init__(
            model,
            client,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_body=effective_extra_body,
            metadata_extractors=metadata_extractors,
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
        thinking: bool | None = None,
        api_surface: str = "chat_completions",
        response_schema: type[BaseModel] | Mapping[str, Any] | None = None,
        schema_name: str | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        cache_group: str | None = None,
        **client_kwargs: Any,
    ) -> "DeepSeekModel":
        """Build a DeepSeekModel; reads ``DEEPSEEK_API_KEY`` when ``api_key`` is
        None. Adds the ``thinking`` toggle and Responses API structured-output
        options (see the class docstring) on top of the shared constructor."""
        return super().from_api_key(
            model,
            api_key,
            base_url=base_url,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_body=extra_body,
            # DeepSeekModel applies JSON mode according to the selected API
            # surface; suppress the base class's Chat-only injection.
            json_mode=False,
            max_connections=max_connections,
            metadata_extractors=metadata_extractors,
            cache_group=cache_group,
            _instance_kwargs={
                "thinking": thinking,
                "api_surface": api_surface,
                "response_schema": response_schema,
                "schema_name": schema_name,
                "json_mode": json_mode,
            },
            **client_kwargs,
        )

    @property
    def artifact_identity_extra(self) -> dict[str, Any]:
        """Replay identity fields that distinguish transport/output contracts."""
        extra: dict[str, Any] = {"api_surface": self.api_surface}
        if self.response_schema_hash is not None:
            extra["response_schema"] = {
                "name": self.response_schema_name,
                "identity": self.response_schema_identity,
                "sha256": self.response_schema_hash,
            }
        elif self._uses_json_mode:
            extra["output_format"] = "json_object"
        return extra

    @property
    def _uses_json_mode(self) -> bool:
        body = self._default_extra_body or {}
        if self.api_surface == "responses":
            text_config = body.get("text")
            output_format = text_config.get("format") if isinstance(text_config, Mapping) else None
        else:
            output_format = body.get("response_format")
        return isinstance(output_format, Mapping) and output_format.get("type") == "json_object"

    def parse_structured_response(self, response: LLMResponse) -> Any:
        """Parse provider-enforced output into its Pydantic type or JSON value."""
        try:
            if self._response_model is not None:
                return self._response_model.model_validate_json(response.text)
            return json.loads(response.text)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise StructuredOutputValidationError(
                f"DeepSeek returned output that did not validate against "
                f"{self.response_schema_name!r}: {exc}"
            ) from exc

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = None,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call the selected DeepSeek API surface and normalize its response."""
        if self.api_surface == "chat_completions":
            result = await super().generate(
                prompt,
                temperature=temperature,
                system_instruction=system_instruction,
                config=config,
            )
            metadata = dict(result.metadata or {})
            metadata["api_surface"] = self.api_surface
            request_id = getattr(result.raw, "id", None)
            if isinstance(request_id, str) and request_id:
                metadata["provider_request_id"] = request_id
            if self._uses_json_mode:
                metadata["output_format"] = "json_object"
            result.metadata = metadata
            return result
        return await self._generate_responses(
            prompt,
            temperature=temperature,
            system_instruction=system_instruction,
            config=config,
        )

    def _extract_tokens(self, response: Any) -> tuple[int, int, int, int]:
        """Extract tokens, preferring DeepSeek's native cache-hit field.

        Falls back to the OpenAI-shaped extraction for the
        input/output/total counts, then overrides the cached count with
        ``usage.prompt_cache_hit_tokens`` when DeepSeek provides it.
        """
        input_tokens, output_tokens, total_tokens, cached_tokens = super()._extract_tokens(response)

        usage = getattr(response, "usage", None)
        if usage is not None:
            hit = getattr(usage, "prompt_cache_hit_tokens", None)
            if hit is not None:
                cached_tokens = int(hit or 0)

        return input_tokens, output_tokens, total_tokens, cached_tokens
