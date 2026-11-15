"""Concrete LLM model implementations.

Model classes wrap a provider's client and handle API calls, token extraction,
and response normalization. Strategies receive an LLMModel and call generate()
without knowing about provider-specific details.

Added in v0.6.0. Split from one ``models.py`` into this package in v1.1 (#322);
import from ``llm_minder.models`` (or ``llm_minder``), not from the private
submodules. The optional SDK handles (``AsyncOpenAI``, ``AsyncAnthropic``, ``genai``) live in
``llm_minder.models._sdk``, and every provider module reads them there at call
time, so that is the one place to patch them.
"""

# Re-exports. There is no ``__all__``, as before the split. The private helpers
# are here because tests import them; they are called, never patched.
from ._anthropic import (  # noqa: F401
    AnthropicModel,
)
from ._deepseek import (  # noqa: F401
    DeepSeekModel,
)
from ._gemini import (  # noqa: F401
    GeminiCachedModel,
    GeminiModel,
)
from ._openai import (  # noqa: F401
    OpenAIModel,
)
from ._openai_compatible import (  # noqa: F401
    OpenAICompatibleModel,
)
from ._openrouter import (  # noqa: F401
    OpenRouterModel,
)
from ._shared import (  # noqa: F401
    MetadataExtractor,
    _coerce_to_messages,
    _decode_tags_from_display_name,
    _diagnostics_miss_reason,
    _encode_tags_to_display_name,
    _extract_metadata,
    _extract_tokens,
    _has_system_message,
    _openai_cache_write_tokens,
    _run_extractors,
    grounding_metadata_extractor,
)
