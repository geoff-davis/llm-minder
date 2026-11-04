"""Batch LLM processing utilities for handling bulk LLM requests.

This module provides a flexible framework for processing multiple LLM requests
efficiently using a strategy pattern for provider-agnostic LLM integration.

Key features:
- Strategy pattern for any LLM provider (OpenAI, Anthropic, Google, LangChain, custom)
- Built-in strategies: PydanticAIStrategy, GeminiStrategy, OpenAIStrategy, OpenRouterStrategy
- Built-in models: GeminiModel, GeminiCachedModel, OpenAIModel, OpenRouterModel
- Provider-agnostic error classification
- Pluggable rate limit strategies
- Middleware pipeline for extensibility
- Observer pattern for monitoring
- Configuration-based setup

Example:
    >>> from llm_minder import process_prompts
    >>> from llm_minder.testing import FakeStrategy
    >>> result = await process_prompts(FakeStrategy("hello"), ["Say hello"])
    >>> print(result.outputs)
    ['hello']

Type Aliases:
    For convenience, type aliases are provided to reduce verbosity:

    - ``SimpleBatchProcessor[T]``: Processor with string input, output type T, no context
      Equivalent to ``ParallelBatchProcessor[str, T, None]``

    - ``SimpleWorkItem[T]``: Work item with string input, output type T, no context
      Equivalent to ``LLMWorkItem[str, T, None]``

    - ``SimpleResult[T]``: Result with output type T, no context
      Equivalent to ``WorkItemResult[T, None]``

    Example using type aliases:
        >>> from llm_minder import SimpleBatchProcessor, SimpleWorkItem
        >>>
        >>> async with SimpleBatchProcessor[MyOutput](config=config) as processor:
        ...     await processor.add_work(SimpleWorkItem[MyOutput](
        ...         item_id="item_1",
        ...         strategy=strategy,
        ...         prompt="Process this",
        ...     ))
"""

from typing import Any, TypeVar

from ._internal.cleanup import CleanupInterruptedError
from .artifacts import (
    ArtifactError,
    ArtifactFormatError,
    ArtifactIdentity,
    ArtifactIdentityError,
    ArtifactIOError,
    ArtifactSerializationError,
    ArtifactStore,
    JsonlArtifactStore,
    ResumePolicy,
)

# Core classes
from .base import (
    AttemptTiming,
    BatchInterruptedError,
    BatchResult,
    BatchTermination,
    CachedTokenRates,
    LLMResponse,
    LLMWorkItem,
    PostProcessorFunc,
    ProgressCallbackFunc,
    RetryState,
    StreamFinalizationError,
    TokenUsage,
    WorkItemResult,
    WorkItemTiming,
)
from .budget import AttemptUsage

# Queue-less convenience surfaces (single call + shared call pool), built on the
# same per-item resilience pipeline as the batch processor.
from .cache_groups import CacheFloorBreach, CacheGroupTracker
from .call_pool import LLMCallPool
from .callable_strategy import CallableStrategy, CallOutcome

# Documented error and timeout category vocabulary
from .categories import ErrorCategory, TimeoutCategory

# Classifiers
from .classifiers import (
    GeminiErrorClassifier,
    OpenAIErrorClassifier,
    OpenRouterErrorClassifier,
    PydanticAIErrorClassifier,
)

# Configuration
from .core import (
    AbortMode,
    GuardrailConfig,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    StartupRampConfig,
)

# Protocols
from .core.protocols import LLMModel, ManagedLLMModel

# String-based strategy factory: llm("openai:gpt-6-luna")
from .factory import llm

# LLM call strategies
from .llm_strategies import (
    DeepSeekStrategy,
    GeminiStrategy,
    LLMCallStrategy,
    ModelStrategy,
    OpenAIStrategy,
    OpenRouterStrategy,
    PydanticAIStrategy,
)

# Middleware
from .middleware import BaseMiddleware, Middleware

# Concrete models
from .models import (
    DeepSeekModel,
    GeminiCachedModel,
    GeminiModel,
    MetadataExtractor,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenRouterModel,
)

# Observers
from .observers import BaseObserver, MetricsObserver, ProcessingEvent, ProcessorObserver

# Main processor
from .parallel import ParallelBatchProcessor

# Structured-output parsing helpers
from .parsing import pydantic_json_parser, strip_code_fences
from .provider_calls import attach_token_usage, provider_call

# Provider auxiliary output (typed metadata views)
from .provider_output import Grounding, GroundingSource, ToolCall
from .serialization import ResultSerializationError
from .single import LLMCallError, call, call_result
from .sqlite_artifacts import SqliteArtifactStore, SqliteDurability

# Error classification and rate limit strategies
from .strategies import (
    AsyncBatchLLMError,
    BatchAbortedError,
    BatchAdmissionClosedError,
    BatchBudgetExceeded,
    BatchDeadlineExceeded,
    DefaultErrorClassifier,
    EmptyResponseError,
    ErrorClassifier,
    ErrorInfo,
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    FrameworkTimeoutError,
    ItemDeadlineExceeded,
    LLMMinderError,
    MiddlewareContractError,
    ProviderResponseError,
    QuotaScopeError,
    RateLimitRetriesExceeded,
    RateLimitStrategy,
    StructuredOutputSchemaError,
    StructuredOutputValidationError,
    TokenEstimateExceedsLimit,
    TokenEstimationError,
    TokenEstimatorRequired,
    TokenTrackingError,
    TruncatedResponseError,
)

# High-level streaming API (built on the processor's streaming mode)
from .streaming import process_prompts, process_stream
from .token_estimation import CharacterTokenEstimator, TokenEstimate, TokenEstimator

# Type variable for output type in simplified aliases
_T = TypeVar("_T")

# Type aliases for common use cases
# These reduce verbosity for the most common pattern: string input, typed output, no context
SimpleBatchProcessor = ParallelBatchProcessor[str, _T, None]
"""Type alias for ParallelBatchProcessor[str, T, None].

Use when you have string prompts, a typed output, and no context.

Example:
    async with SimpleBatchProcessor[MyOutput](config=config) as processor:
        ...
"""

SimpleWorkItem = LLMWorkItem[str, _T, None]
"""Type alias for LLMWorkItem[str, T, None].

Use when creating work items with string prompts, typed output, and no context.

Example:
    item = SimpleWorkItem[MyOutput](item_id="1", strategy=strategy, prompt="Hello")
"""

SimpleResult = WorkItemResult[_T, None]
"""Type alias for WorkItemResult[T, None].

Use when working with results that have no context.

Example:
    result: SimpleResult[MyOutput] = results[0]
"""

__all__ = [
    # Core
    "BatchInterruptedError",
    "AttemptTiming",
    "BatchResult",
    "BatchTermination",
    "CacheFloorBreach",
    "CacheGroupTracker",
    "CachedTokenRates",
    "LLMWorkItem",
    "PostProcessorFunc",
    "ProgressCallbackFunc",
    "RetryState",
    "StreamFinalizationError",
    "TokenUsage",
    "TokenEstimate",
    "TokenEstimator",
    "CharacterTokenEstimator",
    "WorkItemResult",
    "WorkItemTiming",
    "ResultSerializationError",
    "CallOutcome",
    "CleanupInterruptedError",
    "CallableStrategy",
    # Audit/checkpoint artifacts
    "ArtifactError",
    "ArtifactIdentityError",
    "ArtifactFormatError",
    "ArtifactIOError",
    "ArtifactIdentity",
    "ArtifactSerializationError",
    "ArtifactStore",
    "JsonlArtifactStore",
    "ResumePolicy",
    "SqliteArtifactStore",
    "SqliteDurability",
    # Configuration
    "ProcessorConfig",
    "AbortMode",
    "AttemptUsage",
    "GuardrailConfig",
    "RateLimitConfig",
    "RetryConfig",
    "StartupRampConfig",
    # High-level convenience API
    "process_prompts",
    "process_stream",
    # Single-call + shared-call surfaces
    "call",
    "call_result",
    "LLMCallError",
    "LLMCallPool",
    # String-based strategy factory
    "llm",
    # LLM Strategies
    "DeepSeekStrategy",
    "GeminiStrategy",
    "LLMCallStrategy",
    "ModelStrategy",
    "OpenAIStrategy",
    "OpenRouterStrategy",
    "PydanticAIStrategy",
    # Models
    "DeepSeekModel",
    "GeminiModel",
    "GeminiCachedModel",
    "OpenAICompatibleModel",
    "OpenAIModel",
    "OpenRouterModel",
    "MetadataExtractor",
    # Provider auxiliary output (typed metadata views)
    "Grounding",
    "GroundingSource",
    "ToolCall",
    # Protocols
    "LLMModel",
    "LLMResponse",
    "ManagedLLMModel",
    # Error Classification Strategies
    "ErrorClassifier",
    "ErrorInfo",
    "DefaultErrorClassifier",
    "EmptyResponseError",
    "TruncatedResponseError",
    "FrameworkTimeoutError",
    "ItemDeadlineExceeded",
    "MiddlewareContractError",
    "BatchBudgetExceeded",
    "BatchDeadlineExceeded",
    "AsyncBatchLLMError",
    "LLMMinderError",
    "ErrorCategory",
    "TimeoutCategory",
    "BatchAbortedError",
    "BatchAdmissionClosedError",
    "ProviderResponseError",
    "QuotaScopeError",
    "RateLimitRetriesExceeded",
    "StructuredOutputSchemaError",
    "StructuredOutputValidationError",
    "TokenEstimationError",
    "TokenEstimatorRequired",
    "TokenEstimateExceedsLimit",
    "TokenTrackingError",
    "RateLimitStrategy",
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    # Middleware
    "Middleware",
    "BaseMiddleware",
    # Observers
    "ProcessorObserver",
    "BaseObserver",
    "MetricsObserver",
    "ProcessingEvent",
    # Classifiers
    "GeminiErrorClassifier",
    "OpenAIErrorClassifier",
    "OpenRouterErrorClassifier",
    "PydanticAIErrorClassifier",
    # Processor
    "ParallelBatchProcessor",
    # Structured-output parsing helpers
    "pydantic_json_parser",
    "strip_code_fences",
    # Helpers for custom execute() overrides
    "attach_token_usage",
    "provider_call",
    # Type aliases (convenience)
    "SimpleBatchProcessor",
    "SimpleWorkItem",
    "SimpleResult",
]

# Version is read from package metadata (single source of truth in pyproject.toml)
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-minder")
except PackageNotFoundError:
    # Package not installed (e.g., running from source in development)
    __version__ = "0.0.0+dev"


# Names removed from the public API in 1.0. Looking one up raises the usual
# AttributeError (so ``from llm_minder import X`` raises ImportError), with
# a pointer to the replacement.
_REMOVED_NAMES: dict[str, str] = {
    "LLMGateway": "LLMGateway was removed in 1.0; use LLMCallPool, the same class.",
    "BatchProcessor": (
        "BatchProcessor was removed from the public API in 1.0; use "
        "ParallelBatchProcessor, its only implementation."
    ),
    "ProcessingStats": (
        "ProcessingStats was removed from the public API in 1.0; read statistics "
        "from ParallelBatchProcessor.get_stats(), which returns a dict."
    ),
    "grounding_metadata_extractor": (
        "grounding_metadata_extractor was removed from the public API in 1.0; "
        "built-in Gemini models already put grounding in metadata['grounding']."
    ),
}


def __getattr__(name: str) -> Any:
    if name in _REMOVED_NAMES:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. {_REMOVED_NAMES[name]}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
