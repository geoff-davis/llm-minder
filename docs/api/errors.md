# Errors, Classification and Rate Limits

Every exception type the library exports subclasses
[`AsyncBatchLLMError`](core.md#asyncbatchllmerror) and keeps its built-in base
(`TimeoutError`, `RuntimeError` or `ValueError`), so existing `except` clauses
still match. `BatchBudgetExceeded` is documented with the budget settings on the
[core page](core.md#batchbudgetexceeded), `LLMCallError` with
[single calls](single-call-pool.md#llmcallerror), and the artifact errors on the
[artifacts page](artifacts.md#artifact-errors).

For which category each built-in classifier assigns, see
[Provider classifier categories](strategies.md#provider-classifier-categories).

## Error classification

### ErrorClassifier

::: llm_minder.strategies.ErrorClassifier

### ErrorInfo

::: llm_minder.ErrorInfo

### DefaultErrorClassifier

::: llm_minder.DefaultErrorClassifier

### GeminiErrorClassifier

::: llm_minder.GeminiErrorClassifier

### OpenAIErrorClassifier

::: llm_minder.OpenAIErrorClassifier

### OpenRouterErrorClassifier

::: llm_minder.OpenRouterErrorClassifier

### PydanticAIErrorClassifier

::: llm_minder.PydanticAIErrorClassifier

## Rate-limit strategies

`RateLimitConfig` controls cooldown and slow-start; a `RateLimitStrategy`
decides how long each cooldown lasts.

### RateLimitStrategy

::: llm_minder.RateLimitStrategy

### ExponentialBackoffStrategy

::: llm_minder.ExponentialBackoffStrategy

### FixedDelayStrategy

::: llm_minder.FixedDelayStrategy

## Deadlines, aborts and interruptions

::: llm_minder.ItemDeadlineExceeded

::: llm_minder.BatchDeadlineExceeded

::: llm_minder.BatchAbortedError

::: llm_minder.BatchInterruptedError

::: llm_minder.FrameworkTimeoutError

::: llm_minder.RateLimitRetriesExceeded

## Provider responses and structured output

::: llm_minder.EmptyResponseError

::: llm_minder.TruncatedResponseError

::: llm_minder.ProviderResponseError

::: llm_minder.StructuredOutputSchemaError

::: llm_minder.StructuredOutputValidationError

::: llm_minder.TokenTrackingError

## Configuration errors

::: llm_minder.MiddlewareContractError

::: llm_minder.QuotaScopeError

::: llm_minder.TokenEstimationError

::: llm_minder.TokenEstimatorRequired

::: llm_minder.TokenEstimateExceedsLimit
