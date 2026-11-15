"""Anthropic-specific error classification.

Handles exceptions raised by the ``anthropic`` Python SDK (1.x and later):
typed exceptions and HTTP status codes first, message patterns last.

- 429 is a rate limit, with ``retry-after-ms`` / ``retry-after`` as the
  suggested wait. A 429 whose ``error_code`` is
  ``enforced_spend_limit_reached`` is billing exhaustion and fails fast.
- 529 (``overloaded_error``) is provider-wide capacity, so it is treated like a
  rate limit: every worker pauses in the shared cooldown instead of each item
  retrying into the overload. Its category is ``server_overload``.
- 500/502/503/504, connection errors and timeouts are retried.
- 401, 403, 404, 400, 413 and 422 are deterministic and not retried.

Provisional (new in 1.2, #314).
"""

from __future__ import annotations

import asyncio

from ..strategies.errors import (
    ErrorClassifier,
    ErrorInfo,
    _retry_after_seconds,
    http_status,
    matches_any_pattern,
)

RATE_LIMIT_PATTERNS = ("429", "rate limit", "rate_limit_error", "too many requests")
OVERLOAD_PATTERNS = ("529", "overloaded", "overloaded_error")
TIMEOUT_PATTERNS = ("timeout", "timed out", "deadline")
NETWORK_PATTERNS = ("connection", "network", "econnreset", "broken pipe")

# error.details.error_code values (anthropic SDK attribute ``error_code``) that
# mean the account can't spend more: retrying cannot succeed.
_SPEND_LIMIT_CODES = frozenset({"enforced_spend_limit_reached"})

_SPEND_LIMIT_HINT = (
    "The Anthropic workspace or organization reached its enforced spend limit. "
    "Raise the limit in the Claude Console. Not retryable."
)
_OVERLOAD_STATUS = 529


def _error_code(exception: Exception) -> object:
    """``error.details.error_code`` from the SDK attribute or, on SDKs without it, the body."""
    code = getattr(exception, "error_code", None)
    if isinstance(code, str):
        return code
    body = getattr(exception, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    return details.get("error_code") if isinstance(details, dict) else None


class AnthropicErrorClassifier(ErrorClassifier):
    """Classifier for Anthropic SDK exceptions plus generic fallbacks."""

    # Transient server-side failures.
    _RETRYABLE_STATUS = frozenset({408, 500, 502, 503, 504})
    # Deterministic client errors.
    _NON_RETRYABLE_STATUS = frozenset({400, 404, 405, 409, 410, 412, 413, 422, 431})

    def classify(self, exception: Exception) -> ErrorInfo:
        info = self._framework_prelude(exception)
        if info is not None:
            return info

        info = self._classify_anthropic_exception(exception)
        if info is not None:
            return info

        # A non-SDK exception carrying a status, then validation and
        # programming-error types, all outrank message text.
        status = http_status(exception)
        if status is not None:
            return self._classify_status(status, exception)
        info = self._typed_failure(exception)
        if info is not None:
            return info

        error_str = str(exception)
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError)) or matches_any_pattern(
            error_str, TIMEOUT_PATTERNS
        ):
            return ErrorInfo(True, False, True, "api_timeout")
        if isinstance(exception, ConnectionError) or matches_any_pattern(
            error_str, NETWORK_PATTERNS
        ):
            return ErrorInfo(True, False, False, "network_error")
        if matches_any_pattern(error_str, OVERLOAD_PATTERNS):
            return ErrorInfo(True, True, False, "server_overload")
        if matches_any_pattern(error_str, RATE_LIMIT_PATTERNS):
            return ErrorInfo(True, True, False, "rate_limit")
        return self._generic_tail(exception)

    def _classify_anthropic_exception(self, exception: Exception) -> ErrorInfo | None:
        """Return ErrorInfo for anthropic-SDK exceptions, or None to defer."""
        try:
            import anthropic
        except ImportError:
            return None

        # APITimeoutError subclasses APIConnectionError, so it goes first.
        if isinstance(exception, anthropic.APITimeoutError):
            return ErrorInfo(True, False, True, "api_timeout")
        if isinstance(exception, anthropic.APIConnectionError):
            return ErrorInfo(True, False, False, "network_error")
        if isinstance(exception, anthropic.APIStatusError):
            status = getattr(exception, "status_code", None)
            if not isinstance(status, int) or isinstance(status, bool):
                # SDK-shaped doubles may omit a concrete status; trust the type.
                status = 429 if isinstance(exception, anthropic.RateLimitError) else None
            if status is None:
                return ErrorInfo(True, False, False, "api_error")
            return self._classify_status(status, exception)
        # Credentials that can't be loaded locally fail before any request.
        credentials_error = getattr(anthropic, "CredentialsError", None)
        if isinstance(credentials_error, type) and isinstance(exception, credentials_error):
            return ErrorInfo(False, False, False, "authentication")
        return None

    def _classify_status(self, status: int, exception: Exception) -> ErrorInfo:
        """Branch on an HTTP status code."""
        if status == 429:
            if _error_code(exception) in _SPEND_LIMIT_CODES:
                return ErrorInfo(
                    False, False, False, "insufficient_balance", hint=_SPEND_LIMIT_HINT
                )
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )
        if status == _OVERLOAD_STATUS:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="server_overload",
                suggested_wait=_retry_after_seconds(exception),
            )
        if status == 401:
            return ErrorInfo(False, False, False, "authentication")
        if status == 403:
            return ErrorInfo(False, False, False, "permission_denied")
        if status in self._RETRYABLE_STATUS:
            return ErrorInfo(True, False, status in (408, 504), "server_error")
        if status in self._NON_RETRYABLE_STATUS:
            return ErrorInfo(False, False, False, "client_error")
        if 500 <= status < 600:
            return ErrorInfo(True, False, False, "server_error")
        # Unrecognized status: be conservative and retry.
        return ErrorInfo(True, False, False, "api_error")
