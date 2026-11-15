"""Every branch of AnthropicErrorClassifier (#314), with real anthropic SDK exceptions."""

from __future__ import annotations

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from llm_minder import (  # noqa: E402
    AnthropicErrorClassifier,
    EmptyResponseError,
    ErrorCategory,
    FrameworkTimeoutError,
    TruncatedResponseError,
)

_URL = "https://api.anthropic.com/v1/messages"


def _request():
    return httpx2.Request("POST", _URL)


def _status_error(status, cls=None, *, error_type="api_error", headers=None, details=None):
    error = {"type": error_type, "message": "m"}
    if details is not None:
        error["details"] = details
    body = {"type": "error", "error": error}
    response = httpx2.Response(status, request=_request(), json=body, headers=headers or {})
    return (cls or anthropic.APIStatusError)(f"Error code: {status}", response=response, body=body)


def _classify(exception):
    return AnthropicErrorClassifier().classify(exception)


def _summary(info):
    return info.is_retryable, info.is_rate_limit, info.is_timeout, info.error_category


@pytest.mark.parametrize(
    ("status", "cls", "expected"),
    [
        (429, anthropic.RateLimitError, (True, True, False, "rate_limit")),
        (529, anthropic.APIStatusError, (True, True, False, "server_overload")),
        (500, anthropic.InternalServerError, (True, False, False, "server_error")),
        (502, anthropic.APIStatusError, (True, False, False, "server_error")),
        (503, anthropic.APIStatusError, (True, False, False, "server_error")),
        (504, anthropic.APIStatusError, (True, False, True, "server_error")),
        (408, anthropic.APIStatusError, (True, False, True, "server_error")),
        (599, anthropic.APIStatusError, (True, False, False, "server_error")),
        (401, anthropic.AuthenticationError, (False, False, False, "authentication")),
        (403, anthropic.PermissionDeniedError, (False, False, False, "permission_denied")),
        (404, anthropic.NotFoundError, (False, False, False, "client_error")),
        (400, anthropic.BadRequestError, (False, False, False, "client_error")),
        (413, anthropic.APIStatusError, (False, False, False, "client_error")),
        (422, anthropic.UnprocessableEntityError, (False, False, False, "client_error")),
        (418, anthropic.APIStatusError, (True, False, False, "api_error")),
    ],
)
def test_status_codes(status, cls, expected):
    assert _summary(_classify(_status_error(status, cls))) == expected


def test_overloaded_error_subclass_is_a_shared_cooldown():
    overloaded = getattr(anthropic, "OverloadedError", anthropic.APIStatusError)
    info = _classify(_status_error(529, overloaded, error_type="overloaded_error"))
    assert info.is_rate_limit and info.is_retryable
    assert info.error_category == ErrorCategory.SERVER_OVERLOAD


@pytest.mark.parametrize(
    ("headers", "wait"),
    [
        ({"retry-after": "12"}, 12.0),
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),
        ({"retry-after": "soon"}, None),
        ({}, None),
    ],
)
def test_rate_limit_retry_after(headers, wait):
    info = _classify(_status_error(429, anthropic.RateLimitError, headers=headers))
    assert info.suggested_wait == wait


def test_overload_retry_after():
    info = _classify(_status_error(529, headers={"retry-after": "3"}))
    assert info.suggested_wait == 3.0


def test_spend_limit_429_is_billing_exhaustion():
    error = _status_error(
        429,
        anthropic.RateLimitError,
        error_type="rate_limit_error",
        details={"error_code": "enforced_spend_limit_reached"},
    )
    info = _classify(error)
    assert _summary(info) == (False, False, False, "insufficient_balance")
    assert "spend limit" in info.hint


def test_status_error_without_int_status_trusts_type():
    error = _status_error(429, anthropic.RateLimitError)
    error.status_code = None
    assert _summary(_classify(error)) == (True, True, False, "rate_limit")
    other = _status_error(500)
    other.status_code = "500"
    assert _summary(_classify(other)) == (True, False, False, "api_error")


def test_timeout_and_connection_errors():
    timeout = anthropic.APITimeoutError(request=_request())
    connection = anthropic.APIConnectionError(request=_request())
    assert _summary(_classify(timeout)) == (True, False, True, "api_timeout")
    assert _summary(_classify(connection)) == (True, False, False, "network_error")


def test_credentials_error_fails_fast():
    credentials_error = getattr(anthropic, "CredentialsError", None)
    if credentials_error is None:
        pytest.skip("this anthropic release has no CredentialsError")
    assert _summary(_classify(credentials_error("no credentials"))) == (
        False,
        False,
        False,
        "authentication",
    )


def test_framework_exceptions_keep_their_categories():
    assert _classify(FrameworkTimeoutError("t")).error_category == "framework_timeout"
    assert _classify(TruncatedResponseError("t")).error_category == "response_truncated"
    assert _classify(EmptyResponseError("e")).error_category == "empty_response"


class _StatusException(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    ("status", "category", "rate_limit"),
    [(429, "rate_limit", True), (529, "server_overload", True), (401, "authentication", False)],
)
def test_non_sdk_exception_status_outranks_message(status, category, rate_limit):
    info = _classify(_StatusException("connection timeout text", status))
    assert (info.error_category, info.is_rate_limit) == (category, rate_limit)


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (TimeoutError("x"), (True, False, True, "api_timeout")),
        (ConnectionError("x"), (True, False, False, "network_error")),
        (RuntimeError("Request timed out"), (True, False, True, "api_timeout")),
        (RuntimeError("connection reset"), (True, False, False, "network_error")),
        (RuntimeError("Overloaded"), (True, True, False, "server_overload")),
        (RuntimeError("rate limit hit"), (True, True, False, "rate_limit")),
        (ValueError("bad"), (False, False, False, "logic_error")),
        (RuntimeError("something else"), (True, False, False, "unknown")),
    ],
)
def test_untyped_fallbacks(exception, expected):
    assert _summary(_classify(exception)) == expected


def test_categories_are_documented():
    produced = {
        "rate_limit",
        "server_overload",
        "server_error",
        "api_error",
        "api_timeout",
        "network_error",
        "authentication",
        "permission_denied",
        "client_error",
        "insufficient_balance",
    }
    assert produced <= {member.value for member in ErrorCategory}
