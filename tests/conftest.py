"""Shared fixtures for the test suite.

The retry/rate-limit fixtures exist to keep tests fast: the library's
default ``RetryConfig(initial_wait=1.0)`` means any test that exercises a
retry against the defaults pays 1-2s of real sleep per retry. Tests that
don't specifically assert timing behavior should use these.
"""

import os

import pytest

from llm_minder.core import RateLimitConfig, RetryConfig

# Property-based tests (#327). The default ``ci`` profile is derandomized and
# small, so the default run is deterministic and fast; it keeps no example
# database, so a failure is pinned as an explicit ``@example`` or regression
# test rather than replayed from local state. Tests marked ``slow`` apply the
# ``thorough`` profile explicitly. HYPOTHESIS_PROFILE overrides the default.
# Hypothesis is a dev-group dependency. Jobs that install only selected packages (the
# sdk-compat matrix) don't collect the property tests, so the profiles are registered only
# when it is importable.
try:
    from hypothesis import HealthCheck, settings
except ImportError:  # pragma: no cover - isolated SDK jobs
    pass
else:
    settings.register_profile(
        "ci",
        derandomize=True,
        deadline=None,
        database=None,
        max_examples=150,
        stateful_step_count=30,
        suppress_health_check=[HealthCheck.too_slow],
        print_blob=True,
    )
    settings.register_profile(
        "thorough",
        deadline=None,
        database=None,
        max_examples=600,
        stateful_step_count=80,
        suppress_health_check=[HealthCheck.too_slow],
        print_blob=True,
    )
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture
def fast_retry() -> RetryConfig:
    """Retry config with millisecond waits (deterministic: jitter off)."""
    return RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False)


@pytest.fixture
def fast_rate_limit() -> RateLimitConfig:
    """Rate-limit config with a near-zero cooldown and no slow-start ramp."""
    return RateLimitConfig(
        cooldown_seconds=0.01,
        slow_start_items=0,
        slow_start_initial_delay=0.0,
        slow_start_final_delay=0.0,
        backoff_multiplier=1.0,
    )
