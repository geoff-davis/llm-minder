"""attempt_timeout, which replaced timeout_per_item (issue #98; the old name was
removed in 1.0, see test_removed_in_1_0.py)."""

import warnings

import pytest

from async_batch_llm import ProcessorConfig


class TestNewName:
    def test_attempt_timeout_default(self):
        config = ProcessorConfig()
        assert config.attempt_timeout == 120.0

    def test_attempt_timeout_explicit(self):
        config = ProcessorConfig(attempt_timeout=45.0)
        assert config.attempt_timeout == 45.0

    def test_new_name_emits_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            config = ProcessorConfig(attempt_timeout=45.0)
        assert config.attempt_timeout == 45.0

    def test_attempt_timeout_validated(self):
        with pytest.raises(ValueError, match="attempt_timeout must be > 0"):
            ProcessorConfig(attempt_timeout=0)
