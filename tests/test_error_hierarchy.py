"""Every exception type the library exports shares AsyncBatchLLMError."""

from __future__ import annotations

import inspect
import pickle

import pytest

import llm_minder
from llm_minder import AsyncBatchLLMError

EXPORTED = sorted(
    (
        (name, obj)
        for name in llm_minder.__all__
        if inspect.isclass(obj := getattr(llm_minder, name)) and issubclass(obj, BaseException)
    ),
    key=lambda pair: pair[0],
)


def test_every_exported_exception_type_uses_the_base():
    missing = [name for name, cls in EXPORTED if not issubclass(cls, AsyncBatchLLMError)]
    assert missing == []
    assert len(EXPORTED) >= 25


@pytest.mark.parametrize(
    ("name", "builtin"),
    [
        ("ItemDeadlineExceeded", TimeoutError),
        ("BatchDeadlineExceeded", TimeoutError),
        ("FrameworkTimeoutError", TimeoutError),
        ("ArtifactError", RuntimeError),
        ("BatchAbortedError", RuntimeError),
        ("LLMCallError", RuntimeError),
        ("EmptyResponseError", ValueError),
        ("MiddlewareContractError", ValueError),
        ("ResultSerializationError", ValueError),
    ],
)
def test_existing_builtin_bases_are_kept_first(name, builtin):
    cls = getattr(llm_minder, name)
    assert issubclass(cls, builtin)
    mro = cls.__mro__
    assert mro.index(builtin) < mro.index(AsyncBatchLLMError)


def test_catching_the_base_and_pickling_still_work():
    for exc in (
        llm_minder.ItemDeadlineExceeded("item deadline"),
        llm_minder.ArtifactIOError("disk full"),
        llm_minder.BatchBudgetExceeded("over budget"),
    ):
        with pytest.raises(AsyncBatchLLMError):
            raise exc
        copy = pickle.loads(pickle.dumps(exc))
        assert type(copy) is type(exc) and str(copy) == str(exc)
