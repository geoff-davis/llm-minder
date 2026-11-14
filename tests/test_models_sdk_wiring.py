"""The optional SDK handles are patched in one place: ``llm_minder.models._sdk`` (#322).

Before the split, tests patched ``llm_minder.models.AsyncOpenAI`` and friends. With
``models`` a package, any other binding of those names is a second copy: a patch would
change one copy while the code read the other, and the tests patching it would pass or
fail for the wrong reason. Provider modules do bind the names their constructor
annotations use, so ``typing.get_type_hints()`` resolves them, but code must read them as
``_sdk.<name>``. These tests fail if that wiring breaks.
"""

import ast
import importlib
import inspect
import pkgutil
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import patch

import pytest

import llm_minder.models as models_pkg
from llm_minder import llm
from llm_minder.models import (
    DeepSeekModel,
    GeminiCachedModel,
    GeminiModel,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenRouterModel,
    _sdk,
)

SDK_NAMES = ("AsyncOpenAI", "genai", "Content", "_build_openai_http_client", "_close_unused_client")
OPENAI_FAMILY = (OpenAICompatibleModel, OpenAIModel, OpenRouterModel, DeepSeekModel)


class _RecordingFactory:
    """Stands in for an SDK class or helper and records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        # ``responses.create`` lets OpenAIModel's Responses check accept it as a client.
        return SimpleNamespace(
            args=args,
            kwargs=kwargs,
            close=lambda: None,
            responses=SimpleNamespace(create=lambda **_: None),
        )


def _submodules():
    for info in pkgutil.iter_modules(models_pkg.__path__):
        if info.name != "_sdk":
            yield importlib.import_module(f"llm_minder.models.{info.name}")


def test_package_does_not_rebind_sdk_handles():
    # A stale patch("llm_minder.models.AsyncOpenAI") must fail loudly, not patch a copy.
    for name in SDK_NAMES:
        assert not hasattr(models_pkg, name), name
    with pytest.raises(AttributeError), patch("llm_minder.models.AsyncOpenAI"):
        pass


@pytest.mark.parametrize("module", list(_submodules()), ids=lambda m: m.__name__)
def test_provider_modules_read_sdk_handles_through_sdk(module):
    # Annotations are strings, so any Name node for an SDK handle outside the import is
    # code reading the module's own binding, which a patch on _sdk would not reach.
    tree = ast.parse(inspect.getsource(module))
    reads = [
        f"{node.id} at line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in SDK_NAMES
    ]
    assert not reads, f"{module.__name__} reads {reads}; use _sdk.<name>"
    for name in SDK_NAMES:
        bound = getattr(module, name, _sdk)
        assert bound is _sdk or bound is getattr(_sdk, name), (module.__name__, name)


@pytest.mark.parametrize("model_cls", OPENAI_FAMILY, ids=lambda c: c.__name__)
def test_patched_async_openai_is_the_one_from_api_key_calls(model_cls):
    factory = _RecordingFactory()
    with patch.object(_sdk, "AsyncOpenAI", factory):
        model = model_cls.from_api_key("m", api_key="k", base_url="http://localhost:1")
    assert len(factory.calls) == 1
    assert model._client.kwargs["api_key"] == "k"


@pytest.mark.parametrize("model_cls", OPENAI_FAMILY, ids=lambda c: c.__name__)
def test_patched_http_client_builder_is_the_one_from_api_key_calls(model_cls):
    builder = _RecordingFactory()
    factory = _RecordingFactory()
    with (
        patch.object(_sdk, "_build_openai_http_client", builder),
        patch.object(_sdk, "AsyncOpenAI", factory),
    ):
        model_cls.from_api_key("m", api_key="k", base_url="http://localhost:1", max_connections=3)
    assert builder.calls == [((3,), {})]
    assert factory.calls[0][1]["http_client"].args == (3,)


@pytest.mark.parametrize("model_cls", OPENAI_FAMILY, ids=lambda c: c.__name__)
def test_missing_openai_sdk_is_seen_by_every_openai_model(model_cls):
    with patch.object(_sdk, "AsyncOpenAI", None), pytest.raises(ImportError, match="openai"):
        model_cls.from_api_key("m", api_key="k", base_url="http://localhost:1")


@pytest.mark.parametrize(
    "build",
    [
        lambda: GeminiModel("m", client=object()),
        lambda: GeminiCachedModel("m", client=object(), cached_content=[]),
    ],
    ids=["GeminiModel", "GeminiCachedModel"],
)
def test_missing_genai_is_seen_by_every_gemini_model(build):
    with patch.object(_sdk, "genai", None), pytest.raises(ImportError, match="google-genai"):
        build()


def test_factory_reads_the_patched_genai(monkeypatch):
    factory = _RecordingFactory()
    monkeypatch.setattr(_sdk, "genai", SimpleNamespace(Client=factory))
    llm("gemini:m", api_key="k")
    assert factory.calls == [((), {"api_key": "k"})]


def test_factory_reads_the_patched_async_openai():
    factory = _RecordingFactory()
    with patch.object(_sdk, "AsyncOpenAI", factory):
        llm("openai:m", api_key="k")
    assert len(factory.calls) == 1


@pytest.mark.parametrize("model_cls", OPENAI_FAMILY, ids=lambda c: c.__name__)
def test_openai_constructor_annotations_resolve(model_cls):
    # get_type_hints evaluates string annotations in the defining module, so the SDK
    # names must be reachable there (as _sdk.<name>), not only under TYPE_CHECKING.
    openai = pytest.importorskip("openai")
    hints = get_type_hints(model_cls.__init__)
    assert hints["client"] is openai.AsyncOpenAI


def test_gemini_constructor_annotations_resolve():
    genai = pytest.importorskip("google.genai")
    from google.genai.types import Content

    assert get_type_hints(GeminiModel.__init__)["client"] is genai.Client
    hints = get_type_hints(GeminiCachedModel.__init__)
    assert hints["client"] is genai.Client
    assert hints["cached_content"] == list[Content]
