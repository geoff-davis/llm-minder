"""async-batch-llm is now llm-minder.

This package exists so code that still says ``import async_batch_llm`` keeps working.
Importing it warns once, then makes ``async_batch_llm`` and every
``async_batch_llm.<submodule>`` the same module objects as their ``llm_minder``
counterparts, so classes, isinstance checks and exceptions are shared.

Replace the ``async-batch-llm`` dependency with ``llm-minder`` and import
``llm_minder`` instead. See https://geoff-davis.github.io/llm-minder/migration/v1.0/.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType

_OLD = "async_batch_llm"
_NEW = "llm_minder"


class _AliasLoader(importlib.abc.Loader):
    """Hands back the already-imported ``llm_minder`` module instead of a copy."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._real_spec: importlib.machinery.ModuleSpec | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._target)
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The target module has already run. importlib just replaced its __spec__
        # with the alias spec; put the real one back so reload() and other
        # introspection of llm_minder.<name> keep using its own source loader.
        module.__spec__ = self._real_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves ``async_batch_llm.<name>`` to ``llm_minder.<name>``."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_OLD + "."):
            return None
        new_name = _NEW + fullname[len(_OLD) :]
        real = importlib.util.find_spec(new_name)
        if real is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(new_name),
            is_package=real.submodule_search_locations is not None,
        )


def _importer_stacklevel() -> int:
    """Point the warning at the code doing the import.

    warnings already skips importlib's _bootstrap frames; this also skips
    importlib.import_module, so a dynamic import is attributed to its caller.
    """
    level = 2
    frame = sys._getframe(2)  # 0 is this function, 1 the module body
    while frame is not None:
        filename = frame.f_code.co_filename
        if "importlib" in filename and "_bootstrap" in filename:
            pass  # warnings doesn't count these frames
        elif filename == importlib.__file__:
            level += 1
        else:
            break
        frame = frame.f_back  # type: ignore[assignment]
    return level


warnings.warn(
    "async-batch-llm has been renamed to llm-minder. Depend on llm-minder and "
    "import llm_minder instead; async_batch_llm is only an alias for it. See "
    "https://geoff-davis.github.io/llm-minder/migration/v1.0/",
    DeprecationWarning,
    stacklevel=_importer_stacklevel(),
)

if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

import llm_minder  # noqa: E402

sys.modules[__name__] = llm_minder
