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

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return importlib.import_module(self._target)

    def exec_module(self, module: ModuleType) -> None:
        pass  # the target module has already run


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


warnings.warn(
    "async-batch-llm has been renamed to llm-minder. Depend on llm-minder and "
    "import llm_minder instead; async_batch_llm is only an alias for it. See "
    "https://geoff-davis.github.io/llm-minder/migration/v1.0/",
    DeprecationWarning,
    # warnings skips importlib's own frames, so 2 is the line doing the import.
    stacklevel=2,
)

if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

import llm_minder  # noqa: E402

sys.modules[__name__] = llm_minder
