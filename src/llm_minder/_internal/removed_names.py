"""Errors for public names removed in 1.0, raised from a module ``__getattr__``.

``from llm_minder import LLMGateway`` reaches the module ``__getattr__``, but
CPython's ``IMPORT_FROM`` swallows any ``AttributeError`` it raises and builds a
generic "cannot import name" ``ImportError`` without context, so the hint
naming the replacement would be lost in the most common upgrade form. When the
lookup comes from an ``IMPORT_FROM`` instruction, raise the ``ImportError``
ourselves with the hint; every other lookup (attribute access, ``getattr``,
``hasattr``) still gets an ``AttributeError``, so ``hasattr()`` returns False
and ``getattr(module, name, default)`` returns the default.
"""

from __future__ import annotations

import sys


def _looked_up_by_import_from() -> bool:
    """True when the frame that triggered the module ``__getattr__`` is running
    ``from module import name``.

    Frames: 0 is this function, 1 is :func:`removed_name_error`, 2 is the module
    ``__getattr__`` and 3 is the code doing the lookup (CPython calls
    ``__getattr__`` from C, so no frame sits in between). Any failure to inspect
    the frame falls back to ``AttributeError``.
    """
    try:
        import opcode

        frame = sys._getframe(3)
        return frame.f_code.co_code[frame.f_lasti] == opcode.opmap["IMPORT_FROM"]
    except Exception:
        return False


def removed_name_error(module: str, path: str | None, name: str, hint: str) -> Exception:
    """Build the error a module ``__getattr__`` raises for a removed name.

    Call it directly from the module ``__getattr__`` (the frame depth matters).
    """
    if _looked_up_by_import_from():
        location = f" ({path})" if path else ""
        return ImportError(
            f"cannot import name {name!r} from {module!r}{location}. {hint}",
            name=module,
            path=path,
        )
    return AttributeError(f"module {module!r} has no attribute {name!r}. {hint}")
