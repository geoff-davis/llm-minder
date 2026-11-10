"""The HTTP library the installed OpenAI SDK is built on, for tests.

OpenAI SDK releases before 3.x use ``httpx``; 3.x uses ``httpx2`` and no longer
pulls in ``httpx`` at all. Tests that build SDK requests, responses, exceptions
or mock transports must use the library the SDK itself uses, derived the same
way ``llm_minder.models._build_openai_http_client`` derives its limits class.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def openai_http() -> ModuleType:
    """Return ``httpx`` or ``httpx2``, whichever the installed openai uses."""
    import openai

    package = type(openai.DEFAULT_CONNECTION_LIMITS).__module__.split(".")[0]
    return importlib.import_module(package)
