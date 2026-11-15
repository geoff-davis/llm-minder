"""Optional provider SDK handles and client helpers.

Provider modules read these as ``_sdk.<name>`` at call time, so tests patch them
here, in one place (``llm_minder.models._sdk.AsyncOpenAI``,
``llm_minder.models._sdk.AsyncAnthropic`` and so on)."""

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any

# Conditional imports for optional dependencies
if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from google import genai
    from google.genai.types import Content
    from openai import AsyncOpenAI
else:
    try:
        from google import genai
        from google.genai.types import Content
    except ImportError:
        genai = None  # type: ignore[assignment]
        Content = Any  # type: ignore[misc,assignment]
    try:
        from openai import AsyncOpenAI
    except ImportError:
        AsyncOpenAI = None  # type: ignore[assignment,misc]
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        AsyncAnthropic = None  # type: ignore[assignment,misc]

# Every models submodule logs as llm_minder.models, the logger name before the split.
logger = logging.getLogger("llm_minder.models")


def _build_openai_http_client(max_connections: int) -> Any:
    """Build a sized client using the transport bundled with the OpenAI SDK.

    OpenAI SDK releases before 3.x use ``httpx`` while 3.x uses ``httpx2``.
    Both expose their matching limits instance and default async client at the
    package root, so deriving the limits class from the SDK supports both
    transport generations without taking a direct dependency on either one.
    """
    from openai import DEFAULT_CONNECTION_LIMITS, DefaultAsyncHttpxClient

    limits_type = type(DEFAULT_CONNECTION_LIMITS)
    return DefaultAsyncHttpxClient(
        limits=limits_type(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
    )


def _build_anthropic_http_client(max_connections: int) -> Any:
    """Build a sized client using the transport bundled with the Anthropic SDK.

    ``anthropic`` 1.x and later run on ``httpx2``, not ``httpx``; the SDK
    rejects plain ``httpx`` objects. Deriving the limits class from the SDK's
    own default keeps llm-minder free of a direct dependency on either.
    """
    from anthropic import DEFAULT_CONNECTION_LIMITS, DefaultAsyncHttpxClient

    limits_type = type(DEFAULT_CONNECTION_LIMITS)
    return DefaultAsyncHttpxClient(
        limits=limits_type(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
    )


# Close tasks for unused clients, held so the loop doesn't drop them mid-close.
_PENDING_CLIENT_CLOSES: set["asyncio.Task[None]"] = set()


def _close_unused_client(resource: Any, method: str) -> None:
    """Close a client whose model failed to construct, from synchronous code.

    The client has sent no request, so closing only releases its transport
    objects. Inside a running event loop the close runs as a task; otherwise
    on a private loop. Failures are logged, never raised over the original
    construction error.
    """

    async def close() -> None:
        try:
            result = getattr(resource, method)()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Failed to close unused %s", type(resource).__name__, exc_info=True)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(close())
        except Exception:
            logger.debug("Failed to close unused %s", type(resource).__name__, exc_info=True)
        return
    task = loop.create_task(close())
    _PENDING_CLIENT_CLOSES.add(task)
    task.add_done_callback(_PENDING_CLIENT_CLOSES.discard)
