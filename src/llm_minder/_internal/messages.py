"""Structured message prompts (#315): validation, rendering, and per-provider mapping.

One canonical shape for every built-in model: Chat Completions messages, each a
mapping with a string ``role`` and ``content`` that is a string or a list of
``{"type": "text", "text": ..., **extras}`` parts. Other keys pass through
validation untouched; each provider mapping decides what it can express and
raises ``ValueError`` (before any provider call) for what it can't.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..base import LLMWorkItem
    from ..llm_strategies import LLMCallStrategy

MESSAGES_PATH = "$.messages"


def _is_message_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def normalize_messages(messages: object, *, path: str = MESSAGES_PATH) -> list[dict[str, Any]]:
    """Validate the message shape and return a JSON-normalized deep copy.

    Shape check only: a non-empty sequence of mappings, each with a string
    ``role`` and ``content`` that is a string or a list of mappings. Every
    other key passes through. Values are normalized to JSON primitives (so the
    copy is detached from the caller's objects) and checked for lone
    surrogates. Errors name a JSON path such as ``$.messages[2].content``.
    """
    # Imported here: this module is loaded lazily from base.py, and both of
    # these modules import base.
    from ..serialization import _to_fingerprint_value
    from .artifact_codec import _validate_utf8

    if isinstance(messages, Mapping) or not _is_message_sequence(messages):
        raise TypeError(
            f"{path} must be a list of message mappings such as "
            f'[{{"role": "user", "content": "..."}}] (got {type(messages).__name__}).'
        )
    assert isinstance(messages, Sequence)
    if len(messages) == 0:
        raise ValueError(f"{path} must contain at least one message.")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        message_path = f"{path}[{index}]"
        if not isinstance(message, Mapping):
            raise TypeError(
                f"{message_path} must be a mapping with 'role' and 'content' "
                f"(got {type(message).__name__})."
            )
        role = message.get("role")
        if not isinstance(role, str):
            raise TypeError(f"{message_path}.role must be a string (got {type(role).__name__}).")
        if "content" not in message:
            raise ValueError(f"{message_path}.content is required.")
        content = message["content"]
        if not isinstance(content, str):
            if isinstance(content, Mapping) or not isinstance(content, (list, tuple)):
                raise TypeError(
                    f"{message_path}.content must be a string or a list of part mappings "
                    f"(got {type(content).__name__})."
                )
            for part_index, part in enumerate(content):
                if not isinstance(part, Mapping):
                    raise TypeError(
                        f"{message_path}.content[{part_index}] must be a mapping "
                        f"(got {type(part).__name__})."
                    )
        value = _to_fingerprint_value(message, path=message_path)
        _validate_utf8(value, path=message_path)
        assert isinstance(value, dict)  # a Mapping always normalizes to a dict
        normalized.append(value)
    return normalized


def _part_text(part: Mapping[str, Any]) -> str | None:
    if part.get("type") == "text" and isinstance(part.get("text"), str):
        return str(part["text"])
    return None


def render_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Flatten messages to ``"role: text"`` lines, keeping text parts only.

    Used wherever the framework needs one string for a messages item: token
    estimators, the dry-run placeholder, and diagnostics.
    """
    lines: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, (list, tuple)):
            text = "\n".join(
                t for part in content if isinstance(part, Mapping) and (t := _part_text(part))
            )
        else:
            text = ""
        lines.append(f"{message.get('role')}: {text}")
    return "\n".join(lines)


def item_text(work_item: LLMWorkItem[Any, Any, Any]) -> str:
    """The prompt string, or the flattened rendering of a messages item."""
    if work_item.messages is not None:
        return render_messages(work_item.messages)
    return work_item.prompt


def messages_for_attempt(work_item: LLMWorkItem[Any, Any, Any]) -> list[dict[str, Any]]:
    """A private copy per attempt: a strategy that edits its list can't change the item."""
    assert work_item.messages is not None
    return [copy.deepcopy(dict(message)) for message in work_item.messages]


def _defining_class(cls: type, name: str) -> type | None:
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass
    return None


def _execute_overrides_messages_hook(cls: type) -> bool:
    """True when ``execute()`` is overridden below the class that defines ``execute_messages()``.

    A subclass of a built-in strategy that customizes only ``execute()`` would otherwise have
    message items silently bypass its customization through the inherited hook.
    """
    from ..llm_strategies import LLMCallStrategy

    messages_owner = _defining_class(cls, "execute_messages")
    execute_owner = _defining_class(cls, "execute")
    return (
        messages_owner is not None
        and messages_owner is not LLMCallStrategy
        and execute_owner is not None
        and execute_owner is not messages_owner
        and issubclass(execute_owner, messages_owner)
    )


def supports_messages(strategy: object) -> bool:
    """True when the strategy's class overrides ``execute_messages()`` at or below its ``execute()``."""
    from ..llm_strategies import LLMCallStrategy

    cls = type(strategy)
    hook = getattr(cls, "execute_messages", None)
    if hook is None or hook is LLMCallStrategy.execute_messages:
        return False
    return not _execute_overrides_messages_hook(cls)


def unsupported_messages_error(strategy: LLMCallStrategy[Any] | object) -> TypeError:
    name = type(strategy).__name__
    if _execute_overrides_messages_hook(type(strategy)):
        return TypeError(
            f"{name} overrides execute() but not execute_messages(), so its customization "
            "would not apply to structured messages. Override execute_messages() on "
            f"{name} as well, or pass a string prompt instead."
        )
    return TypeError(
        f"{name} does not support structured messages: it does not override "
        "execute_messages(). Use a built-in model strategy (OpenAIStrategy, "
        "GeminiStrategy, AnthropicStrategy, ...) or FakeStrategy, override "
        f"execute_messages() on {name}, or pass a string prompt instead."
    )


def require_messages_support(strategy: object) -> None:
    if not supports_messages(strategy):
        raise unsupported_messages_error(strategy)


def build_work_item(
    item_id: str,
    strategy: LLMCallStrategy[Any],
    prompt: str | Sequence[Mapping[str, Any]],
    context: Any = None,
) -> LLMWorkItem[Any, Any, Any]:
    """Route a call-side ``str | messages`` argument to the right work-item field."""
    from ..base import LLMWorkItem

    if isinstance(prompt, str):
        return LLMWorkItem(item_id=item_id, strategy=strategy, prompt=prompt, context=context)
    return LLMWorkItem(item_id=item_id, strategy=strategy, context=context, messages=prompt)


# ── Per-provider mapping ────────────────────────────────────────────────


def _shape_error(provider: str, path: str, detail: str) -> ValueError:
    return ValueError(f"{provider} messages: {path} {detail}.")


def _iter_messages(provider: str, messages: object) -> list[Mapping[str, Any]]:
    if isinstance(messages, Mapping) or not _is_message_sequence(messages):
        raise _shape_error(provider, "$.messages", "must be a list of message mappings")
    assert isinstance(messages, Sequence)
    checked: list[Mapping[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            raise _shape_error(
                provider, f"$.messages[{index}]", "must be a mapping with a string 'role'"
            )
        checked.append(message)
    return checked


def _text_parts(
    provider: str, content: object, path: str, *, keep_extras: bool
) -> list[dict[str, Any]]:
    """Return ``content`` as text-part dicts; raise for anything that isn't text."""
    if isinstance(content, str):
        return [{"type": "text", "text": content} if keep_extras else {"text": content}]
    if isinstance(content, Mapping) or not isinstance(content, (list, tuple)):
        raise _shape_error(provider, path, "must be a string or a list of text parts")
    parts: list[dict[str, Any]] = []
    for index, part in enumerate(content):
        part_path = f"{path}[{index}]"
        if not isinstance(part, Mapping):
            raise _shape_error(provider, part_path, "must be a mapping")
        if part.get("type") != "text":
            raise _shape_error(
                provider,
                f"{part_path}.type",
                f"is {part.get('type')!r}; only 'text' parts are supported",
            )
        if not isinstance(part.get("text"), str):
            raise _shape_error(provider, f"{part_path}.text", "must be a string")
        parts.append(copy.deepcopy(dict(part)) if keep_extras else {"text": part["text"]})
    return parts


def _check_message_keys(provider: str, message: Mapping[str, Any], path: str) -> None:
    extra = sorted(str(key) for key in message if key not in ("role", "content"))
    if extra:
        raise _shape_error(provider, path, f"has keys {extra} that {provider} cannot express")


def to_anthropic(
    messages: object,
) -> tuple[list[dict[str, Any]], str | list[dict[str, Any]] | None]:
    """Map Chat-shaped messages to Anthropic ``(messages, system)``.

    Leading ``system`` messages become the top-level ``system``: a string when
    every part is plain text, else a list of text blocks (which keep extras
    such as ``cache_control``). ``user``/``assistant`` messages pass through;
    their text parts already are Anthropic text blocks. A ``system`` message
    after a non-system one, any other role, or a non-text part raises
    ``ValueError``. ``system`` is ``None`` when the list has no system message.
    """
    provider = "Anthropic"
    system_blocks: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    for index, message in enumerate(_iter_messages(provider, messages)):
        path = f"$.messages[{index}]"
        _check_message_keys(provider, message, path)
        role = message["role"]
        if "content" not in message:
            raise _shape_error(provider, f"{path}.content", "is required")
        content = message["content"]
        if role == "system":
            if converted:
                raise _shape_error(
                    provider,
                    path,
                    "is a system message after a non-system one; Anthropic takes system "
                    "text only at the start",
                )
            system_blocks.extend(
                _text_parts(provider, content, f"{path}.content", keep_extras=True)
            )
        elif role in ("user", "assistant"):
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            else:
                parts = _text_parts(provider, content, f"{path}.content", keep_extras=True)
                converted.append({"role": role, "content": parts})
        else:
            raise _shape_error(
                provider, f"{path}.role", f"is {role!r}; use 'system', 'user' or 'assistant'"
            )
    if not converted:
        raise _shape_error(provider, "$.messages", "needs at least one user or assistant message")
    system: str | list[dict[str, Any]] | None = None
    if system_blocks:
        if all(set(block) == {"type", "text"} for block in system_blocks):
            system = "\n\n".join(block["text"] for block in system_blocks)
        else:
            system = system_blocks
    return converted, system


def anthropic_system_text(system: object) -> str | None:
    """The text a cache group is labelled by, for a string or text-block ``system``."""
    if isinstance(system, str):
        return system
    if isinstance(system, (list, tuple)):
        texts = [
            block["text"]
            for block in system
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        return "\n\n".join(texts) if texts else None
    return None


def is_chat_message_list(prompt: object) -> bool:
    """Whether a Gemini ``prompt`` list is Chat-shaped messages rather than raw contents.

    The rule: a list is messages when any element is a mapping with a
    ``content`` key. google-genai contents never use that key (a content dict
    has ``role`` and ``parts``; parts are strings, ``Part`` objects or part
    dicts), so every raw contents list passed to ``generate()`` today keeps
    its meaning.
    """
    if not isinstance(prompt, (list, tuple)):
        return False
    return any(isinstance(element, Mapping) and "content" in element for element in prompt)


def to_gemini(messages: object, *, provider: str = "Gemini") -> tuple[list[Any], str | None]:
    """Map Chat-shaped messages to Gemini ``(contents, system_instruction)``.

    ``system`` messages (only at the start) become ``system_instruction``,
    ``assistant`` becomes role ``model``, and text parts become
    ``{"text": ...}`` (extras such as ``cache_control`` have no Gemini
    meaning and are dropped). Any other role or part type raises
    ``ValueError``.
    """
    system_texts: list[str] = []
    # list[Any]: google-genai types contents as TypedDicts, which dicts don't match.
    contents: list[Any] = []
    for index, message in enumerate(_iter_messages(provider, messages)):
        path = f"$.messages[{index}]"
        _check_message_keys(provider, message, path)
        role = message["role"]
        if "content" not in message:
            raise _shape_error(provider, f"{path}.content", "is required")
        parts = _text_parts(provider, message["content"], f"{path}.content", keep_extras=False)
        if role == "system":
            if contents:
                raise _shape_error(
                    provider,
                    path,
                    "is a system message after a non-system one; Gemini takes a system "
                    "instruction only at the start",
                )
            system_texts.extend(part["text"] for part in parts)
        elif role in ("user", "assistant"):
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
        else:
            raise _shape_error(
                provider, f"{path}.role", f"is {role!r}; use 'system', 'user' or 'assistant'"
            )
    if not contents:
        raise _shape_error(provider, "$.messages", "needs at least one user or assistant message")
    return contents, ("\n\n".join(system_texts) if system_texts else None)
