"""Render or update the public API snapshot that tests/test_api_snapshot.py checks.

The snapshot covers what docs/stability.md promises: every name in
``llm_minder.__all__`` with its signature, dataclass fields, enum members and public
methods, plus the documented ``ProcessingEvent`` payload keys. A change to any of
these fails the test until the snapshot is updated on purpose:

    uv run python scripts/api_snapshot.py --update
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import re
import sys
import typing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "tests" / "api_snapshot.txt"
OBSERVERS_DOC = REPO_ROOT / "docs" / "api" / "observers.md"

# Members every class inherits from these bases; only the public API's own are listed.
_INHERITED_FROM = (object, BaseException, enum.Enum, typing.Protocol)
_SKIP_ATTRS = {"__init__", "__new__", "__class_getitem__", "__init_subclass__"}


def _optional_to_union(text: str) -> str:
    """Rewrite ``Optional[X]`` as ``X | None``; Python 3.15 renders the latter."""
    while (start := text.find("Optional[")) != -1:
        depth, end = 0, start + len("Optional")
        for end in range(start + len("Optional"), len(text)):
            depth += {"[": 1, "]": -1}.get(text[end], 0)
            if depth == 0:
                break
        inner = text[start + len("Optional[") : end]
        text = f"{text[:start]}{inner} | None{text[end + 1 :]}"
    return text


def _normalize(text: str) -> str:
    text = re.sub(r" at 0x[0-9a-f]+", "", text)
    # Annotation reprs differ across Python versions only in these spellings.
    text = re.sub(r"\btyping(_extensions)?\.", "", text)
    text = re.sub(r"\bcollections\.abc\.", "", text)
    text = re.sub(r"<class '([\w.]+)'>", r"\1", text)
    text = re.sub(r"\bllm_minder\.[\w.]*\.(?=[A-Z])", "", text)
    text = text.replace("NoneType", "None")
    text = re.sub(r"ForwardRef\('([^']*)'(?:, module='[\w.]+')?\)", r"\1", text)
    return _optional_to_union(text)


def _annotation(value: object) -> str:
    if value is inspect.Parameter.empty:
        return ""
    if isinstance(value, str):
        return value
    return inspect.formatannotation(value)


def _factory_default(factory: typing.Callable[[], object]) -> str:
    """Render what a default factory builds, so a changed default changes the snapshot."""
    try:
        value = factory()
    except Exception as exc:  # pragma: no cover - a factory that needs arguments
        return f"<factory raised {type(exc).__name__}>"
    return f"<factory: {_normalize(repr(value))}>"


def _signature(obj: object) -> str:
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "(<no signature>)"
    params = []
    for param in sig.parameters.values():
        if param.name.startswith("_"):
            continue  # private parameters aren't covered by the stability promise
        text = param.name
        if param.kind is param.VAR_POSITIONAL:
            text = "*" + text
        elif param.kind is param.VAR_KEYWORD:
            text = "**" + text
        annotation = _annotation(param.annotation)
        if annotation:
            text += f": {annotation}"
        if param.default is not param.empty:
            text += f" = {param.default!r}"
        params.append((param.kind, text))
    rendered: list[str] = []
    seen_kw_only = False
    for index, (kind, text) in enumerate(params):
        if kind is inspect.Parameter.KEYWORD_ONLY and not seen_kw_only:
            if not any(k is inspect.Parameter.VAR_POSITIONAL for k, _ in params[:index]):
                rendered.append("*")
            seen_kw_only = True
        rendered.append(text)
        if kind is inspect.Parameter.POSITIONAL_ONLY and (
            index + 1 == len(params)
            or params[index + 1][0] is not inspect.Parameter.POSITIONAL_ONLY
        ):
            rendered.append("/")
    result = f"({', '.join(rendered)})"
    returns = _annotation(sig.return_annotation)
    if returns:
        result += f" -> {returns}"
    return _normalize(result)


def _public_members(cls: type) -> list[tuple[str, object]]:
    inherited: set[str] = set()
    builtin_bases = [base for base in cls.__mro__ if base.__module__ == "builtins"]
    for base in (*_INHERITED_FROM, *builtin_bases):
        inherited.update(dir(base))
    members = []
    for name in sorted(set(dir(cls)) - inherited):
        if name.startswith("_") or name in _SKIP_ATTRS:
            continue
        static = inspect.getattr_static(cls, name)
        members.append((name, static))
    return members


def _render_class(name: str, cls: type) -> list[str]:
    bases = ", ".join(base.__name__ for base in cls.__bases__ if base is not object)
    lines = [f"class {name}({bases})" if bases else f"class {name}"]
    if typing.is_typeddict(cls):
        required = cls.__required_keys__  # type: ignore[attr-defined]
        for key, annotation in cls.__annotations__.items():
            kind = "required" if key in required else "optional"
            lines.append(f"    key {key}: {_normalize(_annotation(annotation))} ({kind})")
        return lines
    if issubclass(cls, enum.Enum):
        lines.extend(f"    {member.name} = {member.value!r}" for member in cls)
        return lines
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if field.name.startswith("_"):
                continue
            text = f"    field {field.name}: {_normalize(_annotation(field.type))}"
            if not field.init:
                text += " (init=False)"
            elif field.kw_only:
                text += " (kw_only)"
            if field.default is not dataclasses.MISSING:
                text += f" = {_normalize(repr(field.default))}"
            elif field.default_factory is not dataclasses.MISSING:
                text += f" = {_factory_default(field.default_factory)}"
            lines.append(text)
    is_protocol = getattr(cls, "_is_protocol", False)
    if not is_protocol and ("__init__" in cls.__dict__ or dataclasses.is_dataclass(cls)):
        lines.append(f"    __init__{_signature(cls.__init__)}")
    field_names = (
        {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
    )
    for member_name, static in _public_members(cls):
        if member_name in field_names:
            continue
        if isinstance(static, property):
            returns = _normalize(_annotation(inspect.signature(static.fget).return_annotation))
            lines.append(f"    property {member_name}" + (f" -> {returns}" if returns else ""))
        elif isinstance(static, (staticmethod, classmethod)):
            kind = "staticmethod" if isinstance(static, staticmethod) else "classmethod"
            lines.append(f"    {kind} {member_name}{_signature(getattr(cls, member_name))}")
        elif inspect.isfunction(static):
            prefix = "async def" if inspect.iscoroutinefunction(static) else "def"
            lines.append(f"    {prefix} {member_name}{_signature(static)}")
        elif isinstance(static, (bool, int, float, str, type(None))):
            lines.append(f"    attribute {member_name} = {static!r}")
        else:
            lines.append(f"    attribute {member_name}")
    return lines


def _documented_payloads() -> list[str]:
    lines = []
    pattern = re.compile(r"^- `([A-Z_]+)`(?: / `([A-Z_]+)`)?: `\{(.*?)\}`", re.S | re.M)
    text = OBSERVERS_DOC.read_text()
    for match in pattern.finditer(text):
        keys = ", ".join(key.strip() for key in " ".join(match.group(3).split()).split(","))
        for event in filter(None, match.group(1, 2)):
            lines.append(f"event {event}: {{{keys}}}")
    return sorted(lines)


def render() -> str:
    import llm_minder

    lines = [
        "# Public API snapshot. Regenerate on purpose with:",
        "#   uv run python scripts/api_snapshot.py --update",
        "",
    ]
    for name in sorted(llm_minder.__all__):
        obj = getattr(llm_minder, name)
        if inspect.isclass(obj) and obj.__name__ != name:
            lines.append(f"{name} = {obj.__name__}")
        elif inspect.isclass(obj):
            lines.extend(_render_class(name, obj))
        elif inspect.isfunction(obj):
            prefix = "async def" if inspect.iscoroutinefunction(obj) else "def"
            lines.append(f"{prefix} {name}{_signature(obj)}")
        else:
            lines.append(f"{name}: {_normalize(repr(obj))}")
    lines.append("")
    lines.extend(_documented_payloads())
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    rendered = render()
    if "--update" in argv:
        SNAPSHOT_PATH.write_text(rendered)
        print(f"wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)}")
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
