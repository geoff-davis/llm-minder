# Structured Message Prompts

A work item normally carries one string, `prompt`. When you need a system
message, earlier conversation turns, or per-block markers such as Anthropic's
`cache_control`, give it a message list with `messages=` instead. This is
provisional in 1.2 (see [API stability](stability.md)).

```python
from llm_minder import LLMWorkItem, OpenAIModel, OpenAIStrategy

strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-6-luna"))

item = LLMWorkItem(
    item_id="q1",
    strategy=strategy,
    messages=[
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": "What is a mutex?"},
        {"role": "assistant", "content": "A lock that lets one task in at a time."},
        {"role": "user", "content": "And a semaphore?"},
    ],
)
```

An item has exactly one of `prompt` and `messages`. String items behave
exactly as before.

## The message shape

Messages use the OpenAI Chat Completions shape, whichever provider serves
them:

- each message is a mapping with a string `role` and a `content`;
- `content` is a string, or a list of parts, each a mapping;
- a text part is `{"type": "text", "text": "..."}`, and may carry extra keys
  such as `cache_control`.

The work item checks this shape when it is built and raises `TypeError` or
`ValueError` naming the bad element, for example `$.messages[1].content[0]`.
Every value must be JSON-serializable. The item stores a deep copy, so
changing your list afterwards does not change the item, and each attempt hands
the strategy its own fresh copy.

## Entry points

Every place you pass a prompt accepts a message list too:

```python
from llm_minder import LLMCallPool, call, process_prompts

messages = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "What is a mutex?"},
]

text = await call(strategy, messages)

batch = await process_prompts(
    strategy,
    [
        "a plain string prompt",
        messages,  # a bare message list
        ("item-7", messages),  # (item_id, messages)
        ("item-8", messages, {"row": 8}),  # (item_id, messages, context)
    ],
)

async with LLMCallPool(strategy) as pool:
    text = await pool.submit(messages)
```

To pass a single conversation to `process_prompts`, wrap it: `[messages]`.
A list of messages on its own reads as a list of prompts, and each message
would be rejected.

## What each built-in model does

| Model | Mapping |
| --- | --- |
| OpenAI-compatible Chat Completions (`OpenRouterModel`, `DeepSeekModel`, `OpenAICompatibleModel`) | Sent as written, including extra part keys such as `cache_control` |
| Responses (`OpenAIModel`, `DeepSeekModel(api_surface="responses")`) | Translated by the existing Chat-to-Responses translation: text parts become `input_text`; extra part keys are dropped |
| `AnthropicModel` | Leading `system` messages become the top-level `system` (text blocks, keeping `cache_control`); `user` and `assistant` pass through |
| `GeminiModel`, `GeminiCachedModel` | Leading `system` messages become `system_instruction`; `assistant` becomes role `model`; text parts become `{"text": ...}` |
| `PydanticAIStrategy`, `CallableStrategy` | Not supported yet: string prompts only |

Anthropic and Gemini accept only what they can express: the `system`, `user`
and `assistant` roles, text parts, and no message keys besides `role` and
`content`. System messages must come first, and at least one user or
assistant message is required. Anything else raises `ValueError` before any
request is sent, and the item fails without a retry.

### System instructions

A system message in the list wins. Without one, a per-call
`system_instruction` applies, then the model's default:

1. `system` messages in the list;
2. `system_instruction=` passed to `generate()`;
3. the model's `system_instruction`.

On Gemini, a `system_instruction` you set inside the `config` passed to
`generate()` still overrides all three, as before. `GeminiCachedModel` keeps
its system instruction in the cached content, so a system message raises
`ValueError` there.

### Gemini and raw `contents` lists

`GeminiModel.generate()` already accepted a google-genai `contents` list (part
strings, part dicts, or content dicts with `parts`), and still does. A list is
treated as chat messages when any element is a mapping with a `content` key,
which google-genai contents never have; anything else is passed to the SDK
unchanged.

## Strategies

Built-in model strategies (`OpenAIStrategy`, `OpenRouterStrategy`,
`DeepSeekStrategy`, `AnthropicStrategy`, `GeminiStrategy`, and any
`ModelStrategy`) and `FakeStrategy` support messages. A custom strategy opts
in by overriding `execute_messages()`, which mirrors `execute()`:

```python
from collections.abc import Mapping, Sequence
from typing import Any

from llm_minder import LLMCallStrategy, RetryState, TokenUsage


class MyStrategy(LLMCallStrategy[str]):
    def __init__(self, client: Any) -> None:
        self.client = client

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, dict[str, Any] | None]:
        return await self.execute_messages(
            [{"role": "user", "content": prompt}], attempt, timeout, state
        )

    async def execute_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage, dict[str, Any] | None]:
        response = await self.client.chat(messages=list(messages))
        return response.text, response.usage, None
```

A strategy that doesn't override it can't run message items. Building such an
item, adding it to a processor, or passing messages to `call()` or
`LLMCallPool.submit()` raises `TypeError` naming the strategy, before any
request. A subclass that overrides `execute()` but inherits `execute_messages()`
(for example an `OpenAIStrategy` subclass that rewrites the prompt) counts as
unsupported and gets the same `TypeError`, so its customization is never silently
skipped; override `execute_messages()` as well.

`FakeStrategy` returns its fixed response for a message item. A callable
response receives a flattened rendering of the messages (one `role: text`
line per message); pass `messages_handler=` to receive the list itself:

```python
from llm_minder.testing import FakeStrategy

fake = FakeStrategy(lambda prompt: prompt.upper(), messages_handler=lambda msgs: len(msgs))
```

## Token estimation, dry run, and middleware

Token estimators (`ProcessorConfig(token_estimator=)` and a strategy's
`estimate_tokens()`) receive the same flattened rendering, which counts only
text parts. In a dry run, message items return the base placeholder built
from that rendering; a strategy's own `dry_run(prompt)` is for string prompts.

Middleware sees the work item with its `messages` and may change or replace
them. The returned item is validated again, so an invalid list fails that item
with a middleware error.

## Checkpoints and resume

A message item's input fingerprint is a SHA-256 of the canonical JSON of the
list, under a prefix that keeps it apart from string fingerprints. String
fingerprints are unchanged, so existing checkpoints replay as before. A string
item and a message item never replay against each other, even with the same
text, and any change to the list (including a `cache_control` marker) is a
new input.

With `include_prompt=True`, JSONL records store the list as `raw_messages`
(`raw_prompt` stays `null`), and SQLite stores it as a JSON array in the
existing raw-prompt column. The artifact schema version is unchanged:
llm-minder 1.1 reads these records in inspection, and never replays them.
