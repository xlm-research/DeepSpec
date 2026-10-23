"""Adapt assistant reasoning to a checkpoint's native chat template."""

import hashlib
import json
import re

from jinja2 import TemplateError

REASONING_FORMAT_VERSION = "assistant-reasoning-v1"
_FIELDS = ("reasoning_content", "thinking", "thinking_content", "reasoning")
_USER = "TORCHSPEC_USER_PROBE_8e730a"
_REASON = "TORCHSPEC_REASONING_PROBE_8e730a"
_ANSWER = "TORCHSPEC_ANSWER_PROBE_8e730a"


class ReasoningFormatError(RuntimeError):
    """Do not let a parser's missing-template fallback hide reasoning errors."""


def _split_content(content):
    # Preserve rich content and media; only a leading text part can carry the
    # inline reasoning prefix. Tags quoted later in an answer are ordinary text.
    if isinstance(content, list):
        if content and isinstance(content[0], dict) and isinstance(content[0].get("text"), str):
            reasoning, text = _split_content(content[0]["text"])
            if reasoning is not None:
                return reasoning, [{**content[0], "text": text}, *content[1:]]
        return None, content
    if not isinstance(content, str) or not content.lstrip().startswith("<think>"):
        return None, content
    head, separator, answer = content.lstrip()[len("<think>") :].partition("</think>")
    if not separator or "<think>" in head:
        raise ReasoningFormatError("Malformed leading assistant reasoning block")
    return head, answer.lstrip("\n")


def _inline_content(reasoning, content):
    prefix = f"<think>{reasoning}</think>"
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}, *content]
    return prefix + (content or "")


def normalize_reasoning_messages(messages, *, inline=False):
    """Return a copy only where assistant reasoning needs adaptation.

    Empty fields do not override inline reasoning. Conflicting nonempty sources
    are rejected, rather than silently selecting one and losing the other.
    """
    result = []
    for message in messages:
        if message.get("role") != "assistant":
            result.append(message)
            continue
        content = message.get("content")
        embedded, answer = _split_content(content)
        values = []
        for field in _FIELDS:
            value = message.get(field)
            if value is not None and not isinstance(value, str):
                raise ReasoningFormatError(f"Assistant {field} must be a string or null")
            if value and value.strip():
                values.append(value)
        if embedded is not None and embedded.strip():
            values.append(embedded)
        if len({value.strip() for value in values}) > 1:
            raise ReasoningFormatError("Conflicting assistant reasoning sources")
        if embedded is None and not values:
            result.append(message)
            continue
        reasoning = values[0] if values else embedded or ""
        entry = {key: value for key, value in message.items() if key not in _FIELDS}
        if inline:
            # An already-inline message should keep its exact whitespace.
            entry["content"] = (
                content
                if embedded is not None and embedded.strip() == reasoning.strip()
                else _inline_content(reasoning, answer)
                if reasoning
                else answer
            )
        else:
            entry.update(content=answer, reasoning_content=reasoning)
        result.append(entry)
    return messages if all(a is b for a, b in zip(result, messages)) else result


def rendering_cache_fingerprint(tokenizer):
    """Version both the adaptation and the checkpoint's effective templates."""
    payload = json.dumps(
        [REASONING_FORMAT_VERSION, getattr(tokenizer, "chat_template", None)],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _valid_probe(tokenizer, rendered):
    if not isinstance(rendered, str):
        # Some remote tokenizers return IDs even for tokenize=False.
        rendered = tokenizer.decode(rendered, skip_special_tokens=False)
    if (
        rendered.count(_REASON) != 1
        or rendered.count(_ANSWER) != 1
        or rendered.index(_REASON) > rendered.index(_ANSWER)
    ):
        return False
    start = rendered.index(_USER) + len(_USER) if _USER in rendered else 0
    body = rendered[start : rendered.index(_ANSWER)]
    if "<think>" not in body and "</think>" not in body:
        return True  # Native protocols may use channels instead of think tags.
    blocks = re.findall(r"<think>(.*?)</think>", body, re.DOTALL)
    return (
        body.count("<think>") == body.count("</think>") == 1
        and len(blocks) == 1
        and _REASON in blocks[0]
    )


def _reasoning_mode(tokenizer, options):
    probe_options = dict(options)
    for key in (
        "return_dict",
        "return_tensors",
        "return_assistant_tokens_mask",
        "padding",
        "truncation",
        "max_length",
        "add_special_tokens",
        "continue_final_message",
    ):
        probe_options.pop(key, None)
    probe_options.update(tokenize=False, add_generation_prompt=False)
    template = options.get("chat_template", getattr(tokenizer, "chat_template", None))
    if hasattr(tokenizer, "get_chat_template"):
        # Preserve the native missing-template error, which GeneralParser can
        # handle with its existing fallback. Never cache a guessed template.
        template = tokenizer.get_chat_template(
            chat_template=options.get("chat_template"), tools=options.get("tools")
        )
    key = (template if isinstance(template, str) else repr(template), repr(probe_options))
    cache = getattr(tokenizer, "_torchspec_reasoning_modes", None)
    if cache is None:
        cache = tokenizer._torchspec_reasoning_modes = {}
    if key in cache:
        return cache[key]
    for mode in ("separate", "inline"):
        assistant = {"role": "assistant", "content": _ANSWER, "reasoning_content": _REASON}
        if mode == "inline":
            assistant = {"role": "assistant", "content": f"<think>{_REASON}</think>{_ANSWER}"}
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": _USER}, assistant], **probe_options
            )
            valid = _valid_probe(tokenizer, rendered)
        except (ValueError, TypeError, TemplateError):
            valid = False
        if valid:
            # Per-row tool schemas/options must not grow the cache unboundedly.
            if len(cache) >= 32:
                cache.clear()
            cache[key] = mode
            return mode
    raise ReasoningFormatError(
        "Chat template cannot preserve assistant reasoning without loss or duplicate think blocks; "
        "use a renderer that supports this checkpoint's reasoning format"
    )


def apply_chat_template_with_reasoning(tokenizer, messages, **options):
    """Render reasoning once, preserving the native protocol and caller options."""
    normalized = normalize_reasoning_messages(messages)
    if any(
        adapted is not original or adapted.get("reasoning_content")
        for original, adapted in zip(messages, normalized)
        if adapted.get("role") == "assistant"
    ):
        mode = _reasoning_mode(tokenizer, options)
        if mode == "inline":
            normalized = normalize_reasoning_messages(messages, inline=True)
    return tokenizer.apply_chat_template(normalized, **options)
