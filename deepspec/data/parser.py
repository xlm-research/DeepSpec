import json
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChatTemplate:
    assistant_header: str | None
    user_header: str | None
    system_prompt: str | None
    end_of_turn_token: str | None
    assistant_loss_prefix: str | None = None
    tokenizer_chat_template: str | None = None


class TemplateRegistry:
    def __init__(self):
        self._templates = {}

    def register(self, name, template):
        assert name not in self._templates, f"Chat template {name} already exists."
        self._templates[name] = template

    def get(self, name):
        return self._templates[name]


TEMPLATE_REGISTRY = TemplateRegistry()

TEMPLATE_REGISTRY.register(
    "qwen",
    ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    "deepseek_v4",
    ChatTemplate(
        assistant_header="<｜Assistant｜></think>",
        user_header="<｜User｜>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<｜end▁of▁sentence｜>",
        tokenizer_chat_template=(
            "{% for message in messages %}"
            "{% if loop.first %}{{ bos_token }}{% endif %}"
            "{% if message['role'] == 'system' %}{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}<｜User｜>{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}<｜Assistant｜></think>"
            "{{ message['content'] }}{{ eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        ),
    ),
)

TEMPLATE_REGISTRY.register(
    "deepseek_v4_flash_body",
    ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<｜end▁of▁sentence｜>",
        tokenizer_chat_template=(
            "{% for message in messages %}"
            "{% if loop.first %}{{ bos_token }}{% endif %}"
            "{% if message['role'] == 'system' %}{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}<｜User｜>{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}<｜Assistant｜>"
            "{{ message['content'] }}{{ eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        ),
    ),
)


_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
}
_MEDIA_PLACEHOLDER_PATTERN = re.compile(r"(<image>|<video>)")


class MultimodalTruncationError(ValueError):
    """Raised when truncation would leave partial multimodal placeholders."""


def normalize_media_uri_map(media_uri_map=None):
    """Validate and order URI-prefix mappings from most to least specific."""
    if media_uri_map is None:
        return {}
    if not isinstance(media_uri_map, Mapping):
        raise TypeError("media_uri_map must be a mapping of source to replacement prefixes.")

    normalized = {}
    for source_prefix, replacement_prefix in media_uri_map.items():
        if not isinstance(source_prefix, str) or not source_prefix:
            raise ValueError("media_uri_map source prefixes must be non-empty strings.")
        if not isinstance(replacement_prefix, str) or not replacement_prefix:
            raise ValueError(
                "media_uri_map replacement prefixes must be non-empty strings."
            )
        normalized[source_prefix] = replacement_prefix
    return dict(
        sorted(normalized.items(), key=lambda item: len(item[0]), reverse=True)
    )


def parse_media_uri_map_entries(entries=None):
    """Parse repeatable SOURCE_PREFIX=REPLACEMENT_PREFIX CLI entries."""
    media_uri_map = {}
    for entry in entries or []:
        source_prefix, separator, replacement_prefix = entry.partition("=")
        if not separator or not source_prefix or not replacement_prefix:
            raise ValueError(
                "Media URI mappings must use SOURCE_PREFIX=REPLACEMENT_PREFIX."
            )
        if source_prefix in media_uri_map:
            raise ValueError(f"Duplicate media URI source prefix: {source_prefix}")
        media_uri_map[source_prefix] = replacement_prefix
    return normalize_media_uri_map(media_uri_map)


def _remap_media_uri(value, media_uri_map):
    if not isinstance(value, str):
        return value
    for source_prefix, replacement_prefix in media_uri_map.items():
        if value.startswith(source_prefix):
            return replacement_prefix + value[len(source_prefix) :]
    return value


class _MediaCursor:
    def __init__(self, values, media_type, media_root=None, media_uri_map=None):
        if values is None:
            values = []
        elif not isinstance(values, (list, tuple)):
            values = [values]
        self.values = list(values)
        self.media_type = media_type
        self.media_root = media_root
        self.media_uri_map = normalize_media_uri_map(media_uri_map)
        self.index = 0

    def _resolve(self, value):
        if isinstance(value, dict):
            nested = value.get(self.media_type)
            if nested is None:
                nested = value.get(f"{self.media_type}_url")
            if isinstance(nested, dict):
                nested = nested.get("url") or nested.get("path")
            value = nested or value.get("path") or value.get("url")
        if value is None:
            raise ValueError(f"Missing {self.media_type} path or object.")
        value = _remap_media_uri(value, self.media_uri_map)
        if (
            isinstance(value, str)
            and self.media_root
            and not os.path.isabs(value)
            and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value)
            and not value.startswith("data:")
        ):
            value = os.path.join(self.media_root, value)
        return value

    def take(self):
        if self.index >= len(self.values):
            raise ValueError(
                f"Conversation contains more <{self.media_type}> placeholders "
                f"than record.{self.media_type}s entries."
            )
        value = self._resolve(self.values[self.index])
        self.index += 1
        return value

    def resolve_explicit(self, value):
        resolved = self._resolve(value)
        if self.index < len(self.values):
            candidate = self._resolve(self.values[self.index])
            if candidate == resolved:
                self.index += 1
        return resolved

    def assert_consumed(self):
        if self.index != len(self.values):
            raise ValueError(
                f"record.{self.media_type}s contains {len(self.values) - self.index} "
                f"unreferenced item(s). Add matching <{self.media_type}> placeholders "
                "or content blocks."
            )

TEMPLATE_REGISTRY.register(
    "gemma4",
    ChatTemplate(
        assistant_header="<|turn>model\n",
        user_header="<|turn>user\n",
        system_prompt=None,
        end_of_turn_token="<turn|>\n",
        assistant_loss_prefix="<|channel>thought\n<channel|>",
    ),
)


class GeneralParser:
    def __init__(self, tokenizer, chat_template):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.system_prompt = chat_template.system_prompt
        self.assistant_loss_prefix = chat_template.assistant_loss_prefix or ""
        self.assistant_message_separator = chat_template.assistant_header or ""
        self.assistant_pattern = (
            re.escape(self.assistant_message_separator)
            + r"([\s\S]*?(?:"
            + re.escape(chat_template.end_of_turn_token or "")
            + "|$))"
        )

    def parse(
        self,
        conversation,
        max_length,
    ):
        messages = []
        if conversation[0]["role"] == "system":
            warnings.warn(
                "System prompt from the sample overrides the registered template.",
                stacklevel=2,
            )
            messages.append({"role": "system", "content": conversation[0]["content"]})
            conversation = conversation[1:]
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for idx, sentence in enumerate(conversation):
            role = sentence["role"]
            assert idx != 0 or role == "user", (
                f"Conversation must start with user, got {role}."
            )
            tool_calls = sentence.get("tool_calls")
            if isinstance(tool_calls, str):
                try:
                    sentence["tool_calls"] = json.loads(tool_calls)
                except json.JSONDecodeError:
                    assert False, f"Failed to parse tool_calls JSON: {tool_calls}"
            messages.append(sentence)
        render_messages = self._prepare_render_messages(messages)
        conversation_text = render_chat_messages(
            self.tokenizer,
            render_messages,
            add_generation_prompt=False,
            chat_template=self.chat_template.tokenizer_chat_template,
        )

        encoding = self.tokenizer(
            conversation_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        matches = list(re.finditer(self.assistant_pattern, conversation_text, re.DOTALL))
        for match in matches:
            content_start_char = match.start(1)
            if self.assistant_loss_prefix and conversation_text.startswith(
                self.assistant_loss_prefix,
                content_start_char,
            ):
                content_start_char += len(self.assistant_loss_prefix)
            content_end_char = match.end(1)
            prefix_ids = self.tokenizer.encode(
                conversation_text[:content_start_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            full_ids = self.tokenizer.encode(
                conversation_text[:content_end_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            start_token_idx = min(len(prefix_ids), len(input_ids))
            end_token_idx = min(len(full_ids), len(input_ids))
            if start_token_idx < end_token_idx:
                loss_mask[start_token_idx:end_token_idx] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    def _prepare_render_messages(self, messages):
        if not self.assistant_loss_prefix:
            return messages

        render_messages = []
        for message in messages:
            if message["role"] != "assistant":
                render_messages.append(message)
                continue

            content = message["content"]
            assert isinstance(content, str), (
                "Gemma4 non-thinking training expects assistant content to be text."
            )
            render_message = dict(message)
            if not content.startswith(self.assistant_loss_prefix):
                render_message["content"] = f"{self.assistant_loss_prefix}{content}"
            render_messages.append(render_message)
        return render_messages


def render_chat_messages(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
    chat_template: str | None = None,
) -> str:
    chat_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking
    if chat_template is not None:
        chat_kwargs["chat_template"] = chat_template
    return tokenizer.apply_chat_template(messages, **chat_kwargs)


def encode_chat_messages(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> torch.LongTensor:
    conversation_text = render_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    return tokenizer(
        conversation_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


def preprocess_record(
    record,
    tokenizer,
    chat_template,
    max_length,
):
    try:
        template = TEMPLATE_REGISTRY.get(chat_template)
    except KeyError:
        assert False, f"Unknown chat template: {chat_template}"
    parser = GeneralParser(tokenizer=tokenizer, chat_template=template)
    assert "conversations" in record, "Expected `conversations` field for JSONL records."
    return parser.parse(
        record["conversations"],
        max_length=max_length,
    )


def _normalize_media_block(block, image_cursor, video_cursor):
    block = dict(block)
    block_type = str(block.get("type", "")).lower()
    if block_type in ("image", "image_url", "input_image"):
        value = block.get("image")
        if value is None:
            value = block.get("image_url")
        if isinstance(value, dict):
            value = value.get("url") or value.get("path")
        value = image_cursor.take() if value is None else image_cursor.resolve_explicit(value)
        return {"type": "image", "image": value}
    if block_type in ("video", "video_url", "input_video"):
        value = block.get("video")
        if value is None:
            value = block.get("video_url")
        if isinstance(value, dict):
            value = value.get("url") or value.get("path")
        value = video_cursor.take() if value is None else video_cursor.resolve_explicit(value)
        return {"type": "video", "video": value}
    if block_type in ("text", "input_text"):
        return {"type": "text", "text": str(block.get("text", ""))}
    return block


def _normalize_message_content(content, image_cursor, video_cursor):
    if isinstance(content, list):
        normalized = []
        for block in content:
            if isinstance(block, str):
                normalized.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                normalized.append(
                    _normalize_media_block(block, image_cursor, video_cursor)
                )
            else:
                raise TypeError(
                    "Multimodal message content items must be strings or dictionaries, "
                    f"got {type(block)!r}."
                )
        return normalized

    if content is None:
        return ""
    if not isinstance(content, str):
        return str(content)
    if not _MEDIA_PLACEHOLDER_PATTERN.search(content):
        return content

    normalized = []
    for part in _MEDIA_PLACEHOLDER_PATTERN.split(content):
        if not part:
            continue
        if part == "<image>":
            normalized.append({"type": "image", "image": image_cursor.take()})
        elif part == "<video>":
            normalized.append({"type": "video", "video": video_cursor.take()})
        else:
            normalized.append({"type": "text", "text": part})
    return normalized


def normalize_multimodal_messages(
    record,
    *,
    chat_template,
    media_root=None,
    media_uri_map=None,
):
    """Normalize OpenAI and ms-swift style records for HF multimodal processors."""
    conversation = record.get("messages")
    if conversation is None:
        conversation = record.get("conversations")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(
            "Expected a non-empty `messages` or `conversations` list in each record."
        )

    try:
        template = TEMPLATE_REGISTRY.get(chat_template)
    except KeyError:
        raise ValueError(f"Unknown chat template: {chat_template}") from None

    media_uri_map = normalize_media_uri_map(media_uri_map)
    image_cursor = _MediaCursor(
        record.get("images"), "image", media_root, media_uri_map
    )
    video_cursor = _MediaCursor(
        record.get("videos"), "video", media_root, media_uri_map
    )
    messages = []
    for sentence in conversation:
        if not isinstance(sentence, dict):
            raise TypeError(
                f"Conversation messages must be dictionaries, got {type(sentence)!r}."
            )
        role = sentence.get("role", sentence.get("from"))
        role = _ROLE_ALIASES.get(str(role).lower(), str(role).lower())
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported conversation role: {role!r}")
        content = sentence.get("content", sentence.get("value", ""))
        normalized = {
            key: value
            for key, value in sentence.items()
            if key not in ("role", "from", "content", "value")
        }
        normalized["role"] = role
        normalized["content"] = _normalize_message_content(
            content,
            image_cursor,
            video_cursor,
        )
        tool_calls = normalized.get("tool_calls")
        if isinstance(tool_calls, str):
            try:
                normalized["tool_calls"] = json.loads(tool_calls)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse tool_calls JSON: {tool_calls}") from exc
        messages.append(normalized)

    image_cursor.assert_consumed()
    video_cursor.assert_consumed()
    if messages[0]["role"] != "system" and template.system_prompt:
        messages.insert(
            0,
            {"role": "system", "content": template.system_prompt},
        )
    return messages


def _apply_multimodal_chat_template(processor, messages, *, add_generation_prompt):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "text_kwargs": {
                "padding": False,
                "return_mm_token_type_ids": True,
            }
        },
    )


def encode_multimodal_generation_record(
    record,
    *,
    processor,
    chat_template,
    media_root=None,
    media_uri_map=None,
    strip_assistant=True,
):
    """Encode one multimodal prompt for generation without flattening media."""
    messages = normalize_multimodal_messages(
        record,
        chat_template=chat_template,
        media_root=media_root,
        media_uri_map=media_uri_map,
    )
    if strip_assistant:
        first_assistant = next(
            (
                index
                for index, message in enumerate(messages)
                if message["role"] == "assistant"
            ),
            len(messages),
        )
        messages = messages[:first_assistant]
    if not messages or not any(message["role"] == "user" for message in messages):
        raise ValueError("A multimodal generation prompt must contain a user message.")

    encoding = dict(
        _apply_multimodal_chat_template(
            processor,
            messages,
            add_generation_prompt=True,
        )
    )
    _validate_visual_tokens(processor, encoding, after_truncation=False)
    return {
        key: value
        for key, value in encoding.items()
        if isinstance(value, torch.Tensor) and key != "assistant_masks"
    }


def _visual_token_count(processor, encoding, media_type):
    if media_type == "image":
        token_id = getattr(processor, "image_token_id", None)
        grid = encoding.get("image_grid_thw")
        media_processor = getattr(processor, "image_processor", None)
    else:
        token_id = getattr(processor, "video_token_id", None)
        grid = encoding.get("video_grid_thw")
        media_processor = getattr(processor, "video_processor", None)
    if grid is None:
        return 0, 0
    merge_size = int(getattr(media_processor, "merge_size", 1))
    expected = int((grid.prod(dim=-1) // (merge_size**2)).sum().item())
    actual = int((encoding["input_ids"] == int(token_id)).sum().item())
    return actual, expected


def _validate_visual_tokens(processor, encoding, *, after_truncation):
    for media_type in ("image", "video"):
        actual, expected = _visual_token_count(processor, encoding, media_type)
        if actual == expected:
            continue
        message = (
            f"{media_type} placeholder count ({actual}) does not match processed "
            f"feature count ({expected})."
        )
        if after_truncation:
            raise MultimodalTruncationError(
                f"{message} Increase max_length or remove media that falls beyond "
                "the truncation boundary."
            )
        raise ValueError(message)


def preprocess_multimodal_record(
    record,
    processor,
    chat_template,
    max_length,
    media_root=None,
    media_uri_map=None,
):
    """Process a multimodal record and create an assistant-only loss mask."""
    messages = normalize_multimodal_messages(
        record,
        chat_template=chat_template,
        media_root=media_root,
        media_uri_map=media_uri_map,
    )
    encoding = dict(
        _apply_multimodal_chat_template(
            processor,
            messages,
            add_generation_prompt=False,
        )
    )
    _validate_visual_tokens(processor, encoding, after_truncation=False)

    full_input_ids = encoding["input_ids"]
    if full_input_ids.ndim != 2 or full_input_ids.shape[0] != 1:
        raise ValueError(
            "Multimodal record preprocessing expects one conversation at a time."
        )
    full_length = int(full_input_ids.shape[1])
    loss_mask = torch.zeros(full_length, dtype=torch.long)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start_encoding = _apply_multimodal_chat_template(
            processor,
            messages[:index],
            add_generation_prompt=True,
        )
        end_encoding = _apply_multimodal_chat_template(
            processor,
            messages[: index + 1],
            add_generation_prompt=False,
        )
        start = min(int(start_encoding["input_ids"].shape[1]), full_length)
        end = min(int(end_encoding["input_ids"].shape[1]), full_length)
        if start < end:
            loss_mask[start:end] = 1

    max_length = int(max_length)
    sequence_length = min(full_length, max_length)
    processed = {}
    for key, value in encoding.items():
        if key == "assistant_masks" or not isinstance(value, torch.Tensor):
            continue
        if value.ndim >= 2 and value.shape[0] == 1 and value.shape[1] == full_length:
            processed[key] = value[0, :sequence_length]
        else:
            processed[key] = value
    processed["loss_mask"] = loss_mask[:sequence_length]
    _validate_visual_tokens(processor, processed, after_truncation=True)
    return processed
