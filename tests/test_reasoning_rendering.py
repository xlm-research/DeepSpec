"""Reasoning must survive the actual rendering and assistant supervision paths."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from torchspec.data.parse import create_parser
from torchspec.data.preprocessing import preprocess_conversations
from torchspec.data.renderers.qwen3_5_full_data import Qwen35FullDataRenderer
from torchspec.data.renderers.qwen3_5_multimodal import Qwen35MultimodalRenderer
from torchspec.data.template import TEMPLATE_REGISTRY

INLINE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{% endfor %}"
)

# Reproduces the checkpoint's field-only reasoning branch, including its
# preserve_thinking option and the system-message guard used by the renderer.
FIELD_TEMPLATE = """
{%- for message in messages %}
    {%- if message.role == 'system' and not loop.first %}
        {{- raise_exception('System message must be at the beginning.') }}
    {%- endif %}
    {{- '<|im_start|>' + message.role + '\n' }}
    {%- if message.role == 'assistant' %}
        {%- if preserve_thinking is undefined or preserve_thinking or loop.last %}
            {{- '<think>\n' + (message.reasoning_content|default('')) + '\n</think>\n\n' }}
        {%- endif %}
    {%- endif %}
    {{- message.content }}
    {%- if message.tool_calls %}{{- message.tool_calls|tojson }}{%- endif %}
    {{- '<|im_end|>\n' }}
{%- endfor %}
"""


def make_tokenizer(template):
    vocab = [
        "<unk>",
        "<pad>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "system",
        "user",
        "assistant",
        "question",
        "reasoning",
        "answer",
        "first",
        "second",
    ]
    backend = Tokenizer(WordLevel(dict(zip(vocab, range(len(vocab)))), "<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=vocab[2:6],
        chat_template=template,
    )


def conversation(**assistant):
    return [{"role": "user", "content": "question"}, {"role": "assistant", **assistant}]


def test_inline_reasoning_is_not_duplicated_in_training_tokens_or_loss():
    tokenizer = make_tokenizer(FIELD_TEMPLATE)
    messages = conversation(content="<think>reasoning</think>answer")
    original = deepcopy(messages)
    batch = preprocess_conversations(
        tokenizer,
        [messages],
        TEMPLATE_REGISTRY.get("qwen"),
        max_length=512,
        last_turn_loss_only=True,
        use_packed_loss_mask=False,
        return_formatted_text=True,
    )
    text = batch["formatted_text"][0]
    ids, mask = batch["input_ids"][0][0], batch["loss_mask"][0][0]
    supervised = tokenizer.decode(ids[mask.bool()])
    assert text.count("<think>") == supervised.count("<think>") == 1
    assert text.count("</think>") == supervised.count("</think>") == 1
    assert "reasoning" in supervised and "answer" in supervised
    assert "<think>\n\n</think>" not in text
    assert messages == original


@pytest.mark.parametrize(
    "field", ["reasoning_content", "thinking", "thinking_content", "reasoning"]
)
def test_content_only_templates_preserve_separate_reasoning(field):
    tokenizer = make_tokenizer(INLINE_TEMPLATE)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen"))
    output = parser.format(conversation(content="answer", **{field: "reasoning"}))
    assert "<think>reasoning</think>answer" in output


def test_empty_separate_field_does_not_invent_an_inline_block():
    tokenizer = make_tokenizer(INLINE_TEMPLATE)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen"))
    output = parser.format(conversation(content="answer", reasoning_content=""))
    assert "<think>" not in output


def test_conflicting_aliases_and_unclosed_blocks_are_rejected():
    parser = create_parser(make_tokenizer(FIELD_TEMPLATE), TEMPLATE_REGISTRY.get("qwen"))
    with pytest.raises(RuntimeError, match="Conflicting.*reasoning"):
        parser.format(conversation(content="answer", reasoning_content="first", thinking="second"))
    with pytest.raises(RuntimeError, match="Malformed"):
        parser.format(conversation(content="<think>reasoning"))


def test_tools_and_rich_content_survive_normalization():
    from torchspec.data.reasoning import normalize_reasoning_messages

    tools = [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}]
    messages = conversation(
        content=[
            {"type": "text", "text": "<think>reasoning</think>answer"},
            {"type": "image_url", "image_url": {"url": "file:///image.png"}},
        ],
        tool_calls=tools,
    )
    original = deepcopy(messages)
    normalized = normalize_reasoning_messages(messages)
    assert normalized[-1]["reasoning_content"] == "reasoning"
    assert normalized[-1]["content"][0]["text"] == "answer"
    assert normalized[-1]["content"][1] == messages[-1]["content"][1]
    assert normalized[-1]["tool_calls"] == tools
    assert normalize_reasoning_messages(normalized) == normalized
    assert messages == original


def test_remote_renderer_preserves_inline_reasoning_as_native_fields():
    from tests.test_kimi_k3_renderer import StubK3Tokenizer
    from torchspec.data.renderers.kimi_k3 import K3Renderer

    tokenizer = StubK3Tokenizer()
    messages = conversation(content="<think>reasoning</think>answer")
    renderer = K3Renderer(tokenizer)
    ids, mask = renderer.render(messages, max_seq_length=512)
    supervised = tokenizer.decode([token for token, keep in zip(ids, mask) if keep])
    assert supervised.count("reasoning") == supervised.count("answer") == 1
    assert tokenizer.calls[-1]["messages"][-1]["reasoning_content"] == "reasoning"
    assert messages[-1]["content"] == "<think>reasoning</think>answer"


@pytest.mark.parametrize("reasoning", ["", "reasoning"])
def test_empty_field_does_not_hide_inline_reasoning(reasoning):
    tokenizer = make_tokenizer(FIELD_TEMPLATE)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen"))
    output = parser.format(
        conversation(
            content=f"<think>{reasoning}</think>answer",
            reasoning_content="",
        )
    )
    assert output.count("<think>") == output.count("</think>") == 1
    assert f"<think>\n{reasoning}\n</think>\n\nanswer" in output


def test_conflicting_reasoning_fails_instead_of_falling_back_to_a_different_template():
    tokenizer = make_tokenizer(FIELD_TEMPLATE)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen"))
    with pytest.raises(RuntimeError, match="Conflicting.*reasoning"):
        parser.format(
            conversation(content="<think>first</think>answer", reasoning_content="second")
        )


@pytest.mark.parametrize("renderer_type", [Qwen35MultimodalRenderer, Qwen35FullDataRenderer])
def test_native_renderers_supervise_each_reasoning_once(renderer_type):
    tokenizer = make_tokenizer(FIELD_TEMPLATE)
    renderer = renderer_type(tokenizer)
    messages = [
        *conversation(content="<think>first reasoning</think>first answer"),
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think>second reasoning</think>second answer"},
    ]
    ids, mask = renderer.render(messages, max_seq_length=512)
    supervised = tokenizer.decode([token for token, keep in zip(ids, mask) if keep])
    assert supervised.count("<think>") == 2
    assert supervised.count("reasoning") == 2
    assert supervised.count("answer") == 2


def test_inline_templates_and_literal_tags_in_user_messages_are_unchanged():
    tokenizer = make_tokenizer(INLINE_TEMPLATE)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen"))
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "Explain <think>example</think>."},
        {"role": "assistant", "content": "  <think>\nreasoning\n</think>\nanswer"},
    ]
    assert parser.format(messages) == tokenizer.apply_chat_template(messages, tokenize=False)


def test_effective_template_change_rechecks_reasoning_support():
    tokenizer = make_tokenizer(FIELD_TEMPLATE)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen"))
    messages = conversation(content="<think>reasoning</think>answer")
    assert parser.format(messages).count("<think>") == 1
    tokenizer.chat_template = INLINE_TEMPLATE
    assert "<think>reasoning</think>answer" in parser.format(messages)


def test_template_that_discards_reasoning_is_rejected():
    tokenizer = make_tokenizer("{{ messages[-1].content.split('</think>')[-1] }}")
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen"))
    with pytest.raises(RuntimeError, match="reasoning.*template|template.*reasoning"):
        parser.format(conversation(content="<think>reasoning</think>answer"))


@pytest.mark.parametrize(
    "assistant",
    [
        {"content": "<think>reasoning</think>answer"},
        {"content": "answer", "reasoning_content": "reasoning"},
    ],
)
def test_thinking_parser_reuses_the_generation_prompt_opener(assistant):
    tokenizer = make_tokenizer(
        INLINE_TEMPLATE
        + "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"
    )
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen3-thinking"))
    rendered = parser.format(conversation(**assistant))
    assert rendered.count("<think>") == rendered.count("</think>") == 1
    assert "reasoning</think>answer" in rendered


def test_dataset_cache_changes_when_the_checkpoint_template_changes(tmp_path, monkeypatch):
    from torchspec.data import dataset as dataset_module

    tokenizer = make_tokenizer(FIELD_TEMPLATE)
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(dataset_module, "_worker_state", {})
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversations": conversation(content="<think>reasoning</think>answer"),
            }
        )
    )
    args = SimpleNamespace(
        train_data_path=str(source),
        target_model_path="same-checkpoint-path",
        cache_dir=str(tmp_path / "cache"),
        chat_template="qwen",
        prompt_key="conversations",
        max_seq_length=512,
        last_turn_loss_only=True,
        train_with_decode=False,
        min_loss_tokens=0,
        num_proc=1,
    )
    first = dataset_module.load_conversation_dataset(args)
    tokenizer.chat_template = INLINE_TEMPLATE
    second = dataset_module.load_conversation_dataset(args)
    assert first[0]["formatted_prompt"] != second[0]["formatted_prompt"]
    assert len(list((tmp_path / "cache" / "tokenized_dataset").glob("*.pt"))) == 2
    assert second[0]["formatted_prompt"].count("<think>") == 1
