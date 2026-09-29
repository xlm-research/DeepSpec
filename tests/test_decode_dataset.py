"""Decode prompts must reach generation before supervised-token filtering."""

import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from torchspec.data import dataset as dataset_module
from torchspec.data.utils import unpack_loss_mask


@pytest.fixture
def decode_tokenizer(monkeypatch):
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
    ]
    backend = Tokenizer(WordLevel({token: i for i, token in enumerate(vocab)}, "<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=vocab[2:6],
        chat_template=(
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"
        ),
    )
    monkeypatch.setattr(dataset_module, "_worker_state", {})
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    dataset_module._init_tokenize_worker("local-test-tokenizer", True, "qwen", "auto", 14)
    return tokenizer


@pytest.mark.parametrize("last_turn_only", ["auto", True, False])
@pytest.mark.parametrize(
    "answer",
    [None, "answer", "<think>reasoning</think>" + "answer " * 20],
    ids=["prompt-only", "short-answer", "thinking-answer"],
)
def test_decode_keeps_prompts_without_requiring_existing_supervision(
    decode_tokenizer, last_turn_only, answer
):
    dataset_module._worker_state["last_turn_loss_only"] = last_turn_only
    messages = [{"role": "user", "content": "question"}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    result = dataset_module._tokenize_single((messages, None, None, 512, True))

    assert result is not None
    assert result["formatted_prompt"].endswith("<|im_start|>assistant\n<think>\n")
    expected_ids = decode_tokenizer.encode(result["formatted_prompt"], add_special_tokens=False)
    assert result["input_ids"].tolist() == [expected_ids]
    mask = unpack_loss_mask(result["packed_loss_mask"])
    assert len(mask) == len(expected_ids)
    assert mask.sum().item() == 0  # Supervision comes from the generated completion.


def test_decode_keeps_truncation_and_mask_lengths_aligned(decode_tokenizer):
    messages = [{"role": "user", "content": "question " * 100}]

    result = dataset_module._tokenize_single((messages, None, None, 32, True))

    assert result is not None
    assert result["input_ids"].shape == (1, 32)
    assert len(unpack_loss_mask(result["packed_loss_mask"])) == 32


def test_decode_still_skips_empty_conversations(decode_tokenizer):
    assert dataset_module._tokenize_single(([], None, None, 512, True)) is None


@pytest.mark.parametrize(
    "answer, retained", [(None, False), ("answer", False), ("answer " * 20, True)]
)
def test_prefill_still_requires_supervised_tokens(decode_tokenizer, answer, retained):
    messages = [{"role": "user", "content": "question"}]
    if answer is not None:
        messages.append({"role": "assistant", "content": "<think>reasoning</think>" + answer})

    result = dataset_module._tokenize_single((messages, None, None, 512, False))

    assert (result is not None) is retained
    if retained:
        assert unpack_loss_mask(result["packed_loss_mask"]).sum().item() >= 14


def test_decode_dataset_loads_and_caches_prompts(decode_tokenizer, tmp_path, monkeypatch, caplog):
    source = tmp_path / "conversations.jsonl"
    source.write_text(json.dumps({"conversations": [{"role": "user", "content": "question"}]}))
    args = SimpleNamespace(
        train_data_path=str(source),
        target_model_path="local-test-tokenizer",
        cache_dir=str(tmp_path / "cache"),
        chat_template="qwen",
        prompt_key="conversations",
        max_seq_length=512,
        train_with_decode=True,
        last_turn_loss_only="auto",
        min_loss_tokens=14,
        num_proc=1,
    )
    monkeypatch.setattr(dataset_module.logger, "propagate", True)

    prompts = dataset_module.load_conversation_dataset(args)

    assert len(prompts) == 1
    assert "unbalanced" not in caplog.text  # An open thinking prefix is expected for generation.

    def unexpected_tokenization(*_args):
        pytest.fail("The second load should reuse the prompt cache")

    monkeypatch.setattr(dataset_module, "_init_tokenize_worker", unexpected_tokenization)
    cached = dataset_module.load_conversation_dataset(args)
    assert len(cached) == 1
    assert cached[0]["packed_loss_mask"] == prompts[0]["packed_loss_mask"]


@pytest.mark.parametrize(
    ("engine_type", "train_with_decode", "expected_length"),
    [("vllm", False, 63), ("vllm", True, 64), ("sgl", False, 64)],
)
def test_vllm_prefill_reserves_one_context_token_without_changing_other_modes(
    decode_tokenizer, tmp_path, engine_type, train_with_decode, expected_length
):
    """A full-length prefill must fit vLLM's one-token extraction request."""
    source = tmp_path / "conversations.jsonl"
    source.write_text(json.dumps({"conversations": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think>reasoning</think>" + "answer " * 100},
    ]}))
    args = SimpleNamespace(
        train_data_path=str(source),
        target_model_path="local-test-tokenizer",
        cache_dir=str(tmp_path / "cache"),
        chat_template="qwen",
        prompt_key="conversations",
        max_seq_length=64,
        inference_engine_type=engine_type,
        train_with_decode=train_with_decode,
        last_turn_loss_only="auto",
        min_loss_tokens=14,
        num_proc=1,
    )

    prompts = dataset_module.load_conversation_dataset(args)

    assert len(prompts) == 1
    assert prompts[0]["input_ids"].shape[-1] == expected_length
    mask = unpack_loss_mask(prompts[0]["packed_loss_mask"])
    assert len(mask) == expected_length
    if train_with_decode:
        assert mask.sum().item() == 0
    else:
        assert mask.sum().item() >= 14
    cached = dataset_module.load_conversation_dataset(args)
    assert cached[0]["input_ids"].shape[-1] == expected_length
