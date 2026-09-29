"""Batched preprocessing preserves samples while bounding temporary raw data."""

import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.data import dataset as dataset_module


def _args(tmp_path, **overrides):
    values = dict(
        train_data_path=str(tmp_path / "rows.jsonl"),
        target_model_path=str(tmp_path / "tokenizer"),
        cache_dir=str(tmp_path / "cache"),
        chat_template="qwen",
        prompt_key="conversations",
        max_seq_length=64,
        last_turn_loss_only="auto",
        min_loss_tokens=1,
        dynamic_loss_mask=True,
        num_proc=1,
        tokenization_batch_size=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def local_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr(dataset_module, "_worker_state", {})
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
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    rows = [
        {
            "conversations": [
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "answer"},
            ]
        },
        {"id": "prompt-only", "conversations": [{"role": "user", "content": "question"}]},
        {
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<think>reasoning</think>answer"},
            ]
        },
        {
            "id": "long",
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer " * 100},
            ],
        },
        {
            "data_id": "image",
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "example.png"},
                        {"type": "text", "text": "question"},
                    ],
                },
                {"role": "assistant", "content": "answer"},
            ],
        },
    ]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return tmp_path


def _assert_same_samples(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row.keys() == expected_row.keys()
        for key, expected_value in expected_row.items():
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_row[key], expected_value), key
                assert actual_row[key].dtype == expected_value.dtype
            else:
                assert actual_row[key] == expected_value, key


@pytest.mark.parametrize("mode", ["prefill", "decode", "deferred", "deferred-decode"])
@pytest.mark.parametrize("num_proc,batch_size", [(1, 1), (1, 2), (1, 8), (2, 2)])
def test_batched_outputs_match_whole_dataset(local_dataset, mode, num_proc, batch_size):
    overrides = dict(
        train_with_decode=mode in ("decode", "deferred-decode"),
        defer_tokenization=mode in ("deferred", "deferred-decode"),
    )
    expected = dataset_module.load_conversation_dataset(
        _args(local_dataset, tokenization_batch_size=0, **overrides)
    )
    actual = dataset_module.load_conversation_dataset(
        _args(
            local_dataset,
            cache_dir=str(local_dataset / "batched-cache"),
            num_proc=num_proc,
            tokenization_batch_size=batch_size,
            **overrides,
        )
    )

    _assert_same_samples(actual, expected)
    expected_ids = ["sample_0", "sample_2", "long", "image"]
    if mode != "prefill":
        expected_ids.insert(1, "prompt-only")
    assert [row["data_id"] for row in actual] == expected_ids
    thinking = next(row for row in actual if row["data_id"] == "sample_2")
    assert thinking["metadata"] == {"has_thinking": True}
    assert actual[-1]["multimodal_inputs"] == {"images": ["example.png"], "videos": None}
    assert "packed_loss_mask" not in actual[-1]
    if not overrides["defer_tokenization"]:
        assert next(row for row in actual if row["data_id"] == "long")["input_ids"].shape == (1, 64)


class _TrackedConversation(list):
    """Allow weak references without retaining source messages in the test."""


def test_batching_bounds_read_ahead_and_releases_raw_rows(tmp_path, monkeypatch):
    batch_size = 3
    produced = processed = 0
    source_refs = []

    def rows():
        nonlocal produced
        for idx in range(7):
            assert produced - processed < batch_size, "Read beyond the current processing batch"
            messages = _TrackedConversation([{"role": "user", "content": str(idx)}])
            source_refs.append(weakref.ref(messages))
            # The source/normalizer generators can still reference their last row.
            assert sum(ref() is not None for ref in source_refs) <= batch_size + 1
            produced += 1
            yield {"conversations": messages}

    def tokenize(item):
        nonlocal processed
        processed += 1
        idx = int(item[0][0]["content"])
        if idx < batch_size:
            return None  # An entirely filtered batch must not stop iteration.
        return {
            "input_ids": np.asarray([idx], dtype=np.int64),
            "packed_loss_mask": "0,1",
            "formatted_prompt": str(idx),
        }

    monkeypatch.setattr(dataset_module, "load_hf_dataset", lambda _path: rows())
    monkeypatch.setattr(dataset_module, "estimate_row_count", lambda _path: 7)
    monkeypatch.setattr(dataset_module, "_init_tokenize_worker", lambda *_args: None)
    monkeypatch.setattr(dataset_module, "_tokenize_single", tokenize)
    prompts = dataset_module.load_conversation_dataset(
        _args(tmp_path, tokenization_batch_size=batch_size)
    )

    assert produced == processed == 7
    assert [row["data_id"] for row in prompts] == ["sample_3", "sample_4", "sample_5", "sample_6"]
    assert all(ref() is None for ref in source_refs)


@pytest.mark.parametrize("first_batch_size,second_batch_size", [(0, 2), (2, 0)])
def test_batching_reuses_existing_cache(
    local_dataset, monkeypatch, first_batch_size, second_batch_size
):
    args = _args(local_dataset, tokenization_batch_size=first_batch_size)
    expected = dataset_module.load_conversation_dataset(args)

    def unexpected_tokenization(*_args):
        pytest.fail("Changing only the preprocessing batch size must reuse the cache")

    monkeypatch.setattr(dataset_module, "_init_tokenize_worker", unexpected_tokenization)
    args.tokenization_batch_size = second_batch_size
    _assert_same_samples(dataset_module.load_conversation_dataset(args), expected)


def test_batching_preserves_source_index_in_errors(local_dataset):
    source = local_dataset / "rows.jsonl"
    with source.open("a") as stream:
        stream.write('\n{"conversations": "not a conversation"}\n')
    with pytest.raises(ValueError, match="sample 5"):
        dataset_module.load_conversation_dataset(_args(local_dataset))
    assert not list((local_dataset / "cache").rglob("*.pt"))


@pytest.mark.parametrize("batch_size", [0, 2])
def test_empty_dataset_with_batching(local_dataset, batch_size):
    (local_dataset / "rows.jsonl").write_text("")
    assert (
        dataset_module.load_conversation_dataset(
            _args(local_dataset, tokenization_batch_size=batch_size)
        )
        == []
    )


@pytest.mark.parametrize("batch_size", [0, 1, 1000])
def test_tokenization_batch_size_config(batch_size):
    config = load_config(cli_args=[f"dataset.tokenization_batch_size={batch_size}"])
    assert config_to_flat_args(config).tokenization_batch_size == batch_size
    assert load_config().dataset.tokenization_batch_size == 0


def test_negative_tokenization_batch_size_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="tokenization_batch_size"):
        load_config(cli_args=["dataset.tokenization_batch_size=-1"])
    with pytest.raises(ValueError, match="tokenization_batch_size"):
        dataset_module.load_conversation_dataset(_args(tmp_path, tokenization_batch_size=-1))
