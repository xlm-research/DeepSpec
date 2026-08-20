from types import SimpleNamespace

import torch
from torch import nn

from deepspec.modeling.target import online
from deepspec.modeling.target.common import TargetForwardResult


def test_online_target_hands_local_cp_shard_directly_to_draft(monkeypatch):
    teacher = online.DeepseekV4OnlineTarget.__new__(
        online.DeepseekV4OnlineTarget
    )
    teacher.model = object()
    teacher.target_layer_ids = [1, 21, 42]
    teacher.device = torch.device("cpu")
    teacher.topology = SimpleNamespace(context_parallel_size=2)

    hidden = torch.randn(1, 128, 12)
    last_hidden = torch.randn(1, 128, 4)

    def fake_cp_forward(**kwargs):
        model_inputs = kwargs["model_inputs"]
        assert model_inputs["input_ids"].shape == (1, 256)
        assert model_inputs["input_ids"][0, :3].tolist() == [10, 11, 12]
        assert not bool(model_inputs["input_ids"][0, 3:].any())
        assert model_inputs["attention_mask"].shape == (1, 256)
        assert model_inputs["attention_mask"].sum().item() == 3
        return TargetForwardResult(
            target_hidden_states=hidden,
            target_last_hidden_states=last_hidden,
            context_start=128,
        )

    monkeypatch.setattr(
        online,
        "_run_target_forward_context_parallel",
        fake_cp_forward,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "loss_mask": torch.tensor([[0, 1, 1, 0]]),
    }

    result = teacher.forward_training_batch(batch)

    assert result["input_ids"].shape == (1, 256)
    assert result["input_ids"][0, :3].tolist() == [10, 11, 12]
    assert not bool(result["input_ids"][0, 3:].any())
    assert result["loss_mask"].shape == (1, 256)
    assert result["loss_mask"][0, :3].tolist() == [0, 1, 1]
    assert not bool(result["loss_mask"][0, 3:].any())
    assert result["target_hidden_states"] is hidden
    assert result["target_last_hidden_states"] is last_hidden
    assert result["context_start"].item() == 128
    assert result["context_len"].item() == 128
    assert result["seq_len"].item() == 256
    assert "attention_mask" not in result


def test_rank_local_cache_model_matches_source_dtype_policy(monkeypatch):
    class TinyTarget(nn.Module):
        _keep_in_fp32_modules_strict = {"e_score_correction_bias"}

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32))
            self.register_buffer(
                "rotary_inv_freq",
                torch.ones(2, dtype=torch.float32),
            )
            self.register_buffer(
                "e_score_correction_bias",
                torch.ones(2, dtype=torch.bfloat16),
            )
            self.register_buffer(
                "ordinary_buffer",
                torch.ones(2, dtype=torch.bfloat16),
            )

    monkeypatch.setattr(
        online.AutoModel,
        "from_config",
        lambda _config: TinyTarget(),
    )

    model = online._build_rank_local_cache_model(object())

    assert model.weight.is_meta
    assert model.weight.dtype == torch.bfloat16
    assert model.rotary_inv_freq.is_meta
    assert model.rotary_inv_freq.dtype == torch.float32
    assert model.e_score_correction_bias.is_meta
    assert model.e_score_correction_bias.dtype == torch.float32
    assert model.ordinary_buffer.is_meta
    assert model.ordinary_buffer.dtype == torch.bfloat16


def test_online_target_rejects_more_than_one_local_sample():
    teacher = online.DeepseekV4OnlineTarget.__new__(
        online.DeepseekV4OnlineTarget
    )
    batch = {
        "input_ids": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "loss_mask": torch.ones((2, 3), dtype=torch.long),
    }

    try:
        teacher.forward_training_batch(batch)
    except ValueError as exc:
        assert "local_batch_size=1" in str(exc)
    else:
        raise AssertionError("Expected online batch-size validation to fail.")
