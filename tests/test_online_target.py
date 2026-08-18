from types import SimpleNamespace

import torch

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

    hidden = torch.randn(1, 2, 12)
    last_hidden = torch.randn(1, 2, 4)

    def fake_cp_forward(**kwargs):
        assert kwargs["model_inputs"]["input_ids"].shape == (1, 3)
        return TargetForwardResult(
            target_hidden_states=hidden,
            target_last_hidden_states=last_hidden,
            context_start=1,
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

    assert result["input_ids"].tolist() == [[10, 11, 12]]
    assert result["loss_mask"].tolist() == [[0, 1, 1]]
    assert result["target_hidden_states"] is hidden
    assert result["target_last_hidden_states"] is last_hidden
    assert result["context_start"].item() == 1
    assert result["context_len"].item() == 2
    assert result["seq_len"].item() == 3
    assert "attention_mask" not in result


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
