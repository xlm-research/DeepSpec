"""The native meta initialization must initialize the TP norm wrappers."""

import torch
from torchtitan.models.dspark_draft.config_registry import qwen38_debug
from torchtitan.models.dspark_draft.tensor_parallel import DraftNorm


def test_tp_norm_is_initialized_after_meta_materialization():
    config = qwen38_debug().model_spec.model
    with torch.device("meta"):
        model = config.build()
    model.hidden_norm = DraftNorm(model.hidden_norm, group=None, partial=False)
    model.to_empty(device="cpu")
    # Poison the uninitialized allocation to make a skipped initializer fail
    # deterministically, independent of allocator reuse or zero-filled pages.
    with torch.no_grad():
        model.hidden_norm.weight.fill_(float("nan"))
    model.init_weights()
    torch.testing.assert_close(
        model.hidden_norm.weight, torch.ones_like(model.hidden_norm.weight)
    )
