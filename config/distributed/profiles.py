from __future__ import annotations

from pathlib import Path
import runpy


def build(parallel: dict):
    base_path = Path(__file__).resolve().parents[1] / "dflash2" / "dflash2_qwen3_8_27b.py"
    namespace = {
        key: value
        for key, value in runpy.run_path(str(base_path)).items()
        if not key.startswith("__")
    }
    namespace["train"] = dict(namespace["train"])
    namespace["train"]["parallel"] = dict(parallel)
    return namespace


def defaults(**overrides):
    values = dict(
        dp_replicate=1,
        dp_shard=1,
        tp=1,
        cp=1,
        ep=1,
        expert_tp=1,
        pp=1,
        use_fsdp=True,
        use_sequence_parallel=False,
        use_loss_parallel=False,
        use_activation_checkpoint=False,
        use_compile=False,
        expert_dispatch_backend="auto",
        context_parallel_backend="pytorch",
        dynamic_context_parallel=False,
        reshard_after_forward=True,
    )
    values.update(overrides)
    return values
