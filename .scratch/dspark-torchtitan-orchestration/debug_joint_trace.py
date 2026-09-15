"""Temporary first-microbatch traces for the ticket 19 numerical investigation."""

import os
from pathlib import Path
import unittest

import torch
import torch.distributed as dist


def tensors(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return tuple(tensors(v) for v in value)
    return None


def install(model):
    root = Path(os.environ["DEEPSPEC_JOINT_TRACE"])
    root.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    counts = {}

    def record(name):
        def hook(module, args, output):
            index = counts.get(name, 0)
            counts[name] = index + 1
            limit = 2 if name.endswith(("k_proj", "v_proj", "k_norm")) else 1
            if index < limit:
                torch.save(
                    {"args": tensors(args), "output": tensors(output)},
                    root / f"rank-{rank}.{name}.{index}.pt",
                )
        return hook

    for name, module in model.named_modules():
        if name in ("embed_tokens", "fc", "hidden_norm", "norm") or name.startswith("layers."):
            module.register_forward_hook(record(name))


def reference():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from deepspec.distributed import ParallelConfig, ParallelContext
    from deepspec.training.loss import configure_loss_reduction_group
    from deepspec.utils.metrics import configure_reduction_group
    from tests.distributed_test_utils import require_torchrun
    from tests.test_dspark_training_baseline import FixedFeatureTrainer

    runtime = require_torchrun(unittest.TestCase(), world_size=1)
    topology = ParallelContext.build(ParallelConfig(dp_shard=1, reduce_dtype="fp32"))
    configure_loss_reduction_group(topology.loss_mesh.get_group())
    configure_reduction_group(topology.loss_mesh.get_group())
    fixture = torch.load(
        Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"]) / "torch.bfloat16_rank0.pt",
        weights_only=True,
    )
    trainer = FixedFeatureTrainer(
        runtime, topology, torch.bfloat16, fixture=fixture,
        model_config=Qwen3_5TextConfig.from_dict(fixture["model_config"]),
    )
    install(trainer.model)
    torch.set_rng_state(fixture["initial_cpu_rng"])
    torch.cuda.set_rng_state(fixture["initial_cuda_rng"], runtime.device)
    result = trainer.train_and_observe()
    for key, value in result["microbatches"][0]["output"].items():
        torch.testing.assert_close(value, fixture["result"]["microbatches"][0]["output"][key], rtol=0, atol=0)
    print("Trace matches archived reference first forward exactly", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    reference()
