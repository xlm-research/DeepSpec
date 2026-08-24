"""Two-phase worker used to validate distributed-checkpoint resharding."""

import argparse
import json
import os

import torch
from torch import nn

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.distributed_checkpoint import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from deepspec.distributed.fsdp import apply_fsdp2
from deepspec.distributed.mesh import ParallelContext
from deepspec.distributed.runtime import initialize_runtime
from deepspec.training.optimizer import BF16Optimizer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))]
        )

    def forward(self, x):
        return self.layers[0](x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("save", "load"), required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    runtime = initialize_runtime()
    config = ParallelConfig(dp_shard=runtime.world_size)
    context = ParallelContext.build(config, device_type=runtime.device.type)
    torch.manual_seed(314)
    model = Model().to(runtime.device)
    apply_fsdp2(model, context, config, param_dtype=torch.float32)
    optimizer = BF16Optimizer(model, 1e-3, 3, 0, 0)
    x = torch.arange(32, device=runtime.device, dtype=torch.float32).reshape(4, 8) / 32
    progress = TrainingProgress(0, 0, 0, 0, 4, runtime.world_size, {}, {})
    if args.phase == "save":
        model(x).square().mean().backward()
        optimizer.step()
        expected = model(x).detach().cpu().tolist()
        progress = TrainingProgress(
            1, 1, 0, 4, 4, runtime.world_size, config.to_dict(), {"hidden": 8}
        )
        save_training_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
            optimizer_bundle=optimizer,
            progress=progress,
        )
        if runtime.global_rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(os.path.join(args.checkpoint_dir, "expected.json"), "w", encoding="utf-8") as handle:
                json.dump(expected, handle)
    else:
        load_training_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
            optimizer_bundle=optimizer,
            progress=progress,
        )
        with open(os.path.join(args.checkpoint_dir, "expected.json"), encoding="utf-8") as handle:
            expected = torch.tensor(json.load(handle), device=runtime.device)
        torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
        assert progress.next_micro_step == 1


if __name__ == "__main__":
    main()
