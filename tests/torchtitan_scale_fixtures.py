"""Bounded full-geometry workloads for the native phase performance entry."""

import os
import random
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from torchtitan.models.dspark_draft.config_registry import qwen38_27b_tp4
from torchtitan.models.dspark_draft.trainer import DSparkTrainer


def capture_rng(device):
    numpy_rng = np.random.get_state()
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "python": random.getstate(),
        "numpy": (
            numpy_rng[0],
            torch.from_numpy(numpy_rng[1].copy()),
            *numpy_rng[2:],
        ),
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    numpy_rng = state["numpy"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy(), *numpy_rng[2:]))


class ScaleTrainer(DSparkTrainer):
    """Record small supervision facts and an optional shared benchmark seed."""

    @dataclass(kw_only=True, slots=True)
    class Config(DSparkTrainer.Config):
        capture_initialization: str = ""
        reference_initialization: str = ""

    def __init__(self, config):
        super().__init__(config)
        self.observations = []
        model = self.model_parts[0]
        if not config.checkpoint.initial_load_path:
            if config.reference_initialization:
                reference = torch.load(
                    Path(config.reference_initialization)
                    / f"rng-rank{dist.get_rank()}.pt",
                    weights_only=True,
                )
                restore_rng(reference, self.device)
            if config.capture_initialization:
                root = Path(config.capture_initialization)
                root.mkdir(parents=True, exist_ok=True)
                with self.phase_timing.measure("validation_capture"):
                    weights = get_model_state_dict(
                        model,
                        options=StateDictOptions(
                            full_state_dict=True, cpu_offload=True
                        ),
                    )
                    if dist.get_rank() == 0:
                        path = root / "initial-weights.pt"
                        if path.exists():
                            raise FileExistsError(path)
                        torch.save(weights, path)
                    torch.save(
                        capture_rng(self.device),
                        root / f"rng-rank{dist.get_rank()}.pt",
                    )
                    dist.barrier()
        model.register_forward_hook(self.observe_forward)

    def observe_forward(self, module, args, output):
        self.observations.append(
            {
                "target_ids": output.target_ids.detach().cpu(),
                "eval_mask": output.eval_mask.detach().cpu(),
                "block_keep_mask": output.block_keep_mask.detach().cpu(),
                "next_microbatch": self.dataloader.next_global_microbatch,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device),
            }
        )

    def close(self):
        if self.completed_updates == self.phase_stop_update:
            path = Path(os.environ["DEEPSPEC_PHASE_RESULT"]).parent
            torch.save(
                self.observations, path / f"supervision-rank{dist.get_rank()}.pt"
            )
        super().close()


def qwen38_128k_acceptance():
    base = qwen38_27b_tp4()
    config = ScaleTrainer.Config(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    config.training.steps = 10
    config.training.num_tokens_per_train_step = 131072 * 4
    config.measure_phase = True
    config.preparation.source_paths = [os.environ["DEEPSPEC_SCALE_DATA"]]
    config.preparation.epochs = 1
    config.dump_folder = str(Path(os.environ["DEEPSPEC_SCALE_OUTPUT"]) / "native-logs")
    config.capture_initialization = os.environ.get("DEEPSPEC_SCALE_CAPTURE", "")
    config.reference_initialization = os.environ.get("DEEPSPEC_SCALE_REFERENCE", "")
    if config.reference_initialization:
        config.initial_weights = str(
            Path(config.reference_initialization) / "initial-weights.pt"
        )
    return config
