from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pickle
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as distributed_checkpoint
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)


DISTRIBUTED_CHECKPOINT_DIR_NAME = "distributed_checkpoint"


def distributed_checkpoint_path(checkpoint_dir: str | os.PathLike) -> str:
    return os.path.join(os.fspath(checkpoint_dir), DISTRIBUTED_CHECKPOINT_DIR_NAME)


def has_distributed_checkpoint(checkpoint_dir: str | os.PathLike) -> bool:
    path = distributed_checkpoint_path(checkpoint_dir)
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


class _StatefulAdapter:
    def __init__(self, obj):
        self.obj = obj

    def state_dict(self):
        return self.obj.state_dict()

    def load_state_dict(self, state_dict):
        self.obj.load_state_dict(state_dict)


@dataclass
class TrainingProgress:
    next_micro_step: int
    global_step: int
    epoch: int
    data_position: int
    local_batch_size: int
    saved_world_size: int
    parallel_config: dict[str, Any]
    model_config: dict[str, Any]
    scaler: object | None = None
    partition_id: int | None = None
    partition_start_next_micro_step: int | None = None
    partition_end_next_micro_step: int | None = None
    checkpointed: bool = False

    def state_dict(self):
        cuda_rng = (
            torch.cuda.get_rng_state(torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.empty(0, dtype=torch.uint8)
        )
        state = {
            "next_micro_step": self.next_micro_step,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "data_position": self.data_position,
            "local_batch_size": self.local_batch_size,
            "saved_world_size": self.saved_world_size,
            "parallel_config_pickle": pickle.dumps(self.parallel_config),
            "model_config_pickle": pickle.dumps(self.model_config),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": cuda_rng,
            "numpy_rng_pickle": pickle.dumps(np.random.get_state()),
            "python_rng_pickle": pickle.dumps(random.getstate()),
            "scaler": self.scaler.state_dict() if self.scaler is not None else {},
        }
        if self.partition_id is not None:
            state.update(
                partition_id=int(self.partition_id),
                partition_start_next_micro_step=int(
                    self.partition_start_next_micro_step
                ),
                partition_end_next_micro_step=int(
                    self.partition_end_next_micro_step
                ),
                checkpointed=bool(self.checkpointed),
            )
        return state

    def load_state_dict(self, state_dict):
        for name in (
            "next_micro_step",
            "global_step",
            "epoch",
            "data_position",
            "local_batch_size",
            "saved_world_size",
        ):
            setattr(self, name, state_dict[name])
        self.parallel_config = pickle.loads(state_dict["parallel_config_pickle"])
        self.model_config = pickle.loads(state_dict["model_config_pickle"])
        torch.set_rng_state(state_dict["torch_rng"].cpu())
        cuda_rng = state_dict["torch_cuda_rng"]
        if torch.cuda.is_available() and cuda_rng.numel():
            torch.cuda.set_rng_state(cuda_rng.cpu(), torch.cuda.current_device())
        np.random.set_state(pickle.loads(state_dict["numpy_rng_pickle"]))
        random.setstate(pickle.loads(state_dict["python_rng_pickle"]))
        if self.scaler is not None and state_dict.get("scaler"):
            self.scaler.load_state_dict(state_dict["scaler"])
        if "partition_id" in state_dict:
            self.partition_id = int(state_dict["partition_id"])
            self.partition_start_next_micro_step = int(
                state_dict["partition_start_next_micro_step"]
            )
            self.partition_end_next_micro_step = int(
                state_dict["partition_end_next_micro_step"]
            )
            self.checkpointed = bool(state_dict["checkpointed"])


def _model_config_dict(model) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if isinstance(value, (str, int, float, bool, type(None), list, dict))
        }
    return {"repr": repr(config)}


def save_training_checkpoint(
    *,
    checkpoint_dir: str,
    model,
    optimizer_bundle,
    progress: TrainingProgress,
) -> None:
    path = distributed_checkpoint_path(checkpoint_dir)
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer_bundle.optimizer,
    )
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": _StatefulAdapter(optimizer_bundle.scheduler),
        "training": progress,
    }
    distributed_checkpoint.save(state, checkpoint_id=path)


def load_training_checkpoint(
    *,
    checkpoint_dir: str,
    model,
    optimizer_bundle,
    progress: TrainingProgress,
) -> TrainingProgress:
    if not has_distributed_checkpoint(checkpoint_dir):
        raise FileNotFoundError(
            f"No {DISTRIBUTED_CHECKPOINT_DIR_NAME} found under {checkpoint_dir}."
        )
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer_bundle.optimizer,
    )
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": _StatefulAdapter(optimizer_bundle.scheduler),
        "training": progress,
    }
    distributed_checkpoint.load(
        state,
        checkpoint_id=distributed_checkpoint_path(checkpoint_dir),
    )
    set_state_dict(
        model,
        optimizer_bundle.optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
    )
    # Keep BF16 parameters consistent with the restored FP32 master state.
    optimizer_bundle._copy_master_weights_to_model()
    return progress


def full_model_state_dict(model) -> dict[str, torch.Tensor]:
    """Collect a CPU full state for the backward-compatible HF export."""

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def write_checkpoint_metadata(
    checkpoint_dir: str,
    *,
    progress: TrainingProgress,
) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    metadata = {
        "format": "torch.distributed.checkpoint",
        "version": 1,
        "next_micro_step": progress.next_micro_step,
        "global_step": progress.global_step,
        "epoch": progress.epoch,
        "data_position": progress.data_position,
        "saved_world_size": progress.saved_world_size,
        "parallel_config": progress.parallel_config,
        "model_config": progress.model_config,
    }
    if progress.partition_id is not None:
        metadata.update(
            partition_id=int(progress.partition_id),
            partition_start_next_micro_step=int(
                progress.partition_start_next_micro_step
            ),
            partition_end_next_micro_step=int(
                progress.partition_end_next_micro_step
            ),
            checkpointed=bool(progress.checkpointed),
        )
    path = os.path.join(checkpoint_dir, "distributed_checkpoint_metadata.json")
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def read_checkpoint_metadata(checkpoint_dir: str) -> dict[str, Any]:
    path = os.path.join(checkpoint_dir, "distributed_checkpoint_metadata.json")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid distributed checkpoint metadata: {path}")
    return metadata


__all__ = [
    "DISTRIBUTED_CHECKPOINT_DIR_NAME",
    "TrainingProgress",
    "distributed_checkpoint_path",
    "full_model_state_dict",
    "has_distributed_checkpoint",
    "load_training_checkpoint",
    "read_checkpoint_metadata",
    "save_training_checkpoint",
    "write_checkpoint_metadata",
]
