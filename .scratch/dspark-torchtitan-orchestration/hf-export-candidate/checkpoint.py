"""Commit complete native DCP state at an orchestration phase boundary."""

import hashlib
import json
import os
from dataclasses import dataclass, field
import time

import fsspec
import torch
import torch.distributed as dist

from torchtitan.components.checkpointer import AsyncMode, CheckpointManager, MODEL
from torchtitan.tools import filesystem


def read_commit(path):
    with fsspec.open(filesystem.join(path, "commit.json"), "rt") as stream:
        commit = json.load(stream)
    with fsspec.open(filesystem.join(path, ".metadata"), "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if commit["metadata_sha256"] != digest:
        raise ValueError("Checkpoint metadata differs from the committed state")
    return commit


def write_marker(directory, name, value):
    marker = filesystem.join(directory, name)
    if filesystem.is_remote(marker):
        with fsspec.open(marker, "wt") as stream:
            json.dump(value, stream, sort_keys=True)
        return
    temporary = marker + ".incomplete"
    with open(temporary, "w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker)
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


class PhaseCheckpointer(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        initial_load_model_only: bool = False
        last_save_model_only: bool = False
        last_save_in_hf: bool = False
        keep_latest_k: int = 2
        milestone_steps: list[int] = field(default_factory=list)
        export_hf_steps: list[int] = field(default_factory=list)
        export_hf_final: bool = False

        def __post_init__(self):
            CheckpointManager.Config.__post_init__(self)
            if self.async_mode != "disabled":
                raise ValueError("Phase checkpoints require synchronous DCP commits")
            if (
                self.load_only
                or self.initial_load_model_only
                or self.initial_load_in_hf
                or self.last_save_model_only
                or self.last_save_in_hf
                or self.exclude_from_loading
            ):
                raise ValueError("Phase checkpoints must save and restore full state")

    def __init__(self, config, **kwargs):
        self.last_commit = None
        self.phase_config = config
        self.exports = []
        super().__init__(config, **kwargs)

    def _purge_stale_checkpoints(self, *, saving_step, staging_dir_prefix=None):
        # Native DCP calls this before writing. Retention starts only after commit.
        if self.last_commit and self.last_commit["completed_updates"] == saving_step:
            super()._purge_stale_checkpoints(
                saving_step=saving_step, staging_dir_prefix=staging_dir_prefix
            )

    def _is_purge_exempt(self, step):
        return (
            step in self.phase_config.milestone_steps
            or step == self.states["train_state"].config.training.steps
            or super()._is_purge_exempt(step)
        )

    def _should_save(self, curr_step, last_step=False):
        return curr_step == self.states[
            "train_state"
        ].phase_stop_update or super()._should_save(curr_step, last_step)

    def _save(self, curr_step, last_step=False):
        if not self._should_save(curr_step, last_step):
            return False
        with self.states["train_state"].phase_timing.measure("save", step=curr_step):
            return self._save_phase(curr_step, last_step)

    def _save_phase(self, curr_step, last_step):
        checkpoint_id = self._create_checkpoint_id(curr_step)
        if self._is_valid_checkpoint(checkpoint_id):
            raise FileExistsError(f"A committed update already exists: {checkpoint_id}")
        if not super()._save(curr_step, last_step):
            return False
        torch.cuda.synchronize()
        dist.barrier()
        trainer = self.states["train_state"]
        commit = {
            "format_version": 1,
            "checkpoint": checkpoint_id,
            "run_id": trainer.config.run_id,
            "training_identity": trainer.training_identity,
            "completed_updates": curr_step,
            "next_global_microbatch": trainer.dataloader.next_global_microbatch,
            "partition_identity": trainer.dataloader.identity,
            "world_size": dist.get_world_size(),
            "resolved_recipe": trainer.config.to_dict(),
            "input_plan_identity": trainer.dataloader.plan_identity,
        }
        if dist.get_rank() == 0:
            with fsspec.open(
                filesystem.join(checkpoint_id, ".metadata"), "rb"
            ) as stream:
                commit["metadata_sha256"] = hashlib.sha256(stream.read()).hexdigest()
            write_marker(checkpoint_id, "commit.json", commit)
        dist.barrier()
        self.last_commit = read_commit(checkpoint_id)
        self._purge_stale_checkpoints(saving_step=curr_step)
        if curr_step in self.phase_config.export_hf_steps or (
            self.phase_config.export_hf_final
            and curr_step == trainer.config.training.steps
        ):
            self._export_hf(curr_step)
        return True

    def _export_hf(self, step):
        started = time.monotonic()
        path = filesystem.join(filesystem.join(self.folder, "hf"), f"step-{step}")
        states = {
            name: value.to(self.export_dtype)
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for name, value in self.states[MODEL].state_dict().items()
        }
        self.dcp_save(
            states,
            checkpoint_id=path,
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=True,
        )
        if dist.get_rank() == 0:
            from transformers.models.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5TextConfig,
            )

            config = Qwen3_5TextConfig(
                **self.states["train_state"].config.model_spec.model.hf_config
            )
            config.architectures = ["Qwen3DSparkModel"]
            config.dtype = self.phase_config.export_dtype
            write_marker(path, "config.json", config.to_dict())
            write_marker(
                path,
                "export.json",
                {
                    "checkpoint": self.last_commit,
                    "export_dtype": self.phase_config.export_dtype,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        dist.barrier()
        self.exports.append(
            {"step": step, "path": path, "elapsed_seconds": time.monotonic() - started}
        )

    def _is_valid_checkpoint(self, checkpoint_dir):
        return self._storage.isfile(
            filesystem.join(checkpoint_dir, "commit.json")
        ) and self._storage.isfile(filesystem.join(checkpoint_dir, ".metadata"))

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        commit = read_commit(checkpoint_id)
        trainer = self.states["train_state"]
        if (
            commit["format_version"] != 1
            or commit["run_id"] != trainer.config.run_id
            or commit["training_identity"] != trainer.training_identity
            or commit["world_size"] != dist.get_world_size()
        ):
            raise ValueError("Checkpoint training identity or topology differs")
        with trainer.phase_timing.measure("restore"):
            super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)
            trainer.restore_pending_state()
        self.last_commit = commit
