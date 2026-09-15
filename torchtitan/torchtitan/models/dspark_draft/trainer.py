"""DSpark extensions to the native Trainer's component construction and loop."""

import hashlib
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.trainer import Trainer

from .metrics import collect_metrics
from .checkpoint import PhaseCheckpointer
from .preparation import PreparationConfig
from .timing import PhaseTiming


class DSparkTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        initial_weights: str = ""
        initial_target_path: str = ""
        run_id: str = ""
        phase_stop_update: int = 0
        preparation: PreparationConfig | None = None
        measure_phase: bool = False

    def __init__(self, config):
        self.phase_timing = PhaseTiming(config.measure_phase)
        self.phase_stop_update = config.phase_stop_update or config.training.steps
        if not 0 < self.phase_stop_update <= config.training.steps:
            raise ValueError("Phase stop must be inside the full training plan")
        if config.checkpoint.enable and (
            not isinstance(config.checkpoint, PhaseCheckpointer.Config)
            or not config.run_id
            or not config.dataloader.plan_path
        ):
            raise ValueError(
                "Resumable DSpark phases require a run ID, input plan and PhaseCheckpointer"
            )
        if config.dataloader.require_producer_manifest and (
            config.dataloader.target_layer_ids
            != config.model_spec.model.hf_config["target_layer_ids"]
            or config.dataloader.hidden_size
            != config.model_spec.model.hf_config["hidden_size"]
        ):
            raise ValueError(
                "Feature requirements must match the draft model configuration"
            )
        super().__init__(config)
        if config.checkpoint.enable and self.dataloader.plan_run_id != config.run_id:
            raise ValueError("Input plan belongs to a different training run")
        self.completed_updates = 0
        self._pending_state = None
        resolved = config.to_dict()
        identity = {
            key: resolved[key]
            for key in (
                "model_spec",
                "training",
                "parallelism",
                "loss",
                "optimizer",
                "lr_scheduler",
                "activation_checkpoint",
                "compile",
                "debug",
            )
        }
        identity["input_plan"] = self.dataloader.plan_identity
        self.training_identity = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        self.loss_fn.group = loss_mesh.get_group() if loss_mesh is not None else None
        self.loss_fn.gas = self.gradient_accumulation_steps
        if len(self.dataloader.entries) % self.gradient_accumulation_steps:
            raise ValueError("Feature partition ends inside an optimizer update")
        start = self.dataloader.global_microbatch_start
        if start < 0 or start % self.gradient_accumulation_steps:
            raise ValueError("Feature partition must start on an update boundary")
        required = self.phase_stop_update * self.gradient_accumulation_steps - start
        if not 0 < required <= len(self.dataloader.entries):
            raise ValueError("Feature partition is shorter than the requested training")
        if config.initial_weights and not config.checkpoint.initial_load_path:
            weights = torch.load(
                config.initial_weights, map_location="cpu", weights_only=True
            )
            set_model_state_dict(
                self.model_parts[0],
                weights,
                options=StateDictOptions(full_state_dict=True, strict=True),
            )
            # The native Trainer builds optimizers before loading initialization.
            # Master state must follow the loaded model, not random initialization.
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        optimizer.state[parameter]["master_param"].copy_(
                            parameter.detach().float()
                        )
        elif config.initial_target_path and not config.checkpoint.initial_load_path:
            from pathlib import Path

            from safetensors import safe_open

            root = Path(config.initial_target_path)
            index = json.loads((root / "model.safetensors.index.json").read_text())
            for destination, source in (
                ("embed_tokens.weight", "model.language_model.embed_tokens.weight"),
                ("lm_head.weight", "lm_head.weight"),
            ):
                with safe_open(
                    root / index["weight_map"][source], framework="pt"
                ) as file:
                    tensor = file.get_tensor(source)
                set_model_state_dict(
                    self.model_parts[0],
                    {destination: tensor},
                    options=StateDictOptions(full_state_dict=True, strict=False),
                )
                del tensor
        self.phase_timing.record("initialize", self.phase_timing.started)

    def should_continue_training(self):
        return self.step < self.phase_stop_update

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "run_id": self.config.run_id,
                "training_identity": self.training_identity,
                f"rank_{dist.get_rank()}": {
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(self.device),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "buffers": {
                        f"{index}.{name}": buffer.detach().clone()
                        for index, part in enumerate(self.model_parts)
                        for name, buffer in part.named_buffers()
                    },
                },
            }
        )
        return state

    def load_state_dict(self, state):
        if (
            state["run_id"] != self.config.run_id
            or state["training_identity"] != self.training_identity
        ):
            raise ValueError("Checkpoint and resolved training plan differ")
        super().load_state_dict(state)
        self.completed_updates = self.step
        self._pending_state = state[f"rank_{dist.get_rank()}"]

    def restore_pending_state(self):
        state = self._pending_state
        if state is None:
            raise ValueError("Full checkpoint did not restore this rank's state")
        if (
            self.dataloader.next_global_microbatch
            != self.step * self.gradient_accumulation_steps
        ):
            raise ValueError("Checkpoint data progress differs from completed updates")
        with torch.no_grad():
            for index, part in enumerate(self.model_parts):
                for name, buffer in part.named_buffers():
                    buffer.copy_(state["buffers"][f"{index}.{name}"])
        # Run after every DCP component and model storage has finished loading.
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        self._pending_state = None

    def train_step(self, data_iterator):
        with self.phase_timing.measure("training", step=self.step):
            return self._train_dspark_step(data_iterator)

    def _train_dspark_step(self, data_iterator):
        if (
            self.dataloader.next_global_microbatch
            != (self.step - 1) * self.gradient_accumulation_steps
        ):
            raise ValueError("Feature position requires a matching full checkpoint")
        self._accumulation_index = 0
        with collect_metrics() as values:
            result = super().train_step(data_iterator)
        self.completed_updates = self.step
        if self.metrics_processor.should_log(self.step):
            metrics = {}
            for name, pair in values.items():
                if self.loss_fn.group is not None:
                    dist.all_reduce(pair, group=self.loss_fn.group)
                metrics[name] = float((pair[0] / pair[1].clamp_min(1e-6)).item())
            self.metrics_processor.logger.log(metrics, self.step)
        return result

    def forward_backward_step(self, *args, **kwargs):
        self._accumulation_index += 1
        last = self._accumulation_index == self.gradient_accumulation_steps
        for part in self.model_parts:
            part.set_requires_gradient_sync(last, recurse=True)
            part.set_reshard_after_backward(last, recurse=True)
        try:
            return super().forward_backward_step(*args, **kwargs)
        finally:
            for part in self.model_parts:
                part.set_requires_gradient_sync(True, recurse=True)
                part.set_reshard_after_backward(True, recurse=True)
