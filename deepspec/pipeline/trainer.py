"""Materialize DSpark features at Titan's existing microbatch boundary."""

import math
import socket
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torchtitan.models.dspark_draft.trainer import DSparkTrainer

from .store import FEATURE_FIELDS
from .topology import consumer_dp


class StreamingDSparkTrainer(DSparkTrainer):
    def _get(self, ref, *, timeout=None):
        return self.dataloader._get(ref, timeout=timeout)

    @dataclass(kw_only=True, slots=True)
    class Config(DSparkTrainer.Config):
        pass

    def __init__(self, config):
        from .training import TrainingHandshake

        self.handshake = TrainingHandshake.begin(config.dataloader.pipeline_config)
        try:
            super().__init__(config)
            self.active_position = None
            self._update_losses = []
            if self.parallel_dims.cp != 1 or self.parallel_dims.pp != 1:
                raise ValueError("The initial streaming recipe requires CP=PP=1")
            self._get(
                self.dataloader.buffer.event.remote(
                    "consumer_rank_initialized",
                    reader=dist.get_rank(),
                    hostname=socket.gethostname(),
                    dp_rank=self.dataloader.dp_rank,
                    tp_rank=self.parallel_dims.get_mesh("tp").get_local_rank(),
                    world_size=dist.get_world_size(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                )
            )
            dist.barrier()
            if self.handshake is not None:
                self.handshake.initialized(self)
            elif dist.get_rank() == 0:
                self._get(self.dataloader.buffer.consumer_initialized.remote())
        except BaseException as error:
            if self.handshake is not None:
                self.handshake.failed(error)
            raise

    def materialize_batch(self, input_dict, labels):
        descriptor = input_dict.pop("_mooncake_features")
        started = time.monotonic()
        pipeline = self.dataloader.pipeline
        direct = pipeline["receive_device"] == "cuda"
        features = self.dataloader.take_features(
            descriptor, self.device if direct else "cpu"
        )
        features_ready = time.monotonic()
        # All source reads have completed; this rank now owns independent buffers.
        self._get(
            self.dataloader.buffer.acknowledge.remote(
                descriptor["position"],
                dist.get_rank(),
                verified=True,
                nbytes=sum(field["nbytes"] for field in descriptor["fields"].values()),
                duration_seconds=features_ready - started,
            )
        )
        acknowledged = time.monotonic()
        input_dict.update(features)
        result = super().materialize_batch(input_dict, labels)
        torch.cuda.current_stream(self.device).synchronize()
        self.active_position = descriptor["position"]
        self._get(
            self.dataloader.buffer.reader_copy_state.remote(
                self.active_position, dist.get_rank(), "active"
            ),
            timeout=pipeline["timeout_seconds"],
        )
        self._get(
            self.dataloader.buffer.event.remote(
                "gpu_ready",
                position=self.active_position,
                reader=dist.get_rank(),
                seconds=time.monotonic() - started,
                feature_wait_seconds=features_ready - started,
                acknowledge_seconds=acknowledged - features_ready,
                materialize_seconds=time.monotonic() - acknowledged,
                receive_device=pipeline["receive_device"],
            )
        )
        return result

    def forward_backward_step(self, *args, **kwargs):
        started = time.monotonic()
        self._get(
            self.dataloader.buffer.event.remote(
                "compute_start",
                position=self.active_position,
                reader=dist.get_rank(),
                step=self.step,
            )
        )
        result = super().forward_backward_step(*args, **kwargs)
        torch.cuda.current_stream(self.device).synchronize()
        if self.handshake is not None:
            detached = result.detach()
            if hasattr(detached, "to_local"):
                detached = detached.to_local()
            self._update_losses.append(float(detached.item()))
        if (
            self.step == 1
            and self._accumulation_index == self.gradient_accumulation_steps
        ):
            norms = {}
            parameters = {}
            for part in self.model_parts:
                for path in (
                    "fc",
                    "layers.0.self_attn.k_proj",
                    "layers.0.self_attn.v_proj",
                ):
                    # CheckpointWrapper forwards module access, but its
                    # internal prefix appears in root.named_parameters().
                    parameter = part.get_submodule(path).weight
                    name = f"{path}.weight"
                    gradient = parameter.grad
                    if gradient is None:
                        raise RuntimeError(
                            f"DSpark context projection has no gradient: {name}"
                        )
                    local = (
                        gradient.to_local()
                        if hasattr(gradient, "to_local")
                        else gradient
                    )
                    norms[name] = float(local.float().norm())
                    weight = (
                        parameter.to_local()
                        if hasattr(parameter, "to_local")
                        else parameter
                    )
                    parameters[name] = {
                        "weight_norm": float(weight.detach().float().norm()),
                        "gradient_elements": local.numel(),
                    }
            if len(norms) != 3:
                raise RuntimeError("Did not find the three DSpark context projections")
            self._get(
                self.dataloader.buffer.event.remote(
                    "context_gradient_observed",
                    reader=dist.get_rank(),
                    position=self.active_position,
                    norms=norms,
                    parameters=parameters,
                    feature_norms={
                        name: float(kwargs["input_dict"][name].float().norm())
                        for name in FEATURE_FIELDS
                    },
                    loss=float(
                        result.detach().to_local()
                        if hasattr(result, "to_local")
                        else result.detach()
                    ),
                )
            )
            for name, norm in norms.items():
                if not 0 < norm < float("inf"):
                    raise RuntimeError(
                        f"DSpark context gradient is invalid: {name}={norm}"
                    )
            self._get(
                self.dataloader.buffer.event.remote(
                    "context_gradient_verified",
                    reader=dist.get_rank(),
                    norms=norms,
                )
            )
        self._get(
            self.dataloader.buffer.event.remote(
                "compute_end",
                position=self.active_position,
                reader=dist.get_rank(),
                step=self.step,
                seconds=time.monotonic() - started,
            )
        )
        # Titan retains its input dictionaries until the optimizer step. These
        # large features are no longer needed after backward/recomputation and
        # stream completion; keep only the small batch metadata in that list.
        for name in FEATURE_FIELDS:
            del kwargs["input_dict"][name]
        self._get(
            self.dataloader.buffer.reader_copy_state.remote(
                self.active_position, dist.get_rank(), "retired"
            ),
            timeout=self.dataloader.pipeline["timeout_seconds"],
        )
        return result

    def train_step(self, data_iterator):
        started = time.monotonic()
        self._update_losses = []
        result = super().train_step(data_iterator)
        torch.cuda.current_stream(self.device).synchronize()
        self._get(
            self.dataloader.buffer.event.remote(
                "optimizer_update_complete",
                reader=dist.get_rank(),
                step=self.step,
                next_global_microbatch=self.dataloader.next_global_microbatch,
                next_global_sample=self.dataloader.next_global_microbatch
                * consumer_dp(self.dataloader.pipeline),
            )
        )
        if self.handshake is not None:
            self.handshake.update(
                self,
                math.fsum(self._update_losses),
                duration_seconds=time.monotonic() - started,
            )
        return result

    def train(self):
        result = super().train()
        if self.handshake is not None:
            if self.checkpointer.last_commit is None:
                raise ValueError("Training finished without a native checkpoint commit")
            self.handshake.checkpoint(self.checkpointer.last_commit)
        return result

    def close(self):
        try:
            return super().close()
        finally:
            if self.handshake is not None:
                self.handshake.close()
