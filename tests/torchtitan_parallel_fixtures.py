"""Observe native dense combinations without replacing training mathematics."""

import os
import random
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.models.dspark_draft import model_spec
from torchtitan.models.dspark_draft.checkpoint import PhaseCheckpointer
from torchtitan.models.dspark_draft.config_registry import qwen38_debug
from torchtitan.models.dspark_draft.loss import DSparkLoss
from torchtitan.models.dspark_draft.trainer import DSparkTrainer

from tests.test_dspark_training_baseline import cpu_tensor
from tests.torchtitan_phase_fixtures import ObservedMetrics


class RecordingLoss(DSparkLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(DSparkLoss.Config):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []

    def __call__(self, prediction, labels, *args, **kwargs):
        loss, terms = super().__call__(prediction, labels, *args, **kwargs)
        self.observations.append(
            {
                "loss": cpu_tensor(loss),
                "terms": {key: cpu_tensor(value) for key, value in terms.items()},
                "output": {
                    field.name: cpu_tensor(value)
                    for field in fields(prediction)
                    if (value := getattr(prediction, field.name)) is not None
                },
            }
        )
        return loss, terms


class ParallelTrainer(DSparkTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(DSparkTrainer.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.updates = []
        self.activation_shapes = {}
        for name, module in self.model_parts[0].named_modules():
            if name in (
                "fc",
                "norm",
                "hidden_norm",
                "layers.0.input_layernorm",
                "layers.0.post_attention_layernorm",
            ):
                module.register_forward_hook(
                    lambda module,
                    args,
                    output,
                    name=name: self.activation_shapes.setdefault(
                        name,
                        {"input": list(args[0].shape), "output": list(output.shape)},
                    )
                    and None
                )
        fixture = torch.load(reference_path(), weights_only=True)
        random.seed(1219)
        np.random.seed(1219)
        torch.set_rng_state(fixture["initial_cpu_rng"])
        torch.cuda.set_rng_state(fixture["initial_cuda_rng"], self.device)
        if config.checkpoint.initial_load_path:
            # Deliberately perturb rebuild RNG; complete DCP restore must undo it.
            torch.rand(17)
            torch.rand(17, device=self.device)
            random.random()
            np.random.rand()

    def train_step(self, data_iterator):
        super().train_step(data_iterator)
        parameters = {
            canonical_fqn(name): value
            for part in self.model_parts
            for name, value in part.named_parameters()
        }
        states = {
            parameter: state
            for optimizer in self.optimizers
            for parameter, state in optimizer.state.items()
        }
        self.updates.append(
            {
                "gradients_after_clip": {
                    name: cpu_tensor(p.grad)
                    for name, p in parameters.items()
                    if p.requires_grad
                },
                "parameters": {name: cpu_tensor(p) for name, p in parameters.items()},
                "adam": {
                    name: {key: cpu_tensor(value) for key, value in states[p].items()}
                    for name, p in parameters.items()
                    if p.requires_grad
                },
                "scheduler": self.lr_schedulers.schedulers[0].state_dict(),
            }
        )

    def close(self):
        if self.completed_updates == self.phase_stop_update:
            torch.save(
                {
                    "updates": self.updates,
                    "microbatches": self.loss_fn.observations,
                    "grad_norm": self.metrics_processor.norms,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(self.device),
                    "cursor": self.dataloader.next_global_microbatch,
                    "peak_allocated": torch.cuda.max_memory_allocated(self.device),
                    "peak_reserved": torch.cuda.max_memory_reserved(self.device),
                    "activation_shapes": self.activation_shapes,
                },
                Path(os.environ["DEEPSPEC_PARALLEL_ROOT"])
                / f"rank-{dist.get_rank()}.pt",
            )
        super().close()


def reference_path():
    tp = int(os.environ.get("DEEPSPEC_PARALLEL_TP", "1"))
    cp = int(os.environ.get("DEEPSPEC_PARALLEL_CP", "1"))
    pp = int(os.environ.get("DEEPSPEC_PARALLEL_PP", "1"))
    world = int(os.environ.get("WORLD_SIZE", "8"))
    dp_rank = (int(os.environ.get("RANK", "0")) % (world // pp)) // (tp * cp)
    return (
        Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"])
        / f"torch.{os.environ['DEEPSPEC_PARALLEL_DTYPE']}_rank{dp_rank}.pt"
    )


def parallel_features():
    base = qwen38_debug()
    config = ParallelTrainer.Config(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    fixture = torch.load(reference_path(), weights_only=True)
    config.model_spec = model_spec(fixture["model_config"])
    root = Path(os.environ["DEEPSPEC_PARALLEL_ROOT"])
    config.initial_weights = str(root / "initial-weights.pt")
    config.dump_folder = str(root / "native-logs")
    config.dataloader.manifest = str(root / "features.json")
    dtype = os.environ["DEEPSPEC_PARALLEL_DTYPE"]
    config.training.dtype = config.training.mixed_precision_param = dtype
    tp = int(os.environ.get("DEEPSPEC_PARALLEL_TP", "1"))
    cp = int(os.environ.get("DEEPSPEC_PARALLEL_CP", "1"))
    pp = int(os.environ.get("DEEPSPEC_PARALLEL_PP", "1"))
    replicate = int(os.environ.get("DEEPSPEC_PARALLEL_REPLICATE", "1"))
    workers = int(os.environ.get("WORLD_SIZE", "8"))
    config.parallelism.tensor_parallel_degree = tp
    config.parallelism.context_parallel_degree = cp
    config.parallelism.pipeline_parallel_degree = pp
    config.parallelism.data_parallel_replicate_degree = replicate
    config.parallelism.data_parallel_shard_degree = workers // (
        replicate * tp * cp * pp
    )
    config.parallelism.enable_sequence_parallel = (
        os.environ.get("DEEPSPEC_PARALLEL_SP") == "1"
    )
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.num_pp_microbatches = 2 if pp > 1 else 1
    config.training.num_tokens_per_train_step = 32 * workers // (tp * cp * pp)
    config.loss = RecordingLoss.Config(
        enable_vocab_parallel=os.environ.get("DEEPSPEC_PARALLEL_VOCAB") == "1"
    )
    config.metrics = ObservedMetrics.Config(log_freq=1, enable_tensorboard=True)
    config.run_id = "dspark-dense-12-19"
    config.checkpoint = PhaseCheckpointer.Config(
        enable=True,
        interval=1000,
        keep_latest_k=0,
        enable_first_step_checkpoint=False,
    )
    config.measure_phase = True
    if os.environ.get("DEEPSPEC_PARALLEL_AC") == "1":
        from torchtitan.distributed.activation_checkpoint import SelectiveAC

        config.activation_checkpoint = SelectiveAC.Config()
    return config
