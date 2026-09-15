"""Observe the real native phase entry against immutable DSpark inputs."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.distributed as dist

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.models.dspark_draft import model_spec
from torchtitan.models.dspark_draft.config_registry import qwen38_debug
from torchtitan.models.dspark_draft.trainer import DSparkTrainer

from tests import test_dspark_training_baseline as baseline


def qwen38_live():
    """Short-sequence acceptance on the released teacher and full draft geometry."""
    from torchtitan.models.dspark_draft.config_registry import qwen38_27b

    config = qwen38_27b()
    config.training.max_context_length = 256
    config.training.num_tokens_per_microbatch_per_dp_rank = 256
    config.training.num_tokens_per_train_step = 256 * 4
    config.training.steps = 2
    config.model_spec.model.hf_config["num_anchors"] = 8
    config.lr_scheduler.total_steps = 2
    config.lr_scheduler.warmup_steps = 1
    config.preparation.source_paths = [os.environ["DEEPSPEC_LIVE_DATA"]]
    config.preparation.epochs = 1
    return config


def retention_features():
    config = checkpoint_features()
    config.training.steps = 4
    config.checkpoint.interval = 1
    config.checkpoint.keep_latest_k = 2
    config.checkpoint.milestone_steps = [1]
    config.checkpoint.export_hf_steps = [3]
    return config


def interruption_features():
    config = checkpoint_features()
    config.checkpoint.interval = 1
    config.checkpoint.keep_latest_k = 2
    return config


class ObservedMetrics(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def log(
        self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None
    ):
        if not hasattr(self, "norms"):
            self.norms = []
        self.norms.append(grad_norm)
        return super().log(
            step, global_avg_loss, global_max_loss, grad_norm, extra_metrics
        )


class ObservedTrainer(DSparkTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(DSparkTrainer.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.microbatches = []
        self.updates = []
        fixture = torch.load(fixture_path(), weights_only=True)
        torch.set_rng_state(fixture["initial_cpu_rng"])
        torch.cuda.set_rng_state(fixture["initial_cuda_rng"], self.device)
        if os.environ.get("DEEPSPEC_PHASE_PERTURB_INITIALIZATION"):
            torch.rand(100, device=self.device)
            torch.rand(100)
        self.model_parts[0].register_forward_hook(self.observe_forward)

    def observe_forward(self, module, args, output):
        group = self.loss_fn.group
        if group is None:
            group = self.parallel_dims.get_mesh("dp_shard").get_group()
        _, terms, denominator = baseline.reference_loss(output, group)
        self.microbatches.append(
            {
                "terms": baseline.cpu_tensor(terms),
                "denominator": baseline.cpu_tensor(denominator),
                "output": {
                    field.name: baseline.cpu_tensor(value)
                    for field in fields(output)
                    if (value := getattr(output, field.name)) is not None
                },
            }
        )

    def _forward_backward_body(self, *args, **kwargs):
        loss = super()._forward_backward_body(*args, **kwargs)
        self.microbatches[-1]["loss"] = baseline.cpu_tensor(
            loss * self.gradient_accumulation_steps
        )
        return loss

    def train_step(self, data_iterator):
        super().train_step(data_iterator)
        parameters = {
            canonical_fqn(name): parameter
            for name, parameter in self.model_parts[0].named_parameters()
        }
        optimizer = self.optimizers.optimizers[0]
        self.updates.append(
            {
                "gradients_after_clip": {
                    name: baseline.cpu_tensor(p.grad)
                    for name, p in parameters.items()
                    if p.requires_grad
                },
                "parameters": {
                    name: baseline.cpu_tensor(p) for name, p in parameters.items()
                },
                "adam": {
                    name: {
                        key: baseline.cpu_tensor(value)
                        for key, value in optimizer.state[p].items()
                    }
                    for name, p in parameters.items()
                    if p.requires_grad
                },
                "scheduler": self.lr_schedulers.schedulers[0].state_dict(),
            }
        )

    def close(self):
        if self.completed_updates == self.phase_stop_update:
            result = {
                "microbatches": self.microbatches,
                "updates": self.updates,
                "next_micro_step": self.dataloader.next_global_microbatch,
                "cuda_rng": torch.cuda.get_rng_state(self.device),
                "cpu_rng": torch.get_rng_state(),
                "metrics": {"grad_norm": self.metrics_processor.norms},
            }
            torch.save(
                result,
                Path(os.environ["DEEPSPEC_PHASE_TEST_ROOT"])
                / f"native-rank{dist.get_rank()}.pt",
            )
        super().close()


def fixture_path():
    dp_rank = int(os.environ.get("RANK", "0")) // int(
        os.environ.get("DEEPSPEC_PHASE_TP", "1")
    )
    return Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"]) / (
        f"torch.{os.environ['DEEPSPEC_PHASE_TEST_DTYPE']}_rank{dp_rank}.pt"
    )


def fixed_features():
    root = Path(os.environ["DEEPSPEC_PHASE_TEST_ROOT"])
    dtype = os.environ["DEEPSPEC_PHASE_TEST_DTYPE"]
    fixture = torch.load(fixture_path(), weights_only=True)
    base = qwen38_debug()
    config = ObservedTrainer.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    config.model_spec = model_spec(fixture["model_config"])
    config.initial_weights = str(root / "initial-weights.pt")
    config.dataloader.manifest = str(root / "features.json")
    config.dump_folder = str(root / "native-logs")
    config.measure_phase = bool(os.environ.get("DEEPSPEC_PHASE_MEASURE"))
    config.training.dtype = dtype
    config.training.mixed_precision_param = dtype
    workers = int(os.environ.get("WORLD_SIZE", "1"))
    tp = int(os.environ.get("DEEPSPEC_PHASE_TP", "1"))
    replicas = int(os.environ.get("DEEPSPEC_PHASE_DP_REPLICATE", "1"))
    if replicas < 1 or workers % (tp * replicas):
        raise ValueError(
            "The fixture's TP and DP replica degrees must divide its workers"
        )
    config.parallelism.tensor_parallel_degree = tp
    config.parallelism.data_parallel_replicate_degree = replicas
    config.parallelism.data_parallel_shard_degree = workers // (tp * replicas)
    if tp > 1:
        config.parallelism.enable_sequence_parallel = False
    config.training.num_tokens_per_train_step *= workers // tp
    config.metrics = ObservedMetrics.Config(log_freq=1, enable_tensorboard=True)
    if os.environ.get("DEEPSPEC_PHASE_SELECTIVE_AC"):
        from torchtitan.distributed.activation_checkpoint import SelectiveAC

        config.activation_checkpoint = SelectiveAC.Config()
    return config


def checkpoint_features():
    from torchtitan.models.dspark_draft.checkpoint import PhaseCheckpointer

    config = fixed_features()
    config.run_id = "native-dspark-checkpoint-reference"
    config.phase_stop_update = int(os.environ.get("DEEPSPEC_PHASE_STOP_UPDATE", "0"))
    config.dataloader.global_microbatch_start = int(
        os.environ.get("DEEPSPEC_PHASE_MICROBATCH_START", "0")
    )
    config.checkpoint = PhaseCheckpointer.Config(
        enable=True,
        folder=str(Path(os.environ["DEEPSPEC_PHASE_TEST_ROOT"]) / "checkpoints"),
        initial_load_path=os.environ.get("DEEPSPEC_PHASE_RESTORE", ""),
        interval=1000,
        enable_first_step_checkpoint=False,
        keep_latest_k=0,
    )
    return config
