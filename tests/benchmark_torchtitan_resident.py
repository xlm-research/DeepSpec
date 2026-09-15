"""Retained resident trainer on the native acceptance's exact prepared features.

This benchmark uses two FSDP ranks, matching the two logical DP samples per
microbatch. Its six unused GPUs must be reported when comparing with native TP4.
It is a reference measurement, not a native phase-contract acceptance test.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.distributed import ParallelConfig, ParallelContext, apply_parallelism
from deepspec.distributed.context_parallel import FixedContextParallel
from deepspec.distributed.distributed_checkpoint import (
    save_training_checkpoint,
    TrainingProgress,
)
from deepspec.distributed.runtime import initialize_runtime
from deepspec.modeling.dspark.qwen3_8 import Qwen3_8DSparkModel
from deepspec.trainer.dspark_trainer import Qwen3_8DSparkTrainer
from deepspec.training import BF16Optimizer
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.utils import training_logger
from deepspec.utils.config import to_config_node
from deepspec.utils.hfai_suspend import SuspendController
from deepspec.utils.metrics import configure_reduction_group
from torchtitan.models.dspark_draft.data import FeatureLoader, PreparedTokens
from torchtitan.models.dspark_draft.timing import PhaseTiming
from tests.torchtitan_scale_fixtures import restore_rng


class TimedOptimizer(BF16Optimizer):
    def step(self):
        super().step()
        torch.cuda.synchronize()
        finished = time.monotonic()
        self.update_seconds.append(finished - self.last_update)
        self.last_update = finished


class ResidentReference(Qwen3_8DSparkTrainer):
    def __init__(self, request, runtime):
        self.device = runtime.device
        self.global_rank = runtime.global_rank
        self.world_size = runtime.world_size
        if self.world_size != 2:
            raise ValueError(
                "The resident reference requires the matching two DP ranks"
            )
        self.parallel_config = ParallelConfig(
            dp_shard=2,
            reduce_dtype="fp32",
            use_activation_checkpoint=True,
            activation_checkpoint_policy="torchtitan_selective",
        )
        self.parallel = ParallelContext.build(self.parallel_config)
        group = self.parallel.loss_mesh.get_group()
        configure_loss_reduction_group(group)
        configure_reduction_group(group)
        self.fixed_context_parallel = FixedContextParallel(
            self.parallel, backend=self.parallel_config.context_parallel_backend
        )
        self.plan = json.loads(Path(request["plan_path"]).read_text())
        recipe = self.plan["resolved_recipe"]
        model_config = recipe["model_spec"]["model"]["hf_config"]
        expected = {
            "hidden_size": 5120,
            "vocab_size": 248320,
            "num_hidden_layers": 5,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "num_anchors": 512,
            "block_size": 7,
        }
        if (
            any(model_config[key] != value for key, value in expected.items())
            or self.plan["global_batch_size"] != 4
            or self.plan["gradient_accumulation_steps"] != 2
            or self.plan["training_steps"] != 10
            or any(batch["length"] != 131072 for batch in self.plan["batches"])
            or recipe["lr_scheduler"]["warmup_steps"] != 40
            or recipe["lr_scheduler"]["total_steps"] != 1000
        ):
            raise ValueError("The resident reference requires the fixed 128K workload")
        self.args = to_config_node(
            {
                "model": {
                    "ce_loss_alpha": 0.1,
                    "l1_loss_alpha": 0.9,
                    "confidence_head_alpha": 1.0,
                    "loss_decay_gamma": 4.0,
                },
                "train": {"local_batch_size": 1, "max_grad_norm": 1.0},
                "logging": {"save_checkpoints": False},
            }
        )
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        config = Qwen3_5TextConfig.from_dict(model_config)
        config._attn_implementation = "flex_attention"
        torch.manual_seed(42)
        draft = Qwen3_8DSparkModel(config).to(dtype=torch.bfloat16)
        reference = Path(request["initialization"])
        draft.load_state_dict(
            torch.load(reference / "initial-weights.pt", weights_only=True)
        )
        draft = draft.to(self.device)
        draft.set_embedding_head_trainable(False)
        draft.configure_context_parallel(
            size=1,
            rank=0,
            group=self.parallel.cp_mesh.get_group(),
            model_parallel_group=self.parallel.model_mesh.get_group(),
            model_parallel_src_rank=self.parallel.model_parallel_src_rank,
        )
        self.draft_model = draft
        self.model = apply_parallelism(
            draft, self.parallel, self.parallel_config, param_dtype=torch.bfloat16
        )
        self.optimizer = TimedOptimizer(
            self.model, lr=6e-4, total_steps=1000, warmup_ratio=0.04
        )
        self.suspend_controller = SuspendController(self.device)
        self._pure_expert_modules = []
        self._draft_residency_verified = False
        self.partitioned_model_swap_enabled = False
        self.online_target_enabled = self.offline_target_data_batches_enabled = False
        self.data_batch_micro_batches = None
        self._data_batch_end_after_current = False
        self.next_micro_step = 0
        self.gradient_accumulation_steps = self.plan["gradient_accumulation_steps"]
        self.max_train_steps = self.plan["training_steps"]
        self.micro_batches_per_epoch = self.plan["samples_per_epoch"] // 2
        self.request = request
        self.observations = []
        self.model.register_forward_hook(self.observe_forward)
        state = torch.load(
            reference / f"rng-rank{runtime.global_rank * 4}.pt", weights_only=True
        )
        restore_rng(state, self.device)

    def observe_forward(self, module, args, output):
        self.observations.append(
            {
                "target_ids": output.target_ids.detach().cpu(),
                "eval_mask": output.eval_mask.detach().cpu(),
                "block_keep_mask": output.block_keep_mask.detach().cpu(),
                "next_microbatch": self.next_micro_step + 1,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device),
            }
        )

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        if start_offset_samples != 0 or num_samples != self.max_train_steps * 2:
            raise ValueError("This resident measurement runs the entire fixed workload")
        for partition in self.request["partitions"]:
            loader = FeatureLoader(
                FeatureLoader.Config(
                    manifest=partition["manifest"],
                    global_microbatch_start=partition["microbatch_start"],
                    plan_path=self.request["plan_path"],
                    target_layer_ids=[1, 16, 31, 46, 61],
                    hidden_size=5120,
                    require_producer_manifest=True,
                ),
                dp_world_size=2,
                dp_rank=self.global_rank,
                tokenizer=PreparedTokens(
                    PreparedTokens.Config(vocab_size=248320), tokenizer_path=""
                ),
                max_context_length=131072,
                num_tokens_per_batch=131072,
            )
            for batch, _ in loader:
                batch.pop("num_valid_tokens")
                yield batch


def run(request):
    timing = PhaseTiming(True)
    runtime = initialize_runtime()
    trainer = ResidentReference(request, runtime)
    timing.record("initialize", timing.started)
    root = Path(request["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    training_logger.init(logging_steps=1, tensorboard_dir=str(root / "metrics"))
    trainer.optimizer.update_seconds = []
    trainer.optimizer.last_update = time.monotonic()
    with timing.measure("training"):
        trainer.train()
    if trainer.global_step != 10 or trainer.next_micro_step != 20:
        raise RuntimeError("The resident reference did not complete its fixed workload")
    with timing.measure("save"):
        save_training_checkpoint(
            checkpoint_dir=str(root / "checkpoint"),
            model=trainer.model,
            optimizer_bundle=trainer.optimizer,
            progress=TrainingProgress(
                next_micro_step=trainer.next_micro_step,
                global_step=trainer.global_step,
                epoch=0,
                data_position=trainer.next_micro_step,
                local_batch_size=1,
                saved_world_size=2,
                parallel_config=trainer.parallel_config.to_dict(),
                model_config=trainer.draft_model.config.to_dict(),
            ),
        )
    with timing.measure("close"):
        training_logger.close()
        torch.save(
            {
                "supervision": trainer.observations,
                "update_seconds": trainer.optimizer.update_seconds,
            },
            root / f"observations-rank{runtime.global_rank}.pt",
        )
    report = timing.report()
    workers = [None] * runtime.world_size
    dist.all_gather_object(workers, os.getpid())
    if runtime.global_rank == 0:
        (root / "result.json").write_text(
            json.dumps(
                {
                    "completed_updates": 10,
                    "actual_workers": 2,
                    "worker_pids": workers,
                    "timing": report,
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    arguments = parser.parse_args()
    try:
        run(json.loads(arguments.request.read_text()))
    except Exception:
        import sys
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
