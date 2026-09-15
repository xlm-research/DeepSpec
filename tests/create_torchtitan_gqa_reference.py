"""Record an additional real 24Q/4KV DSpark reference for TP4 acceptance."""

import hashlib
import json
import os
from pathlib import Path
import unittest

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.distributed import ParallelConfig, ParallelContext
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.utils.metrics import configure_reduction_group

from tests.distributed_test_utils import require_torchrun
from tests.test_dspark_training_baseline import (
    DSparkTrainingBaselineTest,
    FixedFeatureTrainer,
)


def main():
    workers = int(os.environ.get("DEEPSPEC_GQA_REFERENCE_WORKERS", "2"))
    runtime = require_torchrun(unittest.TestCase(), world_size=workers)
    root = Path(os.environ["DEEPSPEC_GQA_REFERENCE_OUTPUT"]).resolve()
    if runtime.global_rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    topology = ParallelContext.build(
        ParallelConfig(dp_shard=workers, reduce_dtype="fp32")
    )
    configure_loss_reduction_group(topology.loss_mesh.get_group())
    configure_reduction_group(topology.loss_mesh.get_group())
    for dtype in (torch.float32, torch.bfloat16):
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
            layer_types=["full_attention"] * 2,
        )
        trainer = FixedFeatureTrainer(runtime, topology, dtype, model_config=config)
        torch.manual_seed(1000 + runtime.global_rank)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(runtime.device)
        actual = trainer.train_and_observe()
        independent = FixedFeatureTrainer(
            runtime, topology, dtype, model_config=config, independent_loss=True
        )
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, runtime.device)
        expected = independent.train_and_observe()
        DSparkTrainingBaselineTest().assert_state_close(actual, expected)
        path = root / f"{dtype}_rank{runtime.global_rank}.pt"
        torch.save(
            {
                "initial_weights": trainer.initial_weights,
                "features": trainer.features,
                "initial_cpu_rng": cpu_rng,
                "initial_cuda_rng": cuda_rng,
                "model_config": config.to_dict(),
                "result": actual,
                "parallel_config": {"dp_shard": workers, "tp": 1, "cp": 1},
            },
            path,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "query_heads": 24,
                    "kv_heads": 4,
                    "dtype": str(dtype),
                    "updates": 2,
                    "gas": 2,
                    "data_parallel_size": workers,
                    "independent_objective_match": True,
                },
                indent=2,
            )
        )
    dist.barrier()
    if runtime.global_rank == 0:
        print(
            f"PASS: real {workers}-rank 24Q/4KV FP32 and BF16 references, two updates each",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
