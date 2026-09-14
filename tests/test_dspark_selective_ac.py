"""SelectiveAC must preserve complete updates through the Qwen training loop."""

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import time
import tempfile
import unittest

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.distributed import ParallelConfig, ParallelContext
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.trainer.dspark_trainer import Qwen3_8DSparkTrainer
from deepspec.utils.config import to_config_node
from deepspec.utils import training_logger
from deepspec.utils.metrics import configure_reduction_group
from tests.distributed_test_utils import require_torchrun
from tests import test_dspark_training_baseline as baseline
from tests.test_dspark_training_baseline import FixedFeatureTrainer


class DSparkSelectiveACTest(unittest.TestCase):
    def test_online_target_configuration_is_independent_of_draft_checkpoint_policy(
        self,
    ):
        runtime = require_torchrun(self, world_size=2)

        class ModelConstructionReached(Exception):
            pass

        class ConfigurationProbe(Qwen3_8DSparkTrainer):
            def build_models(self):
                # Stop at the external model-loading boundary after exercising
                # the real trainer's configuration and mesh initialization.
                raise ModelConstructionReached(self.target_parallel.config.to_dict())

        target_configs = []
        with tempfile.TemporaryDirectory() as directory:
            for policy in ("full", "torchtitan_selective"):
                args = to_config_node(
                    {
                        "data": {"online_target": True},
                        "train": {
                            "precision": "fp32",
                            "local_batch_size": 1,
                            "parallel": ParallelConfig(
                                dp_shard=2,
                                use_activation_checkpoint=True,
                                activation_checkpoint_policy=policy,
                            ).to_dict(),
                            "target_parallel": {"use_activation_checkpoint": False},
                        },
                        "logging": {
                            "checkpoint_dir": directory,
                            "tensorboard_dir": directory,
                            "logging_steps": 1,
                        },
                    }
                )
                try:
                    with self.assertRaises(ModelConstructionReached) as reached:
                        ConfigurationProbe(runtime.local_rank, args)
                    target_configs.append(reached.exception.args[0])
                finally:
                    training_logger.close()
        self.assertEqual(target_configs[0], target_configs[1])
        self.assertFalse(target_configs[1]["use_activation_checkpoint"])

    def test_selective_ac_preserves_two_fsdp_updates(self):
        runtime = require_torchrun(self, world_size=2)
        plain = ParallelConfig(dp_shard=2, reduce_dtype="fp32")
        selective = replace(
            plain,
            use_activation_checkpoint=True,
            activation_checkpoint_policy="torchtitan_selective",
        )
        topology = ParallelContext.build(selective)
        configure_loss_reduction_group(topology.loss_mesh.get_group())
        configure_reduction_group(topology.loss_mesh.get_group())
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                directory = os.environ.get("DEEPSPEC_BASELINE_REFERENCE")
                fixture = (
                    torch.load(
                        Path(directory) / f"{dtype}_rank{runtime.global_rank}.pt",
                        weights_only=True,
                    )
                    if directory
                    else None
                )
                if fixture is None:
                    reference = FixedFeatureTrainer(
                        runtime, ParallelContext.build(plain), dtype
                    )
                    torch.manual_seed(1000 + runtime.global_rank)
                    fixture = {
                        "initial_weights": reference.initial_weights,
                        "features": reference.features,
                        "initial_cpu_rng": torch.get_rng_state(),
                        "initial_cuda_rng": torch.cuda.get_rng_state(runtime.device),
                    }
                    fixture["result"] = reference.train_and_observe()
                    del reference
                trainer = FixedFeatureTrainer(runtime, topology, dtype, fixture=fixture)
                torch.set_rng_state(fixture["initial_cpu_rng"])
                torch.cuda.set_rng_state(fixture["initial_cuda_rng"], runtime.device)
                actual = trainer.train_and_observe()
                baseline.DSparkTrainingBaselineTest().assert_state_close(
                    actual, fixture["result"]
                )
                self.assertEqual(actual["next_micro_step"], 4)
                self.assertEqual(len(actual["updates"]), 2)
                del trainer
                gc.collect()

    def test_selective_ac_preserves_eight_rank_tp4_updates(self):
        runtime = require_torchrun(self, world_size=8)
        plain = ParallelConfig(dp_shard=2, tp=4, reduce_dtype="fp32")
        topologies = [
            ParallelContext.build(plain),
            ParallelContext.build(
                replace(
                    plain,
                    use_activation_checkpoint=True,
                    activation_checkpoint_policy="torchtitan_selective",
                )
            ),
        ]
        for dtype in (torch.float32, torch.bfloat16):
            fixture = None
            for topology in topologies:
                configure_loss_reduction_group(topology.loss_mesh.get_group())
                configure_reduction_group(topology.loss_mesh.get_group())
                config = Qwen3_5TextConfig(
                    vocab_size=128,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=8,
                    num_key_value_heads=4,
                    head_dim=16,
                    max_position_embeddings=128,
                    layer_types=["full_attention"] * 2,
                )
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(runtime.device)
                started = time.perf_counter()
                trainer = FixedFeatureTrainer(
                    runtime,
                    topology,
                    dtype,
                    fixture=fixture,
                    model_config=config,
                )
                if fixture is None:
                    torch.manual_seed(1000 + topology.data_parallel_rank)
                    fixture = {
                        "initial_weights": trainer.initial_weights,
                        "features": trainer.features,
                        "initial_cpu_rng": torch.get_rng_state(),
                        "initial_cuda_rng": torch.cuda.get_rng_state(runtime.device),
                    }
                torch.set_rng_state(fixture["initial_cpu_rng"])
                torch.cuda.set_rng_state(fixture["initial_cuda_rng"], runtime.device)
                torch.cuda.synchronize(runtime.device)
                built = time.perf_counter()
                actual = trainer.train_and_observe()
                torch.cuda.synchronize(runtime.device)
                print(
                    json.dumps(
                        {
                            "rank": runtime.global_rank,
                            "world_size": runtime.world_size,
                            "dtype": str(dtype),
                            "topology": topology.config.to_dict(),
                            "build_seconds": built - started,
                            "train_seconds": time.perf_counter() - built,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                                runtime.device
                            ),
                            "torch": torch.__version__,
                            "cuda": torch.version.cuda,
                        }
                    ),
                    flush=True,
                )
                if "result" not in fixture:
                    fixture["result"] = actual
                else:
                    baseline.DSparkTrainingBaselineTest().assert_state_close(
                        actual, fixture["result"]
                    )
                del trainer
                gc.collect()


if __name__ == "__main__":
    unittest.main()
