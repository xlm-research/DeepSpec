"""Native TorchTitan draft modules preserve the retained DSpark update contract."""

import gc
import os
from pathlib import Path
import unittest

import torch

from deepspec.distributed import ParallelConfig, ParallelContext
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.utils.metrics import add_metric, configure_reduction_group
from tests import test_dspark_training_baseline as baseline
from tests.distributed_test_utils import require_torchrun


class TorchTitanDSparkDraftTest(unittest.TestCase):
    def test_native_draft_preserves_two_complete_fsdp_updates(self):
        runtime = require_torchrun(self, world_size=2)
        reference = os.environ.get("DEEPSPEC_BASELINE_REFERENCE")
        if not reference:
            self.skipTest("requires captured immutable Qwen features and updates")
        from torchtitan.models.dspark_draft import (
            DSparkStateDictAdapter,
            build_draft_config,
        )

        topology = ParallelContext.build(ParallelConfig(dp_shard=2, reduce_dtype="fp32"))
        configure_loss_reduction_group(topology.loss_mesh.get_group())
        configure_reduction_group(topology.loss_mesh.get_group())
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                fixture = torch.load(
                    Path(reference) / f"{dtype}_rank{runtime.global_rank}.pt", weights_only=True
                )
                config = build_draft_config(fixture["model_config"])
                adapter = DSparkStateDictAdapter(config, None)

                def build(_):
                    model = config.build()
                    model.metric_recorder = add_metric
                    return model

                native_fixture = dict(fixture, initial_weights=adapter.from_hf(fixture["initial_weights"]))
                trainer = baseline.FixedFeatureTrainer(
                    runtime, topology, dtype, fixture=native_fixture, model_factory=build
                )
                torch.set_rng_state(fixture["initial_cpu_rng"])
                torch.cuda.set_rng_state(fixture["initial_cuda_rng"], runtime.device)
                actual = trainer.train_and_observe()
                # Parameter ownership follows TorchTitan's normal FQNs; compare
                # the same weights/gradients through its checkpoint adapter.
                for update in actual["updates"]:
                    for field in ("gradients_after_clip", "parameters", "adam"):
                        update[field] = adapter.to_hf(update[field])
                baseline.DSparkTrainingBaselineTest().assert_state_close(actual, fixture["result"])
                del trainer
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
