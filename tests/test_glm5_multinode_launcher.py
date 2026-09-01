from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh"


class Glm5MultiNodeLauncherTest(unittest.TestCase):
    def _run_launcher(self, **overrides):
        env = os.environ.copy()
        for name in (
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "NNODES",
            "NODE_RANK",
            "NPROC_PER_NODE",
            "SENSECORE_PYTORCH_NNODES",
            "SENSECORE_PYTORCH_NODE_RANK",
            "CUDA_VISIBLE_DEVICES",
            "NUM_TRAIN_EPOCHS",
            "MAX_TRAIN_STEPS",
            "DATA_BATCH_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "SLURM_NNODES",
            "SLURM_NODEID",
            "SLURM_JOB_NODELIST",
            "TMPDIR",
            "TRAIN_DATA_PATH",
            "TARGET_MODEL_PATH",
            "DATA_BATCH_CACHE_DIR",
            "DP_REPLICATE",
            "DP_SHARD",
            "DRAFT_EP",
            "TARGET_DP_REPLICATE",
            "TARGET_DP_SHARD",
            "GLOBAL_BATCH_SIZE",
        ):
            env.pop(name, None)
        env.update(
            {
                "DRY_RUN": "true",
                "SAVE_CHECKPOINTS": "false",
                "OUTPUT_ROOT": "/shared/deepspec-glm-launcher-test",
                "TRAIN_DATA_PATH": os.fspath(REPO_ROOT / "README.md"),
                "TARGET_MODEL_PATH": os.fspath(REPO_ROOT),
            }
        )
        env.update({name: str(value) for name, value in overrides.items()})
        return subprocess.run(
            ["bash", os.fspath(LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_two_nodes_keep_tp4_and_fsdp_shards_node_local(self):
        result = self._run_launcher(
            NNODES=2,
            NODE_RANK=1,
            NPROC_PER_NODE=8,
            MASTER_ADDR="10.0.0.1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training on 16 GPUs", result.stdout)
        self.assertIn(
            "draft HSDP: DP_REPLICATE=2, DP_SHARD=8, EP=8",
            result.stdout,
        )
        self.assertIn(
            "target HSDP: DP_REPLICATE=2, DP_SHARD=2, TP=4, EP=1",
            result.stdout,
        )
        self.assertIn("--nnodes 2 --node_rank 1", result.stdout)
        self.assertIn("train.global_batch_size=16", result.stdout)
        self.assertIn("schedule: dataset-derived, epochs=1", result.stdout)
        self.assertIn("train.max_train_steps=null", result.stdout)

    def test_non_multiple_of_four_local_shape_uses_global_tp4_mesh(self):
        result = self._run_launcher(
            NNODES=2,
            NODE_RANK=0,
            NPROC_PER_NODE=6,
            MASTER_ADDR="10.0.0.1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training on 12 GPUs", result.stdout)
        self.assertIn(
            "draft HSDP: DP_REPLICATE=2, DP_SHARD=6, EP=6",
            result.stdout,
        )
        self.assertIn(
            "target HSDP: DP_REPLICATE=1, DP_SHARD=3, TP=4, EP=1",
            result.stdout,
        )

    def test_torchrun_style_scheduler_world_size_is_gpu_process_count(self):
        result = self._run_launcher(
            WORLD_SIZE=16,
            LOCAL_WORLD_SIZE=8,
            RANK=8,
            MASTER_ADDR="10.0.0.1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node=1/2", result.stdout)
        self.assertIn("WORLD_SIZE=16 (gpu_processes)", result.stdout)

    def test_sensecore_job_needs_no_manual_topology_variables(self):
        result = self._run_launcher(
            SENSECORE_PYTORCH_NNODES=2,
            SENSECORE_PYTORCH_NODE_RANK=1,
            WORLD_SIZE=2,
            RANK=1,
            MASTER_ADDR="10.0.0.1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node=1/2", result.stdout)
        self.assertIn("training on 16 GPUs", result.stdout)
        self.assertIn("topology source=SenseCore", result.stdout)
        self.assertIn(
            "draft HSDP: DP_REPLICATE=2, DP_SHARD=8, EP=8",
            result.stdout,
        )
        self.assertIn(
            "target HSDP: DP_REPLICATE=2, DP_SHARD=2, TP=4, EP=1",
            result.stdout,
        )
        self.assertIn("train.data_batch_size=256", result.stdout)
        self.assertIn(
            "data.data_batch_cache_dir=/tmp/deepspec_glm5_target_cache_29501",
            result.stdout,
        )

    def test_gpu_count_per_node_is_not_fixed_to_eight(self):
        result = self._run_launcher(
            NNODES=2,
            NODE_RANK=0,
            NPROC_PER_NODE=12,
            CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(12)),
            MASTER_ADDR="10.0.0.1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training on 24 GPUs", result.stdout)
        self.assertIn(
            "draft HSDP: DP_REPLICATE=2, DP_SHARD=12, EP=12",
            result.stdout,
        )
        self.assertIn(
            "target HSDP: DP_REPLICATE=2, DP_SHARD=3, TP=4, EP=1",
            result.stdout,
        )

    def test_explicit_step_limit_remains_a_diagnostic_override(self):
        result = self._run_launcher(MAX_TRAIN_STEPS=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data partitions=256", result.stdout)
        self.assertIn("schedule: diagnostic max steps=5", result.stdout)
        self.assertIn("train.max_train_steps=5", result.stdout)
        self.assertNotIn("train.max_train_steps=null", result.stdout)

    def test_total_gpu_count_must_satisfy_requested_target_tp4(self):
        result = self._run_launcher(
            NNODES=3,
            NODE_RANK=0,
            NPROC_PER_NODE=2,
            MASTER_ADDR="10.0.0.1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("target TP=4", result.stderr)


if __name__ == "__main__":
    unittest.main()
