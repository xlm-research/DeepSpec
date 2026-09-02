from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/train/train_qwen3_8_27b_dspark_128gpu.sh"
LOCAL_LAUNCHER = REPO_ROOT / "scripts/train/train_qwen3_8_27b_dspark_128gpu_local.sh"


class Qwen38MultiNodeLauncherTest(unittest.TestCase):
    def _run_launcher(self, *, gpu_count=8, launcher=LAUNCHER, **overrides):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == \"-c\" ]]; then\n"
                "    printf '%s\\n' \"${FAKE_GPU_COUNT}\"\n"
                "    exit 0\n"
                "fi\n"
                "exit 99\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors.index.json").write_text(
                "{}\n", encoding="utf-8"
            )
            source_path = root / "train.jsonl"
            source_path.write_text("{}\n", encoding="utf-8")

            env = os.environ.copy()
            for name in (
                "LOCAL_RANK",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "SENSECORE_PYTORCH_NNODES",
                "SENSECORE_PYTORCH_NODE_RANK",
                "NPROC_PER_NODE",
                "CONTEXT_PARALLEL_SIZE",
                "CP",
                "TENSOR_PARALLEL_SIZE",
                "TP",
                "FSDP_SIZE",
                "TARGET_CACHE_FSDP_SIZE",
                "MAX_LENGTH",
                "BOUNDED_OFFLINE",
                "DATA_PARTITIONS",
                "DATA_BATCH_CACHE_DIR",
                "DATA_BATCH_CACHE_TIMESTAMP",
                "JSONL_INDEX_CACHE_DIR",
            ):
                env.pop(name, None)
            env.update(
                {
                    "PYTHON_BIN": os.fspath(fake_python),
                    "FAKE_GPU_COUNT": str(gpu_count),
                    "NNODES": "2",
                    "NODE_RANK": "0",
                    "MASTER_ADDR": "10.0.0.1",
                    "TARGET_MODEL_PATH": os.fspath(model_dir),
                    "SOURCE_JSONL_PATH": os.fspath(source_path),
                    "TARGET_CACHE_PATH": os.fspath(root / "target-cache"),
                    "OUTPUT_ROOT": os.fspath(root / "output"),
                    "DRY_RUN": "true",
                    "PRODUCTION_RUN": "false",
                    "SAVE_CHECKPOINTS": "false",
                }
            )
            env.update({name: str(value) for name, value in overrides.items()})
            return subprocess.run(
                ["bash", os.fspath(launcher)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_two_node_defaults_use_128k_cp1_tp4(self):
        result = self._run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "target supervision=bounded offline, data partitions per epoch=512",
            result.stdout,
        )
        self.assertIn(
            "topology=DP_REPLICATE=2, DP_SHARD=2, CP=1, TP=4, "
            "effective FSDP shard=2",
            result.stdout,
        )
        self.assertIn("data parallel size=4", result.stdout)
        self.assertIn("gradient accumulation=128", result.stdout)
        self.assertIn("data.max_length=131072", result.stdout)
        self.assertIn("train.parallel.dp_replicate=2", result.stdout)
        self.assertIn("train.parallel.dp_shard=2", result.stdout)
        self.assertIn("train.parallel.cp=1", result.stdout)
        self.assertIn("train.parallel.tp=4", result.stdout)
        self.assertIn("data.offline_target_data_batches=true", result.stdout)
        self.assertIn("train.data_partitions=512", result.stdout)
        self.assertRegex(
            result.stdout,
            r"transient target cache=.*/output/"
            r"model_dp2_fsdp2_cp1_tp4_[0-9]{8}_[0-9]{6}",
        )
        self.assertIn("train.offline_target_parallel.dp_replicate=2", result.stdout)
        self.assertIn("train.offline_target_parallel.cp=1", result.stdout)
        self.assertIn("train.offline_target_parallel.tp=4", result.stdout)
        self.assertNotIn("prepare command:", result.stdout)

    def test_default_cache_dir_uses_model_parallelism_and_timestamp(self):
        result = self._run_launcher(
            DATA_BATCH_CACHE_TIMESTAMP="20260902_153045"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
            result.stdout,
            r"transient target cache=.*/output/"
            r"model_dp2_fsdp2_cp1_tp4_20260902_153045",
        )
        self.assertRegex(
            result.stdout,
            r"data\.data_batch_cache_dir=.*/output/"
            r"model_dp2_fsdp2_cp1_tp4_20260902_153045",
        )

    def test_explicit_cache_dir_overrides_generated_default(self):
        result = self._run_launcher(DATA_BATCH_CACHE_DIR="/tmp/qwen-cache")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("transient target cache=/tmp/qwen-cache", result.stdout)
        self.assertIn(
            "data.data_batch_cache_dir=/tmp/qwen-cache",
            result.stdout,
        )

    def test_data_partition_count_is_configurable(self):
        result = self._run_launcher(DATA_PARTITIONS=73)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "target supervision=bounded offline, data partitions per epoch=73",
            result.stdout,
        )
        self.assertIn("train.data_partitions=73", result.stdout)

    def test_visible_gpus_must_cover_cp_times_tp(self):
        result = self._run_launcher(gpu_count=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "must be divisible by CONTEXT_PARALLEL_SIZE * "
            "TENSOR_PARALLEL_SIZE=4",
            result.stderr,
        )

    def test_sensecore_topology_needs_no_manual_node_variables(self):
        result = self._run_launcher(
            NNODES="",
            NODE_RANK="",
            SENSECORE_PYTORCH_NNODES=2,
            SENSECORE_PYTORCH_NODE_RANK=1,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node=1/2", result.stdout)
        self.assertIn("visible GPUs per node=8", result.stdout)
        self.assertIn("--nproc_per_node 8 --nnodes 2 --node_rank 1", result.stdout)

    def test_gpu_process_world_size_is_converted_to_node_count(self):
        result = self._run_launcher(
            NNODES="",
            NODE_RANK="",
            WORLD_SIZE=16,
            LOCAL_WORLD_SIZE=8,
            RANK=8,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node=1/2", result.stdout)
        self.assertIn("--nproc_per_node 8 --nnodes 2 --node_rank 1", result.stdout)

    def test_local_launcher_uses_single_node_cp1_tp4(self):
        result = self._run_launcher(
            launcher=LOCAL_LAUNCHER,
            NNODES=16,
            NODE_RANK=7,
            NPROC_PER_NODE=4,
            SENSECORE_PYTORCH_NNODES=16,
            SENSECORE_PYTORCH_NODE_RANK=7,
            WORLD_SIZE=128,
            LOCAL_WORLD_SIZE=8,
            RANK=56,
            LOCAL_RANK=0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node=0/1, rendezvous=127.0.0.1:29501", result.stdout)
        self.assertIn("visible GPUs per node=8", result.stdout)
        self.assertIn("--nproc_per_node 8 --nnodes 1 --node_rank 0", result.stdout)
        self.assertIn("CP=1, TP=4", result.stdout)


if __name__ == "__main__":
    unittest.main()
