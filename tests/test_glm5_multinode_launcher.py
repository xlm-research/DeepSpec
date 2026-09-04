from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
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
            "PYTHON_BIN",
            "LOGGING_STEPS",
            "NUM_TRAIN_EPOCHS",
            "MAX_TRAIN_STEPS",
            "DATA_BATCH_SIZE",
            "PARTITIONED_MODEL_SWAP",
            "PARTITION_MAX_SAMPLES",
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
            "TARGET_EP",
            "GLOBAL_BATCH_SIZE",
            "LOG_DIR",
            "TORCHRUN_PER_RANK_LOGS",
            "TARGET_MODEL_PATH",
            "TARGET_MODEL_CACHE_DIR",
            "TARGET_MODEL_CACHE_COPY_WORKERS",
            "DEEPSPEC_DCP_LOAD_THREADS",
        ):
            env.pop(name, None)
        env.update(
            {
                "DRY_RUN": "true",
                "SAVE_CHECKPOINTS": "false",
                "OUTPUT_ROOT": "/shared/deepspec-glm-launcher-test",
                "TRAIN_DATA_PATH": os.fspath(REPO_ROOT / "README.md"),
                "TARGET_MODEL_PATH": os.fspath(REPO_ROOT),
                "TARGET_MODEL_CACHE_DIR": "off",
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

    def test_real_launch_replaces_node_log_and_tees_only_rank_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            for index in range(1, 63):
                (target / f"model-{index:05}-of-00062.safetensors").touch()
            (target / "model.safetensors.index.json").write_text(
                "{}", encoding="utf-8"
            )
            train_data = root / "train.jsonl"
            train_data.write_text("{}\n", encoding="utf-8")
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            output_root = root / "output"
            node_log = output_root / "logs/node_rank_0.log"
            node_log.parent.mkdir(parents=True)
            node_log.write_text("stale launch output\n", encoding="utf-8")

            result = self._run_launcher(
                DRY_RUN="false",
                PYTHON_BIN=fake_python,
                CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(8)),
                TARGET_MODEL_PATH=target,
                TRAIN_DATA_PATH=train_data,
                OUTPUT_ROOT=output_root,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log = node_log.read_text(encoding="utf-8")
            self.assertNotIn("stale launch output", log)
            self.assertRegex(
                log.splitlines()[0],
                r"^\[deepspec-launch-start\] time=\d{4}-\d{2}-\d{2} "
                r"\d{2}:\d{2}:\d{2} [+-]\d{4} host=.+ launch_id=",
            )
            self.assertIn("Launching GLM-5.3-Flash", log)
            self.assertIn("--tee 0:3", log)
            self.assertIn("training command completed successfully", log)

    def test_real_preflight_failure_is_captured_in_node_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"

            result = self._run_launcher(
                DRY_RUN="false",
                PYTHON_BIN="/bin/echo",
                CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(8)),
                TARGET_MODEL_PATH=root / "missing-target",
                TRAIN_DATA_PATH=root / "missing.jsonl",
                OUTPUT_ROOT=output_root,
            )

            self.assertEqual(result.returncode, 2)
            node_log = output_root / "logs/node_rank_0.log"
            log = node_log.read_text(encoding="utf-8")
            self.assertTrue(log.startswith("[deepspec-launch-start] time="))
            self.assertIn("Target model directory does not exist", log)
            self.assertIn("exit_code=2", log)
            self.assertIn("failed before worker logs were created", log)

    def test_real_launch_stages_and_reuses_node_local_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "config.json").write_text("{}\n", encoding="utf-8")
            for index in range(1, 63):
                (target / f"model-{index:05}-of-00062.safetensors").write_text(
                    f"shard {index}\n", encoding="utf-8"
                )
            (target / "model.safetensors.index.json").write_text(
                "{}\n", encoding="utf-8"
            )
            train_data = root / "train.jsonl"
            train_data.write_text("{}\n", encoding="utf-8")
            cache_root = root / "model-cache"

            first = self._run_launcher(
                DRY_RUN="false",
                PYTHON_BIN="/bin/echo",
                CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(8)),
                TARGET_MODEL_PATH=target,
                TARGET_MODEL_CACHE_DIR=cache_root,
                TRAIN_DATA_PATH=train_data,
                OUTPUT_ROOT=root / "output-first",
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("staging complete", first.stdout)
            self.assertIn("with 8 shard workers", first.stdout)
            cached_models = list(cache_root.glob("glm5-*"))
            self.assertEqual(len(cached_models), 1)
            cached_model = cached_models[0]
            self.assertTrue((cached_model / ".deepspec-cache-ready").is_file())
            self.assertIn(
                f"model.target_model_name_or_path={cached_model}", first.stdout
            )

            second = self._run_launcher(
                DRY_RUN="false",
                PYTHON_BIN="/bin/echo",
                CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(8)),
                TARGET_MODEL_PATH=target,
                TARGET_MODEL_CACHE_DIR=cache_root,
                TRAIN_DATA_PATH=train_data,
                OUTPUT_ROOT=root / "output-second",
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("cache hit", second.stdout)
            self.assertIn("target model cache=hit", second.stdout)

    def test_real_worker_failure_reports_error_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            for index in range(1, 63):
                (target / f"model-{index:05}-of-00062.safetensors").touch()
            (target / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            train_data = root / "train.jsonl"
            train_data.write_text("{}\n", encoding="utf-8")
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "log_dir=\n"
                "while (($#)); do\n"
                '    if [[ "$1" == "--log-dir" && $# -ge 2 ]]; then\n'
                "        log_dir=$2\n"
                "        shift 2\n"
                "        continue\n"
                "    fi\n"
                "    shift\n"
                "done\n"
                'mkdir -p "${log_dir}/fake_run/attempt_0/3"\n'
                "printf '%s\\n' '{\"message\":\"boom\"}' "
                '> "${log_dir}/fake_run/attempt_0/3/error.json"\n'
                "exit 17\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            output_root = root / "output"

            result = self._run_launcher(
                DRY_RUN="false",
                PYTHON_BIN=fake_python,
                CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(8)),
                TARGET_MODEL_PATH=target,
                TRAIN_DATA_PATH=train_data,
                OUTPUT_ROOT=output_root,
            )

            self.assertEqual(result.returncode, 17)
            log = (output_root / "logs/node_rank_0.log").read_text(encoding="utf-8")
            self.assertIn("exit_code=17", log)
            self.assertIn("worker failure record=", log)
            self.assertIn("fake_run/attempt_0/3/error.json", log)

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
            "target HSDP: DP_REPLICATE=2, DP_SHARD=2, TP=4, EP=8",
            result.stdout,
        )
        self.assertIn("--nnodes 2 --node_rank 1", result.stdout)
        self.assertIn("train.global_batch_size=16", result.stdout)
        self.assertIn("schedule: dataset-derived, epochs=1", result.stdout)
        self.assertIn("train.max_train_steps=null", result.stdout)

    def test_partitioned_model_swap_sets_distinct_partition_contract(self):
        result = self._run_launcher(
            PARTITIONED_MODEL_SWAP="true",
            PARTITION_MAX_SAMPLES=512,
            SAVE_CHECKPOINTS="true",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("partitioned model swap=true, max global samples=512", result.stdout)
        self.assertIn("train.data_batch_size=null", result.stdout)
        self.assertIn("train.partitioned_model_swap.enabled=true", result.stdout)
        self.assertIn("train.partitioned_model_swap.max_samples=512", result.stdout)

    def test_partitioned_model_swap_requires_checkpoints(self):
        result = self._run_launcher(PARTITIONED_MODEL_SWAP="true")
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "PARTITIONED_MODEL_SWAP=true requires SAVE_CHECKPOINTS=true",
            result.stderr,
        )

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
            "target HSDP: DP_REPLICATE=1, DP_SHARD=3, TP=4, EP=12",
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
            "target HSDP: DP_REPLICATE=2, DP_SHARD=2, TP=4, EP=8",
            result.stdout,
        )
        self.assertIn("train.data_batch_size=256", result.stdout)
        self.assertIn(
            "data.data_batch_cache_dir=/shared/deepspec-glm-launcher-test/"
            "data_batch_cache/"
            "target_dp2_fsdp2_cp1_tp4_ep8_etp1__"
            "draft_dp2_fsdp8_cp1_tp1_ep8_etp1",
            result.stdout,
        )

    def test_explicit_data_batch_cache_dir_overrides_topology_default(self):
        result = self._run_launcher(DATA_BATCH_CACHE_DIR="/local/glm-cache")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "node-local or shared transient cache=/local/glm-cache",
            result.stdout,
        )
        self.assertIn(
            "data.data_batch_cache_dir=/local/glm-cache",
            result.stdout,
        )

    def test_sensecore_topology_takes_precedence_over_stale_manual_values(self):
        result = self._run_launcher(
            NNODES=1,
            NODE_RANK=0,
            SENSECORE_PYTORCH_NNODES=2,
            SENSECORE_PYTORCH_NODE_RANK=1,
            WORLD_SIZE=2,
            RANK=1,
            MASTER_ADDR="10.0.0.1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node=1/2", result.stdout)
        self.assertIn("topology source=SenseCore", result.stdout)
        self.assertIn("--nnodes 2 --node_rank 1", result.stdout)

    def test_uses_selected_python_for_torch_distributed_launcher(self):
        result = self._run_launcher(
            PYTHON_BIN="/bin/echo",
            CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(8)),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "launcher=/bin/echo -m torch.distributed.run",
            result.stdout,
        )
        self.assertIn(
            "/bin/echo -m torch.distributed.run --nproc_per_node",
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
            "target HSDP: DP_REPLICATE=2, DP_SHARD=3, TP=4, EP=12",
            result.stdout,
        )

    def test_explicit_step_limit_remains_a_diagnostic_override(self):
        result = self._run_launcher(MAX_TRAIN_STEPS=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data partitions=256", result.stdout)
        self.assertIn("schedule: diagnostic max steps=5", result.stdout)
        self.assertIn("train.max_train_steps=5", result.stdout)
        self.assertNotIn("train.max_train_steps=null", result.stdout)

    def test_logging_steps_can_be_overridden(self):
        result = self._run_launcher(LOGGING_STEPS=7)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("logging steps=7", result.stdout)
        self.assertIn("logging.logging_steps=7", result.stdout)

    def test_target_dcp_reader_uses_measured_default_and_can_be_overridden(self):
        default = self._run_launcher()
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertIn("target DCP reader threads=8", default.stdout)

        overridden = self._run_launcher(DEEPSPEC_DCP_LOAD_THREADS=3)
        self.assertEqual(overridden.returncode, 0, overridden.stderr)
        self.assertIn("target DCP reader threads=3", overridden.stdout)

        invalid = self._run_launcher(DEEPSPEC_DCP_LOAD_THREADS=0)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("DEEPSPEC_DCP_LOAD_THREADS must be", invalid.stderr)

    def test_total_gpu_count_must_satisfy_requested_target_tp4(self):
        result = self._run_launcher(
            NNODES=3,
            NODE_RANK=0,
            NPROC_PER_NODE=2,
            MASTER_ADDR="10.0.0.1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("target TP=4", result.stderr)

    def test_target_ep_must_divide_sparse_domain_and_experts(self):
        result = self._run_launcher(TARGET_EP=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "TARGET_EP must divide both target DP_SHARD*TP=8 and 288 experts",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
