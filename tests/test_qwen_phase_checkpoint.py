"""Persist a complete draft phase through the retained real Qwen entry."""

import copy
from contextlib import ExitStack
import gc
import json
import os
import pickle
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
import torch.distributed as dist

from deepspec.distributed import ParallelConfig
from deepspec.trainer.draft_phase_checkpoint import validate_draft_phase_checkpoint
from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json
from deepspec.utils import training_logger
from deepspec.utils.config import to_config_node
from tests.distributed_test_utils import require_torchrun
from tests.test_qwen_producer_isolation import ProducerFixtureTrainer, cpu_state


def training_state(trainer, device):
    return {
        "observations": trainer.observations,
        "gradients_after_clip": trainer.gradients,
        "model": cpu_state(trainer.model.state_dict()),
        "optimizer": cpu_state(trainer.optimizer.state_dict()),
        "next_micro_step": trainer.next_micro_step,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device),
        "python_rng": pickle.dumps(random.getstate()),
        "numpy_rng": pickle.dumps(np.random.get_state()),
    }


class QwenDraftPhaseCheckpointTest(unittest.TestCase):
    def test_completed_phase_commits_full_dcp_without_hf_export(self):
        mode = os.environ.get("DEEPSPEC_PHASE_TEST_MODE", "full")
        if mode == "failures":
            for failure in ("first", "later", "truncated", "commit"):
                with self.subTest(failure=failure):
                    self.run_phase_scenario("failure", failure=failure)
        else:
            self.run_phase_scenario(mode)

    def run_phase_scenario(self, mode, *, failure=None):
        runtime = require_torchrun(self, world_size=2)
        reference = os.environ.get("DEEPSPEC_BASELINE_REFERENCE")
        if not reference:
            self.skipTest("set DEEPSPEC_BASELINE_REFERENCE to captured Qwen features")
        fixtures = [
            torch.load(
                Path(reference) / f"torch.float32_rank{rank}.pt", weights_only=True
            )
            for rank in range(2)
        ]
        batches = [batch for fixture in fixtures for batch in fixture["features"]]
        by_tokens = {tuple(batch["input_ids"][0].tolist()): batch for batch in batches}
        persistent_root = os.environ.get("DEEPSPEC_PHASE_TEST_ROOT")
        if failure and persistent_root:
            persistent_root = str(Path(persistent_root) / failure)
        temporary = (
            tempfile.TemporaryDirectory()
            if runtime.global_rank == 0 and not persistent_root
            else None
        )
        roots = [persistent_root or (temporary.name if temporary else None)]
        dist.broadcast_object_list(roots, src=0)
        assert roots[0] is not None
        root = Path(roots[0])
        checkpoint_root = Path(
            os.environ.get("DEEPSPEC_PHASE_CHECKPOINT_DIR", str(root / "checkpoints"))
        )
        input_root = Path(os.environ.get("DEEPSPEC_PHASE_INPUT_ROOT", str(root)))
        dataset = input_root / "inputs.jsonl"
        if runtime.global_rank == 0:
            root.mkdir(parents=True, exist_ok=True)
        if runtime.global_rank == 0 and not dataset.exists():
            input_root.mkdir(parents=True, exist_ok=True)
            dataset.write_text(
                "".join(
                    json.dumps(
                        {
                            "input_ids": batch["input_ids"][0].tolist(),
                            "loss_mask": batch["loss_mask"][0].tolist(),
                        }
                    )
                    + "\n"
                    for batch in batches
                )
            )
        if runtime.global_rank == 0 and not (root / "config.py").exists():
            (root / "config.py").write_text(
                "# Real Qwen fixture; resolved configuration accompanies the DCP.\n"
            )
        dist.barrier()
        args = to_config_node(
            {
                "exp_name": "phase-checkpoint-fixture",
                "_origin_config_path": str(root / "config.py"),
                "_origin_opts": [],
                "model": {
                    "target_model_name_or_path": "fixed-fixture",
                    "target_layer_ids": [1, 3],
                    "ce_loss_alpha": 0.1,
                    "l1_loss_alpha": 0.9,
                    "confidence_head_alpha": 1.0,
                    "loss_decay_gamma": 4.0,
                },
                "train": {
                    "draft_phase_unload": mode == "lifecycle",
                    "parallel": ParallelConfig(dp_shard=2).to_dict(),
                    "offline_target_parallel": ParallelConfig(tp=2).to_dict(),
                    "precision": "fp32",
                    "local_batch_size": 1,
                    "global_batch_size": 4,
                    "num_train_epochs": 1,
                    "max_train_steps": 2,
                    "data_partitions": 2,
                    "lr": 1e-3,
                    "warmup_ratio": 0.25,
                    "weight_decay": 0.0,
                    "max_grad_norm": 0.5,
                    "qwen_vllm": {
                        "python_executable": sys.executable,
                        "source_dir": str(Path("vllm").resolve()),
                        "tensor_parallel_size": 1,
                    },
                },
                "data": {
                    "offline_target_data_batches": True,
                    "train_data_path": str(dataset),
                    "max_length": 16,
                    "num_workers": 0,
                    "jsonl_index_cache_dir": str(root / "json-index"),
                    "data_batch_cache_dir": str(root / "cache"),
                },
                "logging": {
                    "checkpoint_dir": str(checkpoint_root),
                    "tensorboard_dir": str(root / "tensorboard"),
                    "save_checkpoints": mode != "continuous",
                    "checkpointing_steps": 1,
                    "logging_steps": 1,
                },
            }
        )
        teacher_starts = []
        if failure or mode == "resume_failure":
            args.train.max_train_steps = 3
            args.train.num_train_epochs = 2
        if mode == "suspend":
            args.train.data_partitions = 1

        def produce(job_path, config, devices):
            teacher_starts.append(trainer.next_micro_step)
            if mode == "lifecycle" and trainer.next_micro_step:
                self.assertIsNone(
                    trainer.model, "draft model remains resident in target phase"
                )
                self.assertIsNone(
                    trainer.optimizer,
                    "draft optimizer remains resident in target phase",
                )
                self.assertTrue(
                    all(reference() is None for reference in original_state)
                )
                # External extraction must not perturb the restored draft RNG.
                torch.rand(37, device=runtime.device)
                torch.rand(29)
                random.random()
                np.random.rand(31)
            job = load_json(job_path)
            for request in job["requests"]:
                batch = torch.load(request["input_path"], weights_only=True)
                feature = by_tokens[tuple(batch["input_ids"][0].tolist())]
                path = Path(request["output_paths"][0])
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(feature, context_chunk_len=torch.tensor([16])), path)
            atomic_write_json(
                str(job_path) + ".complete",
                {
                    "teacher": job["teacher"],
                    "samples": [{} for _ in job["requests"]],
                },
            )

        fixture = fixtures[runtime.global_rank]
        if mode in ("resume", "resume_failure", "reject"):
            metadata = load_json(
                checkpoint_root / "step_latest" / "distributed_checkpoint_metadata.json"
            )
            fixture = dict(
                fixture,
                model_config=metadata["model_config"],
                initial_weights={
                    name: torch.zeros_like(value)
                    for name, value in fixture["initial_weights"].items()
                },
            )
        if mode == "reject":
            cases = (
                ("model", "ce_loss_alpha", 0.2),
                ("train", "lr", 0.002),
                ("train", "global_batch_size", 8),
                ("train", "parallel", ParallelConfig(dp_replicate=2).to_dict()),
                ("data", "min_loss_tokens", 2),
            )
            for section, name, value in cases:
                changed = copy.deepcopy(args)
                changed[section][name] = value
                with self.subTest(configuration=f"{section}.{name}"):
                    with self.assertRaisesRegex(
                        (ValueError, RuntimeError), "mismatch|changed"
                    ):
                        ProducerFixtureTrainer(runtime.local_rank, changed, fixture)
                training_logger.close()
                gc.collect()
                torch.cuda.empty_cache()
            dist.barrier()
            if temporary:
                temporary.cleanup()
            return
        trainer = ProducerFixtureTrainer(runtime.local_rank, args, fixture)
        original_state = [
            weakref.ref(value)
            for value in (
                trainer.model,
                trainer.optimizer,
                *trainer.model.parameters(),
                *trainer.model.buffers(),
                *(
                    value
                    for state in trainer.optimizer.optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                ),
            )
        ]
        if mode == "save":
            trainer._active_train_end_step = 1
        if mode == "resume_failure":
            trainer._active_train_end_step = 2
        if mode not in ("resume", "resume_failure"):
            torch.manual_seed(1000 + runtime.global_rank)
            random.seed(3000 + runtime.global_rank)
            np.random.seed(4000 + runtime.global_rank)
        try:
            if failure:
                original_fsync = os.fsync
                original_rename = os.rename
                failed_step = 2 if failure in ("later", "commit") else 1
                injected = False

                def fail_storage_sync(descriptor):
                    nonlocal injected
                    path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
                    if (
                        not injected
                        and failure != "commit"
                        and runtime.global_rank == 0
                        and f".step_{failed_step}.incomplete-" in str(path)
                        and path.suffix == ".distcp"
                    ):
                        injected = True
                        if failure == "truncated":
                            os.truncate(path, 0)
                        else:
                            raise OSError("injected checkpoint storage sync failure")
                    return original_fsync(descriptor)

                def fail_commit(source, destination, *args, **kwargs):
                    nonlocal injected
                    if (
                        failure == "commit"
                        and f".step_{failed_step}.incomplete-" in str(source)
                    ):
                        injected = True
                        raise OSError("injected checkpoint commit failure")
                    return original_rename(source, destination, *args, **kwargs)

                with patch(
                    "deepspec.trainer.qwen3_8_vllm_trainer.run_worker_process", produce
                ):
                    with (
                        patch("os.fsync", fail_storage_sync),
                        patch("os.rename", fail_commit),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                            trainer.train()
                self.assertEqual(trainer.next_micro_step, failed_step * 2)
                self.assertFalse((checkpoint_root / f"step_{failed_step}").exists())
                if runtime.global_rank == 0:
                    self.assertTrue(injected)
                    self.assertEqual(
                        teacher_starts, [0, 2] if failed_step == 2 else [0]
                    )
                latest = trainer.discover_resume_checkpoint()
                if failed_step == 2:
                    self.assertEqual(Path(latest), checkpoint_root / "step_1")
                    self.assertEqual(
                        (checkpoint_root / "step_latest").resolve(), Path(latest)
                    )
                else:
                    self.assertIsNone(latest)
                    self.assertFalse((checkpoint_root / "step_latest").exists())
                if persistent_root and failed_step == 2:
                    torch.save(
                        training_state(trainer, runtime.device),
                        root / f"failure_rank{runtime.global_rank}.pt",
                    )
                return
            suspended = []
            with ExitStack() as external:
                external.enter_context(
                    patch(
                        "deepspec.trainer.qwen3_8_vllm_trainer.run_worker_process",
                        produce,
                    )
                )
                if mode == "suspend":
                    external.enter_context(
                        patch.object(
                            trainer.suspend_controller, "requested", return_value=True
                        )
                    )
                    external.enter_context(
                        patch.object(
                            trainer.suspend_controller,
                            "go_suspend",
                            side_effect=lambda: suspended.append(
                                trainer.next_micro_step
                            ),
                        )
                    )
                trainer.train()
            expected_micro_step = 2 if mode == "save" else 4
            self.assertEqual(trainer.next_micro_step, expected_micro_step)
            if mode != "continuous":
                checkpoint = checkpoint_root / f"step_{expected_micro_step // 2}"
                self.assertTrue(
                    (checkpoint / "distributed_checkpoint" / ".metadata").is_file()
                )
                self.assertFalse(list(checkpoint.glob("*.safetensors")))
                self.assertEqual(
                    (checkpoint_root / "step_latest").resolve(), checkpoint
                )
                self.assertTrue((checkpoint / "draft_feature_index.json").is_file())
                self.assertTrue((checkpoint / "draft_phase_commit.json").is_file())
                validate_draft_phase_checkpoint(checkpoint)
            if mode == "suspend" and runtime.global_rank == 0:
                self.assertEqual(suspended, [4])
            if runtime.global_rank == 0:
                expected_starts = (
                    [0]
                    if mode in ("save", "suspend")
                    else [2]
                    if mode in ("resume", "resume_failure")
                    else [0, 2]
                )
                self.assertEqual(teacher_starts, expected_starts)
            if persistent_root:
                torch.save(
                    training_state(trainer, runtime.device),
                    root / f"{mode}_rank{runtime.global_rank}.pt",
                )
        finally:
            trainer.train_dataset.close()
            training_logger.close()
            dist.barrier()
            if temporary:
                temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
