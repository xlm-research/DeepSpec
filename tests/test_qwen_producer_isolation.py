"""A fixed external producer serves independent real draft training layouts."""

from dataclasses import fields
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.distributed import ParallelConfig
from deepspec.modeling.dspark.qwen3_8 import Qwen3_8DSparkModel
from deepspec.trainer.qwen3_8_vllm_trainer import Qwen3_8VllmDSparkTrainer
from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json
from deepspec.utils.config import to_config_node
from deepspec.utils import training_logger
from tests.distributed_test_utils import require_torchrun
from tests import test_dspark_training_baseline as baseline
from tests.test_dspark_training_baseline import cpu_tensor


class ProducerFixtureTrainer(Qwen3_8VllmDSparkTrainer):
    def __init__(self, local_rank, args, fixture):
        self.fixture = fixture
        self._teacher_identity = {"teacher": "immutable-features"}
        super().__init__(local_rank, args)
        self.observations = []
        self.gradients = []
        self._output = None
        self.model.register_forward_hook(self.observe_forward)
        self.optimizer.optimizer.register_step_pre_hook(self.observe_gradients)

    def observe_forward(self, module, args, output):
        self._output = output

    def observe_gradients(self, *unused):
        self.gradients.append(
            {
                name: cpu_tensor(parameter.grad)
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            }
        )

    def run_batch(self, batch):
        loss = super().run_batch(batch)
        assert self._output is not None
        expected, terms, denominator = baseline.reference_loss(
            self._output, self.parallel.loss_mesh.get_group()
        )
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
        self.observations.append(
            {
                "input_ids": cpu_tensor(batch["input_ids"]),
                "loss": cpu_tensor(loss),
                "terms": cpu_tensor(terms),
                "denominator": cpu_tensor(denominator),
                "output": {
                    field.name: cpu_tensor(value)
                    for field in fields(self._output)
                    if (value := getattr(self._output, field.name)) is not None
                },
            }
        )
        self._output = None
        return loss

    def build_models(self):
        config = Qwen3_5TextConfig.from_dict(self.fixture["model_config"])
        config._attn_implementation = "flex_attention"
        model = Qwen3_8DSparkModel(config).to(device=self.device, dtype=torch.float32)
        model.load_state_dict(self.fixture["initial_weights"])
        model.set_embedding_head_trainable(False)
        return model, None

    def _build_conversation_collator(self):
        def collate(records):
            ids = torch.tensor([record["input_ids"] for record in records])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "loss_mask": torch.tensor([record["loss_mask"] for record in records]),
            }

        return collate


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return cpu_tensor(value)
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_state(item) for item in value]
    return value


class QwenProducerIsolationTest(unittest.TestCase):
    def test_fixed_producer_serves_two_draft_layouts_and_waits_for_consumers(self):
        self.compare_training(align_phases=False)

    def test_phase_boundaries_preserve_updates_epoch_shuffle_and_stop(self):
        self.compare_training(align_phases=True)

    def compare_training(self, *, align_phases):
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
        if align_phases:
            batches.append(batches[-1])  # Nine records: each epoch keeps eight.
        by_tokens = {tuple(batch["input_ids"][0].tolist()): batch for batch in batches}
        temporary = tempfile.TemporaryDirectory() if runtime.global_rank == 0 else None
        root: list[str | None] = [temporary.name if temporary else None]
        dist.broadcast_object_list(root, src=0)
        assert root[0] is not None
        directory = Path(root[0])
        dataset = directory / "inputs.jsonl"
        if runtime.global_rank == 0:
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
        dist.barrier()
        results = []
        jobs_by_layout = []
        producer = ParallelConfig(tp=2).to_dict()
        runs = (
            ((ParallelConfig(dp_shard=2), 1), (ParallelConfig(dp_shard=2), 3))
            if align_phases
            else ((ParallelConfig(dp_shard=2), 2), (ParallelConfig(dp_replicate=2), 2))
        )
        for run, (draft, partitions) in enumerate(runs):
            destination = directory / f"run{run}"
            jobs: list[tuple] = []
            args = to_config_node(
                {
                    "model": {
                        "target_model_name_or_path": "fixed-fixture",
                        "target_layer_ids": [1, 3],
                        "ce_loss_alpha": 0.1,
                        "l1_loss_alpha": 0.9,
                        "confidence_head_alpha": 1.0,
                        "loss_decay_gamma": 4.0,
                    },
                    "train": {
                        "parallel": draft.to_dict(),
                        "offline_target_parallel": producer,
                        "precision": "fp32",
                        "local_batch_size": 1,
                        "global_batch_size": 4,
                        "num_train_epochs": 2 if align_phases else 1,
                        "max_train_steps": 3 if align_phases else 2,
                        "data_partitions": partitions,
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
                        "jsonl_index_cache_dir": str(directory / "json-index"),
                        "data_batch_cache_dir": str(destination / "cache"),
                    },
                    "logging": {
                        "checkpoint_dir": str(destination / "checkpoints"),
                        "tensorboard_dir": str(destination / "tensorboard"),
                        "save_checkpoints": False,
                        "logging_steps": 1,
                    },
                }
            )

            trainer = ProducerFixtureTrainer(
                runtime.local_rank, args, fixtures[runtime.global_rank]
            )
            self.assertEqual(trainer.target_parallel_config.to_dict(), producer)

            def producer_fixture(job_path, config, devices, draft_trainer=trainer):
                if align_phases:
                    self.assertEqual(draft_trainer.next_micro_step % 2, 0)
                    self.assertTrue(
                        all(
                            parameter.grad is None
                            for parameter in draft_trainer.model.parameters()
                        )
                    )
                job = load_json(job_path)
                observed_inputs = []
                for request in job["requests"]:
                    batch = torch.load(request["input_path"], weights_only=True)
                    tokens = tuple(batch["input_ids"][0].tolist())
                    observed_inputs.append(tokens)
                    self.assertEqual(len(request["output_paths"]), 1)
                    path = Path(request["output_paths"][0])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        dict(by_tokens[tokens], context_chunk_len=torch.tensor([16])),
                        path,
                    )
                jobs.append((config, list(devices), observed_inputs))
                atomic_write_json(
                    str(job_path) + ".complete",
                    {
                        "teacher": job["teacher"],
                        "samples": [{} for _ in job["requests"]],
                    },
                )

            if runtime.global_rank == 1:

                def delayed_consumer(*unused):
                    time.sleep(0.3)
                    self.assertTrue(
                        list((destination / "cache").glob("rank_*/data_batch_*/*.pt"))
                    )

                trainer.optimizer.optimizer.register_step_post_hook(delayed_consumer)
            torch.manual_seed(1000 + runtime.global_rank)
            try:
                with patch(
                    "deepspec.trainer.qwen3_8_vllm_trainer.run_worker_process",
                    producer_fixture,
                ):
                    trainer.train()
                results.append(
                    {
                        "observations": trainer.observations,
                        "gradients_after_clip": trainer.gradients,
                        "model": cpu_state(trainer.model.state_dict()),
                        "optimizer": cpu_state(trainer.optimizer.state_dict()),
                        "next_micro_step": trainer.next_micro_step,
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(runtime.device),
                    }
                )
                self.assertFalse(
                    list((destination / "cache").glob("rank_*/data_batch_*/*.pt"))
                )
                jobs_by_layout.append(jobs)
            finally:
                trainer.train_dataset.close()
                training_logger.close()
            del trainer
            gc.collect()
        baseline.DSparkTrainingBaselineTest().assert_state_close(results[0], results[1])
        if align_phases and runtime.global_rank == 0:
            expected_indices = torch.randperm(
                9, generator=torch.Generator().manual_seed(42)
            ).tolist()[:8]
            expected_indices += torch.randperm(
                9, generator=torch.Generator().manual_seed(43)
            ).tolist()[:4]
            expected = [
                tuple(batches[index]["input_ids"][0].tolist())
                for index in expected_indices
            ]
            for jobs in jobs_by_layout:
                self.assertEqual(
                    [sample for job in jobs for sample in job[2]], expected
                )
                self.assertTrue(all(len(job[2]) % 4 == 0 for job in jobs))
            self.assertEqual([len(job[2]) for job in jobs_by_layout[0]], [8, 4])
            self.assertEqual([len(job[2]) for job in jobs_by_layout[1]], [4, 4, 4])
        elif not align_phases:
            self.assertEqual(jobs_by_layout[0], jobs_by_layout[1])
            if runtime.global_rank == 0:
                self.assertEqual([len(job[2]) for job in jobs_by_layout[0]], [4, 4])
        if not align_phases:
            for name, value in (
                ("chat_template", "changed-template"),
                ("min_loss_tokens", 2),
            ):
                with self.subTest(preprocessing=name):
                    original = args.data.get(name)
                    args.data[name] = value
                    try:
                        with self.assertRaisesRegex(
                            RuntimeError, "producer configuration changed"
                        ):
                            ProducerFixtureTrainer(
                                runtime.local_rank, args, fixtures[runtime.global_rank]
                            )
                    finally:
                        if original is None:
                            del args.data[name]
                        else:
                            args.data[name] = original
                        training_logger.close()
                        gc.collect()
        dist.barrier()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
