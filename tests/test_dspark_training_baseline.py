"""Real Qwen updates through BaseTrainer.train, without a live target.

Run with the existing training interpreter under torchrun --nproc-per-node=2.
DEEPSPEC_BASELINE_OUTPUT optionally archives inputs and observed training state.
"""

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.distributed import ParallelConfig, ParallelContext, apply_parallelism
from deepspec.distributed.context_parallel import FixedContextParallel
from deepspec.modeling.dspark.qwen3_8 import Qwen3_8DSparkModel
from deepspec.trainer.dspark_trainer import Qwen3_8DSparkTrainer
from deepspec.training import BF16Optimizer
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.utils.config import to_config_node
from deepspec.utils.hfai_suspend import SuspendController
from deepspec.utils.metrics import configure_reduction_group
from deepspec.utils import training_logger
from tests.distributed_test_utils import require_torchrun


def cpu_tensor(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.detach().cpu().clone()


def fixed_features(rank, dtype):
    generator = torch.Generator().manual_seed(901 + rank)
    batches = []
    for mask_positions in (range(16), [3, 4, 5, 7, 8, 15], [], [14, 15]):
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[:, list(mask_positions)] = True
        batches.append(
            {
                "input_ids": torch.randint(0, 127, (1, 16), generator=generator),
                "loss_mask": mask,
                "target_hidden_states": torch.randn(1, 16, 128, generator=generator).to(
                    dtype
                ),
                "target_last_hidden_states": torch.randn(
                    1, 16, 64, generator=generator
                ).to(dtype),
                "seq_len": torch.tensor([16]),
            }
        )
    return batches


def reference_loss(output, group):
    """Independent statement of ADR-0001, including detached BCE targets."""
    logits = output.draft_logits
    weights = output.eval_mask.float() * torch.exp(
        -torch.arange(3, device=logits.device).float() / 4.0
    )
    denominator = weights.sum().detach().clone()
    dist.all_reduce(denominator, group=group)
    probabilities = logits.float().softmax(-1)
    teacher = output.aligned_target_logits.float().softmax(-1)
    distance = (probabilities - teacher).abs().sum(-1)
    ce = F.cross_entropy(
        logits.flatten(0, 2), output.target_ids.flatten(), reduction="none"
    ).reshape_as(weights)
    confidence = F.binary_cross_entropy_with_logits(
        output.confidence_pred.float(),
        (1.0 - distance.detach() / 2.0).clamp(0.0, 1.0),
        reduction="none",
    )
    terms = torch.stack([(term * weights).sum() for term in (ce, distance, confidence)])
    means = terms / (denominator + 1e-6)
    loss = (means * means.new_tensor([0.1, 0.9, 1.0])).sum()
    return loss * dist.get_world_size(group), means.detach(), denominator


class ObservedOptimizer(BF16Optimizer):
    """Observe the public optimizer boundary without replacing its update."""

    def __init__(self, model, parameter_names=None):
        super().__init__(model, lr=1e-3, total_steps=4, warmup_ratio=0.25)
        self.model = model
        # Public parameter paths from the original model remain addressable
        # through checkpoint wrappers; observations use the model's own names.
        self.parameters = {
            name: model.get_parameter(name)
            for name in (
                parameter_names
                if parameter_names is not None
                else dict(model.named_parameters())
            )
        }
        self.updates = []

    def step(self):
        gradients = {
            name: cpu_tensor(parameter.grad)
            for name, parameter in self.parameters.items()
            if parameter.requires_grad and parameter.grad is not None
        }
        super().step()
        self.updates.append(
            {
                "gradients_after_clip": gradients,
                "parameters": {
                    name: cpu_tensor(p) for name, p in self.parameters.items()
                },
                "adam": {
                    name: {
                        key: cpu_tensor(value)
                        for key, value in self.optimizer.state[p].items()
                    }
                    for name, p in self.parameters.items()
                    if p.requires_grad
                },
                "scheduler": self.scheduler.state_dict(),
            }
        )


class FixedFeatureTrainer(Qwen3_8DSparkTrainer):
    """Supply ready features to the retained production training loop."""

    def __init__(
        self,
        runtime,
        topology,
        dtype,
        *,
        independent_loss=False,
        fixture=None,
        model_config=None,
        model_factory=Qwen3_8DSparkModel,
    ):
        self.device = runtime.device
        self.global_rank = runtime.global_rank
        self.world_size = runtime.world_size
        self.parallel = topology
        self.parallel_config = topology.config
        self.fixed_context_parallel = FixedContextParallel(
            topology, backend=topology.config.context_parallel_backend
        )
        self.args = to_config_node(
            {
                "model": {
                    "ce_loss_alpha": 0.1,
                    "l1_loss_alpha": 0.9,
                    "confidence_head_alpha": 1.0,
                    "loss_decay_gamma": 4.0,
                },
                "train": {"local_batch_size": 1, "max_grad_norm": 0.5},
                "logging": {"save_checkpoints": False},
            }
        )
        config = model_config or Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            layer_types=["full_attention"] * 2,
        )
        config.target_layer_ids = [1, 3]
        config.num_target_layers = 4
        config.block_size = 3
        config.mask_token_id = 127
        config.num_anchors = 2
        config.enable_confidence_head = True
        config.markov_rank = 8
        config.markov_head_type = "vanilla"
        config.confidence_head_with_markov = True
        config._attn_implementation = "flex_attention"
        torch.manual_seed(20260914)
        draft = model_factory(config).to(device=self.device, dtype=dtype)
        draft.set_embedding_head_trainable(False)
        if fixture is not None:
            draft.load_state_dict(fixture["initial_weights"])
        self.initial_weights = {
            name: cpu_tensor(p) for name, p in draft.named_parameters()
        }
        self.draft_model = draft
        draft.configure_context_parallel(
            size=topology.config.cp,
            rank=topology.context_parallel_rank,
            group=topology.cp_mesh.get_group(),
            model_parallel_group=topology.model_mesh.get_group(),
            model_parallel_src_rank=topology.model_parallel_src_rank,
        )
        self.model = apply_parallelism(
            draft, topology, topology.config, param_dtype=dtype
        )
        self.optimizer = ObservedOptimizer(self.model, self.initial_weights)
        self.suspend_controller = SuspendController(self.device)
        self._pure_expert_modules = []
        self._draft_residency_verified = False
        self.partitioned_model_swap_enabled = False
        self.online_target_enabled = self.offline_target_data_batches_enabled = False
        self.data_batch_micro_batches = None
        self._data_batch_end_after_current = False
        self.next_micro_step = 0
        self.gradient_accumulation_steps = 2
        self.max_train_steps = 2
        self.micro_batches_per_epoch = 4
        self.features = (
            fixed_features(topology.data_parallel_rank, dtype)
            if fixture is None
            else fixture["features"]
        )
        self.observations = []
        self.independent_loss = independent_loss
        self._output = None
        self.model.register_forward_hook(self.observe_forward)

    def observe_forward(self, module, args, output):
        self._output = output

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        end = start_offset_samples + num_samples
        return [
            {key: value.clone() for key, value in batch.items()}
            for batch in self.features[start_offset_samples:end]
        ]

    def run_batch(self, batch):
        actual = super().run_batch(batch)
        output = self._output
        assert output is not None
        expected, terms, denominator = reference_loss(
            output, self.parallel.loss_mesh.get_group()
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.observations.append(
            {
                "loss": cpu_tensor(actual),
                "terms": cpu_tensor(terms),
                "denominator": cpu_tensor(denominator),
                "output": {
                    field.name: cpu_tensor(value)
                    for field in fields(output)
                    if (value := getattr(output, field.name)) is not None
                },
            }
        )
        self._output = None
        return expected if self.independent_loss else actual

    def result(self):
        return {
            "microbatches": self.observations,
            "updates": self.optimizer.updates,
            "next_micro_step": self.next_micro_step,
            "cuda_rng": torch.cuda.get_rng_state(self.device),
            "cpu_rng": torch.get_rng_state(),
        }

    def train_and_observe(self):
        with tempfile.TemporaryDirectory(prefix="dspark-baseline-") as directory:
            training_logger.init(logging_steps=1, tensorboard_dir=directory)
            try:
                self.train()
            finally:
                training_logger.close()
            metrics: list[dict[str, list[float]] | None] = [None]
            if self.global_rank == 0:
                events = EventAccumulator(directory).Reload()
                metrics[0] = {
                    name: [row.value for row in events.Scalars(f"train/{name}")]
                    for name in ("grad_norm", "ce_loss", "l1_loss", "confidence_loss")
                }
            dist.broadcast_object_list(metrics, src=0)
        observed = metrics[0]
        assert observed is not None
        # The public logger reports token-weighted component means over an
        # optimizer window. Its reporting denominator differs from the
        # microbatch means used for backward, so reconstruct that separately.
        for step in range(2):
            window = self.observations[step * 2 : step * 2 + 2]
            denominator = sum(row["denominator"] for row in window)
            numerators = sum(
                row["terms"] * (row["denominator"] + 1e-6) for row in window
            ).to(self.device)
            dist.all_reduce(numerators, group=self.parallel.loss_mesh.get_group())
            expected = numerators.cpu() / denominator
            for index, name in enumerate(("ce_loss", "l1_loss", "confidence_loss")):
                torch.testing.assert_close(
                    torch.tensor(observed[name][step]),
                    expected[index],
                    rtol=1e-5,
                    atol=1e-6,
                )
        return dict(self.result(), metrics=observed)


class DSparkTrainingBaselineTest(unittest.TestCase):
    def assert_state_close(self, actual, expected):
        if isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_state_close(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for observed, reference in zip(actual, expected):
                self.assert_state_close(observed, reference)
        elif isinstance(actual, (torch.Tensor, float)):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        else:
            self.assertEqual(actual, expected)

    def test_real_qwen_two_updates_match_microbatch_objective(self):
        runtime = require_torchrun(self, world_size=2)
        output = os.environ.get("DEEPSPEC_BASELINE_OUTPUT")
        if output:
            error: list[str | None] = [None]
            if runtime.global_rank == 0:
                try:
                    Path(output).mkdir(parents=True, exist_ok=False)
                except OSError as exc:
                    error[0] = str(exc)
            dist.broadcast_object_list(error, src=0)
            if error[0]:
                self.fail(f"Use a new baseline output directory: {error[0]}")
        topology = ParallelContext.build(
            ParallelConfig(dp_shard=2, reduce_dtype="fp32")
        )
        configure_loss_reduction_group(topology.loss_mesh.get_group())
        configure_reduction_group(topology.loss_mesh.get_group())
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                filename = f"{dtype}_rank{runtime.global_rank}.pt"
                reference_directory = os.environ.get("DEEPSPEC_BASELINE_REFERENCE")
                fixture = (
                    torch.load(Path(reference_directory) / filename, weights_only=True)
                    if reference_directory
                    else None
                )
                trainer = FixedFeatureTrainer(runtime, topology, dtype, fixture=fixture)
                torch.manual_seed(1000 + runtime.global_rank)
                initial_cpu_rng = (
                    fixture["initial_cpu_rng"] if fixture else torch.get_rng_state()
                )
                initial_rng = (
                    fixture["initial_cuda_rng"]
                    if fixture
                    else torch.cuda.get_rng_state(runtime.device)
                )
                torch.set_rng_state(initial_cpu_rng)
                torch.cuda.set_rng_state(initial_rng, runtime.device)
                actual = trainer.train_and_observe()
                if fixture is not None:
                    # An immutable pre-change run catches regressions shared by
                    # both live paths: GAS scaling, clipping and optimizer logic.
                    self.assert_state_close(actual, fixture["result"])
                reference = FixedFeatureTrainer(
                    runtime, topology, dtype, independent_loss=True, fixture=fixture
                )
                torch.set_rng_state(initial_cpu_rng)
                torch.cuda.set_rng_state(initial_rng, runtime.device)
                expected = reference.train_and_observe()
                # Both objectives execute identical kernels except the final
                # scalar association: fixed tolerances precede the comparison.
                self.assert_state_close(actual, expected)
                self.assertEqual(actual["next_micro_step"], 4)
                self.assertEqual(len(actual["updates"]), 2)
                denominators = [
                    row["denominator"].item() for row in actual["microbatches"]
                ]
                self.assertNotEqual(denominators[0], denominators[1])
                self.assertGreater(denominators[0], 0)
                self.assertGreater(denominators[1], 0)
                self.assertEqual(denominators[2], 0)
                trainable = {
                    name
                    for name, p in trainer.model.named_parameters()
                    if p.requires_grad
                }
                for update in actual["updates"]:
                    self.assertEqual(set(update["gradients_after_clip"]), trainable)
                    for name in ("lm_head.weight", "embed_tokens.weight"):
                        torch.testing.assert_close(
                            update["parameters"][name],
                            trainer.initial_weights[name],
                            rtol=0,
                            atol=0,
                        )
                    for state in update["adam"].values():
                        self.assertTrue(
                            all(
                                value.dtype == torch.float32 for value in state.values()
                            )
                        )
                if output:
                    directory = Path(output)
                    artifact = directory / filename
                    torch.save(
                        {
                            "initial_weights": trainer.initial_weights,
                            "features": trainer.features,
                            "initial_cuda_rng": initial_rng,
                            "initial_cpu_rng": initial_cpu_rng,
                            "result": actual,
                            "model_config": trainer.draft_model.config.to_dict(),
                            "parallel_config": topology.config.to_dict(),
                        },
                        artifact,
                    )
                    artifact.with_suffix(".json").write_text(
                        json.dumps(
                            {
                                "rank": runtime.global_rank,
                                "world_size": runtime.world_size,
                                "dtype": str(dtype),
                                "torch": torch.__version__,
                                "cuda": torch.version.cuda,
                                "gas": 2,
                                "updates": 2,
                                "rtol": 1e-4,
                                "atol": 1e-6,
                                "denominators": denominators,
                                "test_sha256": hashlib.sha256(
                                    Path(__file__).read_bytes()
                                ).hexdigest(),
                            },
                            indent=2,
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    unittest.main()
