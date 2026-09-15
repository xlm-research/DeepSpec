import copy
import os
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4SparseMoeBlock,
)

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.mesh import ParallelContext
from deepspec.modeling.deepseek_v4_parallel import _parallelize_moe
from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.pure_ep import (
    synchronize_module_gradients,
    synchronize_pure_expert_gradients,
)
from deepspec.training.loss import configure_loss_reduction_group


def _check_global_mean(parallel):
    context = ParallelContext.build(parallel, device_type="cpu")
    configure_loss_reduction_group(context.loss_mesh.get_group())
    loss_size = int(context.loss_mesh.size())
    logical_rank = dist.get_rank() // parallel.tp
    config = DeepseekV4Config(
        hidden_size=8,
        moe_intermediate_size=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        mlp_layer_types=["moe"],
    )
    config._experts_implementation = "eager"
    torch.manual_seed(2026)
    reference = DeepseekV4SparseMoeBlock(config, layer_idx=0)
    for parameter in reference.parameters():
        torch.nn.init.uniform_(parameter, a=-0.2, b=0.2)
    # Both EP owners receive tokens, and each owns an unused expert. Router
    # weights still participate in the loss through the two selected scores.
    reference.gate.e_score_correction_bias.copy_(torch.tensor([100., 0., 100., 0.]))
    actual = copy.deepcopy(reference)
    with patch("torch.cuda.is_available", return_value=False), patch.dict(
        os.environ, {"DEEPSPEC_V4_EP_TOKEN_CHUNK": "4"}
    ):
        _parallelize_moe(actual, topology=context)

    # Distinct DP/CP token shards, replicated along TP. Unequal token counts
    # also exercise padding and multiple dispatcher collective rounds.
    counts = [3 + 2 * index for index in range(loss_size)]
    generator = torch.Generator().manual_seed(913)
    global_hidden = torch.randn(1, sum(counts), 8, generator=generator)
    global_targets = torch.arange(sum(counts)).remainder(8)
    start = sum(counts[:logical_rank])
    stop = start + counts[logical_rank]
    hidden = global_hidden[:, start:stop].clone().requires_grad_(True)
    reference_hidden = global_hidden.clone().requires_grad_(True)
    expected = reference(reference_hidden)
    output = actual(hidden)
    torch.testing.assert_close(output, expected[:, start:stop], rtol=2e-5, atol=1e-7)

    targets = global_targets[start:stop].view(1, -1, 1)
    outputs = DSparkForwardOutput(
        draft_logits=output.unsqueeze(-2),
        target_ids=targets,
        eval_mask=torch.ones_like(targets, dtype=torch.bool),
        block_keep_mask=torch.ones_like(targets[..., 0], dtype=torch.bool),
    )
    with patch("deepspec.modeling.dspark.loss.add_metric"):
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=None,
            ce_loss_alpha=1.0,
            l1_loss_alpha=0.0,
            confidence_head_alpha=0.0,
        )
    loss.backward()
    expected_loss = F.cross_entropy(
        expected.reshape(-1, 8), global_targets, reduction="sum"
    ) / (sum(counts) + 1e-6)
    expected_loss.backward()

    # The local input cotangent carries DSpark's dense-average compensation;
    # expert normalization must not divide it or router gradients by EP.
    torch.testing.assert_close(
        hidden.grad / loss_size,
        reference_hidden.grad[:, start:stop],
        rtol=3e-5,
        atol=1e-7,
    )
    dense_modules = [actual.gate, actual.shared_experts]
    synchronize_module_gradients(
        dense_modules,
        process_groups=[(context.loss_mesh.get_group(), loss_size)],
    )
    # Overlapping roots must not normalize one parameter twice.
    synchronize_pure_expert_gradients(
        [actual.experts, actual.experts], sparse_mesh=context.sparse_mesh
    )

    def expected_parameter(name, value):
        if name.startswith("experts."):
            return value.chunk(parallel.ep, dim=0)[context.expert_parallel_rank]
        if name.startswith("shared_experts.") and parallel.tp > 1:
            dim = 1 if "down_proj" in name else 0
            return value.chunk(parallel.tp, dim=dim)[context.tensor_parallel_rank]
        return value

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            expected_parameter(name, reference_parameters[name].grad),
            rtol=3e-5,
            atol=1e-7,
            msg=lambda message, name=name: f"{parallel}: {name}: {message}",
        )
    # SGD exposes a scale error that Adam's first normalized update can hide.
    torch.optim.SGD(actual.parameters(), lr=0.25).step()
    torch.optim.SGD(reference.parameters(), lr=0.25).step()
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter,
            expected_parameter(name, reference_parameters[name]),
            rtol=2e-5,
            atol=1e-7,
        )
    with torch.no_grad():
        torch.testing.assert_close(
            actual(hidden), reference(reference_hidden)[:, start:stop],
            rtol=2e-5, atol=1e-7,
        )
    dist.barrier()


def _worker(rank, rendezvous, world_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=world_size, timeout=timedelta(seconds=90),
    )
    try:
        if world_size == 2:
            configurations = (
                ParallelConfig(dp_shard=2, ep=2),
                ParallelConfig(cp=2, ep=2),
                ParallelConfig(tp=2, ep=2),
            )
        else:
            configurations = (
                ParallelConfig(dp_shard=4, ep=2),
                ParallelConfig(dp_replicate=2, dp_shard=2, ep=2),
                ParallelConfig(dp_shard=2, tp=2, ep=2),
                ParallelConfig(dp_shard=2, cp=2, ep=2),
            )
        for parallel in configurations:
            _check_global_mean(parallel)
    finally:
        configure_loss_reduction_group(None)
        dist.destroy_process_group()


class PureExpertGradientNormalizationTest(unittest.TestCase):
    def _run_distributed(self, world_size):
        with tempfile.TemporaryDirectory(prefix="deepspec_ep_gradient_") as directory:
            mp.start_processes(
                _worker, args=(os.path.join(directory, "rendezvous"), world_size),
                nprocs=world_size, start_method="spawn", join=True,
            )

    def test_two_rank_dp_cp_and_tp_match_global_mean(self):
        self._run_distributed(2)

    def test_four_rank_expert_replicas_match_global_mean(self):
        self._run_distributed(4)


if __name__ == "__main__":
    unittest.main()
