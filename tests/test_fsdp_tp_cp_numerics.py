import copy
import unittest

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.context_parallel import FixedContextParallel
from deepspec.distributed.fsdp import apply_fsdp2
from deepspec.distributed.mesh import ParallelContext
from deepspec.distributed.tensor_parallel import apply_tensor_parallelism
from deepspec.training.optimizer import BF16Optimizer
from tests.distributed_test_utils import require_torchrun


class _Attention(nn.Module):
    def __init__(self, hidden=256, heads=4):
        super().__init__()
        self.num_attention_heads = heads
        self.num_key_value_heads = heads
        self.num_key_value_groups = 1
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape
        def heads(tensor):
            return tensor.view(
                batch, seq, self.num_attention_heads, self.head_dim
            ).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            heads(self.q_proj(x)),
            heads(self.k_proj(x)),
            heads(self.v_proj(x)),
            is_causal=True,
        )
        output = output.transpose(1, 2).reshape(
            batch,
            seq,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(output)


class _MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.up_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])

    def forward(self, x):
        return self.layers[0](x)


class FSDP2TensorContextParallelNumericsTest(unittest.TestCase):
    def test_loss_gradients_and_update_match(self):
        runtime = require_torchrun(self, world_size=4)
        torch.manual_seed(707)
        baseline = _Model().to(runtime.device)
        composed = copy.deepcopy(baseline)
        config = ParallelConfig(dp_shard=1, tp=2, cp=2)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        apply_tensor_parallelism(composed, context, config)
        apply_fsdp2(composed, context, config, param_dtype=torch.float32)
        cp_backend = FixedContextParallel(context, backend="pytorch")
        baseline_optimizer = BF16Optimizer(baseline, 1e-3, 2, 0, 0)
        composed_optimizer = BF16Optimizer(composed, 1e-3, 2, 0, 0)
        x = torch.randn(1, 128, 256, device=runtime.device)
        expected = baseline(x)
        expected_loss = expected.float().square().mean()
        expected_loss.backward()

        cp_input = x.clone()
        with cp_backend.forward_context(buffers=[cp_input], sequence_dims=[1]):
            actual = composed(cp_input)
            local_numerator = actual.float().square().sum()
            reported_loss = local_numerator.detach().clone()
            dist.all_reduce(reported_loss, group=context.cp_mesh.get_group())
            reported_loss /= expected.numel()
            backward_loss = (
                local_numerator / expected.numel() * config.cp
            )
            backward_loss.backward()
        torch.testing.assert_close(reported_loss, expected_loss, rtol=3e-4, atol=3e-4)
        baseline_optimizer.step()
        composed_optimizer.step()
        # A full-sequence call outside the CP context explicitly unshards the
        # sequence and verifies the composed parameter update.
        torch.testing.assert_close(
            composed(x), baseline(x), rtol=5e-4, atol=5e-4
        )


if __name__ == "__main__":
    unittest.main()
