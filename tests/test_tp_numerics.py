import copy
import unittest

import torch
from torch import nn

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.mesh import ParallelContext
from deepspec.distributed.tensor_parallel import apply_tensor_parallelism
from deepspec.distributed.tensor_parallel import build_transformer_tp_plan
from deepspec.training.optimizer import BF16Optimizer
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from tests.distributed_test_utils import require_torchrun


class _Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.num_key_value_groups = 1
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x) + self.k_proj(x) + self.v_proj(x)))


class _MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.up_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.mlp = _MLP(hidden)

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class _Model(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.layers = nn.ModuleList([_Block(hidden), _Block(hidden)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TensorParallelNumericsTest(unittest.TestCase):
    def test_real_qwen_draft_fqns_are_discovered(self):
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["full_attention"],
        )
        config.target_layer_ids = [1, 3]
        config.block_size = 3
        config.mask_token_id = 127
        config.num_anchors = 4
        config.enable_confidence_head = False
        config.markov_rank = 0
        model = Qwen3DSparkModel(config)
        plan = build_transformer_tp_plan(model)
        self.assertIn("layers.0.self_attn.q_proj", plan.styles)
        self.assertIn("layers.0.self_attn.o_proj", plan.styles)
        self.assertIn("layers.0.mlp.gate_proj", plan.styles)
        self.assertIn("layers.0.mlp.down_proj", plan.styles)

    def test_forward_backward_and_update_match(self):
        runtime = require_torchrun(self, world_size=2)
        torch.manual_seed(1234)
        baseline = _Model().to(runtime.device)
        parallel_model = copy.deepcopy(baseline)
        config = ParallelConfig(tp=2, use_fsdp=False)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        apply_tensor_parallelism(parallel_model, context, config)
        baseline_optimizer = BF16Optimizer(
            baseline, lr=1e-3, total_steps=2, warmup_ratio=0, weight_decay=0
        )
        parallel_optimizer = BF16Optimizer(
            parallel_model, lr=1e-3, total_steps=2, warmup_ratio=0, weight_decay=0
        )
        x = torch.randn(2, 5, 8, device=runtime.device)
        baseline_output = baseline(x)
        parallel_output = parallel_model(x)
        torch.testing.assert_close(parallel_output, baseline_output, rtol=1e-5, atol=1e-5)
        baseline_output.square().mean().backward()
        parallel_output.square().mean().backward()
        baseline_optimizer.step()
        parallel_optimizer.step()
        torch.testing.assert_close(
            parallel_model(x), baseline(x), rtol=2e-5, atol=2e-5
        )


if __name__ == "__main__":
    unittest.main()
