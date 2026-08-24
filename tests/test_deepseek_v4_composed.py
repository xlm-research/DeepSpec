import copy
import unittest

import torch
import torch.distributed as dist
from transformers import AutoConfig

from deepspec.distributed import ParallelConfig, ParallelContext, apply_parallelism
from deepspec.modeling.dflash2.deepseek_v4 import (
    DeepseekV4DFlash2Model,
    build_draft_config,
)
from deepspec.modeling.pure_ep import (
    get_pure_expert_modules,
    synchronize_pure_expert_gradients,
)
from deepspec.training import MasterWeightAdamW
from deepspec.utils import load_config
from tests.distributed_test_utils import require_torchrun


TARGET = "/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731"


class DeepseekV4ComposedParallelTest(unittest.TestCase):
    def test_fsdp2_dp_tp_cp_ep_forward_backward(self):
        runtime = require_torchrun(self, world_size=8)
        target = AutoConfig.from_pretrained(TARGET)
        target.hidden_size = 64
        target.num_attention_heads = 4
        target.num_key_value_heads = 1
        target.head_dim = 16
        target.q_lora_rank = 32
        target.o_groups = 4
        target.o_lora_rank = 16
        target.hc_mult = 2
        target.moe_intermediate_size = 32
        target.n_routed_experts = 8
        target.n_shared_experts = 1
        target.num_experts_per_tok = 2
        target.vocab_size = 128
        target.expert_dtype = "bf16"
        target.num_hidden_layers = 3
        args = load_config("config/dflash2/dflash2_deepseek_v4.py").model
        args.mask_token_id = 127
        args.num_anchors = 4
        args.target_layer_ids = [0, 1, 2]
        model = DeepseekV4DFlash2Model(
            build_draft_config(copy.deepcopy(target), args)
        ).to(runtime.device, dtype=torch.float32)

        parallel = ParallelConfig(
            dp_shard=2,
            cp=2,
            tp=2,
            ep=1,
            context_parallel_backend="model_native",
        )
        context = ParallelContext.build(parallel, device_type=runtime.device.type)
        model = apply_parallelism(
            model,
            context,
            parallel,
            param_dtype=torch.float32,
            sequence_length=32,
        )
        torch.manual_seed(2026)
        seq = 32
        local = seq // parallel.cp
        experts = get_pure_expert_modules(model)
        self.assertFalse(experts)
        optimizer = MasterWeightAdamW(model.parameters(), lr=1e-4)
        # A single iteration cannot expose stale collective/autograd state.
        # Exercise a real optimizer boundary and different routing twice.
        for step in range(2):
            torch.manual_seed(2026 + step + dist.get_rank() * 17)
            output = model(
                input_ids=torch.randint(0, 127, (1, seq), device=runtime.device),
                target_hidden_states=torch.randn(1, local, 192, device=runtime.device),
                loss_mask=torch.ones(1, seq, dtype=torch.bool, device=runtime.device),
                target_last_hidden_states=None,
                context_start=torch.tensor(
                    [context.context_parallel_rank * local], device=runtime.device
                ),
                context_len=torch.tensor([local], device=runtime.device),
                seq_len=torch.tensor([seq], device=runtime.device),
            )
            loss = output.draft_logits.float().square().mean()
            loss = loss + output.selector_scores.float().square().mean()
            loss.backward()
            synchronize_pure_expert_gradients(
                experts, sparse_mesh=context.sparse_mesh
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        dist.barrier()


if __name__ == "__main__":
    unittest.main()
