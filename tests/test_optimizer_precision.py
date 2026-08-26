import copy
import unittest

import torch

from deepspec.distributed.fsdp import fsdp_reduce_dtype
from deepspec.training.optimizer import MasterWeightAdamW


class OptimizerPrecisionTest(unittest.TestCase):
    def test_bf16_training_uses_bf16_fsdp_reduction(self):
        self.assertIs(fsdp_reduce_dtype(torch.bfloat16), torch.bfloat16)
        self.assertIs(fsdp_reduce_dtype(torch.float16), torch.bfloat16)
        self.assertIs(fsdp_reduce_dtype(torch.float32), torch.float32)

    def test_bf16_parameter_keeps_fp32_optimizer_state_and_accumulation(self):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        )
        optimizer = MasterWeightAdamW([parameter], lr=1e-3)
        parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
        optimizer.step()

        state = optimizer.state[parameter]
        for name in ("step", "master_param", "exp_avg", "exp_avg_sq"):
            self.assertEqual(state[name].dtype, torch.float32)
        self.assertEqual(optimizer.param_groups[0]["step"], 1)
        self.assertEqual(int(state["step"].item()), 1)

    def test_loading_bf16_state_restores_fp32_optimizer_state(self):
        source_parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        )
        source = MasterWeightAdamW([source_parameter], lr=1e-3)
        state_dict = copy.deepcopy(source.state_dict())
        for state in state_dict["state"].values():
            for name in ("step", "master_param", "exp_avg", "exp_avg_sq"):
                state[name] = state[name].to(torch.bfloat16)

        destination_parameter = torch.nn.Parameter(
            torch.tensor([3.0, -4.0], dtype=torch.bfloat16)
        )
        destination = MasterWeightAdamW([destination_parameter], lr=1e-3)
        destination.load_state_dict(state_dict)

        state = destination.state[destination_parameter]
        for name in ("step", "master_param", "exp_avg", "exp_avg_sq"):
            self.assertEqual(state[name].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
