"""A BF16 model must resume from the unrounded FP32 optimizer state."""

import copy
import unittest

import torch

from torchtitan.models.dspark_draft.optimizer import MasterWeightAdamW


class NativeOptimizerStateTest(unittest.TestCase):
    def test_bf16_restore_preserves_state_and_next_update_bitwise(self):
        parameter = torch.nn.Parameter(
            torch.tensor([0.132, -0.217], dtype=torch.bfloat16)
        )
        optimizer = MasterWeightAdamW([parameter], lr=0.003, weight_decay=0.01)
        for gradient in ([0.0283, -0.751], [-0.0425, 0.657]):
            parameter.grad = torch.tensor(gradient, dtype=torch.bfloat16)
            optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = MasterWeightAdamW([restored_parameter], lr=0.003, weight_decay=0.01)
        restored.load_state_dict(saved)
        torch.testing.assert_close(
            restored.state_dict(), optimizer.state_dict(), rtol=0, atol=0
        )
        for candidate in (parameter, restored_parameter):
            candidate.grad = torch.tensor([0.172, -0.259], dtype=torch.bfloat16)
        optimizer.step()
        restored.step()
        torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
        torch.testing.assert_close(
            restored.state_dict(), optimizer.state_dict(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
