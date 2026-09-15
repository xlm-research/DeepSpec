"""Validate this machine's CUDA, NCCL, and compiled attention without model assets."""

import os

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention


rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
try:
    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value)
    assert value.item() == 36.0, value
    matrix = torch.ones(64, 64, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(matrix @ matrix, torch.full_like(matrix, 64))
    print(f"rank={rank} gpu={torch.cuda.get_device_name()} NCCL/BF16 PASS", flush=True)
    if rank == 0:
        torch.manual_seed(42)
        inputs = [
            torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
            for _ in range(3)
        ]
        compiled = torch.compile(flex_attention)
        output = compiled(*inputs)
        reference = torch.nn.functional.scaled_dot_product_attention(*inputs)
        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
        output.float().square().mean().backward()
        assert all(t.grad is not None and t.grad.isfinite().all() for t in inputs)
        torch.cuda.synchronize()
        print("Inductor/Triton FlexAttention forward/backward PASS", flush=True)
    dist.barrier()
finally:
    dist.destroy_process_group()
