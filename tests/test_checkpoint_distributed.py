"""Opt-in checkpoint regression: reserves almost all free VRAM on two GPUs.

TORCHSPEC_TEST_CHECKPOINT_MEMORY_PRESSURE=1 python -m pytest -q tests/test_checkpoint_distributed.py
"""

import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from torchspec.training import checkpoint
from torchspec.training.optimizer import BF16Optimizer
from torchspec.utils.distributed import init_gloo_group

pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHSPEC_TEST_CHECKPOINT_MEMORY_PRESSURE") != "1"
    or torch.cuda.device_count() < 2,
    reason="Requires opt-in and two dedicated CUDA devices",
)


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _checkpoint_roundtrip(rank, directory):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    cpu_group = init_gloo_group()
    torch.manual_seed(42)
    model = torch.nn.Linear(128, 128).cuda().bfloat16()
    fully_shard(model)
    optimizer = BF16Optimizer(model, lr=1e-4, total_steps=10, warmup_ratio=0)
    model(torch.ones(8, 128, device="cuda", dtype=torch.bfloat16)).float().sum().backward()
    optimizer.step()
    actor = SimpleNamespace(
        args=SimpleNamespace(
            checkpoint_dir=directory,
            load_path=directory,
            ckpt_step=None,
            continual_training=False,
        ),
        model=model,
        optimizer=optimizer,
        lr_scheduler=optimizer.scheduler,
        global_step=1,
    )
    parameters = [_local(p).detach().clone() for p in model.parameters()]
    masters = [_local(p).detach().clone() for p in optimizer.fp32_params]
    adam_state = [
        {key: _local(value).clone() for key, value in optimizer.state[p].items()}
        for p in optimizer.fp32_params
    ]
    scheduler_state = optimizer.scheduler.state_dict()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    allocation = torch.empty(free_bytes - 64 * 2**20, device="cuda", dtype=torch.uint8)
    del allocation
    dist.barrier(group=cpu_group)
    assert torch.cuda.mem_get_info()[0] < 64 * 2**20

    # The original NCCL metadata gather fails here while creating P2P buffers.
    checkpoint.save(actor, step=1)
    with torch.no_grad():
        for tensor in model.parameters():
            tensor.zero_()
    optimizer = BF16Optimizer(model, lr=1e-4, total_steps=10, warmup_ratio=0)
    actor.optimizer = optimizer
    actor.lr_scheduler = optimizer.scheduler
    actor.global_step = 0
    payload = checkpoint.load(actor)
    assert payload is not None
    checkpoint.finalize_load(actor, payload)

    for actual, expected in zip(model.parameters(), parameters, strict=True):
        torch.testing.assert_close(_local(actual), expected, rtol=0, atol=0)
    for actual, expected in zip(optimizer.fp32_params, masters, strict=True):
        torch.testing.assert_close(_local(actual), expected, rtol=0, atol=0)
    for master, expected in zip(optimizer.fp32_params, adam_state, strict=True):
        for key, value in expected.items():
            torch.testing.assert_close(_local(optimizer.state[master][key]), value, rtol=0, atol=0)
    assert optimizer.scheduler.state_dict() == scheduler_state
    assert actor.global_step == 1
    saved = Path(directory) / "iter_0000002"
    assert json.loads((saved / "meta.json").read_text())["global_step"] == 1
    for component in ["model", "optimizer", "lr_scheduler"]:
        assert (saved / component / ".metadata").is_file()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


def test_checkpoint_roundtrip_with_full_cuda_allocator_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "0")
    mp.spawn(_checkpoint_roundtrip, args=(str(tmp_path),), nprocs=2, join=True)
