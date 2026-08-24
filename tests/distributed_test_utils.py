from __future__ import annotations

import atexit
import os

import torch
import torch.distributed as dist

from deepspec.distributed.runtime import initialize_runtime


_cleanup_registered = False


def require_torchrun(test_case, *, world_size: int = 2):
    global _cleanup_registered
    if "LOCAL_RANK" not in os.environ:
        test_case.skipTest("run with torchrun")
    runtime = initialize_runtime()
    if not _cleanup_registered:
        def cleanup():
            if dist.is_initialized():
                dist.destroy_process_group()
        atexit.register(cleanup)
        _cleanup_registered = True
    if runtime.world_size != world_size:
        test_case.skipTest(f"requires world_size={world_size}")
    return runtime


def assert_all_ranks_close(test_case, actual, expected, **kwargs):
    torch.testing.assert_close(actual, expected, **kwargs)
    flag = torch.ones((), device=actual.device)
    dist.all_reduce(flag)
    test_case.assertEqual(int(flag.item()), dist.get_world_size())
