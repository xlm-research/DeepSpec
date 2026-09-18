"""Eight-GPU TCP smoke check for the two-node DP2/TP4 process groups."""

import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 8
        dp_groups = [dist.new_group([i, i + 4]) for i in range(4)]
        tp_groups = [dist.new_group(list(range(i, i + 4))) for i in (0, 4)]
        value = torch.full((262144,), rank + 1, dtype=torch.float32, device=device)
        global_sum = value.clone()
        dist.all_reduce(global_sum)
        assert bool(global_sum.eq(36).all())
        dp_group = dp_groups[rank % 4]
        gathered = [torch.empty_like(value) for _ in range(2)]
        dist.all_gather(gathered, value, group=dp_group)
        assert bool(gathered[0].eq(rank % 4 + 1).all())
        assert bool(gathered[1].eq(rank % 4 + 5).all())
        reduced = torch.empty_like(value)
        dist.reduce_scatter_tensor(reduced, torch.cat([value, value]), group=dp_group)
        assert bool(reduced.eq(2 * (rank % 4) + 6).all())
        tp_sum = value.clone()
        dist.all_reduce(tp_sum, group=tp_groups[rank // 4])
        assert bool(tp_sum.eq(10 if rank < 4 else 26).all())
        dist.barrier()
        Path(os.environ["DEEPSPEC_PROBE_OUTPUT"], f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "hostname": socket.gethostname(),
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "global_all_reduce": True,
                    "dp_all_gather": True,
                    "dp_reduce_scatter": True,
                    "tp_all_reduce": True,
                    "network": os.environ["NCCL_NET"],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
