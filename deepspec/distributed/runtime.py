from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedRuntime:
    device: torch.device
    global_rank: int
    local_rank: int
    world_size: int
    owns_process_group: bool


def initialize_runtime(
    local_rank: int | None = None,
    *,
    timeout_minutes: int = 60,
) -> DistributedRuntime:
    """Initialize once, supporting torchrun and the legacy local spawner."""

    if dist.is_initialized():
        resolved_local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                torch.cuda.current_device() if torch.cuda.is_available() else 0,
            )
        )
        device = (
            torch.device("cuda", resolved_local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return DistributedRuntime(
            device=device,
            global_rank=dist.get_rank(),
            local_rank=resolved_local_rank,
            world_size=dist.get_world_size(),
            owns_process_group=False,
        )

    torchrun_launch = "LOCAL_RANK" in os.environ
    if torchrun_launch:
        resolved_local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        init_method = "env://"
    else:
        if local_rank is None:
            local_rank = 0
        resolved_local_rank = int(local_rank)
        local_world_size = max(torch.cuda.device_count(), 1)
        node_rank = int(os.environ.get("RANK", "0"))
        node_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = node_rank * local_world_size + resolved_local_rank
        world_size = node_world_size * local_world_size
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{master_addr}:{master_port}"

    if torch.cuda.is_available():
        torch.cuda.set_device(resolved_local_rank)
        device = torch.device("cuda", resolved_local_rank)
        backend = "nccl"
        device_id = device
    else:
        device = torch.device("cpu")
        backend = "gloo"
        device_id = None
    kwargs = dict(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )
    if device_id is not None:
        kwargs["device_id"] = device_id
    dist.init_process_group(**kwargs)
    return DistributedRuntime(
        device=device,
        global_rank=rank,
        local_rank=resolved_local_rank,
        world_size=world_size,
        owns_process_group=True,
    )


def destroy_runtime() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
