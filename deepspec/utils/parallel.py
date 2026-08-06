"""Small 3-D process topology shared by DSpark cache and training paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class ParallelTopology:
    world_size: int
    global_rank: int
    context_parallel_size: int
    fsdp_size: int
    data_parallel_size: int
    data_parallel_rank: int
    sample_parallel_size: int
    sample_parallel_rank: int
    context_parallel_rank: int
    fsdp_rank: int
    fsdp_group: object
    context_parallel_group: object
    context_parallel_cpu_group: object
    model_parallel_group: object
    model_parallel_src_rank: int


def build_parallel_topology(
    *,
    context_parallel_size: int,
    fsdp_size: int,
    create_fsdp_groups: bool = True,
) -> ParallelTopology:
    """Build ``DP x CP x FSDP`` groups using contiguous FSDP shards.

    Rank layout is ``((dp * cp_size) + cp) * fsdp_size + fsdp``.  Every
    process calls every ``new_group`` below in the same order.
    """

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    cp_size = int(context_parallel_size)
    shard_size = int(fsdp_size)
    if cp_size < 1 or shard_size < 1:
        raise ValueError(
            "context_parallel_size and fsdp_size must both be positive, got "
            f"{cp_size} and {shard_size}."
        )
    model_parallel_size = cp_size * shard_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "world_size must be divisible by context_parallel_size * fsdp_size: "
            f"world_size={world_size}, context_parallel_size={cp_size}, "
            f"fsdp_size={shard_size}."
        )

    dp_size = world_size // model_parallel_size
    fsdp_rank = global_rank % shard_size
    cp_rank = (global_rank // shard_size) % cp_size
    dp_rank = global_rank // model_parallel_size

    local_fsdp_group = None
    if create_fsdp_groups:
        for dp_idx in range(dp_size):
            for cp_idx in range(cp_size):
                ranks = [
                    ((dp_idx * cp_size + cp_idx) * shard_size) + fsdp_idx
                    for fsdp_idx in range(shard_size)
                ]
                group = dist.new_group(ranks=ranks)
                if global_rank in ranks:
                    local_fsdp_group = group

    local_cp_group = None
    local_cp_cpu_group = None
    for dp_idx in range(dp_size):
        for fsdp_idx in range(shard_size):
            ranks = [
                ((dp_idx * cp_size + cp_idx) * shard_size) + fsdp_idx
                for cp_idx in range(cp_size)
            ]
            group = dist.new_group(ranks=ranks)
            cpu_group = dist.new_group(ranks=ranks, backend="gloo")
            if global_rank in ranks:
                local_cp_group = group
                local_cp_cpu_group = cpu_group

    assert local_cp_group is not None
    assert local_cp_cpu_group is not None
    if create_fsdp_groups:
        assert local_fsdp_group is not None

    return ParallelTopology(
        world_size=world_size,
        global_rank=global_rank,
        context_parallel_size=cp_size,
        fsdp_size=shard_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        sample_parallel_size=dp_size * shard_size,
        sample_parallel_rank=dp_rank * shard_size + fsdp_rank,
        context_parallel_rank=cp_rank,
        fsdp_rank=fsdp_rank,
        fsdp_group=local_fsdp_group,
        context_parallel_group=local_cp_group,
        context_parallel_cpu_group=local_cp_cpu_group,
        model_parallel_group=local_cp_group,
        model_parallel_src_rank=(
            (dp_rank * cp_size) * shard_size + fsdp_rank
        ),
    )


__all__ = ["ParallelTopology", "build_parallel_topology"]
