"""Orthogonal process-group topology shared by cache generation and training."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class ParallelTopology:
    """Orthogonal ``DP x CP x EP x TP x FSDP`` rank layout.

    Global ranks use ``FSDP`` as the fastest-changing coordinate::

        ((((dp * cp_size) + cp) * ep_size + ep) * tp_size + tp)
        * fsdp_size + fsdp

    Ranks with the same ``(dp, fsdp)`` coordinate consume one logical sample.
    CP splits its tokens, EP splits routed experts, and TP splits tensor math.
    Varying the FSDP coordinate selects different samples while sharding the
    same ``(cp, ep, tp)`` parameter partition.
    """

    world_size: int
    global_rank: int
    context_parallel_size: int
    expert_parallel_size: int
    tensor_parallel_size: int
    fsdp_size: int
    data_parallel_size: int
    data_parallel_rank: int
    sample_parallel_size: int
    sample_parallel_rank: int
    context_parallel_rank: int
    expert_parallel_rank: int
    tensor_parallel_rank: int
    fsdp_rank: int
    fsdp_group: object | None
    fsdp_replica_group: object | None
    fsdp_replica_size: int
    context_parallel_group: object | None
    expert_parallel_group: object | None
    tensor_parallel_group: object | None
    model_parallel_group: object | None
    model_parallel_src_rank: int


def compute_context_parallel_range(
    *, sequence_length: int, context_parallel_rank: int, context_parallel_size: int
) -> tuple[int, int]:
    """Return the contiguous token interval owned by one CP rank."""

    sequence_length = int(sequence_length)
    rank = int(context_parallel_rank)
    size = int(context_parallel_size)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    if size < 1 or not 0 <= rank < size:
        raise ValueError(
            "Invalid context-parallel rank/size: "
            f"rank={rank}, size={size}."
        )
    base, remainder = divmod(sequence_length, size)
    start = rank * base + min(rank, remainder)
    length = base + int(rank < remainder)
    if length == 0:
        raise ValueError(
            "A sequence must contain at least one token per CP rank: "
            f"sequence_length={sequence_length}, context_parallel_size={size}."
        )
    return start, start + length


def build_parallel_topology(
    *,
    context_parallel_size: int,
    fsdp_size: int,
    expert_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    create_fsdp_groups: bool = True,
) -> ParallelTopology:
    """Create deterministic, mutually orthogonal distributed process groups.

    Every process calls :func:`dist.new_group` in the same order.  An FSDP
    shard group fixes ``(dp, cp, ep, tp)`` and varies ``fsdp``.  Its optional
    hybrid replica group fixes ``(fsdp, ep, tp)`` and varies ``(dp, cp)`` so
    replicated CP copies and outer data replicas stay synchronized without
    mixing different expert or tensor shards.
    """

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    cp_size = int(context_parallel_size)
    ep_size = int(expert_parallel_size)
    tp_size = int(tensor_parallel_size)
    shard_size = int(fsdp_size)
    sizes = {
        "context_parallel_size": cp_size,
        "expert_parallel_size": ep_size,
        "tensor_parallel_size": tp_size,
        "fsdp_size": shard_size,
    }
    invalid = {name: value for name, value in sizes.items() if value < 1}
    if invalid:
        raise ValueError(f"Parallel sizes must be positive, got {invalid}.")

    model_parallel_size = cp_size * ep_size * tp_size * shard_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "world_size must be divisible by CP * EP * TP * FSDP: "
            f"world_size={world_size}, CP={cp_size}, EP={ep_size}, "
            f"TP={tp_size}, FSDP={shard_size}."
        )
    dp_size = world_size // model_parallel_size

    fsdp_rank = global_rank % shard_size
    coordinate = global_rank // shard_size
    tp_rank = coordinate % tp_size
    coordinate //= tp_size
    ep_rank = coordinate % ep_size
    coordinate //= ep_size
    cp_rank = coordinate % cp_size
    dp_rank = coordinate // cp_size

    def rank_of(dp_idx, cp_idx, ep_idx, tp_idx, fsdp_idx):
        return (
            ((((dp_idx * cp_size) + cp_idx) * ep_size + ep_idx) * tp_size
              + tp_idx)
            * shard_size
            + fsdp_idx
        )

    local_fsdp_group = None
    if create_fsdp_groups:
        for dp_idx in range(dp_size):
            for cp_idx in range(cp_size):
                for ep_idx in range(ep_size):
                    for tp_idx in range(tp_size):
                        ranks = [
                            rank_of(dp_idx, cp_idx, ep_idx, tp_idx, fsdp_idx)
                            for fsdp_idx in range(shard_size)
                        ]
                        group = dist.new_group(ranks=ranks)
                        if global_rank in ranks:
                            local_fsdp_group = group

    # FSDP HYBRID_SHARD replication may cross outer DP replicas and CP ranks,
    # but must never cross EP/TP because those ranks own different parameters.
    local_fsdp_replica_group = None
    fsdp_replica_size = dp_size * cp_size
    if create_fsdp_groups and fsdp_replica_size > 1:
        for fsdp_idx in range(shard_size):
            for ep_idx in range(ep_size):
                for tp_idx in range(tp_size):
                    ranks = [
                        rank_of(dp_idx, cp_idx, ep_idx, tp_idx, fsdp_idx)
                        for dp_idx in range(dp_size)
                        for cp_idx in range(cp_size)
                    ]
                    group = dist.new_group(ranks=ranks)
                    if global_rank in ranks:
                        local_fsdp_replica_group = group

    local_cp_group = None
    if cp_size > 1:
        for dp_idx in range(dp_size):
            for ep_idx in range(ep_size):
                for tp_idx in range(tp_size):
                    for fsdp_idx in range(shard_size):
                        ranks = [
                            rank_of(dp_idx, cp_idx, ep_idx, tp_idx, fsdp_idx)
                            for cp_idx in range(cp_size)
                        ]
                        group = dist.new_group(ranks=ranks)
                        if global_rank in ranks:
                            local_cp_group = group

    local_ep_group = None
    if ep_size > 1:
        for dp_idx in range(dp_size):
            for cp_idx in range(cp_size):
                for tp_idx in range(tp_size):
                    for fsdp_idx in range(shard_size):
                        ranks = [
                            rank_of(dp_idx, cp_idx, ep_idx, tp_idx, fsdp_idx)
                            for ep_idx in range(ep_size)
                        ]
                        group = dist.new_group(ranks=ranks)
                        if global_rank in ranks:
                            local_ep_group = group

    local_tp_group = None
    if tp_size > 1:
        for dp_idx in range(dp_size):
            for cp_idx in range(cp_size):
                for ep_idx in range(ep_size):
                    for fsdp_idx in range(shard_size):
                        ranks = [
                            rank_of(dp_idx, cp_idx, ep_idx, tp_idx, fsdp_idx)
                            for tp_idx in range(tp_size)
                        ]
                        group = dist.new_group(ranks=ranks)
                        if global_rank in ranks:
                            local_tp_group = group

    local_model_group = None
    model_axis_size = cp_size * ep_size * tp_size
    if model_axis_size > 1:
        for dp_idx in range(dp_size):
            for fsdp_idx in range(shard_size):
                ranks = [
                    rank_of(dp_idx, cp_idx, ep_idx, tp_idx, fsdp_idx)
                    for cp_idx in range(cp_size)
                    for ep_idx in range(ep_size)
                    for tp_idx in range(tp_size)
                ]
                group = dist.new_group(ranks=ranks)
                if global_rank in ranks:
                    local_model_group = group

    required = (
        (create_fsdp_groups, local_fsdp_group, "FSDP"),
        (create_fsdp_groups and fsdp_replica_size > 1,
         local_fsdp_replica_group, "FSDP replica"),
        (cp_size > 1, local_cp_group, "context-parallel"),
        (ep_size > 1, local_ep_group, "expert-parallel"),
        (tp_size > 1, local_tp_group, "tensor-parallel"),
        (model_axis_size > 1, local_model_group, "model-parallel"),
    )
    for enabled, group, name in required:
        if enabled and group is None:
            raise RuntimeError(f"Failed to assign the local {name} group.")

    return ParallelTopology(
        world_size=world_size,
        global_rank=global_rank,
        context_parallel_size=cp_size,
        expert_parallel_size=ep_size,
        tensor_parallel_size=tp_size,
        fsdp_size=shard_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        sample_parallel_size=dp_size * shard_size,
        sample_parallel_rank=dp_rank * shard_size + fsdp_rank,
        context_parallel_rank=cp_rank,
        expert_parallel_rank=ep_rank,
        tensor_parallel_rank=tp_rank,
        fsdp_rank=fsdp_rank,
        fsdp_group=local_fsdp_group,
        fsdp_replica_group=local_fsdp_replica_group,
        fsdp_replica_size=fsdp_replica_size,
        context_parallel_group=local_cp_group,
        expert_parallel_group=local_ep_group,
        tensor_parallel_group=local_tp_group,
        model_parallel_group=local_model_group,
        model_parallel_src_rank=rank_of(dp_rank, 0, 0, 0, fsdp_rank),
    )


__all__ = [
    "ParallelTopology",
    "build_parallel_topology",
    "compute_context_parallel_range",
]
