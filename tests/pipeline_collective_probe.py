"""Explicit native TP/DP collective probe (Gloo CPU or NCCL GPU, no model)."""

import argparse
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from deepspec.pipeline.runtime import atomic_json


def rank_contract(plan, *, rank, local_rank, world, node_id):
    ranks = plan["training_ranks"]
    if world != len(ranks) or not 0 <= rank < world:
        raise ValueError("Probe world identity differs from plan")
    expected = ranks[rank]
    if (expected["global_rank"], expected["local_rank"], expected["node_id"]) != (
        rank,
        local_rank,
        node_id,
    ):
        raise ValueError("Probe rank/node identity differs from plan")
    positions = [s["position"] for s in plan["samples"] if rank in s["reader_ranks"]]
    return {
        **expected,
        "tp_group": [
            r["global_rank"] for r in ranks if r["dp_rank"] == expected["dp_rank"]
        ],
        "dp_group": [
            r["global_rank"] for r in ranks if r["tp_rank"] == expected["tp_rank"]
        ],
        "positions": positions,
        "native_cursor": plan["counts"]["native_cursor"],
        "sample_cursor": plan["counts"]["sample_cursor"],
        "input_plan_hash": plan["input_plan_hash"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--output", type=Path, default=os.environ.get("DEEPSPEC_PROBE_OUTPUT")
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--fail-rank", type=int)
    args = parser.parse_args()
    if args.output is None or not 0 < args.timeout < float("inf"):
        parser.error(
            "A probe output directory and positive finite timeout are required"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    plan = None
    if args.plan:
        from deepspec.pipeline.planning import TopologyPlan
        from deepspec.pipeline.execution import validate_artifacts
        import ray

        plan = TopologyPlan.from_dict(json.loads(args.plan.read_text())).to_dict()
        validate_artifacts(plan)
        ray.init(
            address=plan["config"]["ray_address"],
            namespace=f"collective-{plan['run_id']}",
            log_to_driver=False,
        )
        try:
            node_id = ray.get_runtime_context().get_node_id()
        finally:
            ray.shutdown()
        contract = rank_contract(
            plan, rank=rank, local_rank=local_rank, world=world, node_id=node_id
        )
        timeout = min(args.timeout, plan["timeouts_seconds"]["collective"])
    else:
        if world != 8:
            parser.error("The standalone historical probe requires eight ranks")
        contract = {
            "tp_group": list(range(rank // 4 * 4, rank // 4 * 4 + 4)),
            "dp_group": [rank % 4, rank % 4 + 4],
        }
        timeout = args.timeout
    if args.backend == "gloo":
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") or torch.cuda.is_initialized():
            raise ValueError(
                "CPU probe requires empty CUDA visibility and no initialized CUDA"
            )
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timedelta(seconds=timeout))
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            "nccl", device_id=device, timeout=timedelta(seconds=timeout)
        )
    report = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "backend": args.backend,
        "scope": "planned_nodes" if plan else "standalone_same_or_multiple_hosts",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "network": os.environ.get("NCCL_NET") if args.backend == "nccl" else "gloo",
        "contract": contract,
        "status": "failed",
        "model_initializations": 0,
    }
    if plan:
        report.update(run_id=plan["run_id"], plan_hash=plan["plan_hash"])
    try:
        # Every rank creates all groups in the same global order.
        if plan:
            all_contracts = [
                rank_contract(
                    plan,
                    rank=r["global_rank"],
                    local_rank=r["local_rank"],
                    world=world,
                    node_id=r["node_id"],
                )
                for r in plan["training_ranks"]
            ]
            memberships = sorted(
                {tuple(c[k]) for c in all_contracts for k in ("tp_group", "dp_group")}
            )
        else:
            memberships = sorted(
                {tuple([i, i + 4]) for i in range(4)}
                | {tuple(range(i, i + 4)) for i in (0, 4)}
            )
        groups = {
            members: dist.new_group(list(members), timeout=timedelta(seconds=timeout))
            for members in memberships
        }
        atomic_json(args.output / f"identity-{rank}.json", report)
        dist.barrier()
        for other in range(world):
            identity = json.loads((args.output / f"identity-{other}.json").read_text())
            if identity["rank"] != other or (
                plan and identity.get("plan_hash") != plan["plan_hash"]
            ):
                raise ValueError("Shared output rank identity differs")
        if rank == args.fail_rank:
            raise RuntimeError("Injected owned probe rank failure")
        value = torch.full(
            (1024 if args.backend == "gloo" else 262144,),
            rank + 1,
            dtype=torch.float32,
            device=device,
        )
        global_sum = value.clone()
        dist.all_reduce(global_sum)
        assert bool(global_sum.eq(world * (world + 1) // 2).all())
        dp_members = contract["dp_group"]
        dp_group = groups[tuple(dp_members)]
        gathered = [torch.empty_like(value) for _ in dp_members]
        dist.all_gather(gathered, value, group=dp_group)
        assert all(
            bool(v.eq(member + 1).all()) for member, v in zip(dp_members, gathered)
        )
        reduced = torch.empty_like(value)
        dist.reduce_scatter_tensor(
            reduced, torch.cat([value] * len(dp_members)), group=dp_group
        )
        assert bool(reduced.eq(sum(r + 1 for r in dp_members)).all())
        tp_sum = value.clone()
        dist.all_reduce(tp_sum, group=groups[tuple(contract["tp_group"])])
        assert bool(tp_sum.eq(sum(r + 1 for r in contract["tp_group"])).all())
        dist.barrier()
        report.update(
            status="passed",
            global_all_reduce=True,
            dp_all_gather=True,
            dp_reduce_scatter=True,
            tp_all_reduce=True,
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        atomic_json(args.output / f"rank-{rank}.json", report)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
