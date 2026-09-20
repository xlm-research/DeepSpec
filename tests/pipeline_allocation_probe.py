"""Explicit real Ray placement probe. No model or CUDA initialization occurs."""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from deepspec.pipeline.runtime import Deadline, atomic_json


class PlacementShell:
    def __init__(self, plan):
        self.plan = plan

    def allocation(self, participant):
        import ray
        from ray._private.worker import get_resource_ids
        from deepspec.pipeline.cluster import gpu_inventory
        from deepspec.pipeline.runtime import actor_identity
        from deepspec.pipeline.vllm_adapter import observed_bundle

        identity = actor_identity(self.plan)
        group = ray.util.get_current_placement_group()
        inventory = gpu_inventory(timeout=10)
        by_id = {str(g["index"]): g["uuid"] for g in inventory}
        by_id.update({g["uuid"]: g["uuid"] for g in inventory})
        devices = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            raise RuntimeError("Allocation probe initialized CUDA")
        identity.update(
            participant=participant,
            gpu_uuids=[by_id[str(d)] for d in devices],
            pg_id=None if group is None else group.id.hex(),
            bundle_index=None
            if group is None
            else observed_bundle(get_resource_ids(), group.id.hex()),
            pid=identity["process"]["pid"],
            start_ticks=identity["process"]["start_ticks"],
            model_initializations=0,
        )
        return identity


def execute(plan, output, *, inject_partial=False):
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from deepspec.pipeline.controller import (
        ActorAllocator,
        AllocationGate,
        NodeAgents,
        PlacementAllocator,
        RayPlacementBackend,
        ResourceRegistry,
    )

    output = Path(output)
    registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
    probe_id = uuid.uuid4().hex
    actors = ActorAllocator(
        plan,
        registry=registry,
        name_prefix=probe_id,
        namespace=f"allocation-{probe_id}",
        allocation_path=output / "allocation.json",
    )
    agents = NodeAgents(plan, actors, report_dir=str(output))
    allocator = PlacementAllocator(
        plan, registry=registry, backend=RayPlacementBackend()
    )
    deadline = Deadline.after(plan["timeouts_seconds"]["allocation"])
    result = {
        "run_id": plan["run_id"],
        "plan_hash": plan["plan_hash"],
        "evidence_level": "ray_allocation",
        "probe_id": probe_id,
        "model_initializations": 0,
        "allocations": [],
        "injected_partial": inject_partial,
    }
    injected = False
    try:
        ray.init(
            address=plan["config"]["ray_address"],
            namespace=actors.namespace,
            log_to_driver=False,
        )
        agents.start(deadline=deadline)
        groups = allocator.allocate(
            deadline=deadline, cleanup_timeout=plan["timeouts_seconds"]["cleanup"]
        )
        actors._persist()
        gate = AllocationGate(plan, pg_ids={n: h.id.hex() for n, h in groups.items()})
        specs = [
            (
                p,
                e["node_id"],
                groups[next(n for n, h in groups.items() if h.id.hex() == e["pg_id"])],
                e["bundle_index"],
                e["gpu_count"],
                e["gpu_count"] * 2 if e["role"] == "training" else 0,
            )
            for p, e in gate.expected.items()
        ]
        specs += [
            (
                f"core/{r['replica_id']}",
                r["core_node"],
                groups[f"inference-{r['replica_id']}"],
                r["core_cpu_bundle"],
                0,
                1,
            )
            for r in plan["replicas"]
        ]
        specs.append(("frontend", plan["replicas"][0]["core_node"], None, None, 0, 1))
        for index, (participant, node, pg, bundle, gpus, cpus) in enumerate(specs):
            options = {
                "num_gpus": gpus,
                "num_cpus": cpus,
                "runtime_env": {
                    "env_vars": {"DEEPSPEC_PIPELINE_RUN_ID": plan["run_id"]}
                },
            }
            if pg is not None:
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle,
                    placement_group_capture_child_tasks=False,
                )
            name = "shell-" + participant.replace("/", "-")
            shell = actors.create(
                name,
                PlacementShell,
                node_id=node,
                role="allocation_probe",
                deadline=deadline,
                args=(plan,),
                options=options,
            )
            report = actors.resolve(
                name, shell.allocation.remote(participant), deadline=deadline
            )
            if (
                report["node_id"] != node
                or report["model_initializations"] != 0
                or len(report["gpu_uuids"]) != gpus
            ):
                raise ValueError("Observed shell placement differs")
            agents.register_process(report, deadline=deadline)
            result["allocations"].append(report)
            if participant in gate.expected:
                asyncio.run(gate.report_allocation(report))
            if inject_partial and index == 1:
                injected = True
                raise RuntimeError("Injected partial allocation failure")
        if set(gate.allocations) != set(gate.expected):
            raise ValueError("All-role gate is incomplete")
        result["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- rollback remains mandatory after partial allocation
        result.update(status="passed" if injected else "failed", error=repr(error))
    finally:
        cleanup = Deadline.after(plan["timeouts_seconds"]["cleanup"])
        actor_report = actors.stop(
            deadline=cleanup,
            allocation_ids=[
                n for n in actors.handles if not n.startswith("node-agent-")
            ],
        )
        allocator.stop(deadline=cleanup)
        node_report = agents.stop(deadline=cleanup)
        actors._persist()
        result["cleanup"] = {
            "cleanup_complete": registry.cleanup_complete
            and actor_report["cleanup_complete"]
            and node_report["cleanup_complete"],
            "nodes": node_report,
            "registry": registry.to_dict(),
        }
        if not result["cleanup"]["cleanup_complete"]:
            result["status"] = "failed"
        ray.shutdown()
        atomic_json(output / "result.json", result)
    return result


def main():
    from deepspec.pipeline.cli import load_run
    from tests.run_pipeline_acceptance import record_result

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output-root", default="outputs/ray-topology-acceptance")
    args = parser.parse_args()
    root, plan, _ = load_run(Path(args.plan).parent)
    output = root / "probes" / uuid.uuid4().hex
    output.mkdir(parents=True)
    started = time.time()
    result = execute(plan, output, inject_partial=args.partial)
    record = record_result(
        args.output_root,
        "allocation-probe",
        status=result["status"],
        evidence_level="ray_allocation",
        reason=None
        if result["status"] == "passed"
        else result.get("error", "cleanup failed"),
        evidence={
            "executed": True,
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "started_at": started,
            "ended_at": time.time(),
            "checks": {
                "allocation": result["status"] == "passed",
                "cleanup": result["cleanup"]["cleanup_complete"],
            },
            "artifacts": [str(output / "result.json")],
            "probe": result,
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "artifact": str(output / "result.json"),
                "record_id": record["record_id"],
            }
        )
    )
    return 0 if result["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
