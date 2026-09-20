"""Finite, independently named TCP/CPU transport probes on planned nodes."""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .runtime import Deadline, PipelineError, atomic_json, validate_message


def participants(plan):
    writers = {replica["worker_nodes"][0] for replica in plan["replicas"]}
    readers = {rank["node_id"] for rank in plan["training_ranks"]}
    return writers, readers


def validate_matrix(plan, matrix):
    writers, readers = participants(plan)
    pairs = {(row["writer_node"], row["reader_node"]) for row in matrix}
    if pairs != {(w, r) for w in writers for r in readers} or len(pairs) != len(matrix):
        raise ValueError("Transport probe has incomplete or duplicate node coverage")
    if any(
        row.get("verified") is not True
        or row["nbytes"] <= 0
        or row["nbytes"] != row["expected_bytes"]
        for row in matrix
    ):
        raise ValueError("Transport probe requires full verification of every byte")
    return True


def probe_config(plan, output):
    return {
        "run_id": plan["run_id"],
        "plan_hash": plan["plan_hash"],
        "output_dir": str(output),
        "samples": [
            {
                "position": i,
                "sample_id": str(i),
                "input_identity": str(i),
                "length": 1,
                "nbytes": 100,
            }
            for i in range(4)
        ],
        "producer_dp": 1,
        "capacity_bytes": 32 * 1024**2,
        "window": 4,
        "consumer_world_size": 4,
        "samples_per_update": 4,
        "pool_bytes": 64 * 1024**2,
        "feature_memory_budget": 64 * 1024**2,
        "memory_reserve_bytes": 0,
        "scratch_bound_bytes": 0,
        "timeout_seconds": plan["timeouts_seconds"]["transfer"],
        "events_path": str(output / "buffer-events.jsonl"),
        "store": {"protocol": "tcp", "verify_mode": "full", "rdma_devices": ""},
    }


def execute_probe(plan, probe_id, output):
    """Runs only inside a killable driver; never reserves GPUs or joins model groups."""
    import ray

    from .cluster import StoreProbe
    from .controller import ActorAllocator, NodeAgents, ResourceRegistry
    from .groups import StoreService
    from .planning import TopologyPlan

    plan = TopologyPlan.from_dict(plan).to_dict()
    output = Path(output)
    policy = plan["timeouts_seconds"]
    deadline = Deadline.after(
        min(
            policy["run"],
            policy["allocation"] + policy["initialization"] + policy["transfer"],
        )
    )
    result = {
        "schema_version": 3,
        "run_id": plan["run_id"],
        "plan_hash": plan["plan_hash"],
        "probe_id": probe_id,
        "namespace": f"probe-{probe_id}",
        "evidence_level": "transport",
        "pool_bytes": 64 * 1024**2,
        "protocol": "tcp",
        "receive_device": "cpu",
        "verify_mode": "full",
        "scope": "Small-pool transport only; no GPU, model or feature-capacity acceptance.",
        "matrix": [],
        "deleted": [],
    }
    registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
    actors = ActorAllocator(
        plan,
        registry=registry,
        name_prefix=probe_id,
        namespace=result["namespace"],
        allocation_path=output / "allocation.json",
    )
    agents = NodeAgents(plan, actors, report_dir=str(output))
    service, clients = StoreService(), {}
    try:
        ray.init(
            address=plan["config"]["ray_address"],
            namespace=result["namespace"],
            log_to_driver=False,
        )
        live = {n["NodeID"]: n for n in ray.nodes() if n["Alive"]}
        nodes = {n["node_id"]: n for n in plan["nodes"].values()}
        for node, fact in nodes.items():
            if node not in live or live[node]["NodeManagerAddress"] != fact["ip"]:
                raise PipelineError(
                    "NODE_UNAVAILABLE",
                    "Planned node is absent or its address changed",
                    node_id=node,
                    exit_code=4,
                )
        agents.start(deadline=deadline)
        config = probe_config(plan, output)
        service.allocate(
            plan,
            resources={
                "actors": actors,
                "config": config,
                "register_process": agents.register_process,
            },
            deadline=deadline,
        ).result(deadline=deadline)
        service.start(gate=None, deadline=deadline).result(deadline=deadline)
        writers, readers = participants(plan)
        for node in sorted(writers | readers):
            agents.check()
            store_config = {**service.config["store"], "host": nodes[node]["ip"]}
            actor = actors.create(
                f"probe-{node}",
                StoreProbe,
                node_id=node,
                role="transport_probe",
                deadline=deadline,
                args=(store_config,),
                kwargs={"identity_config": config, "defer_store": True},
                options={
                    "runtime_env": {
                        "env_vars": {
                            "DEEPSPEC_PIPELINE_RUN_ID": plan["run_id"],
                            "CUDA_VISIBLE_DEVICES": "",
                        }
                    }
                },
            )
            clients[node] = actor
            identity = actors.resolve(
                f"probe-{node}", actor.identity.remote(), deadline=deadline
            )
            validate_message(
                identity, run_id=plan["run_id"], plan_hash=plan["plan_hash"]
            )
            if identity["node_id"] != node or identity[
                "actor_id"
            ] != actors.backend.identity(actor):
                raise ValueError("Transport actor allocation identity differs")
            agents.register_process(identity, deadline=deadline)
            actors.backend.get(actor.initialize.remote(), timeout=deadline.remaining())
        for writer in sorted(writers):
            agents.check()
            fields = actors.backend.get(
                clients[writer].write.remote(f"probe/{probe_id}/{writer}"),
                timeout=deadline.remaining(),
            )
            expected = sum(field["nbytes"] for field in fields.values())
            for reader in sorted(readers):
                report = actors.backend.get(
                    clients[reader].read.remote(fields), timeout=deadline.remaining()
                )
                result["matrix"].append(
                    {
                        "writer_node": writer,
                        "reader_node": reader,
                        "expected_bytes": expected,
                        **report,
                    }
                )
            deleted = actors.backend.get(
                clients[writer].remove.remote(fields), timeout=deadline.remaining()
            )
            if deleted.get("confirmed_absent") is not True:
                raise RuntimeError("Transport object deletion was not confirmed")
            result["deleted"].append(writer)
        validate_matrix(plan, result["matrix"])
        result["status"] = "passed"
    except BaseException as error:  # noqa: BLE001 -- every interruption must attempt exact-owned cleanup
        result["status"] = "failed"
        result["error"] = (
            error.to_dict()
            if isinstance(error, PipelineError)
            else {"code": "TRANSPORT_FAILED", "message": repr(error)}
        )
    finally:
        cleanup = Deadline.after(policy["cleanup"])
        started = time.monotonic()
        errors = []
        for node, actor in clients.items():
            try:
                report = actors.backend.get(
                    actor.close.remote(timeout=cleanup.remaining()),
                    timeout=cleanup.remaining(),
                )
                if report.get("cleanup_complete") is not True:
                    raise RuntimeError("Probe client close was not confirmed")
            except Exception as error:  # noqa: BLE001 -- keep all cleanup errors within one shared deadline
                errors.append(f"{node}: {error}")
        actors.stop(
            deadline=cleanup, allocation_ids=[f"probe-{node}" for node in clients]
        )
        service_report = None
        if service.allocation is not None:
            service_report = service.stop("transport probe finished", deadline=cleanup)
            errors.extend(service_report.errors)
        node_report = agents.stop(deadline=cleanup)
        complete = (
            not errors
            and registry.cleanup_complete
            and node_report["cleanup_complete"]
            and (service_report is None or not service_report.unknown)
        )
        result["cleanup"] = {
            "cleanup_complete": complete,
            "seconds": time.monotonic() - started,
            "errors": errors,
            "nodes": node_report,
            "allocation": registry.to_dict(),
        }
        if not complete:
            result["status"] = "failed"
        ray.shutdown()
        atomic_json(output / "transport-probe.json", result)
    return result


def transport_check(plan_path):
    from deepspec.orchestration.process import start_owned

    from .cli import load_run

    plan_path = Path(plan_path).resolve()
    if plan_path.name != "plan.json":
        raise PipelineError(
            "PLAN_PATH_INVALID", "Use the frozen run plan.json", field_path="plan"
        )
    output, plan, _ = load_run(plan_path.parent)
    probe_id = uuid.uuid4().hex
    probe_dir = output / "probes" / probe_id
    probe_dir.mkdir(parents=True, exist_ok=False)
    request = probe_dir / "request.json"
    atomic_json(request, {"plan": plan, "probe_id": probe_id, "output": str(probe_dir)})
    policy = plan["timeouts_seconds"]
    duration = min(
        policy["run"],
        policy["allocation"] + policy["initialization"] + policy["transfer"],
    )
    with (probe_dir / "driver.log").open("w") as log:
        handle = start_owned(
            [sys.executable, "-m", "deepspec.pipeline.transport", str(request)],
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                DEEPSPEC_PIPELINE_RUN_ID=plan["run_id"],
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=duration,
            cleanup_timeout=policy["cleanup"],
            report_path=probe_dir / "supervisor.json",
        )
        failure = None
        try:
            handle.result()
        except Exception as error:  # noqa: BLE001 -- retain the worker's structured result on failure
            failure = error
        finally:
            supervisor = handle.stop(timeout=policy["cleanup"])
    path = probe_dir / "transport-probe.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "schema_version": 3,
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "probe_id": probe_id,
            "status": "failed",
            "error": {"message": str(failure)},
        }
    )
    result["supervisor_cleanup"] = supervisor
    if failure is not None or not supervisor["cleanup_complete"]:
        result["status"] = "failed"
    result["artifact_path"] = str(path)
    atomic_json(path, result)
    atomic_json(output / "transport-probe.json", result)
    return result


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    report = execute_probe(request["plan"], request["probe_id"], request["output"])
    raise SystemExit(0 if report["status"] == "passed" else 3)
