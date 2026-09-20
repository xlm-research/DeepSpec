"""Explicit real CPU lifecycle probe; never a GPU topology/training acceptance.

Run only under deepspec.orchestration.process. It creates its own local Ray
cluster with zero GPUs; existing clusters and unrelated processes are borrowed
by neither the service nor its cleanup ledger.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from deepspec.pipeline.controller import ActorAllocator, NodeAgents, ResourceRegistry
from deepspec.pipeline.groups import StoreService
from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import Deadline, atomic_json
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


class BlockedControlProbe:
    """An intentionally stuck CPU actor RPC, registered before entry."""

    def __init__(self, config):
        self.config = config

    def identity(self):
        from deepspec.pipeline.runtime import actor_identity

        return actor_identity(self.config)

    def block(self, path):
        atomic_json(path, self.identity())
        time.sleep(300)


def cancellation_probe(base_config, output, phase):
    """Cancel one real blocked CPU RPC, without models."""
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import redirect_stdout
    from io import StringIO

    import ray

    from deepspec.orchestration.process import signal_process
    from deepspec.pipeline import cli
    from deepspec.pipeline.controller import RunController

    directory = Path(output) / f"cancel-{phase}"
    directory.mkdir()
    config = json.loads(json.dumps(base_config))
    config["output_dir"] = str(directory)
    config["timeouts_seconds"].update(run=90, allocation=45, cleanup=15)
    facts = node_facts(config)
    facts[0]["ip"] = next(n for n in ray.nodes() if n["Alive"])["NodeManagerAddress"]
    plan = build_plan(
        config, facts, input_plan(config), run_id=uuid.uuid4().hex, now=100
    ).to_dict()
    atomic_json(directory / "plan.json", plan)
    atomic_json(directory / "config.normalized.json", plan["config"])
    registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
    actors = ActorAllocator(plan, registry=registry)
    entered = directory / "blocked.json"

    class Operations:
        identity = None

        def check(self):
            pass

        def __getattr__(self, step):
            def call(*, deadline, **kwargs):
                if step == "allocate":
                    self.actor = actors.create(
                        "blocked-control",
                        BlockedControlProbe,
                        node_id=facts[0]["node_id"],
                        role="cpu_fault",
                        deadline=deadline,
                        args=(
                            {
                                "run_id": plan["run_id"],
                                "plan_hash": plan["plan_hash"],
                            },
                        ),
                        options={
                            "runtime_env": {
                                "env_vars": {
                                    "DEEPSPEC_PIPELINE_RUN_ID": plan["run_id"],
                                    "CUDA_VISIBLE_DEVICES": "",
                                }
                            }
                        },
                    )
                    self.identity = actors.resolve(
                        "blocked-control",
                        self.actor.identity.remote(),
                        deadline=deadline,
                    )
                if step == phase:
                    ray.get(
                        self.actor.block.remote(str(entered)),
                        timeout=deadline.remaining(),
                    )
                    raise AssertionError("Blocked RPC returned without cancellation")
                if step == "release_allocations":
                    return actors.stop(deadline=deadline)
                if step == "verify_cleanup":
                    return {
                        "cleanup_complete": registry.cleanup_complete
                        and self.identity is not None
                        and signal_process(self.identity["process"], 0) == "released"
                    }
                return {"ready": True, "cleanup_complete": True}

            return call

    operations = Operations()
    controller = RunController(plan, operations)
    worker = ThreadPoolExecutor(max_workers=1)
    future = worker.submit(controller.run)
    try:
        waiting = Deadline.after(60)
        while not entered.exists():
            if future.done():
                raise AssertionError(
                    f"Controller exited before {phase}: {future.result()}"
                )
            time.sleep(min(0.05, waiting.remaining()))
        started = time.monotonic()
        with redirect_stdout(StringIO()):
            exit_code = cli.main(["cancel", "--run-dir", str(directory)])
        status = future.result(timeout=20)
        assert exit_code == 130 and status["state"] == "cancelled", status
        assert status["reason"]["code"] == "CANCELLED", status
        assert status["cleanup_complete"], status
        repeated = cli.cancel(directory)
        assert repeated == status
        report = {
            "phase": phase,
            "passed": True,
            "cancel_seconds": time.monotonic() - started,
            "exit_code": exit_code,
            "cleanup_complete": True,
            "repeated_cancel_same": True,
            "allocation": registry.to_dict(),
        }
        atomic_json(directory / "probe.json", report)
        return report
    finally:
        actors.stop(deadline=Deadline.after(15))
        worker.shutdown(wait=True)


def victim_driver(request_path):
    """The parent deliberately SIGKILLs this driver during a blocked actor RPC."""
    import ray

    request = json.loads(Path(request_path).read_text())
    plan, output = request["plan"], Path(request["output"])
    ray.init(
        address=plan["config"]["ray_address"],
        namespace=f"deepspec-{plan['run_id']}",
        log_to_driver=False,
    )
    registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
    actors = ActorAllocator(plan, registry=registry)
    agents = NodeAgents(plan, actors)
    deadline = Deadline.after(180)
    agents.start(deadline=deadline)
    from deepspec.pipeline.transport import probe_config

    service = StoreService()
    service.allocate(
        plan,
        resources={
            "actors": actors,
            "config": probe_config(plan, output),
            "register_process": agents.register_process,
        },
        deadline=deadline,
    ).result(deadline=deadline)
    service.start(gate=None, deadline=deadline).result(deadline=deadline)
    node = next(iter(agents.nodes))
    blocked = actors.create(
        "blocked-control",
        BlockedControlProbe,
        node_id=node,
        role="cpu_fault",
        deadline=deadline,
        args=({"run_id": plan["run_id"], "plan_hash": plan["plan_hash"]},),
        options={
            "runtime_env": {
                "env_vars": {
                    "DEEPSPEC_PIPELINE_RUN_ID": plan["run_id"],
                    "CUDA_VISIBLE_DEVICES": "",
                }
            }
        },
    )
    identity = actors.resolve(
        "blocked-control", blocked.identity.remote(), deadline=deadline
    )
    agents.register_process(identity, deadline=deadline)
    ref = blocked.block.remote(str(output / "blocked.json"))
    atomic_json(
        output / "driver-ready.json",
        {
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "node_id": node,
            "agent_epoch": agents.agent_epochs[node],
            "allocation": registry.to_dict(),
        },
    )
    # Fault injection: control is occupied, but the independent heartbeat thread
    # keeps the NodeAgent lease alive until the driver is killed.
    ray.get(ref, timeout=deadline.remaining())
    raise AssertionError("Fault driver unexpectedly returned")


def driver_loss_probe(base_config, output):
    import signal
    import uuid

    from ray._private.state import actors as actor_state

    from deepspec.orchestration.process import capture_process, signal_process
    from deepspec.pipeline.runtime import validate_message

    output = Path(output) / "driver-loss"
    output.mkdir()
    config = json.loads(json.dumps(base_config))
    config["output_dir"] = str(output)
    config["timeouts_seconds"].update(lease=5, heartbeat=1, cleanup=10)
    facts = node_facts(config)
    import ray

    facts[0]["ip"] = next(n for n in ray.nodes() if n["Alive"])["NodeManagerAddress"]
    run_id = uuid.uuid4().hex
    plan = build_plan(
        config, facts, input_plan(config), run_id=run_id, now=100
    ).to_dict()
    request = output / "request.json"
    atomic_json(request, {"plan": plan, "output": str(output)})
    report = {
        "run_id": run_id,
        "plan_hash": plan["plan_hash"],
        "scope": "Real CPU driver SIGKILL with blocked control and owned master/pool.",
    }
    with (output / "driver.log").open("w") as log:
        driver = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.pipeline_lifecycle_probe",
                "--victim-request",
                str(request),
            ],
            env=dict(
                os.environ, CUDA_VISIBLE_DEVICES="", DEEPSPEC_PIPELINE_RUN_ID=run_id
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        identity = capture_process(driver.pid, run_id)
        try:
            deadline = Deadline.after(150)
            while not (
                (output / "driver-ready.json").exists()
                and (output / "blocked.json").exists()
            ):
                if driver.poll() is not None:
                    raise RuntimeError(
                        f"Victim driver exited during startup: {driver.returncode}"
                    )
                time.sleep(min(0.05, deadline.remaining()))
            ready = json.loads((output / "driver-ready.json").read_text())
            started = time.monotonic()
            assert signal_process(identity, signal.SIGKILL) in ("signalled", "released")
            driver.wait(timeout=5)
            deadline = Deadline.after(20)
            orphan = output / f"orphan-{ready['node_id']}.json"
            while not orphan.exists():
                time.sleep(min(0.05, deadline.remaining()))
            cleanup = json.loads(orphan.read_text())
            validate_message(cleanup, run_id=run_id, plan_hash=plan["plan_hash"])
            assert cleanup["agent_epoch"] == ready["agent_epoch"]
            assert cleanup["orphan"] and cleanup["cleanup_complete"], cleanup
            ray_ids = [
                r["ray_id"]
                for r in ready["allocation"]["resources"]
                if r["kind"] == "actor"
            ]
            while not all(
                actor_state(actor_id).get("State") == "DEAD" for actor_id in ray_ids
            ):
                time.sleep(min(0.05, deadline.remaining()))
            report.update(
                cleanup=cleanup,
                ray_actor_ids=ray_ids,
                all_actors_dead=True,
                cleanup_seconds=time.monotonic() - started,
                passed=True,
            )
        finally:
            if driver.poll() is None:
                signal_process(identity, signal.SIGKILL)
                driver.wait(timeout=5)
            atomic_json(output / "probe.json", report)
    return report


def service_probe(output):
    import ray

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = os.environ["DEEPSPEC_PIPELINE_RUN_ID"]
    deadline = Deadline.after(600)
    bystander = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(900)"],
        env=dict(os.environ, DEEPSPEC_PIPELINE_RUN_ID=run_id + "-bystander"),
    )
    agents = service = actors = None
    outcome = {
        "evidence_level": "cpu_process",
        "run_id": run_id,
        "scope": "Synthetic topology metadata; real CPU Ray actors, lease, 64 MiB Mooncake pool and process cleanup. No GPU/model.",
    }
    try:
        ray.init(
            address="local",
            num_cpus=6,
            num_gpus=0,
            include_dashboard=False,
            namespace=f"deepspec-{run_id}",
            log_to_driver=False,
            object_store_memory=100 * 1024**2,
        )
        local = next(n for n in ray.nodes() if n["Alive"])
        config = task_config("M0", output_dir=output)
        config["ray_address"] = ray.get_runtime_context().gcs_address
        config["nodes"][0]["selector"] = {"node_id": local["NodeID"]}
        config["timeouts_seconds"].update(
            heartbeat=1, lease=30, cleanup=30, initialization=90, transfer=30, run=240
        )
        facts = node_facts(config)
        facts[0]["ip"] = local["NodeManagerAddress"]
        plan = build_plan(
            config, facts, input_plan(config), run_id=run_id, now=100
        ).to_dict()
        atomic_json(output / "synthetic-plan.json", plan)
        atomic_json(output / "plan.json", plan)
        atomic_json(output / "config.normalized.json", plan["config"])
        atomic_json(
            output / "status.json",
            {
                "run_id": run_id,
                "plan_hash": plan["plan_hash"],
                "state": "preparing",
                "phase_detail": "synthetic_cpu_fixture",
            },
        )
        registry = ResourceRegistry(run_id, plan["plan_hash"])
        actors = ActorAllocator(plan, registry=registry)
        agents = NodeAgents(plan, actors)
        agents.start(deadline=deadline)
        compatibility = {
            "run_id": run_id,
            "plan_hash": plan["plan_hash"],
            "output_dir": str(output),
            "samples": [
                {
                    "position": i,
                    "sample_id": str(i),
                    "input_identity": str(i),
                    "length": 10,
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
            "timeout_seconds": 30,
            "events_path": str(output / "buffer-events.jsonl"),
            "store": {"protocol": "tcp", "verify_mode": "full"},
        }
        service = StoreService()
        service.allocate(
            plan,
            resources={
                "actors": actors,
                "config": compatibility,
                "register_process": agents.register_process,
            },
            deadline=deadline,
        ).result(deadline=deadline)
        service.start(gate=None, deadline=deadline).result(deadline=deadline)
        ready = service.ready(deadline=deadline)
        assert ready["ready"] and ready["pool_bytes"] == 64 * 1024**2
        agents.check()
        outcome.update(ready=ready, plan_hash=plan["plan_hash"])
        cleanup = Deadline.after(30)
        assert not service.stop("service phase finished", deadline=cleanup).unknown
        assert agents.stop(deadline=cleanup)["cleanup_complete"]
        # The production entrypoint runs a second supervised driver against this
        # owned CPU cluster. Synthetic GPU topology is never allocated or accepted.
        from deepspec.pipeline.transport import transport_check

        outcome["transport"] = transport_check(output / "plan.json")
        assert outcome["transport"]["status"] == "passed", outcome["transport"]
        outcome["driver_loss"] = driver_loss_probe(config, output)
        outcome["phase_cancellation"] = [
            cancellation_probe(config, output, phase)
            for phase in ("allocate", "initialize", "ready", "run", "drain", "verify")
        ]
    except BaseException as error:
        outcome["error"] = repr(error)
        raise
    finally:
        cleanup = Deadline.after(30)
        started = time.monotonic()
        try:
            if service is not None:
                report = service.stop("probe finished", deadline=cleanup)
                outcome["service_cleanup"] = {
                    "released": report.released,
                    "unknown": report.unknown,
                    "errors": report.errors,
                }
                outcome["repeated_stop_same"] = (
                    service.stop("repeat", deadline=cleanup) == report
                )
            if agents is not None:
                outcome["node_cleanup"] = agents.stop(deadline=cleanup)
            outcome["cleanup_seconds"] = time.monotonic() - started
            outcome["bystander_survived"] = bystander.poll() is None
            if actors is not None:
                outcome["allocation"] = actors.registry.to_dict()
                outcome["cleanup_complete"] = (
                    actors.registry.cleanup_complete
                    and not outcome.get("service_cleanup", {}).get(
                        "unknown", ["missing"]
                    )
                    and outcome.get("node_cleanup", {}).get("cleanup_complete", False)
                )
        finally:
            # This process created this exact bystander and retains its Popen handle.
            bystander.terminate()
            bystander.wait(timeout=5)
            ray.shutdown()
            atomic_json(output / "probe.json", outcome)
    assert (
        outcome["cleanup_complete"]
        and outcome["bystander_survived"]
        and outcome["repeated_stop_same"]
    ), outcome
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--victim-request")
    args = parser.parse_args(argv)
    if args.victim_request:
        victim_driver(args.victim_request)
        return
    if not args.output:
        parser.error("--output is required for the main probe")
    print(json.dumps(service_probe(args.output)))


if __name__ == "__main__":
    main()
