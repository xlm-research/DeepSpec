"""Native controller operations; all Ray RPCs live in a killable driver."""

import json
import time
from pathlib import Path

from .controller import (
    ActorAllocator,
    NodeAgents,
    PlacementAllocator,
    RayPlacementBackend,
    ResourceRegistry,
)
from .execution import (
    CPUVerifier,
    read_events,
    revalidate_environment,
    verification_budget,
)
from .groups import InferenceGroup, StoreService, TrainingGroup
from .runtime import (
    Deadline,
    PipelineError,
    atomic_json,
    message_envelope,
    validate_message,
)


class NativeRunOperations:
    def __init__(self, plan):
        self.plan = plan
        self.output = Path(plan["config"]["output_dir"])
        self.registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
        self.actors = ActorAllocator(plan, registry=self.registry)
        self.placements = PlacementAllocator(
            plan, registry=self.registry, backend=RayPlacementBackend()
        )
        self.agents = NodeAgents(plan, self.actors)
        self.store, self.inference, self.training = (
            StoreService(),
            InferenceGroup(),
            TrainingGroup(),
        )
        self.gate = None
        self.connected = self.models_stopped = self.services_stopped = False
        self.verification = None
        self.group_cleanup = None

    def _get(self, ref, deadline):
        return self.actors.backend.get(ref, timeout=deadline.remaining())

    def _message(self, **fields):
        return message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "controller"},
            **fields,
        )

    def check(self):
        self.agents.check()

    def allocate(self, *, deadline, stop):
        import ray

        from .transport import transport_check, validate_matrix
        from .vllm_adapter import NativeCoordination

        revalidate_environment(self.plan)
        if stop.is_set():
            raise RuntimeError("Allocation cancelled")
        probe_path = self.output / "transport-probe.json"
        probe = (
            json.loads(probe_path.read_text())
            if probe_path.exists()
            else transport_check(self.output / "plan.json")
        )
        if (
            (probe.get("run_id"), probe.get("plan_hash"))
            != (self.plan["run_id"], self.plan["plan_hash"])
            or probe.get("status") != "passed"
            or not probe.get("cleanup", {}).get("cleanup_complete")
        ):
            raise PipelineError(
                "TRANSPORT_NOT_VERIFIED",
                "All-node transport and probe cleanup must pass before model allocation",
                exit_code=3,
            )
        validate_matrix(self.plan, probe["matrix"])
        deadline.remaining()
        if stop.is_set():
            raise RuntimeError("Allocation cancelled")
        ray.init(
            address=self.plan["config"]["ray_address"],
            namespace=self.actors.namespace,
            log_to_driver=False,
        )
        self.connected = True
        self.agents.start(deadline=deadline)
        groups = self.placements.allocate(
            deadline=deadline, cleanup_timeout=self.plan["timeouts_seconds"]["cleanup"]
        )
        self.actors._persist()
        self.gate = self.actors.create(
            "native-gate",
            NativeCoordination,
            node_id=self.plan["replicas"][0]["core_node"],
            role="coordination",
            deadline=deadline,
            args=(self.plan,),
            kwargs={
                "pg_ids": {n: h.id.hex() for n, h in groups.items()},
                "node_agents": self.agents.handles,
                "tokens": self.agents.tokens,
            },
        )
        self.actors.resolve(
            "native-gate", self.gate.snapshot.remote(), deadline=deadline
        )
        config = json.loads((self.output / "pipeline.json").read_text())
        config.update(
            ray_address=self.plan["config"]["ray_address"],
            native_gate_name=f"{self.actors.name_prefix}-native-gate",
            remaining_run_seconds=deadline.remaining()
            + self.plan["timeouts_seconds"]["run"],
        )
        self.store.allocate(
            self.plan,
            resources={
                "actors": self.actors,
                "config": config,
                "register_process": self.agents.register_process,
                "node_monitors": list(self.agents.handles.values()),
            },
            deadline=deadline,
        ).result(deadline=deadline)
        runtime_path = self.output / "pipeline.runtime.json"
        atomic_json(runtime_path, self.store.config)
        resources = {
            "actors": self.actors,
            "gate": self.gate,
            "placement_groups": groups,
            "config_path": runtime_path,
            "register_process": self.agents.register_process,
        }
        # Both launcher sets must exist before either role may initialize CUDA.
        handles = [
            g.allocate(self.plan, resources=resources, deadline=deadline)
            for g in (self.training, self.inference)
        ]
        for handle in handles:
            handle.result(deadline=deadline)
        return {"allocated": True}

    def initialize(self, *, deadline, stop):
        self.store.start(gate=self.gate, deadline=deadline).result(deadline=deadline)
        handles = [
            g.start(gate=self.gate, deadline=deadline)
            for g in (self.training, self.inference)
        ]
        for handle in handles:
            handle.result(deadline=deadline)
        self._wait_native_gate(
            self.gate.wait_for_allocation.remote(timeout=deadline.remaining()),
            deadline=deadline,
            stop=stop,
        )
        self._get(
            self.gate.report_initialized.remote(
                self._message(
                    participant="store",
                    node_id=self.plan["services"]["pool_node_id"],
                    transport_verified=True,
                )
            ),
            deadline,
        )
        return {"started": True}

    def _wait_native_gate(self, pending, *, deadline, stop, frontend_ready=False):
        import ray

        initialized = False
        while True:
            if stop.is_set():
                raise RuntimeError("Native initialization was stopped")
            reports = {
                name: group.status(deadline=deadline)
                for name, group in (
                    ("training", self.training),
                    ("inference", self.inference),
                )
            }
            for report in reports.values():
                if report.get("error") or report.get("state") in ("failed", "stopped"):
                    raise RuntimeError(
                        f"Native role failed during initialization: {report}"
                    )
            if not initialized:
                completed, _ = ray.wait(
                    [pending], timeout=min(0.1, deadline.remaining())
                )
                if completed:
                    self._get(pending, deadline)
                    initialized = True
            if initialized and (
                not frontend_ready
                or reports["inference"].get("ready")
                or reports["inference"]["state"] == "finished"
            ):
                break
            self.check()
            time.sleep(min(0.05, deadline.remaining()))

    def ready(self, *, deadline, stop):
        self._wait_native_gate(
            self.gate.wait_for_initialization.remote(timeout=deadline.remaining()),
            deadline=deadline,
            stop=stop,
            frontend_ready=True,
        )
        snapshot = self._get(self.gate.snapshot.remote(), deadline)
        atomic_json(self.output / "actual-placement.json", snapshot)
        return {"ready": True}

    def run(self, *, deadline, stop):
        while True:
            self.check()
            if stop.is_set():
                raise RuntimeError("Native run was stopped")
            reports = [
                g.status(deadline=deadline) for g in (self.inference, self.training)
            ]
            for report in reports:
                if report.get("error") or report.get("state") in ("failed", "stopped"):
                    raise RuntimeError(f"Native role failed: {report}")
            if all(r["state"] == "finished" for r in reports):
                return {"finished": True, "roles": reports}
            time.sleep(min(0.1, deadline.remaining()))

    def drain(self, *, deadline, stop):
        source = self._get(self.store.pool.source_snapshot.remote(), deadline)
        atomic_json(self.output / "source-release.json", source)
        if (
            source["resident_bytes"]
            or source["reserved_bytes"]
            or len(source["records"]) != len(self.plan["samples"])
            or any(r["state"] != "released" for r in source["records"])
        ):
            raise RuntimeError("Source objects did not drain completely")
        return {"drained": True}

    def verify(self, *, deadline, stop):
        # Verification is sequenced after real model process/PG teardown. The
        # pool and watchdogs retain their approved CPU/memory reservations.
        report = self.stop_groups(deadline=deadline, stop=stop)
        if not report["cleanup_complete"]:
            raise RuntimeError("Model process cleanup must finish before verification")
        report = self.release_allocations(deadline=deadline, stop=stop)
        if not report["cleanup_complete"]:
            raise RuntimeError("GPU allocations remain before CPU verification")
        actor = self.actors.create(
            "cpu-verifier",
            CPUVerifier,
            node_id=self.plan["services"]["pool_node_id"],
            role="verifier",
            deadline=deadline,
            args=(self.plan,),
            options={
                "num_cpus": 1,
                "num_gpus": 0,
                "memory": verification_budget(self.plan),
                "runtime_env": {
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": "",
                        "DEEPSPEC_PIPELINE_RUN_ID": self.plan["run_id"],
                    }
                },
            },
        )
        identity = self.actors.resolve(
            "cpu-verifier", actor.identity.remote(), deadline=deadline
        )
        self.agents.register_process(identity, deadline=deadline)
        self.verification = self._get(actor.verify.remote(), deadline)
        atomic_json(self.output / "checkpoint-verification.json", self.verification)
        cleanup = self.actors.stop(deadline=deadline, allocation_ids=("cpu-verifier",))
        if not cleanup["cleanup_complete"]:
            raise RuntimeError("CPU verifier cleanup is unknown")
        return self.verification

    def stop_admission(self, *, deadline, stop):
        if self.gate is not None and not self.services_stopped:
            self._get(
                self.gate.native_failed.remote(
                    self._message(reason="controller stopping")
                ),
                deadline,
            )
        if self.store.pool is not None and not self.services_stopped:
            self._get(self.store.pool.fail.remote("controller stopping"), deadline)
        return {"cleanup_complete": True}

    def stop_groups(self, *, deadline, stop):
        if self.group_cleanup is not None:
            return self.group_cleanup
        reports = []
        for group in (self.inference, self.training):
            if group.allocation is not None:
                reports.append(group.stop("controller stopping", deadline=deadline))
            else:
                group.executor.shutdown(wait=False)
        self.models_stopped = True
        self.group_cleanup = {
            "cleanup_complete": all(not r.unknown for r in reports),
            "groups": [r.__dict__ for r in reports],
        }
        return self.group_cleanup

    def close_objects(self, *, deadline, stop):
        if self.store.pool is None or self.services_stopped:
            return {"cleanup_complete": True}
        return self._get(
            self.store.pool.close.remote(timeout=deadline.remaining()), deadline
        )

    def release_allocations(self, *, deadline, stop):
        self.placements.stop(deadline=deadline)
        self.actors._persist()
        records = {r["allocation_id"]: r for r in self.registry.to_dict()["resources"]}
        return {
            "cleanup_complete": all(
                records[n]["release_state"] == "released"
                for n in self.placements.handles
            )
        }

    def stop_services(self, *, deadline, stop):
        complete = True
        if self.store.allocation is not None:
            result = self.store.stop("controller stopping", deadline=deadline)
            complete = not result.unknown
        else:
            self.store.executor.shutdown(wait=False)
        if self.gate is not None:
            self._get(self.gate.close_events.remote(), deadline)
        ids = [
            name for name in self.actors.handles if not name.startswith("node-agent-")
        ]
        report = self.actors.stop(deadline=deadline, allocation_ids=ids)
        nodes = self.agents.stop(deadline=deadline)
        self.services_stopped = True
        return {
            "cleanup_complete": complete
            and report["cleanup_complete"]
            and nodes["cleanup_complete"],
            "nodes": nodes,
        }

    def verify_cleanup(self, *, deadline, stop):
        from .verification import verify_release

        self.actors._persist()
        if not self.connected:
            return {"cleanup_complete": self.registry.cleanup_complete}
        native_path = self.output / "native-allocation.json"
        native = (
            json.loads(native_path.read_text())
            if native_path.exists()
            else {
                "schema_version": 3,
                "run_id": self.plan["run_id"],
                "plan_hash": self.plan["plan_hash"],
                "resources": [],
            }
        )
        registries = [self.registry.to_dict(), native]
        observations = observe_cleanup(
            self.plan,
            registries,
            deadline=deadline,
            node_reports=(self.agents.cleanup_result or {}).get("nodes", {}),
        )
        complete = self.registry.cleanup_complete and all(
            o["release_state"] == "released" for o in observations
        )
        atomic_json(
            self.output / "resource-cleanup.json",
            self._message(observations=observations, cleanup_complete=complete),
        )
        if self.verification is not None:
            from .metrics import require_metrics

            metrics = require_metrics(self.plan, read_events(self.output))
            release = verify_release(
                self.plan,
                source=json.loads((self.output / "source-release.json").read_text()),
                registries=registries,
                cleanup=observations,
            )
            result = {
                **self.verification,
                "release": release,
                "verified": complete,
                "metrics": metrics,
            }
            atomic_json(self.output / "verification.json", result)
        import ray

        ray.shutdown()
        return {"cleanup_complete": complete}


def observe_cleanup(plan, registries, *, deadline, node_reports):
    """Read exact IDs again after teardown, independently of ledger transitions."""
    from ray._private.state import actors
    from ray.util.placement_group import placement_group_table

    groups = placement_group_table()
    reports = []
    for registry in registries:
        for resource in registry["resources"]:
            deadline.remaining()
            kind, identity = resource["kind"], resource["ray_id"]
            released = False
            if kind in ("actor", "native_actor"):
                observed = actors(identity)
                released = bool(observed and observed.get("State") == "DEAD")
            elif kind == "placement_group":
                observed = groups.get(identity)
                released = bool(observed and observed.get("state") == "REMOVED")
            elif kind == "process":
                process = resource["process"]
                node = node_reports.get(resource["node_id"], {})
                released = (
                    node.get("release_states", {}).get(
                        f"{process['pid']}:{process['start_ticks']}"
                    )
                    == "released"
                )
            elif kind == "service" and resource["owner"] == "external":
                # Releasing a borrowed connection never means stopping its master.
                released = (
                    resource["release_state"] == "released"
                    and bool(node_reports)
                    and all(n["cleanup_complete"] for n in node_reports.values())
                )
            reports.append(
                message_envelope(
                    plan["run_id"],
                    plan["plan_hash"],
                    {"component": "cleanup_observer"},
                    resource_id=resource["allocation_id"],
                    ray_id=identity,
                    release_state="released" if released else "unknown",
                )
            )
    return reports


def verify_completed_run(plan):
    """Use the plan's verifier CPU slot; preserve the original lifecycle files."""
    import ray

    deadline = Deadline.after(plan["timeouts_seconds"]["run"])
    registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
    actors = ActorAllocator(
        plan,
        registry=registry,
        name_prefix=f"verify-{__import__('uuid').uuid4().hex}",
        allocation_path=Path(plan["config"]["output_dir"])
        / "verification/last-allocation.json",
    )
    try:
        ray.init(address=plan["config"]["ray_address"], log_to_driver=False)
        actor = actors.create(
            "verifier",
            CPUVerifier,
            node_id=plan["services"]["pool_node_id"],
            role="verifier",
            deadline=deadline,
            args=(plan,),
            options={
                "num_cpus": 1,
                "num_gpus": 0,
                "memory": verification_budget(plan),
                "runtime_env": {"env_vars": {"CUDA_VISIBLE_DEVICES": ""}},
            },
        )
        result = actors.resolve(
            "verifier", actor.verify.remote(full=True), deadline=deadline
        )
        validate_message(result, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
    finally:
        cleanup = actors.stop(
            deadline=Deadline.after(plan["timeouts_seconds"]["cleanup"])
        )
        ray.shutdown()
    if not cleanup["cleanup_complete"]:
        raise RuntimeError("Independent verifier cleanup is unknown")
    return {**result, "verifier_cleanup": cleanup}
