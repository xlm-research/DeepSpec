"""Backend contracts. Implementations retain native vLLM and TorchTitan engines.

allocate/start return supervised handles promptly. Blocking engine construction
and training run outside the control/heartbeat executor. Every operation uses
the controller's shared deadline rather than granting a fresh per-RPC timeout.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .runtime import Deadline


class SupervisedHandle(Protocol):
    def done(self) -> bool: ...
    def result(self, *, deadline: Deadline) -> Mapping: ...


@dataclass(frozen=True)
class StopResult:
    released: tuple[str, ...]
    unknown: tuple[str, ...]
    errors: tuple[str, ...] = ()


class ManagedGroup(Protocol):
    def allocate(self, plan, *, resources, deadline: Deadline) -> SupervisedHandle: ...
    def start(self, *, gate, deadline: Deadline) -> SupervisedHandle: ...
    def ready(self, *, deadline: Deadline) -> Mapping: ...
    def status(self, *, deadline: Deadline) -> Mapping: ...
    def stop(self, reason: str, *, deadline: Deadline) -> StopResult: ...


class InferenceGroupContract(ManagedGroup, Protocol):
    """Borrow controller-owned PGs; native vLLM owns its worker/core graph."""


class TrainingGroupContract(ManagedGroup, Protocol):
    """Node-local launchers supervise one native torchrun/TorchTitan task."""


class StoreServiceContract(ManagedGroup, Protocol):
    """Single pool; external masters are borrowed and never stopped."""


class FutureHandle:
    """A timed out wait does not cancel or release the native operation."""

    def __init__(self, future):
        self.future = future

    def done(self):
        return self.future.done()

    def result(self, *, deadline):
        return self.future.result(timeout=deadline.remaining())


class InferenceGroup:
    """One CPU frontend borrowing controller-owned PGs for native vLLM actors."""

    def __init__(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference-group"
        )
        self.stopped = threading.Event()
        self.frontend = self.allocation = self.startup = None
        self.cleanup_result = None

    def _check(self, deadline):
        deadline.remaining()
        if self.stopped.is_set():
            raise RuntimeError("Inference group is stopped")

    def allocate(self, plan, *, resources, deadline):
        from .planning import TopologyPlan

        self._check(deadline)
        if self.allocation is not None:
            raise RuntimeError("Inference allocation may only start once")
        self.plan = TopologyPlan.from_dict(plan).to_dict()
        self.actors, self.gate = resources["actors"], resources["gate"]
        self.register_process = resources["register_process"]
        self.config_path = str(resources["config_path"])
        self.groups = [
            resources["placement_groups"][f"inference-{r['replica_id']}"]
            for r in self.plan["replicas"]
        ]
        self.allocation = FutureHandle(self.executor.submit(self._allocate, deadline))
        return self.allocation

    def _allocate(self, deadline):
        from .actors import Producer
        from .run import environment
        from .runtime import validate_message

        self._check(deadline)
        node = self.plan["replicas"][0]["core_node"]
        self.runtime_env = {
            "env_vars": {
                **environment(
                    self.plan["config"]["model_path"],
                    inference_tp=self.plan["config"]["inference"]["tp"],
                ),
                "DEEPSPEC_PIPELINE_RUN_ID": self.plan["run_id"],
                "DEEPSPEC_RUN_REMAINING_SECONDS": str(
                    self.plan["timeouts_seconds"]["run"]
                ),
            }
        }
        self.frontend = self.actors.create(
            "inference-frontend",
            Producer,
            node_id=node,
            role="inference_frontend",
            deadline=deadline,
            args=(self.config_path,),
            kwargs={
                "native_plan": self.plan,
                "placement_groups": self.groups,
                "gate": self.gate,
                "runtime_env": self.runtime_env,
            },
            options={"num_cpus": 1, "num_gpus": 0, "runtime_env": self.runtime_env},
        )
        identity = self.actors.resolve(
            "inference-frontend", self.frontend.identity.remote(), deadline=deadline
        )
        validate_message(
            identity, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )
        if identity["node_id"] != node or identity[
            "actor_id"
        ] != self.actors.backend.identity(self.frontend):
            raise ValueError("Native frontend allocation differs from its identity")
        self.register_process(identity, deadline=deadline)
        return {
            "allocated": True,
            "frontend": identity,
            "borrowed_pg_ids": [p.id.hex() for p in self.groups],
        }

    def _get(self, ref, deadline):
        return self.actors.backend.get(ref, timeout=deadline.remaining())

    def start(self, *, gate, deadline):
        self._check(deadline)
        if gate is not self.gate:
            raise ValueError("Inference group must use its allocation gate")
        if self.startup is None:
            self.startup = FutureHandle(self.executor.submit(self._start, deadline))
        return self.startup

    def _start(self, deadline):
        self.allocation.result(deadline=deadline)
        self._check(deadline)
        return self._get(
            self.frontend.start.remote(initialization_timeout=deadline.remaining()),
            deadline,
        )

    def ready(self, *, deadline):
        import time

        self._check(deadline)
        report = self._get(
            self.gate.wait_for_initialization.remote(timeout=deadline.remaining()),
            deadline,
        )
        while True:
            self._check(deadline)
            status = self.status(deadline=deadline)
            if status.get("error") or status.get("state") in ("failed", "stopped"):
                raise RuntimeError(status.get("error") or "Inference frontend stopped")
            if status.get("ready") or status.get("state") == "finished":
                return report
            time.sleep(min(0.05, deadline.remaining()))

    def status(self, *, deadline):
        return self._get(self.frontend.status.remote(), deadline)

    def stop(self, reason, *, deadline):
        if self.cleanup_result is not None:
            return self.cleanup_result
        self.stopped.set()
        released, unknown, errors = [], [], []
        try:
            self._get(
                self.gate.native_failed.remote(self.gate_message(reason)), deadline
            )
            if self.frontend is not None:
                result = self._get(
                    self.frontend.stop.remote(reason, timeout=deadline.remaining()),
                    deadline,
                )
                if not result["cleanup_complete"]:
                    unknown.append("inference-frontend-processes")
        except Exception as error:  # noqa: BLE001 -- complete native and frontend cleanup after failure
            errors.append(str(error))
        try:
            result = self._get(
                self.gate.stop_native.remote(reason, timeout=deadline.remaining()),
                deadline,
            )
            for resource in result["resources"]:
                (
                    released if resource["release_state"] == "released" else unknown
                ).append(resource["allocation_id"])
            errors.extend(str(error) for error in result["errors"])
        except Exception as error:  # noqa: BLE001 -- missing native release evidence stays unknown
            unknown.append("native-actors")
            errors.append(str(error))
        if self.frontend is not None:
            result = self.actors.stop(
                deadline=deadline, allocation_ids=("inference-frontend",)
            )
            (released if result["cleanup_complete"] else unknown).append(
                "inference-frontend"
            )
            errors.extend(str(error) for error in result["errors"])
        self.executor.shutdown(wait=False)
        self.cleanup_result = StopResult(tuple(released), tuple(unknown), tuple(errors))
        return self.cleanup_result

    def gate_message(self, reason):
        from .runtime import message_envelope

        return message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "inference_group"},
            reason=reason,
        )


class TrainingGroup:
    """Node-local launchers join one native torchrun TP/FSDP task."""

    def __init__(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="training-group"
        )
        self.stopped = threading.Event()
        self.allocation = self.startup = self.launcher = None
        self.launchers = {}
        self.cleanup_result = None

    def _check(self, deadline):
        deadline.remaining()
        if self.stopped.is_set():
            raise RuntimeError("Training group is stopped")

    def _get(self, ref, deadline):
        return self.actors.backend.get(ref, timeout=deadline.remaining())

    def allocate(self, plan, *, resources, deadline):
        from .planning import TopologyPlan

        self._check(deadline)
        if self.allocation is not None:
            raise RuntimeError("Training allocation may only start once")
        self.plan = TopologyPlan.from_dict(plan).to_dict()
        self.actors, self.gate = resources["actors"], resources["gate"]
        self.groups = resources["placement_groups"]
        self.config_path = str(resources["config_path"])
        self.register_process = resources["register_process"]
        self.allocation = FutureHandle(self.executor.submit(self._allocate, deadline))
        return self.allocation

    def _allocate(self, deadline):
        import json
        from pathlib import Path
        from .runtime import atomic_json, validate_message

        reports = [
            self._allocate_node(rank, deadline)
            for rank in self.plan["training_ranks"]
            if rank["local_rank"] == 0
        ]
        self.launcher = self.launchers[0]
        if len(self.launchers) > 1:
            report = self._get(self.launcher.reserve_rendezvous.remote(), deadline)
            validate_message(
                report, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
            )
            first = self.plan["training_ranks"][0]["node_id"]
            host = next(
                n["ip"] for n in self.plan["nodes"].values() if n["node_id"] == first
            )
            if (
                report["node_id"] != first
                or report["host"] != host
                or report["actor_id"] != self.actors.backend.identity(self.launcher)
            ):
                raise ValueError("Rendezvous reservation belongs to another launcher")
            endpoint = {"host": report["host"], "port": report["port"]}
            config = json.loads(Path(self.config_path).read_text())
            config["consumer_rendezvous"] = endpoint
            atomic_json(self.config_path, config)
            atomic_json(
                Path(self.plan["config"]["output_dir"]) / "rendezvous.json", report
            )
            refs = [
                launcher.configure_rendezvous.remote(endpoint)
                for launcher in self.launchers.values()
            ]
            for ref in refs:
                self._get(ref, deadline)
        return {"allocated": True, "launchers": reports}

    def _allocate_node(self, rank, deadline):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from .actors import Consumer
        from .controller import ProcessIdentity
        from .run import environment
        from .runtime import validate_message

        self._check(deadline)
        node_rank = rank["node_rank"]
        name = f"training-launcher-{node_rank}"
        launcher = self.actors.create(
            name,
            Consumer,
            node_id=rank["node_id"],
            role="training_launcher",
            deadline=deadline,
            args=(self.config_path,),
            kwargs={
                "node_rank": node_rank,
                "native_plan": self.plan,
                "gate": self.gate,
            },
            options={
                "num_cpus": rank["local_world_size"] * 2,
                "num_gpus": rank["local_world_size"],
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=self.groups[f"training-{node_rank}"],
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=False,
                ),
                "runtime_env": {
                    "env_vars": {
                        **environment(self.plan["config"]["model_path"]),
                        "DEEPSPEC_PIPELINE_RUN_ID": self.plan["run_id"],
                    }
                },
            },
        )
        self.launchers[node_rank] = launcher
        report = self.actors.resolve(
            name,
            launcher.allocation.remote(timeout=deadline.remaining()),
            deadline=deadline,
        )
        validate_message(
            report, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )
        if (
            report["actor_id"] != self.actors.backend.identity(launcher)
            or report["node_id"] != rank["node_id"]
        ):
            raise ValueError("Actual training launcher identity differs")
        self.register_process(report, deadline=deadline)
        process = report["process"]
        self.actors.registry.observe(
            name,
            process=ProcessIdentity(
                node_id=rank["node_id"],
                **{
                    k: process[k]
                    for k in ("pid", "start_ticks", "run_id", "parent_pid", "group_id")
                },
            ),
            gpu_uuids=report["gpu_uuids"],
        )
        self.actors._persist()
        self._get(
            self.gate.observe_launcher.remote(report, timeout=deadline.remaining()),
            deadline,
        )
        return report

    def start(self, *, gate, deadline):
        self._check(deadline)
        if gate is not self.gate:
            raise ValueError("Training group must use its allocation gate")
        if self.startup is None:
            self.startup = FutureHandle(self.executor.submit(self._start, deadline))
        return self.startup

    def _start(self, deadline):
        self.allocation.result(deadline=deadline)
        self._check(deadline)
        refs = [
            launcher.start.remote(initialization_timeout=deadline.remaining())
            for launcher in self.launchers.values()
        ]
        return {
            "started": True,
            "launchers": [self._get(ref, deadline) for ref in refs],
        }

    def ready(self, *, deadline):
        self._check(deadline)
        return self._get(
            self.gate.wait_for_initialization.remote(timeout=deadline.remaining()),
            deadline,
        )

    def status(self, *, deadline):
        reports = {
            rank: self._get(launcher.status.remote(), deadline)
            for rank, launcher in self.launchers.items()
        }
        failed = next(
            (
                r
                for r in reports.values()
                if r.get("error") or r["state"] in ("failed", "stopped")
            ),
            None,
        )
        return {
            "state": "failed"
            if failed
            else "finished"
            if all(r["state"] == "finished" for r in reports.values())
            else "running",
            "error": None
            if failed is None
            else failed.get("error") or "Training launcher stopped",
            "launchers": reports,
        }

    def stop(self, reason, *, deadline):
        from .runtime import message_envelope

        if self.cleanup_result is not None:
            return self.cleanup_result
        self.stopped.set()
        released, unknown, errors = [], [], []
        try:
            message = message_envelope(
                self.plan["run_id"],
                self.plan["plan_hash"],
                {"component": "training_group"},
                reason=reason,
            )
            self._get(self.gate.native_failed.remote(message), deadline)
        except Exception as error:  # noqa: BLE001 -- keep stopping owned launcher after notification failure
            errors.append(str(error))
        for rank, launcher in self.launchers.items():
            name = f"training-launcher-{rank}"
            try:
                report = self._get(
                    launcher.stop.remote(reason, timeout=deadline.remaining()),
                    deadline,
                )
                if not report["cleanup_complete"]:
                    unknown.append(f"training-native-processes-{rank}")
            except Exception as error:  # noqa: BLE001 -- retain unknown until process-level cleanup is independently verified
                unknown.append(f"training-native-processes-{rank}")
                errors.append(str(error))
            report = self.actors.stop(deadline=deadline, allocation_ids=(name,))
            (released if report["cleanup_complete"] else unknown).append(name)
            errors.extend(str(error) for error in report["errors"])
        self.executor.shutdown(wait=False)
        self.cleanup_result = StopResult(tuple(released), tuple(unknown), tuple(errors))
        return self.cleanup_result


class StoreService:
    """One FeatureBuffer-owned pool and an optional owned master on its node.

    ``resources`` supplies the shared ActorAllocator, compatibility config and a
    register_process callback which acknowledges NodeAgent registration before
    any service starts. External endpoints are recorded as borrowed resources.
    """

    def __init__(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="store-service"
        )
        self.stopped = threading.Event()
        self.allocation = self.startup = None
        self.master = self.pool = None
        self.actor_ids = []
        self.resource_ids = []
        self.cleanup_result = None

    def allocate(self, plan, *, resources, deadline):
        from .planning import TopologyPlan

        if self.allocation is not None:
            raise RuntimeError("Store allocation may only start once")
        self.plan = TopologyPlan.from_dict(plan).to_dict()
        self.actors = resources["actors"]
        self.registry = self.actors.registry
        self.register_process = resources["register_process"]
        self.config = dict(resources["config"])
        self.node_monitors = resources.get("node_monitors", [])
        self.config.update(
            run_id=plan["run_id"],
            plan_hash=plan["plan_hash"],
            timeouts_seconds=plan["timeouts_seconds"],
            consumer_node_id=plan["services"]["pool_node_id"],
            producer_node_ids=[r["core_node"] for r in plan["replicas"]],
        )
        self.node_id = plan["services"]["pool_node_id"]
        self.node = next(
            n for n in plan["nodes"].values() if n["node_id"] == self.node_id
        )
        self.allocation = FutureHandle(self.executor.submit(self._allocate, deadline))
        return self.allocation

    def _check(self, deadline):
        deadline.remaining()
        if self.stopped.is_set():
            raise RuntimeError("Store service is stopped")

    def _get(self, ref, deadline):
        return self.actors.backend.get(ref, timeout=deadline.remaining())

    def _actor(self, name, cls, args, kwargs, deadline):
        from .runtime import validate_message

        self._check(deadline)
        actor = self.actors.create(
            name,
            cls,
            node_id=self.node_id,
            role="store",
            args=args,
            kwargs=kwargs,
            deadline=deadline,
            options={
                "runtime_env": {
                    "env_vars": {
                        "DEEPSPEC_PIPELINE_RUN_ID": self.plan["run_id"],
                        "CUDA_VISIBLE_DEVICES": "",
                        "DEEPSPEC_STORE_HOST": self.node["ip"],
                    }
                }
            },
        )
        self.actor_ids.append(name)
        identity = self.actors.resolve(name, actor.identity.remote(), deadline=deadline)
        validate_message(
            identity, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )
        if identity["node_id"] != self.node_id or identity[
            "actor_id"
        ] != self.actors.backend.identity(actor):
            raise ValueError(
                "Store actor identity differs from its registered allocation"
            )
        self.register_process(identity, deadline=deadline)
        self._check(deadline)
        return actor

    def _allocate(self, deadline):
        from .actors import OwnedMasterActor
        from .buffer import FeatureBuffer
        from .controller import Allocation

        policy = self.plan["services"]["master"]
        if policy["mode"] == "owned":
            self.master = self._actor(
                "store-master",
                OwnedMasterActor,
                (self.config, self.node["ip"], policy.get("endpoint")),
                {},
                deadline,
            )
            endpoint = self._get(self.master.configuration.remote(), deadline)[
                "endpoint"
            ]
        else:
            endpoint = policy["endpoint"]
            self.registry.register(
                Allocation(
                    allocation_id="external-master",
                    run_id=self.plan["run_id"],
                    plan_hash=self.plan["plan_hash"],
                    owner="external",
                    kind="service",
                    role="store",
                    ray_id=f"endpoint:{endpoint}",
                    node_id=self.node_id,
                    borrower="DeepSpec",
                )
            )
            self.resource_ids.append("external-master")
            self.registry.transition("external-master", "acquiring")
            self.registry.transition("external-master", "acquired")
        self.config["buffer_name"] = f"{self.actors.name_prefix}-feature-buffer"
        self.config["namespace"] = self.actors.namespace
        self.config["store"] = {
            **self.config.get("store", {}),
            "host": self.node["ip"],
            "master": endpoint,
        }
        self.pool = self._actor(
            "feature-buffer",
            FeatureBuffer,
            (self.config,),
            {"defer_store": True, "node_monitors": self.node_monitors},
            deadline,
        )
        return {
            "allocated": True,
            "store": self.config["store"],
            "buffer_name": self.config["buffer_name"],
        }

    def start(self, *, gate, deadline):
        self._check(deadline)
        if self.startup is None:
            self.startup = FutureHandle(self.executor.submit(self._start, deadline))
        return self.startup

    def _wait_ready(self, method, deadline):
        import time

        while True:
            self._check(deadline)
            result = self._get(method.remote(), deadline)
            if result.get("error"):
                raise RuntimeError(result["error"])
            if result["ready"]:
                return result
            time.sleep(min(0.05, deadline.remaining()))

    def _start(self, deadline):
        self.allocation.result(deadline=deadline)
        self._check(deadline)
        if self.master is not None:
            self._get(self.master.start.remote(), deadline)
            master = self._wait_ready(self.master.ready, deadline)
            self._register_master_process(master, deadline)
        self._get(self.pool.start.remote(), deadline)
        result = self._wait_ready(self.pool.service_status, deadline)
        return {**result, "master": self.config["store"]["master"]}

    def _register_master_process(self, report, deadline):
        from .controller import Allocation, ProcessIdentity
        from .runtime import message_envelope

        identity = report["process"]
        self.registry.register(
            Allocation(
                allocation_id="master-process",
                run_id=self.plan["run_id"],
                plan_hash=self.plan["plan_hash"],
                owner="DeepSpec",
                kind="process",
                role="store",
                node_id=self.node_id,
                ray_id=f"process:{self.node_id}:{identity['pid']}:{identity['start_ticks']}",
                process=ProcessIdentity(
                    node_id=self.node_id,
                    **{
                        k: identity[k]
                        for k in (
                            "pid",
                            "start_ticks",
                            "run_id",
                            "parent_pid",
                            "group_id",
                        )
                    },
                ),
            )
        )
        self.resource_ids.append("master-process")
        self.registry.transition("master-process", "acquiring")
        self.register_process(
            message_envelope(
                self.plan["run_id"],
                self.plan["plan_hash"],
                {"component": "store"},
                node_id=self.node_id,
                process=identity,
                supervisor_report_path=report["result"]["supervisor_report_path"],
            ),
            deadline=deadline,
        )
        self.registry.transition("master-process", "acquired")
        self.actors._persist()

    def ready(self, *, deadline):
        self._check(deadline)
        if self.startup is None:
            raise RuntimeError("Store service has not started")
        return self.startup.result(deadline=deadline)

    def status(self, *, deadline):
        deadline.remaining()
        if self.stopped.is_set():
            return {"state": "stopped", "ready": False}
        if self.startup is None or not self.startup.done():
            return {"state": "initializing", "ready": False}
        self.startup.result(deadline=deadline)
        result = self._get(self.pool.service_status.remote(), deadline)
        if (
            self.master is not None
            and not self._get(self.master.ready.remote(), deadline)["ready"]
        ):
            raise RuntimeError("Owned master exited after startup")
        return result

    def stop(self, reason, *, deadline):
        self.stopped.set()
        if self.cleanup_result is not None:
            return self.cleanup_result
        errors = []
        pending = []
        for handle in (self.allocation, self.startup):
            if handle is None:
                continue
            try:
                handle.result(deadline=deadline)
            except Exception as error:  # noqa: BLE001 -- allocation failures must not prevent cleanup
                if not handle.done():
                    pending.append("store-operation")
                    errors.append(str(error))
        pool_closed = self.pool is None
        master_closed = self.master is None
        if self.pool is not None:
            try:
                pool_closed = self._get(
                    self.pool.close.remote(timeout=deadline.remaining()), deadline
                )["cleanup_complete"]
            except Exception as error:  # noqa: BLE001 -- retain failed pool close and continue exact owned resources
                errors.append(str(error))
        if self.master is not None:
            try:
                master_closed = self._get(
                    self.master.stop.remote(reason, timeout=deadline.remaining()),
                    deadline,
                )["cleanup_complete"]
            except Exception as error:  # noqa: BLE001 -- unknown native children cannot be reported released
                errors.append(str(error))
        for name in self.resource_ids:
            self.registry.transition(name, "releasing")
            confirmed = pool_closed if name == "external-master" else master_closed
            self.registry.transition(name, "released" if confirmed else "unknown")
        self.actors.stop(deadline=deadline, allocation_ids=self.actor_ids)
        self.executor.shutdown(wait=False, cancel_futures=False)
        records = {r["allocation_id"]: r for r in self.registry.to_dict()["resources"]}
        ids = self.actor_ids + self.resource_ids
        released = tuple(
            name for name in ids if records[name]["release_state"] == "released"
        )
        unknown = [name for name in ids if records[name]["release_state"] != "released"]
        if not pool_closed:
            unknown.append("store-pool")
        if not master_closed:
            unknown.append("owned-master")
        self.cleanup_result = StopResult(
            released, tuple(unknown + pending), tuple(errors)
        )
        return self.cleanup_result
