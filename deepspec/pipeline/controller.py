"""Pure lifecycle and resource ownership ledger; runtime handles stay outside."""

from dataclasses import asdict, dataclass, replace

from .runtime import PipelineError

RUN_ORDER = (
    "preparing",
    "allocating",
    "initializing",
    "ready",
    "running",
    "draining",
    "succeeded",
)
TERMINAL = {"succeeded", "failed", "cancelled"}
RESOURCE_TRANSITIONS = {
    "declared": {"acquiring", "releasing"},
    "acquiring": {"acquired", "releasing"},
    "acquired": {"releasing"},
    "releasing": {"released", "unknown"},
    "released": set(),
    "unknown": set(),
}


@dataclass
class RunState:
    state: str = "preparing"
    reason: str | None = None

    def transition(self, target, *, reason=None):
        if target == self.state:
            return
        if self.state in TERMINAL or (
            target not in ("failed", "cancelled")
            and target != RUN_ORDER[RUN_ORDER.index(self.state) + 1]
        ):
            raise PipelineError(
                "INVALID_STATE_TRANSITION", f"Cannot move {self.state} to {target}"
            )
        self.state = target
        if self.reason is None:
            self.reason = reason


@dataclass(frozen=True)
class ProcessIdentity:
    node_id: str
    pid: int
    start_ticks: int
    run_id: str
    parent_pid: int
    group_id: int


@dataclass(frozen=True)
class Allocation:
    allocation_id: str
    run_id: str
    plan_hash: str
    owner: str
    kind: str
    role: str
    ray_id: str
    node_id: str
    bundle_index: int | None = None
    gpu_uuids: tuple[str, ...] = ()
    process: ProcessIdentity | None = None
    borrower: str | None = None
    release_state: str = "declared"


class RunController:
    """Single status writer around bounded backend operations.

    Operations implement the named lifecycle/cleanup steps with a shared
    deadline and a stop Event. Native blocking calls belong in supervised
    actors/processes; this executor keeps cancellation and lease checks live.
    A concrete native operations adapter is required before CLI run is enabled.
    """

    CLEANUP_STEPS = (
        "stop_admission",
        "stop_groups",
        "close_objects",
        "release_allocations",
        "stop_services",
        "verify_cleanup",
    )

    def __init__(self, plan, operations):
        import threading
        from pathlib import Path

        from .planning import TopologyPlan

        self.plan = TopologyPlan.from_dict(plan).to_dict()
        self.output = Path(plan["config"]["output_dir"]).resolve()
        self.operations = operations
        self.state, self.failure = RunState(), None
        self.phase = "preparing"
        self.stop_requested = threading.Event()
        self.cleanup, self.cleanup_complete, self.cleanup_finished = {}, False, False
        self.verification = None
        self._last_heartbeat = 0.0

    def _heartbeat(self):
        import os
        import time
        from pathlib import Path
        from .runtime import atomic_json

        now = time.monotonic()
        if now - self._last_heartbeat >= 0.5:
            atomic_json(
                self.output / "controller-heartbeat.json",
                {
                    "run_id": self.plan["run_id"],
                    "plan_hash": self.plan["plan_hash"],
                    "execution_id": self.execution_id,
                    "pid": os.getpid(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id")
                    .read_text()
                    .strip(),
                    "local_monotonic": now,
                    "phase": self.phase,
                },
            )
            self._last_heartbeat = now

    def _status(self):
        return {
            "schema_version": 3,
            "run_id": self.plan["run_id"],
            "plan_hash": self.plan["plan_hash"],
            "state": self.state.state,
            "phase_detail": self.phase,
            "reason": self.failure,
            "cleanup": self.cleanup,
            "cleanup_complete": self.cleanup_complete,
            "cleanup_finished": self.cleanup_finished,
            "verification": self.verification,
        }

    def _save(self):
        from .runtime import atomic_json

        atomic_json(self.output / "status.json", self._status())

    def _cancel_request(self):
        import json

        from .runtime import validate_message

        path = self.output / "control/cancel.json"
        if path.exists():
            request = json.loads(path.read_text())
            validate_message(
                request, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
            )
            if request.get("execution_id") != self.execution_id:
                raise PipelineError(
                    "CANCEL_IDENTITY_MISMATCH",
                    "Cancel request belongs to another execution",
                )
            raise PipelineError(
                "CANCELLED", "User requested cancellation", exit_code=130
            )

    def _poll(self):
        self._cancel_request()
        self.operations.check()

    def _wait(self, future, deadline, *, poll=True):
        import time

        while True:
            self._heartbeat()
            if poll:
                self._poll()
            remaining = deadline.remaining()
            if future.done():
                return future.result()
            time.sleep(min(0.025, remaining))

    def _failed(self, error):
        self.stop_requested.set()
        if self.failure is None:
            if isinstance(error, TimeoutError):
                error = PipelineError(
                    "RUN_DEADLINE",
                    "Run exceeded its shared deadline",
                    phase=self.phase,
                    exit_code=3,
                )
            self.failure = (
                error.to_dict()
                if isinstance(error, PipelineError)
                else PipelineError(
                    "RUN_FAILED",
                    str(error),
                    run_id=self.plan["run_id"],
                    phase=self.phase,
                    exit_code=3,
                ).to_dict()
            )
            self.failure["run_id"] = self.failure.get("run_id") or self.plan["run_id"]
            self.failure["phase"] = self.failure.get("phase") or self.phase
            self.state.transition(
                "cancelled" if self.failure["code"] == "CANCELLED" else "failed",
                reason=self.failure["message"],
            )

    def run(self):
        import json
        import os
        import uuid
        from concurrent.futures import ThreadPoolExecutor

        from .runtime import Deadline, atomic_json_once, message_envelope

        policy = self.plan["timeouts_seconds"]
        run_deadline = Deadline.after(policy["run"])
        status_path = self.output / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if (status.get("run_id"), status.get("plan_hash")) != (
                self.plan["run_id"],
                self.plan["plan_hash"],
            ):
                raise PipelineError(
                    "RUN_IDENTITY_MISMATCH", "Status belongs to another plan"
                )
            if status["state"] in TERMINAL:
                raise PipelineError(
                    "RUN_ALREADY_EXECUTED", "Plan was already executed or terminated"
                )
        self.execution_id = uuid.uuid4().hex
        execution = message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "controller", "pid": os.getpid()},
            execution_id=self.execution_id,
        )
        if not atomic_json_once(self.output / "execution.json", execution):
            raise PipelineError("RUN_ALREADY_EXECUTED", "Plan was already executed")
        work = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline-phases")
        cleanup_work = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pipeline-cleanup"
        )
        try:
            self._save()
            for state, step, limit in (
                ("allocating", "allocate", "allocation"),
                ("initializing", "initialize", "initialization"),
                ("ready", "ready", "initialization"),
                ("running", "run", "run"),
                ("draining", "drain", "transfer"),
            ):
                self.phase = state
                # The ready transition follows, rather than precedes, all-role readiness.
                if state != "ready":
                    self.state.transition(state)
                self._save()
                deadline = Deadline(
                    min(
                        run_deadline.expires_at,
                        Deadline.after(policy[limit]).expires_at,
                    )
                )
                future = work.submit(
                    getattr(self.operations, step),
                    deadline=deadline,
                    stop=self.stop_requested,
                )
                try:
                    result = self._wait(future, deadline)
                except TimeoutError as error:
                    raise PipelineError(
                        "RUN_DEADLINE",
                        f"{state} exceeded its shared deadline",
                        phase=state,
                        exit_code=3,
                    ) from error
                if state == "ready":
                    if result.get("ready") is not True:
                        raise RuntimeError("All-role readiness was not confirmed")
                    self.state.transition("ready")
                    self._save()
            self.phase = "verifying"
            self._save()
            self.verification = self._wait(
                work.submit(
                    self.operations.verify,
                    deadline=run_deadline,
                    stop=self.stop_requested,
                ),
                run_deadline,
            )
            if not (
                self.verification.get("verified") is True
                and self.verification.get("independent") is True
                and self.verification.get("run_id") == self.plan["run_id"]
                and self.verification.get("plan_hash") == self.plan["plan_hash"]
            ):
                raise RuntimeError(
                    "Independent checkpoint verification was not confirmed"
                )
            self._poll()
        except BaseException as error:  # noqa: BLE001 -- interruption must still enter owned cleanup
            self._failed(error)
        finally:
            self.stop_requested.set()
            self.phase = "cleanup"
            cleanup_deadline = Deadline.after(policy["cleanup"])
            self._save()
            for step in self.CLEANUP_STEPS:
                try:
                    cleanup_deadline.remaining()
                    future = cleanup_work.submit(
                        getattr(self.operations, step),
                        deadline=cleanup_deadline,
                        stop=self.stop_requested,
                    )
                    report = self._wait(future, cleanup_deadline, poll=False)
                    self.cleanup[step] = report
                    if report.get("cleanup_complete") is not True:
                        self.cleanup[step] = {**report, "cleanup_complete": False}
                except BaseException as error:  # noqa: BLE001 -- record incomplete cleanup without losing the first cause
                    self.cleanup[step] = {
                        "cleanup_complete": False,
                        "error": repr(error),
                    }
            self.cleanup_complete = all(
                r.get("cleanup_complete") is True for r in self.cleanup.values()
            )
            self.cleanup_finished = True
            work.shutdown(wait=False, cancel_futures=True)
            cleanup_work.shutdown(wait=False, cancel_futures=True)
            if self.failure is None:
                try:
                    self._cancel_request()
                except BaseException as error:  # noqa: BLE001 -- cancellation during cleanup must remain terminal
                    self._failed(error)
            if self.failure is None and not self.cleanup_complete:
                self._failed(
                    PipelineError(
                        "CLEANUP_UNKNOWN",
                        "Cleanup could not be fully confirmed",
                        phase="cleanup",
                        exit_code=3,
                    )
                )
            if self.failure is None:
                self.state.transition("succeeded")
            self.phase = "complete"
            self._save()
        return self._status()


class ResourceRegistry:
    def __init__(self, run_id, plan_hash):
        import threading

        self.run_id, self.plan_hash = run_id, plan_hash
        self._resources = {}
        self._lock = threading.RLock()

    def register(self, allocation):
        with self._lock:
            self._register(allocation)

    def _register(self, allocation):
        for field in (
            "allocation_id",
            "run_id",
            "plan_hash",
            "owner",
            "kind",
            "role",
            "ray_id",
            "node_id",
        ):
            if not isinstance(getattr(allocation, field), str) or not getattr(
                allocation, field
            ):
                raise PipelineError(
                    "RESOURCE_IDENTITY_INVALID",
                    f"{field} requires a persistent string identity",
                )
        if not isinstance(allocation.gpu_uuids, tuple) or any(
            not isinstance(x, str) or not x for x in allocation.gpu_uuids
        ):
            raise PipelineError(
                "RESOURCE_IDENTITY_INVALID",
                "GPU identities must be an immutable tuple of UUIDs",
            )
        if (allocation.run_id, allocation.plan_hash) != (self.run_id, self.plan_hash):
            raise PipelineError(
                "RESOURCE_IDENTITY_MISMATCH",
                "Resource belongs to a different run or plan",
            )
        if (
            allocation.owner not in ("DeepSpec", "external")
            or allocation.release_state != "declared"
        ):
            raise PipelineError(
                "RESOURCE_OWNER_INVALID",
                "Register a declared owned or external resource",
            )
        if allocation.kind == "placement_group" and allocation.owner != "DeepSpec":
            raise PipelineError(
                "RESOURCE_OWNER_INVALID", "DeepSpec must own pipeline placement groups"
            )
        if allocation.process and (
            allocation.process.run_id != self.run_id
            or allocation.process.node_id != allocation.node_id
        ):
            raise PipelineError(
                "RESOURCE_IDENTITY_MISMATCH",
                "Process belongs to a different run or node",
            )
        previous = self._resources.get(allocation.allocation_id)
        if previous is not None:
            if previous == allocation:
                return
            raise PipelineError("RESOURCE_CONFLICT", "Allocation ID already registered")
        if not allocation.ray_id or len(set(allocation.gpu_uuids)) != len(
            allocation.gpu_uuids
        ):
            raise PipelineError(
                "RESOURCE_CONFLICT",
                "Exact resource ID and unique GPU UUIDs are required",
            )
        for item in self._resources.values():
            if item.ray_id == allocation.ray_id or (
                item.node_id == allocation.node_id
                and set(item.gpu_uuids) & set(allocation.gpu_uuids)
            ):
                raise PipelineError(
                    "RESOURCE_CONFLICT",
                    "Ray ID or physical GPU already belongs to a resource",
                )
        self._resources[allocation.allocation_id] = allocation

    def transition(self, allocation_id, target):
        with self._lock:
            self._transition(allocation_id, target)

    def _transition(self, allocation_id, target):
        item = self._resources[allocation_id]
        if target == item.release_state:
            return
        if target not in RESOURCE_TRANSITIONS[item.release_state]:
            raise PipelineError(
                "INVALID_STATE_TRANSITION",
                f"Cannot move {item.release_state} to {target}",
            )
        self._resources[allocation_id] = replace(item, release_state=target)

    def owned_resources(self):
        with self._lock:
            return tuple(x for x in self._resources.values() if x.owner == "DeepSpec")

    def observe(self, allocation_id, *, process=None, gpu_uuids=None):
        """Bind acquired facts to an existing owned allocation exactly once."""
        with self._lock:
            item = self._resources[allocation_id]
            if item.owner != "DeepSpec" or item.release_state not in (
                "acquiring",
                "acquired",
            ):
                raise PipelineError(
                    "RESOURCE_OBSERVATION_INVALID",
                    "Only active owned allocations accept facts",
                )
            if process is not None:
                if (
                    not isinstance(process, ProcessIdentity)
                    or process.node_id != item.node_id
                    or process.run_id != self.run_id
                    or process.pid <= 0
                    or process.start_ticks <= 0
                    or (item.process is not None and item.process != process)
                ):
                    raise PipelineError(
                        "RESOURCE_IDENTITY_MISMATCH",
                        "Observed process identity changed",
                    )
            devices = item.gpu_uuids if gpu_uuids is None else tuple(gpu_uuids)
            if (
                len(devices) != len(set(devices))
                or any(not isinstance(device, str) or not device for device in devices)
                or (item.gpu_uuids and devices != item.gpu_uuids)
            ):
                raise PipelineError(
                    "RESOURCE_IDENTITY_MISMATCH", "Observed device identity changed"
                )
            for other in self._resources.values():
                if (
                    other.allocation_id != allocation_id
                    and other.node_id == item.node_id
                    and set(other.gpu_uuids) & set(devices)
                ):
                    raise PipelineError(
                        "RESOURCE_CONFLICT",
                        "Observed GPU already belongs to another allocation",
                    )
            self._resources[allocation_id] = replace(
                item, process=process or item.process, gpu_uuids=devices
            )

    @property
    def cleanup_complete(self):
        with self._lock:
            return all(x.release_state == "released" for x in self._resources.values())

    def to_dict(self):
        with self._lock:
            return {
                "schema_version": 3,
                "run_id": self.run_id,
                "plan_hash": self.plan_hash,
                "resources": [asdict(x) for x in self._resources.values()],
            }


class PlacementAllocator:
    """Own exact PG handles; register creation before any readiness wait."""

    def __init__(self, plan, *, registry, backend):
        self.plan, self.registry, self.backend = plan, registry, backend
        self.handles = {}
        self.errors = []

    def allocate(self, *, deadline, cleanup_timeout):
        from .runtime import Deadline

        if self.handles:
            raise PipelineError("ALLOCATION_REUSED", "Allocation may only start once")
        try:
            for spec in self.plan["placement_groups"]:
                deadline.remaining()
                handle = self.backend.create(
                    spec, name=f"{self.plan['run_id']}-{spec['id']}"
                )
                self.handles[spec["id"]] = handle
                nodes = {b["node_id"] for b in spec["bundles"]}
                self.registry.register(
                    Allocation(
                        allocation_id=spec["id"],
                        run_id=self.plan["run_id"],
                        plan_hash=self.plan["plan_hash"],
                        owner="DeepSpec",
                        kind="placement_group",
                        role=spec["id"].split("-")[0],
                        ray_id=self.backend.identity(handle),
                        node_id=next(iter(nodes)) if len(nodes) == 1 else "multi-node",
                        borrower=spec["borrower"],
                    )
                )
                self.registry.transition(spec["id"], "acquiring")
            for name, handle in self.handles.items():
                self.backend.ready(handle, timeout=deadline.remaining())
                self.registry.transition(name, "acquired")
            return dict(self.handles)
        except BaseException:
            self.stop(deadline=Deadline.after(cleanup_timeout))
            raise

    def stop(self, *, deadline):
        resources = {r.allocation_id: r for r in self.registry.owned_resources()}
        for name, handle in reversed(tuple(self.handles.items())):
            record = resources[name]
            if record.release_state in ("released", "unknown"):
                continue
            self.registry.transition(name, "releasing")
            state = "unknown"
            try:
                deadline.remaining()
                self.backend.remove(handle)
                if self.backend.removed_state(handle, timeout=deadline.remaining()):
                    state = "released"
            except Exception as error:  # noqa: BLE001 -- preserve all cleanup diagnostics and continue exact IDs
                self.errors.append({"resource_id": name, "error": str(error)})
            self.registry.transition(name, state)
        return {
            "cleanup_complete": self.registry.cleanup_complete,
            "errors": list(self.errors),
        }


class RayPlacementBackend:
    def create(self, spec, *, name):
        from ray.util.placement_group import placement_group

        return placement_group(
            [b["resources"] for b in spec["bundles"]], strategy="PACK", name=name
        )

    def identity(self, handle):
        return handle.id.hex()

    def ready(self, handle, *, timeout):
        import ray

        ray.get(handle.ready(), timeout=timeout)

    def remove(self, handle):
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(handle)

    def removed_state(self, handle, *, timeout):
        import time

        from ray.util.placement_group import placement_group_table

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = placement_group_table(handle)
            if record and record.get("state") == "REMOVED":
                return True
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        return False


class ActorAllocator:
    """Register an actor's exact ID before waiting for its constructor or RPCs."""

    def __init__(
        self,
        plan,
        *,
        registry,
        backend=None,
        name_prefix=None,
        allocation_path=None,
        namespace=None,
    ):
        self.plan, self.registry = plan, registry
        self.backend = RayActorBackend() if backend is None else backend
        self.handles, self.errors = {}, []
        self.name_prefix = name_prefix or plan["run_id"]
        self.namespace = namespace or f"deepspec-{plan['run_id']}"
        self.allocation_path = allocation_path

    def _persist(self):
        from pathlib import Path

        from .runtime import atomic_json

        output = Path(self.plan["config"]["output_dir"])
        if output.is_dir():
            with self.registry._lock:
                atomic_json(
                    self.allocation_path or output / "allocation.json",
                    self.registry.to_dict(),
                )

    def create(
        self,
        allocation_id,
        cls,
        *,
        node_id,
        role,
        deadline,
        args=(),
        kwargs=None,
        options=None,
        borrower=None,
    ):
        if allocation_id in self.handles:
            raise PipelineError(
                "ALLOCATION_REUSED", "Actor allocation may only start once"
            )
        deadline.remaining()
        actor_options = {"num_cpus": 1, "num_gpus": 0, **(options or {})}
        if actor_options.get("lifetime") == "detached" and role != "node_agent":
            raise PipelineError(
                "RESOURCE_OWNER_INVALID",
                "Only leased NodeAgents may outlive the driver",
            )
        actor_options.update(
            name=f"{self.name_prefix}-{allocation_id}",
            max_restarts=0,
            max_task_retries=0,
        )
        actor_options.setdefault("namespace", self.namespace)
        actor = self.backend.create(
            cls, args=args, kwargs=kwargs or {}, options=actor_options, node_id=node_id
        )
        self.handles[allocation_id] = actor
        self.registry.register(
            Allocation(
                allocation_id=allocation_id,
                run_id=self.plan["run_id"],
                plan_hash=self.plan["plan_hash"],
                owner="DeepSpec",
                kind="actor",
                role=role,
                ray_id=self.backend.identity(actor),
                node_id=node_id,
                borrower=borrower,
            )
        )
        self.registry.transition(allocation_id, "acquiring")
        self._persist()
        return actor

    def resolve(self, allocation_id, ref, *, deadline):
        result = self.backend.get(ref, timeout=deadline.remaining())
        self.registry.transition(allocation_id, "acquired")
        self._persist()
        return result

    def stop(self, *, deadline, allocation_ids=None):
        selected = (
            tuple(self.handles) if allocation_ids is None else tuple(allocation_ids)
        )
        resources = {r.allocation_id: r for r in self.registry.owned_resources()}
        for name in reversed(selected):
            if resources[name].release_state in ("released", "unknown"):
                continue
            self.registry.transition(name, "releasing")
            state = "unknown"
            try:
                deadline.remaining()
                self.backend.kill(self.handles[name])
                if self.backend.dead(self.handles[name], timeout=deadline.remaining()):
                    state = "released"
            except Exception as error:  # noqa: BLE001 -- preserve every exact-ID cleanup failure
                self.errors.append({"resource_id": name, "error": str(error)})
            self.registry.transition(name, state)
            self._persist()
        resources = {r.allocation_id: r for r in self.registry.owned_resources()}
        return {
            "cleanup_complete": all(
                resources[n].release_state == "released" for n in selected
            ),
            "errors": list(self.errors),
        }


class RayActorBackend:
    def create(self, cls, *, args, kwargs, options, node_id):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        options = dict(options)
        if getattr(cls, "owns_supervised_processes", False):
            from deepspec.orchestration.process import supervised_actor_runtime_env

            options["runtime_env"] = supervised_actor_runtime_env(
                options.get("runtime_env")
            )
        options.setdefault(
            "scheduling_strategy", NodeAffinitySchedulingStrategy(node_id, soft=False)
        )
        return ray.remote(cls).options(**options).remote(*args, **kwargs)

    def identity(self, actor):
        return actor._actor_id.hex()

    def get(self, ref, *, timeout):
        import ray

        return ray.get(ref, timeout=timeout)

    def kill(self, actor):
        import ray

        ray.kill(actor, no_restart=True)

    def dead(self, actor, *, timeout):
        import time

        from ray._private.state import actors

        from .runtime import Deadline

        deadline = Deadline.after(timeout)
        while True:
            # Native GCS calls run only in the independently supervised driver.
            record = actors(self.identity(actor))
            if record and record.get("State") == "DEAD":
                return True
            time.sleep(min(0.05, deadline.remaining()))


class NodeAgents:
    """Independent leased actors, with heartbeat work outside model execution."""

    def __init__(self, plan, actors, *, report_dir=None):
        import threading
        import uuid

        self.plan, self.actors, self.report_dir = plan, actors, report_dir
        self.nodes = {n["node_id"]: n for n in plan["nodes"].values()}
        self.tokens = {node: uuid.uuid4().hex for node in self.nodes}
        self.handles = {}
        self.agent_epochs = {}
        self.sequence = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self.cleanup_result = None

    def start(self, *, deadline):
        import threading

        from .cluster import NodeAgent

        for node_id in self.nodes:
            self.handles[node_id] = self.actors.create(
                f"node-agent-{node_id}",
                NodeAgent,
                node_id=node_id,
                role="node_agent",
                deadline=deadline,
                args=(self.plan, node_id, self.tokens[node_id]),
                kwargs={
                    "exit_on_orphan": True,
                    "report_dir": self.report_dir,
                    "sample_resources": self.plan.get("evidence_capture", False),
                },
                options={
                    "lifetime": "detached",
                    "runtime_env": {
                        "env_vars": {
                            "DEEPSPEC_PIPELINE_RUN_ID": self.plan["run_id"],
                            "CUDA_VISIBLE_DEVICES": "",
                        }
                    },
                },
            )
        self._heartbeat(deadline, initial=True)
        self._thread = threading.Thread(
            target=self._watch, name="node-agent-heartbeats", daemon=True
        )
        self._thread.start()

    def _request(self, node_id, **payload):
        from .runtime import message_envelope

        return message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "controller"},
            fencing_token=self.tokens[node_id],
            **payload,
        )

    def _heartbeat(self, deadline, *, initial=False):
        from .runtime import validate_message

        self.sequence += 1
        replies = {
            node: actor.heartbeat.remote(self._request(node, sequence=self.sequence))
            for node, actor in self.handles.items()
        }
        for node, ref in replies.items():
            if initial:
                reply = self.actors.resolve(
                    f"node-agent-{node}", ref, deadline=deadline
                )
            else:
                reply = self.actors.backend.get(ref, timeout=deadline.remaining())
            validate_message(
                reply, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
            )
            if not reply["accepted"] or reply["node_id"] != node:
                raise RuntimeError(
                    "Node lease expired or heartbeat identity mismatched"
                )
            if (
                node in self.agent_epochs
                and self.agent_epochs[node] != reply["agent_epoch"]
            ):
                raise RuntimeError("Node agent epoch changed within one execution")
            self.agent_epochs[node] = reply["agent_epoch"]

    def _watch(self):
        from .runtime import Deadline

        policy = self.plan["timeouts_seconds"]
        while not self._stop.wait(policy["heartbeat"]):
            try:
                self._heartbeat(
                    Deadline.after(min(policy["transfer"], policy["heartbeat"]))
                )
            except Exception as error:  # noqa: BLE001 -- one failed lease makes the run unhealthy
                self.error = self.error or error
                return

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Node heartbeat failed: {self.error}") from self.error

    def register_process(self, identity, *, deadline):
        from .runtime import validate_message

        self.check()
        validate_message(
            identity, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )
        node = identity["node_id"]
        request = self._request(
            node,
            process=identity["process"],
            supervisor_report_path=identity.get("supervisor_report_path"),
        )
        response = self.actors.backend.get(
            self.handles[node].register_process.remote(request),
            timeout=deadline.remaining(),
        )
        validate_message(
            response, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )
        return response

    def stop(self, *, deadline):
        import time

        if self.cleanup_result is not None:
            return self.cleanup_result
        self._stop.set()
        errors, reports = [], {}
        if self._thread is not None:
            self._thread.join(timeout=max(0, deadline.expires_at - time.monotonic()))
            if self._thread.is_alive():
                errors.append("heartbeat worker did not stop before cleanup deadline")
        refs = {}
        for node, actor in self.handles.items():
            try:
                refs[node] = actor.cleanup.remote(timeout=deadline.remaining())
            except Exception as error:  # noqa: BLE001 -- continue reachable nodes under the same deadline
                errors.append(f"{node}: {error}")
        for node, ref in refs.items():
            try:
                reports[node] = self.actors.backend.get(
                    ref, timeout=deadline.remaining()
                )
            except Exception as error:  # noqa: BLE001 -- unreachable is unknown, never clean
                errors.append(f"{node}: {error}")
        # Keep unreachable leased agents alive: they must attempt node-local cleanup
        # independently after driver loss, then exit themselves.
        confirmed = [
            f"node-agent-{node}"
            for node, report in reports.items()
            if report["cleanup_complete"]
        ]
        for node in self.handles:
            name = f"node-agent-{node}"
            if name not in confirmed:
                self.actors.registry.transition(name, "releasing")
                self.actors.registry.transition(name, "unknown")
        actor_cleanup = self.actors.stop(deadline=deadline, allocation_ids=confirmed)
        self.cleanup_result = {
            "cleanup_complete": not errors
            and len(confirmed) == len(self.handles)
            and actor_cleanup["cleanup_complete"],
            "nodes": reports,
            "errors": errors,
        }
        return self.cleanup_result


class AllocationGate:
    """CPU coordination only: no model code and no membership inferred from reports."""

    def __init__(self, plan, *, pg_ids):
        import asyncio

        from .planning import TopologyPlan

        self.plan = TopologyPlan.from_dict(plan).to_dict()
        self.pg_ids = dict(pg_ids)
        if set(pg_ids) != {p["id"] for p in plan["placement_groups"]}:
            raise PipelineError(
                "GATE_PG_MISSING", "Gate requires the complete owned PG set"
            )
        self.changed = asyncio.Condition()
        self.error = None
        self.allocations, self.initialized = {}, {}
        self.expected = {}
        for replica in plan["replicas"]:
            for rank, node in enumerate(replica["worker_nodes"]):
                self.expected[f"inference/{replica['replica_id']}/{rank}"] = {
                    "node_id": node,
                    "pg_id": pg_ids[f"inference-{replica['replica_id']}"],
                    "bundle_index": rank,
                    "gpu_count": 1,
                    "role": "inference",
                }
        for rank in plan["training_ranks"]:
            if rank["local_rank"] != 0:
                continue
            self.expected[f"training/{rank['node_rank']}"] = {
                "node_id": rank["node_id"],
                "pg_id": pg_ids[f"training-{rank['node_rank']}"],
                "bundle_index": 0,
                "gpu_count": rank["local_world_size"],
                "role": "training",
            }
        self.nodes = {n["node_id"]: n for n in plan["nodes"].values()}
        self.expected_initialized = {
            key for key in self.expected if key.startswith("inference/")
        }
        self.expected_initialized.update(
            f"rank/{r['global_rank']}" for r in plan["training_ranks"]
        )
        self.expected_initialized.add("store")

    def _check(self):
        if self.error is not None:
            raise self.error

    def _reply(self, **payload):
        from .runtime import message_envelope

        return message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "allocation_gate"},
            **payload,
        )

    def _fail(self, error):
        if self.error is None:
            self.error = (
                error
                if isinstance(error, PipelineError)
                else PipelineError("GATE_FAILED", str(error))
            )
        self.changed.notify_all()

    async def abort(self, reason):
        async with self.changed:
            self._fail(PipelineError("GATE_ABORTED", reason))

    async def report_allocation(self, message):
        from .runtime import validate_message

        async with self.changed:
            self._check()
            try:
                validate_message(
                    message,
                    run_id=self.plan["run_id"],
                    plan_hash=self.plan["plan_hash"],
                )
                key = message["participant"]
                expected = self.expected.get(key)
                if expected is None or any(
                    message.get(k) != expected[k]
                    for k in ("node_id", "pg_id", "bundle_index")
                ):
                    raise PipelineError(
                        "ALLOCATION_MISMATCH",
                        "Unexpected node, PG, bundle or participant",
                    )
                gpus = message.get("gpu_uuids", [])
                available = {g["uuid"] for g in self.nodes[expected["node_id"]]["gpus"]}
                alias = next(
                    a
                    for a, node in self.plan["nodes"].items()
                    if node["node_id"] == expected["node_id"]
                )
                role_node = next(
                    n
                    for n in self.plan["config"][expected["role"]]["nodes"]
                    if n["node"] == alias
                )
                allowed = set(role_node.get("allowed_gpu_uuids", available))
                if (
                    len(gpus) != expected["gpu_count"]
                    or len(set(gpus)) != len(gpus)
                    or set(gpus) - available
                    or set(gpus) - allowed
                ):
                    raise PipelineError(
                        "ALLOCATION_GPU_MISMATCH",
                        "Actual GPU UUIDs differ from the declared role",
                    )
                if not message.get("actor_id") or any(
                    type(message.get(k)) is not int or message[k] <= 0
                    for k in ("pid", "start_ticks")
                ):
                    raise PipelineError(
                        "ALLOCATION_PROCESS_MISSING",
                        "Actor and PID/start-time identity are required",
                    )
                payload = {k: v for k, v in message.items() if k not in ("event_id",)}
                previous = self.allocations.get(key)
                if previous is not None:
                    if previous != payload:
                        raise PipelineError(
                            "ALLOCATION_CONFLICT", "Conflicting allocation report"
                        )
                    return self._reply(accepted=True)
                for item in self.allocations.values():
                    if item["actor_id"] == message["actor_id"] or (
                        item["node_id"] == message["node_id"]
                        and set(item["gpu_uuids"]) & set(gpus)
                    ):
                        raise PipelineError(
                            "ALLOCATION_GPU_CONFLICT",
                            "GPU or actor identity reported by multiple roles",
                        )
                self.allocations[key] = payload
                self.changed.notify_all()
                return self._reply(accepted=True)
            except Exception as error:
                self._fail(error)
                raise self.error from error

    async def report_initialized(self, message):
        from .runtime import validate_message

        async with self.changed:
            self._check()
            try:
                validate_message(
                    message,
                    run_id=self.plan["run_id"],
                    plan_hash=self.plan["plan_hash"],
                )
                if set(self.allocations) != set(self.expected):
                    raise PipelineError(
                        "INITIALIZATION_EARLY",
                        "All allocations must arrive before GPU initialization",
                    )
                key = message["participant"]
                if key not in self.expected_initialized:
                    raise PipelineError(
                        "INITIALIZATION_UNKNOWN", "Unexpected initialized participant"
                    )
                if key.startswith("rank/"):
                    rank = self.plan["training_ranks"][int(key.split("/")[1])]
                    fields = (
                        "node_id",
                        "global_rank",
                        "local_rank",
                        "node_rank",
                        "tp_rank",
                        "dp_rank",
                    )
                    if (
                        any(message.get(k) != rank[k] for k in fields)
                        or message.get("world") != len(self.plan["training_ranks"])
                        or message.get("gas") != self.plan["counts"]["gas"]
                    ):
                        raise PipelineError(
                            "TRAINING_IDENTITY_MISMATCH",
                            "Native rank/world/GAS differs from plan",
                        )
                    if (
                        message.get("input_plan_hash") != self.plan["input_plan_hash"]
                        or message.get("model_identity")
                        != self.nodes[rank["node_id"]]["identities"]["model"]
                    ):
                        raise PipelineError(
                            "TRAINING_INPUT_MISMATCH",
                            "Native model/input identity differs from plan",
                        )
                elif key == "store":
                    if (
                        message.get("node_id") != self.plan["services"]["pool_node_id"]
                        or message.get("transport_verified") is not True
                    ):
                        raise PipelineError(
                            "STORE_NOT_READY",
                            "Store and all-node transport must be verified",
                        )
                else:
                    allocation = self.allocations[key]
                    replica, tp = map(int, key.split("/")[1:])
                    if (
                        message.get("node_id") != allocation["node_id"]
                        or message.get("actor_id") != allocation["actor_id"]
                        or message.get("dp_rank") != replica
                        or message.get("tp_rank") != tp
                    ):
                        raise PipelineError(
                            "INFERENCE_IDENTITY_MISMATCH",
                            "Native inference rank identity differs from allocation",
                        )
                payload = {k: v for k, v in message.items() if k != "event_id"}
                if key in self.initialized and self.initialized[key] != payload:
                    raise PipelineError(
                        "INITIALIZATION_CONFLICT", "Conflicting initialized identity"
                    )
                self.initialized[key] = payload
                self.changed.notify_all()
                return self._reply(accepted=True)
            except Exception as error:
                self._fail(error)
                raise self.error from error

    async def _wait(self, deadline, *, initialization):
        import asyncio

        async with self.changed:
            expected = (
                self.expected_initialized if initialization else set(self.expected)
            )
            reports = self.initialized if initialization else self.allocations
            try:
                await asyncio.wait_for(
                    self.changed.wait_for(
                        lambda: self.error is not None or set(reports) == expected
                    ),
                    deadline.remaining(),
                )
                self._check()
                return self._reply(ready=True, participants=sorted(reports))
            except TimeoutError as error:
                self._fail(
                    PipelineError(
                        "GATE_TIMEOUT",
                        "Missing participant reports before shared deadline",
                    )
                )
                raise TimeoutError("Gate shared deadline expired") from error

    async def wait_allocation(self, deadline):
        return await self._wait(deadline, initialization=False)

    async def wait_initialized(self, deadline):
        return await self._wait(deadline, initialization=True)
