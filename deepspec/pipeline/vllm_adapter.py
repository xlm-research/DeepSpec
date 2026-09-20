"""Native vLLM placement, process ownership and all-role startup coordination.

Ray handles are runtime arguments only. The persisted ledger contains exact
resource identities; it never serializes a handle or derives a GPU from DP rank.
"""

import asyncio
import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .controller import Allocation, AllocationGate, ProcessIdentity, ResourceRegistry
from .runtime import (
    Deadline,
    EventWriter,
    PipelineError,
    atomic_json,
    message_envelope,
    validate_message,
)


def observed_bundle(resource_ids, pg_id):
    """Read the indexed PG resource actually assigned to this Ray worker."""
    pattern = re.compile(r".+_group_(\d+)_" + re.escape(pg_id) + r"$")
    indices = {
        int(match.group(1))
        for name, allocations in resource_ids.items()
        if (match := pattern.fullmatch(name))
        and any(float(amount) > 0 for _, amount in allocations)
    }
    if len(indices) != 1:
        raise PipelineError(
            "NATIVE_BUNDLE_UNKNOWN", "Exactly one actual indexed PG bundle is required"
        )
    return indices.pop()


def _stable_observation(message):
    """Process scheduler state can change between deliveries of one identity."""
    result = copy.deepcopy({k: v for k, v in message.items() if k != "event_id"})
    if "process" in result:
        result["process"].pop("state", None)
    return result


class NativeCoordination(AllocationGate):
    """CPU actor with an async gate and one writer for native resource evidence."""

    def __init__(self, plan, *, pg_ids, node_agents, tokens):
        super().__init__(plan, pg_ids=pg_ids)
        if set(node_agents) != set(self.nodes) or set(tokens) != set(self.nodes):
            raise ValueError("Every planned node needs a leased NodeAgent")
        self.node_agents, self.tokens = dict(node_agents), dict(tokens)
        self.native_registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
        self.handles, self.native_records, self.observations = {}, {}, {}
        self.rank_processes = {}
        self.inference_initialized, self.connectors = {}, {}
        self.observation_lock = asyncio.Lock()
        self.accepting_native = True
        self.output = Path(plan["config"]["output_dir"])
        self.events = EventWriter(
            self.output / "events/native-gate.jsonl",
            run_id=plan["run_id"],
            plan_hash=plan["plan_hash"],
            sender_identity={"component": "native_gate"},
        )

    def _validate(self, message):
        validate_message(
            message, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )

    def _persist_native(self):
        atomic_json(
            self.output / "native-allocation.json",
            {
                **self.native_registry.to_dict(),
                "placements": self.native_records,
                "training_rank_processes": self.rank_processes,
                "initialized": self.initialized,
                "connectors": self.connectors,
            },
        )

    def _native_expected(self, participant):
        if participant.startswith("inference/") and participant in self.expected:
            return self.expected[participant]
        cores = {
            f"core/{r['replica_id']}": {
                "node_id": r["core_node"],
                "pg_id": self.pg_ids[f"inference-{r['replica_id']}"],
                "bundle_index": r["core_cpu_bundle"],
                "gpu_count": 0,
                "role": "core",
            }
            for r in self.plan["replicas"]
        }
        if participant in cores:
            return cores[participant]
        raise PipelineError("NATIVE_PARTICIPANT_UNKNOWN", "Unplanned native actor")

    def _record_native(self, message, actor=None):
        self._validate(message)
        if not self.accepting_native:
            raise PipelineError("NATIVE_STOPPED", "Native allocation is stopped")
        key = message["participant"]
        expected = self._native_expected(key)
        fields = ("actor_id", "node_id", "pg_id", "bundle_index")
        record = {k: message[k] for k in fields}
        if not record["actor_id"] or record["node_id"] not in self.nodes:
            raise ValueError("Native actor requires an exact ID and participating node")
        if actor is not None and actor._actor_id.hex() != record["actor_id"]:
            raise ValueError("Native handle differs from its declared actor ID")
        previous = self.native_records.get(key)
        if previous is None:
            self.native_registry.register(
                Allocation(
                    allocation_id=key,
                    run_id=self.plan["run_id"],
                    plan_hash=self.plan["plan_hash"],
                    owner="DeepSpec",
                    borrower="vLLM",
                    kind="native_actor",
                    role=expected["role"],
                    ray_id=record["actor_id"],
                    node_id=record["node_id"],
                    bundle_index=record["bundle_index"],
                )
            )
            self.native_registry.transition(key, "acquiring")
            self.native_records[key] = record
        elif previous != record:
            raise PipelineError(
                "NATIVE_IDENTITY_CHANGED", "Native actor identity changed"
            )
        if actor is not None:
            self.handles[key] = actor
        # Persist ownership before placement validation, including a wrong slot.
        self._persist_native()
        if any(record[k] != expected[k] for k in ("node_id", "pg_id", "bundle_index")):
            raise PipelineError("NATIVE_PLACEMENT_MISMATCH", "Native placement differs")
        return key

    async def _abort_error(self, error):
        async with self.changed:
            self._fail(error)

    async def register_native(self, message, actor):
        """Creation can be reported before or after the child's self-report."""
        try:
            self._record_native(message, actor)
            self._check()
            return self._reply(accepted=True)
        except Exception as error:
            await self._abort_error(error)
            raise

    async def _register_process(self, message, deadline):
        node = message["node_id"]
        process = message["process"]
        if node not in self.node_agents:
            raise ValueError("Process is outside participating nodes")
        request = message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "native_gate"},
            process=process,
            fencing_token=self.tokens[node],
            supervisor_report_path=message.get("supervisor_report_path"),
        )
        reply = await asyncio.wait_for(
            self.node_agents[node].register_process.remote(request),
            deadline.remaining(),
        )
        self._validate(reply)
        fields = ("pid", "start_ticks", "run_id", "parent_pid", "group_id")
        if any(reply["process"].get(k) != process.get(k) for k in fields):
            raise ValueError("NodeAgent did not confirm the native process identity")
        return ProcessIdentity(node_id=node, **{k: process[k] for k in fields})

    async def _check_devices(self, message, deadline):
        devices = message.get("gpu_uuids", [])
        if not devices:
            return
        request = message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "native_gate"},
            fencing_token=self.tokens[message["node_id"]],
            gpu_uuids=devices,
        )
        reply = await asyncio.wait_for(
            self.node_agents[message["node_id"]].check_allocated_devices.remote(
                request
            ),
            deadline.remaining(),
        )
        self._validate(reply)
        if (
            reply.get("node_id") != message["node_id"]
            or sorted(reply.get("gpu_uuids", [])) != sorted(devices)
            or reply.get("external_processes") != []
        ):
            raise ValueError("NodeAgent did not confirm exclusive allocated GPU use")

    async def observe_native(self, message, *, timeout):
        """Fence the real process before allowing any worker to initialize CUDA."""
        deadline = Deadline.after(timeout)
        try:
            self._validate(message)
            async with self.observation_lock:
                self._check()
                key = message["participant"]
                payload = _stable_observation(message)
                if key in self.observations:
                    if self.observations[key] != payload:
                        raise ValueError("Native observed identity changed")
                    return self._reply(accepted=True)
                process = await self._register_process(message, deadline)
                self._record_native(message)
                if (message.get("pid"), message.get("start_ticks")) != (
                    process.pid,
                    process.start_ticks,
                ):
                    raise ValueError(
                        "Native report differs from registered PID/start-time"
                    )
                self.native_registry.observe(
                    key, process=process, gpu_uuids=message.get("gpu_uuids", [])
                )
                self._persist_native()
                if key.startswith("inference/"):
                    await self._check_devices(message, deadline)
                    await self.report_allocation(message)
                elif message.get("gpu_uuids"):
                    raise ValueError("A CPU core cannot own GPU devices")
                self.native_registry.transition(key, "acquired")
                self.observations[key] = payload
                self._persist_native()
                self.events.emit(
                    "node_environment",
                    {
                        "identities": {"process": message["process"]},
                        "placement": {
                            k: message[k]
                            for k in (
                                "participant",
                                "actor_id",
                                "node_id",
                                "pg_id",
                                "bundle_index",
                                "gpu_uuids",
                            )
                        },
                    },
                    basis="observed",
                    causes=(message["event_id"],),
                )
                return self._reply(accepted=True)
        except Exception as error:
            await self._abort_error(error)
            raise

    async def wait_for_allocation(self, *, timeout):
        return await self.wait_allocation(Deadline.after(timeout))

    async def wait_for_initialization(self, *, timeout):
        return await self.wait_initialized(Deadline.after(timeout))

    async def observe_launcher(self, message, *, timeout):
        """Launchers own their GPU bundle before torchrun is allowed to start."""
        deadline = Deadline.after(timeout)
        try:
            self._validate(message)
            self._check()
            if not message["participant"].startswith("training/"):
                raise ValueError("Expected a planned training launcher")
            process = await self._register_process(message, deadline)
            if (message.get("pid"), message.get("start_ticks")) != (
                process.pid,
                process.start_ticks,
            ):
                raise ValueError("Training launcher process identity differs")
            await self._check_devices(message, deadline)
            return await self.report_allocation(message)
        except Exception as error:
            await self._abort_error(error)
            raise

    async def register_training_process(self, message, *, timeout):
        """Capture torchrun supervisor and rank ownership before native GPU init."""
        deadline = Deadline.after(timeout)
        try:
            self._validate(message)
            self._check()
            if set(self.allocations) != set(self.expected):
                raise ValueError(
                    "Training processes cannot precede all-role allocation"
                )
            key = message["participant"]
            if key.startswith("rank/"):
                rank = self.plan["training_ranks"][int(key.split("/")[1])]
                if key != f"rank/{rank['global_rank']}" or (
                    message.get("global_rank") != rank["global_rank"]
                    or any(
                        message.get(k) != rank[k]
                        for k in (
                            "node_id",
                            "node_rank",
                            "local_rank",
                            "local_world_size",
                        )
                    )
                    or message.get("world") != len(self.plan["training_ranks"])
                ):
                    raise ValueError("Torchrun rank topology differs from the plan")
                previous = self.rank_processes.get(key)
                stable = _stable_observation(message)
                if previous is not None and previous != stable:
                    raise ValueError("Native rank process identity changed")
                await self._register_process(message, deadline)
                self.rank_processes[key] = stable
                self._persist_native()
            elif key.startswith("training/"):
                launcher = self.allocations.get(key)
                if launcher is None or any(
                    message.get(k) != launcher[k] for k in ("node_id", "actor_id")
                ):
                    raise ValueError(
                        "Supervisor belongs to an unknown training launcher"
                    )
                await self._register_process(message, deadline)
            else:
                raise ValueError("Unexpected training process")
            return self._reply(accepted=True)
        except Exception as error:
            await self._abort_error(error)
            raise

    async def report_initialized(self, message):
        if message.get("participant", "").startswith("inference/"):
            try:
                self._validate(message)
                self._check()
                key = message["participant"]
                allocation = self.allocations.get(key)
                if allocation is None or set(self.allocations) != set(self.expected):
                    raise ValueError("Inference initialization preceded allocation")
                replica, tp = map(int, key.split("/")[1:])
                if (
                    message.get("actor_id") != allocation["actor_id"]
                    or message.get("node_id") != allocation["node_id"]
                    or message.get("dp_rank") != replica
                    or message.get("tp_rank") != tp
                ):
                    raise ValueError("Native inference rank differs from allocation")
                previous = self.inference_initialized.get(key)
                stable = {k: v for k, v in message.items() if k != "event_id"}
                if previous is not None and previous != stable:
                    raise ValueError("Native inference initialization identity changed")
                self.inference_initialized[key] = stable
                if key not in self.connectors:
                    return self._reply(accepted=True, connector_pending=True)
            except Exception as error:
                await self._abort_error(error)
                raise
        if message.get("participant", "").startswith("rank/"):
            try:
                self._validate(message)
                started = self.rank_processes.get(message["participant"])
                if started is None or any(
                    message.get(k) != started["process"][k]
                    for k in ("pid", "start_ticks")
                ):
                    raise ValueError(
                        "Initialized rank must match its pre-GPU process registration"
                    )
                rank = self.plan["training_ranks"][message["global_rank"]]
                launcher = self.allocations[f"training/{rank['node_rank']}"]
                if message.get("gpu_uuid") != launcher["gpu_uuids"][rank["local_rank"]]:
                    raise ValueError(
                        "Rank GPU differs from its allocated launcher slot"
                    )
                expected_tp = [
                    r["global_rank"]
                    for r in self.plan["training_ranks"]
                    if r["dp_rank"] == rank["dp_rank"]
                ]
                expected_dp = [
                    r["global_rank"]
                    for r in self.plan["training_ranks"]
                    if r["tp_rank"] == rank["tp_rank"]
                ]
                if (
                    message.get("tp_members") != expected_tp
                    or message.get("dp_members") != expected_dp
                ):
                    raise ValueError(
                        "Native TP/FSDP group members differ from the plan"
                    )
            except Exception as error:
                await self._abort_error(error)
                raise
        result = await super().report_initialized(message)
        self._persist_native()
        return result

    async def report_connector(self, message, *, ready=True):
        """A rank is ready only after native init and its real connector setup."""
        try:
            self._validate(message)
            self._check()
            key = message["participant"]
            allocation = self.allocations.get(key)
            if (
                allocation is None
                or not key.startswith("inference/")
                or set(self.allocations) != set(self.expected)
            ):
                raise ValueError("Connector has no allocated native worker")
            replica, tp = map(int, key.split("/")[1:])
            if (
                any(
                    message.get(k) != allocation[k]
                    for k in ("actor_id", "node_id", "gpu_uuids", "pid", "start_ticks")
                )
                or message.get("tp_rank") != tp
                or message.get("dp_rank") != replica
                or message.get("tp_world_size")
                != self.plan["config"]["inference"]["tp"]
                or message.get("writer") is not (tp == 0)
            ):
                raise ValueError(
                    "Connector GPU, process or TP/DP writer identity differs"
                )
            if (
                tp == 0
                and message["node_id"] != self.plan["replicas"][replica]["core_node"]
            ):
                raise ValueError("TP0 writer must share the planned core node")
            if not ready:
                return self._reply(accepted=True)
            stable = {k: v for k, v in message.items() if k != "event_id"}
            if key in self.connectors and self.connectors[key] != stable:
                raise ValueError("Connector identity changed")
            self.connectors[key] = stable
            if key in self.inference_initialized:
                result = await super().report_initialized(
                    message_envelope(
                        self.plan["run_id"],
                        self.plan["plan_hash"],
                        self.inference_initialized[key]["sender_identity"],
                        **{
                            k: v
                            for k, v in self.inference_initialized[key].items()
                            if k
                            not in (
                                "schema_version",
                                "run_id",
                                "plan_hash",
                                "sender_identity",
                            )
                        },
                    )
                )
                self._persist_native()
                return result
            self._persist_native()
            return self._reply(accepted=True, native_init_pending=True)
        except Exception as error:
            await self._abort_error(error)
            raise

    async def native_failed(self, message):
        self._validate(message)
        await self.abort(message["reason"])
        return self._reply(accepted=True)

    async def register_native_process(self, message, *, timeout):
        """Register native non-actor children with the same exact process fence."""
        try:
            self._validate(message)
            self._check()
            if not self.accepting_native or message["participant"] != "dp_coordinator":
                raise ValueError("Unexpected native child process")
            if message["node_id"] != self.plan["replicas"][0]["core_node"]:
                raise ValueError("Native coordinator must share the frontend node")
            process = await self._register_process(message, Deadline.after(timeout))
            return self._reply(
                accepted=True, pid=process.pid, start_ticks=process.start_ticks
            )
        except Exception as error:
            await self._abort_error(error)
            raise

    async def stop_native(self, reason, *, timeout):
        """Stop exact native actor IDs; PG release remains the controller's job."""
        from .controller import RayActorBackend

        deadline = Deadline.after(timeout)
        self.accepting_native = False
        await self.abort(reason)
        backend = RayActorBackend()
        errors = []
        resources = self.native_registry.owned_resources()
        for resource in reversed(resources):
            if resource.release_state in ("released", "unknown"):
                continue
            key = resource.allocation_id
            self.native_registry.transition(key, "releasing")
            actor = self.handles.get(key)
            try:
                deadline.remaining()
                if actor is None:
                    raise RuntimeError(
                        "Creator did not provide the exact native actor handle"
                    )
                backend.kill(actor)
            except Exception as error:  # noqa: BLE001 -- continue exact-ID cleanup and preserve unknown
                errors.append({"resource_id": key, "error": str(error)})
        self._persist_native()
        for resource in self.native_registry.owned_resources():
            if resource.release_state != "releasing":
                continue
            key, state = resource.allocation_id, "unknown"
            try:
                actor = self.handles[key]
                if await asyncio.wait_for(
                    asyncio.to_thread(
                        backend.dead, actor, timeout=deadline.remaining()
                    ),
                    deadline.remaining(),
                ):
                    state = "released"
            except Exception as error:  # noqa: BLE001 -- retain unconfirmed release evidence
                errors.append({"resource_id": key, "error": str(error)})
            self.native_registry.transition(key, state)
            self._persist_native()
            self.events.emit(
                "cleanup",
                {"resource_id": key, "release_state": state},
                basis="observed",
            )
        return self._reply(
            cleanup_complete=self.native_registry.cleanup_complete,
            errors=errors,
            resources=self.native_registry.to_dict()["resources"],
        )

    def snapshot(self):
        return self._reply(
            resources=self.native_registry.to_dict()["resources"],
            allocations=self.allocations,
            initialized=self.initialized,
            connectors=self.connectors,
            error=self.error.to_dict() if self.error else None,
        )

    def close_events(self):
        self.events.close()


@dataclass
class NativePlacementHooks:
    """Bound runtime callback used by the unmodified native engine hierarchy."""

    plan: dict
    gate: object

    def _message(self, **payload):
        return message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            {"component": "vllm_native", "pid": os.getpid()},
            **payload,
        )

    def _call(self, method, *args, deadline, **kwargs):
        import ray

        result = ray.get(
            getattr(self.gate, method).remote(*args, **kwargs),
            timeout=deadline.remaining(),
        )
        validate_message(
            result, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )
        return result

    def __call__(self, event, payload, *, timeout):
        from deepspec.orchestration.process import capture_process

        deadline = Deadline.after(timeout)
        if event in ("core_created", "worker_created"):
            values = dict(payload)
            actor = values.pop("actor")
            participant = (
                f"core/{values['replica']}"
                if event == "core_created"
                else f"inference/{values['replica']}/{values['rank']}"
            )
            return self._call(
                "register_native",
                self._message(
                    **values, participant=participant, actor_id=actor._actor_id.hex()
                ),
                actor,
                deadline=deadline,
            )
        if event in ("core_starting", "worker_allocated"):
            values = dict(payload)
            resources = values.pop("resource_ids")
            bundle = observed_bundle(resources, values["pg_id"])
            values["bundle_index"] = bundle
            is_core = event == "core_starting"
            participant = (
                f"core/{values['replica']}"
                if is_core
                else f"inference/{values['replica']}/{values['rank']}"
            )
            devices = []
            if not is_core:
                from .cluster import gpu_inventory

                inventory = {
                    str(g["index"]): g["uuid"]
                    for g in gpu_inventory(timeout=deadline.remaining())
                }
                devices = [inventory[str(i)] for i in values.pop("physical_gpu_ids")]
            process = capture_process(os.getpid(), self.plan["run_id"])
            return self._call(
                "observe_native",
                self._message(
                    **values,
                    participant=participant,
                    gpu_uuids=devices,
                    process=process,
                    pid=process["pid"],
                    start_ticks=process["start_ticks"],
                ),
                timeout=deadline.remaining(),
                deadline=deadline,
            )
        if event == "allocation_ready":
            return self._call(
                "wait_for_allocation", timeout=deadline.remaining(), deadline=deadline
            )
        if event == "worker_initialized":
            if payload["tp_world_size"] != self.plan["config"]["inference"]["tp"]:
                raise ValueError("Native TP group size differs from the immutable plan")
            values = dict(payload)
            values.update(
                participant=f"inference/{payload['replica']}/{payload['rank']}",
                dp_rank=payload["replica"],
            )
            return self._call(
                "report_initialized", self._message(**values), deadline=deadline
            )
        if event in ("connector_check", "connector_initialized"):
            from .cluster import gpu_inventory

            values = dict(payload)
            inventory = gpu_inventory(timeout=deadline.remaining())
            devices = {str(g["index"]): g["uuid"] for g in inventory}
            devices.update({g["uuid"]: g["uuid"] for g in inventory})
            gpu_uuids = [devices[str(i)] for i in values.pop("physical_gpu_ids")]
            process = capture_process(os.getpid(), self.plan["run_id"])
            return self._call(
                "report_connector",
                self._message(
                    **values,
                    participant=f"inference/{values['replica']}/{values['tp_rank']}",
                    dp_rank=values["replica"],
                    gpu_uuids=gpu_uuids,
                    pid=process["pid"],
                    start_ticks=process["start_ticks"],
                ),
                ready=event == "connector_initialized",
                deadline=deadline,
            )
        if event == "process_created":
            import ray

            process = capture_process(payload["pid"], self.plan["run_id"])
            return self._call(
                "register_native_process",
                self._message(
                    participant=payload["role"],
                    process=process,
                    node_id=ray.get_runtime_context().get_node_id(),
                ),
                timeout=deadline.remaining(),
                deadline=deadline,
            )
        if event == "failed":
            return self._call(
                "native_failed", self._message(reason=str(payload)), deadline=deadline
            )
        raise ValueError(f"Unknown native placement callback: {event}")


def validate_native_environment(plan, environment=None):
    """Reject placement overrides before config normalization or actor creation."""
    inference = plan["config"]["inference"]
    environment = os.environ if environment is None else environment
    node = next(
        n
        for n in plan["nodes"].values()
        if n["node_id"] == plan["replicas"][0]["core_node"]
    )
    values = {
        "VLLM_DP_SIZE": str(inference["dp"]),
        "VLLM_DP_RANK": "0",
        "VLLM_DP_MASTER_IP": node["ip"],
        "VLLM_RAY_BUNDLE_INDICES": ",".join(map(str, range(inference["tp"]))),
        "VLLM_RAY_PER_WORKER_GPUS": "1",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
    }
    for key, expected in values.items():
        actual = environment.get(key)
        if actual is not None and actual != expected:
            raise PipelineError(
                "NATIVE_ENV_CONFLICT", f"{key} conflicts with the frozen plan"
            )
    for key in ("VLLM_DP_RANK_LOCAL", "VLLM_RAY_DP_PLACEMENT_NODE_IPS"):
        if environment.get(key):
            raise PipelineError(
                "NATIVE_ENV_CONFLICT", f"{key} conflicts with borrowed placement"
            )
    if environment.get("VLLM_RAY_DP_PACK_STRATEGY", "pack") != "pack":
        raise PipelineError(
            "NATIVE_ENV_CONFLICT",
            "Automatic spanning conflicts with borrowed placement",
        )
    return node["ip"]


def bind_native_config(vllm_config, plan, groups, gate, *, timeout, runtime_env):
    """Bind only after native normalization and validate the effective topology."""
    deadline = Deadline.after(timeout)
    ip = validate_native_environment(plan)
    parallel = vllm_config.parallel_config
    inference = plan["config"]["inference"]
    if inference["dp"] == 1 and parallel.data_parallel_rank_local == 0:
        # The singleton fallback defaults to offline local rank zero even when
        # EngineArgs requests the online Ray backend. Explicit env ranks were
        # rejected above; cores receive their actual local rank from the plan.
        parallel.data_parallel_rank_local = None
    expected = {
        "tensor_parallel_size": inference["tp"],
        "pipeline_parallel_size": 1,
        "data_parallel_size": inference["dp"],
        "data_parallel_size_local": inference["dp"],
        "data_parallel_rank": 0,
        "data_parallel_rank_local": None,
        "distributed_executor_backend": "ray",
        "data_parallel_backend": "ray",
        "enable_elastic_ep": False,
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
    }
    for key, value in expected.items():
        if getattr(parallel, key, None) != value:
            raise PipelineError(
                "NATIVE_CONFIG_CONFLICT", f"Effective {key} differs from the plan"
            )
    if inference["dp"] == 1 and parallel.data_parallel_master_ip == "127.0.0.1":
        # ParallelConfig's singleton environment fallback ignores the explicit
        # EngineArgs address. Restore it only after rejecting every env override.
        parallel.data_parallel_master_ip = ip
    if parallel.data_parallel_master_ip != ip:
        raise PipelineError(
            "NATIVE_CONFIG_CONFLICT", "Effective DP address differs from the plan"
        )
    if not callable(getattr(parallel, "bind_ray_placement", None)):
        raise PipelineError(
            "NATIVE_CAPABILITY_MISSING", "Native borrowed placement API is missing"
        )
    groups = list(groups)
    if len(groups) != len(plan["replicas"]):
        raise ValueError("One controller-owned placement group is required per replica")
    replicas, local_ranks, per_node = [], [], {}
    for replica, group in zip(plan["replicas"], groups):
        node = replica["core_node"]
        local_ranks.append(per_node.get(node, 0))
        per_node[node] = per_node.get(node, 0) + 1
        replicas.append(
            {
                "placement_group_id": group.id.hex(),
                "worker_bundle_indices": list(range(inference["tp"])),
                "bundle_node_ids": [*replica["worker_nodes"], node],
                "core_bundle_index": replica["core_cpu_bundle"],
            }
        )
    native_plan = {
        "version": 1,
        "run_id": plan["run_id"],
        "plan_hash": plan["plan_hash"],
        "timeout_seconds": deadline.remaining(),
        "replicas": replicas,
    }
    explicit_env = copy.deepcopy(runtime_env)
    env_vars = explicit_env.setdefault("env_vars", {})
    validate_native_environment(plan, env_vars)
    if "CUDA_VISIBLE_DEVICES" in env_vars:
        raise ValueError(
            "Worker visibility must be derived from actual Ray GPU assignments"
        )
    env_vars.update(
        DEEPSPEC_PIPELINE_RUN_ID=plan["run_id"],
        VLLM_USE_RAY_V2_EXECUTOR_BACKEND="1",
        VLLM_RAY_BUNDLE_INDICES=",".join(map(str, range(inference["tp"]))),
    )
    parallel.ray_runtime_env = explicit_env
    parallel.bind_ray_placement(
        groups, local_ranks, native_plan, NativePlacementHooks(plan, gate)
    )
    return vllm_config


class NativeInferenceAdapter:
    """One AsyncLLM entry for DP1 and DP2, using vLLM's native Ray executor."""

    def __init__(self, plan, groups, gate, *, runtime_env):
        self.plan, self.groups, self.gate = plan, groups, gate
        self.runtime_env = runtime_env
        self.engine = None
        self.state, self.error = "allocated", None

    def start(self, model_args, *, timeout):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.executor.abstract import Executor
        from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

        if self.state != "allocated":
            raise RuntimeError("Native engine start is single-use")
        deadline = Deadline.after(timeout)
        self.state = "initializing"
        try:
            verify_native_capabilities()
            ip = validate_native_environment(self.plan)
            inference = self.plan["config"]["inference"]
            values = dict(model_args)
            required = {
                "model": self.plan["config"]["model_path"],
                "tensor_parallel_size": inference["tp"],
                "pipeline_parallel_size": 1,
                "data_parallel_size": inference["dp"],
                "data_parallel_size_local": inference["dp"],
                "distributed_executor_backend": "ray",
                "data_parallel_backend": "ray",
                "data_parallel_address": ip,
            }
            for key, expected in required.items():
                if key in values and values[key] != expected:
                    raise ValueError(
                        f"Engine argument {key} conflicts with the frozen plan"
                    )
                values[key] = expected
            config = AsyncEngineArgs(**values).create_engine_config()
            bind_native_config(
                config,
                self.plan,
                self.groups,
                self.gate,
                timeout=deadline.remaining(),
                runtime_env=self.runtime_env,
            )
            if Executor.get_class(config) is not RayExecutorV2:
                raise PipelineError(
                    "NATIVE_EXECUTOR_MISMATCH", "Native Ray V2 executor is required"
                )
            self.engine = AsyncLLM.from_vllm_config(config)
            deadline.remaining()
            self.state = "initialized"
            return self.engine
        except BaseException as error:
            self.state, self.error = "failed", repr(error)
            try:
                NativePlacementHooks(self.plan, self.gate)(
                    "failed", {"error": repr(error)}, timeout=0.1
                )
            except Exception as notification_error:  # noqa: BLE001 -- preserve the native startup failure
                error.add_note(f"Native gate notification failed: {notification_error}")
            raise

    def status(self):
        return {"state": self.state, "error": self.error}

    def ready(self, *, timeout):
        if self.state != "initialized":
            raise RuntimeError("Native engine initialization is incomplete")
        return NativePlacementHooks(self.plan, self.gate)._call(
            "wait_for_initialization", timeout=timeout, deadline=Deadline.after(timeout)
        )

    def stop(self, *, timeout):
        deadline = Deadline.after(timeout)
        if self.engine is not None:
            self.engine.shutdown(timeout=deadline.remaining())
            self.engine = None
        self.state = "stopped"


def verify_native_capabilities():
    """An installed version alone cannot enable borrowed native placement."""
    import importlib

    for name in (
        "vllm.config.parallel",
        "vllm.v1.engine.utils",
        "vllm.v1.engine.core",
        "vllm.v1.executor.ray_executor_v2",
    ):
        module = importlib.import_module(name)
        if getattr(module, "RAY_PLACEMENT_API_VERSION", None) != 1:
            raise PipelineError(
                "NATIVE_CAPABILITY_MISSING", f"Missing native placement seam: {name}"
            )
