"""Two-node TP4 producer/consumer launch on an existing Ray cluster."""

import hashlib
import importlib.metadata
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .memory import feature_budget, node_memory
from .topology import consumer_dp, producer_dp

logger = logging.getLogger(__name__)
PACKAGES = ("torch", "vllm", "ray", "mooncake-transfer-engine", "transformers", "numpy")


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def source_hashes(root):
    paths = [
        *sorted((root / "deepspec/pipeline").rglob("*.py")),
        root / "deepspec/orchestration/process.py",
        root / "deepspec/trainer/qwen3_8_vllm.py",
        root / "torchtitan/torchtitan/trainer.py",
        *sorted((root / "torchtitan/torchtitan/models/dspark_draft").rglob("*.py")),
        root / "vllm/vllm/config/parallel.py",
        root / "vllm/vllm/v1/engine/utils.py",
        root / "vllm/vllm/v1/engine/core.py",
        root / "vllm/vllm/v1/executor/ray_executor_v2.py",
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def native_placement_capabilities(root):
    """Require explicit implementation markers across native placement seams.

    A package version or a synthetic actor is not evidence of these hooks.
    Native integration tests must validate the marker-bearing implementation.
    """
    import ast

    paths = {
        "borrowed_pg": ("vllm/vllm/config/parallel.py", "vllm/vllm/v1/engine/utils.py"),
        "cpu_core": ("vllm/vllm/v1/engine/core.py",),
        "allocation_gate": ("vllm/vllm/v1/executor/ray_executor_v2.py",),
    }
    result = {}
    for capability, files in paths.items():
        supported = []
        for path in files:
            if not (Path(root) / path).is_file():
                supported.append(False)
                continue
            tree = ast.parse((Path(root)/path).read_text())
            supported.append(any(
                isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "RAY_PLACEMENT_API_VERSION" for target in statement.targets)
                and isinstance(statement.value, ast.Constant) and statement.value.value == 1
                for statement in tree.body
            ))
        result[capability] = all(supported)
    return result


class TaskNodeInspector:
    """Short-lived CPU actor: inspect existing resources, never allocate GPUs."""

    def __init__(self, config, run, token):
        self.config, self.run, self.token = config, run, token
        self.sample_sequence = 1

    def ready(self):
        return True

    def sample_startup_memory(self, request_id):
        """Sample after slow identity hashing, with request-specific freshness."""
        import ray

        self.sample_sequence += 1
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "agent_epoch": self.token,
            "request_id": request_id,
            "sample_seq": self.sample_sequence,
            "memory": node_memory(),
            "observed_at": time.monotonic(),
        }

    def inspect(self):
        import ray
        from torchtitan.models.dspark_draft.planning import model_identity

        from .run import ROOT
        from .schema import content_hash

        config = self.config
        output = Path(self.run["output_dir"])
        if (output / f".inspection-{self.token}").read_text() != self.token:
            raise ValueError("Shared output witness is not visible on this node")
        source = Path(config["data"]["source_path"])
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024**2), b""):
                digest.update(block)
        context = ray.get_runtime_context()
        node_id = context.get_node_id()
        ip = ray.util.get_node_ip_address()
        inventory = gpu_inventory()
        busy = set(subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines())
        witness = output / f".inspection-{self.token}-{node_id}"
        witness.write_text(self.token)
        versions_value = {"python": sys.version, "python_executable": sys.executable, "packages": versions()}
        return {"node_id": node_id, "ip": ip, "hostname": socket.gethostname(), "alive": True,
                "gpus": inventory, "free_gpu_uuids": [g["uuid"] for g in inventory if g["uuid"] not in busy],
                "gpu_uuids_in_use": sorted(busy),
                "memory": node_memory(), "identities": {
                    "source": content_hash(source_hashes(ROOT)), "dependencies": content_hash(versions_value),
                    "model": content_hash(model_identity(config["model_path"])), "input": digest.hexdigest()},
                "environment": versions_value, "network_interface": interface_for_ip(ip),
                "shared_paths": {"readable": True, "writable": True, "witness_path": str(witness)},
                "capabilities": native_placement_capabilities(ROOT),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "agent_epoch": self.token, "sample_seq": 1, "observed_at": time.monotonic(),
                "evidence_level": "observed"}


def _inspect_task_nodes_worker(config, run):
    import ray
    from ray._private.state import available_resources_per_node
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from .runtime import Deadline, PipelineError

    deadline = Deadline.after(config["timeouts_seconds"]["allocation"])
    token = uuid.uuid4().hex
    output = Path(run["output_dir"])
    witness = output / f".inspection-{token}"
    witness.write_text(token)
    actors, local_witnesses = [], []
    try:
        ray.init(address=config["ray_address"], namespace=f"{run['namespace']}-inspect-{token}", log_to_driver=False)
        resolved_address = (
            ray.get_runtime_context().gcs_address
            if config["ray_address"] == "auto" else config["ray_address"]
        )
        selected = []
        for spec in config["nodes"]:
            key, value = next(iter(spec["selector"].items()))
            field = "NodeID" if key == "node_id" else "NodeManagerAddress"
            matches = [n for n in ray.nodes() if n["Alive"] and n[field] == value]
            if len(matches) != 1 or any(n["NodeID"] == matches[0]["NodeID"] for n in selected):
                raise PipelineError("NODE_SELECTION", "Expected unique live node selectors", field_path="nodes", exit_code=4)
            node = matches[0]
            selected.append(node)
            actor = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0)(TaskNodeInspector).options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
            ).remote(config, run, token)
            actors.append(actor)
        ray.get([actor.ready.remote() for actor in actors], timeout=deadline.remaining())
        reports = ray.get([actor.inspect.remote() for actor in actors], timeout=deadline.remaining())
        available = available_resources_per_node()
        request_id = uuid.uuid4().hex
        sent_at = time.monotonic()
        snapshots = ray.get(
            [actor.sample_startup_memory.remote(request_id) for actor in actors],
            timeout=min(deadline.remaining(), config["timeouts_seconds"]["budget_snapshot"]),
        )
        if len(snapshots) != len(reports):
            raise PipelineError("NODE_FACTS_MISSING", "Missing startup memory snapshots", field_path="nodes", exit_code=4)
        for report, snapshot in zip(reports, snapshots):
            if (snapshot.get("request_id") != request_id
                    or snapshot.get("node_id") != report["node_id"]
                    or snapshot.get("agent_epoch") != report["agent_epoch"]
                    or snapshot.get("sample_seq", 0) <= report["sample_seq"]):
                raise PipelineError("NODE_FACTS_MISMATCH", "Startup memory snapshot identity mismatch", field_path="nodes", exit_code=4)
            report.update(snapshot)
        for report in reports:
            path = Path(report["shared_paths"]["witness_path"])
            local_witnesses.append(path)
            if path.read_text() != token:
                raise PipelineError("SHARED_PATH_UNAVAILABLE", "Remote write is not visible to controller", field_path="output_dir", exit_code=4)
            free = available.get(report["node_id"], {})
            sharing = config.get("gpu_sharing", "exclusive")
            eligible = report["gpus"] if sharing == "shared" else report["free_gpu_uuids"]
            report.update(ray_address=resolved_address, request_sent_at=sent_at, cpu_available=free.get("CPU", 0)+1,
                          gpu_sharing=sharing, ray_gpu_available=free.get("GPU", 0),
                          gpu_available=min(free.get("GPU", 0), len(eligible)))
        return reports
    finally:
        cleanup_errors = []
        try:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as error:  # noqa: BLE001 -- collect cleanup failures for every owned actor
                    cleanup_errors.append(str(error))
        finally:
            ray.shutdown()
            witness.unlink(missing_ok=True)
            for path in local_witnesses:
                path.unlink(missing_ok=True)
        if cleanup_errors:
            raise PipelineError("INSPECTION_CLEANUP_UNKNOWN", "; ".join(cleanup_errors), field_path="nodes", exit_code=4)


def inspect_task_nodes(config, run):
    """Bound Ray connection and native inspection in a killable CPU driver."""
    from .runtime import PipelineError, atomic_json

    output = Path(run.output_dir)
    token = uuid.uuid4().hex
    request, result = output/f"inspection-{token}.request.json", output/f"inspection-{token}.result.json"
    atomic_json(request, {"config": config, "run": run.to_dict(), "result_path": str(result)})
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "deepspec.pipeline.cluster", "--inspect-task", str(request)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, check=False,
            timeout=config["timeouts_seconds"]["allocation"]+config["timeouts_seconds"]["cleanup"],
        )
        if result.exists():
            payload = json.loads(result.read_text())
            if "error" in payload:
                details = payload["error"]
                raise PipelineError(**details, exit_code=payload["exit_code"])
            if completed.returncode == 0:
                return payload["nodes"]
        raise PipelineError("NODE_INSPECTION_FAILED", completed.stderr[-4000:] or "Inspection process failed", field_path="nodes", exit_code=4)
    except subprocess.TimeoutExpired as error:
        raise PipelineError("NODE_INSPECTION_TIMEOUT", "CPU node inspection exceeded its shared deadline", field_path="nodes", exit_code=4) from error
    finally:
        request.unlink(missing_ok=True)
        result.unlink(missing_ok=True)


def select_nodes(nodes, producer, consumer, *, consumer_dp=1, producer_dp=1):
    selected = []
    for selector in (producer, consumer):
        matches = [
            node
            for node in nodes
            if node["Alive"]
            and selector in (node["NodeID"], node["NodeManagerAddress"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one live Ray node for {selector}; found {len(matches)}"
            )
        node = matches[0]
        if node["Resources"].get("GPU", 0) < 4:
            raise ValueError(f"Node {selector} must advertise at least four GPUs")
        selected.append(node)
    if selected[0]["NodeID"] == selected[1]["NodeID"]:
        raise ValueError("Producer and consumer must occupy distinct nodes")
    for node, dp, role in zip(
        selected, (producer_dp, consumer_dp), ("producer", "consumer"), strict=True
    ):
        if dp == 2 and node["Resources"].get("GPU", 0) < 8:
            raise ValueError(f"DP2 requires eight GPUs on the {role} node")
    return selected


def interface_for_ip(address):
    devices = json.loads(subprocess.check_output(["ip", "-j", "addr"], text=True))
    matches = [
        device["ifname"]
        for device in devices
        if any(item.get("local") == address for item in device.get("addr_info", []))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one local network interface for {address}")
    return matches[0]


def gpu_inventory(*, timeout=30):
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=timeout,
    ).splitlines()
    result = []
    for row in rows:
        index, identity, name, total, used = [item.strip() for item in row.split(",")]
        result.append(
            {
                "index": index,
                "uuid": identity,
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
            }
        )
    return result


def gpu_processes(run_id, known_owned=None, *, proc_root=Path("/proc")):
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).splitlines()
    marker = f"DEEPSPEC_PIPELINE_RUN_ID={run_id}".encode()
    if known_owned is None:
        known_owned = set()
    result = []
    for row in rows:
        pid, identity, used = [item.strip() for item in row.split(",")]
        try:
            process = proc_root / pid
            state = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if state[0] in ("Z", "X"):
                continue
            process_identity = (int(pid), int(state[19]))
            try:
                environment = (process / "environ").read_bytes().split(b"\0")
            except PermissionError:
                # Non-root monitors may lose environ access during worker exit.
                # Only an already observed PID/start-time remains trusted.
                environment = []
        except (FileNotFoundError, ProcessLookupError):
            continue
        if marker in environment:
            known_owned.add(process_identity)
        # NVML can still report a worker while its exit clears /proc/environ.
        # Keep an observed identity through exit, but never trust a reused PID.
        result.append(
            {
                "pid": int(pid),
                "gpu_uuid": identity,
                "used_mib": used,
                "owned": process_identity in known_owned,
                "process_start_ticks": process_identity[1],
            }
        )
    return result


def process_memory(run_id):
    marker = f"DEEPSPEC_PIPELINE_RUN_ID={run_id}".encode()
    processes = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if marker not in (path / "environ").read_bytes().split(b"\0"):
                continue
            status = dict(
                line.split(":", 1)
                for line in (path / "status").read_text().splitlines()
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        processes.append(
            {
                "pid": int(path.name),
                "name": status["Name"].strip(),
                "rss_bytes": int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
                "anonymous_bytes": int(status.get("RssAnon", "0 kB").split()[0]) * 1024,
            }
        )
    # Shared pages can be counted in several processes. Use this for trends,
    # while node_memory remains the source for actual admission headroom.
    return {
        "rss_bytes": sum(p["rss_bytes"] for p in processes),
        "anonymous_bytes": sum(p["anonymous_bytes"] for p in processes),
        "processes": processes,
    }


def stop_run_processes(run_id):
    """Stop residual descendants after Ray actors exit, matching the exact run."""
    import psutil

    def owned():
        result = []
        for row in process_memory(run_id)["processes"]:
            if row["pid"] == os.getpid():
                continue
            try:
                process = psutil.Process(row["pid"])
                if process.environ().get("DEEPSPEC_PIPELINE_RUN_ID") == run_id:
                    result.append(process)
            except psutil.NoSuchProcess:
                pass
        return result

    signals = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        processes = owned()
        for process in processes:
            try:
                # psutil checks its cached PID/start-time identity before signals.
                process.send_signal(sig)
                signals.append({"pid": process.pid, "signal": int(sig)})
            except psutil.NoSuchProcess:
                pass
        deadline = time.monotonic() + 5
        while owned() and time.monotonic() < deadline:
            time.sleep(0.1)
    remaining = [p.pid for p in owned()]
    if remaining:
        raise RuntimeError(f"Run-owned processes survived cleanup: {remaining}")
    return signals


def rdma_counters(devices):
    counters = {}
    for name in filter(None, devices.split(",")):
        for port in (Path("/sys/class/infiniband") / name / "ports").glob("*"):
            values = {}
            for field in (
                "port_xmit_data",
                "port_rcv_data",
                "port_xmit_packets",
                "port_rcv_packets",
                "port_rcv_errors",
                "port_xmit_discards",
                "link_downed",
            ):
                try:
                    values[field] = int((port / "counters" / field).read_text())
                except (OSError, ValueError):
                    continue
            counters[f"{name}/{port.name}"] = values
    return counters


class NodeMonitor:
    """Node-local preflight and memory accounting, without a GPU allocation."""

    def __init__(self, config, role):
        import ray

        self.config = config
        self.role = role
        self.node_id = ray.get_runtime_context().get_node_id()
        self.budget = None
        self.known_owned = set()
        self.agent_epoch = uuid.uuid4().hex
        self.sample_seq = 0
        self.path = Path(config["output_dir"]) / f"node-{role}.jsonl"

    def inspect(self, expected_config_digest):
        from .run import ROOT

        config_path = Path(self.config["output_dir"]) / "pipeline.json"
        if (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            != expected_config_digest
        ):
            raise ValueError(
                "Node cannot see the same pipeline configuration in shared storage"
            )
        for name in ("model_path", "source_path", "plan_path"):
            if not Path(self.config[name]).exists():
                raise FileNotFoundError(self.config[name])
        processes = gpu_processes(self.config["run_id"], self.known_owned)
        if processes:
            raise RuntimeError(
                f"Node {self.role} has GPU processes before allocation: {processes}"
            )
        inventory = gpu_inventory()
        producer = self.role == "producer"
        dp = consumer_dp(self.config)
        writers = producer_dp(self.config) if producer else 0
        required_gpus = 4 * (producer_dp(self.config) if producer else dp)
        if len(inventory) < required_gpus:
            raise RuntimeError(
                f"The selected node must contain at least {required_gpus} physical GPUs"
            )
        self.budget = feature_budget(
            0 if producer else self.config["pool_bytes"],
            max(sample["nbytes"] for sample in self.config["samples"]),
            self.config["window"],
            0 if producer else self.config["consumer_world_size"],
            self.config["samples_per_update"] // dp,
            writer=writers,
        )
        return {
            "node_id": self.node_id,
            "hostname": socket.gethostname(),
            "python": sys.executable,
            "python_version": sys.version,
            "versions": versions(),
            "source_sha256": source_hashes(ROOT),
            "gpu_inventory": inventory,
            "memory_budget": self.budget,
            "network_interface": (
                interface_for_ip(os.environ["DEEPSPEC_STORE_HOST"]) if dp > 1 else None
            ),
        }

    def rendezvous_port(self):
        from .store import free_port

        return free_port()

    def cleanup(self):
        report = {"signals": stop_run_processes(self.config["run_id"])}
        report["gpu_processes"] = gpu_processes(self.config["run_id"], self.known_owned)
        (Path(self.config["output_dir"]) / f"cleanup-node-{self.role}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report

    def memory(self):
        memory = node_memory()
        return {
            "node_id": self.node_id,
            "role": self.role,
            **memory,
            "pressure": memory["headroom_bytes"]
            < (
                self.budget["memory_reserve_bytes"] + self.budget["scratch_bound_bytes"]
                + self.budget.get("transport_staging_bound_bytes", 0)
            ),
        }

    def sample_budget(self, request):
        from .runtime import validate_message

        validate_message(request, run_id=self.config["run_id"], plan_hash=self.config["plan_hash"])
        self.sample_seq += 1
        return {**request, "sender_identity": {"component": "node_agent", "node_id": self.node_id},
                "event_id": uuid.uuid4().hex, "node_id": self.node_id,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "agent_epoch": self.agent_epoch, "sample_seq": self.sample_seq,
                "memory": node_memory(), "sampled_at": time.monotonic(),
                # Configured pool size/RSS cannot establish a retained charge.
                "retained_charges": []}

    def sample(self):
        record = {
            "time": time.time(),
            "memory": self.memory(),
            "gpu_processes": gpu_processes(self.config["run_id"], self.known_owned),
            "process_memory": process_memory(self.config["run_id"]),
            "rdma_counters": rdma_counters(
                self.config["store"].get("rdma_devices", "")
            ),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        foreign = [p for p in record["gpu_processes"] if not p["owned"]]
        if foreign:
            raise RuntimeError(f"Unexpected GPU processes on {self.role}: {foreign}")
        return record


class NodeAgent:
    """Lease-bound node control, independent of model actor execution.

    The runtime must create this CPU actor with a run-unique detached lifetime
    and exit_on_orphan=True. Its lease then outlives a lost driver, cleans only
    registered identities, writes a node-local report, and exits its own actor.
    """

    def __init__(self, plan, node_id, fencing_token, *, exit_on_orphan=False, report_dir=None, sample_resources=False):
        import re
        import threading

        from deepspec.orchestration.process import NodeLease

        from .planning import TopologyPlan

        self.plan = TopologyPlan.from_dict(plan).to_dict()
        if node_id not in {n["node_id"] for n in plan["nodes"].values()} or not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
            raise ValueError("NodeAgent must belong to a planned node")
        self.node_id, self.fencing_token = node_id, fencing_token
        self.agent_epoch, self.sample_seq = uuid.uuid4().hex, 0
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.output = Path(plan["config"]["output_dir"])
        if report_dir is not None:
            report_dir = Path(report_dir).resolve()
            if not report_dir.is_relative_to(self.output.resolve()):
                raise ValueError("Node reports must remain within their run directory")
            self.output = report_dir
        self.exit_on_orphan = exit_on_orphan
        self._processes, self.known_owned = {}, set()
        self._lock, self._cleanup_lock = threading.Lock(), threading.Lock()
        self._stopped = threading.Event()
        self.accepting, self.cleanup_result = True, None
        self.lease = NodeLease(fencing_token, timeout=plan["timeouts_seconds"]["lease"], on_expire=self._expire)
        self._watcher = threading.Thread(target=self._watch, name="deepspec-node-lease", daemon=True)
        self.sampler = None
        if sample_resources:
            from .observation import ResourceSampler

            self.sampler = ResourceSampler(plan, node_id, self.output)
        self._watcher.start()

    def _validate(self, request):
        from .runtime import validate_message

        validate_message(request, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"])

    def _reply(self, **payload):
        from .runtime import message_envelope

        return message_envelope(self.plan["run_id"], self.plan["plan_hash"],
                                {"component": "node_agent", "node_id": self.node_id}, **payload)

    def _watch(self):
        interval = min(0.1, self.plan["timeouts_seconds"]["heartbeat"])
        while not self._stopped.wait(interval):
            if not self.lease.check():
                return

    def _expire(self):
        self.cleanup(orphan=True)
        if self.exit_on_orphan:
            os._exit(1)

    def heartbeat(self, request):
        self._validate(request)
        accepted = self.lease.heartbeat(request.get("fencing_token"), request.get("sequence"))
        return self._reply(accepted=accepted, agent_epoch=self.agent_epoch, node_id=self.node_id)

    def register_process(self, request):
        from deepspec.orchestration.process import capture_process

        self._validate(request)
        identity = request["process"]
        if request.get("fencing_token") != self.fencing_token or identity["pid"] == os.getpid():
            raise ValueError("Invalid registration token or agent self-registration")
        observed = capture_process(identity["pid"], self.plan["run_id"])
        if observed["start_ticks"] != identity["start_ticks"] or identity["run_id"] != self.plan["run_id"]:
            raise ValueError("Registered process PID/start-time or run marker differs")
        report = request.get("supervisor_report_path")
        if report and not Path(report).resolve().is_relative_to(self.output.resolve()):
            raise ValueError("Supervisor report must belong to this run directory")
        with self._lock:
            if not self.accepting or not self.lease.is_active():
                raise ValueError("Expired node cannot register new work")
            key = (observed["pid"], observed["start_ticks"])
            value = {**observed, "supervisor_report_path": report}
            if key in self._processes and self._processes[key] != value:
                raise ValueError("Conflicting registered process")
            self._processes[key] = value
            self.known_owned.add(key)
        return self._reply(process=observed)

    def sample_budget(self, request):
        self._validate(request)
        with self._lock:
            if not self.accepting or not self.lease.is_active():
                raise RuntimeError("Node admission is stopped")
            self.sample_seq += 1
            sequence = self.sample_seq
        return {**request, "sender_identity": {"component": "node_agent", "node_id": self.node_id},
                "event_id": uuid.uuid4().hex, "node_id": self.node_id, "boot_id": self.boot_id,
                "agent_epoch": self.agent_epoch, "sample_seq": sequence,
                "sampled_at": time.monotonic(), "memory": node_memory(), "retained_charges": []}

    def check_allocated_devices(self, request):
        self._validate(request)
        selected = set(request["gpu_uuids"])
        observed = gpu_processes(self.plan["run_id"], self.known_owned)
        foreign = [p for p in observed if p["gpu_uuid"] in selected and not p["owned"]]
        sharing = self.plan["config"].get("gpu_sharing", "exclusive")
        if foreign and sharing != "shared":
            raise RuntimeError(f"Allocated devices contain external processes: {foreign}")
        return self._reply(node_id=self.node_id, gpu_uuids=sorted(selected),
                           gpu_sharing=sharing, external_processes=foreign)

    def cleanup(self, *, orphan=False, timeout=None):
        from deepspec.orchestration.process import signal_process

        from .runtime import Deadline, atomic_json, bounded_lock

        duration = self.plan["timeouts_seconds"]["cleanup"] if timeout is None else min(timeout, self.plan["timeouts_seconds"]["cleanup"])
        deadline = Deadline.after(duration)
        with bounded_lock(self._cleanup_lock, deadline):
            if self.cleanup_result is not None:
                return self.cleanup_result
            with self._lock:
                self.accepting = False
                identities = list(self._processes.values())
            with self.lease.lock:
                self.lease.expired = True
            self._stopped.set()
            term_until = time.monotonic() + min(1, duration / 3)
            supervisor_until = deadline.expires_at - min(0.1, duration / 10)
            states = {}
            while True:
                for identity in identities:
                    key = str(identity["pid"])+":"+str(identity["start_ticks"])
                    # A supervisor must stay alive while it kills/reaps resistant
                    # descendants and writes its identity-bound cleanup report.
                    grace = supervisor_until if identity["supervisor_report_path"] else term_until
                    states[key] = signal_process(identity, signal.SIGTERM if time.monotonic() < grace else signal.SIGKILL)
                if all(state == "released" for state in states.values()) or time.monotonic() >= deadline.expires_at:
                    break
                time.sleep(min(0.05, max(0, deadline.expires_at-time.monotonic())))
            errors = []
            for identity in identities:
                report_path = identity["supervisor_report_path"]
                if report_path:
                    try:
                        report = json.loads(Path(report_path).read_text())
                        if (report["run_id"] != self.plan["run_id"] or report.get("cleanup_complete") is not True
                                or report.get("supervisor_pid") != identity["pid"]
                                or report.get("supervisor_start_ticks") != identity["start_ticks"]):
                            raise ValueError("Supervisor cleanup was not confirmed")
                    except (OSError, ValueError, KeyError) as error:
                        errors.append({"pid": identity["pid"], "error": str(error)})
            report = self._reply(node_id=self.node_id, agent_epoch=self.agent_epoch, orphan=orphan,
                      reason="lease_expired" if orphan else "requested_cleanup", processes=identities,
                      release_states=states, errors=errors,
                      cleanup_complete=all(s == "released" for s in states.values()) and not errors)
            if self.sampler is not None:
                try:
                    self.sampler.stop(timeout=deadline.remaining())
                except Exception as error:  # noqa: BLE001 -- record sampler failure without skipping cleanup evidence
                    report["cleanup_complete"] = False
                    report["errors"].append({"sampler": str(error)})
            atomic_json(self.output / f"{'orphan' if orphan else 'cleanup'}-{self.node_id}.json", report)
            self.cleanup_result = report
            return report

    def close(self):
        from .runtime import Deadline

        deadline = Deadline.after(self.plan["timeouts_seconds"]["cleanup"])
        report = self.cleanup(timeout=deadline.remaining())
        self._stopped.set()
        self._watcher.join(timeout=deadline.remaining())
        if self._watcher.is_alive():
            raise TimeoutError("Node watchdog did not exit")
        return report


class StoreProbe:
    """Create/read probe tensors locally; only descriptors cross Ray."""

    def __init__(self, store_config, *, identity_config=None, defer_store=False):
        self.store_config, self.identity_config = store_config, identity_config
        self.store = None
        self.objects = {}
        if not defer_store:
            self.initialize()

    def identity(self):
        from .runtime import actor_identity

        return actor_identity(self.identity_config)

    def initialize(self):
        from .store import TensorStore

        if self.store is None:
            self.store = TensorStore(self.store_config)
        return {"ready": True}

    def write(self, prefix):
        import torch

        from .store import describe_tensors

        tensors = {
            "input_ids": torch.arange(4096).reshape(1, -1),
            "loss_mask": torch.ones(1, 4096, dtype=torch.bool),
            "seq_len": torch.tensor([4096]),
            "context_chunk_len": torch.tensor([4096]),
            "target_hidden_states": torch.ones(1, 4096, 1280, dtype=torch.bfloat16),
            "target_last_hidden_states": torch.full(
                (1, 4096, 256), 2, dtype=torch.bfloat16
            ),
        }
        fields = describe_tensors(prefix, tensors)
        self.objects[prefix] = fields
        self.store.put(fields, tensors)
        return fields

    def read(self, fields):
        from .store import FIELDS

        started = time.monotonic()
        tensors = self.store.get(fields, FIELDS)
        return {
            "nbytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "seconds": time.monotonic() - started,
            "verified": self.store.verify_mode == "full",
        }

    def remove(self, fields):
        from .store import object_keys

        self.store.remove(fields)
        if any(self.store.client.is_exist(key) != 0 for key in object_keys(fields)):
            raise RuntimeError("Probe objects were not deleted")
        for prefix, pending in list(self.objects.items()):
            if pending == fields:
                del self.objects[prefix]
        return {"confirmed_absent": True}

    def close(self, *, timeout=35):
        from .runtime import Deadline

        deadline = Deadline.after(timeout)
        if self.store is not None:
            for fields in list(self.objects.values()):
                deadline.remaining()
                self.remove(fields)
            self.store.close(timeout=deadline.remaining())
        return {"cleanup_complete": True}


def launch_cluster(config, config_path):
    """Legacy callers share the versioned planner and native controller."""
    from .legacy import run_config

    return run_config(config)


if __name__ == "__main__":
    from .runtime import PipelineError, atomic_json

    if len(sys.argv) != 3 or sys.argv[1] != "--inspect-task":
        raise SystemExit("Use deepspec.pipeline.cli preview --config FILE")
    request = json.loads(Path(sys.argv[2]).read_text())
    try:
        result = _inspect_task_nodes_worker(request["config"], request["run"])
    except Exception as error:  # noqa: BLE001 -- subprocess boundary must report structured failures
        wrapped = error if isinstance(error, PipelineError) else PipelineError(
            "NODE_INSPECTION_FAILED", str(error), field_path="nodes", exit_code=4)
        atomic_json(request["result_path"], {"error": wrapped.to_dict(), "exit_code": wrapped.exit_code})
        raise SystemExit(wrapped.exit_code)
    atomic_json(request["result_path"], {"nodes": result})
