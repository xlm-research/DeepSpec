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
from pathlib import Path

from .memory import feature_budget, node_memory
from .topology import consumer_dp, consumer_microbatches, producer_dp

logger = logging.getLogger(__name__)
PACKAGES = ("torch", "vllm", "ray", "mooncake-transfer-engine", "transformers", "numpy")


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def source_hashes(root):
    paths = [
        *sorted((root / "deepspec/pipeline").glob("*.py")),
        root / "deepspec/trainer/qwen3_8_vllm.py",
        root / "torchtitan/torchtitan/trainer.py",
        root / "torchtitan/torchtitan/models/dspark_draft/data.py",
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


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


def gpu_inventory():
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
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
            environment = (process / "environ").read_bytes().split(b"\0")
        except FileNotFoundError:
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


class StoreProbe:
    """Create/read probe tensors locally; only descriptors cross Ray."""

    def __init__(self, store_config):
        from .store import TensorStore

        self.store = TensorStore(store_config)

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
        self.store.put(fields, tensors)
        return fields

    def read(self, fields):
        from .store import FIELDS

        started = time.monotonic()
        tensors = self.store.get(fields, FIELDS)
        return {
            "nbytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "seconds": time.monotonic() - started,
        }

    def remove(self, fields):
        from .store import object_keys

        self.store.remove(fields)
        if any(self.store.client.is_exist(key) != 0 for key in object_keys(fields)):
            raise RuntimeError("Probe objects were not deleted")

    def close(self):
        self.store.close()


def launch_cluster(config, config_path):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import (
        NodeAffinitySchedulingStrategy,
        PlacementGroupSchedulingStrategy,
    )

    from .actors import Consumer, Producer
    from .buffer import FeatureBuffer
    from .run import ROOT, environment, summarize_events, write_json
    from .runtime import MooncakeMaster
    from .schema import normalize_pipeline_config
    from .store import free_port

    normalize_pipeline_config(config)
    output = Path(config["output_dir"])
    common_env = {
        **environment(config["model_path"]),
        "DEEPSPEC_PIPELINE_RUN_ID": config["run_id"],
    }
    actors, groups, monitors, probes = [], [], [], []
    master = buffer = producer = None
    try:
        context = ray.init(
            address=config["cluster_address"],
            namespace=config["namespace"],
            runtime_env={"env_vars": common_env},
            log_to_driver=False,
        )
        nodes = select_nodes(
            ray.nodes(),
            config["producer_node"],
            config["consumer_node"],
            consumer_dp=consumer_dp(config),
            producer_dp=producer_dp(config),
        )
        config["producer_node_id"], config["consumer_node_id"] = [
            n["NodeID"] for n in nodes
        ]
        producer_indices = [0] * producer_dp(config)
        config["producer_node_ids"] = [nodes[i]["NodeID"] for i in producer_indices]
        config["producer_node_ips"] = [
            nodes[i]["NodeManagerAddress"] for i in producer_indices
        ]
        consumer_indices = [1]
        config["consumer_nodes"] = 1
        config["role_separation"] = True
        config["consumer_node_ids"] = [nodes[i]["NodeID"] for i in consumer_indices]
        config["ray_address"] = context.address_info["gcs_address"]
        config["store"]["per_node_hosts"] = True
        envs = [
            {**common_env, "DEEPSPEC_STORE_HOST": n["NodeManagerAddress"]}
            for n in nodes
        ]
        affinities = [
            NodeAffinitySchedulingStrategy(n["NodeID"], soft=False) for n in nodes
        ]
        write_json(config_path, config)
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        for role, env, affinity in zip(
            ("producer", "consumer"), envs, affinities, strict=True
        ):
            monitor = (
                ray.remote(NodeMonitor)
                .options(
                    num_cpus=1,
                    num_gpus=0,
                    max_restarts=0,
                    runtime_env={"env_vars": env},
                    scheduling_strategy=affinity,
                )
                .remote(config, role)
            )
            actors.append(monitor)
            monitors.append(monitor)
        facts = ray.get([m.inspect.remote(digest) for m in monitors], timeout=120)
        local_versions, local_hashes = versions(), source_hashes(ROOT)
        for fact in facts:
            if (
                fact["versions"] != local_versions
                or fact["source_sha256"] != local_hashes
            ):
                raise RuntimeError(
                    f"Dependency or source mismatch on {fact['hostname']}: {fact['versions']}"
                )
            if fact["python_version"] != sys.version:
                raise RuntimeError(f"Python build mismatch on {fact['hostname']}")
        config["store"]["hosts_by_hostname"] = {
            fact["hostname"]: node["NodeManagerAddress"]
            for fact, node in zip(facts, nodes, strict=True)
        }
        if len(config["store"]["hosts_by_hostname"]) != 2:
            raise ValueError("The two nodes must have distinct hostnames")
        config["node_memory_budgets"] = {
            f["node_id"]: f["memory_budget"] for f in facts
        }
        config["consumer_hostnames"] = [facts[1]["hostname"]] * consumer_dp(config)
        config.update(facts[1]["memory_budget"])
        write_json(
            output / "environment.json",
            {"driver": socket.gethostname(), "nodes": facts},
        )
        write_json(config_path, config)
        master = MooncakeMaster(
            config["store"]["master"],
            output / "mooncake-master.log",
            metrics_port=free_port(),
            ttl_seconds=300,
            env=dict(
                os.environ, **common_env, DEEPSPEC_ORCHESTRATOR_PID=str(os.getpid())
            ),
        ).start(timeout=30)
        buffer = (
            ray.remote(FeatureBuffer)
            .options(
                name=config["buffer_name"],
                num_cpus=1,
                num_gpus=0,
                max_restarts=0,
                runtime_env={"env_vars": envs[1]},
                scheduling_strategy=affinities[1],
            )
            .remote(config, monitors)
        )
        actors.append(buffer)
        pool = ray.get(buffer.summary.remote(), timeout=120)
        if pool["store_endpoint"].rsplit(":", 1)[0] != nodes[1]["NodeManagerAddress"]:
            raise RuntimeError("Feature pool did not advertise the consumer node")
        for env, affinity in zip(envs, affinities, strict=True):
            probe = (
                ray.remote(StoreProbe)
                .options(
                    num_cpus=1,
                    num_gpus=0,
                    max_restarts=0,
                    runtime_env={"env_vars": env},
                    scheduling_strategy=affinity,
                )
                .remote(config["store"])
            )
            actors.append(probe)
            probes.append(probe)
        fields = ray.get(
            probes[0].write.remote(f"{config['run_id']}/transport-probe"), timeout=120
        )
        probe_result = ray.get(probes[1].read.remote(fields), timeout=120)
        ray.get(probes[1].remove.remote(fields), timeout=30)
        probe_result.update(
            producer_node_id=nodes[0]["NodeID"],
            consumer_node_id=nodes[1]["NodeID"],
            store_endpoint=pool["store_endpoint"],
        )
        write_json(output / "transport-probe.json", probe_result)
        for probe in probes:
            ray.get(probe.close.remote(), timeout=30)
            ray.kill(probe, no_restart=True)
            actors.remove(probe)
        probes.clear()
        if config["transport_only"]:
            write_json(
                output / "result.json",
                {"transport_probe": probe_result, "models_started": False},
            )
            print(json.dumps(probe_result), flush=True)
            return
        # A node resource is required in every bundle; STRICT_PACK alone does
        # not require the producer and consumer groups to occupy different nodes.
        producer_bundles = [{"GPU": 1}] * 4 + [{"CPU": 1}]
        consumer_bundles = [
            {
                "GPU": config["consumer_world_size"],
                "CPU": 2 * config["consumer_world_size"],
            }
        ]
        allocations = [] if producer_dp(config) > 1 else [(producer_bundles, nodes[0])]
        allocations += [(consumer_bundles, nodes[i]) for i in consumer_indices]
        for bundles, node in allocations:
            resources = [
                {**bundle, f"node:{node['NodeManagerAddress']}": 0.001}
                for bundle in bundles
            ]
            group = placement_group(resources, strategy="STRICT_PACK")
            groups.append(group)
        ray.get([g.ready() for g in groups], timeout=120)
        producer_env = {
            **envs[0],
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        if producer_dp(config) > 1:
            # Native AsyncLLM creates its DP/TP groups from the remaining GPUs.
            # Its frontend must not capture child tasks into a consumer group.
            producer_strategy = affinities[0]
            producer_env.update(
                VLLM_RAY_DP_PLACEMENT_NODE_IPS=config["producer_node_ips"][0],
                VLLM_RAY_DP_PACK_STRATEGY="strict",
                VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="DEEPSPEC_PIPELINE_RUN_ID,PYTHONPATH,LD_LIBRARY_PATH,OMP_NUM_THREADS,RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
                LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", ""),
                NCCL_IB_DISABLE="1",
                NCCL_NET="Socket",
                NCCL_SOCKET_IFNAME=f"={facts[0]['network_interface']}",
                NCCL_SOCKET_FAMILY="AF_INET",
                GLOO_SOCKET_IFNAME=facts[0]["network_interface"],
            )
            consumer_groups = groups
        else:
            producer_strategy = PlacementGroupSchedulingStrategy(
                placement_group=groups[0],
                placement_group_bundle_index=4,
                placement_group_capture_child_tasks=True,
            )
            consumer_groups = groups[1:]
        producer = (
            ray.remote(Producer)
            .options(
                num_cpus=1,
                num_gpus=0,
                max_restarts=0,
                runtime_env={"env_vars": producer_env},
                scheduling_strategy=producer_strategy,
            )
            .remote(str(config_path))
        )
        actors.append(producer)
        consumers = []
        for node_rank, index in enumerate(consumer_indices):
            env = dict(envs[index])
            if consumer_dp(config) > 1:
                interface = facts[index]["network_interface"]
                env.update(
                    NCCL_IB_DISABLE="1",
                    NCCL_NET="Socket",
                    NCCL_SOCKET_IFNAME=f"={interface}",
                    NCCL_SOCKET_FAMILY="AF_INET",
                    GLOO_SOCKET_IFNAME=interface,
                    NCCL_DEBUG="INFO",
                )
            consumer = (
                ray.remote(Consumer)
                .options(
                    num_cpus=2 * config["consumer_world_size"],
                    num_gpus=config["consumer_world_size"],
                    max_restarts=0,
                    runtime_env={"env_vars": env},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=consumer_groups[node_rank],
                        placement_group_bundle_index=0,
                    ),
                )
                .remote(str(config_path), node_rank)
            )
            actors.append(consumer)
            consumers.append(consumer)
        pending = {producer.run.remote(): "producer"}
        pending.update(
            {
                consumer.run.remote(): "consumer"
                if rank == 0
                else f"consumer_node_{rank}"
                for rank, consumer in enumerate(consumers)
            }
        )
        results = {"transport_probe": probe_result}
        deadline = time.monotonic() + config["timeout_seconds"]
        while pending:
            finished, _ = ray.wait(list(pending), timeout=5)
            ray.get([m.sample.remote() for m in monitors], timeout=30)
            for ref in finished:
                role = pending.pop(ref)
                results[role] = ray.get(ref)
                print(f"{role} completed", flush=True)
            if time.monotonic() >= deadline:
                raise TimeoutError("Two-node pipeline timed out")
        if consumer_dp(config) > 1:
            ray.get(
                buffer.event.remote(
                    "consumer_finished",
                    completed_updates=results["consumer"]["completed_updates"],
                    completed_nodes=len(consumers),
                )
            )
        results["buffer"] = ray.get(buffer.summary.remote(), timeout=30)
        if results["buffer"]["remaining"] or results["buffer"]["released"] != len(
            config["samples"]
        ):
            raise RuntimeError(
                "Feature objects were not completely consumed and released"
            )
        results["events"] = summarize_events(Path(config["events_path"]), config)
        from torchtitan.models.dspark_draft.checkpoint import read_commit

        commit = read_commit(results["consumer"]["commit"]["checkpoint"])
        if (
            commit != results["consumer"]["commit"]
            or commit["completed_updates"] != config["steps"]
            or commit["next_global_microbatch"] != consumer_microbatches(config)
            or results["consumer"]["consumed_microbatches"]
            != consumer_microbatches(config)
            or commit["run_id"] != config["run_id"]
        ):
            raise RuntimeError(
                "Checkpoint does not match the completed two-node stream"
            )
        results["consumer"]["consumed_samples"] = len(config["samples"])
        write_json(output / "result.json", results)
        print(
            json.dumps(
                {
                    "output_dir": str(output),
                    "buffer": results["buffer"],
                    "events": results["events"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error)})
        if buffer is not None:
            try:
                ray.get(buffer.fail.remote(repr(error)), timeout=10)
            except Exception:
                logger.exception("Could not notify buffer of failure")
        raise
    finally:
        if producer is not None:
            try:
                ray.get(producer.close.remote(), timeout=15)
            except Exception:
                logger.exception("Producer close did not finish before actor cleanup")
        for actor in reversed(actors):
            try:
                if actor in monitors:
                    ray.get(actor.cleanup.remote(), timeout=20)
                if actor == buffer or actor in probes:
                    ray.get(actor.close.remote(), timeout=10)
            except Exception:
                logger.exception("Store close did not finish before actor cleanup")
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.exception("Could not stop a job-owned actor")
        for group in groups:
            remove_placement_group(group)
        if producer_dp(config) > 1 and ray.is_initialized():
            from ray.util.placement_group import get_placement_group

            # Names resolve only in this run's unique namespace, including
            # partially constructed native groups after engine startup failure.
            for rank in range(producer_dp(config)):
                try:
                    group = get_placement_group(f"dp_rank_{rank}")
                except ValueError:
                    continue
                remove_placement_group(group)
        ray.shutdown()
        if master is not None:
            master.stop()
